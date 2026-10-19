"""The entry points: read the challenge's input, prepare the references, run the probe, build the file."""

import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .combiner import ORION_TYPES, TYPES, combiner_tables, context_xc, fit_combiner
from .constants import CIS_PRIOR, E_ROW_EXTRA, E_TRAIN_EXTRA
from .generator import bulk_target, cell_target, fit_controls, mean_cpm, pinned_cells
from .inputs import (
    CLIP,
    TYPE_W,
    Pool,
    Sources,
    blas_threads,
    cis_prior,
    direction,
    gate,
    neighbour_table,
    probe_direction,
    rescale_rows,
    source_tables,
)
from .levels import call_level, power_curves
from .stage1 import moments_final, moments_probe, p53_key2, p53_loadings, type_rows

__all__ = [
    "BASE_SOURCES",
    "CELLS_PER_PERT",
    "EXTRA_SOURCES",
    "FINAL_CONFIG",
    "FINAL_SOURCES",
    "INT32_MAX",
    "JIANG",
    "MAX_COUNTS_PER_CELL",
    "MAX_STORED_ENTRIES",
    "NO_ORION_SOURCES",
    "ORION_SOURCES",
    "PROBE_SOURCES",
    "SOURCE_TYPE",
    "THREADS",
    "GeneMap",
    "build_phase",
    "depmap_mean",
    "depmap_on_axis",
    "detection_line",
    "emit_phase",
    "learn_e",
    "open_tables",
    "predict_context",
    "prepare_phase",
    "probe_phase",
    "read_controls",
    "read_phase",
    "scorer_calls",
    "source_pcs",
]

# ---- the sources, in the order every consensus sums them -------------------------------------------------
# The list of public screens the pipeline reads, in a fixed order (floating-point sums depend on order, and the file
# is rebuilt byte for byte).
# Used in: sections 4 and 13.
JIANG = [
    f"jiang_{line}_{cond}"
    for line in ("A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7")
    for cond in ("IFNB", "IFNG", "INS", "TGFB1", "TNFA")
]
BASE_SOURCES = (
    [
        "arc_h1",
        "bench_h1",
        "bench_hepg2",
        "bench_jurkat",
        "bench_k562",
        "bench_rpe1",
        "cd4_Rest",
        "cd4_Stim48hr",
        "cd4_Stim8hr",
    ]
    + JIANG
    + ["k562gwps_bulk", "kolf", "viperturb"]
)
EXTRA_SOURCES = ["a549_myllia", "a375_perturbfate", "a549_ko", "hesc_gs", "dld1_gx"]
ORION_SOURCES = ["orion_hct116", "orion_hek293t"]  # X-Atlas/Orion, CC BY-NC-SA 4.0
PROBE_SOURCES = BASE_SOURCES + ["a549_myllia", "a375_perturbfate", "a549_ko", "hesc_gs"]
NO_ORION_SOURCES = BASE_SOURCES + EXTRA_SOURCES
FINAL_SOURCES = NO_ORION_SOURCES + ORION_SOURCES
SOURCE_TYPE = {
    "a375_perturbfate": "a375",
    "a549_ko": "a549",
    "a549_myllia": "a549",
    "arc_h1": "h1",
    "bench_h1": "h1",
    "hesc_gs": "h1",
    "bench_hepg2": "hepg2",
    "bench_jurkat": "jurkat",
    "bench_k562": "k562",
    "k562gwps_bulk": "k562",
    "viperturb": "k562",
    "bench_rpe1": "rpe1",
    "cd4_Rest": "cd4t",
    "cd4_Stim48hr": "cd4t",
    "cd4_Stim8hr": "cd4t",
    "dld1_gx": "dld1",
    "kolf": "ipsc",
    "orion_hct116": "hct116",
    "orion_hek293t": "hek293t",
    **{n: n.split("_")[1].lower() for n in JIANG},
}


def open_tables(tables_dir: Path, names: list) -> Sources:
    """The source tables in `tables_dir` (<name>.npz, optional <name>.json from the loader with the
    screen's control-cell count and depth; phi.npz, the bench controls' overdispersion)."""
    tables_dir = Path(tables_dir)
    files = {}
    for n in names:
        f = tables_dir / f"{n}.npz"
        assert f.exists(), f"missing source table {f}"
        j = tables_dir / f"{n}.json"
        files[n] = (f, j if j.exists() else None)
    phi = dict(np.load(tables_dir / "phi.npz"))
    return Sources(files, phi, extra=[n for n, (_, j) in files.items() if j is not None])


def source_pcs(tables_dir: Path, names: list) -> dict:
    """Each main source's top principal axes (10 x genes), from srcpc/<name>.npy."""
    out = {}
    for n in names:
        f = Path(tables_dir) / "srcpc" / f"{n}.npy"
        if f.exists():
            out[n] = np.load(f).astype(np.float64)
    return out


# ---- the phase (the challenge's input) -------------------------------------------------------------------
# Reading the challenge's input: the manifest, the gene axis, the panel of knockdowns and each line's control cells.
# Used in: every section that reads the final lines.
def read_phase(phase_dir: Path):
    """The challenge's input description in `phase_dir`: (manifest, the gene axis, the panel of knockdowns)."""
    manifest = json.loads((phase_dir / "manifest.json").read_text())
    genes = pd.read_csv(phase_dir / "gene_names.csv")["gene_name"].tolist()
    perts = pd.read_csv(phase_dir / "pert_counts.csv")["target_gene"].tolist()
    assert len(genes) == manifest["n_genes"] and len(perts) == manifest["n_constructs"]
    return manifest, genes, perts


def read_controls(phase_dir: Path, context: str, genes: list, manifest: dict):
    """One context's control cells, its label checked from inside the file (a swapped context is the
    costliest silent mistake)."""
    import anndata as ad

    a = ad.read_h5ad(phase_dir / f"context_{context}.h5ad")
    assert set(a.obs[manifest["context_col"]].astype(str)) == {context}, context
    assert list(a.var_names) == genes, f"context {context}: gene axis differs"
    assert set(a.obs[manifest["pert_col"]].astype(str)) == {manifest["control_label"]}
    return a


# ---- E's matrices for a phase ----------------------------------------------------------------------------
# Training the learned combiner E for this phase's panel, from the public screens only, and its prediction in each
# line's own terms.
# Used in: section 13 (and the folds).
def learn_e(srcs: Sources, phase_dir: Path, drop_types=(), log=print) -> dict:
    """E trained on the public screens for this phase's panel, then its prediction XC in each context's
    own control shares (rows: the panel; own gene 0)."""
    import anndata as ad

    man, genes, panel = read_phase(phase_dir)
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    train_keys = set(panel) | set(E_TRAIN_EXTRA)
    keys = sorted(train_keys | set(E_ROW_EXTRA))
    kix = {k: i for i, k in enumerate(keys)}
    D = combiner_tables(srcs, keys, G)
    drop = list(drop_types)
    thc, nrows = fit_combiner(D, keys, train_keys, gi, drop)
    log(f"E: {len(keys)} keys ({len(train_keys)} trained on), training rows per target {nrows}")
    rows_idx = np.array([kix[p] for p in panel])
    own = np.array([gi.get(p, -1) for p in panel])
    ftypes = [s for s in TYPES if s not in drop]
    out = {"keys": list(panel), "theta": thc, "XC": {}, "q": {}}
    for X in man["contexts"]:
        a = ad.read_h5ad(phase_dir / f"context_{X}.h5ad")
        csum = np.asarray(sp.csr_matrix(a.X).sum(0)).ravel().astype(np.float64)
        q = csum / csum.sum()
        del a
        out["XC"][X] = context_xc(D, thc, rows_idx, ftypes, q, own).astype(np.float32)
        out["q"][X] = q
    return out


# ---- the scorer's calls and the detection line -----------------------------------------------------------
# The official scorer's statistical test run on our own cells (the probe), and from its calls each line's detection
# line: how large a change must be, at each expression level, before the test calls it.
# Used in: section 13 (and the folds).
def scorer_calls(cells_h5ad: Path, controls, workdir: Path, pert_col="target_gene", control="non-targeting"):
    """Run the official scorer's DE (cell-eval2, vcc2026 preset, profile `de`) on predicted cells against
    the line's controls; returns its per-gene table (de_pred.parquet)."""
    import subprocess
    import sys

    import anndata as ad
    import polars as pl

    workdir.mkdir(parents=True, exist_ok=True)
    pred = ad.read_h5ad(cells_h5ad)
    c = ad.AnnData(X=controls.X, obs=controls.obs[[pert_col]].copy(), var=pred.var.copy())
    p = ad.AnnData(X=pred.X, obs=pred.obs[[pert_col]].copy(), var=pred.var.copy())
    pair = ad.concat([c, p], join="outer", index_unique="-")
    pair.obs[pert_col] = pair.obs[pert_col].astype(str).astype("category")
    pf = workdir / "pair.h5ad"
    pair.write_h5ad(pf)
    del pair, pred, c, p
    exe = Path(sys.executable).parent / "cell-eval2"
    cmd = [
        str(exe) if exe.exists() else "cell-eval2",
        "run",
        "-ap",
        str(pf),
        "-ar",
        str(pf),
        "-o",
        str(workdir),
        "--preset",
        "vcc2026",
        "--pert-col",
        pert_col,
        "--control",
        control,
        "--profile",
        "de",
        "--write-degenes",
    ]
    if os.environ.get("VCC_DE_BACKEND"):  # e.g. gpudge: its exact path on the CPU when no GPU is visible
        cmd += ["--set", f"de.backend={os.environ['VCC_DE_BACKEND']}"]
    with open(workdir / "run.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    pf.unlink()
    assert rc == 0, f"cell-eval2 failed ({workdir / 'run.log'})"
    return pl.read_parquet(workdir / "de_pred.parquet")


def detection_line(de_by_ctx: dict, cpm_by_ctx: dict, genes: list) -> dict:
    """Where this line's test starts calling genes: per control-CPM bin, the |log2FC| at which half the
    tested genes are called (p_adj < 0.05), then a power law thr = a CPM^b through the bins."""
    import polars as pl

    gi = {g: i for i, g in enumerate(genes)}
    edges = np.exp(np.linspace(np.log(5), np.log(5e4), 25))
    X, Y, bins = [], [], []
    for ctx, de in de_by_ctx.items():
        cpm = cpm_by_ctx[ctx]
        de = de.select(["target", "feature", "p_adj", "log2_fold_change"]).filter(pl.col("target") != pl.col("feature"))
        f = de["feature"].to_numpy()
        c = cpm[np.array([gi[x] for x in f])]
        lf = np.abs(de["log2_fold_change"].to_numpy())
        sig = de["p_adj"].to_numpy() < 0.05
        for lo, hi in pairwise(edges):
            m = (c >= lo) & (c < hi) & np.isfinite(lf)
            if m.sum() < 2000:
                continue
            q = np.quantile(lf[m], np.linspace(0, 0.999, 200))
            share = np.array(
                [
                    sig[m & (lf >= a) & (lf < b)].mean() if (m & (lf >= a) & (lf < b)).sum() > 30 else np.nan
                    for a, b in pairwise(q)
                ]
            )
            mid = (q[:-1] + q[1:]) / 2
            k = np.flatnonzero(np.isfinite(share) & (share >= 0.5))
            if k.size:
                X.append(np.sqrt(lo * hi))
                Y.append(float(mid[k[0]]))
                bins.append({"ctx": ctx, "cpm": X[-1], "thr": Y[-1], "n": int(m.sum())})
    b, loga = np.polyfit(np.log(X), np.log(Y), 1)
    return {"bins": bins, "a": float(np.exp(loga)), "b": float(b)}


# ---- emitting a phase's cells ----------------------------------------------------------------------------
# Writing the prediction: 400 cells per knockdown and line, one line at a time, with the submission's checks
# asserted as the file is written.
# Used in: sections 8, 10, 13 and 15.
CELLS_PER_PERT = 400
MAX_COUNTS_PER_CELL = 1_000_000
MAX_STORED_ENTRIES = 4_750_000_000
INT32_MAX = 2**31 - 1


def emit_phase(
    phase_dir: Path,
    moments_fn,
    out: Path,
    seed: int = 0,
    n_states: int = 120,
    nnz_frac: float | None = 0.98,
    keep_parts: bool = False,
    log=print,
) -> dict:
    """Every knockdown's 400 cells in every context, one context at a time (each written to its own part
    and freed, then merged on disk). `moments_fn(controls_csr, perts, context) -> ({pert: (lb, lc)}, info)`.
    The control model is fitted with `seed`, the sampler's generator is seeded [seed, context index].
    `nnz_frac`: stop early if the merged file's stored entries would pass that share of the int32 range
    (the submission's limit for 32-bit indices; None = no guard, as for the probe, which is never uploaded)."""
    import gc

    import anndata as ad
    import h5py
    from anndata.experimental import concat_on_disk
    from anndata.io import read_elem

    manifest, genes, perts = read_phase(phase_dir)
    pc, cc = manifest["pert_col"], manifest["context_col"]
    out = Path(out)
    profiles = {}
    for context in manifest["contexts"]:  # each context's controls, for the label check
        c = read_controls(phase_dir, context, genes, manifest)
        profiles[context] = np.log1p(np.asarray(c.X.sum(axis=0)).ravel() / c.n_obs)
        del c
    n_of = manifest.get("cells_per_knockdown") or {}  # a held-out training line keeps its real cell counts
    report = {"nnz": 0, "cells": 0, "label_check": {}, "contexts": {}}
    parts, offset, largest = [], 0, 0
    for ci, context in enumerate(manifest["contexts"]):
        controls = read_controls(phase_dir, context, genes, manifest)
        Xc = sp.csr_matrix(controls.X)
        moments, info = moments_fn(Xc, perts, context)
        report["contexts"][context] = info
        csum = np.asarray(Xc.sum(axis=0)).ravel()
        ccpm = mean_cpm(Xc)
        targets = {}
        for p, (lb, lc) in moments.items():
            lc = np.asarray(lc, np.float64)
            targets[p] = (np.exp2(np.clip(lc, -CLIP, CLIP)), cell_target(ccpm, lc), bulk_target(csum, lb))
        model = fit_controls(controls.X, n_states=n_states, seed=seed)
        rng = np.random.default_rng([seed, ci])
        blocks, obs = [], []
        for pert in perts:
            fc, p_cell, q_bulk = targets[pert]
            n = int(n_of.get(pert, CELLS_PER_PERT))
            blocks.append(pinned_cells(model, n, rng, fc_cell=fc, p_cell=p_cell, q_bulk=q_bulk))
            obs += [(pert, context)] * n
        X = sp.vstack(blocks, format="csr")
        del blocks
        assert X.data.min() >= 0 and np.array_equal(X.data, np.round(X.data)), "counts must be whole"
        assert np.asarray(X.sum(axis=1)).max() <= MAX_COUNTS_PER_CELL
        mine = np.log1p(np.asarray(X.sum(axis=0)).ravel() / X.shape[0])
        corr = {k: float(np.corrcoef(mine, v)[0, 1]) for k, v in profiles.items()}
        assert max(corr, key=corr.get) == context, f"context {context} looks most like another line: {corr}"
        report["label_check"][context] = corr
        largest = max(largest, int(X.nnz))  # the merged file's stored entries must fit int32 indices
        projected = report["nnz"] + int(X.nnz) + largest * (len(manifest["contexts"]) - ci - 1)
        assert nnz_frac is None or projected <= nnz_frac * INT32_MAX, (
            f"projected stored entries {projected} exceed int32"
        )
        report["nnz"] += int(X.nnz)
        report["cells"] += int(X.shape[0])
        obs_df = pd.DataFrame(obs, columns=[pc, cc])
        obs_df.index = [f"cell{i}" for i in range(offset, offset + len(obs_df))]
        for col in obs_df:
            obs_df[col] = obs_df[col].astype("category")
        part = Path(f"{out}.ctx_{context}.h5ad")
        ad.AnnData(X=X, obs=obs_df, var=pd.DataFrame(index=genes)).write_h5ad(part)
        os.sync()
        parts.append(part)
        offset += len(obs_df)
        log(f"[{context}] {X.shape[0]} cells, nnz {X.nnz}")
        del X, obs, obs_df, controls, Xc, model, moments, targets
        gc.collect()
    if out.exists():
        out.unlink()
    concat_on_disk(parts, out)
    os.sync()
    with h5py.File(out, "r") as f:  # the submission rules, on the merged file without loading its matrix
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        assert f["X"].attrs["encoding-type"] == "csr_matrix"
        assert int(f["X"]["data"].shape[0]) == report["nnz"] and int(f["X"].attrs["shape"][0]) == report["cells"]
    assert list(var.index.astype(str)) == genes
    counts = obs.groupby([cc, pc], observed=True).size()
    assert all(int(v) == int(n_of.get(k[1], CELLS_PER_PERT)) for k, v in counts.items())
    for context in manifest["contexts"]:
        assert set(obs.loc[obs[cc] == context, pc].astype(str)) == set(perts), context
    assert manifest["control_label"] not in set(obs[pc].astype(str))
    assert report["nnz"] <= MAX_STORED_ENTRIES
    report["parts"] = [str(p) for p in parts]
    if not keep_parts:
        for p in parts:
            p.unlink()
    return report


# ---- the whole phase -------------------------------------------------------------------------------------
# The entry points: `prepare_phase` (everything before cells are drawn), `probe_phase` (the detection line),
# `predict_context` (stage 1 for one line) and `build_phase` (the whole file).
# Used in: sections 8, 10, 13 and 15.
FINAL_CONFIG = {
    "budget": 1.0,
    "cell": 2.0,
    "floor": 0.25,
    "de_floor": 0.75,
    "de_floor_down": 0.5,
    "cell_table_cap": 6.0,
    "gate_z": 3.0,
    "gate_bg": 0.3,
    "bulk_cis_k": 3.0,
    "whiten_tol": 1e-6,
    "cells_gatecap": 1.0,
    "comp_knee": 1.0,
    "comp_kappa": 0.5,
    "ps": 20.0,
    "k562_weight": 1.0,
    "srcpc_k": 1,
    "smooth": (300, 0.5, 0.2, 3),
    "learn": (0.5, 1.0, 4, 0.2),
    "l2a": (10, "half"),
    "probe": {"budget": 0.9, "cell": 2.0, "floor": 0.25},
    "tp53_k": 0.5,
}
THREADS = {"probe_direction": 4, "directions": 6, "levels": 4, "E": 6, "build": 6}


def prepare_phase(
    phase_dir: Path,
    tables_dir: Path,
    work: Path,
    cfg=FINAL_CONFIG,
    log=print,
    sources=FINAL_SOURCES,
    held=frozenset(),
    p53_wild_type=(),
) -> dict:
    """Everything the final recipe needs before any cell is drawn, from the phase's controls and the
    public tables only: directions, gate, neighbour prior, call levels, E. (The detection line comes from
    the probe, `probe_phase`.) `sources` / `held`: a held-out training line's fold drops its own cell types."""
    man, genes, perts = read_phase(phase_dir)
    work.mkdir(parents=True, exist_ok=True)
    sources = [n for n in sources if SOURCE_TYPE[n] not in held]
    probe_src = [n for n in PROBE_SOURCES if n in sources]
    lvl_src = [n for n in NO_ORION_SOURCES if n in sources]
    orion = any(n in ORION_SOURCES for n in sources)
    drop = sorted({t for t in TYPES if t in held} | (set() if orion else set(ORION_TYPES)))
    srcs = open_tables(tables_dir, sources)
    pcs = source_pcs(tables_dir, sources)
    tabs = source_tables(srcs, perts, genes, log=lambda *_: None)
    inp = {"perts": perts, "genes": genes, "contexts": list(man["contexts"])}
    with blas_threads(THREADS["probe_direction"]):
        inp["probe_dir"] = probe_direction(Pool(tabs, probe_src, perts))
    cpms = {}
    for ctx in man["contexts"]:
        cpms[ctx] = mean_cpm(read_controls(phase_dir, ctx, genes, man).X)
    inp["ctrl_cpm"] = cpms
    tss = pd.read_csv(Path(tables_dir) / "tss.csv")
    shared = np.mean(np.stack(list(cpms.values())), axis=0)
    inp["cis"] = cis_prior(perts, genes, shared, neighbour_table(tss, genes), CIS_PRIOR)
    pool = Pool(tabs, sources, perts)
    tw = dict(TYPE_W, k562=cfg["k562_weight"])
    with blas_threads(THREADS["directions"]):
        P, _ = direction(pool, tw, orion, pcs, cfg["srcpc_k"])
        inp["stack_dir"] = P.astype(np.float32)
        k0s = {}

        def k0_of(nm):
            """The control CPM of source `nm` (NaN as 0), loaded once: the damping's pseudocount scale."""
            if nm not in k0s:
                k0s[nm] = np.nan_to_num(np.asarray(srcs.load(nm)["ctrl_cpm"], np.float64), nan=0.0)
            return k0s[nm]

        P, _ = direction(pool, tw, orion, pcs, cfg["srcpc_k"], cfg["ps"], k0_of)
        inp["ps_dir"] = rescale_rows(P.astype(np.float32), inp["stack_dir"])
    inp["gate"] = gate(pool, orion, cfg["gate_z"])
    log("directions, gate, cis ready")
    inp["medB"] = {}
    with blas_threads(THREADS["levels"]):
        lpool = Pool(tabs, lvl_src, perts)
        for ctx in man["contexts"]:
            pw = power_curves(sp.csr_matrix(read_controls(phase_dir, ctx, genes, man).X))
            inp["medB"][ctx] = call_level(lpool, pw)
            log(f"levels {ctx}: median {np.nanmedian(inp['medB'][ctx]):.0f}")
    del tabs
    with blas_threads(THREADS["E"]):
        e = learn_e(srcs, phase_dir, drop_types=drop, log=log)
    inp["E_keys"], inp["E_XC"], inp["E_theta"] = e["keys"], e["XC"], e["theta"]
    inp["T"] = type_rows(srcs, perts, {"ipsc"})
    inp["tp53"], inp["tp53_status"] = {}, {}
    if p53_wild_type:  # key 1 (the line's curated status) is the caller's; key 2 is read from the controls here
        dmean = depmap_mean(Path(tables_dir) / "depmap_expression.csv", genes, Path(tables_dir) / "hgnc.txt")
        load = p53_loadings(srcs, perts, genes)
        for ctx in man["contexts"]:
            csum = np.asarray(read_controls(phase_dir, ctx, genes, man).X.sum(axis=0, dtype=np.float64)).ravel()
            k2 = p53_key2(csum, dmean, genes)
            fire = ctx in p53_wild_type and k2["call"] == "agree wild-type"
            inp["tp53_status"][ctx] = {
                "key1": "wild-type" if ctx in p53_wild_type else "not called",
                "key2": k2,
                "fire": fire,
            }
            if fire:
                inp["tp53"][ctx] = load
            log(f"p53 {ctx}: {inp['tp53_status'][ctx]}")
    inp["T_probe"] = type_rows(srcs.subset(probe_src), perts, {"ipsc"})
    return inp


class GeneMap:
    """Other datasets' gene names -> positions on the challenge's axis: the exact symbol first, else both sides
    resolved to HGNC ids (approved symbol, then a previous or alias symbol naming one gene only); two names on one
    axis gene: the better route keeps it."""

    ORDER = ("exact", "approved", "ensembl", "previous", "alias")

    def __init__(self, axis: list, hgnc_txt: Path):
        """`axis`: the challenge's gene names, in order; `hgnc_txt`: HGNC's complete set (tab-separated)."""
        import re
        from collections import defaultdict

        self._re = re
        self.axis = pd.Index(axis)
        h = pd.read_csv(hgnc_txt, sep="\t", dtype=str)[
            ["hgnc_id", "symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id"]
        ]
        self.approved, self.prev, self.alias, self.ensembl = {}, defaultdict(set), defaultdict(set), {}
        for hid, sym, prev, alias, ens in h.itertuples(index=False):
            self.approved[str(sym).upper()] = hid
            if isinstance(ens, str):
                self.ensembl[ens] = hid
            for s_ in str(prev).split("|") if isinstance(prev, str) else ():
                self.prev[s_.upper()].add(hid)
            for s_ in str(alias).split("|") if isinstance(alias, str) else ():
                self.alias[s_.upper()].add(hid)
        ids = pd.Series([self._resolve(s_)[0] for s_ in axis])
        once = ids[ids.notna() & ~ids.duplicated(keep=False)]
        self.axis_by_id = dict(zip(once.to_numpy(), once.index.to_numpy(), strict=True))

    def _resolve(self, sym: str, ens: str | None = None):
        """The HGNC id of a symbol (or of its Ensembl id `ens`) and the route that found it; (None, "unmapped")."""
        u = sym.upper()
        if u in self.approved:
            return self.approved[u], "approved"
        if ens and ens in self.ensembl:
            return self.ensembl[ens], "ensembl"
        for table, how in ((self.prev, "previous"), (self.alias, "alias")):
            if len(table.get(u, ())) == 1:
                return next(iter(table[u])), how
        return None, "unmapped"

    def map(self, symbols, ensembl=None):
        """Positions of `symbols` on the axis (-1 = none) and how each was matched (`ensembl` ids help, if given).

        An axis gene claimed twice keeps the name with the better route.
        """
        symbols = [str(s_) for s_ in symbols]
        exact = self.axis.get_indexer(symbols)
        pos = np.full(len(symbols), -1)
        how = np.full(len(symbols), "unmapped", dtype=object)
        for i, s_ in enumerate(symbols):
            if exact[i] >= 0:
                pos[i], how[i] = exact[i], "exact"
                continue
            variants = [s_] + (
                [s_.rsplit("-", 1)[0] + "." + s_.rsplit("-", 1)[1]]
                if self._re.fullmatch(r"[A-Z]{1,2}\d{6}-\d+", s_)
                else []
            )
            for v in variants:
                j = self.axis.get_indexer([v])[0]
                if j >= 0:
                    pos[i], how[i] = j, "exact"
                    break
                hid, route = self._resolve(v, None if ensembl is None else ensembl[i])
                if hid in self.axis_by_id:
                    pos[i], how[i] = self.axis_by_id[hid], route
                    break
        rank = np.array([self.ORDER.index(h) if h in self.ORDER else 9 for h in how])
        seen = set()
        for i in np.lexsort((np.arange(len(pos)), rank)):
            if pos[i] >= 0:
                if pos[i] in seen:
                    pos[i], how[i] = -1, "duplicate"
                else:
                    seen.add(pos[i])
        return pos, how


def depmap_on_axis(csv: Path, gmap: GeneMap) -> pd.DataFrame:
    """DepMap's expression matrix (log2 TPM + 1, rows = cell lines) with its columns on the axis's gene names."""
    df = pd.read_csv(csv, index_col=0)
    pos, _ = gmap.map([c.split(" (")[0] for c in df.columns])
    out = df.loc[:, pos >= 0]
    out.columns = gmap.axis[pos[pos >= 0]]
    return out


def depmap_mean(csv: Path, genes: list, hgnc_txt: Path) -> np.ndarray:
    """DepMap's mean log2(TPM + 1) per axis gene over its ~1,500 cell lines (NaN where it has none)."""
    expr = depmap_on_axis(csv, GeneMap(genes, hgnc_txt))
    return expr.mean(axis=0).reindex(pd.Index(genes)).to_numpy(np.float64)


def probe_phase(phase_dir: Path, inp: dict, work: Path, cfg=FINAL_CONFIG, log=print, keep: bool = False) -> dict:
    """The probe: cells from a simpler recipe, the scorer's calls on them, and from those this line's
    detection line. Its cells are deleted afterwards (keep=True leaves them at work/probe.h5ad, so the test
    harness can score the probe as a baseline)."""
    man, genes, _ = read_phase(phase_dir)
    work.mkdir(parents=True, exist_ok=True)
    L = inp["probe_dir"].astype(np.float64)
    cis = inp["cis"].astype(np.float64)
    pc = cfg["probe"]

    def mfn(controls, perts_, ctx):
        """The probe's targets for one context (the callback `emit_phase` calls)."""
        return moments_probe(controls, perts_, genes, L, cis, inp["T_probe"], pc["budget"], pc["cell"], pc["floor"])

    out = work / "probe.h5ad"
    with blas_threads(THREADS["build"]):
        rep = emit_phase(phase_dir, mfn, out, nnz_frac=None, keep_parts=True, log=log)
    de = {}
    for ctx, part in zip(man["contexts"], rep["parts"]):
        de[ctx] = scorer_calls(Path(part), read_controls(phase_dir, ctx, genes, man), work / f"probe_calls_{ctx}")
        Path(part).unlink()
    if not keep:
        out.unlink()
    dl = detection_line(de, inp["ctrl_cpm"], genes)
    log(f"detection line: a {dl['a']:.4f}, b {dl['b']:.4f}")
    return dl


def predict_context(controls, perturbations, ctx, inp, cfg=FINAL_CONFIG):
    """Stage 1 for one context: {knockdown: (bulk log2FC, per-cell log2FC)}. `inp` holds the references
    built for the phase (prepare_phase + the probe's detection line)."""
    assert list(perturbations) == inp["perts"]
    return moments_final(sp.csr_matrix(controls), list(perturbations), inp["genes"], ctx, inp, cfg)


def build_phase(
    phase_dir: Path, inp: dict, out: Path, cfg=FINAL_CONFIG, keep_parts=False, log=print, seed: int = 0
) -> dict:
    """The final prediction for every context of the phase: stage 1 then stage 2, one context at a time.
    `seed` seeds stage 2's sampler (the final file uses 0; the harness's noise floor reruns with another)."""

    def mfn(controls, perts_, ctx):
        """Stage 1's targets for one context (the callback `emit_phase` calls)."""
        return predict_context(controls, perts_, ctx, inp, cfg)

    with blas_threads(THREADS["build"]):
        return emit_phase(phase_dir, mfn, out, seed=seed, keep_parts=keep_parts, log=log)
