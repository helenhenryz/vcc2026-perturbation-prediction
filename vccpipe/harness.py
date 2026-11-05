"""The test harness: a training line held out like the challenge, scored by the official scorer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .phase import FINAL_CONFIG, build_phase, prepare_phase, probe_phase, read_controls, read_phase
from .stage1 import PIECES

__all__ = [
    "FOLD_LINES",
    "HELD_TYPES",
    "SCORER_METRICS",
    "WATERFALL",
    "cell_eval",
    "fold_phase",
    "run_fold",
    "score_fold",
    "score_toy_screen",
    "waterfall_arms",
]

# ---- the test harness (leave one line out) ---------------------------------------------------------------
# Used in: sections 8, 10 and 15.
# A training line becomes a one-context "phase" shaped like the challenge: its control cells, its panel, and
# the real perturbed cells kept aside as the answer. Every source of the held-out line's own cell type (and
# the iPSC atlas) is removed from the training data before anything is built, and E drops that type from its
# features and its targets. The prediction is scored by the official scorer against the real cells, on the
# challenge's scale (0 = the control-mean baseline, 1 = a split-half replicate of the real data).
HELD_TYPES = {
    "h1": {"h1", "ipsc"},
    "k562gwps": {"k562", "ipsc"},
    "k562": {"k562", "ipsc"},
    "jurkat": {"jurkat", "ipsc"},
    "rpe1": {"rpe1", "ipsc"},
    "hepg2": {"hepg2", "ipsc"},
}


def fold_phase(reference_h5ad: Path, panel: list, line: str, out: Path) -> Path:
    """Write a one-context phase (manifest, gene list, panel, controls) from a training line's reference."""
    import anndata as ad

    out.mkdir(parents=True, exist_ok=True)
    ref = ad.read_h5ad(reference_h5ad, backed="r")
    lab = ref.obs["target_gene"].astype(str).to_numpy()
    ctrl = np.flatnonzero(lab == "non-targeting")
    genes = list(ref.var_names.astype(str))
    c = ref[ctrl].to_memory()
    c.obs = pd.DataFrame({"target_gene": "non-targeting", "context": line}, index=c.obs_names)
    c.obs["target_gene"] = c.obs["target_gene"].astype("category")
    c.obs["context"] = c.obs["context"].astype("category")
    c.write_h5ad(out / f"context_{line}.h5ad")
    pd.DataFrame({"gene_name": genes}).to_csv(out / "gene_names.csv", index=False)
    pd.DataFrame({"target_gene": panel}).to_csv(out / "pert_counts.csv", index=False)
    counts = {p: int((lab == p).sum()) for p in panel}
    man = {
        "contexts": [line],
        "pert_col": "target_gene",
        "context_col": "context",
        "control_label": "non-targeting",
        "n_genes": len(genes),
        "n_constructs": len(panel),
        "cells_per_knockdown": counts,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return out


SCORER_METRICS = {
    "pds": "pds_cosine",
    "mse": "expr_mse_unbiased_capped_norm",
    "nmae": "de_wilcoxon_lfc_nmae",
    "fid": "de_wilcoxon_direction_fidelity_yield_raw",
    "reach": "de_wilcoxon_direction_reach_raw",
    "jac": "de_wilcoxon_sig_jaccard",
}


def cell_eval(args, log: Path):
    """Run the official scorer's command line, `cell-eval2 *args`, with its output in `log`; raise if it fails."""
    import subprocess
    import sys

    exe = Path(sys.executable).parent / "cell-eval2"
    with open(log, "w") as f:
        rc = subprocess.run(
            [str(exe) if exe.exists() else "cell-eval2", *args], stdout=f, stderr=subprocess.STDOUT, check=False
        ).returncode
    assert rc == 0, f"cell-eval2 {args[0]} failed ({log})"


def score_fold(pred_h5ad: Path, phase_dir: Path, reference_h5ad: Path, bundle: Path | None, work: Path) -> dict:
    """The official scorer on a held-out line: our cells plus the line's real controls against the real
    cells, raw metrics, then placed on the bundle's scale (0 = baseline, 1 = replicate). bundle None: the
    reference is cut to the phase's panel and its own bundle is built first (a subset of a line's knockdowns)."""
    import anndata as ad
    import polars as pl

    work.mkdir(parents=True, exist_ok=True)
    man, genes, panel = read_phase(phase_dir)
    flags = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting"]
    if bundle is None:
        ref = ad.read_h5ad(reference_h5ad)
        keep = ref.obs["target_gene"].astype(str).isin(set(panel) | {"non-targeting"}).to_numpy()
        ref[keep].copy().write_h5ad(work / "reference.h5ad")
        del ref
        reference_h5ad = work / "reference.h5ad"
        cell_eval(
            [
                "baseline",
                "-ar",
                str(reference_h5ad),
                "-o",
                str(work / "baseline"),
                "--save-pred",
                str(work / "baseline_pred.h5ad"),
                *flags,
            ],
            work / "baseline.log",
        )
        cell_eval(
            [
                "prep-real-bundle",
                "--real",
                str(reference_h5ad),
                "--baseline",
                str(work / "baseline_pred.h5ad"),
                "-o",
                str(work / "bundle"),
                *flags,
            ],
            work / "bundle.log",
        )
        bundle = work / "bundle"
    ctx = man["contexts"][0]
    c = read_controls(phase_dir, ctx, genes, man)
    p = ad.read_h5ad(pred_h5ad)
    both = ad.AnnData(
        sp.vstack([sp.csr_matrix(c.X), sp.csr_matrix(p.X)], format="csr"),
        obs=pd.DataFrame(
            {"target_gene": list(c.obs["target_gene"].astype(str)) + list(p.obs["target_gene"].astype(str))},
            index=[f"c{i}" for i in range(c.n_obs + p.n_obs)],
        ),
        var=pd.DataFrame(index=genes),
    )
    both.write_h5ad(work / "pred_with_controls.h5ad")
    del both, c, p
    cell_eval(
        [
            "run",
            "-ap",
            str(work / "pred_with_controls.h5ad"),
            "-ar",
            str(reference_h5ad),
            "-o",
            str(work / "run"),
            *flags,
        ],
        work / "run.log",
    )
    cell_eval(
        [
            "score",
            "--user-agg",
            str(work / "run" / "agg_results.csv"),
            "--real-bundle",
            str(bundle),
            "-o",
            str(work / "scored.csv"),
        ],
        work / "score.log",
    )
    (work / "pred_with_controls.h5ad").unlink()
    df = pl.read_csv(work / "scored.csv")
    v = dict(zip(df["metric"].to_list(), df["from_replicate"].to_list(), strict=True))
    out = {k: float(v[m]) for k, m in SCORER_METRICS.items()}
    out["overall"] = sum(out[k] for k in SCORER_METRICS) / len(SCORER_METRICS)
    return out


# The waterfall: the final recipe's pieces in the order they were added. Its arms: the probe's consensus as a
# baseline, no piece, each piece added in turn (the last arm is the final recipe), and the final recipe again with
# another sampler seed, whose score difference is the noise floor.
WATERFALL = [
    "gate",
    "weakwh",
    "cis3",
    "agree_clean",
    "cell_caps",
    "down_floor",
    "ps20",
    "whiten_tol",
    "k_loto",
    "s2",
    "e",
    "l2a",
]
FOLD_LINES = ["h1", "k562gwps", "k562", "jurkat", "rpe1", "hepg2"]


def waterfall_arms() -> dict:
    """The waterfall's arms, {name: spec}: the probe's cells, no piece, each piece added in WATERFALL's order
    (the last is the final recipe), and the final recipe again with sampler seed 1.
    """
    assert set(WATERFALL) == set(PIECES)
    arms = {"probe": "probe", "none": []}
    arms |= {f"+{p}": WATERFALL[: i + 1] for i, p in enumerate(WATERFALL)}
    arms["final_seed1"] = {"pieces": WATERFALL, "seed": 1}
    return arms


def run_fold(
    line: str, tables_dir: Path, work: Path, arms: dict, cfg=None, log=print, cache: Path | None = None
) -> dict:
    """One held-out training line: its phase, the references without its own cell types, the probe, then one
    build and one official score per arm. An arm is a list of pieces (seed 0), {"pieces": [...], "seed": n},
    or "probe" (the probe's own cells, the simple borrowed consensus, scored as a baseline). Returns
    {arm: scaled metrics}.
    `cache`: each arm's score is written to cache/<line>/<arm>.json as soon as it exists, with when and how
    long it took; an arm already there is read, not re-scored, so a long chain resumes where it stopped and a
    rerun of the notebook shows the numbers of the run that made them."""
    import gc
    import time

    cfg = dict(cfg or FINAL_CONFIG)
    out, todo = {}, {}
    for arm, spec in arms.items():
        f = None if cache is None else Path(cache) / line / f"{arm}.json"
        if f is not None and f.exists():
            out[arm] = json.loads(f.read_text())["scores"]
        else:
            todo[arm] = spec
    if not todo:
        return {a: out[a] for a in arms}
    bench = Path(tables_dir) / "bench" / line
    panel = pd.read_csv(bench / "panel.csv")["target_gene"].astype(str).tolist()
    phase = fold_phase(bench / "reference.h5ad", panel, line, work / "phase")
    t0 = time.time()
    inp = prepare_phase(phase, tables_dir, work, cfg, log=log, held=HELD_TYPES[line])
    inp["detline"] = probe_phase(phase, inp, work, cfg, log=log, keep="probe" in todo)
    t_prep = time.time() - t0
    for arm, spec in todo.items():
        t1 = time.time()
        if spec == "probe":
            pred, meta = work / "probe.h5ad", {"arm": "probe"}
        else:
            spec = spec if isinstance(spec, dict) else {"pieces": list(spec), "seed": 0}
            pred, meta = work / f"pred_{arm}.h5ad", {"pieces": list(spec["pieces"]), "seed": int(spec.get("seed", 0))}
            build_phase(phase, inp, pred, dict(cfg, pieces=tuple(spec["pieces"])), log=log, seed=meta["seed"])
        out[arm] = score_fold(
            pred, phase, bench / "reference.h5ad", bench / "bundle_work" / "bundle", work / f"score_{arm}"
        )
        pred.unlink()
        gc.collect()
        log(f"fold {line} arm {arm}: " + ", ".join(f"{k} {v:.4f}" for k, v in out[arm].items()))
        if cache is not None:
            f = Path(cache) / line / f"{arm}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            meta |= {
                "line": line,
                "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - t1, 1),
                "prepare_and_probe_seconds": round(t_prep, 1),
                "detection_line": {k: inp["detline"][k] for k in ("a", "b")},
                "scores": out[arm],
            }
            f.write_text(json.dumps(meta, indent=1))
    return {a: out[a] for a in arms}


def score_toy_screen(real_h5ad: Path, predictions: dict, work: Path) -> dict:
    """Score toy predictions with the official scorer, as section 2 does: the scorer's own baseline and bundle from
    the real data, then one run and one score per prediction. Returns {name: {measure: score}} on the
    challenge's scale (0 = the control-mean baseline, 1 = a split-half replicate of the real data).

    `predictions`: {name: AnnData}, each written to `work`/pred_<i>.h5ad; every log lands in `work` too."""
    import polars as pl

    work = Path(work)
    flags = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting"]
    baseline = ["baseline", "-ar", str(real_h5ad), "-o", str(work / "baseline")]
    cell_eval([*baseline, "--save-pred", str(work / "baseline_pred.h5ad"), *flags], work / "baseline.log")
    bundle = ["prep-real-bundle", "--real", str(real_h5ad), "--baseline", str(work / "baseline_pred.h5ad")]
    cell_eval([*bundle, "-o", str(work / "bundle"), *flags], work / "bundle.log")
    scores = {}
    for i, (name, cells) in enumerate(predictions.items()):
        pred, run_dir, scored = work / f"pred_{i}.h5ad", work / f"run_{i}", work / f"scored_{i}.csv"
        cells.write_h5ad(pred)
        cell_eval(["run", "-ap", str(pred), "-ar", str(real_h5ad), "-o", str(run_dir), *flags], work / f"run_{i}.log")
        inputs = ["--user-agg", str(run_dir / "agg_results.csv"), "--real-bundle", str(work / "bundle")]
        cell_eval(["score", *inputs, "-o", str(scored)], work / f"score_{i}.log")
        df = pl.read_csv(scored)
        v = dict(zip(df["metric"].to_list(), df["from_replicate"].to_list(), strict=True))
        scores[name] = {k: float(v[m]) for k, m in SCORER_METRICS.items()}
    return scores
