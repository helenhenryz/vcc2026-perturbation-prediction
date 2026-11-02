"""A toy line through the whole write path: the phase folder, 400-cell-style emission, and the checks from disk."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from conftest import nb_counts

from vccpipe import (
    CLIP,
    GeneMap,
    bulk_target,
    detection_line,
    emit_phase,
    fold_phase,
    group_sums,
    read_controls,
    read_phase,
    waterfall_arms,
)


def toy_reference(path, rng, n_genes=60):
    """A training line shaped like the challenge: controls plus three knockdowns, each with real cells."""
    genes = [f"G{i}" for i in range(n_genes)]
    mean = np.exp(rng.normal(1.5, 1.0, n_genes))
    blocks, labels = [nb_counts(rng, mean, 250)], ["non-targeting"] * 250
    for k, target in enumerate(["G0", "G1", "G2"]):
        m = mean.copy()
        m[k] *= 0.2
        blocks.append(nb_counts(rng, m, 40 + 10 * k))
        labels += [target] * (40 + 10 * k)
    X = sp.csr_matrix(np.vstack(blocks))
    obs = pd.DataFrame({"target_gene": pd.Categorical(labels)}, index=[f"c{i}" for i in range(len(labels))])
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
    return genes


def test_fold_phase_and_emit_phase(tmp_path, rng):
    genes = toy_reference(tmp_path / "reference.h5ad", rng)
    phase = fold_phase(tmp_path / "reference.h5ad", ["G0", "G1", "G2"], "toy", tmp_path / "phase")
    manifest, axis, panel = read_phase(phase)
    assert axis == genes and panel == ["G0", "G1", "G2"]
    assert manifest["cells_per_knockdown"] == {"G0": 40, "G1": 50, "G2": 60}
    controls = read_controls(phase, "toy", genes, manifest)
    assert controls.n_obs == 250

    def moments(Xc, perts, context):
        lfc = {p: np.where(np.array(genes) == p, np.log2(0.2), 0.0) for p in perts}
        return {p: (lfc[p], lfc[p]) for p in perts}, {"context": context}

    out = tmp_path / "prediction.h5ad"
    report = emit_phase(phase, moments, out, n_states=2, log=lambda *_: None)
    assert report["cells"] == 150 and report["contexts"]["toy"] == {"context": "toy"}

    stats = {}
    keys, sums, file_genes = group_sums(out, stats=stats)
    assert file_genes == genes and stats["whole"] and stats["min_count"] >= 0
    assert stats["cells_per_group"] == {("toy", "G0"): 40, ("toy", "G1"): 50, ("toy", "G2"): 60}
    csum = np.asarray(controls.X.sum(axis=0)).ravel()
    for (_, target), row in zip(keys, sums, strict=True):  # each knockdown's pseudobulk is pinned to its target
        lfc = np.clip(np.where(np.array(genes) == target, np.log2(0.2), 0.0), -CLIP, CLIP)
        assert np.all(np.abs(row - bulk_target(csum, lfc) * row.sum()) < 1.0)


def test_detection_line_recovers_a_power_law():
    a, b = 2.0, -0.4
    edges = np.exp(np.linspace(np.log(5), np.log(5e4), 25))
    centres = np.sqrt(edges[:-1] * edges[1:])
    genes = [f"G{i}" for i in range(len(centres) * 100)]
    cpm = np.repeat(centres, 100)
    rng = np.random.default_rng(1)
    frames = []
    for t in range(80):  # > 30 calls per quantile slice of each bin
        lf = rng.uniform(0, 3 * a * cpm**b)
        frames.append(
            pl.DataFrame(
                {
                    "target": f"T{t}",
                    "feature": genes,
                    "log2_fold_change": lf,
                    "p_adj": np.where(lf > a * cpm**b, 0.01, 0.5),
                }
            )
        )
    fit = detection_line({"X": pl.concat(frames)}, {"X": cpm}, genes)
    assert abs(fit["a"] / a - 1) < 0.05 and abs(fit["b"] - b) < 0.02


def test_gene_map_routes(tmp_path):
    hgnc = pd.DataFrame(
        {
            "hgnc_id": ["H:1", "H:2", "H:3", "H:4"],
            "symbol": ["ALPHA", "BETA", "GAMMA", "DELTA"],
            "prev_symbol": ["", "OLDBETA", "", ""],
            "alias_symbol": ["SHARED", "", "SHARED", ""],
            "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        }
    )
    hgnc.to_csv(tmp_path / "hgnc.txt", sep="\t", index=False)
    gmap = GeneMap(["ALPHA", "BETA", "GAMMA", "AC012345.1"], tmp_path / "hgnc.txt")
    pos, how = gmap.map(["ALPHA", "OLDBETA", "SHARED", "AC012345-1", "beta", "NOPE"])
    assert list(pos) == [0, -1, -1, 3, 1, -1]
    assert list(how) == ["exact", "duplicate", "unmapped", "exact", "approved", "unmapped"]


def test_waterfall_arms():
    arms = waterfall_arms()
    assert arms["probe"] == "probe" and arms["none"] == []
    assert arms["final_seed1"]["seed"] == 1 and json.dumps(arms)
