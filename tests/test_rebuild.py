"""The table builders of the appendix, on toy single cells (no downloads)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from conftest import nb_counts

from vccpipe import (
    SOURCE_CLIP,
    align_genes,
    same_table,
    save_table,
    split_labels,
    stream_mean_cpm,
    table_from_cells,
)

GENES = [f"G{i}" for i in range(30)]


@pytest.fixture
def cells(rng):
    """Controls and two knockdowns over GENES; G0's knockdown cuts G0 to a fifth."""
    mean = np.exp(rng.normal(3.0, 0.5, len(GENES)))
    low = mean.copy()
    low[0] *= 0.2
    X = np.vstack([nb_counts(rng, mean, 200), nb_counts(rng, low, 80), nb_counts(rng, mean, 30)])
    labels = np.array(["non-targeting"] * 200 + ["G0"] * 80 + ["G5"] * 30)
    return sp.csr_matrix(X), labels


def test_align_genes_sums_duplicates_and_drops_unknowns():
    X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0, 4.0]]))
    out = align_genes(X, ["G1", "G1", "OTHER", "G3"], GENES)
    assert out[0, 1] == 3.0 and out[0, 3] == 4.0 and out.sum() == 7.0


def test_table_from_cells(cells):
    X, labels = cells
    t = table_from_cells([(X, labels, GENES)], "non-targeting", "toy", "3p", GENES, min_cells=50)
    assert list(t["target"]) == ["G0"]  # G5 has 30 cells, under min_cells
    assert t["lfc"].shape == (1, len(GENES)) and t["control_cells"] == 200
    assert np.log2(0.2) - 0.3 < t["lfc"][0, 0] < np.log2(0.2) + 0.3
    assert np.nanmax(np.abs(t["lfc"])) <= SOURCE_CLIP


def test_split_labels_halves(rng):
    target = np.array(["a"] * 7 + ["b"] * 4 + ["ctrl"] * 3)
    lab = split_labels(target, {"a", "b"}, rng, control="ctrl")
    assert (lab == "a|A").sum() == 3 and (lab == "a|B").sum() == 4 and (lab == "b|A").sum() == 2
    assert (lab == "ctrl").sum() == 3


def test_stream_mean_cpm_in_parts(tmp_path, cells):
    X, labels = cells
    ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])).write_h5ad(tmp_path / "x.h5ad")
    col_map = sp.identity(len(GENES), format="csr")
    one = stream_mean_cpm(tmp_path / "x.h5ad", pd.Categorical(labels), col_map, block=64)
    three = stream_mean_cpm(tmp_path / "x.h5ad", pd.Categorical(labels), col_map, block=64, parts=3)
    assert one[0] == three[0] == ["G0", "G5", "non-targeting"]
    assert np.allclose(one[1], three[1]) and np.array_equal(one[2], three[2])
    ctrl = labels == "non-targeting"
    expected = (X[ctrl].toarray() / X[ctrl].sum(axis=1).A * 1e6).sum(axis=0)
    assert np.allclose(one[1][2], expected)


def test_save_and_compare_table(tmp_path, cells):
    X, labels = cells
    t = table_from_cells([(X, labels, GENES)], "non-targeting", "toy", "3p", GENES)
    save_table(t, tmp_path / "toy")
    report = same_table(t, tmp_path / "toy.npz")
    assert all(v["equal"] for v in report.values())
    t["lfc"] = t["lfc"] + np.float32(0.5)
    assert not same_table(t, tmp_path / "toy.npz")["lfc"]["equal"]
