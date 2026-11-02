"""The source tables and the effect direction, on toy screens written to a temporary folder."""

import numpy as np
import pandas as pd
import pytest

from vccpipe import (
    Pool,
    Sources,
    cis_prior,
    combine,
    direction,
    drop_pcs,
    gate,
    neighbour_table,
    own_out_mean,
    rescale_rows,
    shrink,
    source_tables,
    whiten_shrunk,
)

GENES = [f"G{i}" for i in range(40)]


def write_source(folder, name, cell_type, targets, rng):
    """A toy screen: log2 fold changes of `targets` over GENES, as the pipeline's loaders write them."""
    lfc = rng.normal(0.0, 0.5, (len(targets), len(GENES))).astype(np.float32)
    lfc[:, -3:] = np.nan  # three genes this screen did not measure
    path = folder / f"{name}.npz"
    np.savez(
        path,
        target=np.array(targets),
        lfc=lfc,
        n_cells=np.full(len(targets), 100.0),
        cell_type=np.array(cell_type),
        chem=np.array("3p"),
        ctrl_cpm=np.full(len(GENES), 50.0),
    )
    return path


@pytest.fixture
def sources(tmp_path, rng):
    files = {
        "bench_k562": write_source(tmp_path, "bench_k562", "k562", GENES[:12], rng),
        "bench_rpe1": write_source(tmp_path, "bench_rpe1", "rpe1", GENES[4:16], rng),
        "kolf": write_source(tmp_path, "kolf", "ipsc", GENES[:20], rng),
    }
    return Sources({n: (f, None) for n, f in files.items()}, {"phi_median": np.full(len(GENES), 0.1)})


def test_own_out_mean_leaves_each_gene_out():
    L = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    out = own_out_mean(L, ["a", "b"], {"a": 0, "b": 1})
    assert np.allclose(out, [4.0, 2.0, 4.5])  # gene a: only row b counts; gene b: row b is its own, and NaN


def test_combine_and_shrink():
    L1, L2 = np.array([[1.0, np.nan]]), np.array([[3.0, 2.0]])
    V = np.ones((1, 2))
    lfc, se2, ns = combine({"a": (L1, V, np.array([1])), "b": (L2, V, np.array([1]))}, {"a": 1.0, "b": 3.0})
    assert np.allclose(lfc, [[2.5, 2.0]]) and np.all(ns == 2)
    post = shrink(lfc, se2, ns, tau=np.ones(2), generic=np.zeros(2))
    assert np.all(np.abs(post) < np.abs(lfc))  # shrunk towards zero by the noise


def test_drop_pcs_and_whitening(rng):
    P = rng.normal(size=(30, 10)) + np.outer(rng.normal(size=30), np.ones(10)) * 5
    Q = drop_pcs(P, 1)
    assert np.linalg.svd(Q, compute_uv=False)[0] < np.linalg.svd(P, compute_uv=False)[0]
    W = whiten_shrunk(Q, 1.0, lam=0.0)
    assert np.allclose(W.std(0), np.median(Q.std(0)))
    R = rescale_rows(W.astype(np.float32), P)
    assert np.allclose(np.linalg.norm(R, axis=1), np.linalg.norm(P, axis=1), rtol=1e-5)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")  # genes no screen measured: their mean is NaN, then 0
def test_tables_pool_and_direction(sources):
    keys = GENES[:18]
    tables = source_tables(sources, keys, GENES, log=lambda *_: None)
    assert tables["bench_k562"]["present"].sum() == 12 and tables["bench_rpe1"]["present"].sum() == 12
    assert np.isnan(tables["bench_k562"]["rows"][:, -1]).all()
    pool = Pool(tables, sources.names(), keys)
    P, info = direction(pool, {"k562": 2.0}, orion=False, pcs={}, srcpc_k=0)
    assert P.shape == (len(keys), len(GENES)) and np.isfinite(P).all()
    assert info["n_fill"] == 2  # G16 and G17: only the iPSC atlas measured them
    keep = gate(pool, orion=False, z=3.0)
    assert keep.shape == P.shape and set(np.unique(keep)) <= {0, 1}


def test_neighbour_prior():
    tss = pd.DataFrame(
        {
            "chrom": ["1", "1", "1", "2"],
            "tss": [1_000, 1_500, 90_000, 1_000],
            "strand": ["+", "-", "+", "+"],
            "gene_name": ["A", "B", "C", "D"],
        }
    )
    genes = ["A", "B", "C", "D"]
    nb = neighbour_table(tss, genes)
    assert list(nb["A"][0]) == [1, 2]  # B and C share A's chromosome; D does not
    prior = {"mode": "ratio", "table": np.full((8, 4), 0.5), "cpm_mult": [0.0, 1.0, 1.0, 1.0, 1.0]}
    out = cis_prior(["A"], genes, np.full(4, 10.0), nb, prior)
    assert out[0, 0] == 0.0  # the target's own column is left to the target
    assert out[0, 1] < 0 and out[0, 3] == 0.0  # a neighbour is silenced; another chromosome is not
