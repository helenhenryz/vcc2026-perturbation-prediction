"""Stage 1's building blocks: whitening, the randomized SVD, the p53 restore and the per-cell amplitude table."""

import numpy as np

from vccpipe import (
    CELL_TABLE,
    agreement,
    gram_whiten,
    k_loto_of,
    p53_restore,
    rsvd,
    table_multiplier,
)


def test_gram_whiten_keeps_row_norms(rng):
    P = rng.normal(size=(20, 50)) @ np.diag(np.linspace(5, 0.1, 50))
    W = gram_whiten(P)
    assert np.allclose(np.linalg.norm(W, axis=1), np.linalg.norm(P, axis=1))
    s = np.linalg.svd(W / np.linalg.norm(W, axis=1, keepdims=True), compute_uv=False)
    assert s[0] / s[-1] < np.linalg.svd(P, compute_uv=False)[0] / np.linalg.svd(P, compute_uv=False)[-1]


def test_rsvd_finds_the_top_axes(rng):
    X = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 40)) * 10 + rng.normal(size=(200, 40)) * 0.01
    V = rsvd(X, 3)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    assert np.allclose(np.abs(V @ Vt[:3].T), np.eye(3), atol=1e-6)


def test_agreement_of_identical_types_is_one(rng):
    rows = rng.normal(size=(5, 30))
    T = {"a": rows, "b": rows * 2.0}
    expr = np.ones(30, bool)
    assert np.allclose(agreement(T, expr), 1.0)
    # when every type agrees, a knockdown's amplitude is its row's norm relative to the median row
    norms = np.linalg.norm(rows, axis=1)
    assert np.allclose(k_loto_of({**T, "c": rows * 3.0}, expr), norms / np.median(norms))


def test_p53_restore_adds_the_loading_and_keeps_the_target():
    genes = ["TP", "CDKN1A", "MDM2", "X"]
    load = {"keys": ["TP"], "A": np.array([[1.0, 2.0]]), "genes": ["CDKN1A", "MDM2"]}
    Lb = np.zeros((2, 4))
    out, n = p53_restore(Lb, ["TP", "OTHER"], genes, load, k=0.5)
    assert n == 1
    assert np.allclose(out[0], [0.0, 0.5, 1.0, 0.0]) and np.allclose(out[1], 0.0)


def test_table_multiplier_reads_the_bins():
    L = np.array([[0.005, 0.5]])
    cpm = np.array([0.5, 500.0])
    m = table_multiplier(L, cpm, CELL_TABLE)
    assert m[0, 0] == CELL_TABLE[0][0] and m[0, 1] == CELL_TABLE[-1][-1]
