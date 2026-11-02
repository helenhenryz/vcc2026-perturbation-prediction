"""Stage 2, the cell generator: whole counts, each cell's depth kept, and the pseudobulk pinned exactly."""

import numpy as np
import scipy.sparse as sp

from vccpipe import (
    bulk_scale,
    bulk_target,
    cell_target,
    control_noise,
    dual_moment_counts,
    fit_controls,
    jackknife,
    mean_cpm,
    pinned_cells,
    pseudobulk,
    repair_bulk_totals,
    sample_cells,
    spread,
)


def test_targets_are_shares(controls):
    lfc = np.zeros(controls.shape[1])
    csum = np.asarray(controls.sum(axis=0)).ravel()
    assert np.allclose(bulk_target(csum, lfc), csum / csum.sum())
    assert np.allclose(cell_target(mean_cpm(controls), lfc).sum(), 1.0)
    lfc[0] = 1.0  # one gene doubled: its share grows, the others shrink by the same factor
    q = bulk_target(csum, lfc)
    ratio = q / (csum / csum.sum())
    assert ratio[0] > 1.9 and np.allclose(ratio[1:], ratio[1])


def test_mean_cpm_normalises_each_cell():
    X = sp.csr_matrix(np.array([[1.0, 3.0], [10.0, 10.0]]))
    assert np.allclose(mean_cpm(X), [0.5e6 * (0.25 + 0.5), 0.5e6 * (0.75 + 0.5)])


def test_fit_and_sample(controls, rng):
    model = fit_controls(controls, n_states=3, seed=0)
    assert np.isclose(model.weights.sum(), 1.0) and model.proportions.shape == (3, controls.shape[1])
    cells = sample_cells(model, 50, rng)
    assert cells.shape == (50, controls.shape[1])
    assert np.array_equal(cells.data, np.round(cells.data)) and cells.data.min() >= 0


def test_pinned_cells_hit_the_bulk_target(controls, rng):
    model = fit_controls(controls, n_states=2, seed=0)
    lfc = np.where(np.arange(controls.shape[1]) % 7 == 0, -0.8, 0.0)
    q_bulk = bulk_target(np.asarray(controls.sum(axis=0)).ravel(), lfc)
    p_cell = cell_target(mean_cpm(controls), lfc)
    cells = pinned_cells(model, 60, rng, fc_cell=np.exp2(lfc), p_cell=p_cell, q_bulk=q_bulk)
    counts = cells.toarray()
    assert np.array_equal(counts, np.round(counts))
    total = counts.sum()
    # each gene's summed counts is its share of the total, rounded (largest remainders): within one count
    assert np.all(np.abs(counts.sum(axis=0) - q_bulk * total) < 1.0)


def test_dual_moment_counts_keep_depths(rng):
    template = rng.poisson(5.0, size=(30, 12)).astype(float) + 1
    depths = template.sum(axis=1).astype(np.int64)
    p = template.mean(axis=0) / template.mean(axis=0).sum()
    counts = dual_moment_counts(template, p, p, depths=depths, seed=1)
    assert np.array_equal(counts.sum(axis=1), depths)
    assert (counts >= 0).all()


def test_repair_bulk_totals_moves_single_counts():
    expected = np.array([[1.6, 0.4], [0.4, 1.6]])
    integer = np.array([[2, 0], [1, 1]])
    fixed = repair_bulk_totals(integer.copy(), expected)
    assert np.array_equal(fixed.sum(axis=1), integer.sum(axis=1))
    assert np.array_equal(fixed.sum(axis=0), [2, 2])


def test_jackknife_and_spread(controls, rng):
    assert jackknife(controls[:1]) == 0.0
    assert control_noise(controls, n=50, reps=2) > 0
    csum = np.asarray(controls.sum(axis=0)).ravel()
    lfc = rng.normal(0, 0.5, (5, controls.shape[1]))
    own = np.full(5, -1)
    budget = 0.5 * spread(pseudobulk(np.vstack([bulk_target(csum, r) for r in lfc])), own)
    s = bulk_scale(lfc, csum, own, budget)
    reached = spread(pseudobulk(np.vstack([bulk_target(csum, s * r) for r in lfc])), own)
    assert 0 < s < 1 and np.isclose(reached, budget, rtol=1e-6)
