"""The rank test against fixed controls, checked against scipy's Mann-Whitney U."""

import numpy as np
import scipy.sparse as sp
from conftest import nb_counts
from scipy.stats import mannwhitneyu, norm

from vccpipe import ControlRanks, Power, power_curves
from vccpipe.levels import cpm_dense


def test_control_ranks_match_scipy(rng, controls):
    mean = np.asarray(controls.mean(axis=0)).ravel()
    shifted = mean * np.exp2(rng.choice([-1.0, 0.0, 1.0], mean.size))
    cells = sp.csr_matrix(nb_counts(rng, shifted, 80))
    tested = np.asarray(controls.sum(axis=0)).ravel() > 0
    z = ControlRanks(controls, tested).z(cells)

    x = cpm_dense(cells).astype(np.float32)
    y = cpm_dense(controls).astype(np.float32)
    for j in np.flatnonzero(tested)[:40]:
        res = mannwhitneyu(x[:, j], y[:, j], use_continuity=False, method="asymptotic")
        expected = np.sign(res.statistic - x.shape[0] * y.shape[0] / 2) * norm.isf(res.pvalue / 2)
        assert np.isclose(z[j], expected, rtol=1e-6, atol=1e-6), j
    assert np.all(z[~tested] == 0)


def test_power_grows_with_the_shift(rng):
    controls = sp.csr_matrix(nb_counts(rng, np.exp(rng.normal(2.0, 1.0, 60)), 500))  # 400 cells are drawn from them
    pw = power_curves(controls, repeats=1)
    assert pw["z"].shape == (len(pw["shifts"]), controls.shape[1])
    power = Power(pw)
    genes = np.flatnonzero(pw["tested"])
    small = power.expected_absz(np.where(pw["tested"], 0.2, 0.0))
    large = power.expected_absz(np.where(pw["tested"], 0.8, 0.0))
    assert np.all(large[genes] >= small[genes])
    assert np.all(small[~pw["tested"]] == 0)
