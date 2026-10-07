"""The rank test's power, measured from a line's own controls."""

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

__all__ = ["POWER_SHIFTS", "ControlRanks", "Power", "call_level", "cpm_dense", "power_curves"]

# ---- the rank test's power from a line's own controls ----------------------------------------------------
# Used in: section 13.
# The scorer calls a gene significant by a Wilcoxon rank-sum test of the 400 predicted cells against the
# line's ~18,400 real controls. How large a shift it needs depends on the gene's expression in that line,
# so we measure it: plant shifts of known size into 400 real control cells (binomial thinning down,
# Poisson additions up) and record the test's z per gene.

POWER_SHIFTS = np.array([-1.0, -0.6, -0.4, -0.25, -0.15, 0.15, 0.25, 0.4, 0.6, 1.0])


def cpm_dense(X) -> np.ndarray:
    """Counts per million of every cell (row) of `X`, as a dense float64 array."""
    X = sp.csr_matrix(X, dtype=np.float64)
    L = np.asarray(X.sum(axis=1)).ravel()
    return np.asarray((sp.diags(1e6 / np.maximum(L, 1.0)) @ X).todense())


class ControlRanks:
    """The scorer's rank-sum test against fixed controls, exactly (average ranks, tie-corrected normal
    approximation, no continuity correction). Per tested gene: the controls' sorted CPM and tie sum."""

    def __init__(self, controls, tested: np.ndarray):
        """`controls`: the line's control cells (cells x genes, counts); `tested`: a mask of the genes tested."""
        C = cpm_dense(controls)
        self.tested = np.asarray(tested, bool)
        self.idx = np.flatnonzero(self.tested)
        self.n2 = C.shape[0]
        self.sorted = np.sort(C[:, self.idx], axis=0).astype(np.float32)
        self.tie_c = np.zeros(len(self.idx))
        for j in range(len(self.idx)):
            _, cnt = np.unique(self.sorted[:, j], return_counts=True)
            self.tie_c[j] = float((cnt.astype(np.float64) ** 3 - cnt).sum())
        self.G = C.shape[1]

    def z(self, cells) -> np.ndarray:
        """Signed z per gene (our cells above the controls -> positive); 0 for untested genes."""
        X = cpm_dense(cells).astype(np.float32)[:, self.idx]
        n1, n2 = X.shape[0], self.n2
        N = n1 + n2
        out = np.zeros(self.G)
        zs = np.empty(len(self.idx))
        for j in range(len(self.idx)):
            x = X[:, j]
            c = self.sorted[:, j]
            lo = np.searchsorted(c, x, side="left")
            hi = np.searchsorted(c, x, side="right")
            xs = np.sort(x)
            olo = np.searchsorted(xs, x, side="left")
            ohi = np.searchsorted(xs, x, side="right")
            below = lo + olo
            ties = (hi - lo) + (ohi - olo)
            r = below + (ties + 1) / 2.0
            U1 = r.sum() - n1 * (n1 + 1) / 2.0
            mu = n1 * n2 / 2.0
            vals, cnt_o = np.unique(x, return_counts=True)
            clo = np.searchsorted(c, vals, side="left")
            chi = np.searchsorted(c, vals, side="right")
            t_c = (chi - clo).astype(np.float64)
            t = t_c + cnt_o
            tie = self.tie_c[j] + ((t**3 - t) - (t_c**3 - t_c)).sum()
            var = n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1)))
            zs[j] = (U1 - mu) / np.sqrt(max(var, 1e-12))
        out[self.idx] = zs
        return out


def power_curves(controls, repeats: int = 2, seed: int = 0) -> dict:
    """z (shifts x genes, NaN where untested) of planted shifts, two emissions per shift averaged."""
    rng = np.random.default_rng(seed)
    C = sp.csr_matrix(controls)
    depth = np.asarray(C.sum(1)).ravel()
    cpm = np.asarray(C.multiply(1e6 / np.maximum(depth, 1)[:, None]).mean(0)).ravel()
    tested = cpm > 5.0
    ranks = ControlRanks(C, tested)
    acc = np.zeros((len(POWER_SHIFTS), C.shape[1]))
    for _ in range(repeats):
        idx = rng.choice(C.shape[0], 400, replace=False)
        Xd = C[idx].toarray()
        perm = rng.permutation(np.flatnonzero(tested))
        for si, s in enumerate(POWER_SHIFTS):
            for b in np.array_split(perm, 10):
                Y = Xd.copy()
                sub = Y[:, b]
                Y[:, b] = (
                    rng.binomial(sub.astype(np.int64), 2.0**s) if s < 0 else sub + rng.poisson(sub * (2.0**s - 1.0))
                )
                acc[si, b] += ranks.z(sp.csr_matrix(Y))[b]
    z = acc / repeats
    z[:, ~tested] = np.nan
    return {"z": z.astype(np.float32), "shifts": POWER_SHIFTS, "tested": tested, "n_cells": C.shape[0]}


class Power:
    """Expected |z| per gene as a function of a signed log2 shift, monotone in |shift|."""

    def __init__(self, pw: dict):
        """`pw`: the output of `power_curves`; the measured |z| is made non-decreasing in |shift| on each side."""
        self.shifts = pw["shifts"]
        self.z = np.abs(np.nan_to_num(pw["z"].astype(np.float64)))
        self.tested = pw["tested"]
        self.m = int(self.tested.sum())
        self.dn = np.flatnonzero(self.shifts < 0)[::-1]
        self.up = np.flatnonzero(self.shifts > 0)
        self.zd = np.maximum.accumulate(self.z[self.dn], axis=0)
        self.zu = np.maximum.accumulate(self.z[self.up], axis=0)
        self.gd = np.abs(self.shifts[self.dn])
        self.gu = self.shifts[self.up]

    def expected_absz(self, delta):
        """The expected |z| of each gene under a signed log2 shift `delta` (one per gene), interpolated between the
        measured shifts; 0 for untested genes.
        """
        out = np.zeros_like(delta)
        a = np.abs(delta)
        for sign, grid, tab in ((-1, self.gd, self.zd), (1, self.gu, self.zu)):
            m = (np.sign(delta) == sign) & (a > 0) & self.tested
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            x = a[idx]
            lo = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
            x0, x1 = grid[lo], grid[lo + 1]
            y0, y1 = tab[lo, idx], tab[lo + 1, idx]
            res = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
            small = x < grid[0]
            res[small] = tab[0, idx[small]] * (x[small] / grid[0])
            big = x > grid[-1]
            res[big] = np.minimum(res[big], 2.0 * tab[-1, idx[big]])
            out[idx] = np.maximum(res, 0.0)
        return out

    def expected_calls(self, delta, k0=300.0, iters=8):
        """Self-consistent Benjamini-Hochberg: k = sum_g P(|Z| > z*(k)), Z ~ N(E|z|, 1)."""
        ez = self.expected_absz(delta)
        k = k0
        for _ in range(iters):
            zstar = norm.isf(0.5 * 0.05 * max(k, 1.0) / self.m)
            p = norm.sf(zstar - ez) + norm.sf(zstar + ez)
            p[~self.tested] = 0.0
            k = float(p.sum())
        return k, p


def call_level(pool, pw: dict) -> np.ndarray:
    """medB per knockdown: over the screens that measured it, the median number of genes this line's test
    would call if the knockdown moved exactly the genes the screen itself called (|z| > 2). It sets how
    much of the bulk budget a knockdown gets."""
    power = Power(pw)
    names = [n for n in pool.allowed({"ipsc"}) if not n.startswith("jiang_") and not n.startswith("orion_")]
    per_src = {}
    for n in names:
        s = pool.src[n]
        if s["rows"] is None:
            continue
        pos = np.flatnonzero(s["present"])
        R = np.clip(s["rows"].astype(np.float64), -6, 6)
        V = np.maximum(s["se2"].astype(np.float64), 1e-6)
        ok = np.isfinite(R) & np.isfinite(V)
        R = np.where(ok, R, 0.0)
        V = np.where(ok, V, 1e6)
        dB = np.where(np.abs(R) / np.sqrt(V) > 2.0, R, 0.0)
        per_src[n] = {int(i): power.expected_calls(dB[j])[0] for j, i in enumerate(pos)}
    P = len(pool.keys)
    medB = np.full(P, np.nan)
    for i in range(P):
        vals = [per_src[n][i] for n in per_src if i in per_src[n]]
        if vals:
            medB[i] = np.median(np.array(vals))
    return medB
