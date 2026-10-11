"""Stage 2, the cell generator: from a predicted change to 400 cells of whole-number counts."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2

from .inputs import CLIP, TS

__all__ = [
    "ControlModel",
    "StateModel",
    "bulk_scale",
    "bulk_target",
    "cell_states",
    "cell_target",
    "control_noise",
    "dual_moment_counts",
    "feasible_cell_target",
    "fit_controls",
    "jackknife",
    "mean_cpm",
    "pinned_cells",
    "pseudobulk",
    "repair_bulk_totals",
    "sample_cells",
    "spread",
]

# ---- the cell generator ----------------------------------------------------------------------------------
# Used in: sections 9, 11 and 13.
# Stage 2 turns a predicted effect into 400 cells of whole-number counts. The line's controls are split
# into 120 cell states (k-means on principal components); within a state each gene is a negative binomial
# whose mean follows the cell's library size. Cells are sampled at the per-cell fold change, then refitted
# so that (a) their mean per-cell share is the per-cell target and (b) their summed counts (the
# pseudobulk) equal the bulk target exactly. No control cell is copied.


@dataclass(frozen=True)
class StateModel:
    """One cell state of a line's controls: its cells' depths and each gene's share and overdispersion."""

    library_sizes: np.ndarray  # (cells,) total counts of the state's control cells
    proportions: np.ndarray  # (genes,) share of all counts per gene
    inv_theta: np.ndarray  # (genes,) negative-binomial overdispersion; 0 = Poisson


@dataclass(frozen=True)
class ControlModel:
    """Per-context control statistics: what a cell of this line looks like at rest."""

    weights: np.ndarray  # (states,) share of control cells in each state
    states: list[StateModel]
    proportions: np.ndarray  # (states, genes), the states' proportions stacked for sampling
    inv_theta: np.ndarray  # (states, genes)


def _fit_state(X: sp.csr_matrix) -> StateModel:
    """Method-of-moments fit with each cell's library size as the offset:
    E[x_ig] = L_i p_g and Var[x_ig] = mu + mu**2 / theta_g."""
    X = sp.csr_matrix(X, dtype=np.float64)
    L = np.asarray(X.sum(axis=1)).ravel()
    p = np.asarray(X.sum(axis=0)).ravel() / L.sum()
    sum_x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()
    sum_Lx = X.T @ L
    sum_L2 = float((L**2).sum())
    sq = sum_x2 - 2 * p * sum_Lx + p**2 * sum_L2  # sum_i (x - L p)^2
    excess = sq - p * L.sum()  # minus the Poisson part, sum_i L p
    denom = p**2 * sum_L2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_theta = np.where(denom > 0, excess / denom, 0.0)
    return StateModel(L, p, np.clip(inv_theta, 0.0, None))


def cell_states(X: sp.csr_matrix, n_states: int, seed: int = 0, n_genes: int = 2000, n_pcs: int = 30) -> np.ndarray:
    """A state label per cell: k-means on the top principal components of log-normalized
    expression over the most variable genes."""
    X = sp.csr_matrix(X, dtype=np.float64)
    L = np.asarray(X.sum(axis=1)).ravel()
    Y = sp.diags(1e4 / L) @ X
    Y.data = np.log1p(Y.data)
    mean = np.asarray(Y.mean(axis=0)).ravel()
    var = np.asarray(Y.multiply(Y).mean(axis=0)).ravel() - mean**2
    top = np.argsort(var)[::-1][:n_genes]
    Z = Y[:, top].toarray() - mean[top]
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :n_pcs] * S[:n_pcs]
    _, labels = kmeans2(pcs, n_states, minit="++", seed=seed)
    return labels


def fit_controls(X: sp.csr_matrix, n_states: int = 1, seed: int = 0) -> ControlModel:
    """The control model of one line, from its control cells `X` (cells x genes, counts).

    `n_states` cell states are found by k-means (`seed`); 1 means a single state. Returns a ControlModel.
    """
    X = sp.csr_matrix(X)
    labels = np.zeros(X.shape[0], dtype=int) if n_states == 1 else cell_states(X, n_states, seed)
    present = np.unique(labels)
    weights = np.array([(labels == k).mean() for k in present])
    states = [_fit_state(X[labels == k]) for k in present]
    return ControlModel(
        weights, states, np.stack([s.proportions for s in states]), np.stack([s.inv_theta for s in states])
    )


def sample_cells(
    model: ControlModel,
    n_cells: int,
    rng: np.random.Generator,
    fold_change: np.ndarray | None = None,
) -> sp.csr_matrix:
    """n_cells cells of whole-number counts; `fold_change` (genes,) scales each gene's mean
    in every state, None meaning no effect. All cells are drawn in one pass: a state per
    cell, a library size from that state's controls, then a negative binomial per gene."""
    state = rng.choice(len(model.states), size=n_cells, p=model.weights)
    L = np.empty(n_cells)
    for s in np.unique(state):
        idx = np.flatnonzero(state == s)
        L[idx] = rng.choice(model.states[s].library_sizes, size=idx.size, replace=True)
    p = model.proportions[state]
    if fold_change is not None:
        p = p * np.asarray(fold_change)[None, :]
    mu = L[:, None] * p
    k = model.inv_theta[state]
    # gamma-Poisson mixture = negative binomial; where k == 0 the gamma step is skipped
    shape = np.where(k > 0, 1.0 / np.maximum(k, 1e-300), 1.0)
    lam = np.where(k > 0, rng.gamma(shape, mu * k), mu)
    return sp.csr_matrix(rng.poisson(lam).astype(np.float32))


def mean_cpm(X: sp.csr_matrix) -> np.ndarray:
    """Per gene: the mean over cells of counts per million, each cell normalized to its own
    total (the scorer's per-cell normalization)."""
    X = sp.csr_matrix(X, dtype=np.float64)
    L = np.asarray(X.sum(axis=1)).ravel()
    return np.asarray((sp.diags(1e6 / np.maximum(L, 1)) @ X).mean(axis=0)).ravel()


# The dual-moment fit and its integer repair are copied from a public challenge entry under the MIT
# licence; the notice follows.
# Source: [kaipengm2] github.com/kaipengm2/Virtual-Cell-Challenge-2026 @ d24ce4f, model.py
#   (dual_moment_counts, repair_bulk_totals). Licence of the copied functions:
#
#   MIT License
#   Copyright (c) 2026 Virtual Cell Challenge 2026 solution contributors
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify, merge,
#   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#   to whom the Software is furnished to do so, subject to the following conditions:
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#   FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.


def bulk_target(ctrl_sum: np.ndarray, lfc: np.ndarray) -> np.ndarray:
    """The pseudobulk to pin: the controls' pooled gene shares times 2**lfc, renormalised."""
    p = np.asarray(ctrl_sum, np.float64)
    q = (p / p.sum()) * np.exp2(np.clip(np.asarray(lfc, np.float64), -CLIP, CLIP))
    return q / q.sum()


def cell_target(ctrl_cpm: np.ndarray, lfc: np.ndarray) -> np.ndarray:
    """The per-cell moment: the controls' mean per-cell share times 2**lfc, renormalised."""
    p = np.asarray(ctrl_cpm, np.float64)
    q = (p / p.sum()) * np.exp2(np.clip(np.asarray(lfc, np.float64), -CLIP, CLIP))
    return q / q.sum()


def repair_bulk_totals(integer: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Move single counts between cells, keeping each cell's depth, until every gene's column
    total is its expected total rounded (largest remainders). [kaipengm2] model.py, MIT."""
    expected = np.asarray(expected, dtype=np.float64)
    if integer.shape != expected.shape or integer.ndim != 2:
        raise ValueError("Count and expectation axes differ")
    row_totals = integer.sum(axis=1, dtype=np.int64)
    columns = expected.sum(axis=0)
    target = np.floor(columns).astype(np.int64)
    remaining = int(row_totals.sum() - target.sum())
    if remaining < 0 or remaining > len(target):
        raise ValueError("Expected and integer grand totals differ")
    if remaining:
        chosen = np.argsort(columns - target, kind="stable")[-remaining:]
        target[chosen] += 1
    difference = integer.sum(axis=0, dtype=np.int64) - target
    capacity = np.zeros(len(integer), dtype=np.int64)
    for gene in np.flatnonzero(difference > 0):
        excess = int(difference[gene])
        rounded_up = integer[:, gene] - np.floor(expected[:, gene]).astype(np.int64)
        rows = np.flatnonzero(rounded_up > 0)
        if rounded_up[rows].sum() < excess:
            raise ValueError("Initial rounding fell below its integer floor")
        rows = rows[np.argsort(-(integer[rows, gene] - expected[rows, gene]), kind="stable")]
        if excess <= len(rows):
            selected = rows[:excess]
            integer[selected, gene] -= 1
            capacity[selected] += 1
        else:
            for row in rows:
                take = min(excess, int(rounded_up[row]))
                integer[row, gene] -= take
                capacity[row] += take
                excess -= take
                if not excess:
                    break
    deficit_genes = np.flatnonzero(difference < 0)
    deficit_genes = deficit_genes[np.argsort(difference[deficit_genes], kind="stable")]
    for gene in deficit_genes:
        deficit = int(-difference[gene])
        while deficit:
            rows = np.flatnonzero(capacity > 0)
            if not len(rows):
                raise AssertionError("Column repair exhausted row capacity")
            take = min(deficit, len(rows))
            score = expected[rows, gene] - integer[rows, gene]
            selected = rows[np.argsort(-score, kind="stable")[:take]]
            integer[selected, gene] += 1
            capacity[selected] -= 1
            deficit -= take
    if capacity.any() or (integer < 0).any():
        raise AssertionError("Column repair lost counts")
    if not np.array_equal(integer.sum(axis=1, dtype=np.int64), row_totals):
        raise AssertionError("Column repair changed a cell depth")
    if not np.array_equal(integer.sum(axis=0, dtype=np.int64), target):
        raise AssertionError("Column repair failed its bulk totals")
    return integer


def dual_moment_counts(
    template: np.ndarray,
    probability: np.ndarray,
    bulk_probability: np.ndarray,
    *,
    depths: np.ndarray,
    seed: int,
    iterations: int = 100,
    tolerance: float = 2e-4,
    max_projection: float = 0.03,
    exact_bulk: bool = True,
) -> np.ndarray:
    """Whole counts (cells x genes) with each cell's depth, the mean of per-cell shares equal
    to `probability` and the depth-weighted share (the pseudobulk) equal to `bulk_probability`.
    [kaipengm2] model.py, MIT; input checks condensed, `max_projection` exposed."""
    x = np.asarray(template, dtype=np.float64).copy()
    p = np.asarray(probability, dtype=np.float64)
    bulk = np.asarray(bulk_probability, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.int64)
    if x.ndim != 2 or p.shape != (x.shape[1],) or bulk.shape != p.shape:
        raise ValueError("Moment gene axes differ from the template")
    if (x < 0).any() or (x.sum(axis=1) <= 0).any() or (depths < 1).any():
        raise ValueError("Template rows and depths must be positive")
    if not (np.isclose(p.sum(), 1, atol=1e-8) and np.isclose(bulk.sum(), 1, atol=1e-8)):
        raise ValueError("Moment probabilities must sum to one")
    z = depths / depths.mean()
    desired = len(x) * p
    requested_bulk = bulk.copy()
    lower = (z.min() + 0.01 * (1 - z.min())) * p
    upper = (z.max() - 0.01 * (z.max() - 1)) * p
    lo, hi = 0.0, 1.0
    if np.ptp(depths) == 0:
        bulk = p.copy()
    else:
        for _ in range(1024):
            if np.clip(bulk * hi, lower, upper).sum() >= 1:
                break
            hi *= 2
        else:
            raise ValueError("Cannot bracket the bulk projection")
    for _ in range(70):
        mid = (lo + hi) / 2
        if np.clip(bulk * mid, lower, upper).sum() < 1:
            lo = mid
        else:
            hi = mid
    bulk = np.clip(bulk * ((lo + hi) / 2), lower, upper)
    projection_error = float(np.abs(bulk - requested_bulk).sum())
    if projection_error > max_projection:
        raise ValueError(f"Requested bulk projection too large: {projection_error}")
    ratio = np.divide(bulk, p, out=np.ones_like(p), where=p > 0)
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    missing = (x.sum(axis=0) == 0) & (desired > 0)
    x[:, missing] = p[missing]
    x = 0.999 * x + 0.001 * p[None, :]
    for step in range(iterations):
        col = x.sum(axis=0)
        x *= np.divide(desired, col, out=np.zeros_like(col), where=col > 0)
        first = z @ x
        mean = np.divide(first, desired, out=np.ones_like(first), where=desired > 0)
        second = (z * z) @ x
        var = np.maximum(np.divide(second, desired, out=np.zeros_like(second), where=desired > 0) - mean**2, 0)
        tilt = np.divide(ratio - mean, var, out=np.zeros_like(mean), where=var > 1e-12)
        tilt = np.clip(tilt, -0.95 / np.maximum(z.max() - mean, 1e-12), 0.95 / np.maximum(mean - z.min(), 1e-12))
        x *= 1 + tilt[None, :] * (z[:, None] - mean[None, :])
        x /= np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
        if step % 5 == 4:
            cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
            bulk_error = float(np.abs(z @ x / len(x) - bulk).sum())
            if max(cpm_error, bulk_error) < tolerance:
                break
    cpm_error = float(np.abs(x.mean(axis=0) - p).sum())
    bulk_error = float(np.abs(z @ x / len(x) - bulk).sum())
    if max(cpm_error, bulk_error) > 1e-2 and not exact_bulk:
        raise ValueError(f"Moment fitting failed: CPM {cpm_error}, bulk {bulk_error}")
    expected = x * depths[:, None]
    if exact_bulk:  # pin the REQUESTED bulk, not the fitted one (they differ < 1e-2)
        col = expected.sum(axis=0)
        want = requested_bulk * depths.sum()
        expected *= np.divide(want, col, out=np.zeros_like(col), where=col > 0)[None, :]
        lost = (col <= 0) & (want > 0)
        expected[np.argmax(depths), lost] += want[lost]
        expected *= (depths / expected.sum(axis=1))[:, None]  # rows back to their depths
    integer = np.floor(expected).astype(np.int64)
    fractions = expected - integer
    rng = np.random.default_rng(seed)
    for i in range(len(x)):
        residual = int(depths[i]) - int(integer[i].sum())
        if residual:
            cumulative = np.cumsum(fractions[i])
            cumulative *= residual / cumulative[-1]
            locations = np.searchsorted(cumulative, np.arange(residual) + rng.random(), side="right")
            np.add.at(integer[i], locations, 1)
    integer = repair_bulk_totals(integer, expected)
    if (integer < 0).any() or not np.array_equal(integer.sum(axis=1), depths):
        raise AssertionError("Integer emission lost row depths")
    return integer


def feasible_cell_target(
    p: np.ndarray, q: np.ndarray, z: np.ndarray, margin: float = 0.1, rounds: int = 20
) -> np.ndarray:
    """A depth-weighted mean of shares can only lie between the smallest and largest relative
    depth times the plain mean, so a gene whose bulk / per-cell ratio is outside that range
    cannot have both moments. Keep the bulk (MSE, PDS read it) and move the per-cell target
    of such genes to the nearest feasible ratio (with a margin), renormalising."""
    lo = z.min() + margin * (1 - z.min())
    hi = z.max() - margin * (z.max() - 1)
    p = np.asarray(p, np.float64).copy()
    q = np.asarray(q, np.float64)
    for _ in range(rounds):
        r = np.divide(q, p, out=np.ones_like(p), where=p > 0)
        bad = (r < lo) | (r > hi) | ((p <= 0) & (q > 0))
        if not bad.any():
            break
        p[bad] = q[bad] / np.clip(np.where(p[bad] > 0, r[bad], 1.0), lo, hi)
        p /= p.sum()
    return p


def pinned_cells(
    model: ControlModel,
    n: int,
    rng: np.random.Generator,
    *,
    fc_cell: np.ndarray,
    p_cell: np.ndarray,
    q_bulk: np.ndarray,
    max_projection: float = 0.03,
) -> sp.csr_matrix:
    """`n` cells for one knockdown: sampled at the per-cell fold change, then fitted so their
    mean share is `p_cell` and their pseudobulk is exactly `q_bulk` (rounded)."""
    cells = sample_cells(model, n, rng, fold_change=np.asarray(fc_cell))
    template = np.asarray(cells.todense(), dtype=np.float64)
    depths = template.sum(axis=1).astype(np.int64)
    if (depths < 1).any():  # an empty draw: give it one count of the commonest gene
        template[depths < 1, int(np.argmax(q_bulk))] = 1.0
        depths = template.sum(axis=1).astype(np.int64)
    p_cell = feasible_cell_target(p_cell, q_bulk, depths / depths.mean())
    counts = dual_moment_counts(
        template, p_cell, q_bulk, depths=depths, seed=int(rng.integers(2**32)), max_projection=max_projection
    )
    return sp.csr_matrix(counts.astype(np.float32))


def pseudobulk(q: np.ndarray) -> np.ndarray:
    """log1p(TS * share), the scorer's group profile, for shares q (rows sum to 1)."""
    return np.log1p(TS * np.asarray(q, np.float64))


def jackknife(Y) -> float:
    """The scorer's sampling correction C_p for one group of cells `Y` (cells x genes):
    (n-1)/n * sum_g sum_i (v_ig - mean_i v_ig)^2, v the delete-one-cell profile."""
    Y = np.asarray(Y.todense() if sp.issparse(Y) else Y, dtype=np.float64)
    n = len(Y)
    if n < 2:
        return 0.0
    P, S, ell = Y.sum(0), Y.sum(), Y.sum(1)
    v = np.log1p(TS * (P[None, :] - Y) / (S - ell)[:, None])
    return float((n - 1) / n * ((v - v.mean(0)) ** 2).sum())


def control_noise(ctrl: sp.csr_matrix, n: int = 400, reps: int = 8, seed: int = 0) -> float:
    """The expected C_p of an n-cell group drawn from these controls: the mean jackknife of
    `reps` random n-cell subsets. A knockdown barely changes it, so it estimates the reference's
    per-knockdown correction from the controls alone (usable on unseen lines)."""
    rng = np.random.default_rng(seed)
    ctrl = sp.csr_matrix(ctrl)
    return float(np.mean([jackknife(ctrl[rng.choice(ctrl.shape[0], n, replace=False)]) for _ in range(reps)]))


def spread(pb: np.ndarray, own: np.ndarray) -> float:
    """B: the across-knockdown spread of the submitted profiles (rows = knockdowns), each
    row's own target column (`own`, -1 = none) left out, as the scorer's rho uses it."""
    pb = np.asarray(pb, np.float64)
    keep = np.ones(pb.shape, bool)
    rows = np.flatnonzero(np.asarray(own) >= 0)
    keep[rows, np.asarray(own)[rows]] = False
    mean = (pb * keep).sum(0) / np.maximum(keep.sum(0), 1)
    return float((((pb - mean[None, :]) ** 2) * keep).sum())


def bulk_scale(lfc: np.ndarray, ctrl_sum: np.ndarray, own: np.ndarray, budget: float, hi: float = 8.0) -> float:
    """The scale s for which the pinned profiles of s * lfc (rows = knockdowns) have spread
    B = `budget` (bisection; B grows with s)."""

    def B(s):
        """The spread B of the pinned profiles of `s` x lfc."""
        return spread(pseudobulk(np.vstack([bulk_target(ctrl_sum, s * r) for r in lfc])), own)

    lo = 0.0
    if B(hi) < budget:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if B(mid) < budget:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
