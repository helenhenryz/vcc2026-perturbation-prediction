"""E, the learned combiner: a small ridge regression across cell types."""

import numpy as np

from .inputs import TS
from .stage1 import type_rows

__all__ = [
    "NAMES",
    "ORION_TYPES",
    "SIB",
    "TARGETS",
    "TYPES",
    "cleaned_type_rows",
    "combiner_tables",
    "context_xc",
    "features",
    "fit_combiner",
    "scorer_space",
    "type_calls",
    "type_shares",
]

# ---- E, the learned combiner -----------------------------------------------------------------------------
# Used in: sections 10 and 13.
# A small ridge regression (23 weights) in the scorer's own space: for each knockdown, predict one cell
# type's measured response from the other types' responses and summaries of their agreement. It is
# trained on the public screens only, each target type in turn left out of its own features (and its
# sibling line too), so nothing about the line being predicted is seen. Its prediction is blended into
# the loudest, best-covered knockdowns of the bulk.
TYPES = ["k562", "cd4t", "hct116", "hek293t", "h1", "dld1", "a549", "a375", "rpe1", "hepg2", "jurkat"]
TARGETS = ["k562", "cd4t", "hct116", "hek293t", "h1", "dld1", "a549", "rpe1", "hepg2", "jurkat"]
SIB = {"hct116": "hek293t", "hek293t": "hct116"}
ORION_TYPES = ["hct116", "hek293t"]  # dropped from every feature and target when Orion is not used
NAMES = TYPES + [
    "M",
    "M_ag<=1",
    "M_ag2",
    "M_ag>=3",
    "M_ex1",
    "M_ex2",
    "M_ex3",
    "M_x_cos",
    "M_raw",
    "M_call0",
    "M_call1",
    "M_call>=2",
]


def type_shares(S):
    """Each cell type's control shares, the mean of its sources' ctrl_cpm shares."""
    qacc = {}
    for name in S.names():
        if name.startswith("jiang_"):
            continue
        s = S.load(name)
        c = np.nan_to_num(s["ctrl_cpm"])
        a = qacc.setdefault(s["cell_type"], [np.zeros_like(c), 0])
        a[0] += c / c.sum()
        a[1] += 1
        del s
    return {t: qacc[t][0] / qacc[t][1] for t in TYPES if t in qacc}


def cleaned_type_rows(S, keys):
    """The type rows from cleaned sources (each source loses its own top
    right singular direction, 40 power iterations from default_rng(0), before the type means)."""
    acc = {}
    rng = np.random.default_rng(0)
    for name in S.names():
        if name.startswith("jiang_"):
            continue
        s_ = S.load(name)
        if s_["cell_type"] not in TYPES:
            continue
        idx = {t: i for i, t in enumerate(s_["target"])}
        r = np.array([idx.get(k, -1) for k in keys])
        if (r >= 0).sum() == 0:
            continue
        Z = np.nan_to_num(np.clip(s_["lfc"], -6, 6)).astype(np.float32)
        v = rng.standard_normal(Z.shape[1]).astype(np.float32)
        for _ in range(40):  # the top right singular vector by power iteration
            v = Z.T @ (Z @ v)
            v /= np.linalg.norm(v)
        X = np.full((len(keys), Z.shape[1]), np.nan, np.float32)
        rows_ = Z[r[r >= 0]]
        rows_ = rows_ - np.outer(rows_ @ v, v)
        Xs_ = np.clip(s_["lfc"][r[r >= 0]], -6, 6)
        X[r >= 0] = np.where(np.isfinite(Xs_), rows_, np.nan)
        a = acc.setdefault(s_["cell_type"], [np.zeros(X.shape), np.zeros(X.shape)])
        a[0] += np.nan_to_num(X)
        a[1] += np.isfinite(X)
        del s_, Z, X, rows_, Xs_
    return {t: np.where(acc[t][1] > 0, acc[t][0] / np.maximum(acc[t][1], 1), np.nan) for t in TYPES if t in acc}


def type_calls(S, keys, G):
    """Per type, how many of its screens called each entry at |z| > 3 (the analytic standard error)."""
    CALLS = {t: np.zeros((len(keys), G), np.uint8) for t in TYPES}
    for name in S.names():
        if name.startswith("jiang_"):
            continue
        s_ = S.load(name)
        if s_["cell_type"] in TYPES:
            idx = {t: i for i, t in enumerate(s_["target"])}
            r = np.array([idx.get(k, -1) for k in keys])
            if (r >= 0).any():
                se2 = S.se2(name, s_)
                zz = np.abs(np.clip(s_["lfc"][r[r >= 0]], -6, 6)) / np.sqrt(np.maximum(se2[r[r >= 0]], 1e-6))
                CALLS[s_["cell_type"]][r >= 0] += np.where(np.isfinite(zz) & (zz > 3), 1, 0).astype(np.uint8)
                del se2, zz
        del s_
    return CALLS


def combiner_tables(S, keys, G):
    """The four tables E reads, for the knockdowns `keys` over `G` genes: {TR, TRraw, QC, CALLS, G}.

    TRraw: the plain type rows; TR: the type rows from cleaned sources (both float32 values held as float64);
    QC: each type's control shares; CALLS: each type's count of significant entries. A type without sources gets
    all-NaN rows (it must then be dropped).
    """
    T = type_rows(S, keys, {"ipsc"})  # the allocation's type rows (stage 1 section)
    nan = np.full((len(keys), G), np.nan)
    TRraw = {t: T[t].astype(np.float32).astype(np.float64) if t in T else nan for t in TYPES}
    del T
    QC = type_shares(S)
    C = cleaned_type_rows(S, keys)
    TR = {t: C[t].astype(np.float32).astype(np.float64) if t in C else nan for t in TYPES}
    del C
    CALLS = type_calls(S, keys, G)
    return {"TR": TR, "TRraw": TRraw, "QC": QC, "CALLS": CALLS, "G": G}


def scorer_space(L, q):
    """lfc rows -> scorer-space effects on control shares q (no renormalisation)."""
    a = TS * q[None, :]
    return np.log1p(a * np.exp2(np.clip(L, -6, 6))) - np.log1p(a)


def features(D, rows_idx, ftypes, q, own):
    """Feature maps (list of n x G arrays, names) for keys rows_idx, using source types ftypes, in shares q."""
    TR, TRraw, CALLS, G = D["TR"], D["TRraw"], D["CALLS"], D["G"]
    n = len(rows_idx)
    Xs, have = {}, {}
    for t in TYPES:
        R = TR[t][rows_idx]
        h = np.isfinite(R).mean(1) > 0.3
        if t not in ftypes:
            h[:] = False
        X = np.zeros((n, G))
        if h.any():
            X[h] = scorer_space(np.nan_to_num(R[h]), q)
        Xs[t], have[t] = X, h
    cnt = sum(have[t].astype(float) for t in TYPES)
    M = sum(Xs[t] for t in TYPES) / np.maximum(cnt, 1)[:, None]
    agree_n = np.zeros((n, G))
    sgn = np.sign(M)
    for t in TYPES:
        if t in ftypes:
            R = np.nan_to_num(TR[t][rows_idx])
            agree_n += (np.sign(R) == sgn) & (np.abs(R) > 0.3) & have[t][:, None]
    cosk = np.zeros(n)
    for i in range(n):
        rs = [Xs[t][i] for t in TYPES if have[t][i]]
        cs = [a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12) for ai, a in enumerate(rs) for b in rs[ai + 1 :]]
        cosk[i] = np.mean(cs) if cs else 0.0
    lq = np.log10(1 + 1e6 * q)
    ex = lq > np.log10(1 + 1.0)
    t1, t2 = np.quantile(lq[ex], [1 / 3, 2 / 3])
    eb = [(lq <= t1), (lq > t1) & (lq <= t2), (lq > t2)]
    feats = [Xs[t] for t in TYPES] + [M, M * (agree_n <= 1), M * (agree_n == 2), M * (agree_n >= 3)]
    feats += [M * e[None, :] for e in eb] + [M * cosk[:, None]]
    Mr = np.zeros((n, G))
    for t in TYPES:
        if t in ftypes:
            Rr = TRraw[t][rows_idx]
            hr = have[t]
            if hr.any():
                Mr[hr] += scorer_space(np.nan_to_num(Rr[hr]), q)
    feats += [Mr / np.maximum(cnt, 1)[:, None]]
    ncall = np.zeros((n, G))
    for t in TYPES:
        if t in ftypes:
            ncall += (CALLS[t][rows_idx] > 0) & have[t][:, None]
    feats += [M * (ncall == 0), M * (ncall == 1), M * (ncall >= 2)]
    names = list(NAMES)
    keep = np.ones((n, G), bool)
    r = np.flatnonzero(own >= 0)
    keep[r, own[r]] = False
    return [f * keep for f in feats], names, keep


def fit_combiner(D, keys, train_keys, gi, drop):
    """The training statistics (a centred fit) with the `drop` types out of every
    feature set and target list; returns theta (ridge 1e-3 x the mean diagonal). Also the rows per target."""
    TR, QC = D["TR"], D["QC"]
    st = {}
    nrows = {}
    for t in TARGETS:
        if t in drop:
            continue
        R = TR[t]
        ri = np.flatnonzero((np.isfinite(R).mean(1) > 0.3) & np.array([k in train_keys for k in keys]))
        nrows[t] = len(ri)
        ft = [s_ for s_ in TYPES if s_ != t and s_ != SIB.get(t) and s_ not in drop]
        q = QC[t]
        ow = np.array([gi.get(keys[i], -1) for i in ri])
        Fm, _, kp = features(D, ri, ft, q, ow)
        Y = scorer_space(np.nan_to_num(R[ri]), q) * kp
        Yc = Y - Y.mean(0)[None, :]
        Fc = [f - f.mean(0)[None, :] for f in Fm]
        nf = len(Fc)
        A = np.zeros((nf, nf))
        b = np.zeros(nf)
        for a_ in range(nf):
            b[a_] = float((Fc[a_] * Yc).sum())
            for c_ in range(a_, nf):
                A[a_, c_] = A[c_, a_] = float((Fc[a_] * Fc[c_]).sum())
        st[t] = (A, b)
        del Fm, Fc, Y, Yc
    A = sum(v[0] for v in st.values())
    b = sum(v[1] for v in st.values())
    return np.linalg.solve(A + 1e-3 * np.diag(A).mean() * np.eye(len(b)), b), nrows


def context_xc(D, thc, rows_idx, ftypes, q, own):
    """Per context: XC = sum_f theta_f F_f in the context's shares q (own entries zero)."""
    Fm, _, _ = features(D, rows_idx, ftypes, q, own)
    XC = sum(thc[i] * Fm[i] for i in range(len(Fm)))
    del Fm
    return XC
