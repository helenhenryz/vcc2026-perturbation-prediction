"""Stage 1's pieces: the allocation, the p53 response, and each knockdown's targets."""

import numpy as np
import scipy.sparse as sp

from .constants import CELL_E_EDGES, CELL_L_EDGES, CELL_TABLE
from .generator import bulk_scale, bulk_target, control_noise, mean_cpm, pseudobulk, spread
from .inputs import CLIP, TS, Sources

__all__ = [
    "P53_CORE",
    "P53_KEY2",
    "P53_SOURCES",
    "PIECES",
    "agreement",
    "control_pcs",
    "gram_whiten",
    "k_loto_of",
    "l2a",
    "moments_final",
    "moments_probe",
    "p53_key2",
    "p53_loadings",
    "p53_restore",
    "rsvd",
    "table_multiplier",
    "type_rows",
    "weights",
]


# ---- the allocation (how loud each knockdown is) ---------------------------------------------------------
# The allocation: how large each knockdown's predicted change is, from how well the cell types agree on it (a
# knockdown every screen sees the same way is predicted louder).
# Used in: sections 10 and 13.
def type_rows(srcs: Sources, keys: list, held: set) -> dict:
    """Per source cell type: the mean clipped row of each knockdown over that type's sources (NaN where
    none measured); the pathway screens and the `held` types left out."""
    acc = {}
    for name in srcs.names():
        s = srcs.load(name)
        if s["cell_type"] in held or name.startswith("jiang_"):
            continue
        idx = {t: i for i, t in enumerate(s["target"])}
        r = np.array([idx.get(k, -1) for k in keys])
        if (r >= 0).sum() == 0:
            continue
        X = np.full((len(keys), s["lfc"].shape[1]), np.nan, np.float32)
        X[r >= 0] = np.clip(s["lfc"][r[r >= 0]], -6, 6)
        a = acc.setdefault(s["cell_type"], [np.zeros(X.shape), np.zeros(X.shape)])
        a[0] += np.nan_to_num(X)
        a[1] += np.isfinite(X)
    return {t: np.where(n > 0, a / np.maximum(n, 1), np.nan) for t, (a, n) in acc.items()}


def agreement(T: dict, expr: np.ndarray) -> np.ndarray:
    """Per knockdown: the mean pairwise cosine of the cell types' rows over the expressed genes."""
    n = next(iter(T.values())).shape[0]
    out = np.zeros(n)
    for i in range(n):
        rows = [T[t][i, expr] for t in T if np.isfinite(T[t][i, expr]).mean() > 0.5]
        cs = []
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                m = np.isfinite(rows[a]) & np.isfinite(rows[b])
                x, y = rows[a][m], rows[b][m]
                cs.append(x @ y / np.sqrt((x @ x) * (y @ y) + 1e-12))
        out[i] = np.mean(cs) if cs else 0.0
    return out


def weights(agree: np.ndarray, floor: float) -> np.ndarray:
    """Per knockdown, sqrt of its (non-negative) agreement, floored at `floor` x the median positive weight."""
    w = np.maximum(agree, 0) ** 0.5
    med = np.median(w[w > 0]) if (w > 0).any() else 1.0
    return np.maximum(w, floor * med)


def k_loto_of(T, expr):
    """Per knockdown, the mean over source types t of <s_t, unit mean of the other types' rows>, each type
    cleaned of its panel top component and divided by its median row norm (NaN with < 2 types): how much
    of a knockdown's response the types agree on, the bulk amplitude."""
    Tc = []
    for t in sorted(T):
        Mx = T[t][:, expr].copy()
        X0 = np.nan_to_num(Mx)
        have = np.isfinite(Mx).mean(1) > 0.5
        if have.sum() >= 10:
            _, _, Vt = np.linalg.svd(X0[have] - X0[have].mean(0), full_matrices=False)
            v = Vt[0]
            Mx = Mx - np.outer(X0 @ v, v)
        nr = np.sqrt(np.nansum(Mx**2, 1))
        med = np.median(nr[have]) if have.any() else 1.0
        Tc.append((Mx / med, have))
    n = Tc[0][0].shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        rows = [np.nan_to_num(M[i]) for M, h in Tc if h[i]]
        if len(rows) < 2:
            continue
        lo = []
        for a in range(len(rows)):
            o = np.mean([rows[b] for b in range(len(rows)) if b != a], 0)
            lo.append(float(rows[a] @ (o / max(np.linalg.norm(o), 1e-12))))
        out[i] = max(np.mean(lo), 0.0)
    return out


def gram_whiten(P, beta=0.5, tol=None):
    """Whiten the rows of `P` against each other: singular values raised to 1 - 2 `beta` (0.5: all equal), each
    row's norm kept; with `tol`, directions below `tol` x the largest are dropped.
    """
    nrm = np.linalg.norm(P, axis=1)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    w = S * (S**2) ** (-beta)
    if tol is not None:  # drop the null directions
        keep = S > tol * S[0]
        U, w, Vt = U[:, keep], w[keep], Vt[keep]
    Pw = (U * w) @ Vt
    return Pw * (nrm / np.maximum(np.linalg.norm(Pw, axis=1), 1e-12))[:, None]


def rsvd(X, k, iters=4, seed=0):
    """Top-k right singular vectors of X by a randomized range finder."""
    rng = np.random.default_rng(seed)
    Y = X @ rng.standard_normal((X.shape[1], k + 10)).astype(X.dtype)
    for _ in range(iters):
        Q, _ = np.linalg.qr(Y)
        Y = X @ (X.T @ Q)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return Vt[:k]


def control_pcs(controls, k, n_ctl=4000, k_ctl=1000):
    """The line's own top-k principal axes: 4,000 random controls (seed 0), log1p CPM, centred."""
    n = controls.shape[0]
    rows = np.sort(np.random.default_rng(0).choice(n, min(n_ctl, n), replace=False))
    X = sp.csr_matrix(controls[rows], dtype=np.float64)
    lib_size = np.asarray(X.sum(1)).ravel()
    X = np.log1p((sp.diags(1e6 / np.maximum(lib_size, 1)) @ X).toarray()).astype(np.float32)
    X -= X.mean(0, keepdims=True)
    V = rsvd(X, k_ctl, iters=3)
    return V[:k].astype(np.float64)


def l2a(Lb, w, csum, genes, perts, own, K_l=10, form_l="half", skip=None):
    """Weak-row whitening: each knockdown below the median weight is softly whitened (in the scorer's
    space, panel targets out) against the other rows' top K axes, so weak knockdowns stop all looking like
    the panel's dominant response. Norm and own entry kept; genes the test barely sees (J < 0.05) unchanged."""
    q_l = csum / csum.sum()
    bc_l = pseudobulk(q_l)

    def _met(rows):
        """Rows of log2 changes, moved to the scorer's space: the pinned pseudobulk minus the controls'."""
        return pseudobulk(np.vstack([bulk_target(csum, x) for x in rows])) - bc_l[None, :]

    def _to_lfc(X):
        """Rows in the scorer's space, moved back to log2 changes against the controls (clipped to +-6)."""
        qh = np.expm1(bc_l[None, :] + X) / TS
        with np.errstate(divide="ignore", invalid="ignore"):
            R_ = np.log2(np.maximum(qh, 1e-12) / q_l[None, :])
        R_[:, q_l <= 0] = 0.0
        return np.clip(np.nan_to_num(R_), -6, 6)

    gi_l = {g: k for k, g in enumerate(genes)}
    pk = np.ones(len(genes), bool)
    pk[[gi_l[p] for p in perts if p in gi_l]] = False
    okg = np.log(2) * TS * q_l / (1 + TS * q_l) >= 0.05
    Xs = _met(Lb)[:, pk]
    Xu = Xs / np.maximum(np.linalg.norm(Xs, axis=1), 1e-12)[:, None]
    weak_l = w < np.median(w)
    out_l = Lb.copy()
    for j in np.flatnonzero(weak_l):
        if skip is not None and skip[j]:
            continue
        keep_r = np.ones(len(Xu), bool)
        keep_r[j] = False
        Y = Xu[keep_r]
        ev, V = np.linalg.eigh(Y @ Y.T)
        o = np.argsort(ev)[::-1][:32]
        ev, V = np.maximum(ev[o], 1e-18), V[:, o]
        A = (V.T @ Y) / np.sqrt(ev)[:, None]
        lam = ev / len(Y)
        sh = np.sqrt(lam[K_l] / (lam[:K_l] + lam[K_l])) if form_l == "half" else lam[K_l] / (lam[:K_l] + lam[K_l])
        A = A[:K_l]
        x = _met(Lb[j][None])[0]
        c = A @ x[pk]
        xp = x.copy()
        xp[pk] = x[pk] - ((1.0 - sh) * c) @ A
        dl = _to_lfc(xp[None])[0] - _to_lfc(x[None])[0]
        new = Lb[j] + np.where(okg, dl, 0.0)
        new *= np.linalg.norm(Lb[j]) / max(np.linalg.norm(new), 1e-12)
        if own[j] >= 0:
            new[own[j]] = Lb[j, own[j]]
        out_l[j] = new
    return out_l, weak_l


# ---- the p53 response ------------------------------------------------------------------------------------
# Used in: section 13.
# A line whose p53 works answers a growth-limiting knockdown with the p53 target programme (CDKN1A, MDM2, ...).
# The consensus direction averages it away, because most screened lines are p53-deficient. On a line with
# working p53 (two keys: the line's curated status, and its own controls expressing the p53 targets above
# expression-matched genes) each knockdown measured in the p53-wild-type screens (RPE1, HepG2) gets 0.5 x
# their mean response on 45 core p53 targets added to its bulk row.
# fmt: off
P53_CORE = ["CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "FAS", "TNFRSF10B", "GADD45A", "SESN1", "SESN2", "RRM2B",
            "TP53I3", "TP53INP1", "ZMAT3", "PLK3", "BTG2", "TIGAR", "DDB2", "XPC", "POLH", "FDXR", "AEN", "CCNG1",
            "TRIAP1", "APAF1", "PERP", "SFN", "GDF15", "SPATA18", "PHLDA3", "ZNF385A", "DRAM1", "PIDD1", "ISCU",
            "KITLG", "EDA2R", "SUSD6", "CMBL", "ASCC3", "FBXO22", "NINJ1", "PANK1", "PPM1D", "RRAD", "E2F7"]
P53_SOURCES = ("bench_rpe1", "bench_hepg2")
P53_KEY2 = {"genes": ["EDA2R", "ZMAT3", "MDM2", "CDKN1A", "XPC", "RRM2B", "SESN1", "DDB2", "AEN", "SPATA18", "BAX",
                      "FDXR", "CCNG1", "TNFRSF10B", "TRIAP1"], "t_wt": 0.123, "t_def": -0.2851, "nbin": 50}
# fmt: on


def p53_loadings(srcs: Sources, panel: list, genes: list, sources=P53_SOURCES) -> dict:
    """Per panel knockdown, the mean over the wild-type screens that measured it of their log2FC on the core
    p53 targets (clipped +-6, NaN -> 0); only the covered knockdowns."""
    gi = {g: i for i, g in enumerate(genes)}
    core = [g for g in P53_CORE if g in gi]
    core_ix = np.array([gi[g] for g in core])
    A = np.zeros((len(panel), len(core)))
    cnt = np.zeros(len(panel))
    for src in sources:
        s = srcs.load(src)
        tix = {t: i for i, t in enumerate(s["target"])}
        for j, p in enumerate(panel):
            if p in tix:
                row = np.asarray(s["lfc"][tix[p]], np.float64)
                A[j] += np.nan_to_num(np.clip(row[core_ix], -CLIP, CLIP))
                cnt[j] += 1
    A = A / np.maximum(cnt, 1)[:, None]
    cov = cnt > 0
    return {"keys": [p for p, c in zip(panel, cov) if c], "A": A[cov], "genes": core}


def p53_restore(Lb, perts, genes, load: dict, k: float = 0.5):
    """Each covered row gets k x its p53 loading on the core genes; its own-target entry is kept."""
    Lb = np.asarray(Lb, np.float64)
    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi[g] for g in load["genes"]])
    kz = {p: i for i, p in enumerate(load["keys"])}
    out = Lb.copy()
    n = 0
    for i, p in enumerate(perts):
        if p not in kz:
            continue
        out[i, cols] = Lb[i, cols] + k * load["A"][kz[p]]
        if p in gi:
            out[i, gi[p]] = Lb[i, gi[p]]
        n += 1
    return out, n


def p53_key2(ctrl_sum: np.ndarray, depmap_mean: np.ndarray, genes: list) -> dict:
    """Key 2 from the line's own controls: the mean residual of 15 p53 targets' log2(CPM + 1) against genes of
    matched typical expression (50 equal-count bins of DepMap's mean); thresholds from DepMap lines of known status."""
    v = np.nan_to_num(np.asarray(ctrl_sum, np.float64))
    m = np.flatnonzero((v > 0) & np.isfinite(depmap_mean))
    nbin = P53_KEY2["nbin"]
    logx = np.log2(v[m] / v[m].sum() * 1e6 + 1.0)
    order = np.argsort(depmap_mean[m], kind="stable")
    b = np.empty(len(m), np.int64)
    b[order] = np.arange(len(m)) * nbin // len(m)
    med = np.array([np.median(logx[b == i]) for i in range(nbin)])
    res = np.full(len(v), np.nan)
    res[m] = logx - med[b]
    gi = {g: i for i, g in enumerate(genes)}
    r = res[[gi[g] for g in P53_KEY2["genes"] if g in gi]]
    sc = float(np.nanmean(r))
    call = "agree wild-type" if sc >= P53_KEY2["t_wt"] else "deficient-like" if sc <= P53_KEY2["t_def"] else "ambiguous"
    return {"score": sc, "n_genes": int(np.isfinite(r).sum()), "call": call}


# ---- the per-knockdown targets (stage 1's output) --------------------------------------------------------
# Stage 1 for one line: every piece of section 10 in order, ending in each knockdown's pooled change and its per-
# cell change, the two targets stage 2 draws cells for.
# Used in: sections 10 and 13.
# The pieces of the final recipe, each switchable for the waterfall on held-out lines (all on = the final).
PIECES = (
    "gate",
    "weakwh",
    "whiten_tol",
    "cis3",
    "agree_clean",
    "k_loto",
    "ps20",
    "s2",
    "e",
    "l2a",
    "cell_caps",
    "down_floor",
)


def table_multiplier(L, cpm, table):
    """The per-cell amplitude table: a multiplier by |L| bin x control-CPM bin."""
    bl = np.digitize(np.abs(L), CELL_L_EDGES)
    be = np.digitize(cpm, CELL_E_EDGES)[None, :].repeat(L.shape[0], 0)
    return np.asarray(table)[bl, be]


def moments_probe(controls, perts, genes, L, cis, T, budget=0.9, cell=2.0, floor=0.25):
    """The probe's targets (an older, simpler recipe, used only to measure where this line's test starts
    calling genes): bulk s w_p L_p + cis, cells 2 L_p + cis, own gene at min(L, log2 0.2)."""
    gi = {g: i for i, g in enumerate(genes)}
    own = np.array([gi.get(p, -1) for p in perts])
    ccpm = mean_cpm(controls)
    w = weights(agreement(T, ccpm > 5), floor)
    csum = np.asarray(controls.sum(axis=0)).ravel()
    C = control_noise(controls, 400, 8, 0)
    s = bulk_scale(L * w[:, None], csum, own, budget * C * len(perts))
    out = {}
    for i, p in enumerate(perts):
        lb = s * w[i] * L[i] + cis[i]
        lc = cell * L[i] + cis[i]
        if own[i] >= 0:
            lb[own[i]] = lc[own[i]] = min(L[i, own[i]], np.log2(0.2))
        out[p] = (lb, lc)
    return out, {"scale": s, "C_kd": C}


def moments_final(controls, perts, genes, ctx, inp, cfg):
    """The final recipe's targets for one context: {pert: (bulk lfc, per-cell lfc)}.

    Bulk: the ps20 direction, its weak rows Gram-whitened, entries outside the sources' gate x0.3, an
    amplitude per knockdown from the types' leave-one-type-out agreement (k_loto), the loudest 20% pulled
    toward the line's own control axes (S2), E blended into the loudest well-covered rows, weak-row
    whitening (l2a), the p53 response restored on a line whose p53 works, all scaled so the profiles'
    spread is the measured noise budget; plus 3 x cis.
    Cells: the stack direction x max(2, table), compressed above the detection line, capped at the line
    outside the gate, small changes lifted to a floor (0.75 x the line up, 0.5 x down); plus cis."""
    on = set(cfg.get("pieces", PIECES)).__contains__  # the waterfall switches every piece; all on = the final
    gi = {g: i for i, g in enumerate(genes)}
    own = np.array([gi.get(p, -1) for p in perts])
    L = (inp["ps_dir"] if on("ps20") else inp["stack_dir"]).astype(np.float64)
    Lc = inp["stack_dir"].astype(np.float64)
    cis = inp["cis"].astype(np.float64)
    T = inp["T"]
    ccpm = mean_cpm(controls)
    floor = cfg["floor"]
    # agreement of the type rows after each type loses its panel top component (the bulk's key)
    expr = ccpm > 5
    Tc = {}
    for t, Mt in T.items():
        Xe = np.nan_to_num(Mt[:, expr])
        have = np.isfinite(Mt[:, expr]).mean(1) > 0.5
        Z = np.full(Mt.shape, np.nan)
        if have.sum() >= 10:
            _, _, Vt = np.linalg.svd(Xe[have] - Xe[have].mean(0), full_matrices=False)
            v = Vt[0]
            Z[:, expr] = Mt[:, expr] - np.outer(Xe @ v, v)
        else:
            Z[:, expr] = Mt[:, expr]
        Tc[t] = Z
    agree_c = agreement(Tc, expr)
    agree = agreement(T, ccpm > 5)
    w = weights(agree, floor)
    medB = np.asarray(inp["medB"][ctx], np.float64)
    medB = np.where(np.isfinite(medB), medB, np.nanmedian(medB))
    w = np.maximum(agree, 0) ** 0.5 * medB**0.5
    w = np.maximum(w / np.median(w[w > 0]), floor)
    w_c = np.maximum(agree_c, 0) ** 0.5 * medB**0.5
    w_c = np.maximum(w_c / np.median(w_c[w_c > 0]), floor)
    Lb = L.copy()
    nrm = np.linalg.norm(Lb, axis=1)
    if on("weakwh"):  # weak rows (below the median weight) fully Gram-whitened
        W = gram_whiten(Lb, tol=cfg["whiten_tol"] if on("whiten_tol") else None)
        t_ = np.where(w < np.median(w), 1.0, 0.0)
        Lb = (1 - t_)[:, None] * Lb + t_[:, None] * W
        Lb = Lb * (nrm / np.maximum(np.linalg.norm(Lb, axis=1), 1e-12))[:, None]
    if on("gate"):
        keep = inp["gate"].astype(bool) & (ccpm > 5)[None, :]
        Lb = np.where(keep, Lb, cfg["gate_bg"] * Lb)
        Lb = Lb * (nrm / np.maximum(np.linalg.norm(Lb, axis=1), 1e-12))[:, None]
    csum = np.asarray(controls.sum(axis=0)).ravel()
    wb = w_c if on("agree_clean") else w
    # the amplitude: k_loto (fallback: the current amplitude x the median ratio), floored, cis rescaled with it
    info = {}
    cis_r = np.ones(len(perts))
    if on("k_loto"):
        kl = k_loto_of(T, ccpm > 5)
        nb = np.linalg.norm(Lb, axis=1)
        cur = wb * nb
        okl = np.isfinite(kl) & (nb > 0)
        ratio = float(np.median(kl[okl] / np.maximum(cur[okl], 1e-12)))
        amp_new = np.where(okl, kl / ratio, cur)
        amp_new = np.maximum(amp_new, 0.25 * np.median(cur[nb > 0]))
        cis_r = np.where(cur > 0, amp_new / np.maximum(cur, 1e-12), 1.0)
        wb = np.where(nb > 0, amp_new / np.maximum(nb, 1e-12), wb)
        info.update(loto_n=int(okl.sum()), loto_ratio=ratio)
    if on("s2"):  # S2: the loudest rows pulled toward the line's own control axes (PCs K0..K), norms kept
        k_s, lam_s, f_s, k0_s = cfg["smooth"]
        nb_s = np.linalg.norm(Lb, axis=1)
        rows_s = np.argsort(-(wb * nb_s))[: round(f_s * len(perts))]
        p0 = csum / csum.sum()
        dslope = (TS * p0) / (1.0 + TS * p0)
        msk = ccpm >= 2.0
        Vc = control_pcs(controls, k_s)[:, msk]
        qv, _ = np.linalg.qr(Vc.T)
        Vc = qv.T[k0_s - 1 :]
        Y = Lb[rows_s][:, msk] * dslope[msk][None, :]
        Pj = (Y @ Vc.T) @ Vc
        blk = Lb[rows_s].copy()
        blk[:, msk] = (Pj + lam_s * (Y - Pj)) / dslope[msk][None, :]
        blk[:, ~msk] *= lam_s
        blk = blk * (nb_s[rows_s] / np.maximum(np.linalg.norm(blk, axis=1), 1e-12))[:, None]
        Lb = Lb.copy()
        Lb[rows_s] = blk
    info["learn_rows"] = 0
    if on("e"):  # E: the learned direction blended into the loudest rows covered by >= 4 source types
        g_, top_, mint_, qc_ = cfg["learn"]
        Lb = Lb.copy()
        kz = {k: j for j, k in enumerate(inp["E_keys"])}
        XC = inp["E_XC"][ctx]
        q_ = csum / csum.sum()
        bc_ = pseudobulk(q_)
        amp = wb * np.linalg.norm(Lb, axis=1)
        loud = amp >= np.quantile(amp, 1 - top_)
        expr_ = q_ * TS > 0.1
        nty = sum((np.isfinite(Mt[:, expr_]).mean(1) > 0.3).astype(int) for Mt in T.values())
        loud &= nty >= mint_
        XCall = np.zeros((len(perts), len(genes)))
        for j, p in enumerate(perts):
            if p in kz:
                XCall[j] = XC[kz[p]]
                if own[j] >= 0:
                    XCall[j, own[j]] = 0.0
        XCm = XCall.mean(0)
        del XCall
        loud_qc = amp >= np.quantile(amp, 1 - qc_)
        n_bl = 0
        for i in np.flatnonzero(loud):
            if perts[i] not in kz:
                continue
            kp_ = np.ones(len(genes), bool)
            if own[i] >= 0:
                kp_[own[i]] = False
            x21 = (pseudobulk(bulk_target(csum, Lb[i])) - bc_) * kp_
            xl = XC[kz[perts[i]]].astype(np.float64) * kp_
            if not loud_qc[i]:
                xl = xl - XCm
            if np.linalg.norm(xl) == 0 or np.linalg.norm(x21) == 0:
                continue
            xb = (1 - g_) * x21 / np.linalg.norm(x21) + g_ * xl / np.linalg.norm(xl)
            qh = np.expm1(bc_ + xb) / TS
            with np.errstate(divide="ignore", invalid="ignore"):
                R = np.log2(np.maximum(qh, 1e-12) / q_)
            R[q_ <= 0] = 0.0
            R = np.clip(np.nan_to_num(R), -6, 6)
            own_v = Lb[i, own[i]] if own[i] >= 0 else None
            nbi = np.linalg.norm(Lb[i])
            Lb[i] = R / max(np.linalg.norm(R), 1e-12) * nbi
            if own_v is not None:
                Lb[i, own[i]] = own_v
            n_bl += 1
        info["learn_rows"] = n_bl
    info["l2a_rows"] = 0
    if on("l2a"):
        Lb, weak_l = l2a(Lb, w, csum, genes, perts, own, *cfg["l2a"])
        info["l2a_rows"] = int(weak_l.sum())
    p53 = (inp.get("tp53") or {}).get(ctx)
    info["tp53_rows"] = 0
    if p53 is not None:  # the line's p53 works (both keys): its target programme restored on the covered rows
        Lb, info["tp53_rows"] = p53_restore(Lb, perts, genes, p53, cfg["tp53_k"])
    C = control_noise(controls, 400, 8, 0)
    s = bulk_scale(Lb * wb[:, None], csum, own, cfg["budget"] * C * len(perts))
    dl = inp["detline"]
    thr = dl["a"] * np.maximum(ccpm, 5.0) ** dl["b"]
    Ttab = np.minimum(np.asarray(CELL_TABLE), cfg["cell_table_cap"])
    M = table_multiplier(Lc, ccpm, Ttab)
    gate_cells = inp["gate"].astype(bool)
    out = {}
    for i, p in enumerate(perts):
        lb = s * wb[i] * Lb[i] + (cfg["bulk_cis_k"] if on("cis3") else 1.0) * cis_r[i] * cis[i]
        dpart = np.maximum(cfg["cell"], M[i]) * Lc[i]
        if on("cell_caps"):  # compression above the knee, then the cap outside the gate
            u = np.abs(dpart) / thr
            big = (ccpm > 5.0) & (u > cfg["comp_knee"])
            dpart = np.where(
                big,
                np.sign(dpart)
                * thr
                * cfg["comp_knee"]
                * np.power(np.maximum(u, 1e-12) / cfg["comp_knee"], cfg["comp_kappa"]),
                dpart,
            )
        lc = dpart + cis[i]
        if on("cell_caps"):
            capm = (~gate_cells[i]) & (ccpm > 5.0) & (np.abs(cis[i]) < 1e-6) & (np.abs(lc) > cfg["cells_gatecap"] * thr)
            lc = np.where(capm, np.sign(lc) * cfg["cells_gatecap"] * thr, lc)
        flv = np.where(Lc[i] < 0, cfg["de_floor_down"] if on("down_floor") else cfg["de_floor"], cfg["de_floor"])
        lift = (ccpm > 5.0) & (np.abs(lc) < flv * thr) & (Lc[i] != 0)
        lc = np.where(lift, np.sign(Lc[i]) * flv * thr, lc)
        if own[i] >= 0:
            lb[own[i]] = min(Lb[i, own[i]], np.log2(0.2))
            lc[own[i]] = min(Lc[i, own[i]], np.log2(0.2))
        out[p] = (lb, lc)
    pb = pseudobulk(np.vstack([bulk_target(csum, out[p][0]) for p in perts]))
    info.update(C_kd=C, scale=s, B_over_sumC=spread(pb, own) / (C * len(perts)))
    return out, info
