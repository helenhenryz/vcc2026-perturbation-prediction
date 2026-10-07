"""The public screens as tables, and what stage 1 borrows from them.

Each knockdown's borrowed profile (the effect direction), the significance gate and the neighbour (cis) prior.
Every function is plain numpy. The arithmetic and the order of every floating-point sum are those of the build
that made the submitted file, which this code rebuilds byte for byte: a different summation order moves the
near-zero entries.
"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

__all__ = [
    "CIS_CPM_EDGES",
    "CIS_DEPTH",
    "CIS_EDGES",
    "CLIP",
    "DEPTH",
    "DLD",
    "FLEX_TYPES",
    "LN2SQ",
    "N_CTRL_DEFAULT",
    "ORION",
    "SE_INFLATE",
    "TS",
    "TYPE_W",
    "Pool",
    "Sources",
    "blas_threads",
    "cis_prior",
    "combine",
    "direction",
    "drop_pcs",
    "drop_shared_and_whiten",
    "fallback_shrunk",
    "gate",
    "main_sources",
    "neighbour_table",
    "own_out_mean",
    "per_type_rows",
    "probe_direction",
    "rescale_rows",
    "sha256_file",
    "shrink",
    "source_table",
    "source_tables",
    "undamped",
    "whiten_shrunk",
]

# ---- setup -----------------------------------------------------------------------------------------------
# Used in: every section.
TS = 5e4  # the scorer's pseudobulk target sum
CLIP = 6.0  # log2 fold changes are clipped to +-6 everywhere


@contextmanager
def blas_threads(n: int):
    """Fix the BLAS thread count for one step (a context manager).

    Matrix products split their sums by thread, so the rounding of near-zero entries (and their sign) depends on
    the thread count; each step runs at the count the submitted file was built with.
    """
    with threadpool_limits(limits=int(n)):
        yield


def sha256_file(path) -> str:
    """The sha256 of the file at `path`, as hex, read in 16 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---- sources ---------------------------------------------------------------------------------------------
# Used in: sections 4, 6 and 13.
# A source is one public Perturb-seq screen reduced to a table: per knocked-down gene (row), the log2 fold
# change of every gene on the challenge's 18,533-gene axis (NaN = not measured), with the cells behind each
# row and the screen's control expression. The noise of each entry is the analytic standard error of a
# log2 ratio of two negative-binomial means (delta method), inflated by 1.25 (split halves showed the
# analytic variance is 1.19-1.28x too small).
# fmt: off
DEPTH = {  # median UMI per cell of each screen's own cells
    "bench_h1": 20_002, "bench_k562": 9_288, "bench_rpe1": 8_453, "bench_hepg2": 11_412, "bench_jurkat": 7_322,
    "k562gwps_bulk": 7_773, "arc_h1": 20_000, "kolf": 6_000, "cd4_Rest": 10_080, "cd4_Stim8hr": 14_977,
    "cd4_Stim48hr": 13_373, "viperturb": 8_000, "a549_myllia": 12_217, "a375_perturbfate": 2_507,
    "a549_ko": 25_827, "hesc_gs": 9_264, "h358_cas12a": 13_128, "gastric_i": 11_291, "dld1_gx": 5_708,
    "orion_hct116": 12_401, "orion_hek293t": 15_329,
}
# fmt: on
N_CTRL_DEFAULT = 10_000
SE_INFLATE = 1.25
LN2SQ = np.log(2.0) ** 2


class Sources:
    """An ordered set of source tables. `files` maps name -> (npz path, json path or None); the order of
    `files` is the order every consensus sums its sources in. `phi` holds the per-gene overdispersion of
    the bench lines' controls (keys phi_<line>, phi_median). `extra` names the sources whose own json
    gives their control-cell count (the newer loaders); the others use 10,000."""

    def __init__(self, files: dict, phi: dict, extra=()):
        """`files`: {name: (npz path, json path or None)}, in summing order; `phi`: the overdispersion arrays;
        `extra`: the sources whose json gives their control-cell count and depth.
        """
        self.files = dict(files)
        self.phi = phi
        self.extra = set(extra)

    def names(self) -> list:
        """The source names, in the order every consensus sums them."""
        return list(self.files)

    def subset(self, names):
        """The same sources restricted to `names`, in that order."""
        return Sources({n: self.files[n] for n in names}, self.phi, self.extra & set(names))

    def load(self, name: str) -> dict:
        """One source's table: {target, lfc (knockdowns x genes, float32), n_cells, cell_type, chem, ctrl_cpm}."""
        z = np.load(self.files[name][0], allow_pickle=True)
        s = {
            "target": z["target"].astype(str),
            "lfc": z["lfc"].astype(np.float32),
            "n_cells": np.asarray(z["n_cells"], dtype=np.float64),
            "cell_type": str(z["cell_type"]),
            "chem": str(z["chem"]),
            "ctrl_cpm": np.asarray(z["ctrl_cpm"], dtype=np.float64),
        }
        return s

    def _json(self, name):
        """The source's json metadata ({} for a source without one)."""
        j = self.files[name][1]
        return json.loads(Path(j).read_text()) if (name in self.extra and j and Path(j).exists()) else {}

    def depth_of(self, name: str) -> float:
        """The median UMIs per cell of the source's screen: DEPTH, else its json, else 8,000."""
        if name in DEPTH:
            return DEPTH[name]
        mt = self._json(name).get("median_total")
        return float(mt) if mt else 8_000

    def n_ctrl_of(self, name: str) -> float:
        """The source's number of control cells: its json, else 10,000."""
        c = self._json(name).get("control_cells")
        return float(c) if c else float(N_CTRL_DEFAULT)

    def phi_for(self, name: str) -> np.ndarray:
        """The per-gene overdispersion for the source: its own training line's, else the median over the lines."""
        key = f"phi_{name.removeprefix('bench_')}"
        return self.phi[key] if key in self.phi else self.phi["phi_median"]

    def se2(self, name: str, s: dict | None = None) -> np.ndarray:
        """Analytic variance of every entry of the source's table (NaN where unmeasured)."""
        s = s or self.load(name)
        depth = self.depth_of(name)
        mu = np.nan_to_num(s["ctrl_cpm"], nan=0.0) * depth / 1e6  # control counts per cell
        phi = self.phi_for(name)
        n = s["n_cells"][:, None]
        nc = self.n_ctrl_of(name)
        fc = np.exp2(np.clip(np.nan_to_num(s["lfc"], nan=0.0), -CLIP, CLIP))
        with np.errstate(divide="ignore", invalid="ignore"):
            term_p = (1.0 + phi[None, :] * mu[None, :] * fc) / (n * mu[None, :] * fc)
            term_c = (1.0 + phi * mu) / (nc * mu)
            v = (term_p + term_c[None, :]) / LN2SQ
        v = SE_INFLATE * np.where(np.isfinite(s["lfc"]) & (mu[None, :] > 0), v, np.nan)
        return v.astype(np.float32)


# ---- per-source tables over a panel ----------------------------------------------------------------------
# Each screen's table cut to the phase's panel and gene axis: the rows stage 1 borrows, with the noise of each entry
# and a per-gene mean response.
# Used in: section 13 (and the folds of sections 8, 10, 15).
def own_out_mean(L: np.ndarray, targets, gidx: dict) -> np.ndarray:
    """Per gene, the mean over a source's knockdowns of its (clipped) change, each gene's own knockdown left out."""
    ok = np.isfinite(L)
    colsum = np.where(ok, L, 0).sum(0, dtype=np.float64)
    colcnt = ok.sum(0).astype(np.float64)
    for i, t in enumerate(targets):
        j = gidx.get(t)
        if j is not None and ok[i, j]:
            colsum[j] -= L[i, j]
            colcnt[j] -= 1
    return np.where(colcnt > 0, colsum / np.maximum(colcnt, 1), 0.0).astype(np.float32)


def source_table(srcs: Sources, name: str, keys: list, gidx: dict) -> dict:
    """One source restricted to `keys`: its clipped rows and their variances (NaN = absent), and three
    per-gene summaries over ALL its rows: the mean with each gene's own knockdown left out (`cmean`, the
    centring), the plain mean (`generic`, the shift any knockdown causes) and mean(l^2 - se^2) (`tpart`,
    the signal variance used by the empirical-Bayes shrinkage)."""
    s = srcs.load(name)
    L = np.clip(s["lfc"], -CLIP, CLIP)
    if name.startswith("jiang_"):  # small pathway screens: a flat variance
        V = np.where(np.isfinite(L), np.float32(0.35**2), np.nan).astype(np.float32)
    else:
        V = srcs.se2(name, s)
    idx = {t: i for i, t in enumerate(s["target"])}
    r = np.array([idx.get(k, -1) for k in keys])
    rows = np.full((len(keys), L.shape[1]), np.nan, np.float32)
    se2 = np.full_like(rows, np.nan)
    rows[r >= 0] = L[r[r >= 0]]
    se2[r >= 0] = V[r[r >= 0]]
    cmean = own_out_mean(L, s["target"], gidx)
    generic = np.nan_to_num(np.nanmean(L, axis=0)).astype(np.float32)
    okv = np.isfinite(V)
    tterm = np.where(okv, L.astype(np.float64) ** 2 - V, 0.0).sum(0)
    tcnt = okv.sum(0)
    tpart = np.where(tcnt > 0, tterm / np.maximum(tcnt, 1), 0.0).astype(np.float32)
    present = r >= 0
    return {
        "type": s["cell_type"],
        "chem": s["chem"],
        "present": present,
        "rows": rows[present] if present.any() else None,
        "se2": se2[present] if present.any() else None,
        "cmean": cmean,
        "generic": generic,
        "tpart": tpart,
    }


def source_tables(srcs: Sources, keys: list, genes: list, log=print) -> dict:
    """`source_table` for every source of `srcs`, in its order: {name: table}; `log` reports each one's coverage."""
    gidx = {g: i for i, g in enumerate(genes)}
    out = {}
    for n in srcs.names():
        out[n] = source_table(srcs, n, keys, gidx)
        log(f"table {n}: {int(out[n]['present'].sum())} of {len(keys)} knockdowns")
    return out


class Pool:
    """The per-source tables of one source set, in that set's order."""

    def __init__(self, tables: dict, names: list, keys: list):
        """`tables`: {name: source_table(...)}; `names`: this pool's sources, in summing order; `keys`: the panel."""
        self.keys = list(keys)
        self.src = {n: tables[n] for n in names}
        self.G = len(next(iter(self.src.values()))["cmean"])

    def allowed(self, excl: set) -> list:
        """The pool's sources whose cell type is not in `excl`, in order."""
        return [n for n, s in self.src.items() if s["type"] not in excl]

    def tau(self, names) -> np.ndarray:
        """Per gene, the signal variance the shrinkage uses: the mean `tpart` of `names` (pathway screens left out),
        at least 1e-4.
        """
        t = [self.src[n]["tpart"].astype(np.float64) for n in names if not n.startswith("jiang_")]
        return np.clip(np.mean(t, 0), 1e-4, None)

    def generic(self, names, centre=False) -> np.ndarray:
        """Per gene, the mean shift any knockdown causes over `names` (minus each source's `cmean` if `centre`)."""
        g = [self.src[n]["generic"] - (self.src[n]["cmean"] if centre else 0.0) for n in names]
        return np.mean(g, 0).astype(np.float64)

    def per_type(self, names, centre=False):
        """{type: (L_t, V_t, n_src_t)}: the inverse-variance mean of each cell type's sources."""
        acc = {}
        n = len(self.keys)
        for name in names:
            s = self.src[name]
            if s["rows"] is None:
                continue
            pr = s["present"]
            L = s["rows"].astype(np.float64)
            if centre:
                L = L - s["cmean"][None, :]
            V = s["se2"].astype(np.float64)
            ok = np.isfinite(L) & np.isfinite(V)
            w = np.where(ok, 1.0 / np.maximum(V, 1e-6), 0.0)
            t = s["type"]
            if t not in acc:
                acc[t] = [np.zeros((n, self.G)), np.zeros((n, self.G)), np.zeros(n)]
            acc[t][0][pr] += w
            acc[t][1][pr] += w * np.where(ok, L, 0.0)
            acc[t][2] += pr
        return {
            t: (
                np.where(w > 0, num / np.maximum(w, 1e-12), np.nan),
                np.where(w > 0, 1.0 / np.maximum(w, 1e-12), np.nan),
                ns,
            )
            for t, (w, num, ns) in acc.items()
        }


# ---- the effect direction --------------------------------------------------------------------------------
# The effect direction: for each knockdown, the screens' profiles averaged per cell type, the types weighted (K562 x2,
# the Flex-chemistry screens x1, the others x0.5), noisy entries shrunk towards the mean response, the component every
# knockdown shares removed and the genes whitened. This is the shape of the predicted change before its size is set.
# Used in: section 13 (and the folds).
FLEX_TYPES = {"h1", "cd4t"}  # screens on the challenge's own chemistry (10x Flex)
DLD = "dld1_gx"
ORION = ("orion_hct116", "orion_hek293t")
TYPE_W = {"h1": 1.0, "cd4t": 1.0, "k562": 2.0, "dld1": 0.5, "hct116": 0.5, "hek293t": 0.5}  # every other type 0.5


def combine(pt: dict, weights: dict):
    """Across cell types: a weighted mean of the per-type rows (weights per type; missing = 1).
    Returns lfc, se2 (NaN where no type measured) and the number of sources per row."""
    num = den = vnum = None
    ns = None
    for t, (L, V, n) in pt.items():
        w = weights.get(t, 1.0)
        if np.isscalar(w) and w == 0:
            continue
        ok = np.isfinite(L)
        wt = np.where(ok, w, 0.0)
        a = wt * np.where(ok, L, 0.0)
        b = wt * wt * np.where(ok, V, 0.0)
        num = a if num is None else num + a
        den = wt if den is None else den + wt
        vnum = b if vnum is None else vnum + b
        ns = n.copy() if ns is None else ns + n
    lfc = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    se2 = np.where(den > 0, vnum / np.maximum(den, 1e-12) ** 2, np.nan)
    return lfc, se2, ns


def shrink(lfc, se2, ns, tau, generic, c=0.5):
    """Empirical-Bayes shrinkage of each entry toward 0 by its noise (c scales the noise); rows no source
    measured get the generic shift."""
    post = lfc * (tau[None, :] / (tau[None, :] + c * se2))
    post = np.where(np.isfinite(post), post, 0.0)
    post[ns == 0] = generic[None, :]
    return post


def drop_pcs(P, k, beta=0.0):
    """Remove the top k singular components of the panel (the response every knockdown shares)."""
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    keep = np.zeros(k)
    keep[-1] = beta
    return P - (U[:, :k] * (S[:k] * (1 - keep))) @ Vt[:k]


def whiten_shrunk(P, a, lam=1.0):
    """Divide each gene by its spread across the panel (shrunk toward the median spread), to the power a."""
    sd = P.std(0)
    med = np.median(sd[sd > 0])
    sd_s = (sd + lam * med) / (1 + lam)
    return P / ((sd_s / med) ** a)[None, :]


def drop_shared_and_whiten(P):
    """The direction's last steps: remove the panel's top component, then whiten each gene (power 0.75)."""
    return whiten_shrunk(drop_pcs(P, 1), 0.75, 1.0)


def main_sources(pool, excl):
    """The main sources: everything but the iPSC atlas, the pathway screens, DLD-1 and Orion."""
    return [
        n
        for n in pool.allowed(excl)
        if pool.src[n]["type"] != "ipsc" and not n.startswith("jiang_") and n != DLD and not n.startswith("orion_")
    ]


def undamped(name: str, kind: str) -> bool:
    """True for the sources that are neither damped nor cleaned of their own axes: the iPSC atlas and the pathway
    screens.
    """
    return kind == "ipsc" or name.startswith("jiang_")


def per_type_rows(pool, names, pcs=None, k=0, c=0.0, k0_of=None):
    """Per cell type, the inverse-variance mean of its sources' centred rows. Main sources (not the iPSC
    atlas or the pathway screens) first lose their own top-k principal components (`pcs[name]`, the
    screen's batch axes) and, with c > 0, are damped per gene by k0 / (k0 + c), k0 the screen's control
    CPM (a pseudocount: genes a screen barely detects carry noisy ratios)."""
    n, G = len(pool.keys), pool.G
    acc = {}
    for nm in names:
        s = pool.src[nm]
        if s["rows"] is None:
            continue
        pr = s["present"].copy()
        X = s["rows"].astype(np.float64) - s["cmean"][None, :]
        main = not undamped(nm, s["type"])
        if main and k > 0:
            V = pcs[nm][:k]
            if len(V):
                ok0 = np.isfinite(X)
                X0 = np.where(ok0, X, 0.0)
                X = np.where(ok0, X0 - (X0 @ V.T) @ V, np.nan)
        V2 = s["se2"].astype(np.float64)
        if c > 0 and main:
            k0 = k0_of(nm)
            d = k0 / (k0 + c)
            X = X * d[None, :]
            V2 = V2 * (d**2)[None, :]
        ok = np.isfinite(X) & np.isfinite(V2)
        w = np.where(ok, 1.0 / np.maximum(V2, 1e-6), 0.0)
        a = acc.setdefault(s["type"], [np.zeros((n, G)), np.zeros((n, G)), np.zeros(n)])
        a[0][pr] += w
        a[1][pr] += w * np.where(ok, X, 0.0)
        a[2] += pr
    return {
        t: (
            np.where(w > 0, num / np.maximum(w, 1e-12), np.nan),
            np.where(w > 0, 1.0 / np.maximum(w, 1e-12), np.nan),
            ns,
        )
        for t, (w, num, ns) in acc.items()
    }


def fallback_shrunk(pool, excl):
    """For knockdowns no main source measured: the iPSC atlas and the pathway screens, equal across types."""
    names = [n for n in pool.allowed(excl) if pool.src[n]["type"] == "ipsc" or n.startswith("jiang_")]
    pt = per_type_rows(pool, names)
    if not pt:  # no fallback source measured any knockdown of this panel (a small panel): no fallback rows
        return np.zeros((len(pool.keys), pool.G)), np.zeros(len(pool.keys))
    lfc, se2, ns = combine(pt, {})
    base = main_sources(pool, excl)
    return shrink(lfc, se2, ns, pool.tau(base), pool.generic(base, True), 0.25), ns


def direction(
    pool, type_w: dict, orion: bool, pcs: dict, srcpc_k: int = 1, ps_c: float = 0.0, k0_of=None, c: float = 0.25
):
    """The effect direction for pool.keys: main sources + DLD-1 (+ Orion), per-type means, type weights,
    shrinkage, the shared component removed and genes whitened, then the fallback rows. Returns (P, info)."""
    excl = set()
    names = main_sources(pool, excl)
    if DLD in pool.allowed(excl):
        names.append(DLD)
    if orion:
        names += [n for n in ORION if n in pool.allowed(excl)]
    base = main_sources(pool, excl)
    pt = per_type_rows(pool, names, pcs, srcpc_k, ps_c, k0_of)
    tw = {t: float(type_w.get(t, 0.5)) for t in pt}
    lfc, se2, ns = combine(pt, tw)
    P0 = shrink(lfc, se2, ns, pool.tau(base), pool.generic(base, True), c)
    P = drop_shared_and_whiten(P0)
    Pf, nf = fallback_shrunk(pool, excl)
    fill = np.flatnonzero((ns == 0) & (nf > 0))
    if len(fill):  # through the main direction's own transforms (its top axis, its whitening factors)
        _, _, Vt = np.linalg.svd(P0, full_matrices=False)
        v = Vt[0]
        Q = drop_pcs(P0, 1)
        sd = Q.std(0)
        med = np.median(sd[sd > 0])
        fac = (((sd + med) / 2) / med) ** 0.75
        F = Pf[fill] - np.outer(Pf[fill] @ v, v)
        P[fill] = F / fac[None, :]
    return P.astype(np.float64), {"types": tw, "n_fill": len(fill), "names": names}


def rescale_rows(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Each row of P (float32) at ref's row norm; a zero row stays zero."""
    nr, nn = np.linalg.norm(ref, axis=1), np.linalg.norm(P, axis=1)
    return (P * np.where(nn > 0, nr / np.maximum(nn, 1e-12), 0.0)[:, None]).astype(np.float32)


def probe_direction(pool):
    """The probe's direction (an older, simpler recipe): every non-iPSC, non-pathway source, per-type
    inverse-variance means, flex 1 / K562 2 / others 0.5, shrinkage c 0.25, top component removed,
    genes whitened at 0.75."""
    allowed = pool.allowed(set())
    no_ij = [n for n in allowed if pool.src[n]["type"] != "ipsc" and not n.startswith("jiang_")]
    pt = pool.per_type(no_ij, centre=True)
    tau = pool.tau(no_ij)
    gen = pool.generic(no_ij, True)
    w = {t: (1.0 if t in FLEX_TYPES else 2.0 if t == "k562" else 0.0 if t == "ipsc" else 0.5) for t in pt}
    P = shrink(*combine(pt, w), tau, gen, 0.25)
    return whiten_shrunk(drop_pcs(P, 1), 0.75, 1.0).astype(np.float32)


# ---- the significance gate -------------------------------------------------------------------------------
# The significance gate: which entries of a knockdown's profile any screen measured as clearly non-zero; the rest
# are quietened (the `gate` piece of section 10).
# Used in: sections 10 and 13.
def gate(pool, orion: bool, z: float = 3.0) -> np.ndarray:
    """keep[p, g]: some source of the direction called gene g in knockdown p at |lfc| / se > z."""
    names = main_sources(pool, set())
    if DLD in pool.allowed(set()):
        names.append(DLD)
    if orion:
        names += [n for n in ORION if n in pool.allowed(set())]
    cnt = np.zeros((len(pool.keys), pool.G))
    for nm in names:
        s = pool.src[nm]
        if s["rows"] is None:
            continue
        zz = np.abs(s["rows"].astype(float)) / np.sqrt(np.maximum(s["se2"].astype(float), 1e-6))
        cnt[s["present"]] += np.where(np.isfinite(zz) & (zz > float(z)), 1.0, 0.0)
    return (cnt >= 1).astype(np.uint8)


# ---- the neighbour (cis) prior ---------------------------------------------------------------------------
# Used in: sections 10 and 13.
# CRISPRi silences a promoter region, so genes whose transcription start sites sit near the target's are
# silenced too. The prior is a table fitted on Flex screens (h1, CD4 T, K562): log2 change by distance
# bin x orientation, times a factor by the neighbour's control expression, times the target's own
# knockdown depth log2(0.2).
CIS_EDGES = [0, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
CIS_CPM_EDGES = [0, 1, 5, 20, 100, np.inf]
CIS_DEPTH = float(np.log2(0.2))


def neighbour_table(tss: pd.DataFrame, genes: list, max_d: int = 200_000) -> dict:
    """symbol -> (axis column of each on-axis neighbour within max_d, distance, orientation class)."""
    gi = {g: i for i, g in enumerate(genes)}
    out = {}
    for _, g in tss.groupby("chrom"):
        g = g.sort_values("tss")
        pos = g["tss"].to_numpy()
        names = g["gene_name"].to_numpy()
        plus = (g["strand"] == "+").to_numpy()
        col = np.array([gi.get(n, -1) for n in names])
        for i in range(len(g)):
            lo = np.searchsorted(pos, pos[i] - max_d)
            hi = np.searchsorted(pos, pos[i] + max_d, side="right")
            j = np.arange(lo, hi)
            j = j[(j != i) & (col[j] >= 0)]
            s = 1 if plus[i] else -1
            rel = (pos[j] - pos[i]) * s  # > 0: downstream of the target's TSS
            same = plus[j] == plus[i]
            cls = np.where(same, np.where(rel > 0, 3, 2), np.where(rel > 0, 1, 0))
            out[names[i]] = (col[j], np.abs(pos[j] - pos[i]), cls)
    return out


def cis_prior(keys: list, genes: list, ctrl_cpm: np.ndarray, nb: dict, prior: dict) -> np.ndarray:
    """(keys x genes) float32: the predicted neighbour silencing of each knockdown, own column 0."""
    gi = {g: i for i, g in enumerate(genes)}
    tab = np.asarray(prior["table"])
    m = np.asarray(prior["cpm_mult"])
    out = np.zeros((len(keys), len(genes)), np.float32)
    for r, k in enumerate(keys):
        v = np.zeros(len(genes), np.float32)
        if k in nb:
            cols, d, cls = nb[k]
            b = np.searchsorted(CIS_EDGES, d, side="left") - 1
            b = np.where(d == 0, 0, b)
            b = np.where(d > CIS_EDGES[-1], -1, b)
            keep = b >= 0
            cols, b, cls = cols[keep], b[keep], cls[keep]
            cb = np.clip(np.searchsorted(CIS_CPM_EDGES, np.nan_to_num(ctrl_cpm[cols], nan=0.0), side="right") - 1, 0, 4)
            v[cols] = tab[b, cls] * m[cb] * (CIS_DEPTH if prior["mode"] == "ratio" else 1.0)
        out[r] = v
        if k in gi:
            out[r, gi[k]] = 0.0
    return out
