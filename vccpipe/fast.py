"""FAST mode's small inputs, read from the public files by byte range."""

import hashlib
import io
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .generator import mean_cpm
from .phase import GeneMap
from .rebuild import (
    BENCH_LINES,
    PUBLIC_FILES,
    align_genes,
    build_reference,
    fetch,
    halves_table,
    save_table,
    split_labels,
    table_bench,
    table_gwps_bulk,
    table_phi,
    tables_jiang,
)

__all__ = [
    "FAST_CELLS",
    "FAST_CONNECTIONS",
    "FAST_CONTROLS",
    "FAST_H1",
    "FAST_H1_ROWS",
    "FAST_H1_SPANS",
    "FAST_KNOCKDOWNS",
    "FAST_LINES",
    "FAST_PINS",
    "RemoteFile",
    "backoff",
    "build_fast_tables",
    "check_pin",
    "chunked_1d",
    "dense_rows",
    "fast_cells",
    "fast_panel",
    "fast_rows",
    "halves_from_cells",
    "legacy_codes",
    "remote_h5",
]

# ---- FAST mode, small inputs read from the public files by byte range ------------------------------------
# Used in: the FAST-mode data cell, before section 1.
# FAST runs the exploration (sections 3 to 11) on a few thousand cells per training line. We redistribute no
# data, so those cells are read straight from the public files: the Replogle and Nadig files store X dense,
# uncompressed and contiguous, so one cell is one byte range; the 2025 H1 file stores it as uncompressed chunks.
# A partial read cannot be checked against the whole file's checksum, so the row lists and the sha256 of the bytes
# read are pinned here.

PUBLIC_FILES.update(
    {
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv": (
            "https://ndownloader.figshare.com/files/51065489",
            506628654,
            "2a71dc94110efcc0221eae821bb93a9f03b54bea16f005818911a09d33383d56",
        ),
        "Model.csv": (
            "https://ndownloader.figshare.com/files/51065297",
            645696,
            "b7a0c1385e6cef30132b56aff61f1261d11e3f490490b355c430d32ee0dbdcfa",
        ),
    }
)
FAST_LINES = ("k562", "rpe1", "hepg2", "jurkat")  # the essential-gene screens, read cell by cell
FAST_H1 = "adata_Validation.h5ad"  # the 2025 challenge's H1 validation split (FULL uses its training split)
FAST_KNOCKDOWNS, FAST_CELLS, FAST_CONTROLS, FAST_H1_ROWS = 60, 100, 2000, 10_000
FAST_H1_SPANS = {
    "data": (8944, 8944 + 335_547_436),
    "indices": (3_459_812_128, 3_459_812_128 + 335_547_436),
    "meta": (6_919_615_312, 6_928_967_541),
}  # where the H1 file keeps the parts FAST reads
FAST_PINS = {  # sha256 of the row numbers read and of the bytes read (h1: the first 10,000 rows' values)
    "k562": {
        "rows": "0f5b0ca6ac3a07137d38a4510796c7b065e5964b6cdae6a09c73a883da4cf266",
        "X": "2c455908da85a6f48e4cb9cd0ea1c1163974fa4d1fcda575be66a32a7dc09da6",
    },
    "rpe1": {
        "rows": "3b7f7f205ee98e2cd216238e7b15b17bb791a3fcda36c79abb52451ed8299e63",
        "X": "b17093f62560c95db7b3b5aa5fa82d6ce39caa0c64d48ba1a2cb58371c5e7479",
    },
    "hepg2": {
        "rows": "d59911f4bb39af6d5cbb8b5042dc5cbfd8fba99ac12b5dca1e82ab21768187f2",
        "X": "86d282c867974c4b197da8561c97a5063b5eb5bf9ed3bea70baa5011fcd98188",
    },
    "jurkat": {
        "rows": "28a631cfa237eecb22111779c70b773da15a8d9668a5d7c7eeb52752eb1149b5",
        "X": "fae36a82304d645e653667b20a84048e40705841409de2abc879f21ad7727718",
    },
    "h1": {
        "rows": "9e1c19b3fdc185411bd1a987deb6a9fda2531116aac060a84c4d878854d1c099",
        "X": "2a6d7c3b6c84b208f8724087a9ff23d41c4ece2ea9874e055b87bd344cbd5062",
    },
}


_LOCAL = threading.local()
_POOL: list = []


def _pool() -> ThreadPoolExecutor:
    """The one set of worker threads every read goes through: each keeps one connection per server, so no server
    ever sees more than FAST_CONNECTIONS connections from us."""
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(FAST_CONNECTIONS))
    return _POOL[0]


FAST_CONNECTIONS = 4  # at most this many connections to one server at a time: be polite to public servers


def _request(url: str, start: int, stop: int):
    """One range request over a kept-alive connection (one per thread and host: a new TLS connection costs
    seconds). Returns (status, body, headers)."""
    import http.client
    from urllib.parse import urlsplit

    u = urlsplit(url)
    pool = _LOCAL.__dict__.setdefault("conns", {})
    c = pool.get(u.netloc)
    if c is None:
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        c = pool[u.netloc] = cls(u.netloc, timeout=120)
    try:
        c.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers={"Range": f"bytes={start}-{stop - 1}"})
        r = c.getresponse()
        return r.status, r.read(), r
    except Exception:
        pool.pop(u.netloc, None)
        raise


def backoff(r, attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After when it sends one, else 2, 4, 8, ... up to 120."""
    try:
        return min(300.0, float(r.getheader("Retry-After")))
    except Exception:  # noqa: BLE001
        return min(120.0, 2.0 ** (attempt + 1))


class RemoteFile(io.RawIOBase):
    """A public file opened for reading by HTTP byte ranges, cached in 4 MiB blocks (for h5py's metadata reads)."""

    def __init__(self, url: str, size: int, block: int = 4 << 20):
        """`url`: the public file; `size`: its length in bytes; `block`: the cache's block size in bytes."""
        self.src, self.size, self.block, self.pos, self.cache = url, size, block, 0, {}
        self.lock, self.signed, self.when, self.kept = threading.Lock(), None, 0.0, {}

    def target(self) -> str:
        """Where the bytes are: figshare redirects to storage with a link signed for 10 seconds, so it is fetched
        again every 7 seconds (a few requests to figshare per file, not one per range)."""
        from urllib.parse import urljoin

        with self.lock:
            if self.signed is None or time.time() - self.when > 7:
                for attempt in range(10):
                    status, _, r = _request(self.src, 0, 1)
                    if status not in (429, 503):
                        break
                    time.sleep(backoff(r, attempt))
                loc = r.getheader("Location")
                self.signed = urljoin(self.src, loc) if status in (301, 302, 303, 307, 308) else self.src
                self.when = time.time()
            return self.signed

    def get(self, start: int, stop: int) -> bytes:
        """Bytes [start, stop) of the file, retried up to 10 times with a growing wait (a stale signed link is
        renewed)."""
        last, r = None, None
        for attempt in range(10):
            try:
                status, body, r = _request(self.target(), start, stop)
                if status == 206 and len(body) == stop - start:
                    return body
                last = f"HTTP {status}, {len(body)} bytes"
                if status not in (429, 503):  # a stale signed link (403): fetch a new one
                    with self.lock:
                        self.signed = None
            except Exception as e:  # noqa: BLE001
                last, r = repr(e), None
            time.sleep(backoff(r, attempt))
        raise OSError(f"{self.src}: bytes {start}-{stop} unreadable ({last})")

    def readable(self):
        """True: the file is opened for reading (part of the io.RawIOBase interface)."""
        return True

    def seekable(self):
        """True: any position can be read (part of the io.RawIOBase interface)."""
        return True

    def tell(self):
        """The current read position, in bytes."""
        return self.pos

    def seek(self, off, whence=0):
        """Move the read position by `off` (`whence` 0: from the start, 1: from here, 2: from the end); returns it."""
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def prefetch(self, start: int, stop: int) -> None:
        """Read the blocks covering [start, stop) in parallel and keep them (a region h5py will then read)."""
        ks = [k for k in range(start // self.block, (stop - 1) // self.block + 1) if k not in self.kept]
        for k, blob in zip(
            ks, self.ranges([(k * self.block, min(self.size, (k + 1) * self.block)) for k in ks]), strict=True
        ):
            self.kept[k] = blob

    def ranges(self, spans: list) -> list:
        """Many byte ranges at once, over at most FAST_CONNECTIONS kept-alive connections."""
        return list(_pool().map(lambda sp_: self.get(*sp_), spans))

    def readinto(self, b):
        """Fill the buffer `b` from the current position, block by block through the cache; returns the bytes read."""
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = memoryview(b), self.pos
        while p < self.pos + n:
            k = p // self.block
            with self.lock:
                blk = self.kept.get(k) or self.cache.get(k)
            if blk is None:
                blk = _pool().submit(self.get, k * self.block, min(self.size, (k + 1) * self.block)).result()
                with self.lock:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    self.cache[k] = blk
            a = p - k * self.block
            m = min(len(blk) - a, self.pos + n - p)
            out[p - self.pos : p - self.pos + m] = blk[a : a + m]
            p += m
        self.pos += n
        return n


def remote_h5(name: str):
    """Open the public file `name` (a key of PUBLIC_FILES) for reading over HTTP.

    Returns (the RemoteFile, an h5py.File reading through it).
    """
    import h5py

    url, size, _ = PUBLIC_FILES[name]
    rf = RemoteFile(url, size)
    return rf, h5py.File(rf, "r")


def legacy_codes(h, group: str, col: str):
    """A categorical column of an older h5ad: its codes and categories."""
    return h[group][col][:], h[group]["__categories"][col][:].astype(str)


def dense_rows(rf: RemoteFile, X, rows: np.ndarray, gap: int = 8) -> np.ndarray:
    """Rows of a dense, contiguous, uncompressed float32 X, read as byte ranges (nearby rows in one range)."""
    assert X.chunks is None and X.compression is None and X.dtype == np.float32, "X is not dense and contiguous"
    off, G = X.id.get_offset(), X.shape[1]
    rb = G * 4
    runs, s = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i] - rows[i - 1] > gap:
            runs.append((rows[s], rows[i - 1] + 1))
            s = i
    blobs = rf.ranges([(off + a * rb, off + b * rb) for a, b in runs])
    out = np.empty((len(rows), G), np.float32)
    k = 0
    for (a, b), blob in zip(runs, blobs, strict=True):
        M = np.frombuffer(blob, np.float32).reshape(b - a, G)
        sel = rows[(rows >= a) & (rows < b)] - a
        out[k : k + sel.size] = M[sel]
        k += sel.size
    return out


def chunked_1d(rf: RemoteFile, d, stop: int, span: tuple) -> np.ndarray:
    """Elements [0, stop) of an uncompressed chunked 1-D dataset. Its chunks lie in order in `span` (bytes of the
    file, pinned), between the small nodes of their index: the span is read in parallel first, then h5py reads
    the dataset from it (asking HDF5 for each chunk's place over the network costs minutes)."""
    assert d.compression is None and d.chunks is not None, "not uncompressed chunks"
    rf.prefetch(*span)
    return d[:stop]


def fast_panel(labels: dict) -> list:
    """The FAST knockdowns: 60 drawn (seed 0) from those with >= 100 cells in every essential-gene line."""
    common = None
    for lab in labels.values():
        vc = pd.Series(lab[lab != "non-targeting"]).value_counts()
        ok = set(vc.index[vc >= FAST_CELLS])
        common = ok if common is None else common & ok
    return sorted(np.random.default_rng(0).choice(sorted(common), FAST_KNOCKDOWNS, replace=False).tolist())


def fast_rows(lab: np.ndarray, panel: list) -> np.ndarray:
    """2,000 controls and 100 cells of each FAST knockdown (seed 0), as sorted row numbers."""
    rng = np.random.default_rng(0)
    pick = [rng.choice(np.flatnonzero(lab == "non-targeting"), FAST_CONTROLS, replace=False)]
    for p in panel:
        pick.append(rng.choice(np.flatnonzero(lab == p), FAST_CELLS, replace=False))
    return np.sort(np.concatenate(pick))


def check_pin(key: str, rows: np.ndarray, block: np.ndarray) -> None:
    """Assert that the rows read for `key` and their values (`block`) hash to the sha256 pinned in FAST_PINS."""
    got = {
        "rows": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest(),
        "X": hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest(),
    }
    assert got == FAST_PINS[key], f"{key}: the subset read differs from the pinned one {got}"


def fast_cells(genes: list, keep: Path) -> tuple:
    """{line: (counts on the axis as CSR, label per cell, row names)} for the FAST training lines, and the FAST
    knockdowns. What is read is kept in `keep` (checked against the pins again when reused), so a rerun reads
    nothing twice."""
    keep.mkdir(parents=True, exist_ok=True)
    meta = {}
    for line in FAST_LINES:
        f = keep / f"{line}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            meta[line] = (None, None, z["lab"], z["sym"], z)
            continue
        rf, h = remote_h5(BENCH_LINES[line][0])
        codes, cats = legacy_codes(h, "obs", "gene")
        gcodes, gcats = legacy_codes(h, "var", "gene_name")
        meta[line] = (rf, h, cats[codes], gcats[gcodes], None)
    panel = fast_panel({ln: m[2] for ln, m in meta.items()})
    out = {}
    for line, (rf, h, lab, sym, z) in meta.items():
        rows = fast_rows(lab, panel)
        if z is None:
            X = dense_rows(rf, h["X"], rows)
            h.close()
            np.savez(keep / f"{line}.npz", X=X, lab=lab, sym=sym)
        else:
            X = z["X"]
        check_pin(line, rows, X)
        assert np.array_equal(X, np.round(X)), "not whole-number counts"
        out[line] = (align_genes(sp.csr_matrix(X), sym, genes), lab[rows], np.array([f"row{r}" for r in rows]))
    f, W = keep / "h1.npz", FAST_H1_ROWS
    if f.exists():
        z = np.load(f, allow_pickle=False)
        ip, data, ind, lab, sym = z["ip"], z["data"], z["ind"], z["lab"], z["sym"]
    else:
        rf, h = remote_h5(FAST_H1)
        rf.prefetch(*FAST_H1_SPANS["meta"])  # indptr, obs and var, with their index nodes
        ip = h["X"]["indptr"][: W + 1]
        data = chunked_1d(rf, h["X"]["data"], int(ip[-1]), FAST_H1_SPANS["data"])
        rf.kept.clear()
        ind = chunked_1d(rf, h["X"]["indices"], int(ip[-1]), FAST_H1_SPANS["indices"])
        g = h["obs"]["target_gene"]
        lab = g["categories"][:].astype(str)[g["codes"][:W]]
        sym = h["var"]["_index"][:].astype(str)
        h.close()
        np.savez(f, ip=ip, data=data, ind=ind, lab=lab, sym=sym)
    check_pin("h1", np.arange(W), data)
    X = sp.csr_matrix((data.astype(np.float32), ind, ip), shape=(W, len(sym)))
    out["h1"] = (align_genes(X, sym, genes), lab, np.array([f"row{r}" for r in range(W)]))
    return out, panel


def halves_from_cells(X, labels: np.ndarray, knockdowns: set, gmap: GeneMap) -> dict:
    """Split halves (seed 0) of cells already on the axis: section 5's table from an in-memory subset."""
    lab = pd.Categorical(split_labels(labels, knockdowns, np.random.default_rng(0), control="non-targeting"))
    codes = np.asarray(lab.codes)
    keep = np.flatnonzero(codes >= 0)
    Xk = sp.csr_matrix(X[keep], dtype=np.float64)
    L = np.asarray(Xk.sum(axis=1)).ravel()
    CPM = sp.csr_matrix(sp.diags(1e6 / np.maximum(L, 1)) @ Xk)
    S = sp.csr_matrix((np.ones(keep.size), (codes[keep], np.arange(keep.size))), shape=(len(lab.categories), keep.size))
    sums = (S @ CPM).toarray()
    n = np.bincount(codes[keep], minlength=len(lab.categories)).astype(np.float64)
    names = list(lab.categories.astype(str))
    ci = names.index("ctrl")
    return halves_table(names, sums, n, sums[ci] / n[ci], gmap)


def build_fast_tables(out: Path, downloads: Path, phase: Path) -> None:
    """The FAST stand-in for the reference tables: the five training lines' cells, tables and split halves, the
    controls' overdispersion, the K562 genome-wide pseudobulk, the pathway screens and DepMap."""
    out.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(phase / "gene_names.csv")["gene_name"].astype(str).tolist()
    gmap = GeneMap(genes, fetch("hgnc_complete_set_2026-09-04.txt", downloads))
    shutil.copy(downloads / "hgnc_complete_set_2026-09-04.txt", out / "hgnc.txt")
    shutil.copy(fetch("OmicsExpressionProteinCodingGenesTPMLogp1.csv", downloads), out / "depmap_expression.csv")
    shutil.copy(fetch("Model.csv", downloads), out / "depmap_models.csv")
    cells, panel = fast_cells(genes, downloads / "fast_cells")
    refs = {}
    for line in ("h1", "k562", "rpe1", "hepg2", "jurkat"):
        X, lab, names = cells[line]
        ref, pnl = build_reference(
            X,
            lab,
            names,
            max_cells=FAST_CELLS,
            n_control=None,
            min_cells=50,
            panel=panel if line != "h1" else None,
            genes=genes,
        )
        d = out / "bench" / line
        d.mkdir(parents=True, exist_ok=True)
        ref.write_h5ad(d / "reference.h5ad")
        pd.DataFrame({"target_gene": pnl}).to_csv(d / "panel.csv", index=False)
        refs[line] = d / "reference.h5ad"
        lab_r = ref.obs["target_gene"].astype(str).to_numpy()
        t = table_bench(line, X, lab, mean_cpm(ref.X[lab_r == "non-targeting"]))
        save_table(t, out / f"bench_{line}")
        hv = halves_from_cells(X, lab, set(t["target"]), gmap)
        (out / "halves").mkdir(exist_ok=True)
        np.savez(out / "halves" / ("h1train.npz" if line == "h1" else f"bench_{line}.npz"), **hv)
        print(f"FAST {line}: {X.shape[0]:,} cells, {len(t['target'])} knockdowns")
        del ref
    np.savez(out / "phi.npz", **table_phi(refs))
    save_table(table_gwps_bulk(fetch("K562_gwps_raw_bulk_01.h5ad", downloads), genes), out / "k562gwps_bulk")
    for name, t in tables_jiang(fetch("DE_results_all_pathway.zip", downloads), gmap).items():
        save_table(t, out / name)
