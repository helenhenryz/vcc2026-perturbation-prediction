"""Rebuilding the reference tables from the public downloads (used only by the appendix)."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .generator import fit_controls, mean_cpm
from .inputs import CLIP, Sources, blas_threads, own_out_mean, sha256_file
from .phase import GeneMap
from .stage1 import rsvd

__all__ = [
    "A549_KO_GEO",
    "BENCH_LINES",
    "DLD1_FILES",
    "GEO",
    "HESC_LANES",
    "PERTURBFATE_GEO",
    "PF_ARMS",
    "PF_MEMORY_ALIASES",
    "PF_RULE",
    "PUBLIC_FILES",
    "SOURCE_CLIP",
    "SOURCE_FLOOR",
    "SRCPC_THREADS",
    "TABLE_CLIP",
    "TABLE_JSON",
    "Rds",
    "align_genes",
    "assign_guides",
    "build_reference",
    "dld1_channels",
    "dld1_coldata",
    "effects_of",
    "fetch",
    "gencode_names",
    "guide_stats",
    "halves_line",
    "halves_table",
    "hesc_url",
    "labels_table",
    "load_line",
    "quarters",
    "read_rds_counts",
    "same_table",
    "save_table",
    "source_pc_axes",
    "split_labels",
    "stream_group_sums",
    "stream_mean_cpm",
    "table_a549_ko",
    "table_arc_h1",
    "table_bench",
    "table_dld1",
    "table_from_cells",
    "table_from_sums",
    "table_gwps_bulk",
    "table_hesc",
    "table_myllia",
    "table_perturbfate",
    "table_phi",
    "table_viperturb",
    "tables_cd4",
    "tables_jiang",
    "tss_table",
]

# ---- appendix, downloads ---------------------------------------------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
# Every public file, with its size and sha256 as we used it. `fetch` resumes an interrupted download and refuses a
# file whose checksum differs.
PUBLIC_FILES = {
    "K562_gwps_raw_bulk_01.h5ad": (
        "https://ndownloader.figshare.com/files/35774443",
        374587922,
        "7cec96b3b76169abbf6b6ab9d10bf00d71d942d89e63292351f745e130b154db",
    ),
    "DE_results_all_pathway.zip": (
        "https://zenodo.org/api/records/14518762/files/DE_results_all_pathway.zip/content",
        324112587,
        "d2f6d920abaf5e05de6e5343acdfae6a18537d23bf9003e80aa73cce97193c80",
    ),
    "gencode.v49.basic.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz",
        67888546,
        "5ad879da17040b5b032e694b1a766413bcc262d0a6a78bed3e81294320ab7c88",
    ),
    "hgnc_complete_set_2026-09-04.txt": (
        (
            "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/"
            "hgnc_complete_set_2026-09-04.txt"
        ),
        16903161,
        "6f43d6ff43aa9fdfa5fb2f20a20a7cace66e6e02e2a0dcf19d9b726e2e248d20",
    ),
}


def fetch(name: str, dest_dir: Path) -> Path:
    """Download one public file (resumable), then check its size and sha256."""
    import subprocess

    url, size, sha = PUBLIC_FILES[name]
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        if dest.exists() and dest.stat().st_size == size:
            break
        subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--retry",
                "10",
                "--retry-all-errors",
                "--limit-rate",
                "50M",
                "-C",
                "-",
                "-o",
                str(dest),
                url,
            ],
            check=False,
        )  # throttled: the network is shared
    assert dest.stat().st_size == size, f"{name}: {dest.stat().st_size} bytes, expected {size}"
    got = sha256_file(dest)
    assert sha is None or got == sha, f"{name}: sha256 {got}, expected {sha}"
    return dest


TABLE_JSON = ("control_cells", "median_total")


def save_table(t: dict, out: Path) -> None:
    """Write a rebuilt table as the pipeline reads it: <name>.npz, plus <name>.json for the screen's control-cell
    count and depth."""
    np.savez(out.with_suffix(".npz"), **{k: v for k, v in t.items() if k not in TABLE_JSON})
    if any(k in t for k in TABLE_JSON):
        out.with_suffix(".json").write_text(json.dumps({k: t[k] for k in TABLE_JSON if k in t}, indent=1))


def same_table(new: dict, held: Path) -> dict:
    """A rebuilt source table against the held one, array by array (the .npz container itself stores a write
    time, so its bytes differ by construction): equal, or the max |diff| and how many entries differ."""
    z = np.load(held, allow_pickle=True)
    meta = json.loads(Path(held).with_suffix(".json").read_text()) if Path(held).with_suffix(".json").exists() else {}
    out = {}
    for k, v in new.items():
        if k in TABLE_JSON:  # the screen's control-cell count and depth, kept in <name>.json beside the table
            out[k] = {"equal": k in meta and float(v) == float(meta[k]), "new": float(v), "held": meta.get(k)}
            continue
        a, b = np.asarray(v), np.asarray(z[k])
        if a.dtype.kind in "fc" and b.dtype.kind in "fc" and a.shape == b.shape:
            d = np.abs(np.nan_to_num(a.astype(np.float64), nan=0) - np.nan_to_num(b.astype(np.float64), nan=0))
            out[k] = {
                "equal": bool(a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)),
                "nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
                "max_abs_diff": float(d.max()) if d.size else 0.0,
                "n_diff": int((d > 0).sum()),
            }
        else:
            out[k] = {
                "equal": bool(
                    a.shape == b.shape
                    and np.array_equal(
                        a.astype(str) if a.dtype.kind in "US" else a, b.astype(str) if b.dtype.kind in "US" else b
                    )
                )
            }
    return out


# ---- appendix, the K562 genome-wide pseudobulk -----------------------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
SOURCE_CLIP, SOURCE_FLOOR = 6.0, 10.0  # log2 fold changes clipped to +-6; NaN where the control mean < 10 CPM


def table_gwps_bulk(h5ad: Path, genes: list, min_cells: int = 50) -> dict:
    """Replogle 2022's K562 genome-wide screen as pseudobulk: one row per guide group ("<id>_<GENE>_..."), X its mean
    counts per cell. Each gene's groups are pooled by cell count, normalised to proportions and divided by the core
    non-targeting groups' proportions."""
    import anndata as ad

    b = ad.read_h5ad(h5ad)
    X = np.asarray(b.X.toarray() if sp.issparse(b.X) else b.X, dtype=np.float64)
    target = np.array([n.split("_")[1] for n in b.obs_names.astype(str)])
    cells = b.obs["num_cells_filtered"].to_numpy(dtype=np.float64)
    ctrl = (target == "non-targeting") & b.obs["core_control"].to_numpy(dtype=bool)
    prop = X / X.sum(axis=1, keepdims=True)
    ctrl_prop = (prop[ctrl] * cells[ctrl, None]).sum(axis=0) / cells[ctrl].sum()
    col = pd.Index(genes).get_indexer(pd.Index(b.var["gene_name"].astype(str)))
    first = np.flatnonzero(col >= 0)
    first = first[~pd.Index(col[first]).duplicated()]  # a symbol twice: its first column
    perts, fcs, ns = [], [], []
    for g in sorted(set(target[~(target == "non-targeting")])):
        rows = target == g
        n = cells[rows].sum()
        if n < min_cells:
            continue
        p = (prop[rows] * cells[rows, None]).sum(axis=0) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ctrl_prop > 0, p / ctrl_prop, 1.0)
        fc = np.ones(len(genes), dtype=np.float32)
        fc[col[first]] = r[first]
        perts.append(g), fcs.append(fc), ns.append(int(n))
    c = np.full(len(genes), np.nan)
    c[col[first]] = ctrl_prop[first] * 1e6
    lfc = np.log2(np.clip(np.vstack(fcs).astype(np.float64), 2.0**-SOURCE_CLIP, 2.0**SOURCE_CLIP))
    lfc[:, ~(c >= SOURCE_FLOOR)] = np.nan
    return {
        "target": np.asarray(perts).astype(str),
        "lfc": lfc.astype(np.float32),
        "n_cells": np.array(ns),
        "cell_type": np.array("k562"),
        "chem": np.array("3p"),
        "ctrl_cpm": c,
    }


# ---- appendix, the pathway screens (Jiang 2025) ----------------------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
def tables_jiang(zip_path: Path, gmap: GeneMap) -> dict:
    """Jiang et al. 2025: one published DE table per regulator x pathway, genes x log2FC per cell line; a gene the
    table does not list is NaN. One source per line x pathway."""
    import io
    import re
    import zipfile

    rows, lfcs, mapped = [], [], {}
    pattern = r"DE_results_all_pathway/Parse_(\w+)/(.+)_\1_pathway_DE_results\.txt"
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if re.fullmatch(pattern, n)):
            pathway, regulator = re.fullmatch(pattern, name).groups()
            df = pd.read_csv(io.BytesIO(z.read(name)), sep=r"\s+", engine="c")
            g = df["gene_ID"].astype(str).to_numpy()
            for x in set(g) - mapped.keys():
                mapped[x] = None
            for line in [c.removeprefix("log2FC_") for c in df.columns if c.startswith("log2FC_")]:
                rows.append((regulator, f"{line}_{pathway}", line.lower()))
                lfcs.append((g, df[f"log2FC_{line}"].to_numpy(dtype=np.float64)))
    names = sorted(mapped)
    gpos, _ = gmap.map(names)
    where = dict(zip(names, gpos, strict=True))
    L = np.full((len(rows), len(gmap.axis)), np.nan, dtype=np.float32)
    for i, (g, v) in enumerate(lfcs):
        p = np.array([where[x] for x in g])
        ok = (p >= 0) & np.isfinite(v)
        L[i, p[ok]] = v[ok]
    regs = np.array([r[0] for r in rows])
    uregs = sorted(set(regs))
    tpos, _ = gmap.map(uregs)
    tmap = {r: (gmap.axis[p] if p >= 0 else r) for r, p in zip(uregs, tpos, strict=True)}
    target = np.array([tmap[r] for r in regs]).astype(str)
    context = np.array([r[1] for r in rows]).astype(str)
    cell_type = np.array([r[2] for r in rows]).astype(str)
    out = {}
    for ctx in sorted(set(context)):
        k = np.flatnonzero(context == ctx)
        out[f"jiang_{ctx}"] = {
            "target": target[k],
            "lfc": L[k],
            "n_cells": np.full(len(k), -1),
            "cell_type": np.array(str(cell_type[k[0]])),
            "chem": np.array("3p"),
            "ctrl_cpm": np.full(len(gmap.axis), np.nan),
        }
    return out


# ---- appendix, transcription start sites (GENCODE 49) ----------------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
def tss_table(gtf_gz: Path) -> pd.DataFrame:
    """One TSS per gene symbol from GENCODE's basic annotation (the first record: chrX before chrY)."""
    import gzip
    import re

    rows = []
    name_re, type_re, id_re = (
        re.compile(r'gene_name "([^"]+)"'),
        re.compile(r'gene_type "([^"]+)"'),
        re.compile(r'gene_id "([^"]+)"'),
    )
    with gzip.open(gtf_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 8)
            if p[2] != "gene":
                continue
            start, end, strand = int(p[3]), int(p[4]), p[6]
            rows.append(
                {
                    "chrom": p[0],
                    "tss": start if strand == "+" else end,
                    "strand": strand,
                    "gene_name": name_re.search(p[8]).group(1),
                    "gene_type": type_re.search(p[8]).group(1),
                    "gene_id": id_re.search(p[8]).group(1),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("gene_name", keep="first").reset_index(drop=True)


# ---- appendix, VIPerturb-seq (K562 on 10x Flex, a Seurat .rds) -------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
PUBLIC_FILES["genome_wide_filtered.rds"] = (
    "https://zenodo.org/api/records/18460279/files/genome_wide_filtered.rds/content",
    3611371127,
    "8012b3ac300034bb0ce6b22dfd258b1cc9581699bf93bfe7b0c52a8c4dbd9fe1",
)


class Rds:
    """An .rds read with rdata's XDR parser from the gzip stream; every vector over 64 MB is spilled to a .npy on
    disk as it is read (a whole-file parse needs three copies of a ~20 GB payload in memory)."""

    def __init__(self, rds: Path, spill: Path):
        """Parse the .rds file `rds`; vectors over 64 MB are spilled to .npy files in `spill`."""
        import gzip
        import sys

        from rdata.parser._parser import DEFAULT_ALTREP_MAP, Parser, RObjectType
        from rdata.parser._xdr import ParserXDR

        self.T = RObjectType
        spill.mkdir(parents=True, exist_ok=True)

        class Spilling(ParserXDR):
            """rdata's XDR parser, spilling large vectors to disk as it reads them."""

            def __init__(self, f):
                """A parser reading the open stream `f`; `n` numbers the spilled vectors."""
                Parser.__init__(self, expand_altrep=True, altrep_constructor_dict=DEFAULT_ALTREP_MAP)
                self.file, self.n = f, 0

            def _parse_array_values(self, dtype, length):
                """One vector: read into memory under 64 MB, else streamed to a .npy file and memory-mapped."""
                dtype = np.dtype(dtype)
                if length * dtype.itemsize < 64 * 2**20:
                    return super()._parse_array_values(dtype, length)
                path = spill / f"v{self.n:03d}_{dtype.name}_{length}.npy"
                self.n += 1
                with open(path, "wb") as fo:
                    np.lib.format.write_array_header_1_0(
                        fo, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": (length,)}
                    )
                    for st in range(0, length, 2**24):
                        k = min(2**24, length - st)
                        buf = self.file.read(k * dtype.itemsize)
                        assert len(buf) == k * dtype.itemsize, "truncated vector"
                        np.frombuffer(buf, dtype=dtype.newbyteorder(">")).astype(dtype).tofile(fo)
                return np.load(path, mmap_mode="c")

        sys.setrecursionlimit(1_000_000)
        with gzip.open(rds, "rb") as f:
            assert f.read(2) == b"X\n", "not an XDR .rds"
            self.obj = self.deref(Spilling(f).parse_all().object)

    def deref(self, o):
        """Follow a reference to the object it points at (other objects are returned as they are)."""
        return o.referenced_object if o.info.type == self.T.REF else o

    def name_of(self, sym):
        """A symbol's name, as a string."""
        sym = self.deref(sym)
        return self.deref(sym.value).value.decode() if sym.info.type == self.T.SYM else sym.value.decode()

    def pairlist(self, o):
        """An R pairlist as a list of (tag, value)."""
        out, o = [], self.deref(o)
        while o is not None and o.info.type == self.T.LIST:
            car, cdr = o.value
            out.append((self.name_of(o.tag) if o.tag is not None else None, self.deref(car)))
            o = self.deref(cdr)
        return out

    def attrs(self, o):
        """An object's attributes as {name: value}."""
        o = self.deref(o)
        return dict(self.pairlist(o.attributes)) if o.attributes is not None else {}

    def strings(self, o):
        """An R character vector as a list of str (None for NA)."""
        return [None if c.value is None else c.value.decode() for c in (self.deref(x) for x in self.deref(o).value)]

    def listing(self, o):
        """A named R list as {name: value}."""
        o = self.deref(o)
        return dict(zip(self.strings(self.attrs(o)["names"]), (self.deref(x) for x in o.value), strict=True))

    def matrix(self, m):
        """A dgCMatrix / dgTMatrix: (i, p or j, x, dim, row names, column names)."""
        a = self.attrs(m)
        dim = [int(v) for v in np.asarray(self.deref(a["Dim"]).value)]
        dn = self.deref(a["Dimnames"]).value
        names = [self.strings(self.deref(d)) if self.deref(d).info.type == self.T.STR else None for d in dn]
        second = "p" if "p" in a else "j"
        return (
            self.deref(a["i"]).value,
            self.deref(a[second]).value,
            self.deref(a["x"]).value,
            dim,
            names[0],
            names[1],
        )


def read_rds_counts(rds: Path, spill: Path):
    """The RNA counts (genes x cells, column-compressed) and the cell metadata of a Seurat .rds."""
    r = Rds(rds, spill)
    slots = dict(r.pairlist(r.obj.attributes))
    rs = dict(r.pairlist(r.listing(slots["assays"])["RNA"].attributes))
    counts = r.listing(rs["layers"])["counts"] if "layers" in rs else rs["counts"]
    i, p, x, _, genes, _ = r.matrix(counts)
    if genes is None:  # Seurat v5: the names live on the assay's feature map
        genes = r.strings(r.deref(r.attrs(rs["features"])["dimnames"]).value[0])
    meta = {}
    for k, v in r.listing(slots["meta.data"]).items():
        a = r.attrs(v)
        if v.info.type == r.T.STR:
            meta[k] = r.strings(v)
        elif "levels" in a:
            lev = np.array(r.strings(a["levels"]), dtype=object)
            codes = np.asarray(v.value)
            meta[k] = pd.Categorical.from_codes(np.where(codes > 0, codes - 1, -1), lev)
    return i, np.asarray(p), x, genes, pd.DataFrame(meta)


def quarters(codes: np.ndarray, n_groups: int, seed: int = 0) -> np.ndarray:
    """Each cell's quarter (0-3): within each group, a random permutation dealt in order."""
    rng = np.random.default_rng(seed)
    q = np.empty(codes.size, dtype=np.int8)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    for g in range(n_groups):
        idx = order[bounds[g] : bounds[g + 1]]
        if idx.size:
            q[rng.permutation(idx)] = (np.arange(idx.size) * 4 // idx.size).astype(np.int8)
    return q


def table_viperturb(
    rds: Path, gmap: GeneMap, spill: Path, control: str = "NO-TARGET", block: int = 8192, min_cells: int = 50
) -> dict:
    """VIPerturb-seq (K562 on 10x Flex) as a source table.

    Per knockdown: the mean per-cell CPM over the axis genes (accumulated in quarters, the order the shipped table
    was summed in), against the non-targeting cells; log2, clipped, NaN below 10 control CPM; a zero mean is read
    as half a molecule of the row.
    """
    i, p, x, genes0, meta = read_rds_counts(rds, spill)
    lab = meta["gene"].astype(str).to_numpy()
    pos, _ = gmap.map(genes0)
    keep = np.flatnonzero(pos >= 0)
    perts = np.array(sorted(set(lab)))
    codes = np.searchsorted(perts, lab)
    grp = codes.astype(np.int64) * 4 + quarters(codes, perts.size)
    ctrl_code = int(np.flatnonzero(perts == control)[0])
    sums = np.zeros((perts.size * 4, keep.size))
    totals = np.zeros(lab.size)
    cells = np.arange(lab.size)
    for s in range(0, cells.size, block):
        cols = cells[s : s + block]
        a, b = int(p[cols[0]]), int(p[cols[-1] + 1])
        X = sp.csr_matrix(
            (np.asarray(x[a:b], dtype=np.float64), np.asarray(i[a:b]), p[cols[0] : cols[-1] + 2] - a),
            shape=(cols.size, len(genes0)),
        )[:, keep]
        tot = np.asarray(X.sum(axis=1)).ravel()
        totals[s : s + cols.size] = tot
        cpm = sp.diags(1e6 / np.maximum(tot, 1)) @ X
        g = grp[s : s + cols.size]
        uniq, inv = np.unique(g, return_inverse=True)
        ind = sp.csr_matrix((np.ones(cols.size), (inv, np.arange(cols.size))), shape=(uniq.size, cols.size))
        sums[uniq] += (ind @ cpm).toarray()
    n = np.bincount(grp, minlength=perts.size * 4).reshape(perts.size, 4)
    means = (sums.reshape(perts.size, 4, -1) / np.maximum(n, 1)[:, :, None]).astype(np.float32)
    target_pos, _ = gmap.map(list(perts))
    axis_pos = pos[keep]
    nn = n.astype(np.float64)
    tot = nn.sum(axis=1)
    median_total = float(np.median(totals.astype(np.float32)))
    rows = np.flatnonzero((tot >= min_cells) & (np.arange(len(perts)) != ctrl_code))
    G = len(gmap.axis)
    ctrl_cpm = np.full(G, np.nan)
    c = (means[ctrl_code] * nn[ctrl_code][:, None]).sum(axis=0) / tot[ctrl_code]
    ctrl_cpm[axis_pos] = c
    lfc = np.full((rows.size, G), np.nan, np.float32)
    for r, k in enumerate(rows):
        m = (means[k] * nn[k][:, None]).sum(axis=0) / tot[k]
        detect = 0.5 / (tot[k] * median_total) * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(np.maximum(m, detect) / c)
        v[~(c >= SOURCE_FLOOR)] = np.nan
        lfc[r, axis_pos] = np.clip(v, -SOURCE_CLIP, SOURCE_CLIP)
    target = np.array([gmap.axis[target_pos[k]] if target_pos[k] >= 0 else perts[k] for k in rows])
    return {
        "target": target.astype(str),
        "lfc": lfc,
        "n_cells": tot[rows],
        "cell_type": np.array("k562"),
        "chem": np.array("flex"),
        "ctrl_cpm": ctrl_cpm,
    }


# ---- appendix, single-cell screens to tables -------------------------------------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
PUBLIC_FILES.update(
    {
        "GSE337804_A549_ch_1.RDS.h5ad": (
            f"{GEO}/GSE337nnn/GSE337804/suppl/GSE337804_A549_ch_1.RDS.h5ad",
            563808587,
            "15230cd19d5958f3b1efa2f08160faf7b6a319dad3fbaeacbb0ed7f42ab9b448",
        ),
        "GSE337804_A549_ch_2.RDS.h5ad": (
            f"{GEO}/GSE337nnn/GSE337804/suppl/GSE337804_A549_ch_2.RDS.h5ad",
            608829330,
            "a29d724ab5e4fa3dc698da2ef75ec5bc7cbdabfcb7b55efc43882bcd64d39215",
        ),
    }
)


def table_from_cells(blocks: list, control: str, cell_type: str, chem: str, genes: list, min_cells: int = 50) -> dict:
    """Per-knockdown log2FC rows on the axis from single cells; blocks = [(X counts, label per cell, gene names)].
    Columns land on the axis by exact symbol (duplicates summed); each knockdown's mean per-cell CPM against the
    controls'; a zero mean read as half a molecule of the pool; NaN below 10 control CPM; clipped to +-6."""
    parts, labs = [], []
    for X, labels, g in blocks:
        pos = pd.Index(genes).get_indexer(pd.Index(g))
        keep = np.flatnonzero(pos >= 0)
        M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(X.shape[1], len(genes)))
        parts.append(sp.csr_matrix(X @ M))
        labs.append(labels)
    Xa = sp.vstack(parts, format="csr")
    labels = np.concatenate(labs)
    ctrl = mean_cpm(Xa[labels == control])
    median_total = float(np.median(np.asarray(Xa.sum(axis=1)).ravel()))
    targets, rows, n_cells = [], [], []
    for t in sorted(set(labels) - {control}):
        r = np.flatnonzero(labels == t)
        if r.size < min_cells:
            continue
        m = mean_cpm(Xa[r])
        detect = 0.5 / (r.size * median_total) * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(np.maximum(m, detect) / ctrl)
        v[~(ctrl >= SOURCE_FLOOR)] = np.nan
        targets.append(t), rows.append(np.clip(v, -SOURCE_CLIP, SOURCE_CLIP).astype(np.float32)), n_cells.append(r.size)
    return {
        "target": np.array(targets, dtype=str),
        "lfc": np.vstack(rows),
        "n_cells": np.array(n_cells, dtype=np.float64),
        "cell_type": np.array(cell_type),
        "chem": np.array(chem),
        "ctrl_cpm": np.where(ctrl > 0, ctrl, np.nan),
        "control_cells": int((labels == control).sum()),
        "median_total": median_total,
    }


def table_myllia(d: Path, genes: list) -> dict:
    """Myllia's A549 CRISPRi screen (GEO GSE337804): 218 chromatin regulators, two channels."""
    import h5py

    blocks = []
    for ch in (1, 2):
        with h5py.File(d / f"GSE337804_A549_ch_{ch}.RDS.h5ad", "r") as h:
            X = sp.csr_matrix(
                (h["X"]["data"][:], h["X"]["indices"][:], h["X"]["indptr"][:]), shape=tuple(h["X"].attrs["shape"])
            )
            assert np.array_equal(X.data, np.round(X.data)), "not whole-number counts"
            blocks.append((X, h["obs"]["sgrna_symbol"][:].astype(str), h["var"]["_index"][:].astype(str)))
    return table_from_cells(blocks, "non-targeting", "a549", "3p", genes)


# ---- appendix, the training lines with single cells (Replogle 2022, Nadig 2025, the 2025 challenge's H1) ----
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
PUBLIC_FILES.update(
    {
        "K562_essential_raw_singlecell_01.h5ad": (
            "https://ndownloader.figshare.com/files/35773219",
            10661879995,
            "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee",
        ),
        "rpe1_raw_singlecell_01.h5ad": (
            "https://ndownloader.figshare.com/files/35775606",
            8700873216,
            "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1",
        ),
        "hepg2_raw_singlecell_01.h5ad": (
            f"{GEO}/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad",
            5614460941,
            "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e",
        ),
        "jurkat_raw_singlecell_01.h5ad": (
            f"{GEO}/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad",
            9366490264,
            "ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3",
        ),
        "competition_support_set.zip": (
            "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip",
            8716992349,
            "f1d5fd56a2eee240e5585fac52bb75b0b830a7be6a5f6de1bf773bff4f507f7c",
        ),
    }
)
BENCH_LINES = {
    "k562": ("K562_essential_raw_singlecell_01.h5ad", "gene", "gene_name", "k562", "3p"),
    "rpe1": ("rpe1_raw_singlecell_01.h5ad", "gene", "gene_name", "rpe1", "3p"),
    "hepg2": ("hepg2_raw_singlecell_01.h5ad", "gene", "gene_name", "hepg2", "3p"),
    "jurkat": ("jurkat_raw_singlecell_01.h5ad", "gene", "gene_name", "jurkat", "3p"),
    "h1": ("competition_train.h5", "target_gene", None, "h1", "flex"),
}


def align_genes(X, src_symbols, genes: list):
    """Columns onto the axis (exact symbol; absent genes zero; duplicate symbols summed)."""
    src = pd.Index(src_symbols).astype(str)
    pos = pd.Index(genes).get_indexer(src)
    keep = np.flatnonzero(pos >= 0)
    M = sp.csr_matrix((np.ones(keep.size, dtype=np.float32), (keep, pos[keep])), shape=(X.shape[1], len(genes)))
    return sp.csr_matrix(sp.csr_matrix(X) @ M)


def load_line(name: str, d: Path, genes: list):
    """One training line as raw counts on the axis and a label per cell ("non-targeting" for controls)."""
    import anndata as ad

    f, pert_col, sym_col, _, _ = BENCH_LINES[name]
    a = ad.read_h5ad(d / f)
    X = sp.csr_matrix(a.X, dtype=np.float32)
    if name == "h1":  # the 2025 challenge's file stores log(1 + count): recover the whole counts
        X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        X.data = np.expm1(X.data)
        assert float(np.abs(X.data - np.round(X.data)).max()) <= 1e-2, "not log1p of counts"
        X.data = np.round(X.data).astype(np.float32)
        X = sp.csr_matrix(X, dtype=np.float32)
    else:
        assert np.array_equal(X.data, np.round(X.data)), "not whole-number counts"
    labels = a.obs[pert_col].astype(str).to_numpy()
    symbols = a.var[sym_col] if sym_col else a.var_names
    return align_genes(X, symbols, genes), labels


def effects_of(X, labels: np.ndarray, min_cells: int = 50):
    """Each knockdown's fold change (mean per-cell CPM over the controls'), knockdowns with >= min_cells."""
    X = sp.csr_matrix(X)
    ctrl = mean_cpm(X[labels == "non-targeting"])
    perts, fcs, ns = [], [], []
    for p in sorted(set(labels) - {"non-targeting"}):
        rows = labels == p
        if rows.sum() >= min_cells:
            num = mean_cpm(X[rows])
            with np.errstate(divide="ignore", invalid="ignore"):
                fcs.append(np.where(ctrl > 0, num / ctrl, 1.0).astype(np.float32))
            perts.append(p), ns.append(int(rows.sum()))
    return perts, np.vstack(fcs), np.array(ns)


def build_reference(
    X,
    labels: np.ndarray,
    obs_names,
    max_cells=400,
    n_control=18_400,
    median_depth=20_000.0,
    min_cells=100,
    seed=0,
    panel=None,
    genes=None,
):
    """The line reshaped like the challenge: up to 18,400 controls, a panel of knockdowns with >= 100 cells capped at
    400 cells each, every cell thinned by one factor so the median cell total is 20,000."""
    import anndata as ad

    sizes = pd.Series(labels[labels != "non-targeting"]).value_counts()
    eligible = sorted(sizes[sizes >= min_cells].index)
    if panel is None:  # the panel is drawn first, with its own generator
        panel = sorted(np.random.default_rng(seed).choice(eligible, size=min(300, len(eligible)), replace=False))
    rng = np.random.default_rng(seed)
    keepm = np.isin(labels, list(panel) + ["non-targeting"])
    Xk, lk, nk = X[keepm], labels[keepm], np.asarray(obs_names)[keepm]
    ctrl = np.flatnonzero(lk == "non-targeting")
    if n_control is not None and ctrl.size > n_control:
        ctrl = np.sort(rng.choice(ctrl, size=n_control, replace=False))
    keep = [ctrl]
    for p in sorted(set(lk) - {"non-targeting"}):
        rows = np.flatnonzero(lk == p)
        if rows.size >= min_cells:
            keep.append(np.sort(rng.choice(rows, size=min(rows.size, max_cells), replace=False)))
    rows = np.concatenate(keep)
    Xr = sp.csr_matrix(Xk[rows], dtype=np.float32, copy=True)
    factor = min(1.0, median_depth / float(np.median(np.asarray(Xr.sum(axis=1)).ravel())))
    if factor < 1.0:
        Xr.data = rng.binomial(Xr.data.astype(np.int64), factor).astype(np.float32)
        Xr.eliminate_zeros()
    obs = pd.DataFrame({"target_gene": pd.Categorical(lk[rows])}, index=pd.Index(nk[rows]).astype(str))
    ref = ad.AnnData(X=Xr, obs=obs)
    if genes is not None:
        ref.var_names = pd.Index(genes).astype(str)
    return ref, panel


def table_bench(name: str, X, labels: np.ndarray, ref_ctrl_cpm: np.ndarray) -> dict:
    """A training line's source table: log2 of its fold changes, clipped, NaN below 10 CPM of the reference's
    controls."""
    perts, fc, ns = effects_of(X, labels)
    lfc = np.log2(np.clip(fc.astype(np.float64), 2.0**-SOURCE_CLIP, 2.0**SOURCE_CLIP))
    lfc[:, ref_ctrl_cpm < SOURCE_FLOOR] = np.nan
    return {
        "target": np.asarray(perts).astype(str),
        "lfc": lfc.astype(np.float32),
        "n_cells": ns,
        "cell_type": np.array(BENCH_LINES[name][3]),
        "chem": np.array(BENCH_LINES[name][4]),
        "ctrl_cpm": np.asarray(ref_ctrl_cpm, np.float64),
    }


# ---- appendix, PerturbFate (A375, sci-based RNA counts, GEO GSE291147) -----------------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
PERTURBFATE_GEO = f"{GEO}/GSE291nnn/GSE291147/suppl"
PUBLIC_FILES.update(
    {
        "GSE291147_Dual_omics_RNA_gene_count_matrix.RDS": (
            f"{PERTURBFATE_GEO}/GSE291147_Dual_omics_RNA_gene_count_matrix.RDS",
            316581467,
            "96367a6b645df9334e2ce8a7a3e7d6c37847f8510268a3fb01ce3c12aa67ea49",
        ),
        "GSE291147_Dual_omics_RNA_cell_metadata.csv.gz": (
            f"{PERTURBFATE_GEO}/GSE291147_Dual_omics_RNA_cell_metadata.csv.gz",
            1112875,
            "1d82ffcc8888677357548c6272e0d4c41add4bf45bc2b350f4ffd08f544437e0",
        ),
        "GSE291147_Dual_omics_sgRNA_count_matrix.RDS": (
            f"{PERTURBFATE_GEO}/GSE291147_Dual_omics_sgRNA_count_matrix.RDS",
            1952531,
            "10612855dcd8aa8edd9a990db094d562aa9d6fe343222d0dbb99a021212dd085",
        ),
        "GSE291147_UnlabeledRNA_gene_count_matrix.RDS": (
            f"{PERTURBFATE_GEO}/GSE291147_UnlabeledRNA_gene_count_matrix.RDS",
            695502598,
            "77e1973e323f47ee10535ccbb4983d67e8732ed188cc2b55f0dab68f59bf0774",
        ),
        "GSE291147_LabeledRNA_gene_count_matrix.RDS": (
            f"{PERTURBFATE_GEO}/GSE291147_LabeledRNA_gene_count_matrix.RDS",
            422304653,
            "127f42b0c49c475360d61b11b95573c431ceca31f3fd1c1aec9d7548e0dd5611",
        ),
        "GSE291147_Tri_omics_RNA_cell_metadata.csv.gz": (
            f"{PERTURBFATE_GEO}/GSE291147_Tri_omics_RNA_cell_metadata.csv.gz",
            6041684,
            "b8bec28d8309de04b790e7d0464adf6db7ce917fe0f10e4020a882dae27feb2b",
        ),
        "GSE291147_Tri_omics_sgRNA_count_matrix.RDS": (
            f"{PERTURBFATE_GEO}/GSE291147_Tri_omics_sgRNA_count_matrix.RDS",
            2976200,
            "9a05eab0452766853077f144ba199c7f0d3a224a6840269d0cc833fca71cfadb",
        ),
        "GSM7056149_sciPlexGxE_2_gene.annotations.txt.gz": (
            (
                "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7056nnn/GSM7056149/suppl/"
                "GSM7056149_sciPlexGxE_2_gene.annotations.txt.gz"
            ),
            430161,
            "60b62120c4d45fd49f4ef2fe778f969ce23df062c9dbe6af95ac0d8f02eb8938",
        ),
    }
)
PF_ARMS = {
    "dual": {
        "meta": "GSE291147_Dual_omics_RNA_cell_metadata.csv.gz",
        "sgrna": "GSE291147_Dual_omics_sgRNA_count_matrix.RDS",
        "counts": ["GSE291147_Dual_omics_RNA_gene_count_matrix.RDS"],
        "umi_cols": ["UMI_counts"],
    },
    "tri": {
        "meta": "GSE291147_Tri_omics_RNA_cell_metadata.csv.gz",
        "sgrna": "GSE291147_Tri_omics_sgRNA_count_matrix.RDS",
        "counts": ["GSE291147_UnlabeledRNA_gene_count_matrix.RDS", "GSE291147_LabeledRNA_gene_count_matrix.RDS"],
        "umi_cols": ["old_UMI_counts", "new_UMI_counts"],
    },
}
PF_RULE = {"min_top": 5, "min_ratio": 3.0, "min_share": 0.5}
PF_MEMORY_ALIASES = {
    "CCDC101": "SGF29",
    "KIRREL": "KIRREL1",
    "SGK223": "PRAG1",
}  # renamed symbols no annotation carries


def gencode_names(gtf_gz: Path) -> dict:
    """Unversioned Ensembl gene ID -> GENCODE 49 gene name (PAR_Y copies skipped)."""
    import gzip
    import re

    m = {}
    with gzip.open(gtf_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            if "PAR_Y" in gid:
                continue
            m.setdefault(gid.split(".")[0], re.search(r'gene_name "([^"]+)"', p[8]).group(1))
    return m


def assign_guides(M, guides: list, rule: dict) -> np.ndarray:
    """One guide per cell: the top guide when it has >= 5 UMIs, >= 3x the second and >= 50% of the cell's guide UMIs;
    '' otherwise. The 12 non-targeting guides pool to "non-targeting"."""
    import re

    tot = np.asarray(M.sum(axis=0)).ravel()
    top = M.max(axis=0).toarray().ravel()
    second = np.zeros(M.shape[1])
    for c in range(M.shape[1]):
        v = M.data[M.indptr[c] : M.indptr[c + 1]]
        if v.size >= 2:
            second[c] = np.partition(v, -2)[-2]
    ok = (
        (top >= rule["min_top"]) & (top >= rule["min_ratio"] * second) & (top / np.maximum(tot, 1) >= rule["min_share"])
    )
    gene_of_guide = np.array(
        ["non-targeting" if g.startswith("NO-TARGET") else re.sub(r"_\d+$", "", g) for g in guides], dtype=object
    )
    return np.where(ok, gene_of_guide[np.asarray(M.argmax(axis=0)).ravel()], "")


def table_perturbfate(d: Path, genes: list, spill: Path) -> dict:
    """PerturbFate's A375 CRISPRi screen, DMSO cells only, both arms (the tri arm's old + new RNA summed)."""
    m49 = gencode_names(d / "gencode.v49.basic.annotation.gtf.gz")
    blocks = []
    for arm, spec in PF_ARMS.items():
        meta = pd.read_csv(d / spec["meta"])
        cond = meta["conditions"].str.strip()
        r = Rds(d / spec["sgrna"], spill / arm)
        a = r.attrs(r.obj)
        i, pj, x, dim, guides, sgcells = r.matrix(r.obj)
        x, i = np.asarray(x, dtype=np.float64), np.asarray(i, dtype=np.int64)
        if "p" not in a:  # dgTMatrix: triplets, duplicates summed
            M = sp.coo_matrix((x, (i, np.asarray(pj, dtype=np.int64))), shape=tuple(dim)).tocsc()
        else:
            M = sp.csc_matrix((x, i, np.asarray(pj, dtype=np.int64)), shape=tuple(dim))
            M.sum_duplicates()
        lab = pd.Series(assign_guides(M, guides, PF_RULE), index=sgcells)
        meta["label"] = meta["cell_names"].str.replace("_RNA_", "_", regex=False).map(lab).fillna("")
        sub = meta[(cond == "DMSO_treated_cells") & (meta["label"] != "")].copy()
        X, gene_ids = None, None
        for part, umi_col in zip(spec["counts"], spec["umi_cols"], strict=True):
            rp = Rds(d / part, spill / part)
            ci, cp, cx, _, g_ids, cells = rp.matrix(rp.obj)
            if gene_ids is None:
                gene_ids = g_ids
                pos = pd.Index(cells).get_indexer(sub["cell_names"])
                assert (pos >= 0).all(), f"{arm}: cells missing from the counts"
                order = np.argsort(pos)
                sub = sub.iloc[order].reset_index(drop=True)
                cols = pos[order]
            assert g_ids == gene_ids
            cp = np.asarray(cp)
            starts, ends = cp[cols], cp[cols + 1]
            indptr = np.concatenate([[0], np.cumsum(ends - starts)])
            idx = np.concatenate([np.arange(s0, e0) for s0, e0 in zip(starts, ends, strict=True)])
            Xp = sp.csr_matrix(
                (np.asarray(cx[idx], dtype=np.float32), np.asarray(ci[idx], dtype=np.int32), indptr),
                shape=(cols.size, len(gene_ids)),
            )
            assert np.array_equal(Xp.data, np.round(Xp.data)), "not whole-number counts"
            tot = np.asarray(Xp.sum(axis=1)).ravel()
            assert np.mean(np.isclose(tot, sub[umi_col].to_numpy(dtype=float))) > 0.999, f"{part}: totals disagree"
            X = Xp if X is None else X + Xp
        sym = np.array([m49.get(g.split(".")[0], "") for g in gene_ids], dtype=object)
        blocks.append((X, sub["label"].to_numpy(dtype=object), sym))
    # target names that are old symbols -> today's, through an older GENCODE (the sciPlex annotation) and GENCODE 49
    axset = set(genes)
    ann = pd.read_csv(
        d / "GSM7056149_sciPlexGxE_2_gene.annotations.txt.gz", sep="\t", header=None, names=["ens_ver", "symbol"]
    )
    old2ens = dict(zip(ann["symbol"], ann["ens_ver"].str.split(".").str[0], strict=True))
    aliases = {}
    for t in sorted(set(np.concatenate([b[1] for b in blocks])) - {"non-targeting"}):
        if t in axset:
            continue
        new = m49.get(old2ens[t]) if t in old2ens else None
        if new and new in axset and new != t:
            aliases[t] = new
        elif t in PF_MEMORY_ALIASES and PF_MEMORY_ALIASES[t] in axset:
            aliases[t] = PF_MEMORY_ALIASES[t]
    blocks = [(X, np.array([aliases.get(t, t) for t in lab], dtype=object), sym) for X, lab, sym in blocks]
    return table_from_cells(blocks, "non-targeting", "a375", "sci", genes)


# ---- appendix, the A549 1,000-knockout atlas (GEO GSE345058, a 3.6e9-entry Matrix Market file, streamed) ----
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
A549_KO_GEO = f"{GEO}/GSE345nnn/GSE345058/suppl"
PUBLIC_FILES.update(
    {
        "GSE345058_cell_annotations.csv.gz": (
            f"{A549_KO_GEO}/GSE345058_cell_annotations.csv.gz",
            40152708,
            "5561c19d1020b6bcb7bf7e9a6f7a875df00097d9674723063eb0928cbe849fdf",
        ),
        "GSE345058_gene_annotations.csv.gz": (
            f"{A549_KO_GEO}/GSE345058_gene_annotations.csv.gz",
            378455,
            "7cbed598dcda0125bcc7aad1dbb79c23ac17f296368d2d4f2873535acf54d0e9",
        ),
        "GSE345058_raw_counts.mtx.gz": (
            f"{A549_KO_GEO}/GSE345058_raw_counts.mtx.gz",
            10956707618,
            "abbcdd70f7de6f726c5ef6e0f63d28104f4a383fcda7502ff06d481287352189",
        ),
    }
)


def stream_group_sums(path: Path, group_of_row: np.ndarray, n_genes: int, n_groups: int, chunk: int = 20_000_000):
    """Per-group gene sums of a Matrix Market file (cells x genes, integer counts) without holding it. The counts
    are whole numbers, so the float64 sums are exact in any order."""
    import gzip

    sums = np.zeros(n_groups * n_genes)
    totals = np.zeros(len(group_of_row))
    with gzip.open(path, "rt") as f:
        assert f.readline().startswith("%%MatrixMarket matrix coordinate")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n_rows, n_cols, _ = (int(v) for v in line.split())
        assert n_cols == n_genes and n_rows == len(group_of_row)
        for block in pd.read_csv(
            f, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64, chunksize=chunk, engine="c"
        ):
            r, c, v = block["r"].to_numpy() - 1, block["c"].to_numpy() - 1, block["v"].to_numpy().astype(np.float64)
            totals += np.bincount(r, weights=v, minlength=len(group_of_row))
            g = group_of_row[r]
            ok = g >= 0
            sums += np.bincount(g[ok] * n_genes + c[ok], weights=v[ok], minlength=n_groups * n_genes)
    counts = np.bincount(group_of_row[group_of_row >= 0], minlength=n_groups)
    return sums.reshape(n_groups, n_genes), counts, totals


def table_a549_ko(d: Path, genes: list) -> dict:
    """The knockout atlas as pooled shares per knockout (the mean of per-cell CPM is not recoverable from streamed
    sums) against the non-targeting cells', with the same rules as the other screens."""
    ann = pd.read_csv(
        d / "GSE345058_cell_annotations.csv.gz", usecols=["cell_id", "Perturbation", "Condition", "total_counts"]
    )
    gsym = pd.read_csv(d / "GSE345058_gene_annotations.csv.gz")["gene_symbol"].astype(str).to_numpy()
    labels = np.where(
        ann["Condition"].to_numpy() == "Nontargeting", "non-targeting", ann["Perturbation"].astype(str).to_numpy()
    )
    groups = sorted(set(labels))
    gid = {g: i for i, g in enumerate(groups)}
    sums, counts, totals = stream_group_sums(
        d / "GSE345058_raw_counts.mtx.gz", np.array([gid[x] for x in labels]), len(gsym), len(groups)
    )
    from scipy.stats import spearmanr

    assert float(spearmanr(totals, ann["total_counts"].to_numpy(dtype=np.float64))[0]) > 0.98, "row order"
    pos = pd.Index(genes).get_indexer(pd.Index(gsym))
    keep = np.flatnonzero(pos >= 0)
    S = np.zeros((len(groups), len(genes)))
    np.add.at(S.T, pos[keep], sums[:, keep].T)  # duplicate symbols summed
    median_total = float(np.median(totals))
    share = S / np.maximum(S.sum(1, keepdims=True), 1)
    ctrl = share[gid["non-targeting"]] * 1e6
    targets, rows, n_cells = [], [], []
    for g in groups:
        i = gid[g]
        if g == "non-targeting" or counts[i] < 50:
            continue
        detect = 0.5 / (counts[i] * median_total) * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(np.maximum(share[i] * 1e6, detect) / ctrl)
        v[~(ctrl >= SOURCE_FLOOR)] = np.nan
        (
            targets.append(g),
            rows.append(np.clip(v, -SOURCE_CLIP, SOURCE_CLIP).astype(np.float32)),
            n_cells.append(int(counts[i])),
        )
    return {
        "target": np.array(targets, dtype=str),
        "lfc": np.vstack(rows),
        "n_cells": np.array(n_cells, dtype=np.float64),
        "cell_type": np.array("a549"),
        "chem": np.array("3p"),
        "ctrl_cpm": np.where(ctrl > 0, ctrl, np.nan),
        "control_cells": int(counts[gid["non-targeting"]]),
        "median_total": median_total,
    }


# ---- appendix, the H1 hESC genome-scale screen (GEO GSE295214, 85 lanes of Cell Ranger output) -----------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
# fmt: off
HESC_LANES = [
    ("GSM8943713_jan_10_GEX_A7_sg_H6_filtered_feature_bc_matrix.h5", 98759993,
     "44a4bfe83903fda656026840497a0ed6d50b440b4d3b842693508084c9c5df5b"),
    ("GSM8943714_jan_10_GEX_B7_sg_G6_filtered_feature_bc_matrix.h5", 101097499,
     "87447dc5bd17244d5128f36891dd3ffeeb51c8b3c93850ea176e676e53788d97"),
    ("GSM8943715_jan_10_GEX_C7_sg_F6_filtered_feature_bc_matrix.h5", 26477774,
     "35da86b7d0bfca27cfaea666e9f0d53885ff4346be7c23a251c7ec22f025e82d"),
    ("GSM8943716_jan_10_GEX_D7_sg_E6_filtered_feature_bc_matrix.h5", 94433268,
     "d982f219cff9fa548a65c207ead326b75ad31350c53f8352655846f685317f42"),
    ("GSM8943717_jan_10_GEX_E7_sg_D6_filtered_feature_bc_matrix.h5", 107429644,
     "c16b7e2138ba0bbed79b8b4eab93cf328ce9fe0c8c798258fca2e28d7d2829c9"),
    ("GSM8943718_jan_10_GEX_G7_sg_B6_filtered_feature_bc_matrix.h5", 110671293,
     "cabc6398fbb300fae1134b1600a3b5664fb794902074291a7c6aa5daf129eb20"),
    ("GSM8943719_jan_10_GEX_H7_sg_A6_filtered_feature_bc_matrix.h5", 106796563,
     "e8aed5acb3df5b5c976987eb3b4a11d6337f6d8101c0d2a00de40a7fe5ea09ba"),
    ("GSM8943720_jan_12_GEX_A8_sg_H5_filtered_feature_bc_matrix.h5", 87346774,
     "e535af3a97a5e24af9f77bb202c8b026f05845bbddc2e828b859b2cbc0c17497"),
    ("GSM8943721_jan_12_GEX_B8_sg_G5_filtered_feature_bc_matrix.h5", 93843275,
     "942958430d2c674d338f120b2d8aae5a6d6f1ebebbe591da25e40aebefe437a4"),
    ("GSM8943722_jan_12_GEX_C8_sg_F5_filtered_feature_bc_matrix.h5", 104426637,
     "ddab141f4f4fb2dcdfdf72d4b7243cd3f1ea88ca3de4272aae7ec018f886f7d6"),
    ("GSM8943723_jan_12_GEX_D8_sg_E5_filtered_feature_bc_matrix.h5", 98217826,
     "fadb1b98c355c35cf4c3f20265d082a1bd52bb7130f8d19d8e773ecc415f9f13"),
    ("GSM8943724_jan_12_GEX_E8_sg_D5_filtered_feature_bc_matrix.h5", 109785299,
     "df1bd703284e4f2aabf7f0e55733335585c69f77b6b5349be957858ef6133eb8"),
    ("GSM8943725_jan_12_GEX_F8_sg_C5_filtered_feature_bc_matrix.h5", 17971197,
     "0aaec74d75c4f01b05d7597df4ca9034dc917cad02bfd425e4c260591aed63d5"),
    ("GSM8943726_jan_12_GEX_G8_sg_B5_filtered_feature_bc_matrix.h5", 99674683,
     "65a0209652d164b9139aa30292bd8aad004cba00ea849457aae7cb189b56fbb1"),
    ("GSM8943727_jan_12_GEX_H8_sg_A5_filtered_feature_bc_matrix.h5", 102565556,
     "6aea30e8ba034e78bedcea2293f6f89956734d2c8e3f9a6cc67c2664d99243d5"),
    ("GSM8943728_jan_13_GEX_A9_sg_H4_filtered_feature_bc_matrix.h5", 108010814,
     "49ab14693a81ac4dfcece321172d850a72feab5cfb8246ee6d2096f224b36a13"),
    ("GSM8943729_jan_13_GEX_B9_sg_G4_filtered_feature_bc_matrix.h5", 105218317,
     "765d2e882bff86e7c9fdb4053c3b43d8b81d33f4b9ad46801c4ec0d8ce89fde9"),
    ("GSM8943730_jan_13_GEX_C9_sg_F4_filtered_feature_bc_matrix.h5", 106149308,
     "b675ea9a3dc3a78a892ff0f632a91c0f12f1499565471ec5465e3bc066b3dcdb"),
    ("GSM8943731_jan_13_GEX_D9_sg_E4_filtered_feature_bc_matrix.h5", 102513271,
     "67aa2092ef0ca8e0484948a4e255aa383bc4835b9595d76952287ece3c665cfc"),
    ("GSM8943732_jan_13_GEX_E9_sg_D4_filtered_feature_bc_matrix.h5", 111095146,
     "af9e90178546368049e6fbc74768b3b6dc2847f780809b55f23e99949814404e"),
    ("GSM8943733_jan_13_GEX_F9_sg_C4_filtered_feature_bc_matrix.h5", 128354937,
     "3f2d64333b59d104492bf3c22946e266cc4a26b3d08cd1ab0db51437ba5f9c75"),
    ("GSM8943734_jan_13_GEX_G9_sg_B4_filtered_feature_bc_matrix.h5", 110132183,
     "d028ea4729ebe76b5ad6ab6903f2f13d754b87c5e92255334b2fd6a61c139685"),
    ("GSM8943735_jan_13_GEX_H9_sg_A4_filtered_feature_bc_matrix.h5", 113100406,
     "c061f68cdc2b8e405d97631399395924b8789f7c3bd8c46aaf3e430d25c0be0a"),
    ("GSM8943736_jan_14_GEX_A10_sg_H3_filtered_feature_bc_matrix.h5", 84092909,
     "477e74f79c20f7667f0f09ee46b4b6ff811d40cad670ab523675f5618171665e"),
    ("GSM8943737_jan_14_GEX_B10_sg_G3_filtered_feature_bc_matrix.h5", 96671587,
     "63aa114a672464d50da324916b6cd2c2d0fef5eaa2b9e8012e98020b92cdc29c"),
    ("GSM8943738_jan_14_GEX_C10_sg_F3_filtered_feature_bc_matrix.h5", 93442232,
     "12a81e182e70bfa79b1312250758fb78bfdd726dc2ca2881fa4e1a3152c69672"),
    ("GSM8943739_jan_14_GEX_D10_sg_E3_filtered_feature_bc_matrix.h5", 87308870,
     "9427bb14ba5ace332a6cbea789bc1588651ffb09cc37c9aa86cc8b243ede9c38"),
    ("GSM8943740_jan_14_GEX_E10_sg_D3_filtered_feature_bc_matrix.h5", 96904236,
     "ec272c6d244759b4c68765ed7db453ec0cdd8ccb88dd25849fd1a33ef6dd846e"),
    ("GSM8943741_jan_14_GEX_F10_sg_C3_filtered_feature_bc_matrix.h5", 111087429,
     "d60d73b17cd03168241b97fe9a08a1279892d51d97b846ee28af1b42515cfe35"),
    ("GSM8943742_jan_14_GEX_G10_sg_B3_filtered_feature_bc_matrix.h5", 100524168,
     "e3d60edcf7335f968fe5321a6dbf4618664393a521a3cbc1bd5fc264d17e0dc6"),
    ("GSM8943743_jan_14_GEX_H10_sg_A3_filtered_feature_bc_matrix.h5", 99534875,
     "981673380271b69f47e9b1444ff5c02dcc4d791089c5120a7188c2e610b5476d"),
    ("GSM8943744_jan_16_GEX_A11_sg_H2_filtered_feature_bc_matrix.h5", 105382070,
     "5c5c70bb03c895e580eaeaa26f0903eb0034491776156d161dab461fe625766b"),
    ("GSM8943745_jan_16_GEX_B11_sg_G2_filtered_feature_bc_matrix.h5", 108560222,
     "6fe8461bc427337c5fe06d0b669678d9a9d876b4881ebb2629e1ffeedff6c5c0"),
    ("GSM8943746_jan_16_GEX_C11_sg_F2_filtered_feature_bc_matrix.h5", 109411242,
     "7663e47b6dc299a8c1efe33faa170f05c9fef45afdc0fd1a803806833c9a0327"),
    ("GSM8943747_jan_16_GEX_D11_sg_E2_filtered_feature_bc_matrix.h5", 86669020,
     "984d604b2f9963ce30e80e67f83b4aa78f388d2ae55eb78f879caaa5569fa184"),
    ("GSM8943748_jan_16_GEX_E11_sg_D2_filtered_feature_bc_matrix.h5", 115684497,
     "56dd315424209aa647c662307d6126ffafa888919989a7850e14e4845a9144b0"),
    ("GSM8943749_jan_16_GEX_F11_sg_C2_filtered_feature_bc_matrix.h5", 125704346,
     "c47ef6b9a0a7f41a79e9bf95e9210e607d6ee18f505829d3c6871f09602efca1"),
    ("GSM8943750_jan_16_GEX_G11_sg_B2_filtered_feature_bc_matrix.h5", 107930839,
     "c6049d62811903dc44183146483f317cc6f3fd6c88bf9591b431254400e07f0f"),
    ("GSM8943751_jan_16_GEX_H11_sg_A2_filtered_feature_bc_matrix.h5", 118027918,
     "71fa5a2723caa74ee3c16893b2de6ea0846dc193eb520960089ce5fd3fdc4393"),
    ("GSM8943752_jan_17_GEX_A12_sg_H1_filtered_feature_bc_matrix.h5", 90307138,
     "178d581694829ba8f9479717483d45c213631917615f4bbe246dbb564cde9ec7"),
    ("GSM8943753_jan_17_GEX_B12_sg_G1_filtered_feature_bc_matrix.h5", 90785033,
     "9aa40f7355cfe775aaece59ebed62e60f55f27a3fcc00c5932ed85166f052932"),
    ("GSM8943754_jan_17_GEX_C12_sg_F1_filtered_feature_bc_matrix.h5", 88085541,
     "cc00e3571c092e33959eaad9f96b96c50e11e53b7d4f89cedc810b7e65400a38"),
    ("GSM8943755_jan_17_GEX_D12_sg_E1_filtered_feature_bc_matrix.h5", 97914084,
     "82dd76808a1e098a7bb003cb65cd2933e631114ee56ebef7104d0ad75582b04d"),
    ("GSM8943756_jan_17_GEX_E12_sg_D1_filtered_feature_bc_matrix.h5", 92345220,
     "e47f747a16d0ea1ccc84f4bbc02a6905737c4014baa061a903c08bd82ef5667f"),
    ("GSM8943757_jan_17_GEX_F12_sg_C1_filtered_feature_bc_matrix.h5", 122114197,
     "a1cd44c7f31103c486de8339142758c0b32bcc7d6544be009f2a9df866862e50"),
    ("GSM8943758_jan_17_GEX_G12_sg_B1_filtered_feature_bc_matrix.h5", 83908314,
     "7a71533dac40448e30cd2789c7aaf2d3e1d129fb605b0fa6f2362e0494ea6b7c"),
    ("GSM8943759_jan_17_GEX_H12_sg_A1_filtered_feature_bc_matrix.h5", 80719457,
     "2094f8c64a58686e93711bec58f6cfd19a17e0c71a07e4213f72c96b8b665941"),
    ("GSM8943760_jan_3_GEX_A1_sg_H12_filtered_feature_bc_matrix.h5", 96008017,
     "12cb2364ea38c02425c27cdd770e5cfdf986efd07d661af3aa29c778ac126041"),
    ("GSM8943761_jan_3_GEX_B1_sg_G12_filtered_feature_bc_matrix.h5", 83536191,
     "0a76b5a4036cd00f47b70e7a98f4ebdfd89d33d21d3d2124ac3da91f54ce7063"),
    ("GSM8943762_jan_3_GEX_C1_sg_F12_filtered_feature_bc_matrix.h5", 93489069,
     "55be43cda3afd03ba1ad6ccce646ae1da8a1356ef760bf7971f8947d198e4d18"),
    ("GSM8943763_jan_3_GEX_D1_sg_E12_filtered_feature_bc_matrix.h5", 99859558,
     "483d1e4d0598dbc189615ab5e4981e59f1063cb185ffa67a4b8c3345406acd03"),
    ("GSM8943764_jan_3_GEX_E1_sg_D12_filtered_feature_bc_matrix.h5", 92574896,
     "bb42c27aa3f975a76e1213e081a49862a4991cf413b270c21e4f5a05520b57bc"),
    ("GSM8943765_jan_3_GEX_F1_sg_C12_filtered_feature_bc_matrix.h5", 108128658,
     "31f015df4337436c241a68a7ba7fd98b125cd7eb66e4e1bd2e43cf62e944dada"),
    ("GSM8943766_jan_3_GEX_G1_sg_B12_filtered_feature_bc_matrix.h5", 91570155,
     "91133f228421920b270eb9d06d60764728c7aa0cbb4c9a7816257ca4e99dc732"),
    ("GSM8943767_jan_3_GEX_H1_sg_A12_filtered_feature_bc_matrix.h5", 82356557,
     "4f96cebe315537d07703e1c00d752b5e54c098debc2895af1eae08034dcda8d6"),
    ("GSM8943768_jan_4_GEX_A2_sg_H11_filtered_feature_bc_matrix.h5", 90972293,
     "2fe794ab7a1783d76194b458aef23cd923fae734193453f0cd28d45147f52666"),
    ("GSM8943769_jan_4_GEX_B2_sg_G11_filtered_feature_bc_matrix.h5", 82287807,
     "2603c4fda4cdb307a72958fc1afd26c2a63e45111c8200a38340eb9aedacf82e"),
    ("GSM8943770_jan_4_GEX_C2_sg_F11_filtered_feature_bc_matrix.h5", 72863042,
     "8bca2c2b7e80c6098f436d4af5e1ce32be61b31a61bef2a71a28c69e8fcc47da"),
    ("GSM8943771_jan_4_GEX_D2_sg_E11_filtered_feature_bc_matrix.h5", 161561509,
     "db2057983798663f49ce37f18329b00210bc5a42c76987525338e6496e9fcd22"),
    ("GSM8943772_jan_4_GEX_E2_sg_D11_filtered_feature_bc_matrix.h5", 98999377,
     "a1670087fdb00399c32acf25a5a934691b5a5c40c8396aa3905d7f51e23b1688"),
    ("GSM8943773_jan_4_GEX_F2_sg_C11_filtered_feature_bc_matrix.h5", 89264524,
     "86f5a845d0d03ad15f67edaadba4ea0d6c88ae6c65def96d874ebda35369829e"),
    ("GSM8943774_jan_4_GEX_G2_sg_B11_filtered_feature_bc_matrix.h5", 89607437,
     "74dd91cc65e34a2fbe490bcebd2efb6a9a03bae63cf844b6d6b88131e9e9986c"),
    ("GSM8943775_jan_4_GEX_H2_sg_A11_filtered_feature_bc_matrix.h5", 114388866,
     "d9b0f51f5884fdc67df48afcfd0be5687f9964120de31687219a1bb1d8b07e38"),
    ("GSM8943776_jan_6_GEX_A3_sg_H10_filtered_feature_bc_matrix.h5", 97516311,
     "c70b4fea7e72a79845bfd622b76e231e3d1d049c8f9d85594bfeadfda2c65210"),
    ("GSM8943777_jan_6_GEX_B3_sg_G10_filtered_feature_bc_matrix.h5", 92034104,
     "5309142127f77aa4631ed6e69b27d10bde6bc9a93f5e96d2c8118c13d9d9a72c"),
    ("GSM8943778_jan_6_GEX_C3_sg_F10_filtered_feature_bc_matrix.h5", 95399897,
     "6952f85b8dc6d2df14e639a7405d149d9b21d820f0420f645ea96fe9b678dcd7"),
    ("GSM8943779_jan_6_GEX_D3_sg_E10_filtered_feature_bc_matrix.h5", 99890112,
     "4913350251ed7ca5b8ffc8b76564c4e1c00ba1143cbdb763adc3f8d37198d866"),
    ("GSM8943780_jan_6_GEX_E3_sg_D10_filtered_feature_bc_matrix.h5", 91876952,
     "92955a15024e254b0aaa2aa20656674bbbc5d1bc2d92a4b509b36dc16da0ecbc"),
    ("GSM8943781_jan_6_GEX_F3_sg_C10_filtered_feature_bc_matrix.h5", 117421821,
     "24e04f33b85f5b48e97b5614cf6c27c3d80f4e766df80985caddf322b296a858"),
    ("GSM8943782_jan_6_GEX_G3_sg_B10_filtered_feature_bc_matrix.h5", 106526249,
     "5f7b7c0a4bb290e11d070299a049b4791b5f8c46318fabd56c55a6a259eab09b"),
    ("GSM8943783_jan_6_GEX_H3_sg_A10_filtered_feature_bc_matrix.h5", 99897851,
     "09c40e1dbc141cfc8e0635688b2317bcb3114e455f433e21440da0d929151085"),
    ("GSM8943784_jan_7_GEX_A4_sg_H9_filtered_feature_bc_matrix.h5", 89052992,
     "35012c40059676f44f55eba06db17884679c5e81a83b165577eab63a9a9ca7e1"),
    ("GSM8943785_jan_7_GEX_B4_sg_G9_filtered_feature_bc_matrix.h5", 86298558,
     "173bd7dd20bceb771fac2c00ccf5a7b6991348d756e36770b98201097a71bcde"),
    ("GSM8943786_jan_7_GEX_C4_sg_F9_filtered_feature_bc_matrix.h5", 88495481,
     "7f41d1b50f5acac5850bac0770094c5db0301beba8c5808dfb4d644c12ec781c"),
    ("GSM8943787_jan_7_GEX_D4_sg_E9_filtered_feature_bc_matrix.h5", 86806629,
     "a46549800cf7f88fa523d923760149e18473f8e21cdb1059422c4abcb830d4bb"),
    ("GSM8943788_jan_7_GEX_E4_sg_D9_filtered_feature_bc_matrix.h5", 93967394,
     "301f86e96b912a43fefe8307d86198447f48bcf5e9dfc539f749450b22970a7c"),
    ("GSM8943789_jan_7_GEX_F4_sg_C9_filtered_feature_bc_matrix.h5", 107787554,
     "8896df5eb887c8c91cfd88a992ae546184f6361f4d824751366e503376ef0887"),
    ("GSM8943790_jan_7_GEX_G4_sg_B9_filtered_feature_bc_matrix.h5", 96674861,
     "4a150b159f6ef95b3ec2db176d38fb958548736ac3b3b0104aaf8c64ec1897ec"),
    ("GSM8943791_jan_7_GEX_H4_sg_A9_filtered_feature_bc_matrix.h5", 89178914,
     "dda298c03151b95583c5f001d98f77a3209909a6a14fa429db06a8170645510e"),
    ("GSM8943792_jan_8_GEX_A5_sg_H8_filtered_feature_bc_matrix.h5", 112967402,
     "02fc663bd2250af27d5eeaec4854833697b44ee694e1343a03544b8a5436481f"),
    ("GSM8943793_jan_8_GEX_B5_sg_G8_filtered_feature_bc_matrix.h5", 91007666,
     "dce5a8aa2341eab271aba5af7d479a39999f017da22b871aad7aa68657cbb56e"),
    ("GSM8943794_jan_8_GEX_C5_sg_F8_filtered_feature_bc_matrix.h5", 88179242,
     "c6ccebc07744aaebd48ad838e9f36632952658c6e29965d7e5bfd20c3c3ce06b"),
    ("GSM8943795_jan_8_GEX_D5_sg_E8_filtered_feature_bc_matrix.h5", 98523039,
     "35c77ccfb435af11adc8775fa478589fd5ecb20a0a00b83a6e899cbe2d56c022"),
    ("GSM8943796_jan_8_GEX_E5_sg_D8_filtered_feature_bc_matrix.h5", 109453212,
     "11f6c4a3a91cedbb5d153ac4217939f95ab8b41387887ab61320bb886dfccf3f"),
    ("GSM8943797_jan_8_GEX_F5_sg_C8_filtered_feature_bc_matrix.h5", 116837147,
     "496709fcffb0155513c05ee2658336ed3ea435123ba026bdd095b0ae15d8878d"),
]
# fmt: on


def hesc_url(name: str) -> str:
    """The GEO download link of one H1 lane's file, from its GSM number."""
    gsm = name.split("_")[0]
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}"


for _n, _s, _h in HESC_LANES:
    PUBLIC_FILES[_n] = (hesc_url(_n), _s, _h)


def guide_stats(G):
    """Per cell: guide UMIs, the top guide (index, UMIs) and the second's UMIs."""
    G = sp.csr_matrix(G)
    G.sum_duplicates()
    total = np.asarray(G.sum(1)).ravel()
    coo = G.tocoo()
    order = np.lexsort((-coo.data, coo.row))
    r, c, v = coo.row[order], coo.col[order], coo.data[order].astype(np.float64)
    n = G.shape[0]
    top, second, top_i = np.zeros(n), np.zeros(n), np.full(n, -1)
    if r.size:
        first = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        top[r[first]], top_i[r[first]] = v[first], c[first]
        nxt = first + 1
        ok = nxt < r.size
        ok[ok] = r[nxt[ok]] == r[first[ok]]
        second[r[first[ok]]] = v[nxt[ok]]
    return total, top, top_i, second


def table_hesc(d: Path, gmap: GeneMap, min_cells: int = 50) -> dict:
    """Each lane's cells assigned to their top guide (>= 5 UMIs, >= 3x the second, >= 50% of the cell's guide UMIs),
    guides pooled per gene, per-cell CPM over the axis genes summed per gene; lanes alternate between two halves
    (summed in that order, as the shipped table was)."""
    import re

    import h5py

    axis = list(gmap.axis)
    S, n, ref, lax_ok = {}, {}, None, []
    for li, (name, _, _) in enumerate(HESC_LANES):
        with h5py.File(d / name, "r") as h:
            m = h["matrix"]
            feat = {k: m["features"][k][:].astype(str) for k in ("id", "name", "feature_type")}
            X = sp.csc_matrix(
                (m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(int(x) for x in m["shape"][:])
            )
        assert np.array_equal(X.data, np.round(X.data)), "not whole-number counts"
        if ref is None:
            n_gex = int((feat["feature_type"] == "Gene Expression").sum())
            guides = feat["name"][n_gex:]
            gene_of = np.array(
                [
                    "non-targeting" if g.startswith("non-targeting") else re.match(r"^(.+?)_[+-]_\d+", g).group(1)
                    for g in guides
                ]
            )
            groups = sorted(set(gene_of))
            gidx = {g: i for i, g in enumerate(groups)}
            pos, _ = gmap.map(feat["name"][:n_gex], feat["id"][:n_gex])
            keep = np.flatnonzero(pos >= 0)
            M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(len(pos), len(axis))).tocsc()
            ref = (n_gex, np.array([gidx[g] for g in gene_of]), M, feat["id"])
            for k in ("A0", "A1"):
                S[k], n[k] = np.zeros((len(groups), len(axis))), np.zeros(len(groups))
        n_gex, group_of_guide, M, ids = ref
        assert np.array_equal(feat["id"], ids), f"{name}: feature reference differs"
        Xc = sp.csr_matrix(X.T)
        E, G = Xc[:, :n_gex], Xc[:, n_gex:]
        total, top, top_i, second = guide_stats(G)
        Xa = sp.csr_matrix(E @ M)
        Lax = np.asarray(Xa.sum(1)).ravel()
        CPM = sp.csr_matrix(sp.diags(1e6 / np.maximum(Lax, 1)) @ Xa)
        share = np.where(total > 0, top / np.maximum(total, 1), 0.0)
        ok = (top >= 5) & (top >= 3.0 * second) & (share >= 0.5) & (top_i >= 0)
        cells = np.flatnonzero(ok)
        g = group_of_guide[np.maximum(top_i, 0)][cells]
        k = f"A{li % 2}"
        Ind = sp.csr_matrix((np.ones(cells.size), (g, cells)), shape=(S[k].shape[0], CPM.shape[0]))
        S[k] += (Ind @ CPM).toarray()
        n[k] += np.bincount(g, minlength=S[k].shape[0])
        lax_ok.append(Lax[cells].astype(int))
        del X, Xc, E, G, Xa, CPM
    return table_from_sums(
        S["A0"] + S["A1"],
        n["A0"] + n["A1"],
        groups,
        float(np.median(np.concatenate(lax_ok))),
        gmap,
        "h1",
        "5p",
        min_cells,
    )


def table_from_sums(
    S_A, n_A, groups: list, median_total: float, gmap: GeneMap, cell_type: str, chem: str, min_cells: int = 50
) -> dict:
    """Summed per-cell CPM per guide group -> the source table: mean CPM against the non-targeting group's, groups with
    >= min_cells cells, a zero read as half a molecule of the pool, NaN below 10 control CPM, clipped; rows named by
    the axis symbol where HGNC resolves the library's name."""
    axis = list(gmap.axis)
    ci = groups.index("non-targeting")
    ctrl = S_A[ci] / n_A[ci]
    keep = [i for i, gname in enumerate(groups) if gname != "non-targeting" and n_A[i] >= min_cells]
    lib = [groups[i] for i in keep]
    tpos, _ = gmap.map(lib)
    targets = [axis[p] if p >= 0 else t for t, p in zip(lib, tpos, strict=True)]
    rows = []
    for i in keep:
        detect = 0.5 / (n_A[i] * median_total) * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(np.maximum(S_A[i] / n_A[i], detect) / ctrl)
        v[~(ctrl >= SOURCE_FLOOR)] = np.nan
        rows.append(np.clip(v, -SOURCE_CLIP, SOURCE_CLIP).astype(np.float32))
    return {
        "target": np.array(targets, dtype=str),
        "lfc": np.vstack(rows),
        "n_cells": np.array([float(n_A[i]) for i in keep], dtype=np.float64),
        "cell_type": np.array(cell_type),
        "chem": np.array(chem),
        "ctrl_cpm": np.where(ctrl > 0, ctrl, np.nan).astype(np.float64),
        "control_cells": int(n_A[ci]),
        "median_total": median_total,
    }


# ---- appendix, the DLD-1 screen (Genentech, GEO GSE337988, low-MOI channels, the authors' guide calls) ----
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
# fmt: off
DLD1_FILES = [
    ("filelist.txt", 26662,
     "d7e25e183ccfb62e8d06e08bc321a7944c223d9999b70356e7346465471f2de1",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "filelist.txt")),
    ("GSE337988_family.soft.gz", 10047,
     "2d5243b5f32e1dc6e0bbc6f3d8f5f78447a316103842c69724a83c2515a04561",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/soft/"
      "GSE337988_family.soft.gz")),
    ("GSE337988_MULTIseq_hashing.csv.gz", 665,
     "f660c231f37bad0cccad0fe027df7a7ffecad3950c33b240cde8e327f5090596",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_MULTIseq_hashing.csv.gz")),
    ("GSE337988_NGS6475_antibody_index_mapping_hashing_ALL.csv.gz", 405,
     "55c44645522788f5c11c03f0f98fe216c0f353e9144844f141d08b736d6d0080",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_NGS6475_antibody_index_mapping_hashing_ALL.csv.gz")),
    ("GSE337988_NGS6475_crispr_demux_index_map_prod.csv.gz", 197016,
     "debaf4f170ceb4ed6c44e9e4c0e36068039603169c0027fb2bb2092f1f3d2871",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_NGS6475_crispr_demux_index_map_prod.csv.gz")),
    ("GSE337988_NGS6475_crispr_feature_barcode_MENGNTCUPDATE.csv.gz", 224078,
     "ffc3aa664f75f5687a8be45790f91d6e671cac9b2b80a89b652f1e44d883d10a",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_NGS6475_crispr_feature_barcode_MENGNTCUPDATE.csv.gz")),
    ("GSE337988_NGS6475_Sample_Mapping_File_All.csv.gz", 522,
     "750f5046455e5201bb186d2102490a2da9c753bc704e849a63723665338325c6",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_NGS6475_Sample_Mapping_File_All.csv.gz")),
    ("GSE337988_sublib2_de_matrices_lfc_matrix_Low1.csv.gz", 155201822,
     "aaa70afe2dfefd22b72fe6294b87887bc6ee928590e419f459bcdf9616fba517",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_sublib2_de_matrices_lfc_matrix_Low1.csv.gz")),
    ("GSE337988_sublib2_processed_objects_High_se.rds", 3116339792,
     "fe75e8b2c6f1949d9169d8717df87690ed58829395b87908820a6b6bee4e6e4d",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_sublib2_processed_objects_High_se.rds")),
    ("GSE337988_sublib2_processed_objects_Low_se.rds", 670285340,
     "f894db3509f6d0367c4b5eb89623ed9488ea80b107e33d9433ee7464e80deb4c",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_sublib2_processed_objects_Low_se.rds")),
    ("GSE337988_sublib2_processed_objects_Med_se.rds", 1017440010,
     "ca3c8000b646248ce6e2bdff9e7e07e4a25191c9b991b675b4f67a7290c7760f",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_sublib2_processed_objects_Med_se.rds")),
    ("GSE337988_sublib2_samplesheet.csv.gz", 787,
     "7f074300e3a5405ae2c9597f887dd61b5ac809653424a81edb11a04584e2d94b",
     ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/"
      "GSE337988_sublib2_samplesheet.csv.gz")),
    ("GSM9864227_LIB6266234_SAM24482910.filtered_feature_bc_matrix.h5", 225278460,
     "5ef6e1b48b0680f36e3df660fe45d88012de0dca0de0149e820aa083b211505d",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864227/suppl/"
      "GSM9864227_LIB6266234_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864227_LIB6266234_SAM24482910.metrics_summary.csv.gz", 387,
     "8477d1077e9fff2ba5ebb5f0154547e3ae440a24596b8dbd08b62c8223d0c392",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864227/suppl/"
      "GSM9864227_LIB6266234_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864227_LIB6266278_SAM24482910.crispr.umi_correct.h5", 16381098,
     "957b615b6a0bb50a692d837a067b11699cd9d379a9be5fe5f8dffa8f0fe99384",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864227/suppl/"
      "GSM9864227_LIB6266278_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864228_LIB6266235_SAM24482910.filtered_feature_bc_matrix.h5", 239537401,
     "08d7295336d65784780c51985ada35175df17cb2a67b4331795d4d37c0517251",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864228/suppl/"
      "GSM9864228_LIB6266235_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864228_LIB6266235_SAM24482910.metrics_summary.csv.gz", 385,
     "82de772cc0f9f5ecdb9fe12b33f22946ec914e1276f3f4eb3f834b676d1d6213",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864228/suppl/"
      "GSM9864228_LIB6266235_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864228_LIB6266279_SAM24482910.crispr.umi_correct.h5", 5413171,
     "8e410337334b108093184561cf8333005aa5b21f838e545e877ab51a52a41a3f",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864228/suppl/"
      "GSM9864228_LIB6266279_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864229_LIB6266236_SAM24482910.filtered_feature_bc_matrix.h5", 240825205,
     "b6febf570291f6704efb5a9f4b2f8772473ef3e5bb5750a8bc61e1b0016ec3d5",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864229/suppl/"
      "GSM9864229_LIB6266236_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864229_LIB6266236_SAM24482910.metrics_summary.csv.gz", 385,
     "a7eaaf6e9bd840c62df4a30ff64f349d723ef400bcc1fcbe7d8d8d674e771da6",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864229/suppl/"
      "GSM9864229_LIB6266236_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864229_LIB6266280_SAM24482910.crispr.umi_correct.h5", 27287949,
     "b65c78b21070cdd0c282528098e818458638cff9ab6fe2fa0f1b635c7a7542b7",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864229/suppl/"
      "GSM9864229_LIB6266280_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864230_LIB6266237_SAM24482910.filtered_feature_bc_matrix.h5", 239548855,
     "7a0c40c82975d0e9430dfaf3c096fc3faaf58e021e4e382617e0d4488dfccc19",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864230/suppl/"
      "GSM9864230_LIB6266237_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864230_LIB6266237_SAM24482910.metrics_summary.csv.gz", 387,
     "74731306a8547b2d70e20d63f3240a2024b5d96a97bb825668875b6f3f0123ca",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864230/suppl/"
      "GSM9864230_LIB6266237_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864230_LIB6266281_SAM24482910.crispr.umi_correct.h5", 25263997,
     "e742ba529938524bec84493146ac3ce782b287abb446aea3b87d207340a6c4c4",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864230/suppl/"
      "GSM9864230_LIB6266281_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864231_LIB6266238_SAM24482910.filtered_feature_bc_matrix.h5", 237476197,
     "a79196d0a5da7e952c001bc42cec5eda5f45b9f53a623a52de94e6359cc94f29",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864231/suppl/"
      "GSM9864231_LIB6266238_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864231_LIB6266238_SAM24482910.metrics_summary.csv.gz", 386,
     "8398b3d3b02e78b6a45c76b6648ed1cfae39850c8bab5d6b2a15fde03dfbee8c",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864231/suppl/"
      "GSM9864231_LIB6266238_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864231_LIB6266282_SAM24482910.crispr.umi_correct.h5", 21167616,
     "c9078f0a8731b11d7f3119f38223cdc44401097b37a2405d635047482666c51f",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864231/suppl/"
      "GSM9864231_LIB6266282_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864232_LIB6266239_SAM24482910.filtered_feature_bc_matrix.h5", 238899064,
     "79fea3e3704e2ec887f565c0ddc661c688da6020db1f770080164ca9c90f66b0",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864232/suppl/"
      "GSM9864232_LIB6266239_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864232_LIB6266239_SAM24482910.metrics_summary.csv.gz", 386,
     "51c5fdc3a6b5528bf2cb7b034ccc9a72e8a79aca875f0c95de57691757aa0e42",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864232/suppl/"
      "GSM9864232_LIB6266239_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864232_LIB6266283_SAM24482910.crispr.umi_correct.h5", 20972675,
     "f61301e179f78b9f579feab3a06c6159f4dcc81764d1ed7b2e5a34ea932f8f6e",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864232/suppl/"
      "GSM9864232_LIB6266283_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864233_LIB6266240_SAM24482910.filtered_feature_bc_matrix.h5", 248480426,
     "d91fe2fb0d35f3b2a4084249f43cab614c164039a35855d10f6c34d73b546d7e",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864233/suppl/"
      "GSM9864233_LIB6266240_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864233_LIB6266240_SAM24482910.metrics_summary.csv.gz", 389,
     "9ff09913193e641c002d712eb04f0407500e4fd7d7d384d09b10afaa478ed6c8",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864233/suppl/"
      "GSM9864233_LIB6266240_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864233_LIB6266284_SAM24482910.crispr.umi_correct.h5", 22947312,
     "d821c58d4c18627ade9e0b2da4bddfe26873c5ddc3f07a9a6348d9419251b557",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864233/suppl/"
      "GSM9864233_LIB6266284_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864234_LIB6266241_SAM24482910.filtered_feature_bc_matrix.h5", 230353222,
     "fd2ac1048cfb1556f394b6b7d7976e70cf340e21366253e3b08c813fa4126826",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864234/suppl/"
      "GSM9864234_LIB6266241_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864234_LIB6266241_SAM24482910.metrics_summary.csv.gz", 388,
     "a10e5e8d9fac64dd0469fdf91ce697f9e5e6ffa62fec2e673fb30683f0d77705",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864234/suppl/"
      "GSM9864234_LIB6266241_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864234_LIB6266285_SAM24482910.crispr.umi_correct.h5", 26067217,
     "292fddb341de78c868cccc99cd9dbec9a6de46fdb2b316eb0e7b981922d10a53",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864234/suppl/"
      "GSM9864234_LIB6266285_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864235_LIB6266242_SAM24482910.filtered_feature_bc_matrix.h5", 264023749,
     "076eae6d6fbc7f71869ffcbbdcd30e10f45333f9c42265862a1a7367a1d0af8b",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864235/suppl/"
      "GSM9864235_LIB6266242_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864235_LIB6266242_SAM24482910.metrics_summary.csv.gz", 384,
     "115248b0c2137bb95c1ca0cf419bd958acbbc79fc6a55591850bdf29d3575b89",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864235/suppl/"
      "GSM9864235_LIB6266242_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864235_LIB6266286_SAM24482910.crispr.umi_correct.h5", 27358614,
     "1ce1e8866377876a048203f9f62a424ec2d514c888dff3c28811085eecfeec56",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864235/suppl/"
      "GSM9864235_LIB6266286_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864236_LIB6266243_SAM24482910.filtered_feature_bc_matrix.h5", 211647933,
     "3f208e3cd13dd867f2fa26095c160ab5e3e99903685a328a334b0d05791b2cf7",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864236/suppl/"
      "GSM9864236_LIB6266243_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864236_LIB6266243_SAM24482910.metrics_summary.csv.gz", 385,
     "cf1f17fb3d9194435a85feeed64e8d1d40962366e8c67d1be4671c172baba207",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864236/suppl/"
      "GSM9864236_LIB6266243_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864236_LIB6266287_SAM24482910.crispr.umi_correct.h5", 27092340,
     "1e996efed18a6b4aa5292aaac1acd46faf1657b05e8fef44dfe5efe6805a5467",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864236/suppl/"
      "GSM9864236_LIB6266287_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864237_LIB6266244_SAM24482910.filtered_feature_bc_matrix.h5", 232130344,
     "9418f92f91cdcd364141a2880b37fa3de363f17bbc07eb0101dc2b6b443fc4f3",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864237/suppl/"
      "GSM9864237_LIB6266244_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864237_LIB6266244_SAM24482910.metrics_summary.csv.gz", 387,
     "a9f682eb478f231e29382de32090f081672bc1fafbbb8705f4b7f0dfd71ffcbe",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864237/suppl/"
      "GSM9864237_LIB6266244_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864237_LIB6266288_SAM24482910.crispr.umi_correct.h5", 26421542,
     "00d265477f1db1e3ee99ce3b580ecd790510bc71042212d457f62c760f8a6ec6",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864237/suppl/"
      "GSM9864237_LIB6266288_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864238_LIB6266245_SAM24482910.filtered_feature_bc_matrix.h5", 217850906,
     "de5530ac0d03f4ec1b320731b633aca70bc0662e1c16fe0cbcbd123b4aacdf4f",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864238/suppl/"
      "GSM9864238_LIB6266245_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864238_LIB6266245_SAM24482910.metrics_summary.csv.gz", 387,
     "115e7b623c8e9a312c9d7b8261d566e4d9f23581d1e6ee564c6439ebbb81853b",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864238/suppl/"
      "GSM9864238_LIB6266245_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864238_LIB6266289_SAM24482910.crispr.umi_correct.h5", 28467289,
     "52bbed2dfd7f7f1f508f7e7f85d0cfa1121d921f1fdf1414cdcb137abeb62045",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864238/suppl/"
      "GSM9864238_LIB6266289_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864239_LIB6266246_SAM24482910.filtered_feature_bc_matrix.h5", 215357022,
     "22fa32643c4519b5b09c758b7681b800b63eaeeb17191383e4fb7fd7bc356cca",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864239/suppl/"
      "GSM9864239_LIB6266246_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864239_LIB6266246_SAM24482910.metrics_summary.csv.gz", 389,
     "c5e4ff4d479efe61e825bd32095bfdd06162c8fad6093ac97e353d9a3eb8da74",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864239/suppl/"
      "GSM9864239_LIB6266246_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864239_LIB6266290_SAM24482910.crispr.umi_correct.h5", 32722270,
     "0596a99b005feeda2d51722b00c867001a5fd51e5a01ad75675905b8b60db9d1",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864239/suppl/"
      "GSM9864239_LIB6266290_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864240_LIB6266247_SAM24482910.filtered_feature_bc_matrix.h5", 223875819,
     "b5c01865c5d152c4c1326f8e1efa681840bae2d4e18f0387d83d1c7add2bea54",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864240/suppl/"
      "GSM9864240_LIB6266247_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864240_LIB6266247_SAM24482910.metrics_summary.csv.gz", 386,
     "9ed52db71d50ca40ae35cd21f040a003ba086cebce9dcaad2f5feb27594947da",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864240/suppl/"
      "GSM9864240_LIB6266247_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864240_LIB6266291_SAM24482910.crispr.umi_correct.h5", 32686936,
     "2066b6c3bc2d88c81069c672a9afe89d259d21ccd11fd7033e575d93d8865c9c",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864240/suppl/"
      "GSM9864240_LIB6266291_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864241_LIB6266248_SAM24482910.filtered_feature_bc_matrix.h5", 239650178,
     "9708b5afeb21a0a4cd69ced2bbdf77d1f6673ac12af6c838a7f2a8f104deaa8e",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864241/suppl/"
      "GSM9864241_LIB6266248_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864241_LIB6266248_SAM24482910.metrics_summary.csv.gz", 384,
     "8b5f097997573edaceb90afe833e21edd35d28d79824845497a993bd9581f449",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864241/suppl/"
      "GSM9864241_LIB6266248_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864241_LIB6266292_SAM24482910.crispr.umi_correct.h5", 35136481,
     "61e7c9a3d99e30e2cc2962adc041addeedc950a65cfd8f58db2b1bda68f4ad10",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864241/suppl/"
      "GSM9864241_LIB6266292_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864242_LIB6266249_SAM24482910.filtered_feature_bc_matrix.h5", 223939384,
     "6fa16cb417defcd7560fd94228c6f017a8b159ff87051c0c11a239ef732c79e8",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864242/suppl/"
      "GSM9864242_LIB6266249_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864242_LIB6266249_SAM24482910.metrics_summary.csv.gz", 390,
     "839314cf438ff4d0fd6fc06ddf35e6494a8235d4bcaae662851de1d051ed2196",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864242/suppl/"
      "GSM9864242_LIB6266249_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864242_LIB6266293_SAM24482910.crispr.umi_correct.h5", 30239691,
     "ec8f4e24c1cadd3afdc66eea5abbf3638919b9d2a5b17fec96590b607eab7713",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864242/suppl/"
      "GSM9864242_LIB6266293_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864243_LIB6266250_SAM24482910.filtered_feature_bc_matrix.h5", 255343945,
     "e388dc702cdd0f0179fa63c13f2fc49971615af6cabc35ad1020854e377dccd5",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864243/suppl/"
      "GSM9864243_LIB6266250_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864243_LIB6266250_SAM24482910.metrics_summary.csv.gz", 386,
     "6d1fc17bb25411c11392cb6200d8e8dd99631dc6adae2a68b8418b5a46b58574",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864243/suppl/"
      "GSM9864243_LIB6266250_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864243_LIB6266294_SAM24482910.crispr.umi_correct.h5", 13226976,
     "773d9d3ecaea20e648e169a2c046ef3b97f6d4dac1e65c87b0f1ddd9096ed129",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864243/suppl/"
      "GSM9864243_LIB6266294_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864244_LIB6266251_SAM24482910.filtered_feature_bc_matrix.h5", 250403383,
     "36607c52abab4cbc52c8cc926758438e44bc67a6538dc36984b81959f97d1a12",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864244/suppl/"
      "GSM9864244_LIB6266251_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864244_LIB6266251_SAM24482910.metrics_summary.csv.gz", 387,
     "b344bcf036b0fd78208b135dabf83a5e83b01ad525ed68aa66a8c59fd042a928",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864244/suppl/"
      "GSM9864244_LIB6266251_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864244_LIB6266295_SAM24482910.crispr.umi_correct.h5", 7917263,
     "579cd37ab69d4c4b9c3ce2f431433649dae10754ae27c1952c0c3a4368ad0ca6",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864244/suppl/"
      "GSM9864244_LIB6266295_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864245_LIB6266252_SAM24482910.filtered_feature_bc_matrix.h5", 271177024,
     "6e57b886dd42baa1ea7149680341ac4445857d3453263f897b9c39c144b944b0",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864245/suppl/"
      "GSM9864245_LIB6266252_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864245_LIB6266252_SAM24482910.metrics_summary.csv.gz", 390,
     "fb185b3e0afba23fc9e70fa8b6a9d8c7a1b8a06c6c89c839c606a9656e00eb44",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864245/suppl/"
      "GSM9864245_LIB6266252_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864245_LIB6266296_SAM24482910.crispr.umi_correct.h5", 17852232,
     "905a1b5763dd76c72816b6f7db05a0bd58b8ba07dc8570caba04569189e9df0d",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864245/suppl/"
      "GSM9864245_LIB6266296_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864246_LIB6266253_SAM24482910.filtered_feature_bc_matrix.h5", 267956530,
     "7b4eefbe3448076dfaf162827f860e9ccf2fd3e90f6f6c3cc8d4e22b9072e658",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864246/suppl/"
      "GSM9864246_LIB6266253_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864246_LIB6266253_SAM24482910.metrics_summary.csv.gz", 390,
     "a56386c4d075faead2aeed0d0ca4173473134d7e018a8441269eec7e397f0a28",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864246/suppl/"
      "GSM9864246_LIB6266253_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864246_LIB6266297_SAM24482910.crispr.umi_correct.h5", 19719279,
     "cb64d6deb5bdd68d8e437c2a6940ca944100893ddd0e26af16d220e4d9e1b34d",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864246/suppl/"
      "GSM9864246_LIB6266297_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864247_LIB6266254_SAM24482910.filtered_feature_bc_matrix.h5", 242131118,
     "01978d65dfd73705f85c2f7ace040936d885c22847559297045784e398fa7793",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864247/suppl/"
      "GSM9864247_LIB6266254_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864247_LIB6266254_SAM24482910.metrics_summary.csv.gz", 386,
     "829cd04a053e69f93698af14a48afabe8390207d573895fbbdeed2eaab53d723",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864247/suppl/"
      "GSM9864247_LIB6266254_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864247_LIB6266298_SAM24482910.crispr.umi_correct.h5", 18149532,
     "b904bdd84cf32eba28576dc8a39765850fe836c1eacebd4d3d44c3bf700f6fa2",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864247/suppl/"
      "GSM9864247_LIB6266298_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864248_LIB6266255_SAM24482910.filtered_feature_bc_matrix.h5", 270904897,
     "20868b700398d600c75dde7d535ee2cdecbb86259a8f8c8c6422e1a321408c47",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864248/suppl/"
      "GSM9864248_LIB6266255_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864248_LIB6266255_SAM24482910.metrics_summary.csv.gz", 387,
     "81a2891ad6b6bb049aa0f726e95388f15cc472031fed97c1fa8d088f9fae2c2d",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864248/suppl/"
      "GSM9864248_LIB6266255_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864248_LIB6266299_SAM24482910.crispr.umi_correct.h5", 16458548,
     "111aeaad9b96c27c8b6951b3968726d308d0e35e9578227a200a884995c19a29",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864248/suppl/"
      "GSM9864248_LIB6266299_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864249_LIB6266256_SAM24482910.filtered_feature_bc_matrix.h5", 253561363,
     "33dd17ef720432d32a60aa71abdf73f6048778d2a2afc7c20353341adffa7c16",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864249/suppl/"
      "GSM9864249_LIB6266256_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864249_LIB6266256_SAM24482910.metrics_summary.csv.gz", 387,
     "1f91482b537212821f937d2e508741184a76fe529a09cbe4628315562146b859",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864249/suppl/"
      "GSM9864249_LIB6266256_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864249_LIB6266300_SAM24482910.crispr.umi_correct.h5", 13264865,
     "1705215309263037a24744341d2d508d6c3a3535e37d565ea60a2ca553977f04",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864249/suppl/"
      "GSM9864249_LIB6266300_SAM24482910.crispr.umi_correct.h5")),
    ("GSM9864250_LIB6266257_SAM24482910.filtered_feature_bc_matrix.h5", 247461436,
     "06a2fb70013e59f60cc3c278e597c127e7b48d882cb9bcb82f6bcf740621199c",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864250/suppl/"
      "GSM9864250_LIB6266257_SAM24482910.filtered_feature_bc_matrix.h5")),
    ("GSM9864250_LIB6266257_SAM24482910.metrics_summary.csv.gz", 384,
     "724afc988c1d544e9078be36e7f6d55a1001437946c245b459b998a236dccffb",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864250/suppl/"
      "GSM9864250_LIB6266257_SAM24482910.metrics_summary.csv.gz")),
    ("GSM9864250_LIB6266301_SAM24482910.crispr.umi_correct.h5", 13996617,
     "c618b14df4effdf6387c4e287767239e283eebbbce64c9974102efc930d17831",
     ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9864nnn/GSM9864250/suppl/"
      "GSM9864250_LIB6266301_SAM24482910.crispr.umi_correct.h5")),
]
# fmt: on
for _n, _s, _h, _u in DLD1_FILES:
    PUBLIC_FILES[_n] = (_u, _s, _h)


def dld1_channels(d: Path) -> list:
    """The 24 low-MOI channels in order: sample, its half (odd / even), its gene-expression H5."""
    ss = pd.read_csv(d / "GSE337988_sublib2_samplesheet.csv.gz")
    files = [ln.split("\t")[1] for ln in (d / "filelist.txt").read_text().splitlines()[1:]]
    out = []
    for sample, g in ss.groupby("Sample", sort=False):
        if not sample.startswith("SAM24482910_"):
            continue
        lib = dict(zip(g["DataType"], g["Library"], strict=True))
        rna = next(x for x in files if f"_{lib['rna']}" in x and x.endswith("filtered_feature_bc_matrix.h5"))
        k = int(sample.split("_")[1])
        out.append({"sample": sample, "k": k, "half": k % 2, "rna": rna})
    return sorted(out, key=lambda c: c["k"])


def dld1_coldata(rds: Path, spill: Path) -> pd.DataFrame:
    """The SummarizedExperiment's colData (the authors' per-cell hash and guide calls)."""
    import io

    r = Rds(rds, spill)
    cd = dict(r.pairlist(r.obj.attributes))["colData"]
    slots = dict(r.pairlist(r.deref(cd).attributes))
    cols = {}
    for k, v in r.listing(slots["listData"]).items():
        v = r.deref(v)
        if v.info.type == r.T.STR:
            cols[k] = r.strings(v)
        elif "class" in r.attrs(v) and "factor" in r.strings(r.attrs(v)["class"]):
            lev = r.strings(r.attrs(v)["levels"])
            cols[k] = [lev[i - 1] if i > 0 else None for i in np.asarray(v.value)]
        else:
            cols[k] = np.asarray(v.value)
    df = pd.DataFrame(cols, index=r.strings(slots["rownames"]))
    buf = io.StringIO()  # through text, as when the shipped table was built (NA strings read back as "")
    df.to_csv(buf, sep="\t")
    buf.seek(0)
    return pd.read_csv(buf, sep="\t", index_col=0, keep_default_na=False, low_memory=False)


def table_dld1(d: Path, gmap: GeneMap, spill: Path) -> dict:
    """Cells singly assigned by the authors (hash singlet and guide singlet), guides pooled per gene through the
    deposit's guide map (NTC -> non-targeting), per-cell CPM over the axis genes summed per gene, two channel halves."""
    import h5py

    gm = pd.read_csv(d / "GSE337988_NGS6475_crispr_demux_index_map_prod.csv.gz")
    g2g = {g: ("non-targeting" if s == "NTC" else s) for g, s in zip(gm["index"], gm["sample"], strict=True)}
    groups = sorted(set(g2g.values()))
    gid = {g: i for i, g in enumerate(groups)}
    cd_all = dld1_coldata(d / "GSE337988_sublib2_processed_objects_Low_se.rds", spill)
    single = (cd_all["demux_type"] == "singlet") & (cd_all["demux_type.fishash"] == "singlet")
    gene = cd_all["assignment.fishash"].map(g2g).fillna("").to_numpy(dtype=object)
    cd_all = cd_all.assign(main=np.where(single.to_numpy(), gene, "").astype(str))
    G = len(gmap.axis)
    S = {k: np.zeros((len(groups), G)) for k in ("A0", "A1")}
    n = {k: np.zeros(len(groups)) for k in ("A0", "A1")}
    M, lax = None, []
    for ch in dld1_channels(d):
        with h5py.File(d / ch["rna"], "r") as h:
            m = h["matrix"]
            X = sp.csc_matrix(
                (m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(int(x) for x in m["shape"][:])
            )
            bc = m["barcodes"][:].astype(str)
            feat = {k: m["features"][k][:].astype(str) for k in ("id", "name")}
        if M is None:
            pos, _ = gmap.map(feat["name"], feat["id"])
            keep = np.flatnonzero(pos >= 0)
            M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(len(pos), G)).tocsc()
        cd = cd_all[cd_all["Sample"] == ch["sample"]]
        col = pd.Index([f"{ch['sample']}-{b.split('-')[0]}" for b in bc]).get_indexer(cd.index)
        assert (col >= 0).all(), f"{ch['sample']}: colData cells missing from the H5"
        Xc = sp.csr_matrix(X.T)[col]
        Xa = sp.csr_matrix(Xc @ M)
        Lax = np.asarray(Xa.sum(1)).ravel()
        CPM = sp.csr_matrix(sp.diags(1e6 / np.maximum(Lax, 1)) @ Xa)
        lab = cd["main"].to_numpy(dtype=str)
        cells = np.flatnonzero(lab != "")
        key = f"A{ch['half']}"
        g = np.array([gid[x] for x in lab[cells]])
        Ind = sp.csr_matrix((np.ones(cells.size), (g, cells)), shape=(len(groups), CPM.shape[0]))
        S[key] += (Ind @ CPM).toarray()
        n[key] += np.bincount(g, minlength=len(groups))
        lax.append(Lax[cells].astype(int))
    return table_from_sums(
        S["A0"] + S["A1"], n["A0"] + n["A1"], groups, float(np.median(np.concatenate(lax))), gmap, "dld1", "3p"
    )


# ---- appendix, H1 held-out splits (Arc 2025) and CD4 T pseudobulk (Zhu, Dann et al. 2026) ----------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
PUBLIC_FILES.update(
    {
        "adata_Validation.h5ad": (
            (
                "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/"
                "validation/adata_Validation.h5ad"
            ),
            6928967541,
            "376f0bab27d9f22e00dfd888668d3602006d035edc182721c378f4802c77bfe1",
        ),
        "adata_Test.h5ad": (
            (
                "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/test/"
                "adata_Test.h5ad"
            ),
            11950739168,
            "ba5ce66729e4c8e50b6f147c7d73d3f9f28f76c91fa43ff33e3f6bc2ed831f03",
        ),
        "GWCD4i.pseudobulk_merged.h5ad": (
            ("https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad"),
            44566657140,
            "fd2b8c21d357f8699ec34e2d5ebc1639612c27a0147a9ca94d4983822d93247e",
        ),
    }
)
TABLE_CLIP = 10.0  # these tables keep log2 fold changes to +-10 (the consensus clips to +-6 when it reads them)


def stream_mean_cpm(
    path: Path, labels: pd.Categorical, col_map, weights=None, block: int = 20_000, parts: int = 1, counts=None
):
    """Per label: the (weighted) sum of per-row CPM on the axis, the summed weights (cells) and the axis UMIs, read in
    row blocks of the h5ad (each row normalised to its total on the axis). `parts` > 1 sums the rows in that many
    contiguous parts, each from zero, then adds the parts in order: the summation order of a build that split the
    rows over processes. `counts` turns a block, as stored, into whole counts (e.g. from log(1 + count))."""
    import h5py
    from anndata.io import sparse_dataset

    codes = np.asarray(labels.codes)
    n_lab = len(labels.categories)
    acc, n, u = None, np.zeros(n_lab), np.zeros(n_lab)
    with h5py.File(path, "r") as f:
        Xs = sparse_dataset(f["X"]) if isinstance(f["X"], h5py.Group) else f["X"]
        cuts = np.linspace(0, Xs.shape[0], parts + 1).astype(int)
        for p0, p1 in pairwise(cuts):
            pa, pn, pu = np.zeros((n_lab, col_map.shape[1])), np.zeros(n_lab), np.zeros(n_lab)
            for s in range(p0, p1, block):
                e = min(p1, s + block)
                c = codes[s:e]
                keep = np.flatnonzero(c >= 0)
                if not keep.size:
                    continue
                X = Xs[s:e]
                X = sp.csr_matrix(X if sp.issparse(X) else np.asarray(X), dtype=np.float64)
                if counts is not None:
                    X = counts(X)
                X = X[keep] @ col_map
                L = np.asarray(X.sum(axis=1)).ravel()
                w = np.ones(keep.size) if weights is None else weights[s:e][keep]
                X = sp.csr_matrix(sp.diags(w * 1e6 / np.maximum(L, 1)) @ X)
                Sm = sp.csr_matrix((np.ones(keep.size), (c[keep], np.arange(keep.size))), shape=(n_lab, keep.size))
                M = (Sm @ X).tocoo()
                pa[M.row, M.col] += M.data
                pn += np.bincount(c[keep], weights=w, minlength=n_lab)
                pu += np.bincount(c[keep], weights=L, minlength=n_lab)
            if parts == 1:
                acc, n, u = pa, pn, pu
            else:
                if acc is None:
                    acc = np.zeros_like(pa)
                acc += pa
                n += pn
                u += pu
    return list(labels.categories.astype(str)), acc, n, u


def labels_table(targets, contexts, sums, n, umis, gmap: GeneMap, strata=None, mix=None):
    """Mean CPM per label (a target in a context) -> log2FC rows (>= 50 cells) against the context's pooled controls,
    or, for donor-pooled data (`mix` = donors, labels x donors cells), against the controls of the knockdown's own
    donor mix. Returns (target, context, n_cells, lfc, per-context control CPM)."""
    targets, contexts = np.asarray(targets, dtype=str), np.asarray(contexts, dtype=str)
    strata = np.full(targets.size, "") if strata is None else np.asarray(strata, dtype=str)
    is_ctrl = targets == "non-targeting"
    ctx_names = sorted(set(contexts))
    ctrl, pooled = {}, {}
    for c in ctx_names:
        ci = np.flatnonzero(is_ctrl & (contexts == c) & (n > 0))
        for i in ci:
            ctrl[(c, strata[i])] = sums[i] / n[i]
        pooled[c] = (sums[ci].sum(0) / n[ci].sum(), n[ci].sum())
    ko = np.flatnonzero(~is_ctrl & (n >= 50))
    names = sorted(set(targets[ko]))
    tpos, _ = gmap.map(names)
    to_axis = {t: (gmap.axis[p] if p >= 0 else t) for t, p in zip(names, tpos, strict=True)}
    lfc = np.empty((ko.size, sums.shape[1]), dtype=np.float32)
    for r, k in enumerate(ko):
        c = contexts[k]
        if mix is None:
            den = pooled[c][0]
        else:
            donors, w = mix[0], mix[1][k]
            den = sum(x * ctrl[(c, d)] for d, x in zip(donors, w, strict=True) if x > 0) / w.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(np.maximum(sums[k] / n[k], 0.5e6 / max(umis[k], 1.0)) / den)
        v = np.clip(v, -TABLE_CLIP, TABLE_CLIP)
        v[~(den >= SOURCE_FLOOR) | ~(den > 0)] = np.nan
        lfc[r] = v.astype(np.float32)
    ctrl_cpm = {c: pooled[c][0].astype(np.float32) for c in ctx_names}
    return (np.array([to_axis[t] for t in targets[ko]]), contexts[ko], np.round(n[ko]).astype(np.int64), lfc, ctrl_cpm)


def table_arc_h1(d: Path, gmap: GeneMap) -> dict:
    """Arc's 2025 H1 validation and test splits (10x Flex), raw counts; the controls both files carry count once."""
    import h5py
    from anndata.io import read_elem

    sums, ns, us, seen = [], [], [], set()
    for name in ("adata_Validation.h5ad", "adata_Test.h5ad"):
        with h5py.File(d / name, "r") as f:
            obs, var = read_elem(f["obs"]), read_elem(f["var"]).index.astype(str)
        names = obs.index.astype(str)
        lab = obs["target_gene"].astype(str).where(~names.isin(seen))
        seen |= set(names)
        pos, _ = gmap.map(var)
        keep = np.flatnonzero(pos >= 0)
        M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(len(pos), len(gmap.axis)))
        labels, s, k, u = stream_mean_cpm(d / name, pd.Categorical(lab), M)
        (
            sums.append(pd.DataFrame(s, index=labels)),
            ns.append(pd.Series(k, index=labels)),
            us.append(pd.Series(u, index=labels)),
        )
    total = pd.concat(sums).groupby(level=0).sum()
    n = pd.concat(ns).groupby(level=0).sum().reindex(total.index)
    u = pd.concat(us).groupby(level=0).sum().reindex(total.index)
    t, _, nc, lfc, cc = labels_table(
        list(total.index), ["h1"] * len(total), total.to_numpy(), n.to_numpy(), u.to_numpy(), gmap
    )
    return {
        "target": t.astype(str),
        "lfc": lfc,
        "n_cells": nc,
        "cell_type": np.array("h1"),
        "chem": np.array("flex"),
        "ctrl_cpm": np.asarray(cc["h1"], np.float64),
    }


def tables_cd4(d: Path, gmap: GeneMap) -> dict:
    """CD4 T cells, pseudobulk UMI sums per guide x donor x condition: rows passing the authors' cell and count QC,
    pooled per target x condition weighted by cells, against each knockdown's own donor mix of controls; per
    condition, genes whose median change over all knockdowns exceeds 0.5 in size are masked (a condition-wide shift,
    not a knockdown's)."""
    import h5py
    from anndata.io import read_elem

    path = d / "GWCD4i.pseudobulk_merged.h5ad"
    with h5py.File(path, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
    qc = obs["keep_min_cells"].to_numpy(dtype=bool) & obs["keep_total_counts"].to_numpy(dtype=bool)
    nt = obs["guide_type"].astype(str).str.contains("non.?target", case=False).to_numpy()
    target = np.where(nt, "non-targeting", obs["perturbed_gene_name"].astype(str).to_numpy())
    cond, donor = obs["culture_condition"].astype(str).to_numpy(), obs["donor_id"].astype(str).to_numpy()
    label = pd.Series(
        [f"{t}|{c}|{dn}" if x else f"{t}|{c}" for t, c, dn, x in zip(target, cond, donor, nt, strict=True)]
    )
    pos, _ = gmap.map(var["gene_name"].astype(str), var["gene_ids"].astype(str).to_numpy())
    keep = np.flatnonzero(pos >= 0)
    M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(len(pos), len(gmap.axis)))
    w = obs["n_cells"].to_numpy(float)
    names, sums, n, u = stream_mean_cpm(path, pd.Categorical(label.where(qc)), M, weights=w, block=5_000)
    parts = [x.split("|") for x in names]
    donors = sorted(set(donor[qc]))
    cells = (
        pd.DataFrame({"label": label[qc].to_numpy(), "donor": donor[qc], "n": w[qc]})
        .pivot_table(index="label", columns="donor", values="n", aggfunc="sum")
        .reindex(index=names, columns=donors)
        .fillna(0.0)
    )
    t, ctx, nc, lfc, cc = labels_table(
        [p[0] for p in parts],
        [p[1] for p in parts],
        sums,
        n,
        u,
        gmap,
        strata=[p[2] if len(p) == 3 else "" for p in parts],
        mix=(donors, cells.to_numpy()),
    )
    out = {}
    for c in ("Rest", "Stim8hr", "Stim48hr"):
        rows = np.flatnonzero(ctx == c)
        L = lfc[rows].copy()
        bad = np.abs(np.nan_to_num(np.nanmedian(L, axis=0))) > 0.5
        L[:, bad] = np.nan
        out[f"cd4_{c}"] = {
            "target": t[rows].astype(str),
            "lfc": L,
            "n_cells": nc[rows],
            "cell_type": np.array("cd4t"),
            "chem": np.array("flex"),
            "ctrl_cpm": np.asarray(cc[c], np.float64),
        }
    return out


# ---- appendix, the derived tables (per-source principal axes, overdispersion) ----------------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
SRCPC_THREADS = {"dld1_gx": 24, "orion_hct116": 24, "orion_hek293t": 24}  # added later, by a run at 24 BLAS threads


def source_pc_axes(srcs: Sources, name: str, genes: list, k_max: int = 10, threads: int | None = None) -> np.ndarray:
    """A source's top principal axes (k x genes, float32): its clipped rows centred on `cmean` (each gene's mean
    over the source's knockdowns, its own knockdown left out), each knockdown's own gene zeroed (it would
    dominate an axis), by the randomized SVD (seed 0); k = min(10, rows // 3)."""
    gidx = {g: i for i, g in enumerate(genes)}
    s = srcs.load(name)
    L = np.clip(s["lfc"], -CLIP, CLIP)
    X = np.nan_to_num(L - own_out_mean(L, s["target"], gidx)[None, :]).astype(np.float32)
    del L, s["lfc"]
    for i, t in enumerate(s["target"]):
        if t in gidx:
            X[i, gidx[t]] = 0.0
    k = min(k_max, max(X.shape[0] // 3, 0))
    # built at 2 BLAS threads (the later screens at 24): another count moves entries by about one float32 ulp
    with blas_threads(threads or SRCPC_THREADS.get(name, 2)):
        V = rsvd(X, k) if k >= 1 else np.zeros((0, X.shape[1]), np.float32)
    return V.astype(np.float32)


def table_phi(references: dict) -> dict:
    """Per-gene negative-binomial overdispersion of each training line's controls (one-state fit), and the
    median over the lines of the positive values (genes with none: the median of the medians)."""
    import anndata as ad

    rows = {}
    for line, path in references.items():
        ref = ad.read_h5ad(path)
        lab = ref.obs["target_gene"].astype(str).to_numpy()
        rows[line] = fit_controls(sp.csr_matrix(ref.X[lab == "non-targeting"]), n_states=1).states[0].inv_theta
        del ref
    phi = np.vstack(list(rows.values()))
    med = np.nanmedian(np.where(phi > 0, phi, np.nan), axis=0)
    med = np.nan_to_num(med, nan=float(np.nanmedian(med)))
    return {"phi_median": med, **{f"phi_{k}": v for k, v in rows.items()}}


# ---- appendix, split halves (the reliability of one knockdown's profile, section 5) ----------------------
# Used in: the appendix (rebuilding the reference tables from the public downloads); skip it on a first read.
def split_labels(target: np.ndarray, keep: set, rng, control: str | None = None) -> np.ndarray:
    """Per cell: '<target>|A' or '|B' for the knockdowns in `keep` (the first floor(n/2) of a random permutation are A),
    'ctrl' for the controls, else None (not read)."""
    lab = np.full(target.size, None, dtype=object)
    for t in sorted(keep):
        rows = np.flatnonzero(target == t)
        if rows.size < 2:
            continue
        rows = rows[rng.permutation(rows.size)]
        h = rows.size // 2
        lab[rows[:h]] = f"{t}|A"
        lab[rows[h:]] = f"{t}|B"
    if control is not None:
        lab[target == control] = "ctrl"
    return lab


def halves_table(labels: list, sums: np.ndarray, n: np.ndarray, ctrl: np.ndarray, gmap: GeneMap) -> dict:
    """Per knockdown with both halves: each half's log2 ratio of mean CPM over the controls' (clipped to +-10, NaN where
    the controls have none; this table keeps zeros at the clip, not at a detection limit)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    rows = {}
    for lab in labels:
        if lab != "ctrl":
            key, half = lab.rsplit("|", 1)
            rows.setdefault(key, {})[half] = idx[lab]
    out = {"t": [], "n_a": [], "n_b": [], "lfc_a": [], "lfc_b": []}

    def ratio(num):
        """log2 of `num` over the controls' CPM, clipped to +-10, NaN where the controls have none (float32)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.clip(np.log2(num / ctrl), -TABLE_CLIP, TABLE_CLIP)
        r[~(ctrl > 0)] = np.nan
        return r.astype(np.float32)

    for k in sorted(k for k, v in rows.items() if "A" in v and "B" in v):
        a, b = rows[k]["A"], rows[k]["B"]
        if n[a] == 0 or n[b] == 0:
            continue
        out["t"].append(k), out["n_a"].append(n[a]), out["n_b"].append(n[b])
        out["lfc_a"].append(ratio(sums[a] / n[a])), out["lfc_b"].append(ratio(sums[b] / n[b]))
    uniq = sorted(set(out["t"]))
    pos, _ = gmap.map(uniq)
    to = {t: (gmap.axis[p] if p >= 0 else t) for t, p in zip(uniq, pos, strict=True)}
    return {
        "target": np.array([to[t] for t in out["t"]]),
        "source_target": np.array(out["t"]),
        "context": np.array([""] * len(out["t"])),
        "n_a": np.array(out["n_a"]),
        "n_b": np.array(out["n_b"]),
        "lfc_a": np.vstack(out["lfc_a"]).astype(np.float32),
        "lfc_b": np.vstack(out["lfc_b"]).astype(np.float32),
        "ctrl_cpm": np.vstack([ctrl]).astype(np.float32),
        "ctrl_context": np.array([""]),
    }


def halves_line(line: str, d: Path, gmap: GeneMap, knockdowns: set) -> dict:
    """Split halves of one training line's single cells (seed 0), the `knockdowns` with a table row; the Replogle and
    Nadig files are read in three contiguous parts, the 2025 H1 file (log(1 + count)) in one."""
    import anndata as ad

    f, pert_col, sym_col, _, _ = BENCH_LINES[line]
    a = ad.read_h5ad(d / f, backed="r")
    target = a.obs[pert_col].astype(str).to_numpy()
    sym = (a.var[sym_col] if sym_col else a.var_names).astype(str)
    a.file.close()
    pos = pd.Index(gmap.axis).get_indexer(pd.Index(sym))  # align_genes' rule: exact symbol, duplicates summed
    keep = np.flatnonzero(pos >= 0)
    M = sp.csr_matrix((np.ones(keep.size), (keep, pos[keep])), shape=(len(sym), len(gmap.axis)))
    lab = pd.Categorical(split_labels(target, knockdowns, np.random.default_rng(0), control="non-targeting"))

    def from_log1p(X):
        """A block stored as log(1 + count), turned back into whole counts (checked)."""
        X.data = np.expm1(X.data)
        assert (float(np.abs(X.data - np.round(X.data)).max()) if X.nnz else 0.0) < 1e-2, "not log1p of counts"
        X.data = np.round(X.data)
        return X

    if line == "h1":
        names, sums, n, _ = stream_mean_cpm(d / f, lab, M, counts=from_log1p)
    else:
        names, sums, n, _ = stream_mean_cpm(d / f, lab, M, block=10_000, parts=3)
    ci = names.index("ctrl")
    return halves_table(names, sums, n, sums[ci] / n[ci], gmap)
