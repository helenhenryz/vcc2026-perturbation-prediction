"""The shared palette and the helpers the figures use."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .phase import GeneMap, depmap_on_axis

__all__ = [
    "CMAP_DIV",
    "CMAP_SEQ",
    "DIVERGING",
    "GRID",
    "INK",
    "INK2",
    "LIGHT",
    "MUTED",
    "SEQ_BLUE",
    "SERIES",
    "SURFACE",
    "TISSUE_COLOURS",
    "TISSUE_OF",
    "depmap_profiles",
    "group_sums",
    "headline",
    "line_colours",
    "match_lines",
    "tissue_colour",
    "tissue_legend",
]

# ---- figures ---------------------------------------------------------------------------------------------
# Used in: every figure.
# Used by every section's figures. One palette, set once here and checked colour-blind safe as a set: each colour
# has one meaning in the whole notebook.
# - The final lines D, E, F: blue, orange, aqua, in every figure.
# - The training screens: by tissue (blood violet, pluripotent stem cells green, liver yellow, epithelial magenta,
#   other grey), so a reader can see which kind of cell a number comes from.
# - Anything else (our method against an alternative, a check against its control): near-black for the thing we
#   built or the finding, grey for the comparison. Changes: blue = down, red = up, grey = no change.
# Light background, thin marks, recessive axes; every title states the finding.

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LIGHT = "#c3c2b7"  # the faintest mark (background points, the comparison's fill)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TISSUE_COLOURS = {
    "blood": "#6250d6",
    "pluripotent": "#008300",
    "liver": "#eda100",
    "epithelial": "#e87ba4",
    "other": MUTED,
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#256abf", "#f0efec", "#e34948")
mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": LIGHT,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "figure.dpi": 110,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "image.cmap": "Blues",
    }
)
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list("div", DIVERGING)
CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ_BLUE)


def headline(fig, text: str, sub: str | None = None):
    """A figure's title: the finding, left-aligned over all its panels, placed above them after the layout (call it
    after tight_layout); `sub` a smaller line under it, saying how to read the figure."""
    from matplotlib.transforms import offset_copy

    y = max(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1 for ax in fig.axes)
    if sub:
        fig.text(
            0.01,
            y,
            sub,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=INK2,
            transform=offset_copy(fig.transFigure, fig=fig, y=4, units="points"),
        )
    fig.text(
        0.01,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        transform=offset_copy(fig.transFigure, fig=fig, y=20 if sub else 6, units="points"),
    )


def line_colours(contexts):
    """Each final line's colour, in the order given: {context: colour}."""
    return {c: SERIES[i] for i, c in enumerate(contexts)}


# The tissue of every training screen (by table name) and of every held-out training line (by fold name).
TISSUE_OF = {
    "bench_k562": "blood",
    "k562gwps_bulk": "blood",
    "viperturb": "blood",
    "bench_jurkat": "blood",
    "cd4_Rest": "blood",
    "cd4_Stim8hr": "blood",
    "cd4_Stim48hr": "blood",
    "bench_hepg2": "liver",
    "bench_rpe1": "epithelial",
    "orion_hct116": "epithelial",
    "orion_hek293t": "epithelial",
    "dld1_gx": "epithelial",
    "a549_ko": "epithelial",
    "a549_myllia": "epithelial",
    "bench_h1": "pluripotent",
    "arc_h1": "pluripotent",
    "hesc_gs": "pluripotent",
    "kolf": "pluripotent",
    "h1train": "pluripotent",
    "a375_perturbfate": "other",
    "jiang": "other",
    "h1": "pluripotent",
    "k562gwps": "blood",
    "k562": "blood",
    "jurkat": "blood",
    "rpe1": "epithelial",
    "hepg2": "liver",
}


def tissue_colour(name: str) -> str:
    """The colour of a training screen or held-out line `name`, by its tissue (grey when not listed)."""
    return TISSUE_COLOURS[TISSUE_OF.get(name, "other")]


def tissue_legend(ax, tissues=None, **kw):
    """A legend of the tissue colours (only those in `tissues`, if given)."""
    tissues = tissues or [t for t in TISSUE_COLOURS if t != "other"]
    handles = [mpl.patches.Patch(color=TISSUE_COLOURS[t], label=t) for t in tissues]
    return ax.legend(handles=handles, **({"loc": "lower right", "fontsize": 8.5} | kw))


def depmap_profiles(expr_csv: Path, models_csv: Path, hgnc_txt: Path, genes: list):
    """DepMap's expression (log2 TPM + 1) of ~1,500 cell lines on the challenge's gene axis, with each line's
    name and lineage."""
    df = depmap_on_axis(expr_csv, GeneMap(genes, hgnc_txt))
    models = pd.read_csv(models_csv, index_col=0).reindex(df.index)
    return df, models["CellLineName"].fillna("?"), models["OncotreeLineage"].fillna("?")


def match_lines(ctrl_sum: np.ndarray, genes: list, dm: pd.DataFrame, n_top: int = 2000) -> pd.Series:
    """Spearman of a line's pooled controls (log2 CPM + 1) with every DepMap line, over the 2,000 genes that vary
    most across DepMap."""
    from scipy.stats import rankdata

    prof = pd.Series(np.log2(ctrl_sum / ctrl_sum.sum() * 1e6 + 1), index=genes)
    shared = dm.columns.intersection(prof.index)
    top = dm[shared].std(0).sort_values(ascending=False).index[:n_top]
    q = rankdata(prof[top].to_numpy())
    M = np.apply_along_axis(rankdata, 1, dm[top].to_numpy())
    q = (q - q.mean()) / q.std()
    M = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    return pd.Series(M @ q / len(q), index=dm.index).sort_values(ascending=False)


def group_sums(
    h5ad: Path, pert_col: str = "target_gene", ctx_col: str = "context", step: int = 20000, stats: dict | None = None
):
    """Each (context, knockdown) group's summed counts, read from the file in blocks of cells (never the whole
    matrix). Returns (sorted keys, sums (groups x genes), genes). `stats`, if given, is filled on the same pass
    with what the submission rules check: cells per group, the smallest count, whether every count is whole,
    the largest total of one cell and the stored entries."""
    import h5py

    with h5py.File(h5ad, "r") as f:
        X, obs, var = f["X"], f["obs"], f["var"]
        from anndata.io import read_elem

        genes = read_elem(var).index.astype(str)
        indptr = X["indptr"][:]
        ng = int(X.attrs["shape"][1])

        def col(name):
            """The categorical obs column `name`, decoded to one string per cell."""
            g = obs[name]
            return g["categories"][:].astype(str)[g["codes"][:]]

        tg, cx = col(pert_col), col(ctx_col)
        keys = sorted(set(zip(cx, tg)))
        kid = {k: i for i, k in enumerate(keys)}
        gid = np.array([kid[k] for k in zip(cx, tg)])
        tot = np.zeros((len(keys), ng))
        for a in range(0, len(tg), step):
            b = min(len(tg), a + step)
            blk = sp.csr_matrix(
                (X["data"][indptr[a] : indptr[b]], X["indices"][indptr[a] : indptr[b]], indptr[a : b + 1] - indptr[a]),
                shape=(b - a, ng),
            )
            M = sp.csr_matrix((np.ones(b - a), (gid[a:b], np.arange(b - a))), shape=(len(keys), b - a))
            tot += (M @ blk).toarray()
            if stats is not None:
                d = blk.data
                stats["min_count"] = min(stats.get("min_count", np.inf), float(d.min()) if d.size else 0.0)
                stats["whole"] = stats.get("whole", True) and bool(np.array_equal(d, np.round(d)))
                stats["max_cell_total"] = max(stats.get("max_cell_total", 0.0), float(np.asarray(blk.sum(1)).max()))
        if stats is not None:
            stats["cells_per_group"] = dict(zip(keys, np.bincount(gid, minlength=len(keys)).tolist()))
            stats["stored_entries"] = int(indptr[-1])
            stats["dtype"] = str(X["data"].dtype)
    return keys, tot, list(genes)
