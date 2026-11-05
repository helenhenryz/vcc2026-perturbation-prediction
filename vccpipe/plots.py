"""The notebook's figures: one function per figure, so a section's cell shows what it plots, not how.

Used in: every section with a figure. Each function draws one figure with the shared palette of `figures` and
returns it; `title` is the headline, the finding the section computed. The functions only draw: every number they
print on a figure is computed from the arguments, never typed in.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import (
    CMAP_DIV,
    CMAP_SEQ,
    GRID,
    INK,
    INK2,
    LIGHT,
    MUTED,
    SEQ_BLUE,
    SURFACE,
    TISSUE_COLOURS,
    TISSUE_OF,
    headline,
    line_colours,
    tissue_colour,
    tissue_legend,
)
from .harness import FOLD_LINES, SCORER_METRICS, WATERFALL
from .inputs import DEPTH
from .labels import FOLD_LABEL, HALF_LINE_LABEL, HALF_LINE_TISSUE, METRIC_LABEL, SCREEN_LABEL, TYPE_LABEL, TYPE_TISSUE
from .phase import JIANG, SOURCE_TYPE

__all__ = [
    "LOSS",
    "pass_fail_table",
    "plot_cell_shape",
    "plot_dead_end",
    "plot_depth_and_cells",
    "plot_fold_map",
    "plot_fold_scores",
    "plot_generator_check",
    "plot_line_map",
    "plot_line_matches",
    "plot_measures",
    "plot_own_line",
    "plot_pair_similarity",
    "plot_predicted_heatmaps",
    "plot_reliability",
    "plot_screen_coverage",
    "plot_strongest_knockdowns",
    "plot_task",
    "plot_toy_scores",
    "plot_transfer",
    "plot_volcanoes",
    "plot_waterfall",
    "style_checks",
]

LOSS = "#d03b3b"  # a loss below the noise band, a failed check


# ---- section 1 ---------------------------------------------------------------------------------------------------


def _box(ax, x, y, w, h, title, body, colour):
    """A rounded box with a bold title and a few lines of text under it."""
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=SURFACE, ec=colour, lw=2
        )
    )
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.82, body, ha="center", va="top", fontsize=9, color=INK2, linespacing=1.5)


def _arrow(ax, x0, x1, y, label):
    """A horizontal arrow from x0 to x1 at height y, with a label above it."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.5})
    ax.text((x0 + x1) / 2, y + 0.18, label, ha="center", fontsize=9, color=INK2)


def plot_task(contexts, n_control, n_genes, n_knockdowns, cells_per_knockdown=400):
    """The task as three boxes: the unseen lines' controls, the knockdowns, the prediction.

    `n_control` is a line's control-cell count (None when the phase does not give it)."""
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12), ax.set_ylim(0, 3.4)
    ax.axis("off")
    if n_control:
        lines_body = f"{n_control:,} control cells each\n{n_genes:,} genes per cell\nwhole-number counts"
    else:
        lines_body = f"control cells\n{n_genes:,} genes per cell"
    _box(ax, 0.1, 0.4, 3.3, 2.6, f"{len(contexts)} unseen cell lines", lines_body, MUTED)
    knockdowns_body = (
        "CRISPRi: one gene switched\ndown at its promoter\n(no perturbed cell of these\nlines is ever seen)"
    )
    _box(ax, 4.35, 0.4, 3.3, 2.6, f"{n_knockdowns} knockdowns", knockdowns_body, MUTED)
    total = len(contexts) * n_knockdowns * cells_per_knockdown
    prediction_body = f"per line and knockdown:\n{cells_per_knockdown} cells x {n_genes:,} genes\nwhole-number counts\n= {total:,} cells"
    _box(ax, 8.6, 0.4, 3.3, 2.6, "the prediction", prediction_body, INK)
    for i, (c, colour) in enumerate(line_colours(contexts).items()):  # the lines' colours, used in every figure
        ax.add_patch(mpl.patches.Circle((1.15 + 0.6 * i, 0.78), 0.17, fc=colour, ec="none"))
        ax.text(1.15 + 0.6 * i, 0.78, c, ha="center", va="center", fontsize=9, fontweight="bold", color="#ffffff")
    _arrow(ax, 3.45, 4.3, 1.7, "")
    _arrow(ax, 7.7, 8.55, 1.7, "predict")
    ax.set_title("The task: from a line's resting cells to its cells after each knockdown")
    return fig


# ---- section 2 ---------------------------------------------------------------------------------------------------


def plot_toy_scores(scores, kinds, title, highlight="right means, wrong shape"):
    """The official scorer's six measures for each toy prediction, one small panel per measure.

    `scores`: {kind: {measure: score}} on the challenge's scale; `highlight` is the kind drawn in ink."""
    fig, axes = plt.subplots(1, 6, figsize=(13, 3.2), sharey=True)
    for ax, m in zip(axes, SCORER_METRICS):
        vals = [scores[k][m] for k in kinds]
        ax.barh(range(len(kinds)), vals, color=[INK if k == highlight else LIGHT for k in kinds], height=0.62)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(1, color=MUTED, lw=0.8)
        ax.set_title(m.upper(), fontsize=10)
        lo, hi = min(0.0, min(vals)), max(1.0, max(vals))
        ax.set_xlim(lo - (0.45 if lo < 0 else 0.05) * (hi - lo), hi + 0.35 * (hi - lo))
        for i, v in enumerate(vals):
            label, align = (f" {v:.2f} ", "left") if v >= 0 else (f"{v:.2f} ", "right")
            ax.text(v, i, label, va="center", ha=align, fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["0", "1"])
    axes[0].set_yticks(range(len(kinds)), kinds, fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    headline(fig, title, "each measure on the challenge's scale: 0 = predicting no change, 1 = a second real sample")
    return fig


def plot_volcanoes(volcano, truly_changed, false_hits, called_level, title):
    """One volcano plot per toy prediction: each gene's log2 fold change against the rank test's -log10 p.

    `volcano`: {kind: (lfc, p)} per gene; `truly_changed`: the genes the knockdown really moves; `called_level`:
    roughly the p-value at which the test calls a gene (a grey line)."""
    fig, axes = plt.subplots(1, len(volcano), figsize=(13, 3.6), sharey=True)
    unchanged = ~truly_changed
    for ax, (kind, (lfc, p)) in zip(axes, volcano.items()):
        y = -np.log10(np.maximum(p, 1e-300))
        ax.scatter(lfc[unchanged], y[unchanged], s=8, color=LIGHT, linewidths=0, label="unchanged genes")
        ax.scatter(lfc[truly_changed], y[truly_changed], s=12, color=INK, linewidths=0, label="truly changed genes")
        ax.axhline(-np.log10(called_level), color=MUTED, lw=0.8)
        ax.set_title(f"{kind}\n{false_hits[kind]} unchanged genes over the line", fontsize=10)
        ax.set_xlabel("log2 fold change")
        ax.set_yscale("symlog", linthresh=10)
    axes[0].set_ylabel("-log10 p (rank test)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    headline(
        fig,
        title,
        "a volcano plot: each dot a gene, its change (x) against how sure the test is (y); the grey line is roughly "
        "where a gene gets called",
    )
    return fig


# ---- section 3 ---------------------------------------------------------------------------------------------------


def plot_line_map(umap_xy, cell_line, contexts, colours, corr, closest):
    """Left: the lines' control cells on a UMAP; right: the correlation of their pooled profiles.

    `cell_line`: each cell's line; `corr`: lines x lines; `closest`: (r, line a, line b) of the closest pair."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for c in contexts:
        m = np.asarray(cell_line == c)
        ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=2, color=colours[c], alpha=0.5, linewidths=0, rasterized=True)
        cx, cy = np.median(umap_xy[m], axis=0)
        box = {"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": colours[c], "lw": 1.5}
        ax.text(cx, cy, f"line {c}", color=INK, fontsize=11, fontweight="bold", ha="center", va="center", bbox=box)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("UMAP 1"), ax.set_ylabel("UMAP 2")
    ax.set_title("Each line's controls form their own island")
    ax = axes[1]
    values = corr.to_numpy()
    lowest = float(values[np.triu_indices(len(contexts), 1)].min())
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    ax.imshow(values, cmap=cmap, vmin=lowest - 0.01, vmax=1.0)
    for i in range(len(contexts)):
        for j in range(len(contexts)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="#ffffff" if v > 0.97 else INK, fontsize=10)
    labels = [f"line {c}" for c in contexts]
    ax.set_xticks(range(len(contexts)), labels), ax.set_yticks(range(len(contexts)), labels)
    r, a, b = closest
    ax.set_title(f"{a} and {b} are the closest pair (r {r:.3f})")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig


def plot_line_matches(matches, names, lineages, colours, title):
    """Each line's 10 best matches in the DepMap catalogue, as bars of Spearman correlation.

    `matches`: {line: Series of rho by catalogue model, best first}; `names`, `lineages`: by catalogue model."""
    fig, axes = plt.subplots(1, len(matches), figsize=(13.5, 3.8), sharex=True)
    for ax, (c, match) in zip(axes, matches.items()):
        top = match.index[:10][::-1]
        ax.barh(range(10), match[top], color=colours[c], height=0.7)
        lineage = {m: lineages[m] if lineages[m] != "?" else "no lineage given" for m in top}
        ax.set_yticks(range(10), [f"{names[m]} ({lineage[m]})" for m in top], fontsize=8)
        ax.set_xlim(0.6, 0.92)
        ax.set_title(f"line {c}: best {match.iloc[0]:.3f}, next {match.iloc[1]:.3f}", fontsize=10)
        ax.set_xlabel("Spearman with the line's controls")
    fig.tight_layout()
    headline(
        fig,
        title,
        "each line's pooled controls ranked against ~1,500 profiled cancer cell lines (DepMap); the 10 best matches",
    )
    return fig


# ---- section 4 ---------------------------------------------------------------------------------------------------


def plot_screen_coverage(covered, screens, title):
    """Which panel knockdowns each public screen measured: one row per screen, coloured by its tissue.

    `covered`: {screen: bool per panel knockdown}; `screens`: the rows, top first."""
    coverage = np.vstack([covered[n] for n in screens]).astype(float)
    kd_order = np.argsort(-coverage.sum(0), kind="stable")  # the most-covered knockdowns first
    fig, ax = plt.subplots(figsize=(12, 5.4))
    rgb = np.ones((len(screens), coverage.shape[1], 3)) * np.array(mpl.colors.to_rgb(SURFACE))
    for r, n in enumerate(screens):  # each screen's row in its tissue's colour
        rgb[r, coverage[r, kd_order] > 0] = mpl.colors.to_rgb(tissue_colour(n))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    labels = [f"{SCREEN_LABEL.get(n, 'pathway screens (6 lines)')}  ({int(covered[n].sum())})" for n in screens]
    ax.set_yticks(range(len(screens)), labels, fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("the final panel's 300 knockdowns, most-covered first")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = {"loc": "lower right", "bbox_to_anchor": (1.0, 0.0), "title_fontsize": 8.5}
    tissue_legend(ax, list(TISSUE_COLOURS), title="tissue of the screen", **legend)
    fig.tight_layout()
    headline(
        fig, title, "each row a public screen (knockdowns of the panel it measured in brackets), coloured by its tissue"
    )
    return fig


def plot_depth_and_cells(screens, covered, cells_per_kd, final_depth, title):
    """Left: molecules per cell of each screen against the final lines; right: cells per measured knockdown.

    `screens`: the screens, in the coverage map's order; `covered`: {screen: bool per panel knockdown};
    `cells_per_kd`: {screen: median cells per measured panel knockdown}; `final_depth`: {final line: median
    molecules per control cell}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    by_depth = sorted(screens, key=lambda n: DEPTH.get(n, 8000))
    depth = [DEPTH.get(n, 8000) for n in by_depth]
    ax.barh(range(len(by_depth)), depth, color=[tissue_colour(n) for n in by_depth], height=0.7)
    lo, hi = min(final_depth.values()), max(final_depth.values())
    ax.axvspan(lo, hi, color=INK, alpha=0.9, lw=0)
    label = f"  the final lines\n  {', '.join(final_depth)}: {lo:,.0f}-{hi:,.0f}"
    ax.text(hi, 1.5, label, fontsize=8.5, color=INK, va="center")
    ax.set_yticks(range(len(by_depth)), [SCREEN_LABEL[n] for n in by_depth], fontsize=8)
    ax.set_xlabel("median molecules (UMI) per cell")
    ax.set_title("Molecules per cell", fontsize=10.5)
    ax = axes[1]
    by_cells = sorted([n for n in screens if covered[n].any()], key=lambda n: cells_per_kd[n])
    cells = [cells_per_kd[n] for n in by_cells]
    ax.barh(range(len(by_cells)), cells, color=[tissue_colour(n) for n in by_cells], height=0.7)
    ax.axvline(400, color=INK, lw=1)
    ax.text(400, -0.9, " 400: the cells we predict per knockdown", color=INK, fontsize=8, va="bottom")
    ax.set_ylim(-1, len(by_cells) - 0.4)
    ax.set_yticks(range(len(by_cells)), [SCREEN_LABEL[n] for n in by_cells], fontsize=8)
    ax.set_xlabel("median cells per measured panel knockdown")
    ax.set_title("Cells per knockdown", fontsize=10.5)
    tissue_legend(ax, list(TISSUE_COLOURS), loc="lower right")
    fig.tight_layout()
    headline(
        fig,
        title,
        "left: how many RNA molecules a cell carries (more = finer detail); right: how many cells stand behind one "
        "knockdown",
    )
    return fig


# ---- section 5 ---------------------------------------------------------------------------------------------------


def plot_reliability(k562_halves, reliability, strong, weak, title):
    """Two K562 knockdowns' half-against-half profiles (the most reliable and a typical one), and every screen's
    reliability as a box.

    `k562_halves`: K562's halves file; `reliability`: {halves file: reliability per knockdown}; `strong`, `weak`:
    the two knockdowns' rows."""
    k562_rel = reliability["bench_k562"]
    well = k562_halves["ctrl_cpm"][0] > 20
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for ax, i, what in ((axes[0], strong, "The most reliable"), (axes[1], weak, "A typical")):
        a, b = k562_halves["lfc_a"][i, well], k562_halves["lfc_b"][i, well]
        ax.scatter(a, b, s=3, color=tissue_colour("bench_k562"), alpha=0.35, linewidths=0, rasterized=True)
        lim = float(np.nanpercentile(np.abs(np.r_[a, b]), 99.9))
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("log2 FC, half A"), ax.set_ylabel("log2 FC, half B")
        knockdown = k562_halves["target"][i]
        ax.set_title(f"{what} K562 knockdown ({knockdown})\nreliability {k562_rel[i]:.2f}", fontsize=10)
    ax = axes[2]
    order = sorted(reliability, key=lambda n: np.nanmedian(reliability[n]))
    boxes = [reliability[n][np.isfinite(reliability[n])] for n in order]
    medianprops = {"color": INK, "lw": 2}
    bp = ax.boxplot(
        boxes, vert=False, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True, medianprops=medianprops
    )
    for patch, n in zip(bp["boxes"], order):
        patch.set(facecolor=tissue_colour(n), alpha=0.85, edgecolor="none")
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set(color=MUTED, lw=1)
    for y, n in enumerate(order, start=1):
        ax.text(1.02, y, f"{np.nanmedian(reliability[n]):.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(1, len(order) + 1), [HALF_LINE_LABEL[n] for n in order])
    ax.set_xlim(-0.2, 1.1)
    ax.set_xlabel("split-half reliability of one knockdown's profile (box: middle half; line: median)")
    ax.set_title("Reliability per screen (median at right)", fontsize=10)
    tissue_legend(ax, sorted({TISSUE_OF[n] for n in order}), loc="upper left", fontsize=8)
    fig.tight_layout()
    headline(
        fig,
        title,
        "split one knockdown's cells in two halves; if both halves agree (left), the profile is signal; if not "
        "(middle), noise",
    )
    return fig


# ---- section 6 ---------------------------------------------------------------------------------------------------


def plot_transfer(transfer, lines, own_vs_average, within, across, title):
    """Left: the noise-corrected transfer between every pair of lines; right: how often a line's average response
    beats its profile of the same gene, as a guess for another line.

    `transfer`: lines x lines (NaN on the diagonal); `own_vs_average`: {(a, b): (share, knockdowns)}; `within`,
    `across`: the medians within a tissue and across tissues."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    n = len(lines)
    im = ax.imshow(np.where(np.eye(n, dtype=bool), np.nan, transfer), cmap=CMAP_SEQ, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(transfer[i, j]):
                colour = "#ffffff" if transfer[i, j] > 0.55 else INK
                ax.text(j, i, f"{transfer[i, j]:.2f}", ha="center", va="center", fontsize=9, color=colour)
    labels = [f"{HALF_LINE_LABEL[line]}\n{HALF_LINE_TISSUE[line]}" for line in lines]
    ax.set_xticks(range(n), labels, fontsize=8, rotation=45, ha="right"), ax.set_yticks(range(n), labels, fontsize=8)
    ax.spines[:].set_visible(False)
    subtitle = f"(median {within:.2f} within a tissue, {across:.2f} across)"
    ax.set_title(f"Same knockdown, two lines: noise-corrected r\n{subtitle}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="median corrected r")
    ax = axes[1]
    tissue = HALF_LINE_TISSUE
    pairs = sorted(own_vs_average, key=lambda k: (tissue[k[0]] == tissue[k[1]], own_vs_average[k][0]))
    shares = [own_vs_average[k][0] for k in pairs]
    colours = [INK if tissue[a] == tissue[b] else LIGHT for a, b in pairs]
    ax.barh(range(len(pairs)), shares, color=colours, height=0.72)
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.text(0.5, len(pairs) - 0.2, " half", fontsize=8, color=INK2, va="bottom")
    ax.set_ylim(-0.6, len(pairs) + 0.6)
    pair_labels = [f"{HALF_LINE_LABEL[a]} -> {HALF_LINE_LABEL[b]}" for a, b in pairs]
    ax.set_yticks(range(len(pairs)), pair_labels, fontsize=7)
    ax.set_xlabel(
        "share of knockdowns where line A's average response beats\nits profile of the same gene, as a guess for line B"
    )
    handles = [
        mpl.patches.Patch(color=INK, label="same tissue"),
        mpl.patches.Patch(color=LIGHT, label="different tissues"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Borrow the gene's own profile, or the line's average response?", fontsize=10)
    fig.tight_layout()
    headline(
        fig,
        title,
        "a corrected r of 1 would mean the same knockdown does exactly the same thing in both lines, once each "
        "screen's noise is removed",
    )
    return fig


# ---- section 7 ---------------------------------------------------------------------------------------------------


def plot_dead_end(target_cpm, forms, title, well_expressed=300):
    """The covariance test's correlations per knocked-down gene, one panel per form of the test.

    `target_cpm`: each tested gene's CPM in the controls; `forms`: {form: {"planted", "real", "random": r per
    gene}}; genes above `well_expressed` CPM are shaded and summarised in each panel's title."""
    hi = target_cpm > well_expressed
    x = np.log10(target_cpm)
    series = (
        ("random", "a random gene's covariance (negative control)", LIGHT, 9),
        ("planted", "planted fluctuation (positive control)", INK, 11),
        ("real", "real K562 knockdown (noise-corrected)", tissue_colour("bench_k562"), 11),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, (what, r) in zip(axes, forms.items()):
        for key, name, colour, size in series:
            ax.scatter(x, r[key], s=size, color=colour, alpha=0.8, linewidths=0, label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvspan(np.log10(well_expressed), x.max() + 0.1, color=GRID, alpha=0.5, lw=0, zorder=0)
        shade = ax.get_xaxis_transform()
        ax.text(np.log10(well_expressed), 0.98, "  above 300 CPM", transform=shade, fontsize=8, color=INK2, va="top")
        ax.set_xlabel("knocked-down gene's expression in the controls, log10 CPM")
        medians = {key: np.median(r[key][hi]) for key in ("planted", "real", "random")}
        summary = f"planted r {medians['planted']:.2f}, real {medians['real']:.2f}, random {medians['random']:.2f}"
        ax.set_title(f"{what}\nabove 300 CPM: {summary}", fontsize=10)
    axes[0].set_ylabel("correlation with the knockdown's profile")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    headline(
        fig,
        title,
        "each dot one gene X (above 300 CPM shaded); right: removing the controls' 10 strongest shared axes erases the "
        "planted signal too, so that form tests nothing",
    )
    return fig


# ---- section 8 ---------------------------------------------------------------------------------------------------


def _held_square(ax, k, r, cell_type, held):
    """One fold x cell-type square: the tissue's colour when trained on, an empty crossed square when held out."""
    fill = SURFACE if held else TISSUE_COLOURS[TYPE_TISSUE[cell_type]]
    edge = LIGHT if held else "none"
    shape = {"boxstyle": "round,pad=0,rounding_size=0.08", "fc": fill, "ec": edge, "lw": 1}
    ax.add_patch(mpl.patches.FancyBboxPatch((k + 0.08, r + 0.12), 0.84, 0.76, **shape))
    if held:
        ax.plot([k + 0.3, k + 0.7], [r + 0.3, r + 0.7], color=MUTED, lw=1.2)
        ax.plot([k + 0.3, k + 0.7], [r + 0.7, r + 0.3], color=MUTED, lw=1.2)


def _part_held_square(ax, k, r, share_held):
    """The pathway screens' square in a fold that leaves out only its own part: filled for the share kept."""
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (k + 0.08, r + 0.12), 0.84, 0.76, fc=SURFACE, ec=LIGHT, lw=1, boxstyle="round,pad=0,rounding_size=0.08"
        )
    )
    kept = 0.84 * (1 - share_held)
    ax.add_patch(mpl.patches.Rectangle((k + 0.08, r + 0.12), kept, 0.76, fc=TISSUE_COLOURS["other"], ec="none"))


def plot_fold_map(fold_lines, held_types, title=None):
    """The six folds as a grid: one row per held-out line, one column per cell type of the training screens; a
    crossed square is left out of training in that fold.

    `held_types`: {fold line: the cell types it holds out}."""
    types = list(TYPE_LABEL)  # the columns
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    for r, line in enumerate(fold_lines):
        for k, cell_type in enumerate(types):
            held = cell_type in held_types[line]
            if cell_type == "pathway":  # the pathway screens span six lines: a fold leaves out its own type's part
                part = [SOURCE_TYPE[n] in held_types[line] for n in JIANG]
                held = all(part)
                if any(part) and not held:
                    _part_held_square(ax, k, r, np.mean(part))
                    continue
            _held_square(ax, k, r, cell_type, held)
    ax.set_xlim(0, len(types)), ax.set_ylim(len(fold_lines), 0)
    ax.set_xticks(np.arange(len(types)) + 0.5, [TYPE_LABEL[t] for t in types], fontsize=8.5)
    rows = [f"held out: {FOLD_LABEL[line]}" for line in fold_lines]
    ax.set_yticks(np.arange(len(fold_lines)) + 0.5, rows, fontsize=9)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.xaxis.tick_top()
    legend = {"loc": "upper left", "bbox_to_anchor": (1.0, 1.0), "title_fontsize": 8.5}
    tissue_legend(ax, list(TISSUE_COLOURS), title="trained on (tissue)", **legend)
    fig.tight_layout()
    headline(
        fig,
        title or "Six folds: each training line is held out in turn, and every screen of its own cell type is removed",
        "columns: the public screens by cell type; a crossed square is left out of training for that fold (the iPSC "
        "atlas in every fold; a part-filled square: only the line's own part of the pathway screens is left out)",
    )
    return fig


def plot_fold_scores(final, consensus, seed_noise, title):
    """Each fold's overall score: no change (0), the borrowed consensus and the final pipeline, with the seed noise
    as an error bar.

    `final`, `consensus`: DataFrames by fold with an "overall" column; `seed_noise`: by fold."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    order = final["overall"].sort_values().index.tolist()
    for y, line in enumerate(order):
        ours, theirs = final.loc[line, "overall"], consensus.loc[line, "overall"]
        ax.plot([theirs, ours], [y, y], color=GRID, lw=3, zorder=1)
        ax.errorbar(ours, y, xerr=seed_noise[line], color=INK2, lw=1, capsize=3, zorder=2)
        ax.scatter(0, y, s=60, color=SURFACE, edgecolors=MUTED, lw=1.2, zorder=3)
        ax.scatter(theirs, y, s=60, color=LIGHT, zorder=3)
        ax.scatter(ours, y, s=70, color=tissue_colour(line), edgecolors=INK, lw=0.8, zorder=4)
    ax.set_yticks(range(len(order)), [FOLD_LABEL[line] for line in order])
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_xlabel(
        "overall score on the held-out line (mean of the six measures; 0 = no change, 1 = a second real sample)"
    )
    dot = {"ls": "", "marker": "o", "ms": 8}
    handles = [
        mpl.lines.Line2D([], [], mfc=SURFACE, mec=MUTED, label="predict no change", **dot),
        mpl.lines.Line2D([], [], color=LIGHT, label="borrowed consensus", **dot),
        mpl.lines.Line2D([], [], mfc=INK2, mec=INK, label="final pipeline (colour: tissue; bar: seed noise)", **dot),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3, fontsize=8.5)
    fig.tight_layout()
    headline(
        fig,
        title,
        "each row one held-out training line; the seed bar is how much the score moves when only the random cells "
        "change",
    )
    return fig


# ---- section 9 ---------------------------------------------------------------------------------------------------

_GENERATOR_COLOUR = {"real": INK2, "naive generator": LIGHT, "our generator": INK}


def plot_cell_shape(example_counts, called, gene, knockdown, found, false, n_real_calls, title):
    """Left: one gene's counts per cell in one knockdown, real and from each generator, and whether the rank test
    calls it; right: the real calls each generator recovers and the false calls it makes.

    `example_counts`: {"real", "naive generator", "our generator": counts per cell}; `called`: {same keys: bool};
    `found`, `false`: [naive, ours] call counts; `n_real_calls`: the calls on the real cells."""
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1, 1], hspace=0.55, wspace=0.45)
    top = float(np.percentile(np.concatenate(list(example_counts.values())), 99.5))
    bins = np.arange(0, top + 2) - 0.5 if top < 60 else np.linspace(0, top, 40)
    for k, (name, v) in enumerate(example_counts.items()):
        ax = fig.add_subplot(gs[k, 0])
        ax.hist(np.minimum(v, bins[-1]), bins=bins, color=_GENERATOR_COLOUR[name], rwidth=0.9)
        ax.axvline(v.mean(), color=INK, lw=1)
        verdict = "CALLED" if called[name] else "not called"
        style = {"fontsize": 8.5, "loc": "left", "fontweight": "bold" if called[name] else "normal", "color": INK}
        ax.set_title(f"{name}: mean {v.mean():.1f}, sd {v.std():.1f}, {verdict} by the test", pad=3, **style)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if k < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"{gene} counts per cell, after knocking down {knockdown}")
    names = ["naive generator", "our generator"]
    colours = [_GENERATOR_COLOUR[n] for n in names]
    ax = fig.add_subplot(gs[:, 1])
    ax.barh(range(2), [n_real_calls] * 2, color=GRID, height=0.5)
    ax.barh(range(2), found, color=colours, height=0.5)
    for y, v in enumerate(found):
        ax.text(n_real_calls, y, f" {v:,} of {n_real_calls:,}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(2), names), ax.invert_yaxis()
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, n_real_calls * 1.6)
    ax.set_xlabel("genes the real test calls that it also calls")
    ax.set_title("Real calls recovered", fontsize=10)
    ax = fig.add_subplot(gs[:, 2])
    ax.barh(range(2), false, color=colours, height=0.5)
    for y, v in enumerate(false):
        ax.text(v, y, f" {v:,}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(2), ["", ""]), ax.invert_yaxis()
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, max(false) * 1.3)
    ax.set_xlabel("genes it calls that the real test does not")
    ax.set_title("False calls", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.14, top=0.86)
    headline(
        fig,
        title,
        "left: one gene's counts per cell in one knockdown, real and generated (line: the mean); the rank test reads the "
        "whole shape, not only the mean. Right: the calls over all 20 knockdowns",
    )
    return fig


# ---- section 10 --------------------------------------------------------------------------------------------------


def plot_waterfall(mean_by_step, gains, band, consensus_mean, title):
    """The waterfall: the bare recipe's score, each piece's gain on top of the ones before it, and the final score,
    with the seed-noise band around each step.

    `mean_by_step`: the mean overall score after each step (bare recipe first); `gains`: each piece's change, by
    arm name ("+piece"); `band`: the noise band's half-width; `consensus_mean`: the borrowed consensus's score."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    # the levels are drawn as short lines (not bars from zero), so the axis can zoom in on the gains
    start, end = mean_by_step.iloc[0], mean_by_step.iloc[-1]
    ax.plot([-0.35, 0.35], [start] * 2, color=INK2, lw=3, solid_capstyle="butt")
    ax.text(0, start, f"{start:.3f}\n", ha="center", va="bottom", fontsize=8.5, color=INK2)
    for k, g in enumerate(gains, start=1):
        lo = mean_by_step.iloc[k - 1]
        ax.add_patch(mpl.patches.Rectangle((k - 0.5, lo - band), 1.0, 2 * band, color=GRID, lw=0, zorder=0))
        colour = INK if g > band else (LIGHT if g >= -band else LOSS)
        ax.bar(k, g, bottom=lo, color=colour, width=0.6, zorder=2)
        ax.text(k, max(lo, lo + g), f"{g:+.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    k_end = len(gains) + 1
    ax.plot([k_end - 0.35, k_end + 0.35], [end] * 2, color=INK, lw=3, solid_capstyle="butt")
    ax.text(k_end, end, f"{end:.3f}\n", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(consensus_mean, color=MUTED, lw=0.8, zorder=1)
    ax.text(k_end + 0.45, consensus_mean, " borrowed\n consensus", fontsize=8, color=INK2, va="center")
    steps = ["bare recipe"] + [arm[1:] for arm in gains.index] + ["final"]
    ax.set_xticks(range(k_end + 1), steps, rotation=35, ha="right")
    ax.set_xlim(-0.6, k_end + 1.4)
    lo = min(mean_by_step.min(), consensus_mean) - band
    hi = max(mean_by_step.max(), consensus_mean) + band
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("overall score, mean of six held-out lines")
    handles = [
        mpl.patches.Patch(color=INK, label="gain above the noise band"),
        mpl.patches.Patch(color=LIGHT, label="inside the noise band"),
        mpl.patches.Patch(color=LOSS, label="loss below the noise band"),
        mpl.patches.Patch(color=GRID, label=f"noise band (+-{band:.4f}, seed change)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    headline(
        fig,
        title,
        "each bar one piece switched on, on top of the ones to its left; the grey box around its start is the seed "
        "noise; the axis is zoomed (it does not start at 0)",
    )
    return fig


def plot_measures(fold_scores, steps, title):
    """The six measures as small multiples over the waterfall's steps: each fold thin, their mean thick, the borrowed
    consensus as a grey line.

    `fold_scores`: {fold line: {arm: {measure: score}}}; `steps`: the arms in order, bare recipe first."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    x = np.arange(len(steps))
    for ax, m in zip(axes.ravel(), SCORER_METRICS):
        per = pd.DataFrame({line: {a: fold_scores[line][a][m] for a in steps} for line in FOLD_LINES})
        for line in FOLD_LINES:
            ax.plot(x, per[line], color=tissue_colour(line), lw=0.9, alpha=0.55)
        ax.plot(x, per.mean(axis=1), color=INK, lw=2.2)
        ax.axhline(float(np.mean([fold_scores[line]["probe"][m] for line in FOLD_LINES])), color=MUTED, lw=0.8)
        first, last = per.mean(axis=1).iloc[0], per.mean(axis=1).iloc[-1]
        ax.set_title(f"{METRIC_LABEL[m]}\n{first:.3f} -> {last:.3f} ({last - first:+.3f})", fontsize=9.5)
        ax.axhline(0, color=GRID, lw=0.8)
    for ax in axes[-1]:
        ax.set_xticks(x, ["bare"] + WATERFALL, rotation=60, ha="right", fontsize=8)
    handles = [
        mpl.lines.Line2D([], [], color=INK, lw=2.2, label="mean of six lines"),
        mpl.lines.Line2D([], [], color=MUTED, lw=0.8, label="borrowed consensus (mean)"),
    ]
    for t in sorted({TISSUE_OF[line] for line in FOLD_LINES}):
        handles.append(mpl.lines.Line2D([], [], color=TISSUE_COLOURS[t], lw=1, label=f"one fold ({t})"))
    axes[0, 0].legend(handles=handles, fontsize=7.5, loc="best")
    fig.tight_layout()
    headline(
        fig,
        title,
        "each panel one measure on the challenge's scale, as the pieces are switched on from left to right",
    )
    return fig


# ---- section 11 --------------------------------------------------------------------------------------------------


def plot_generator_check(moments, keep, library_gap, false_calls, title):
    """Generated against real cells, gene by gene: the spread, how often a gene is detected, and each knockdown's
    counts per cell.

    `moments`: {"real", "naive generator", "our generator": (mean, variance, detected share) per gene and
    knockdown}; `keep`: the pairs compared; `library_gap`: {generator: log2 gap of median counts per cell, per
    knockdown}; `false_calls`: section 9's (naive, ours) false calls, for the caption."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _, real_var, real_det = moments["real"]
    for ax, what in zip(axes, ("var", "det")):
        for name in ("naive generator", "our generator"):
            _, var, det = moments[name]
            if what == "var":
                x = np.log2((var[keep] + 1e-3) / (real_var[keep] + 1e-3))
                edges = np.linspace(-1.5, 1.5, 61)
                label = f"{name}: within x1.4 for {np.mean(np.abs(x) < 0.5):.0%}"
            else:
                x = det[keep] - real_det[keep]
                edges = np.linspace(-0.15, 0.15, 61)
                label = f"{name}: median |gap| {np.median(np.abs(x)):.3f}"
            counts, _ = np.histogram(x, bins=edges)  # values outside the range are left out of the drawing only
            centres = (edges[:-1] + edges[1:]) / 2
            if name == "naive generator":
                ax.fill_between(centres, counts, step="mid", color=LIGHT, lw=0, label=label)
            else:
                ax.step(centres, counts, where="mid", color=INK, lw=1.6, label=label)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_xlabel("log2 (generated / real variance)\nper gene and knockdown; 0 = equal")
    axes[0].set_title("Cell-to-cell spread", fontsize=10)
    axes[1].set_xlabel("share of generated cells detecting the gene\nminus the real share; 0 = equal")
    axes[1].set_title("How often a gene is seen at all", fontsize=10)
    ax = axes[2]  # per cell: the total count (library size) of generated against real cells, per knockdown
    for y, name in enumerate(("naive generator", "our generator")):
        gap = library_gap[name]
        jitter = np.random.default_rng(y).uniform(-0.15, 0.15, len(gap))
        colour = {"naive generator": LIGHT, "our generator": INK}[name]
        ax.scatter(gap, np.full(len(gap), y) + jitter, s=16, color=colour, linewidths=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(2), ["naive", "ours"]), ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("log2 (generated / real median counts per cell),\none dot per knockdown")
    ax.set_title("Counts per cell", fontsize=10)
    fig.tight_layout()
    naive, ours = false_calls
    headline(
        fig,
        title,
        f"the same 20 knockdowns and true means as section 9 ({naive:,} false calls from the naive cells, {ours:,} "
        "from ours); left and middle: one count per gene and knockdown; right: one dot per knockdown",
    )
    return fig


# ---- section 13 --------------------------------------------------------------------------------------------------


def _pass_fail_colour(value):
    """The CSS of a check table's pass cell: bold green for PASS, bold red for FAIL."""
    if value == "PASS":
        return "color: #006300; font-weight: bold"
    return f"color: {LOSS}; font-weight: bold" if value == "FAIL" else ""


def pass_fail_table(rows):
    """A check table from (check, measured, passed) rows: columns check, measured and pass ("PASS" or "FAIL")."""
    table = pd.DataFrame(rows, columns=["check", "measured", "pass"])
    table["pass"] = table["pass"].map({True: "PASS", False: "FAIL"})
    return table


def style_checks(checks):
    """The check table (columns check, measured, pass) for display: no index, the pass column coloured."""
    return checks.style.hide(axis="index").map(_pass_fail_colour, subset=["pass"])


def plot_predicted_heatmaps(shown, colours, knockdowns, genes, limit, title):
    """The same knockdowns x genes block of predicted change in every line, side by side on one colour scale.

    `shown`: {line: knockdowns x genes, already ordered}; `knockdowns`, `genes`: the row and column labels;
    `limit`: the colour scale's end (log2)."""
    fig, axes = plt.subplots(1, len(shown), figsize=(14, 6.4), sharey=True)
    for ax, (c, block) in zip(axes, shown.items()):
        im = ax.imshow(block, aspect="auto", cmap=CMAP_DIV, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=6.5)
        chip = {"fc": colours[c], "ec": "none", "boxstyle": "round,pad=0.3"}
        ax.set_title(f"line {c}", fontsize=11, color="#ffffff", loc="center", bbox=chip)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(knockdowns)), knockdowns, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="predicted change, log2 (blue down, red up)")
    headline(
        fig,
        title,
        "rows: the knockdowns with the largest predicted effect on average; columns: the genes they move most; "
        "both clustered so similar rows and columns sit together",
    )
    return fig


def plot_pair_similarity(pair_r, most_alike, colours, title):
    """For each pair of lines, the histogram of each knockdown's correlation of predicted change in the two lines.

    `pair_r`: {(a, b): r per knockdown}; `most_alike`: the pair drawn in ink."""
    fig, axes = plt.subplots(len(pair_r), 1, figsize=(9, 4.6), sharex=True)
    edges = np.linspace(-0.2, 1, 49)
    for ax, ((a, b), r) in zip(axes, pair_r.items()):
        ax.hist(r[np.isfinite(r)], bins=edges, color=INK if (a, b) == most_alike else LIGHT, rwidth=0.9)
        ax.axvline(np.nanmedian(r), color=INK2, lw=1)
        for k, c in enumerate((a, b)):  # the two lines, as their colour chips
            ax.plot(0.015 + 0.03 * k, 0.6, "o", ms=9, color=colours[c], transform=ax.transAxes, clip_on=False)
        label = f"{a} vs {b}: median {np.nanmedian(r):.2f}"
        ax.text(0.085, 0.6, label, transform=ax.transAxes, va="center", fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("correlation of one knockdown's predicted change in two lines (all 300 knockdowns)")
    fig.tight_layout()
    headline(
        fig,
        title,
        "for each knockdown, the correlation of its predicted change in two lines; one panel per pair of lines",
    )
    return fig


def plot_strongest_knockdowns(effect_size, knockdowns, target_cpm, colours, title):
    """Each line's 10 knockdowns with the largest predicted effect, with the target gene's own CPM beside each bar.

    `effect_size`, `target_cpm`: {line: one value per knockdown, in the order of `knockdowns`}."""
    fig, axes = plt.subplots(1, len(effect_size), figsize=(13, 3.6), sharex=True)
    for ax, (c, size) in zip(axes, effect_size.items()):
        top = np.argsort(-size)[:10][::-1]
        ax.barh(range(10), size[top], color=colours[c], height=0.7)
        ax.set_yticks(range(10), [knockdowns[k] for k in top], fontsize=8.5)
        for y, k in enumerate(top):
            ax.text(size[k], y, f" target {target_cpm[c][k]:,.0f} CPM", va="center", fontsize=7.5, color=INK2)
        ax.set_title(f"line {c}", fontsize=10.5)
        ax.set_xlabel("predicted effect size")
    for ax in axes:
        ax.set_xlim(0, max(size.max() for size in effect_size.values()) * 1.45)
    fig.tight_layout()
    headline(
        fig,
        title,
        "beside each bar: the knocked-down gene's own expression in that line's controls (counts per million)",
    )
    return fig


# ---- section 15 --------------------------------------------------------------------------------------------------


def plot_own_line(measured, predicted, knockdowns, genes, limit, spread_ratio, title):
    """The held-out line's measured and predicted change side by side, on one colour scale.

    `measured`, `predicted`: knockdowns x genes, already ordered; `spread_ratio`: the predicted changes' spread over
    the measured one, for the caption."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, block, what in ((axes[0], measured, "measured"), (axes[1], predicted, "predicted")):
        im = ax.imshow(block, aspect="auto", cmap=CMAP_DIV, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=6.5)
        ax.set_title(f"{what}", fontsize=10.5)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(knockdowns)), knockdowns, fontsize=7.5)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="log2 fold change")
    headline(
        fig,
        title,
        f"rows: the 20 knockdowns; columns: the 40 genes they move most in the real cells; same colour scale; the "
        f"predicted changes' spread is {spread_ratio:.2f} x the measured one (median over knockdowns)",
    )
    return fig
