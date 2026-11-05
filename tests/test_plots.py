"""Smoke tests of the notebook's figures: every plot function draws its figure from toy inputs (Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from vccpipe import (
    FOLD_LABEL,
    FOLD_LINES,
    HALF_LINE_LABEL,
    HALF_LINE_TABLE,
    HALF_LINE_TISSUE,
    HELD_TYPES,
    METRIC_LABEL,
    SCORER_METRICS,
    TYPE_LABEL,
    TYPE_TISSUE,
    WATERFALL,
    line_colours,
    pass_fail_table,
    plot_cell_shape,
    plot_dead_end,
    plot_depth_and_cells,
    plot_fold_map,
    plot_fold_scores,
    plot_generator_check,
    plot_line_map,
    plot_line_matches,
    plot_measures,
    plot_own_line,
    plot_pair_similarity,
    plot_predicted_heatmaps,
    plot_reliability,
    plot_screen_coverage,
    plot_strongest_knockdowns,
    plot_task,
    plot_toy_scores,
    plot_transfer,
    plot_volcanoes,
    plot_waterfall,
    style_checks,
)

LINES = ["D", "E", "F"]
COLOURS = line_colours(LINES)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def drawn(fig):
    """The figure, rendered once (so layout and text code runs), is a matplotlib Figure."""
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    return True


def test_the_label_tables_agree():
    assert set(TYPE_LABEL) == set(TYPE_TISSUE)
    assert set(HALF_LINE_LABEL) == set(HALF_LINE_TABLE) == set(HALF_LINE_TISSUE)
    assert set(FOLD_LABEL) == set(FOLD_LINES)
    assert set(METRIC_LABEL) == set(SCORER_METRICS)


def test_section_1_and_2(rng):
    assert drawn(plot_task(LINES, 18_000, 1_000, 300))
    assert drawn(plot_task(LINES, None, 1_000, 300))
    kinds = ["replicate", "right means, wrong shape", "noisy", "no effect"]
    scores = {k: {m: float(rng.normal(0.5, 0.5)) for m in SCORER_METRICS} for k in kinds}
    assert drawn(plot_toy_scores(scores, kinds, "a finding"))
    truly = np.zeros(50, bool)
    truly[:5] = True
    volcano = {k: (rng.normal(0, 1, 50), rng.uniform(0, 1, 50)) for k in kinds}
    assert drawn(plot_volcanoes(volcano, truly, {k: 3 for k in kinds}, 0.05 / 50, "a finding"))


def test_section_3(rng):
    xy = rng.normal(0, 1, (90, 2))
    cell_line = pd.Series(np.repeat(LINES, 30))
    corr = pd.DataFrame(np.full((3, 3), 0.95) + 0.05 * np.eye(3), index=LINES, columns=LINES)
    assert drawn(plot_line_map(xy, cell_line, LINES, COLOURS, corr, (0.97, "E", "F")))
    models = [f"M{i}" for i in range(12)]
    names = pd.Series([f"line {i}" for i in range(12)], index=models)
    lineages = pd.Series(["Myeloid"] * 6 + ["?"] * 6, index=models)
    matches = {c: pd.Series(np.sort(rng.uniform(0.6, 0.9, 12))[::-1], index=models) for c in LINES}
    assert drawn(plot_line_matches(matches, names, lineages, COLOURS, "a finding"))


def test_section_4(rng):
    screens = ["bench_k562", "bench_rpe1", "kolf"]
    covered = {n: rng.uniform(0, 1, 300) > 0.5 for n in screens}
    covered["jiang"] = rng.uniform(0, 1, 300) > 0.8
    covered["kolf"][:] = False  # a screen that measured none of the panel
    assert drawn(plot_screen_coverage(covered, [*screens, "jiang"], "a finding"))
    cells = {"bench_k562": 150.0, "bench_rpe1": 500.0, "kolf": 0.0}
    depth = {c: 20_000.0 + 1_000 * i for i, c in enumerate(LINES)}
    assert drawn(plot_depth_and_cells(screens, covered, cells, depth, "a finding"))


def test_sections_5_and_6(rng):
    halves = {
        "ctrl_cpm": rng.uniform(0, 100, (1, 200)),
        "lfc_a": rng.normal(0, 1, (8, 200)),
        "lfc_b": rng.normal(0, 1, (8, 200)),
        "target": np.array([f"g{i}" for i in range(8)]),
    }
    reliability = {n: rng.uniform(-0.1, 0.9, 8) for n in ("bench_k562", "bench_hepg2", "h1train")}
    assert drawn(plot_reliability(halves, reliability, 0, 1, "a finding"))
    lines = ["bench_k562", "bench_jurkat", "bench_hepg2"]
    transfer = rng.uniform(0.2, 0.8, (3, 3))
    np.fill_diagonal(transfer, np.nan)
    own = {(a, b): (float(rng.uniform()), 40) for a in lines for b in lines if a != b}
    assert drawn(plot_transfer(transfer, lines, own, 0.5, 0.4, "a finding"))


def test_section_7(rng):
    cpm = rng.uniform(20, 2000, 30)
    forms = {f"form {i}": {k: rng.normal(0, 0.3, 30) for k in ("planted", "real", "random")} for i in (1, 2)}
    assert drawn(plot_dead_end(cpm, forms, "a finding"))


def test_section_8(rng):
    assert drawn(plot_fold_map(FOLD_LINES, HELD_TYPES))
    final = pd.DataFrame({"overall": rng.uniform(0.2, 0.5, 6)}, index=FOLD_LINES)
    consensus = pd.DataFrame({"overall": rng.uniform(0.1, 0.4, 6)}, index=FOLD_LINES)
    noise = pd.Series(rng.uniform(0, 0.01, 6), index=FOLD_LINES)
    assert drawn(plot_fold_scores(final, consensus, noise, "a finding"))


def test_section_9(rng):
    counts = {n: rng.poisson(3, 200).astype(float) for n in ("real", "naive generator", "our generator")}
    called = {"real": False, "naive generator": True, "our generator": False}
    assert drawn(plot_cell_shape(counts, called, "GENE", "KD", [90, 95], [400, 20], 100, "a finding"))


def test_section_10(rng):
    steps = ["none"] + [f"+{p}" for p in WATERFALL]
    mean_by_step = pd.Series(np.cumsum(rng.uniform(-0.002, 0.01, len(steps))) + 0.36, index=steps)
    gains = mean_by_step.diff().iloc[1:]
    assert drawn(plot_waterfall(mean_by_step, gains, 0.002, 0.30, "a finding"))
    fold_scores = {
        line: {a: {m: float(rng.uniform()) for m in SCORER_METRICS} for a in [*steps, "probe"]} for line in FOLD_LINES
    }
    assert drawn(plot_measures(fold_scores, steps, "a finding"))


def test_section_11(rng):
    n = 400
    moments = {
        k: (rng.uniform(0, 5, n), rng.uniform(0, 5, n), rng.uniform(0, 1, n))
        for k in ("real", "naive generator", "our generator")
    }
    keep = rng.uniform(0, 1, n) > 0.2
    gap = {"naive generator": rng.normal(0, 0.1, 20), "our generator": rng.normal(0, 0.05, 20)}
    assert drawn(plot_generator_check(moments, keep, gap, [400, 20], "a finding"))


def test_section_13(rng):
    shown = {c: rng.normal(0, 1, (6, 8)) for c in LINES}
    kds, genes = [f"k{i}" for i in range(6)], [f"g{i}" for i in range(8)]
    assert drawn(plot_predicted_heatmaps(shown, COLOURS, kds, genes, 2.0, "a finding"))
    pair_r = {
        ("D", "E"): rng.uniform(-0.2, 1, 50),
        ("D", "F"): rng.uniform(-0.2, 1, 50),
        ("E", "F"): rng.uniform(0, 1, 50),
    }
    pair_r[("D", "E")][0] = np.nan
    assert drawn(plot_pair_similarity(pair_r, ("E", "F"), COLOURS, "a finding"))
    panel = [f"k{i}" for i in range(15)]
    size = {c: rng.uniform(0, 5, 15) for c in LINES}
    cpm = {c: [float(x) for x in rng.uniform(0, 100, 15)] for c in LINES}
    cpm["D"][0] = np.nan
    assert drawn(plot_strongest_knockdowns(size, panel, cpm, COLOURS, "a finding"))


def test_the_check_table():
    table = pass_fail_table([("a rule", "measured", True), ("another rule", "measured", False)])
    assert list(table["pass"]) == ["PASS", "FAIL"]
    pytest.importorskip("jinja2")  # pandas' Styler needs it; the notebook's extras install it
    html = style_checks(table).to_html()
    assert "PASS" in html and "FAIL" in html


def test_section_15(rng):
    measured, predicted = rng.normal(0, 1, (5, 10)), rng.normal(0, 0.5, (5, 10))
    kds, genes = [f"k{i}" for i in range(5)], np.array([f"g{i}" for i in range(10)])
    assert drawn(plot_own_line(measured, predicted, kds, genes, 2.0, 0.5, "a finding"))
