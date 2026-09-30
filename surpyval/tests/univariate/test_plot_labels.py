"""Plots can be labelled, coloured and overlaid (#510), and have default
axis labels (#514).

Comparing two populations on one probability plot needs each model in its
own colour -- points, fitted line and bounds -- and a legend label.
"""

import inspect

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import surpyval as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture(scope="module")
def two_fits():
    rng = np.random.default_rng(3)
    north = sp.Weibull.fit(10 * rng.weibull(3, 30))
    south = sp.Weibull.fit(50 * rng.weibull(2, 30))
    return north, south


def _colours(ax):
    """The distinct colours of every line and point collection."""
    lines = {to_rgba(ln.get_color()) for ln in ax.get_lines()}
    points = {
        tuple(c) for coll in ax.collections for c in coll.get_facecolor()
    }
    return lines, points


def test_parametric_plot_takes_a_label_for_the_fitted_line(ax, two_fits):
    north, south = two_fits
    north.plot(ax=ax, label="North")
    south.plot(ax=ax, label="South")
    assert ax.get_legend_handles_labels()[1] == ["North", "South"]


def test_one_colour_per_call_from_the_axes_cycle(ax, two_fits):
    north, south = two_fits
    north.plot(ax=ax)
    first_lines, first_points = _colours(ax)
    # the fitted line, the two bounds and the points share one colour
    assert len(first_lines) == 1 and first_lines == first_points
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert first_lines == {to_rgba(cycle[0])}
    south.plot(ax=ax)
    lines, points = _colours(ax)
    assert lines == points == {to_rgba(cycle[0]), to_rgba(cycle[1])}


def test_color_and_line_keywords_are_used(ax, two_fits):
    north, _ = two_fits
    north.plot(ax=ax, color="red", linewidth=3, linestyle=":")
    lines, points = _colours(ax)
    assert lines == points == {to_rgba("red")}
    fitted = ax.get_lines()[0]
    assert fitted.get_linewidth() == 3 and fitted.get_linestyle() == ":"


def test_overlay_keeps_both_models_in_view(ax, two_fits):
    north, south = two_fits
    north.plot(ax=ax)
    lo = ax.get_xlim()[0]
    south.plot(ax=ax)
    assert ax.get_xlim()[0] == pytest.approx(lo)
    assert ax.get_xlim()[1] > 50


def test_a_model_from_parameters_plots_with_a_label_and_colour(ax):
    sp.Weibull.from_params([10, 2]).plot(ax=ax, label="spec", color="k")
    assert ax.get_legend_handles_labels()[1] == ["spec"]
    assert _colours(ax)[0] == {to_rgba("k")}


def test_plot_returns_axes_and_says_so(ax, two_fits):
    from matplotlib.axes import Axes

    north, _ = two_fits
    assert isinstance(north.plot(ax=ax), Axes)
    ret = inspect.signature(sp.Parametric.plot).return_annotation
    assert ret in ("Axes", Axes)


def test_mixture_plot_takes_a_label_and_colour(ax):
    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    model = sp.MixtureModel.fit(x, dist=sp.Weibull, m=2)
    model.plot(ax=ax, label="mixture", color="purple")
    assert ax.get_legend_handles_labels()[1] == ["mixture"]
    lines, points = _colours(ax)
    assert lines == points == {to_rgba("purple")}


# Default axis labels (#514)


def test_parametric_plot_labels_the_time_axis(ax, two_fits):
    north, _ = two_fits
    north.plot(ax=ax)
    assert ax.get_xlabel() == "Time"


def test_parametric_plot_keeps_the_users_x_label(ax, two_fits):
    north, _ = two_fits
    ax.set_xlabel("Cycles")
    north.plot(ax=ax)
    assert ax.get_xlabel() == "Cycles"


@pytest.mark.parametrize(
    "fitter, title",
    [
        (sp.KaplanMeier, "Kaplan-Meier estimate"),
        (sp.NelsonAalen, "Nelson-Aalen estimate"),
        (sp.FlemingHarrington, "Fleming-Harrington estimate"),
        (sp.Turnbull, "Turnbull estimate"),
    ],
)
def test_non_parametric_plot_labels(ax, fitter, title):
    fitter.fit([1, 2, 3, 5, 8], c=[0, 1, 0, 0, 1]).plot(ax=ax)
    assert ax.get_title() == title
    assert ax.get_ylabel() == "Survival probability"
    assert ax.get_xlabel() == "Time"


def test_non_parametric_plot_keeps_the_users_x_label(ax):
    ax.set_xlabel("Days")
    sp.KaplanMeier.fit([1, 2, 3, 5, 8]).plot(ax=ax)
    assert ax.get_xlabel() == "Days"


def test_kaplan_meier_overlay_labels_and_colours(ax):
    sp.KaplanMeier.fit([1, 2, 3, 5, 8]).plot(ax=ax, label="A")
    sp.KaplanMeier.fit([2, 4, 6, 9, 12]).plot(ax=ax, label="B")
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["A", "B"]
    assert handles[0].get_color() != handles[1].get_color()


def test_competing_risks_plot_labels_the_time_axis(ax):
    from surpyval.univariate.competing_risks import CompetingRisks

    model = CompetingRisks.fit(
        [1, 2, 3, 4, 5, 6, 7, 8],
        ["a", "b", "a", "b", "a", None, "a", "b"],
        c=[0, 0, 0, 0, 0, 1, 0, 0],
    )
    assert model.plot(ax=ax).get_xlabel() == "Time"
