"""Suspensions are not points on a probability plot (#478).

A suspension (right-censored unit) moves the plotting positions of the
failures after it but has no plotting position of its own. It used to be
drawn at the F of the failure before it, where it looked like one more
failure. Following Abernethy's New Weibull Handbook and Weibull++, the
points drawn are the failures only.

``get_plot_data`` keeps its meaning: ``x_`` and ``F`` are every row of the
plotting positions, as before v0.22, and the ``failed`` mask selects the
failures (``d > 0``) that ``plot`` draws. ``x_censored`` holds the
suspension times, which ``plot(show_censored=True)`` marks on the time
axis.
"""

import warnings

import matplotlib
import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.nonparametric import plotting_positions

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = [10, 20, 30, 40, 50, 60]
C = [0, 0, 0, 0, 1, 1]
F_X = [0.15351828, 0.30695938, 0.46025942, 0.61325898]

INTERVALS = [[1, 2], [2, 3], [3, 5], 4, [4, 6], [5, 8]]


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _points(ax):
    """The points scattered onto ``ax`` (the probability plot's only
    scatter)."""
    (points,) = ax.collections
    return np.asarray(points.get_offsets())


def _rug(ax):
    # The rug's label starts with "_" so that it stays out of a legend
    # (#510): the legend is for the models' labelled fitted lines.
    return [ln for ln in ax.get_lines() if ln.get_label() == "_suspensions"]


def test_get_plot_data_keeps_every_row_and_marks_the_failures():
    d = sp.Weibull.fit(X, C).get_plot_data()
    # Every row, as before v0.22: the suspensions carry the F of the 40.
    np.testing.assert_array_equal(d["x_"], X)
    np.testing.assert_allclose(d["F"], F_X + [F_X[-1]] * 2, rtol=1e-7)
    assert d["failed"].dtype == bool
    np.testing.assert_array_equal(d["failed"], [1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(d["F"][d["failed"]], F_X, rtol=1e-7)
    np.testing.assert_array_equal(d["x_censored"], [50, 60])
    # the time axis spans the suspensions
    assert d["x_scale_max"] > 60


@pytest.mark.parametrize(
    "heuristic", ["Nelson-Aalen", "Kaplan-Meier", "Blom", "Filliben"]
)
def test_rows_are_the_plotting_positions(heuristic):
    # x_ and F are plotting_positions' rows, and failed is its d > 0.
    x = [3, 4, 4, 5, 6, 7, 9]
    c = [1, 0, 1, 0, 0, 1, 0]
    model = sp.Weibull.fit(x, c)
    d = model.get_plot_data(heuristic=heuristic)
    x_, _, dd, F = plotting_positions(x, c, heuristic=heuristic)
    np.testing.assert_array_equal(d["x_"], x_)
    np.testing.assert_allclose(d["F"], F)
    np.testing.assert_array_equal(d["failed"], dd > 0)
    np.testing.assert_array_equal(d["x_censored"], [3, 4, 7])


def test_suspensions_between_failures_move_the_positions_only():
    x = [10, 15, 20, 25, 30, 35]
    c = [0, 1, 0, 1, 0, 0]
    d = sp.Weibull.fit(x, c).get_plot_data(heuristic="Kaplan-Meier")
    np.testing.assert_array_equal(d["x_"], x)
    np.testing.assert_array_equal(d["failed"], [1, 0, 1, 0, 1, 1])
    km = sp.KaplanMeier.fit(x, c)
    np.testing.assert_allclose(d["F"], 1 - km.sf(x))
    np.testing.assert_array_equal(d["x_censored"], [15, 25])


def test_an_lfp_fit_marks_its_tail_of_suspensions_not_failed():
    rng = np.random.default_rng(0)
    x = np.r_[sp.Weibull.random(20, 10, 2, random_state=rng), [30] * 30]
    c = np.r_[np.zeros(20), np.ones(30)]
    d = sp.Weibull.fit(x, c, lfp=True).get_plot_data()
    assert d["x_"].size == 21 and d["x_"].max() == 30
    assert d["failed"].sum() == 20 and d["x_"][d["failed"]].max() < 30
    np.testing.assert_array_equal(d["x_censored"], [30])


def test_an_offset_shifts_every_row_and_the_suspensions():
    model = sp.Weibull.fit(np.array(X) + 100, C, offset=True, how="MPS")
    d = model.get_plot_data()
    np.testing.assert_allclose(d["x_"], np.array(X) + 100 - model.gamma)
    np.testing.assert_allclose(
        d["x_censored"], [150 - model.gamma, 160 - model.gamma]
    )


def test_turnbull_rows_without_mass_are_kept_but_not_failed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(INTERVALS)
        d = model.get_plot_data(heuristic="Turnbull")
    # Every endpoint of the Turnbull pieces, as before v0.22, including
    # (1, 0) and (4, 0.307), where the estimate has no mass.
    np.testing.assert_array_equal(d["x_"], [1, 2, 3, 4, 4, 5, 6, 8])
    np.testing.assert_array_equal(d["failed"], [0, 1, 1, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(d["x_"][d["failed"]], [2, 3, 4, 5, 6, 8])


@pytest.mark.parametrize("show", [False, True])
def test_plot_draws_the_failures_and_marks_suspensions_when_asked(show):
    model = sp.Weibull.fit(X, C)
    ax = model.plot(show_censored=show)
    points = _points(ax)
    np.testing.assert_array_equal(points[:, 0], [10, 20, 30, 40])
    np.testing.assert_allclose(points[:, 1], F_X, rtol=1e-7)
    rug = _rug(ax)
    assert len(rug) == int(show)
    if show:
        np.testing.assert_array_equal(rug[0].get_xdata(), [50, 60])
    # the axes span every time, the suspensions included
    assert ax.get_xlim()[1] > 60


def test_plot_draws_only_the_turnbull_points_with_mass():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(INTERVALS)
        ax = model.plot(heuristic="Turnbull")
    np.testing.assert_array_equal(_points(ax)[:, 0], [2, 3, 4, 5, 6, 8])


def test_a_model_without_data_draws_no_points():
    ax = sp.Weibull.from_params([10, 2]).plot(show_censored=True)
    assert _points(ax).shape[0] == 0
    assert _rug(ax) == []


def _mixture():
    x = np.r_[
        sp.Weibull.random(20, 5, 3, random_state=0),
        sp.Weibull.random(20, 50, 3, random_state=1),
    ]
    c = np.zeros(40)
    c[[3, 25, 30]] = 1
    return x, c, sp.MixtureModel.fit(x, c, dist=sp.Weibull, m=2)


def test_mixture_get_plot_data_keeps_every_row_and_marks_the_failures():
    x, c, model = _mixture()
    d = model.get_plot_data()
    x_, _, dd, F = plotting_positions(x, c, heuristic="Nelson-Aalen")
    np.testing.assert_array_equal(d["x_"], x_)
    np.testing.assert_allclose(d["F"], F)
    np.testing.assert_array_equal(d["failed"], dd > 0)
    np.testing.assert_array_equal(d["x_censored"], np.sort(x[c == 1]))


@pytest.mark.parametrize("show", [False, True])
def test_mixture_plot_draws_the_failures_and_marks_suspensions(show):
    x, c, model = _mixture()
    ax = model.plot(show_censored=show)
    np.testing.assert_array_equal(
        np.sort(_points(ax)[:, 0]), np.sort(x[c == 0])
    )
    rug = _rug(ax)
    assert len(rug) == int(show)
    if show:
        np.testing.assert_array_equal(rug[0].get_xdata(), np.sort(x[c == 1]))


def test_nonparametric_get_plot_data_marks_the_failures():
    x = [10, 15, 20, 25, 30, 35]
    c = [0, 1, 0, 1, 0, 0]
    d = sp.KaplanMeier.fit(x, c).get_plot_data()
    np.testing.assert_array_equal(d["x_"], x)
    np.testing.assert_array_equal(d["failed"], [1, 0, 1, 0, 1, 1])
    turnbull = sp.Turnbull.fit(INTERVALS).get_plot_data()
    np.testing.assert_array_equal(turnbull["failed"], [0, 1, 1, 0, 1, 1, 1, 1])
    # A model from an ECDF has no d: its failures are where F steps up.
    ecdf = sp.NonParametric.fit_from_ecdf([1.0, 2.0, 3.0], [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(
        ecdf.get_plot_data(plot_bounds=False)["failed"], [1, 0, 1]
    )
