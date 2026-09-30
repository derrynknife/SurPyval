"""Suspensions are not points on a probability plot (#478).

A suspension (right-censored unit) moves the plotting positions of the
failures after it but has no plotting position of its own. It used to be
drawn at the F of the failure before it, where it looked like one more
failure. Following Abernethy's New Weibull Handbook and Weibull++, the
points are now the failures only; the suspension times are returned as
``x_censored`` and ``plot(show_censored=True)`` marks them on the time axis.
"""

import warnings

import matplotlib
import numpy as np
import pytest

import surpyval as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = [10, 20, 30, 40, 50, 60]
C = [0, 0, 0, 0, 1, 1]


def test_suspensions_are_not_plotted_points():
    # Before: x_ = [10, ..., 60] with F = [0.154, 0.307, 0.460, 0.613,
    # 0.613, 0.613] -- the two suspensions at the F of the 40.
    d = sp.Weibull.fit(X, C).get_plot_data()
    np.testing.assert_array_equal(d["x_"], [10, 20, 30, 40])
    np.testing.assert_allclose(
        d["F"], [0.15351828, 0.30695938, 0.46025942, 0.61325898], rtol=1e-7
    )
    np.testing.assert_array_equal(d["x_censored"], [50, 60])
    # the time axis still spans the suspensions
    assert d["x_scale_max"] > 60


def test_suspensions_between_failures_move_the_positions_only():
    x = [10, 15, 20, 25, 30, 35]
    c = [0, 1, 0, 1, 0, 0]
    d = sp.Weibull.fit(x, c).get_plot_data(heuristic="Kaplan-Meier")
    np.testing.assert_array_equal(d["x_"], [10, 20, 30, 35])
    km = sp.KaplanMeier.fit(x, c)
    np.testing.assert_allclose(d["F"], 1 - km.sf([10, 20, 30, 35]))
    np.testing.assert_array_equal(d["x_censored"], [15, 25])


def test_an_lfp_fit_shows_no_tail_of_suspensions():
    rng = np.random.default_rng(0)
    x = np.r_[sp.Weibull.random(20, 10, 2, random_state=rng), [30] * 30]
    c = np.r_[np.zeros(20), np.ones(30)]
    d = sp.Weibull.fit(x, c, lfp=True).get_plot_data()
    assert d["x_"].size == 20 and d["x_"].max() < 30
    np.testing.assert_array_equal(d["x_censored"], [30])


def test_an_offset_shifts_the_suspensions_too():
    model = sp.Weibull.fit(np.array(X) + 100, C, offset=True, how="MPS")
    d = model.get_plot_data()
    np.testing.assert_allclose(
        d["x_censored"], [150 - model.gamma, 160 - model.gamma]
    )


def test_interval_data_plots_only_where_the_turnbull_estimate_has_mass():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit([[1, 2], [2, 3], [3, 5], 4, [4, 6], [5, 8]])
        d = model.get_plot_data(heuristic="Turnbull")
    # Before: 8 points, including (1, 0) and (4, 0.307), where the
    # Turnbull estimate has no mass.
    np.testing.assert_array_equal(d["x_"], [2, 3, 4, 5, 6, 8])


@pytest.mark.parametrize("show", [False, True])
def test_plot_marks_suspensions_only_when_asked(show):
    plt.close("all")
    model = sp.Weibull.fit(X, C)
    ax = model.plot(show_censored=show)
    scattered = ax.collections[0].get_offsets()
    assert len(scattered) == 4
    rug = [ln for ln in ax.get_lines() if ln.get_label() == "suspensions"]
    assert len(rug) == int(show)
    if show:
        np.testing.assert_array_equal(rug[0].get_xdata(), [50, 60])
    plt.close("all")
