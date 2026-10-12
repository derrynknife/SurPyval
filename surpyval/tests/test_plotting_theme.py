"""SurPyval's plot theme (surpyval.plotting): one palette taken in order,
quiet axes, bands for bounds, and probability and time ticks chosen for
the axis they label."""

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import surpyval as sp
from surpyval import plotting
from surpyval.univariate.parametric.probability_plotting import (
    _linear_ticks,
    _log_ticks,
    _probability_ticks,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = sp.Weibull.random(40, 1000, 2.2, random_state=1)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_palette_taken_in_order_per_axes(ax):
    assert plotting.next_color(ax) == plotting.PALETTE[0]
    assert plotting.next_color(ax) == plotting.PALETTE[1]
    # a colour given is used, and still counts as a series
    assert plotting.next_color(ax, "k") == "k"
    assert plotting.next_color(ax) == plotting.PALETTE[3]


def test_the_theme_styles_the_axes(ax):
    sp.Weibull.fit(X).plot(ax=ax)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert to_rgb(ax.get_facecolor()) == to_rgb(plotting.INK["surface"])
    # the bounds are a band, not dashed lines
    assert len(ax.collections) == 2
    assert all(ln.get_linestyle() == "-" for ln in ax.get_lines())


def test_theme_off_leaves_matplotlib_alone(ax):
    plotting.set_theme(False)
    try:
        assert plotting.next_color(ax) is None
        sp.KaplanMeier.fit(X).plot(ax=ax)
        assert ax.spines["top"].get_visible()
    finally:
        plotting.set_theme(True)
    assert plotting.theme_enabled()


def test_rc_params_apply():
    with matplotlib.rc_context(plotting.rc_params()):
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        assert to_rgb(line.get_color()) == to_rgb(plotting.PALETTE[0])
        assert not ax.spines["top"].get_visible()
        plt.close(fig)


def test_probability_ticks_reach_the_tails_where_there_is_room():
    weibull = sp.Weibull.from_params([1000, 2])

    def tf(p):
        return sp.Weibull.mpp_y_transform(np.asarray(p), *weibull.params)

    ticks, minor = _probability_ticks(tf, 0.005, 0.995)
    assert 0.5 in ticks and 0.01 in ticks and 0.99 in ticks
    assert all(0.005 < p < 0.995 for p in ticks + minor)
    positions = np.sort(tf(np.array(ticks)))
    span = tf(0.995) - tf(0.005)
    assert np.all(np.diff(positions) >= span / 14 - 1e-12)


def test_log_time_ticks_are_one_two_five():
    positions, minor, labels = _log_ticks(150.0, 2600.0)
    assert labels == ["200", "500", "1,000", "2,000"]
    assert positions == [200.0, 500.0, 1000.0, 2000.0]
    assert 300.0 in minor and 3000.0 not in minor
    # a narrow range falls back to every 1 to 9 of a decade
    positions, _, labels = _log_ticks(620.0, 1450.0)
    assert len(labels) >= 3


def test_offset_time_ticks_label_round_times():
    # positions are times less gamma; the labels are the times
    positions, _, labels = _log_ticks(10.0, 2000.0, gamma=500.0)
    shown = [float(s.replace(",", "")) for s in labels]
    np.testing.assert_allclose(np.array(positions) + 500.0, shown)
    assert all(v % 100 == 0 for v in shown)


def test_linear_ticks_are_round():
    positions, labels = _linear_ticks(140.0, 1930.0)
    assert labels == ["500", "1,000", "1,500"]


def test_competing_risks_unstacked_by_default(ax):
    from surpyval.univariate.competing_risks import CompetingRisks

    model = CompetingRisks.fit(
        [1, 2, 3, 4, 5, 6, 7, 8],
        ["a", "b", "a", "b", "a", None, "a", "b"],
        c=[0, 0, 0, 0, 0, 1, 0, 0],
    )
    model.plot(ax=ax)
    assert ax.get_title() == "Cumulative incidence by cause"
    assert [to_rgb(ln.get_color()) for ln in ax.get_lines()] == [
        to_rgb(plotting.PALETTE[0]),
        to_rgb(plotting.PALETTE[1]),
    ]


def test_recurrent_plot_draws_the_data_quietly_and_the_model_loud(ax):
    from surpyval.recurrent import CrowAMSAA

    CrowAMSAA.fit(np.cumsum(X[:25])).plot(ax=ax)
    observed, model = ax.get_lines()
    assert to_rgb(observed.get_color()) == to_rgb(plotting.INK["secondary"])
    assert to_rgb(model.get_color()) == to_rgb(plotting.PALETTE[0])
    assert ax.get_legend() is not None
