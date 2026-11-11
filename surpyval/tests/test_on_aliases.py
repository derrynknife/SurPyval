"""``on="R"`` and ``on="F"`` are accepted spellings of ``"sf"`` and
``"ff"`` wherever an ``on`` argument takes those (docs/Conventions.rst):
each gives exactly what the method name gives. These are the places that
used to refuse them."""

import matplotlib
import numpy as np
import pytest

import surpyval as sp

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ALIASES = [("R", "sf"), ("F", "ff")]

X = np.array([2.0, 3.0, 5.0, 8.0, 4.0, 1.0, 2.0, 2.0, 3.0, 6.0, 7.0, 9.0])
C = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1])
G = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)


def _lines(ax):
    return [
        (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
        for line in ax.get_lines()
    ]


@pytest.mark.parametrize("alias, name", ALIASES)
def test_adjusted_curve_cb(alias, name):
    model = sp.ExponentialPH.fit(X, G.reshape(-1, 1), c=C)
    adjusted = sp.direct_adjustment(model, group=0)
    t = [1.0, 3.0, 5.0]
    for bound in ("two-sided", "lower", "upper"):
        np.testing.assert_array_equal(
            adjusted.cb(t, 1.0, on=alias, bound=bound),
            adjusted.cb(t, 1.0, on=name, bound=bound),
        )


@pytest.mark.parametrize("alias, name", ALIASES)
def test_plot_nonparametric(alias, name):
    km = sp.KaplanMeier.fit(X, C)
    a = sp.plot_nonparametric(km, on=alias, quantiles=0.5)
    b = sp.plot_nonparametric(km, on=name, quantiles=0.5)
    try:
        assert a.ax.get_ylabel() == b.ax.get_ylabel()
        assert a.quantiles == b.quantiles
        lines_a, lines_b = _lines(a.ax), _lines(b.ax)
        assert len(lines_a) == len(lines_b) > 0
        for (xa, ya), (xb, yb) in zip(lines_a, lines_b):
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)
    finally:
        plt.close(a.fig)
        plt.close(b.fig)


@pytest.mark.parametrize("alias, name", ALIASES)
def test_plot_partial_effects(alias, name):
    model = sp.WeibullPH.fit(X, G.reshape(-1, 1), c=C)
    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    try:
        model.plot_partial_effects(0, [0.0, 1.0], on=alias, ax=ax_a)
        model.plot_partial_effects(0, [0.0, 1.0], on=name, ax=ax_b)
        assert ax_a.get_ylabel() == ax_b.get_ylabel()
        for (xa, ya), (xb, yb) in zip(_lines(ax_a), _lines(ax_b)):
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("alias, name", ALIASES)
def test_bayesian_compare_to(alias, name):
    rate = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
    new = sp.Exponential.fit_bayes([300.0, 450, 800], priors=rate)
    old = sp.Exponential.fit_bayes([100.0, 250, 400], priors=rate)
    a = new.compare_to(old, on=alias, x=500.0)
    b = new.compare_to(old, on=name, x=500.0)
    assert a.probability == b.probability
    assert repr(a) == repr(b)
