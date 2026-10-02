"""Targeted review of ``surpyval/utils/__init__.py`` (#399).

Each test pins a bug found by reading the module adversarially (fixed
under #439; they were strict expected failures until then).
"""

import matplotlib
import numpy as np
import pytest

import surpyval as surv
from surpyval.utils.numeric import _round_vals

matplotlib.use("Agg")


def test_probability_plot_with_a_tick_at_zero():
    # The tick rounding (utils._round_vals / round_sig) took log10 of a
    # tick at exactly 0 and raised OverflowError.
    import matplotlib.pyplot as plt

    model = surv.Normal.fit([-1, 0.5, 2, 3, 5.0])
    try:
        model.plot()
    finally:
        plt.close("all")


def test_round_sig_of_zero_negative_and_non_finite_values():
    assert surv.round_sig(0) == 0
    assert surv.round_sig(0.0, 3) == 0.0
    got = surv.round_sig(np.array([-1234.5, 0.0, -0.012345, np.inf]), 2)
    np.testing.assert_array_equal(got, [-1200.0, 0.0, -0.012, np.inf])


@pytest.mark.parametrize(
    "ticks, want",
    [
        ([-2.0, -1.0, 0.0, 1.0, 2.0], [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ([-0.15, 0.0, 0.161, 0.17], [-0.15, 0.0, 0.16, 0.17]),
        ([1.0, 1.0], [1.0, 1.0]),  # equal ticks stop, not loop
    ],
)
def test_round_vals_keeps_ticks_apart_through_zero(ticks, want):
    np.testing.assert_array_equal(_round_vals(np.array(ticks)), want)
