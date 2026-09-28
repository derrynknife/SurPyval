"""Targeted review of ``surpyval/utils/__init__.py`` (#399).

Each test pins a bug found by reading the module adversarially. They are
strict expected failures until the bug is fixed.
"""

import matplotlib
import pytest

import surpyval as surv

matplotlib.use("Agg")


@pytest.mark.xfail(
    strict=True,
    reason="#439: the probability plot's tick rounding (utils._round_vals / "
    "round_sig) takes log10 of a tick at exactly 0, so Normal.fit([-1, "
    "0.5, 2, 3, 5]).plot() raises OverflowError",
)
def test_probability_plot_with_a_tick_at_zero():
    import matplotlib.pyplot as plt

    model = surv.Normal.fit([-1, 0.5, 2, 3, 5.0])
    try:
        model.plot()
    finally:
        plt.close("all")
