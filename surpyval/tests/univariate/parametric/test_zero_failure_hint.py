"""A failure at exactly 0 points to ``zi=True`` (#514).

``Weibull.fit([0, 5, 8, 12, 20])`` raised "... Are some of your observed
values 0, -inf or inf?" without saying that the zero-inflated model is
the fit for units dead on arrival.
"""

import pytest

import surpyval as sp
from surpyval.univariate.parametric.parametric_fitter import (
    OutsideSupportError,
)

HINT = "For units that failed at time 0 (dead on arrival), fit with `zi=True`."


@pytest.mark.parametrize("dist", [sp.Weibull, sp.Gamma, sp.LogNormal])
def test_a_failure_at_zero_points_to_zi(dist):
    with pytest.raises(OutsideSupportError) as info:
        dist.fit([0, 5, 8, 12, 20])
    assert str(info.value).endswith(HINT)
    # and the hint is right: the zero-inflated fit works
    assert dist.fit([0, 5, 8, 12, 20], zi=True).f0 == pytest.approx(0.2)


def test_a_zero_width_interval_at_zero_points_to_zi():
    with pytest.raises(OutsideSupportError, match="zi=True"):
        sp.Weibull.fit(xl=[0, 1, 2], xr=[0, 2, 3], c=[0, 2, 2])


def test_a_negative_value_does_not_point_to_zi():
    # zi models a mass at 0, not values below it
    with pytest.raises(OutsideSupportError) as info:
        sp.Weibull.fit([-1, 5, 8, 12])
    assert "zi=True" not in str(info.value)
