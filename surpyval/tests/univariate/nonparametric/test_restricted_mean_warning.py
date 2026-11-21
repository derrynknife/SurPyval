"""A non-parametric mean, its bound and the mean residual life, given no
``tau``, are the area under the estimate to the largest observed time;
where units are still at risk after it, they warn that the value is the
restricted mean, not the mean life (principle 22).

Reference values are worked by hand: exact fractions for the
Kaplan-Meier, and the closed form of the Turnbull's Fleming-Harrington
steps on non-overlapping intervals.
"""

import warnings
from fractions import Fraction

import numpy as np
import pytest

import surpyval as sp

# Ten units: failures at 500, 800 and 1200, seven still running at 1500.
X = [500, 800, 1200] + [1500] * 7
C = [0, 0, 0] + [1] * 7


def _caught(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    return value, [w for w in caught if "restricted" in str(w.message)]


def test_kaplan_meier_mean_is_the_area_to_the_last_time_and_warns():
    model = sp.KaplanMeier.fit(X, c=C)
    # S: 0.9 from 500, 0.8 from 800, 0.7 from 1200 to 1500
    area = (
        Fraction(500)
        + Fraction(300) * Fraction(9, 10)
        + Fraction(400) * Fraction(8, 10)
        + Fraction(300) * Fraction(7, 10)
    )
    assert area == 1300
    value, caught = _caught(model.mean)
    assert value == pytest.approx(float(area), rel=1e-12)
    assert len(caught) == 1
    message = str(caught[0].message)
    for part in ("mean()", "1500", "0.7", "70%", "tau", "rmst(tau)"):
        assert part in message, message
    assert "parametric" in message
    assert caught[0].category is UserWarning
    # the warning points at the caller, not into the package
    assert caught[0].filename == __file__


def test_a_stated_horizon_does_not_warn():
    model = sp.KaplanMeier.fit(X, c=C)
    value, caught = _caught(lambda: model.mean(tau=1500))
    assert value == pytest.approx(1300.0, rel=1e-12)
    assert caught == []
    result, caught = _caught(model.rmst)
    assert result["rmst"] == pytest.approx(1300.0, rel=1e-12)
    assert caught == []
    _, caught = _caught(lambda: model.mrl(1000, tau=1500))
    assert caught == []


def test_mrl_and_mean_cb_warn_alike():
    model = sp.KaplanMeier.fit(X, c=C)
    # (200 * 0.8 + 300 * 0.7) / 0.8 = 462.5
    value, caught = _caught(lambda: model.mrl(1000))
    assert value == pytest.approx(462.5, rel=1e-12)
    assert len(caught) == 1
    assert "mrl(x)" in str(caught[0].message)
    assert "mean residual life" in str(caught[0].message)
    bounds, caught = _caught(model.mean_cb)
    assert len(caught) == 1
    assert "mean_cb()" in str(caught[0].message)
    np.testing.assert_allclose(bounds, model.mean_cb(tau=1500), rtol=0)


def test_turnbull_mean_past_the_last_inspection():
    # One unit failed in each of (0, 500], (500, 1000], (1000, 1500];
    # three still running at 1500. The Turnbull's default steps are
    # exp(-1/6), exp(-1/6 - 1/5) and exp(-1/6 - 1/5 - 1/4) at the
    # intervals' ends (Fleming-Harrington on the expected counts).
    model = sp.Turnbull.fit(
        x=[[0, 500], [500, 1000], [1000, 1500], 1500],
        c=[2, 2, 2, 1],
        n=[1, 1, 1, 3],
    )
    s1, s2, s3 = np.exp(-1 / 6), np.exp(-11 / 30), np.exp(-37 / 60)
    np.testing.assert_allclose(
        model.sf([500, 1000, 1500]), [s1, s2, s3], rtol=1e-12
    )
    value, caught = _caught(model.mean)
    assert value == pytest.approx(500 * (1 + s1 + s2), rel=1e-12)
    assert len(caught) == 1
    assert "{:.4g}".format(s3) in str(caught[0].message)
    value, caught = _caught(lambda: model.mrl(1000))
    assert value == pytest.approx(500.0, rel=1e-12)
    assert len(caught) == 1


@pytest.mark.parametrize(
    "fitter", [sp.KaplanMeier, sp.NelsonAalen, sp.FlemingHarrington]
)
def test_every_estimator_warns_with_units_left_at_risk(fitter):
    model = fitter.fit(X, c=C)
    _, caught = _caught(model.mean)
    assert len(caught) == 1


@pytest.mark.parametrize(
    "fitter", [sp.KaplanMeier, sp.NelsonAalen, sp.FlemingHarrington]
)
def test_no_warning_when_every_unit_has_failed(fitter):
    # The Nelson-Aalen and Fleming-Harrington curves stay above zero at
    # the last failure, but no unit is left at risk: nothing is missing.
    model = fitter.fit([1, 2, 2, 3, 5, 8])
    _, caught = _caught(model.mean)
    assert caught == []
    _, caught = _caught(lambda: model.mrl(2))
    assert caught == []


def test_turnbull_with_every_unit_failed_does_not_warn():
    model = sp.Turnbull.fit(
        x=[[0, 500], [500, 1000], [1000, 1500]], c=[2, 2, 2]
    )
    _, caught = _caught(model.mean)
    assert caught == []
