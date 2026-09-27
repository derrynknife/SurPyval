"""Fitting leaves its inputs alone, and Fleming-Harrington is robust to
float noise in the risk and death sets."""

import numpy as np
import pytest

import surpyval as surv
from surpyval.univariate.nonparametric.fleming_harrington import (
    fh_h,
    fh_var_h,
)


def test_a_turnbull_fit_does_not_modify_its_interval_array():
    # infinite endpoints are rewritten as one-sided censoring internally;
    # that rewrite used to land in the caller's array
    x = np.array(
        [[1.0, 3.0], [2.0, np.inf], [0.5, 2.5], [-np.inf, 6.0], [5.0, 7.0]]
    )
    original = x.copy()
    first = surv.Turnbull.fit(x)
    np.testing.assert_array_equal(x, original)
    second = surv.Turnbull.fit(x)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(first.R, second.R)


def test_fh_treats_near_integer_sets_as_integers():
    noise = 2e-16
    assert fh_h(1 + noise, 1 + noise) == pytest.approx(1.0)
    assert fh_h(5 + noise, 2 - noise) == pytest.approx(fh_h(5.0, 2.0))
    assert np.isfinite(fh_var_h(1 + noise, 1 + noise))
    # no deaths means no hazard, even if the risk set is (numerically) zero
    assert fh_h(-8.9e-16, 0.0) == 0.0
    assert fh_var_h(-8.9e-16, 0.0) == 0.0
    # genuinely fractional sets keep the fractional formula
    assert fh_h(4.5, 1.5) != pytest.approx(fh_h(5.0, 2.0))


def test_turnbull_fh_equals_fleming_harrington_on_complete_data():
    x = np.arange(1.0, 9.0)
    turnbull = surv.Turnbull.fit(x, turnbull_estimator="Fleming-Harrington")
    direct = surv.FlemingHarrington.fit(x)
    np.testing.assert_allclose(turnbull.sf(x), direct.sf(x), rtol=1e-10)
    assert np.all(np.isfinite(turnbull.sf(x)))
