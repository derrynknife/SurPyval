"""Failures the properties found, pinned as strict xfails (#379).

Each test is a counterexample Hypothesis found and shrank, confirmed by
hand, written out so it reproduces without Hypothesis or its database.
The properties themselves ``assume`` the failing data away (see
``known.py``) so the rest of the data space is still searched. A strict
xfail fails (XPASS) the day the bug is fixed: then delete the test and
its predicate in ``known.py``.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.properties import strategies as gen


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# Turnbull drops its last piece when every row is right truncated
# ---------------------------------------------------------------------------
_TURNBULL_TR = (
    "#391: "
    "Turnbull drops the last piece of its ladder when every row has a "
    "finite right truncation time (the ladder assumes bounds end at +inf)"
)


@pytest.mark.xfail(strict=True, reason=_TURNBULL_TR)
def test_turnbull_failure_at_its_right_truncation_time():
    # One failure at 1, observable up to 1: all the mass is at 1, so
    # sf(1) is 0. Turnbull gives 1 (ladder x = [1], R = [1]).
    model = _quiet(sp.Turnbull.fit, x=[1.0], c=[0], tr=[1.0])
    assert model.sf([1.0])[0] == 0.0


@pytest.mark.xfail(strict=True, reason=_TURNBULL_TR)
def test_turnbull_two_failures_right_truncated_at_the_last():
    # Failures at 1 and 2, both observable up to 2: sf(2) is 0. Turnbull
    # gives 0.5; with the second row untruncated (tr = inf) it gives 0.
    model = _quiet(sp.Turnbull.fit, x=[1.0, 2.0], c=[0, 0], tr=[2.0, 2.0])
    np.testing.assert_allclose(model.sf([1.0, 2.0]), [0.5, 0.0])


@pytest.mark.xfail(strict=True, reason=_TURNBULL_TR)
def test_turnbull_right_censored_inside_its_window():
    # Censored at 1 and observable up to 2: the event is in (1, 2], so
    # sf(2) is 0. Turnbull gives 1.
    model = _quiet(sp.Turnbull.fit, x=[1.0], c=[1], tr=[2.0])
    assert model.sf([2.0])[0] == 0.0


@pytest.mark.xfail(strict=True, reason=_TURNBULL_TR)
def test_turnbull_left_censored_at_its_right_truncation_time():
    # Left censored at 1, observable up to 1: the ladder is empty and sf
    # raises IndexError (index -1 of an empty R).
    model = _quiet(sp.Turnbull.fit, x=[1.0], c=[-1], tr=[1.0])
    assert model.sf([0.5, 1.0])[1] == 0.0


# ---------------------------------------------------------------------------
# Parametric fits to data whose likelihood has no maximum
# ---------------------------------------------------------------------------
_POINT_MASS = (
    "#392: "
    "data whose likelihood has no maximum (one point in every row's set) "
    "are fitted to a degenerate spike, silently, instead of refused as "
    "tied data are"
)


@pytest.mark.xfail(strict=True, reason=_POINT_MASS)
def test_exact_value_and_left_censored_above_it_are_refused():
    # A failure at 0.5 and one known only to be before 1: a spike at 0.5
    # explains both perfectly, so the likelihood grows without bound.
    # Weibull returns alpha 0.500, beta 395.7 (no warning); Normal sigma
    # 5e-324; LogNormal takes about 20 s. Three tied values are refused.
    with pytest.raises(ValueError):
        _quiet(sp.Weibull.fit, x=[0.5, 1.0], c=[0, -1])


@pytest.mark.xfail(strict=True, reason=_POINT_MASS)
def test_overlapping_intervals_are_refused():
    # Two intervals, (1, 3] and (2, 4], share (2, 3]: a spike there has
    # likelihood 1. Weibull returns alpha 2.85, beta 57.9.
    with pytest.raises(ValueError):
        _quiet(sp.Weibull.fit, xl=[1.0, 2.0], xr=[3.0, 4.0])


# ---------------------------------------------------------------------------
# Parametric fits to truncated data depend on the time unit
# ---------------------------------------------------------------------------
_UNITS = (
    "#393: "
    "a parametric fit to truncated data depends on the time unit: on the "
    "rescaled data the optimiser stops at a worse point"
)
_UNITS_DATA = dict(
    x=[10.5, 6.5, 2.5, 11.0],
    c=[1, 0, 1, -1],
    tr=[13.0, np.inf, np.inf, np.inf],
)


def _rescaled_fit(fitter, k):
    scaled = dict(_UNITS_DATA)
    scaled["x"] = np.asarray(_UNITS_DATA["x"]) * k
    scaled["tr"] = np.asarray(_UNITS_DATA["tr"]) * k
    return _quiet(fitter.fit, **scaled)


# Each of these fits takes about five seconds: nightly only.
_NIGHTLY = pytest.mark.skipif(
    not gen.THOROUGH, reason="slow: SURPYVAL_HYPOTHESIS_PROFILE=nightly"
)


@_NIGHTLY
@pytest.mark.xfail(strict=True, reason=_UNITS)
def test_normal_truncated_fit_is_unit_free():
    # mu, sigma = 8.536, 2.461 on the data; on the data x 7.3 it gives
    # 48.87, 23.04, not 62.31, 17.96 (log-likelihood -4.630 against
    # -4.034, in the original unit).
    ref = _quiet(sp.Normal.fit, **_UNITS_DATA)
    got = _rescaled_fit(sp.Normal, 7.3)
    np.testing.assert_allclose(got.params, ref.params * 7.3, rtol=1e-3)


@_NIGHTLY
@pytest.mark.xfail(strict=True, reason=_UNITS)
def test_gumbel_truncated_fit_is_unit_free():
    # On the data x 7.3 Gumbel gives mu, sigma = 49.64, 1.99, under which
    # the right censored row, in (76.65, 94.9], has probability 0.
    ref = _quiet(sp.Gumbel.fit, **_UNITS_DATA)
    got = _rescaled_fit(sp.Gumbel, 7.3)
    np.testing.assert_allclose(got.params, ref.params * 7.3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="#394: CoxPH.fit accepts an exactly observed time of inf (every "
    "univariate fitter refuses it) and returns beta 19.4",
)
def test_cox_refuses_an_infinite_event_time():
    with pytest.raises(ValueError):
        _quiet(sp.CoxPH.fit, x=[np.inf, 0.5], Z=[[-1.0], [1.0]], c=[0, 0])
