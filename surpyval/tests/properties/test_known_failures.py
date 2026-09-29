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


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# Parametric fits to data whose likelihood has no maximum: refused since
# #392 was fixed (they returned a degenerate spike, silently)
# ---------------------------------------------------------------------------
def test_exact_value_and_left_censored_above_it_are_refused():
    # A failure at 0.5 and one known only to be before 1: a spike at 0.5
    # explains both perfectly, so the likelihood grows without bound.
    # Weibull returns alpha 0.500, beta 395.7 (no warning); Normal sigma
    # 5e-324; LogNormal took about 20 s. Three tied values are refused.
    with pytest.raises(ValueError, match="no maximum"):
        _quiet(sp.Weibull.fit, x=[0.5, 1.0], c=[0, -1])


def test_overlapping_intervals_are_refused():
    # Two intervals, (1, 3] and (2, 4], share (2, 3]: a spike there has
    # likelihood 1. Weibull returned alpha 2.85, beta 57.9.
    with pytest.raises(ValueError, match="no maximum"):
        _quiet(sp.Weibull.fit, xl=[1.0, 2.0], xr=[3.0, 4.0])


# ---------------------------------------------------------------------------
# Parametric fits to truncated data are unit free since #393 was fixed:
# the truncation term was NaN once F(tl) rounded to 1 (#412), so every
# gradient rung failed and Powell stopped where it could (5 s a fit)
# ---------------------------------------------------------------------------
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


def test_normal_truncated_fit_is_unit_free():
    # mu, sigma = 8.536, 2.461 on the data; on the data x 7.3 it gave
    # 48.87, 23.04, not 62.31, 17.96 (log-likelihood -4.630 against
    # -4.034, in the original unit). The maximum is 8.745, 2.459 (-4.026).
    ref = _quiet(sp.Normal.fit, **_UNITS_DATA)
    got = _rescaled_fit(sp.Normal, 7.3)
    np.testing.assert_allclose(got.params, ref.params * 7.3, rtol=1e-3)


def test_gumbel_truncated_fit_is_unit_free():
    # On the data x 7.3 Gumbel gave mu, sigma = 49.64, 1.99, under which
    # the right censored row, in (76.65, 94.9], has probability 0.
    ref = _quiet(sp.Gumbel.fit, **_UNITS_DATA)
    got = _rescaled_fit(sp.Gumbel, 7.3)
    np.testing.assert_allclose(got.params, ref.params * 7.3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_cox_refuses_an_infinite_event_time():
    # #394: CoxPH.fit accepted an exactly observed time of inf (every
    # univariate fitter refuses it) and returned beta 19.4.
    with pytest.raises(ValueError):
        _quiet(sp.CoxPH.fit, x=[np.inf, 0.5], Z=[[-1.0], [1.0]], c=[0, 0])
