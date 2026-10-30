"""Fits at extreme time scales.

The conformance suite changes the time unit by factors from 1e-6 to 731.
A change of unit only rescales the scale parameter, so the fit should be
the same far beyond that. These tests widen the range to 1e-150..1e150
(where it holds) and pin, as strict xfails, the scales where a Weibull
fit stops being exact or where a warning is given twice.
"""

import warnings

import numpy as np
import pytest

from surpyval import Exponential, KaplanMeier, LogNormal, Turnbull, Weibull
from surpyval.univariate.regression import WeibullPH

SCALES = [1e-150, 1e-100, 1e-50, 1e50, 1e100, 1e150]
UNVERIFIED = "The maximum-likelihood search did not reach a verified maximum"


def _data():
    rng = np.random.default_rng(1)
    x = rng.weibull(2.0, 30) * 100
    c = (rng.random(30) < 0.3).astype(int)
    Z = rng.normal(size=(30, 2))
    return x, c, Z


def _silent(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


@pytest.mark.parametrize("s", SCALES)
def test_weibull_scale_invariant(s):
    x, *_ = _data()
    unit = Weibull.fit(x)
    m = _silent(Weibull.fit, x * s)
    assert m.maximum == "verified"
    np.testing.assert_allclose(
        [m.params[0] / s, m.params[1]], unit.params, rtol=1e-6
    )


@pytest.mark.parametrize("s", SCALES)
def test_exponential_scale_invariant(s):
    x, *_ = _data()
    unit = Exponential.fit(x)
    m = _silent(Exponential.fit, x * s)
    assert m.maximum == "verified"
    np.testing.assert_allclose(m.params[0] * s, unit.params[0], rtol=1e-6)


@pytest.mark.parametrize("s", SCALES)
def test_lognormal_scale_invariant(s):
    x, *_ = _data()
    unit = LogNormal.fit(x)
    m = _silent(LogNormal.fit, x * s)
    assert m.maximum == "verified"
    np.testing.assert_allclose(
        [m.params[0] - np.log(s), m.params[1]], unit.params, rtol=1e-6
    )


@pytest.mark.parametrize("s", SCALES)
def test_kaplan_meier_scale_invariant(s):
    x, *_ = _data()
    q = np.array([20.0, 50.0, 100.0, 150.0])
    np.testing.assert_array_equal(
        _silent(KaplanMeier.fit, x * s).sf(q * s), KaplanMeier.fit(x).sf(q)
    )


@pytest.mark.parametrize("s", SCALES)
def test_turnbull_scale_invariant(s):
    x, c, _ = _data()
    q = np.array([20.0, 50.0, 100.0, 150.0])
    np.testing.assert_allclose(
        _silent(Turnbull.fit, x * s, c=c).sf(q * s),
        Turnbull.fit(x, c=c).sf(q),
        rtol=1e-10,
    )


@pytest.mark.xfail(
    strict=True,
    reason="a Weibull fit at scale >= 1e170 is 1.5-3% off and unverified",
)
def test_weibull_scale_1e170():
    """Observed at e3412770: ``Weibull.fit(x * 1e170)`` has alpha / 1e170
    0.98473 and beta 0.97142 times the unit fit's, ``maximum ==
    "unverified"``, and warns "did not reach a verified maximum" (x**beta
    overflows). Expected: the unit fit rescaled, verified, no warning.
    """
    x, *_ = _data()
    s = 1e170
    unit = Weibull.fit(x)
    m = _silent(Weibull.fit, x * s)
    np.testing.assert_allclose(
        [m.params[0] / s, m.params[1]], unit.params, rtol=1e-3
    )
    assert m.maximum == "verified"


@pytest.mark.xfail(
    strict=True,
    reason="a Weibull fit at scale 1e-200 is right but reported unverified",
)
def test_weibull_scale_1e_minus_200():
    """Observed at e3412770: ``Weibull.fit(x * 1e-200)`` gives the right
    parameters to 1e-5 but ``maximum == "unverified"``, with the warning.
    Expected: verified, no warning.
    """
    x, *_ = _data()
    m = _silent(Weibull.fit, x * 1e-200)
    assert m.maximum == "verified"


@pytest.mark.xfail(
    strict=True,
    reason="WeibullPH at scale 1e200 gives the 'not verified' warning twice",
)
def test_weibull_ph_unverified_warned_once():
    """Observed at e3412770: ``WeibullPH.fit(x * 1e200, Z, c)`` warns "The
    maximum-likelihood search did not reach a verified maximum" twice,
    from ``optimised_fit.py`` and from ``_fit_skeleton.py``. Expected:
    once (Design Principle 22: a thing is warned once).
    """
    x, c, Z = _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        WeibullPH.fit(x * 1e200, Z, c)
    found = [w for w in record if str(w.message).startswith(UNVERIFIED)]
    assert len(found) == 1
