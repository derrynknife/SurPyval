"""
``fit_from_ecdf`` refuses a curve that falls as ``x`` rises.

A CDF cannot fall, but a survival curve ``R`` passed where ``F`` belongs
(``NonParametric.fit_from_ecdf`` takes ``R``) was fitted silently, to a
negative shape: ``Weibull.fit_from_ecdf([1, 2, 3, 4], [0.9, 0.7, 0.4,
0.1])`` gave beta -2.05. Now every family refuses it, saying to pass
``1 - R``. A curve that rises is fitted exactly as before.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.fitters.mpp import mpp_from_ecfd
from surpyval.univariate.parametric.optimised_fit import _ECDF_FALL_TOLERANCE

FAMILIES = [
    getattr(sp, name)
    for name in (
        "Beta",
        "Beta4",
        "BetaGeometric",
        "DiscreteWeibull",
        "ExpoWeibull",
        "Exponential",
        "Galton",
        "Gamma",
        "Gauss",
        "Geometric",
        "Gumbel",
        "GumbelLEV",
        "LogLogistic",
        "LogNormal",
        "Logistic",
        "NegativeBinomial",
        "Normal",
        "Poisson",
        "Rayleigh",
        "Uniform",
        "Weibull",
    )
]
FITS = [f for f in FAMILIES if f.supports_mpp and hasattr(f, "unpack_rr")]

X = np.array([1.0, 2.0, 3.0, 4.0])
R = np.array([0.9, 0.7, 0.4, 0.1])
# Kept inside (0, 1) for the families on a bounded support.
X01 = np.array([0.2, 0.4, 0.6, 0.8])


def _x(fitter):
    return X01 if fitter.name in ("Beta",) else X


@pytest.mark.parametrize("fitter", FITS, ids=lambda f: f.name)
def test_a_survival_curve_is_refused(fitter):
    with pytest.raises(ValueError, match=r"F must rise with x.*F = 1 - R"):
        fitter.fit_from_ecdf(_x(fitter), R)


@pytest.mark.parametrize("fitter", FITS, ids=lambda f: f.name)
def test_a_rising_curve_fits_as_before(fitter):
    x, F = _x(fitter), 1 - R
    try:
        before = mpp_from_ecfd(fitter, x, F)["params"]
    except Exception:  # the family cannot fit these points either way
        pytest.skip("no fit on these points")
    model = fitter.fit_from_ecdf(x, F)
    assert np.array_equal(model.params, np.array(before))


def test_the_message_says_where_it_falls():
    with pytest.raises(ValueError) as info:
        sp.Weibull.fit_from_ecdf(X, R)
    assert "falls from 0.9 at x = 1 to 0.7 at x = 2" in str(info.value)
    assert "falls overall" in str(info.value)


def test_a_single_dip_is_refused():
    with pytest.raises(ValueError, match=r"1 fall\(s\) in all"):
        sp.Weibull.fit_from_ecdf(X, [0.1, 0.3, 0.25, 0.9])


def test_order_of_the_points_does_not_matter():
    shuffled = sp.Weibull.fit_from_ecdf([4, 1, 3, 2], [0.9, 0.1, 0.6, 0.3])
    ordered = sp.Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, 0.6, 0.9])
    # The least-squares sum runs in the order given, so the last digits
    # may differ (as before this check).
    np.testing.assert_allclose(shuffled.params, ordered.params, rtol=1e-12)


def test_flat_steps_and_rounding_are_accepted():
    # A flat run (an ECDF between failures) and a fall within rounding.
    F = np.array([0.1, 0.3, 0.3, 0.6, 0.6 - _ECDF_FALL_TOLERANCE / 2, 0.9])
    sp.Weibull.fit_from_ecdf(np.arange(1.0, 7.0), F)


def test_tied_x_with_rising_F_is_accepted():
    sp.Weibull.fit_from_ecdf([1, 2, 2, 3], [0.1, 0.3, 0.4, 0.8])


def test_values_outside_0_1_are_still_refused():
    with pytest.raises(ValueError, match=r"F must lie in \[0, 1\]"):
        sp.Weibull.fit_from_ecdf(X, [0.1, 0.3, 1.2, 0.9])


@pytest.mark.parametrize(
    "estimator", [sp.KaplanMeier, sp.NelsonAalen, sp.FlemingHarrington]
)
def test_fit_from_non_parametric_still_works(estimator):
    rng = np.random.default_rng(3)
    x = rng.weibull(1.5, 40) * 10
    c = (rng.random(40) < 0.3).astype(int)
    model = estimator.fit(x, c)
    sp.Weibull.fit_from_non_parametric(model)
