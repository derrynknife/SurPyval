"""Standard errors and bounds of the non-recurrent families under a change
of the time unit.

Writing times in another unit (``t' = k t``) is a reparameterisation, so
a bound on a probability is unchanged and a standard error moves with its
parameter's unit. Finite-difference steps taken as a fraction of each
parameter with a fixed floor (``1e-2`` or ``1``) are not: a parameter
that is small in the new unit (a rate per second, an additive hazard's
coefficient) is stepped by a large part of itself, and its standard
error, and every bound through it, moves with the unit. Each test below
fits a fixture in its own unit and in another, and compares what should
not move; the tolerance is above the optimiser's (the two fits agree to
about 1e-7 here, the gamma process and the frailty fits to about 2e-5)
and below what the old steps gave, which each test notes.
"""

import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.checks import rescaled, rescaled_case
from surpyval.tests.conformance.registry import CASE_BY_NAME

SECONDS = 3600.0  # the fixtures' unit as hours
YEARS = 1.0 / 8760.0


def _fit(name, k):
    case = CASE_BY_NAME[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(rescaled(case, case.data(), k))


def _z(case):
    Z = np.asarray(case.Z, dtype=float)
    return Z[0] if Z.ndim > 1 else Z


def _width(bounds):
    bounds = np.asarray(bounds, dtype=float)
    return bounds[..., 1] - bounds[..., 0]


# -- parametric regressions: the delta method ----------------------------
# Old steps (seconds): WeibullAH 1.5e-4, LogNormalAH 4.7e-5, NormalAH
# 2.3e-5, LogisticAH 1.7e-5 off on the width of the Hf band;
# WeibullAL[InversePower] 1.6e-5. Times in millionths of the unit:
# WeibullAL[Exponential] 2.9e-4, WeibullAL[DualExponential] 7.3e-4; in
# a million units WeibullAL[InverseExponential] 1.6e-2.
REGRESSIONS = [
    ("WeibullAH", SECONDS, 1e-5),
    ("LogNormalAH", SECONDS, 1e-5),
    ("NormalAH", SECONDS, 1e-5),
    ("LogisticAH", SECONDS, 1e-5),
    ("WeibullAL[InversePower]", SECONDS, 1e-6),
    ("WeibullAL[Exponential]", 1e-6, 1e-6),
    ("WeibullAL[DualExponential]", 1e-6, 1e-6),
    ("WeibullAL[InverseExponential]", 1e6, 1e-4),
]


@pytest.mark.parametrize(
    "name, k, rtol", REGRESSIONS, ids=[r[0] for r in REGRESSIONS]
)
def test_regression_wald_bounds_do_not_depend_on_the_unit(name, k, rtol):
    case = CASE_BY_NAME[name]
    z = _z(case)
    ref, model = _fit(name, 1.0), _fit(name, k)
    for on, power in (("sf", 0), ("Hf", 0), ("hf", 1)):
        np.testing.assert_allclose(
            _width(model.cb(case.x * k, z, on=on)) * k**power,
            _width(ref.cb(case.x, z, on=on)),
            rtol=rtol,
            err_msg=on,
        )


def test_regression_robust_bounds_do_not_depend_on_the_unit():
    # The rows' scores of the cluster-robust covariance were differenced
    # in steps floored at 1e-2 of a parameter: with times in a million
    # units the band was nan.
    name = "WeibullAH"
    case = CASE_BY_NAME[name]
    z = _z(case)
    cluster = np.arange(len(case.data()["x"])) % 5
    ref, model = _fit(name, 1.0), _fit(name, 1e6)
    np.testing.assert_allclose(
        _width(model.cb(case.x * 1e6, z, robust=True, cluster=cluster)),
        _width(ref.cb(case.x, z, robust=True, cluster=cluster)),
        rtol=1e-5,
    )


def test_regression_quantile_bounds_do_not_depend_on_the_unit():
    name = "WeibullAH"
    case = CASE_BY_NAME[name]
    z = _z(case)
    ref, model = _fit(name, 1.0), _fit(name, SECONDS)
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        model.quantile_cb(p, z) / SECONDS, ref.quantile_cb(p, z), rtol=1e-5
    )


# -- Royston-Parmar: the information --------------------------------------
@pytest.mark.parametrize("k", [SECONDS, YEARS], ids=["seconds", "years"])
def test_royston_parmar_standard_errors_do_not_depend_on_the_unit(k):
    # The spline coefficients after the intercept do not carry the unit
    # (the knots move with log time). Old steps: 1.7e-4 off in seconds.
    ref, model = _fit("RoystonParmar", 1.0), _fit("RoystonParmar", k)
    np.testing.assert_allclose(
        model.standard_errors()[1:], ref.standard_errors()[1:], rtol=1e-5
    )
    case = CASE_BY_NAME["RoystonParmar"]
    np.testing.assert_allclose(
        _width(model.cb(case.x * k, on="Hf")),
        _width(ref.cb(case.x, on="Hf")),
        rtol=1e-5,
    )


# -- frailty: the numerical information at a variance of 0 -----------------
@pytest.mark.parametrize("name", ["ExponentialFrailty", "GammaFrailty"])
def test_frailty_standard_errors_do_not_depend_on_the_unit(name):
    # The frailty variance is at 0, where the exact Hessian is not used.
    # The baseline's rate (per second, 2.4e-5 and 1e-4) was stepped by
    # 1e-5: its relative standard error was 37% and 7% off.
    ref, model = _fit(name, 1.0), _fit(name, SECONDS)
    est_ref = np.r_[ref.dist_params, ref.beta]
    est = np.r_[model.dist_params, model.beta]
    n = est.size
    np.testing.assert_allclose(
        model.standard_errors()[:n] / np.abs(est),
        ref.standard_errors()[:n] / np.abs(est_ref),
        rtol=1e-4,
    )


# -- degradation -----------------------------------------------------------
@pytest.mark.parametrize("name", ["WienerProcess", "GammaProcess"])
def test_process_model_bounds_do_not_depend_on_the_unit(name):
    # Old: the sf band 1.8e-3 (Wiener) and 7.6e-3 (gamma) off in seconds.
    case = CASE_BY_NAME[name]
    ref, model = _fit(name, 1.0), _fit(name, SECONDS)
    query = rescaled_case(case, SECONDS)
    np.testing.assert_allclose(
        _width(model.cb(case.x * SECONDS, **query.call_kwargs)),
        _width(ref.cb(case.x, **case.call_kwargs)),
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "name", ["DegradationAnalysis[linear]", "DegradationAnalysis[power]"]
)
def test_degradation_analysis_bounds_do_not_depend_on_the_unit(name):
    # The two-stage (analytic) bounds: in a million units the old steps
    # gave a band 1% (linear) and 98 times (power) off, and nan.
    case = CASE_BY_NAME[name]
    k = 1e6
    ref, model = _fit(name, 1.0), _fit(name, k)
    for on in ("sf", "Hf"):
        np.testing.assert_allclose(
            _width(model.cb(case.x * k, on=on)),
            _width(ref.cb(case.x, on=on)),
            rtol=1e-6,
            err_msg=on,
        )


# -- copulas ---------------------------------------------------------------
COPULAS = ["ClaytonCopula", "GumbelCopula", "FrankCopula", "GaussianCopula"]


@pytest.mark.parametrize("k", [SECONDS, 1e-6], ids=["seconds", "millionths"])
@pytest.mark.parametrize("name", COPULAS)
def test_copula_standard_errors_do_not_depend_on_the_unit(name, k):
    # With every parameter's: each margin's scale (the second of its
    # two) carries the unit. Old steps: up to 1.2e-4 off in seconds and
    # 6.4e-3 in millionths of the unit.
    ref, model = _fit(name, 1.0), _fit(name, k)
    se = np.asarray(model.standard_errors(margins=True), dtype=float)
    se_ref = np.asarray(ref.standard_errors(margins=True), dtype=float)
    unit = np.ones(se.size)
    unit[[1, 3]] = k
    np.testing.assert_allclose(se / unit, se_ref, rtol=1e-6)


@pytest.mark.parametrize("name", COPULAS)
def test_copula_bounds_do_not_depend_on_the_unit(name):
    case = CASE_BY_NAME[name]
    k = 1e-6
    ref, model = _fit(name, 1.0), _fit(name, k)
    X = np.asarray(case.x, dtype=float)[:5]
    np.testing.assert_allclose(
        model.cb(X * k, on="sf"), ref.cb(X, on="sf"), rtol=1e-6
    )
