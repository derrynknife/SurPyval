"""Standard errors and bounds of the recurrent likelihood fits under a
change of the time unit.

Writing times in another unit (``t' = k t``) is a reparameterisation:
the likelihood at corresponding parameters differs by a constant, so the
observed information transforms with the Jacobian of the parameter map
and every bound on an expected count is unchanged. To test the
covariance and the bounds alone, not the optimiser's tolerance (the fits
agree to about 1e-5 across units), each model fitted in the new unit is
placed at exactly the estimates the fit in months implies there.
"""

import math
import warnings

import numpy as np
import pytest

from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CauseSpecificNHPP,
    CauseSpecificProportionalIntensity,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.tests._helpers import (
    REPAIR_FLEET_C,
    REPAIR_FLEET_I,
    REPAIR_FLEET_X,
)

# Months as hours, as seconds and as years.
UNITS = [730.0, 730.0 * 3600.0, 1.0 / 12.0]
UNIT_IDS = ["hours", "seconds", "years"]
RTOL = 1e-6

X = np.array(REPAIR_FLEET_X, dtype=float)
Z = np.array([0.0] * 6 + [1.0] * 5)
E = np.array(["A", "B", "A", "B", "A", "A", "B", "A", "B", "A", "A"])
T = np.array([10.0, 30.0, 59.0])


def _baseline_map(name, theta, k):
    """The baseline's parameters in the new unit, and the Jacobian of
    the map from the old ones."""
    theta = np.asarray(theta, dtype=float)
    if name == "HPP":
        return theta / k, np.diag([1.0 / k])
    if name == "Crow-AMSAA":  # cif = (t / alpha) ** beta
        return theta * [k, 1.0], np.diag([k, 1.0])
    if name == "Duane":  # cif = b t ** a
        a, b = theta
        return np.array([a, b * k**-a]), np.array(
            [[1.0, 0.0], [-b * k**-a * math.log(k), k**-a]]
        )
    if name == "Cox-Lewis":  # cif = exp(alpha) expm1(beta t) / beta
        alpha, beta = theta
        return np.array([alpha - math.log(k), beta / k]), np.diag(
            [1.0, 1.0 / k]
        )
    raise ValueError(name)


def _map(name, theta, k, n_coeffs=0):
    """Baseline then coefficients (which do not move)."""
    theta = np.asarray(theta, dtype=float)
    n_base = theta.size - n_coeffs
    base, J = _baseline_map(name, theta[:n_base], k)
    full = np.eye(theta.size)
    full[:n_base, :n_base] = J
    return np.r_[base, theta[n_base:]], full


def _place(model, theta, n_coeffs=0):
    """Put a fitted model at ``theta`` (what it reports and predicts
    with)."""
    model._mle = np.asarray(theta, dtype=float)
    if not hasattr(model, "dist"):
        return  # a renewal model reads ``_mle`` alone
    n_base = theta.size - n_coeffs
    model.params = model._mle[:n_base].copy()
    if n_coeffs:
        model.coeffs = model._mle[n_base:].copy()


def _fit_pair(fit, k):
    with warnings.catch_warnings():
        # HPP in seconds: its own fit stops short (a known, separate
        # issue); the placement below is what is tested.
        warnings.simplefilter("ignore")
        return fit(1.0), fit(k)


def _stationary(model, n_coeffs=0):
    """Polish a fit to a point where the gradient is zero to rounding.

    The observed information transforms with the Jacobian alone only
    where the gradient is zero; a fit stops at a gradient of about 1e-3
    (its tolerance), which through a nonlinear map (Duane's ``b k**-a``)
    adds ``g . d2T`` to the Hessian. A few Newton steps, with a
    fourth-order gradient in steps of the standard errors, remove it. A
    parameter with no variance (a renewal model's restoration parameter
    on the edge of its range) is held."""
    theta = np.asarray(model._mle, dtype=float).copy()
    for _ in range(4):
        cov = model.covariance()
        free = np.isfinite(np.diag(cov))
        h = 1e-3 * np.sqrt(np.diag(cov))
        g = np.zeros(theta.size)
        for i in np.flatnonzero(free):
            e = np.zeros(theta.size)
            e[i] = h[i]
            f = [model._neg_ll(theta + m * e) for m in (2, 1, -1, -2)]
            g[i] = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h[i])
        block = np.ix_(free, free)
        theta[free] = theta[free] - cov[block] @ g[free]
        _place(model, theta, n_coeffs)
    return model


def _assert_covariance_transforms(ref, model, J):
    # On the correlation scale, so that a covariance that is zero to
    # rounding is compared against the variances, not against itself.
    # A parameter with no variance (nan row and column) stays so.
    cov = ref.covariance()
    held = np.isnan(np.diag(cov))
    expected = J @ np.where(np.isnan(cov), 0.0, cov) @ J.T
    expected[held, :] = np.nan
    expected[:, held] = np.nan
    sd = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(
        model.covariance() / np.outer(sd, sd),
        expected / np.outer(sd, sd),
        rtol=RTOL,
        atol=RTOL,
    )
    np.testing.assert_allclose(model.standard_errors(), sd, rtol=RTOL)


BASELINES = {
    "HPP": lambda k: HPP.fit(X * k, REPAIR_FLEET_I, REPAIR_FLEET_C),
    "Crow-AMSAA": lambda k: CrowAMSAA.fit(
        X * k, REPAIR_FLEET_I, REPAIR_FLEET_C
    ),
    "Duane": lambda k: Duane.fit(X * k, REPAIR_FLEET_I, REPAIR_FLEET_C),
    "Cox-Lewis": lambda k: CoxLewis.fit(X * k, REPAIR_FLEET_I, REPAIR_FLEET_C),
}
# The parameters that the map leaves as they are.
UNITLESS = {"Crow-AMSAA": ("beta",), "Duane": ("alpha",)}
NHPP_BASELINES = {
    "Crow-AMSAA": CrowAMSAA,
    "Duane": Duane,
    "Cox-Lewis": CoxLewis,
}


@pytest.mark.parametrize("k", UNITS, ids=UNIT_IDS)
@pytest.mark.parametrize("name", list(BASELINES))
def test_poisson_process_bounds_rescale_exactly(name, k):
    ref, model = _fit_pair(BASELINES[name], k)
    _stationary(ref)
    theta, J = _map(name, ref._mle, k)
    _place(model, theta)
    _assert_covariance_transforms(ref, model, J)
    # An expected count has no unit; an intensity is per unit of time.
    np.testing.assert_allclose(model.cif_cb(T * k), ref.cif_cb(T), rtol=RTOL)
    np.testing.assert_allclose(
        model.iif_cb(T * k) * k, ref.iif_cb(T), rtol=RTOL
    )
    np.testing.assert_allclose(
        model.mtbf_cb(T * k) / k, ref.mtbf_cb(T), rtol=RTOL
    )
    # A parameter that does not carry the unit keeps its interval.
    for p in UNITLESS.get(name, ()):
        np.testing.assert_allclose(
            model.param_cb(p), ref.param_cb(p), rtol=RTOL
        )


def _interior_renewal_data():
    """Eight items whose Kijima-I fit has q = 2.6, inside its range."""
    rng = np.random.default_rng(8)
    rows = []
    for k in range(8):
        end = rng.uniform(6000, 12000)
        n = rng.poisson(2e-4 * end**1.35)
        ts = np.sort(end * rng.random(n) ** (1 / 1.35))
        rows += [(t, k, 0) for t in ts] + [(end, k, 1)]
    x, i, c = map(np.array, zip(*rows))
    return x / 1000.0, i, c


RENEWALS = {
    # (fitter, data, whether the restoration parameter is on its edge)
    "GeneralizedOneRenewal": (
        GeneralizedOneRenewal,
        (X, REPAIR_FLEET_I, REPAIR_FLEET_C),
        False,
    ),
    "GeneralizedRenewal-interior": (
        GeneralizedRenewal,
        _interior_renewal_data(),
        False,
    ),
    "GeneralizedRenewal-edge": (
        GeneralizedRenewal,
        (X, REPAIR_FLEET_I, REPAIR_FLEET_C),
        True,
    ),
    "ARA-edge": (ARA, (X, REPAIR_FLEET_I, REPAIR_FLEET_C), True),
    "ARI-edge": (ARI, (X, REPAIR_FLEET_I, REPAIR_FLEET_C), True),
}


@pytest.mark.parametrize("k", UNITS, ids=UNIT_IDS)
@pytest.mark.parametrize("case", list(RENEWALS))
def test_renewal_covariance_rescales_exactly(case, k):
    # [q or rho, alpha, beta] of a Weibull: only alpha carries the unit.
    # On the edge the restoration parameter has no variance and the
    # others' covariance is that of the model with it held there.
    fitter, (x, i, c), on_edge = RENEWALS[case]
    ref, model = _fit_pair(lambda k: fitter.fit(x * k, i, c=c), k)
    assert (ref._edge_value() is not None) == on_edge
    _stationary(ref)
    J = np.diag([1.0, k, 1.0])
    model._mle = J @ np.asarray(ref._mle, dtype=float)
    _assert_covariance_transforms(ref, model, J)
    restoration = model.parameter_names[0]
    names = ["beta"] if on_edge else [restoration, "beta"]
    for p in names:
        np.testing.assert_allclose(
            model.param_cb(p), ref.param_cb(p), rtol=RTOL
        )
    np.testing.assert_allclose(
        model.param_cb("alpha"), ref.param_cb("alpha") * k, rtol=RTOL
    )


def _regressions():
    out = {
        "PI-HPP": (
            "HPP",
            lambda k: ProportionalIntensityHPP.fit(
                X * k, Z, REPAIR_FLEET_I, REPAIR_FLEET_C
            ),
        )
    }
    for name, dist in NHPP_BASELINES.items():
        out["PI-NHPP " + name] = (
            name,
            lambda k, dist=dist: ProportionalIntensityNHPP.fit(
                X * k, Z, REPAIR_FLEET_I, REPAIR_FLEET_C, baseline=dist
            ),
        )
    return out


REGRESSIONS = _regressions()


@pytest.mark.parametrize("k", UNITS, ids=UNIT_IDS)
@pytest.mark.parametrize("case", list(REGRESSIONS))
def test_proportional_intensity_bounds_rescale_exactly(case, k):
    name, fit = REGRESSIONS[case]
    ref, model = _fit_pair(fit, k)
    _stationary(ref, n_coeffs=1)
    theta, J = _map(name, ref._mle, k, n_coeffs=1)
    _place(model, theta, n_coeffs=1)
    _assert_covariance_transforms(ref, model, J)
    for z in ([0.0], [1.0]):
        np.testing.assert_allclose(
            model.cif_cb(T * k, z), ref.cif_cb(T, z), rtol=RTOL
        )
        np.testing.assert_allclose(
            model.iif_cb(T * k, z) * k, ref.iif_cb(T, z), rtol=RTOL
        )


@pytest.mark.parametrize("k", UNITS, ids=UNIT_IDS)
@pytest.mark.parametrize("name", list(NHPP_BASELINES))
def test_cause_specific_bounds_rescale_exactly(name, k):
    ref, model = _fit_pair(
        lambda k: CauseSpecificNHPP.fit(
            X * k,
            REPAIR_FLEET_I,
            REPAIR_FLEET_C,
            e=E,
            dist=NHPP_BASELINES[name],
        ),
        k,
    )
    for cause in ref.event_types:
        _stationary(ref.models[cause])
        theta, J = _map(name, ref.models[cause]._mle, k)
        _place(model.models[cause], theta)
        _assert_covariance_transforms(
            ref.models[cause], model.models[cause], J
        )
        np.testing.assert_allclose(
            model.cif_cb(T * k, event=cause),
            ref.cif_cb(T, event=cause),
            rtol=RTOL,
        )
    # The total over the types adds their delta-method variances.
    np.testing.assert_allclose(
        model.total_cif_cb(T * k), ref.total_cif_cb(T), rtol=RTOL
    )
    np.testing.assert_allclose(
        model.total_iif_cb(T * k) * k, ref.total_iif_cb(T), rtol=RTOL
    )


@pytest.mark.parametrize("k", UNITS, ids=UNIT_IDS)
@pytest.mark.parametrize("name", list(NHPP_BASELINES))
def test_cause_specific_regression_bounds_rescale_exactly(name, k):
    ref, model = _fit_pair(
        lambda k: CauseSpecificProportionalIntensity.fit(
            X * k,
            Z,
            i=REPAIR_FLEET_I,
            c=REPAIR_FLEET_C,
            e=E,
            baseline=NHPP_BASELINES[name],
        ),
        k,
    )
    for cause in ref.event_types:
        _stationary(ref.models[cause], n_coeffs=1)
        theta, J = _map(name, ref.models[cause]._mle, k, n_coeffs=1)
        _place(model.models[cause], theta, n_coeffs=1)
        _assert_covariance_transforms(
            ref.models[cause], model.models[cause], J
        )
        np.testing.assert_allclose(
            model.cif_cb(T * k, [1.0], cause),
            ref.cif_cb(T, [1.0], cause),
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            model.iif_cb(T * k, [1.0], cause) * k,
            ref.iif_cb(T, [1.0], cause),
            rtol=RTOL,
        )
