"""Coverage of regression coefficient intervals.

- **Cox**: 95% Wald intervals from the observed information of the partial
  likelihood (the standard errors behind ``p_values``) and from the robust
  (Lin-Wei) covariance, n = 200, two covariates, 30% censoring; once with
  continuous times (Breslow) and once with times grouped onto a coarse grid
  (heavy ties, Efron's method).
- **Parametric families**: ``param_cb`` of every parameter of ``WeibullPH``,
  ``WeibullAFT``, ``LogNormalAFT``, ``WeibullPO`` and ``WeibullAH``, each
  simulated from its own model, n = 200.

Slack 0.01, except Cox with heavy ties (0.02): grouping the times makes the
partial likelihood an approximation to the grouped-data likelihood, Efron's
being the better one, and a small attenuation towards zero is expected.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_bias, check_coverage

N = 200
Z_CRIT = norm.ppf(0.975)


def _covariates(rng, n=N):
    return np.column_stack([rng.binomial(1, 0.5, n), rng.normal(0, 1, n)])


def _censor(t, rng, c_max):
    cens = rng.uniform(0, c_max, t.size)
    return np.minimum(t, cens), (cens < t).astype(int)


@pytest.mark.parametrize("tied", [False, True], ids=["continuous", "ties"])
def test_cox_coefficient_coverage(tied):
    rng = np.random.default_rng(301 + tied)
    beta = np.array([0.7, -0.5])
    reps = 1000
    lo, hi = np.empty((reps, 2)), np.empty((reps, 2))
    rlo, rhi = np.empty((reps, 2)), np.empty((reps, 2))
    ties = 0.0
    for r in range(reps):
        Z = _covariates(rng)
        # Weibull(10, 1.5) baseline, proportional hazards exp(beta'Z).
        t = 10.0 * (rng.exponential(size=N) / np.exp(Z @ beta)) ** (1 / 1.5)
        x, c = _censor(t, rng, 25.0)
        if tied:
            x = np.ceil(x)  # unit grid: about 20 distinct times
        ties += (1 - np.unique(x).size / N) / reps
        method = "efron" if tied else "breslow"
        model = sp.CoxPH.fit(x=x, Z=Z, c=c, tie_method=method)
        b = np.asarray(model.params)
        se = np.sqrt(np.diag(np.linalg.inv(model.jac(b)[1])))
        rse = np.sqrt(np.diag(model.robust_covariance()))
        lo[r], hi[r] = b - Z_CRIT * se, b + Z_CRIT * se
        rlo[r], rhi[r] = b - Z_CRIT * rse, b + Z_CRIT * rse
    print("tied rows: {:.0%}".format(ties))
    slack = 0.02 if tied else 0.01
    label = "CoxPH ({})".format("efron, ties" if tied else "breslow")
    check_coverage(lo, hi, beta, 0.95, label + " model-based", slack=slack)
    check_coverage(rlo, rhi, beta, 0.95, label + " robust", slack=slack)


def test_proportional_odds_coefficient_coverage():
    # The semi-parametric proportional odds NPMLE (#341): log-logistic
    # baseline odds (t / 10)^2, survival odds multiplied by exp(beta'Z),
    # about 50% censored. The intervals are param_cb's, from the profile
    # likelihood's information; the standard errors must match the spread.
    rng = np.random.default_rng(341)
    beta = np.array([1.0, -0.5])
    reps = 1000
    lo, hi = np.empty((reps, 2)), np.empty((reps, 2))
    est, se = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        Z = _covariates(rng)
        u = rng.uniform(size=N)
        t = 10.0 * (u / (1 - u) * np.exp(Z @ beta)) ** 0.5
        x, c = _censor(t, rng, 30.0)
        model = sp.ProportionalOdds.fit(x, Z, c=c)
        est[r], se[r] = model.beta, model.se
        bounds = [model.param_cb(name) for name in model.parameter_names]
        lo[r], hi[r] = np.array(bounds).T
    check_coverage(lo, hi, beta, 0.95, "ProportionalOdds param_cb")
    check_bias(est, beta, "ProportionalOdds", standard_errors=se)


def _ph(rng, Z, phi):
    # Weibull(10, 1.5) baseline hazard times phi.
    return 10.0 * (rng.exponential(size=Z.shape[0]) / phi) ** (1 / 1.5)


def _aft_weibull(rng, Z, phi):
    # H(x | Z) = H0(phi x): T = T0 / phi.
    return 10.0 * rng.weibull(1.5, Z.shape[0]) / phi


def _aft_lognormal(rng, Z, phi):
    return np.exp(rng.normal(2.0, 0.5, Z.shape[0])) / phi


def _po(rng, Z, phi):
    # S / F = phi S0 / F0: S0 = S / (phi (1 - S) + S), then T = S0^{-1}.
    u = rng.uniform(size=Z.shape[0])
    s0 = u / (phi * (1 - u) + u)
    return sp.Weibull.from_params([10.0, 1.5]).qf(1 - s0)


def _ah(rng, Z, beta):
    # h = h0(x) + beta'Z: solve H0(t) + (beta'Z) t = E by bisection.
    e = rng.exponential(size=Z.shape[0])
    add = Z @ beta
    lo, hi = np.zeros_like(e), np.full_like(e, 100.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = (mid / 10.0) ** 1.5 + add * mid - e
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


# fitter, simulator, baseline params, coefficients, whether the simulator
# takes phi = exp(beta'Z) (else beta itself), censoring limit, seed
FAMILIES = {
    "WeibullPH": (sp.WeibullPH, _ph, [10.0, 1.5], [0.7, -0.5], 1, 25, 311),
    "WeibullAFT": (
        sp.WeibullAFT,
        _aft_weibull,
        [10.0, 1.5],
        [0.7, -0.5],
        1,
        25,
        312,
    ),
    "LogNormalAFT": (
        sp.LogNormalAFT,
        _aft_lognormal,
        [2.0, 0.5],
        [0.7, -0.5],
        1,
        25,
        313,
    ),
    "WeibullPO": (sp.WeibullPO, _po, [10.0, 1.5], [0.7, -0.5], 1, 30, 314),
    # Additive: both covariates non-negative so the hazard stays positive.
    "WeibullAH": (sp.WeibullAH, _ah, [10.0, 1.5], [0.05, 0.03], 0, 25, 315),
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_parametric_regression_param_cb(name):
    fitter, simulate, base, coef, uses_phi, c_max, seed = FAMILIES[name]
    rng = np.random.default_rng(seed)
    coef = np.asarray(coef)
    truth = np.concatenate([base, coef])
    reps = 500
    k = truth.size
    lo, hi = np.empty((reps, k)), np.empty((reps, k))
    for r in range(reps):
        Z = _covariates(rng)
        if not uses_phi:
            Z = np.column_stack([Z[:, 0], np.abs(Z[:, 1])])
        t = simulate(rng, Z, np.exp(Z @ coef) if uses_phi else coef)
        x, c = _censor(t, rng, c_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit(x=x, Z=Z, c=c)
            names = model.parameter_names
            for j, p in enumerate(names):
                lo[r, j], hi[r, j] = model.param_cb(p)
    check_coverage(lo, hi, truth, 0.95, name + " param_cb")
