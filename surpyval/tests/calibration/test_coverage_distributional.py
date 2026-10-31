"""Coverage of the distributional regression's intervals.

The truth: a Weibull whose scale and shape both depend on a binary
covariate ``z1`` and whose scale also depends on a normal covariate
``z2``,

    alpha(z) = 10 exp(-0.5 z1 + 0.3 z2),    beta(z) = 1.5 exp(0.4 z1),

``n = 200`` units, right censored by uniform times on (0, 25) (about 30%
censored), fitted with ``alpha ~ z1 + z2`` and ``beta ~ z1``. And a
log-normal with ``mu = 2 - 0.4 z1`` and ``log sigma = log 0.6 + 0.5 z1``.

- Wald ``param_cb`` of every parameter and Wald ``cb`` of the survival at
  ``t = 8`` for a unit with ``z = (1, 0)``: 1000 replications each.
- Likelihood-ratio ``param_cb`` of the shape coefficient ``beta_z1`` and
  the scale coefficient ``alpha_z1``: 400 replications.
- Percentile bootstrap ``param_cb`` of ``beta_z1``, ``n_boot = 200``: 100
  replications (the cost: 20 000 refits).

The seeds, replications and slack were fixed before any run. Slack 0.01
for the Wald and likelihood-ratio intervals (the suite's rule), 0.02 for
the bootstrap, whose percentile interval at ``n_boot = 200`` has its own
Monte Carlo error in each replication's ends.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests._helpers import random_right_censoring
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 200
ALPHA = (10.0, -0.5, 0.3)
BETA = (1.5, 0.4)
# In the model's order: alpha, beta, then alpha_z1, alpha_z2 and beta_z1
# (named alpha_coef_0, alpha_coef_1 and beta_coef_0)
TRUTH_WEIBULL = np.array([ALPHA[0], BETA[0], ALPHA[1], ALPHA[2], BETA[1]])
COVARIATES = {"alpha": [0, 1], "beta": [0]}
QUERY = (8.0, np.array([[1.0, 0.0]]))
SEED_WALD, SEED_LR, SEED_BOOT, SEED_LOGNORMAL = 2810, 2811, 2812, 2813


def _weibull_sample(seed, r):
    rng = np.random.default_rng([seed, r])
    Z = np.column_stack([rng.binomial(1, 0.5, N), rng.normal(0, 1, N)])
    alpha = ALPHA[0] * np.exp(ALPHA[1] * Z[:, 0] + ALPHA[2] * Z[:, 1])
    beta = BETA[0] * np.exp(BETA[1] * Z[:, 0])
    t = alpha * rng.weibull(beta)
    x, c = random_right_censoring(t, rng, 25.0)
    return x, c, Z


def _true_sf():
    t, z = QUERY
    alpha = ALPHA[0] * np.exp(ALPHA[1] * z[0, 0] + ALPHA[2] * z[0, 1])
    beta = BETA[0] * np.exp(BETA[1] * z[0, 0])
    return float(np.exp(-((t / alpha) ** beta)))


def _fit(x, c, Z):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.WeibullDistributional.fit(x, Z, c=c, covariates=COVARIATES)


def _wald(r):
    model = _fit(*_weibull_sample(SEED_WALD, r))
    names = model.parameter_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bounds = np.array([model.param_cb(n) for n in names])
        band = np.ravel(model.cb([QUERY[0]], QUERY[1]))
    return bounds[:, 0], bounds[:, 1], band


def _lr(r):
    model = _fit(*_weibull_sample(SEED_LR, r))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(
            [
                np.ravel(model.param_cb(n, method="lr"))
                for n in ("beta_coef_0", "alpha_coef_0")
            ]
        )


def _boot(r):
    model = _fit(*_weibull_sample(SEED_BOOT, r))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.ravel(
            model.param_cb(
                "beta_coef_0", method="bootstrap", n_boot=200, random_state=r
            )
        )


def _map(fn, reps):
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        return list(pool.map(fn, range(reps)))


def test_wald_coverage_weibull_scale_and_shape():
    out = _map(_wald, 1000)
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    check_coverage(
        lower, upper, TRUTH_WEIBULL, 0.95, "WeibullDistributional Wald"
    )
    band = np.array([o[2] for o in out])
    check_coverage(
        band[:, :1],
        band[:, 1:],
        _true_sf(),
        0.95,
        "WeibullDistributional Wald sf(8 | z = (1, 0))",
    )


def test_likelihood_ratio_coverage_of_the_coefficients():
    out = np.array(_map(_lr, 400))
    check_coverage(
        out[:, :, 0],
        out[:, :, 1],
        np.array([BETA[1], ALPHA[1]]),
        0.95,
        "WeibullDistributional LR (beta_z1, alpha_z1)",
    )


def test_bootstrap_coverage_of_the_shape_coefficient():
    out = np.array(_map(_boot, 100))
    check_coverage(
        out[:, :1],
        out[:, 1:],
        BETA[1],
        0.95,
        "WeibullDistributional bootstrap beta_z1",
        slack=0.02,
    )


def _lognormal(r):
    rng = np.random.default_rng([SEED_LOGNORMAL, r])
    z = rng.binomial(1, 0.5, N).astype(float)
    mu = 2.0 - 0.4 * z
    sigma = 0.6 * np.exp(0.5 * z)
    t = np.exp(rng.normal(mu, sigma))
    x, c = random_right_censoring(t, rng, 25.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.LogNormalDistributional.fit(x, z, c=c)
        bounds = np.array([model.param_cb(n) for n in model.parameter_names])
    return bounds[:, 0], bounds[:, 1]


def test_wald_coverage_lognormal_location_and_spread():
    out = _map(_lognormal, 1000)
    check_coverage(
        np.array([o[0] for o in out]),
        np.array([o[1] for o in out]),
        np.array([2.0, 0.6, -0.4, 0.5]),
        0.95,
        "LogNormalDistributional Wald",
    )
