"""Coverage of the mixture-cure regression's Wald intervals.

The truth: a unit with a binary covariate ``z1`` and a normal covariate
``z2`` is susceptible with probability

    pi(z) = expit(-0.5 + 1.0 z1 + 0.5 z2),

and a susceptible unit's life is Weibull with shape 2 and scale
``100 exp(-0.4 z1)``. ``n = 400`` units, each followed to a uniform time
on (300, 500), far past the lives, so the cured fraction shows as a
plateau. Fitted with the defaults: ``lfp_p ~ z1 + z2`` (logit) and
``alpha ~ z1 + z2``.

Wald ``param_cb`` of every parameter and Wald ``cb`` of the population's
survival at ``t = 60`` for a unit with ``z = (1, 0.3)``: 400 replicates.
The seed, replicates and slack (0.01, the suite's rule) were fixed
before any run.
"""

import os
import warnings

import numpy as np
from scipy.special import expit

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 400
REPS = 400
SEED = 13801
# In the model's order: alpha, beta, lfp_p, alpha_coef_0, alpha_coef_1,
# lfp_p_coef_0, lfp_p_coef_1 (the scale's coefficients with SurPyval's
# sign: alpha(z) = alpha exp(-b'z))
TRUTH = np.array([100.0, 2.0, expit(-0.5), 0.4, 0.0, 1.0, 0.5])
QUERY = (60.0, np.array([[1.0, 0.3]]))


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    z1 = rng.binomial(1, 0.5, N)
    z2 = rng.normal(size=N)
    fails = rng.random(N) < expit(-0.5 + 1.0 * z1 + 0.5 * z2)
    life = 100.0 * np.exp(-0.4 * z1) * rng.weibull(2.0, N)
    end = rng.uniform(300.0, 500.0, N)
    seen = fails & (life < end)
    return (
        np.where(seen, life, end),
        np.where(seen, 0, 1),
        np.column_stack([z1, z2]),
    )


def _true_sf():
    t, z = QUERY
    pi = expit(-0.5 + 1.0 * z[0, 0] + 0.5 * z[0, 1])
    scale = 100.0 * np.exp(-0.4 * z[0, 0])
    return float(1 - pi + pi * np.exp(-((t / scale) ** 2)))


def _wald(r):
    x, c, Z = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullMixtureCure.fit(x, Z, c=c)
        bounds = np.array([model.param_cb(n) for n in model.parameter_names])
        band = np.ravel(model.cb([QUERY[0]], QUERY[1]))
    return bounds[:, 0], bounds[:, 1], band


def test_wald_coverage_of_both_parts():
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        out = list(pool.map(_wald, range(REPS)))
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    check_coverage(lower, upper, TRUTH, 0.95, "WeibullMixtureCure Wald")
    band = np.array([o[2] for o in out])
    check_coverage(
        band[:, :1],
        band[:, 1:],
        _true_sf(),
        0.95,
        "WeibullMixtureCure Wald sf(60 | z = (1, 0.3))",
    )
