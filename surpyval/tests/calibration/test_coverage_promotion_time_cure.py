"""Coverage of the promotion-time cure regression's Wald intervals.

The truth: a unit with a binary covariate ``z1`` and a normal covariate
``z2`` carries a Poisson number of latent causes with mean

    theta(z) = 0.6 exp(0.7 z1 - 0.4 z2),

each with a Weibull promotion time (scale 12, shape 1.6); the unit fails
at the first and never if it has none, so its survival is
``exp(-theta(z) F0(t))`` and its cure fraction ``exp(-theta(z))``.
``n = 500`` units, each followed to a uniform time on (30, 60), far
past the lives, so the cured fraction shows as a plateau. Fitted with
the defaults: ``theta ~ z1 + z2`` (log link).

Wald ``param_cb`` of every parameter and Wald ``cb`` of the population's
survival at ``t = 10`` for a unit with ``z = (1, 0.3)``: 400 replicates.
The seed, replicates and slack (0.01, the suite's rule) were fixed
before any run.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 500
REPS = 400
SEED = 13802
# In the model's order: alpha, beta, theta, theta_coef_0, theta_coef_1
TRUTH = np.array([12.0, 1.6, 0.6, 0.7, -0.4])
QUERY = (10.0, np.array([[1.0, 0.3]]))


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    z1 = rng.binomial(1, 0.5, N)
    z2 = rng.normal(size=N)
    k = rng.poisson(0.6 * np.exp(0.7 * z1 - 0.4 * z2))
    life = np.array(
        [(12.0 * rng.weibull(1.6, j)).min() if j else np.inf for j in k]
    )
    end = rng.uniform(30.0, 60.0, N)
    seen = life < end
    return (
        np.where(seen, life, end),
        np.where(seen, 0, 1),
        np.column_stack([z1, z2]),
    )


def _true_sf():
    t, z = QUERY
    theta = 0.6 * np.exp(0.7 * z[0, 0] - 0.4 * z[0, 1])
    return float(np.exp(-theta * -np.expm1(-((t / 12.0) ** 1.6))))


def _wald(r):
    x, c, Z = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullPromotionTimeCure.fit(x, Z, c=c)
        bounds = np.array([model.param_cb(n) for n in model.parameter_names])
        band = np.ravel(model.cb([QUERY[0]], QUERY[1]))
    return bounds[:, 0], bounds[:, 1], band


def run(reps=REPS):
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        out = list(pool.map(_wald, range(reps)))
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    band = np.array([o[2] for o in out])
    return lower, upper, band


def test_wald_coverage():
    lower, upper, band = run()
    check_coverage(lower, upper, TRUTH, 0.95, "WeibullPromotionTimeCure Wald")
    check_coverage(
        band[:, :1],
        band[:, 1:],
        _true_sf(),
        0.95,
        "WeibullPromotionTimeCure Wald sf(10 | z = (1, 0.3))",
    )
