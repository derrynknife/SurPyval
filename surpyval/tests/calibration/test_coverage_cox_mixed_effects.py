"""Coverage and size of the mixed-effects Cox model's inference.

The truth: ``G = 40`` groups of 5 units; each unit has a binary covariate
(probability 0.5) with coefficient ``BETA = 0.6`` and its group's random
intercept, normal with standard deviation 0.7 (variance 0.49) on the
log-hazard; the baseline is exponential with rate 0.1, and each unit is
censored at a uniform (0, 15) time, which leaves about a third censored.

* Coverage, over 400 replicates: the Wald ``param_cb`` of the coefficient,
  the profile-likelihood interval of the variance (``variance_cb``), and
  the Wald interval of ``theta`` from ``CoxFrailty.fit(...,
  family="lognormal")`` (on the log scale, its standard error from the
  profile's curvature).
* Size, over 1000 replicates with no group effect (variance 0): the
  likelihood-ratio test of the random intercept (``lr_test``, the 50:50
  mixture of chi-square(0) and chi-square(1)) at the 5% level.

The seeds, replicates, sizes and slack (0.01, the suite's rule) were fixed
before any run.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    check_rate,
    process_pool,
)

G, PER = 40, 5
BETA, SD = 0.6, 0.7
REPS, REPS_SIZE = 400, 1000
SEED, SEED_SIZE = 31151, 31152


def _sample(rng, sd):
    g = np.repeat(np.arange(G), PER)
    z = rng.binomial(1, 0.5, g.size).astype(float)
    b = rng.normal(0.0, sd, G)[g]
    t = rng.exponential(1.0 / (0.1 * np.exp(BETA * z + b)))
    u = rng.uniform(0.0, 15.0, g.size)
    return np.minimum(t, u), z[:, None], (u < t).astype(int), g


def _coverage(r):
    x, Z, c, g = _sample(np.random.default_rng([SEED, r]), SD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxMixedEffects.fit(x, Z=Z, c=c, groups=g)
        beta = model.param_cb("coef_0")
        variance = model.variance_cb().iloc[0][["lower", "upper"]]
        frailty = sp.CoxFrailty.fit(x, Z=Z, c=c, groups=g, family="lognormal")
        try:
            theta = frailty.param_cb("theta")
        except ValueError:
            theta = np.array([np.nan, np.nan])
    return beta, np.asarray(variance, dtype=float), np.ravel(theta)


def _size(r):
    x, Z, c, g = _sample(np.random.default_rng([SEED_SIZE, r]), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxMixedEffects.fit(x, Z=Z, c=c, groups=g)
    return model.lr_test()["p_value"]


def test_coverage_of_the_coefficient_and_the_variance():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_coverage, range(REPS)))
    beta = np.array([o[0] for o in out])
    variance = np.array([o[1] for o in out])
    theta = np.array([o[2] for o in out])
    check_coverage(
        beta[:, :1], beta[:, 1:], BETA, 0.95, "CoxMixedEffects Wald beta"
    )
    check_coverage(
        variance[:, :1],
        variance[:, 1:],
        SD**2,
        0.95,
        "CoxMixedEffects profile-likelihood variance",
    )
    check_coverage(
        theta[:, :1],
        theta[:, 1:],
        SD**2,
        0.95,
        "CoxFrailty lognormal Wald theta",
    )


def test_size_of_the_random_effect_test():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        p = np.array(list(pool.map(_size, range(REPS_SIZE))))
    check_rate(
        int(np.sum(p < 0.05)),
        REPS_SIZE,
        0.05,
        "CoxMixedEffects lr_test size at 5%",
    )
