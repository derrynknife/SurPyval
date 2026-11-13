"""Coverage of the zero-inflated regression's Wald intervals, by each of
its two routes.

The truth: a unit with a binary covariate ``z1`` and a normal covariate
``z2`` is dead on arrival (fails at time 0) with probability

    f0(z) = expit(-2.0 + 1.0 z1 + 0.5 z2),

and a unit that works has a Weibull life with shape 2 and scale
``100 exp(-0.4 z1)``, censored at a uniform time on (80, 300). ``n =
400`` units. Fitted with the defaults: ``f0 ~ z1 + z2`` (logit) and
``alpha ~ z1 + z2``.

Two-stage route: the data as drawn (every row says whether its unit was
dead on arrival). Joint route: the same draws, where a unit at a site
visited only at day 30 (each with probability 0.3) and found failed
there is recorded as an interval from 0 to 30, which couples the parts.

Wald ``param_cb`` of every parameter and Wald ``cb`` of the population's
survival at ``t = 60`` for a unit with ``z = (1, 0.3)``: 400 replicates
per route. The seed, replicates and slack (0.01, the suite's rule) were
fixed before any run.
"""

import os
import warnings

import numpy as np
from scipy.special import expit

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 400
REPS = 400
SEED = 20261112
# In the model's order: alpha, beta, f0, alpha_coef_0, alpha_coef_1,
# f0_coef_0, f0_coef_1 (the scale's coefficients with SurPyval's sign:
# alpha(z) = alpha exp(-b'z))
TRUTH = np.array([100.0, 2.0, expit(-2.0), 0.4, 0.0, 1.0, 0.5])
QUERY = (60.0, np.array([[1.0, 0.3]]))


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    z1 = rng.binomial(1, 0.5, N)
    z2 = rng.normal(size=N)
    doa = rng.random(N) < expit(-2.0 + 1.0 * z1 + 0.5 * z2)
    life = 100.0 * np.exp(-0.4 * z1) * rng.weibull(2.0, N)
    end = rng.uniform(80.0, 300.0, N)
    x = np.where(doa, 0.0, np.minimum(life, end))
    c = np.where(doa, 0, (life > end).astype(int))
    visited = rng.random(N) < 0.3
    return x, c, np.column_stack([z1, z2]), visited


def _true_sf():
    t, z = QUERY
    f0 = expit(-2.0 + 1.0 * z[0, 0] + 0.5 * z[0, 1])
    scale = 100.0 * np.exp(-0.4 * z[0, 0])
    return float((1 - f0) * np.exp(-((t / scale) ** 2)))


def _wald(args):
    r, route = args
    x, c, Z, visited = _sample(r)
    if route == "joint":
        early = visited & (c == 0) & (x <= 30.0)
        xx = np.column_stack([np.where(early, 0.0, x), np.where(early, 30, x)])
        x, c = xx, np.where(early, 2, c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullZeroInflated.fit(x, Z, c=c)
        assert model.fit_route == route
        bounds = np.array([model.param_cb(n) for n in model.parameter_names])
        band = np.ravel(model.cb([QUERY[0]], QUERY[1]))
    return bounds[:, 0], bounds[:, 1], band


def _check(route):
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        out = list(pool.map(_wald, [(r, route) for r in range(REPS)]))
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    label = f"WeibullZeroInflated ({route}) Wald"
    check_coverage(lower, upper, TRUTH, 0.95, label)
    band = np.array([o[2] for o in out])
    check_coverage(
        band[:, :1],
        band[:, 1:],
        _true_sf(),
        0.95,
        f"{label} sf(60 | z = (1, 0.3))",
    )


def test_wald_coverage_two_stage():
    _check("two-stage")


def test_wald_coverage_joint():
    _check("joint")
