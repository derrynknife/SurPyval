"""Coverage of the interval-censored Cox model's Wald intervals, whose
standard errors are the profile likelihood's curvature.

The truth: ``n = 200`` units with a binary covariate ``z1`` and a normal
covariate ``z2``; a unit's life is Weibull proportional hazards,

    S(t | z) = exp(-(t / 10) ^ 1.5 exp(0.7 z1 - 0.4 z2)),

and it is inspected from a uniform(0, 2) time every uniform(1, 3) until
25. A failure before the first inspection is left censored at it, one
after the last right censored there, and every other failure is known
only to lie between two inspections: no exact times.

Wald ``param_cb`` (95%) of both coefficients from ``IntervalCoxPH.fit``
with its defaults, 400 replicates, and the profile likelihood-ratio
``param_cb(method="lr")`` on the same replicates. The seed, replicates and
slack (0.01, the suite's rule) were fixed before any run.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 200
REPS = 400
SEED = 20261113
TRUTH = np.array([0.7, -0.4])


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    z1 = rng.binomial(1, 0.5, N).astype(float)
    z2 = rng.normal(size=N)
    life = 10.0 * (rng.exponential(size=N) / np.exp(0.7 * z1 - 0.4 * z2)) ** (
        1 / 1.5
    )
    x = np.empty((N, 2))
    c = np.empty(N, dtype=int)
    for i in range(N):
        visits = np.cumsum(
            np.r_[rng.uniform(0.0, 2.0), rng.uniform(1.0, 3.0, 30)]
        )
        visits = visits[visits <= 25.0]
        if life[i] <= visits[0]:
            x[i], c[i] = visits[0], -1
        elif life[i] > visits[-1]:
            x[i], c[i] = visits[-1], 1
        else:
            k = np.searchsorted(visits, life[i]) - 1
            x[i], c[i] = (visits[k], visits[k + 1]), 2
    return x, c, np.column_stack([z1, z2])


def _bounds(args):
    r, method = args
    x, c, Z = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.IntervalCoxPH.fit(x, Z, c=c)
        bounds = np.array(
            [model.param_cb(n, method=method) for n in model.parameter_names]
        )
    return bounds[:, 0], bounds[:, 1]


def _check(method, label):
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_bounds, [(r, method) for r in range(REPS)]))
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    check_coverage(lower, upper, TRUTH, 0.95, label)


def test_wald_coverage():
    _check("wald", "IntervalCoxPH Wald (profile)")


def test_likelihood_ratio_coverage():
    _check("lr", "IntervalCoxPH profile likelihood ratio")
