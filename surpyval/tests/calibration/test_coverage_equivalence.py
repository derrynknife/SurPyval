"""Coverage of the equivalent-time-scale model's intervals, and the bias
of its ``k``.

The truth: a fleet of ``N = 300`` units, each run at its own constant
number of starts per hour, drawn from (0, 0.02, 0.1, 0.5); one start is
worth ``k = 10`` hours, and a unit's life on the combined clock
``hours + 10 starts`` is Weibull with scale 2000 and shape 2. Each unit
is followed for a uniform (500, 3000) hours, so about a third are still
running at the end.

Checked over 400 replicates: the Wald ``param_cb`` of every parameter
(``alpha``, ``beta``, ``k``), the profile-likelihood ``param_cb("k",
method="lr")``, the Wald band ``cb_at_usage`` on the survival at 400
hours for a unit making 0.5 starts per hour, and ``quantile_cb_at_usage``
on its B10 life; and the mean of ``k`` against its truth, with its
reported standard error against its spread. The seed,
replicates, sizes and slack (0.01, the suite's rule) were fixed before
any run.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)

N = 300
REPS = 400
SEED = 24113
K, ALPHA, BETA = 10.0, 2000.0, 2.0
RATES = (0.0, 0.02, 0.1, 0.5)
TRUTH = np.array([ALPHA, BETA, K])
T_QUERY, RATE_QUERY = 400.0, 0.5


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    rate = rng.choice(RATES, N)
    life = ALPHA * rng.weibull(BETA, N) / (1.0 + K * rate)
    end = rng.uniform(500.0, 3000.0, N)
    c = (life > end).astype(int)
    h = np.minimum(life, end)
    return h, rate * h, c


def _true_sf():
    T = T_QUERY * (1.0 + K * RATE_QUERY)
    return float(np.exp(-((T / ALPHA) ** BETA)))


def _true_b10():
    T = ALPHA * (-np.log(0.9)) ** (1.0 / BETA)
    return float(T / (1.0 + K * RATE_QUERY))


def _one(r):
    h, s, c = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullEquivalence.fit(h, s, c=c)
        wald = np.array([model.param_cb(n) for n in model.parameter_names])
        lr = model.param_cb("k", method="lr")
        band = model.cb_at_usage(T_QUERY, RATE_QUERY)
        b10 = model.quantile_cb_at_usage(0.1, RATE_QUERY)
    return model.k, wald, lr, band, b10, model.standard_errors()[-1]


def test_coverage_and_bias_of_k():
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        out = list(pool.map(_one, range(REPS)))
    k = np.array([o[0] for o in out])
    wald = np.array([o[1] for o in out])
    lr = np.array([o[2] for o in out])
    band = np.array([o[3] for o in out])
    b10 = np.array([o[4] for o in out])
    se_k = np.array([o[5] for o in out])
    check_bias(k, K, "WeibullEquivalence k", standard_errors=se_k)
    check_coverage(
        wald[:, :, 0], wald[:, :, 1], TRUTH, 0.95, "WeibullEquivalence Wald"
    )
    check_coverage(
        lr[:, :1], lr[:, 1:], K, 0.95, "WeibullEquivalence k profile LR"
    )
    check_coverage(
        band[:, :1],
        band[:, 1:],
        _true_sf(),
        0.95,
        "WeibullEquivalence sf at 400 h, 0.5 starts/h",
    )
    check_coverage(
        b10[:, :1],
        b10[:, 1:],
        _true_b10(),
        0.95,
        "WeibullEquivalence B10 at 0.5 starts/h",
    )
