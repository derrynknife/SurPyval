"""Coverage of the multi-state Cox model's bounds when effects are shared
by some transitions and two baselines are proportional.

The truth: a Markov illness-death model (healthy -> ill, healthy -> dead,
ill -> dead) with constant hazards 0.3, 0.1 and 0.25 for a unit with
``z = 0``. A binary covariate ``z`` (half the units) multiplies the hazard
of healthy -> ill by ``exp(0.5)``, and that of both transitions into dead
by ``exp(0.4)`` (one shared effect); the baseline of ill -> dead is 2.5
times that of healthy -> dead. ``N = 300`` units start healthy at 0 and
are followed for a uniform (2, 6) time.

The model fitted is the true one: ``shared={"z": [into dead]}`` and
``proportional_baselines=[into dead]``. Checked over 400 replicates: the
coefficients (``z`` on healthy -> ill, the shared ``z``, the log ratio of
the baselines) against their truth, with their standard errors against
their spread; the predicted probabilities of being ill and dead at times
2 and 4 for a unit with ``z = 1``, healthy at 0, with their standard
errors (the Aalen-type variance, which carries the shared parameters'
covariance across transitions) against their spread; and the 95% ``cb``
on them. The truth is the matrix exponential. The seed, replicates, sizes
and slack (0.01 for the coverages, the suite's rule) were fixed before
any run.
"""

import os
import warnings

import numpy as np
from scipy.linalg import expm

from surpyval.multistate import MultiStateProportionalHazards, TransitionMatrix
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)

N = 300
REPS = 400
SEED = 26111
STATES = ("healthy", "ill", "dead")
PAIRS = [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]
BASE = np.array([0.3, 0.1, 0.25])
B_ILL, B_DEAD = 0.5, 0.4
RATIO = 2.5
INTO_DEAD = [("healthy", "dead"), ("ill", "dead")]
TIMES = np.array([2.0, 4.0])
TRUTH_COEF = np.array([B_ILL, B_DEAD, np.log(RATIO)])


def _rates(z):
    return BASE * np.exp(np.array([B_ILL, B_DEAD, B_DEAD]) * z)


def _true_P(t, z):
    r = _rates(z)
    Q = np.array([[-r[0] - r[1], r[0], r[1]], [0, -r[2], r[2]], [0, 0, 0]])
    return expm(Q * t)


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    rows = []
    for unit in range(N):
        z = float(rng.integers(0, 2))
        rate = _rates(z)
        end = rng.uniform(2.0, 6.0)
        t1 = rng.exponential(1 / (rate[0] + rate[1]))
        if t1 >= end:
            rows.append((0.0, end, "healthy", None, unit, z))
            continue
        if rng.uniform() < rate[0] / (rate[0] + rate[1]):
            rows.append((0.0, t1, "healthy", "ill", unit, z))
            t2 = t1 + rng.exponential(1 / rate[2])
            if t2 >= end:
                rows.append((t1, end, "ill", None, unit, z))
            else:
                rows.append((t1, t2, "ill", "dead", unit, z))
        else:
            rows.append((0.0, t1, "healthy", "dead", unit, z))
    tl, x, state, e, i, z = zip(*rows)
    return np.array(x), list(e), list(state), np.array(tl), np.array(i), z


def _one(r):
    x, e, state, tl, i, z = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MultiStateProportionalHazards.fit(
            x,
            np.array(z)[:, None],
            e,
            state,
            tl=tl,
            i=i,
            transitions=TransitionMatrix(PAIRS, states=STATES),
            shared={0: INTO_DEAD},
            proportional_baselines=INTO_DEAD,
        )
        out = [model.params, model.standard_errors()]
        for b in ("ill", "dead"):
            args = (TIMES, [1.0], b)
            out.append(model.occupancy(*args, from_state="healthy"))
            out.append(model.occupancy_se(*args, from_state="healthy"))
            out.append(model.cb(*args, from_state="healthy"))
    return out


def test_shared_effects_and_proportional_baselines():
    with process_pool(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        out = list(pool.map(_one, range(REPS)))
    coef = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    check_bias(coef, TRUTH_COEF, "multi-state Cox shared", standard_errors=se)
    for k, b in enumerate(("ill", "dead")):
        p = np.array([o[2 + 3 * k] for o in out])
        p_se = np.array([o[3 + 3 * k] for o in out])
        band = np.array([o[4 + 3 * k] for o in out])
        truth = np.array([_true_P(t, 1.0)[0, STATES.index(b)] for t in TIMES])
        check_bias(
            p,
            truth,
            f"multi-state Cox shared, P(healthy -> {b}) at {TIMES}",
            standard_errors=p_se,
        )
        check_coverage(
            band[:, :, 0],
            band[:, :, 1],
            truth,
            0.95,
            f"multi-state Cox shared, cb of P(healthy -> {b}) at {TIMES}",
        )
