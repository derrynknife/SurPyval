"""Recovery and coverage of the joint frailty model of recurrent and
terminal events, and the size of its test of no frailty.

The truth: ``N = 200`` units, half with ``z = 1``; a gamma frailty of
variance ``theta = 0.5``; recurrent events with the power-law (Weibull)
intensity of scale 10 and shape 1.2 on the unit's age, times ``u e^{0.5
z}``; a terminal event, Weibull with scale 20 and shape 1.5, times ``u^1
e^{0.3 z}`` (``alpha = 1``); each unit followed to a uniform (10, 20) age
or its terminal event. About 2.4 recurrent events and 0.4 terminal events
a unit.

Checked over 300 replicates: the bias of every parameter (with its
reported standard error against its spread); the coverage of the Wald
``param_cb`` of every parameter, of the profile-likelihood ``param_cb``
of ``theta`` and ``alpha``, and of the delta-method bands ``sf_cb`` and
``mcf_cb`` at age 10 for ``z = 1``; and the rejection rate of the test of
``alpha = 0`` (``frailty_tests``) under the truth, reported as its power.

The size of the test of ``theta = 0``: 300 replicates of the same design
with no frailty (``theta = 0``, so ``alpha`` has no meaning), at the 5%
level. ``alpha`` is not identified under this null, so the 50:50
chi-square mixture is an approximation (Davies 1987); the check is one
sided -- the test must not reject more often than 5% beyond the
tolerance -- and the rate is printed.

The seeds, replicates, sizes and tolerances (the suite's: 3 Monte Carlo
standard errors plus a slack of 0.01 for the rates and coverages, 0.2 of
a standard deviation for the bias) were fixed before any run.
"""

import os
import warnings

import numpy as np

from surpyval.recurrent import JointFrailty
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    check_rate,
    process_pool,
)

N = 200
REPS = 300
SEED = 24117
TRUTH = np.array([10.0, 1.2, 20.0, 1.5, 0.5, 1.0, 0.5, 0.3])
T_QUERY = 10.0
WORKERS = min(os.cpu_count() or 1, 4)


def _truth(theta=0.5, alpha=1.0):
    return JointFrailty.from_params(
        (10.0, 1.2), (20.0, 1.5), theta, alpha, beta=[0.5], gamma=[0.3]
    )


def _sample(r, theta=0.5):
    rng = np.random.default_rng([SEED, r, int(theta * 10)])
    Z = (np.arange(N) % 2)[:, None].astype(float)
    end = rng.uniform(10.0, 20.0, N)
    return _truth(theta).random(N, end=end, Z=Z, random_state=rng)


def _one(r):
    rows = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointFrailty.fit(**rows)
        wald = np.array([model.param_cb(n) for n in model.parameter_names])
        lr = np.array(
            [model.param_cb(n, method="lr") for n in ("theta", "alpha")]
        )
        sf = model.sf_cb(T_QUERY, [1.0])
        mcf = model.mcf_cb(T_QUERY, [1.0])
        p_alpha = float(model.frailty_tests().loc["alpha = 0", "p"])
        se = model.standard_errors()
    return model.params, se, wald, lr, sf, mcf, p_alpha


def _null(r):
    rows = _sample(r, theta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointFrailty.fit(**rows)
        return float(model.frailty_tests().loc["theta = 0", "p"])


def test_recovery_and_coverage():
    with process_pool(max_workers=WORKERS) as pool:
        out = list(pool.map(_one, range(REPS)))
    params = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    wald = np.array([o[2] for o in out])
    lr = np.array([o[3] for o in out])
    sf = np.array([o[4] for o in out])
    mcf = np.array([o[5] for o in out])
    p_alpha = np.array([o[6] for o in out])
    truth = _truth()
    check_bias(params, TRUTH, "JointFrailty", standard_errors=se)
    check_coverage(
        wald[:, :, 0], wald[:, :, 1], TRUTH, 0.95, "JointFrailty Wald"
    )
    check_coverage(
        lr[:, :, 0],
        lr[:, :, 1],
        TRUTH[4:6],
        0.95,
        "JointFrailty profile LR (theta, alpha)",
    )
    check_coverage(
        sf[:, :1],
        sf[:, 1:],
        float(truth.sf(T_QUERY, [1.0])),
        0.95,
        "JointFrailty sf_cb at 10, z = 1",
    )
    check_coverage(
        mcf[:, :1],
        mcf[:, 1:],
        float(truth.mcf(T_QUERY, [1.0])),
        0.95,
        "JointFrailty mcf_cb at 10, z = 1",
    )
    # The power of the test of alpha = 0 at the truth (reported only)
    print(
        "JointFrailty power of the alpha = 0 test at 5%: {:.3f}".format(
            float(np.mean(p_alpha < 0.05))
        )
    )


def test_size_of_no_frailty_test():
    with process_pool(max_workers=WORKERS) as pool:
        p = np.array(list(pool.map(_null, range(REPS))))
    check_rate(
        int(np.sum(p < 0.05)),
        REPS,
        0.05,
        "JointFrailty size of the theta = 0 test at 5%",
        side="upper",
    )
