"""Recovery and coverage of the shared-frailty Poisson process
(``CrowAMSAA.fit(..., frailty=...)``), the size of its test of no
frailty, and the coverage of its forecast interval.

The truth: ``N = 150`` units under a power-law intensity (Crow-AMSAA,
``alpha = 20``, ``beta = 1.3``), each with its own frailty of variance
``theta = 0.5`` (gamma with mean 1; in the second study log-normal,
``log u ~ N(0, 0.5)``), each watched from 0 to a uniform (60, 120) age:
about 6 failures a unit. The data are simulated from the definition (a
frailty, a Poisson count, the times given the count), not with the
package.

Checked over 300 replicates of the gamma study: the bias of every
parameter (with its reported standard error against its spread); the
coverage of the Wald ``param_cb`` of every parameter, of the
profile-likelihood ``param_cb`` of ``theta``, and of the delta-method
``cif_cb`` at age 50; and the coverage of the forecast's 95% prediction
interval for one new unit's failures by age 100, whose count is drawn
with its own frailty (the count is discrete, so the interval holds at
least 95%: the check is one sided). The coverage of the forecast
interval of the fitted fleet's failures over its next 30 time units,
which carries each unit's history but not the parameters' uncertainty,
is printed beside that of the Poisson interval of the plain fit, not
asserted (in the one run: 90.0%, against the Poisson interval's 82.3%;
the shortfall from 95% is the parameters' uncertainty, which neither
interval carries).

Over 200 replicates of the log-normal study: the bias and the Wald
coverage of every parameter.

The size of the test of ``theta = 0`` (``frailty_test``): 300 replicates
of the gamma design with no frailty, at the 5% level.

The seeds, replicates, sizes and tolerances (the suite's: 3 Monte Carlo
standard errors plus a slack of 0.01 for the rates and coverages, 0.2 of
a standard deviation for the bias) were fixed before any run.
"""

import os
import warnings

import numpy as np

import surpyval
from surpyval.recurrent import CrowAMSAA
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    check_rate,
    process_pool,
)

N = 150
REPS = 300
REPS_LOGNORMAL = 200
SEED = 46811
ALPHA, BETA, THETA = 20.0, 1.3, 0.5
T_QUERY = 50.0
T_NEW = 100.0
AHEAD = 30.0
WORKERS = min(os.cpu_count() or 1, 4)


def _cif(t):
    return (np.asarray(t, dtype=float) / ALPHA) ** BETA


def _frailties(rng, theta, family, size):
    if theta == 0:
        return np.ones(size)
    if family == "gamma":
        return rng.gamma(1 / theta, theta, size)
    return np.exp(rng.normal(0.0, np.sqrt(theta), size))


def _sample(r, theta, family):
    rng = np.random.default_rng([SEED, r, int(theta * 10), len(family)])
    u = _frailties(rng, theta, family, N)
    end = rng.uniform(60.0, 120.0, N)
    x, i, c = [], [], []
    for unit in range(N):
        top = _cif(end[unit])
        n = rng.poisson(u[unit] * top)
        times = ALPHA * np.sort(rng.uniform(0.0, top, n)) ** (1 / BETA)
        x += [*times, end[unit]]
        i += [unit] * (n + 1)
        c += [0] * n + [1]
    # What comes next: the fleet's failures over the time ahead, and one
    # new unit's by T_NEW.
    future = rng.poisson(u * (_cif(end + AHEAD) - _cif(end))).sum()
    new = rng.poisson(_frailties(rng, theta, family, 1)[0] * _cif(T_NEW))
    return np.array(x), np.array(i), np.array(c), int(future), int(new)


def _one(r):
    x, i, c, future, new = _sample(r, THETA, "gamma")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        wald = np.array([model.param_cb(n) for n in model.parameter_names])
        lr = model.param_cb("theta", method="lr")
        cif = model.cif_cb(T_QUERY)
        se = model.standard_errors()
        fleet = model.forecast(AHEAD)
        unit = model.forecast(T_NEW, age=[0.0])
        plain = surpyval.forecast(CrowAMSAA.fit(x, i, c), horizon=AHEAD)
    return (
        model.params,
        se,
        wald,
        lr,
        cif,
        (fleet.lower[0], fleet.upper[0], future),
        (unit.lower[0], unit.upper[0], new),
        (plain.lower[0], plain.upper[0]),
    )


def _lognormal(r):
    x, i, c, _, _ = _sample(r, THETA, "lognormal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(x, i, c, frailty="lognormal")
        wald = np.array([model.param_cb(n) for n in model.parameter_names])
        se = model.standard_errors()
    return model.params, se, wald


def _null(r):
    x, i, c, _, _ = _sample(r, 0.0, "gamma")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        return float(model.frailty_test()["p"].iloc[0])


def test_gamma_recovery_and_coverage():
    with process_pool(max_workers=WORKERS) as pool:
        out = list(pool.map(_one, range(REPS)))
    params = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    wald = np.array([o[2] for o in out])
    lr = np.array([o[3] for o in out])
    cif = np.array([o[4] for o in out])
    fleet = np.array([o[5] for o in out], dtype=float)
    unit = np.array([o[6] for o in out], dtype=float)
    plain = np.array([o[7] for o in out], dtype=float)
    truth = np.array([ALPHA, BETA, THETA])
    check_bias(params, truth, "frailty NHPP (gamma)", standard_errors=se)
    check_coverage(
        wald[:, :, 0], wald[:, :, 1], truth, 0.95, "frailty NHPP (gamma) Wald"
    )
    check_coverage(
        lr[:, 0], lr[:, 1], THETA, 0.95, "frailty NHPP (gamma) profile LR"
    )
    check_coverage(
        cif[:, 0],
        cif[:, 1],
        float(_cif(T_QUERY)),
        0.95,
        "frailty NHPP (gamma) cif_cb at 50",
    )
    hits = np.sum((unit[:, 0] <= unit[:, 2]) & (unit[:, 2] <= unit[:, 1]))
    check_rate(
        hits,
        REPS,
        0.95,
        "frailty NHPP (gamma) forecast interval, one new unit to 100",
        side="lower",
    )
    inside = (fleet[:, 0] <= fleet[:, 2]) & (fleet[:, 2] <= fleet[:, 1])
    poisson = (plain[:, 0] <= fleet[:, 2]) & (fleet[:, 2] <= plain[:, 1])
    print(
        "frailty NHPP (gamma) forecast interval of the fitted fleet over "
        "30 ahead (parameters taken as known): {:.4f}; the plain fit's "
        "Poisson interval: {:.4f} (target 0.95, not asserted)".format(
            inside.mean(), poisson.mean()
        )
    )


def test_lognormal_recovery_and_coverage():
    with process_pool(max_workers=WORKERS) as pool:
        out = list(pool.map(_lognormal, range(REPS_LOGNORMAL)))
    params = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    wald = np.array([o[2] for o in out])
    truth = np.array([ALPHA, BETA, THETA])
    check_bias(params, truth, "frailty NHPP (lognormal)", standard_errors=se)
    check_coverage(
        wald[:, :, 0],
        wald[:, :, 1],
        truth,
        0.95,
        "frailty NHPP (lognormal) Wald",
    )


def test_size_of_the_test_of_no_frailty():
    with process_pool(max_workers=WORKERS) as pool:
        p = np.array(list(pool.map(_null, range(REPS))))
    check_rate(
        int(np.sum(p < 0.05)),
        REPS,
        0.05,
        "frailty NHPP size of the theta = 0 test at 5%",
    )
