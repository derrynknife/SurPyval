"""Coverage of the semi-Markov ("clock reset") multi-state model's bounds:
the delta-method bounds on the state occupation probabilities and on the
expected length of stay.

The truth: an illness-death model whose hazards are Weibull functions of
the time spent in the current state, healthy -> ill (scale 10, shape
0.8), healthy -> dead (20, 1.5) and ill -> dead (5, 1.3). Each of ``N =
300`` units starts healthy at 0 and is followed to a uniform (0, 30)
time. The true probabilities and lengths of stay are one-dimensional
integrals (taken by adaptive quadrature).

Checked over 400 replicates, from "healthy" at 0: ``cb`` (log(-log)
scale) on the probability of being ill at 3 and at 10 and dead at 10,
and ``length_of_stay_cb`` (logit of the fraction of time) on the time
spent ill and healthy by 10. The seed, replicates, sizes, times and
slack (0.01, the suite's rule) were fixed before any run.
"""

import os
import warnings

import numpy as np
from scipy.integrate import quad

from surpyval.multistate import ParametricMultiState, TransitionMatrix
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

N = 300
REPS = 400
SEED = 11131
CENS = 30.0
PARAMS = {
    ("healthy", "ill"): (10.0, 0.8),
    ("healthy", "dead"): (20.0, 1.5),
    ("ill", "dead"): (5.0, 1.3),
}
TM = TransitionMatrix.illness_death(("healthy", "ill", "dead"))


def _H(u, pair):
    a, b = PARAMS[pair]
    return (u / a) ** b


def _h(u, pair):
    a, b = PARAMS[pair]
    return b / a * (u / a) ** (b - 1)


def _s_healthy(u):
    return np.exp(-_H(u, ("healthy", "ill")) - _H(u, ("healthy", "dead")))


def _p_ill(t):
    return quad(
        lambda v: _h(v, ("healthy", "ill"))
        * _s_healthy(v)
        * np.exp(-_H(t - v, ("ill", "dead"))),
        0,
        t,
        limit=200,
        epsabs=1e-13,
    )[0]


def _truth():
    p_ill = [_p_ill(3.0), _p_ill(10.0)]
    p_dead = 1.0 - _s_healthy(10.0) - p_ill[1]
    los_ill = quad(_p_ill, 0, 10.0, limit=200, epsabs=1e-11)[0]
    los_healthy = quad(_s_healthy, 0, 10.0, epsabs=1e-12)[0]
    return np.array(p_ill), p_dead, los_ill, los_healthy


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    rows = []
    for _ in range(N):
        c = rng.uniform(0, CENS)
        t, state = 0.0, "healthy"
        while state != "dead":
            draws = {
                b: alpha * (-np.log(rng.uniform())) ** (1 / beta)
                for (a, b), (alpha, beta) in PARAMS.items()
                if a == state
            }
            to = min(draws, key=draws.get)
            end = t + draws[to]
            if end >= c:
                rows.append((t, c, state, None))
                break
            rows.append((t, end, state, to))
            t, state = end, to
    tl, x, st, e = (list(v) for v in zip(*rows))
    return np.array(x), e, st, np.array(tl)


def _one(r):
    x, e, st, tl = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiState.fit(
            x, e, st, tl=tl, transitions=TM, clock="reset"
        )
        kw = {"s": 0.0, "from_state": "healthy"}
        ill = model.cb([3.0, 10.0], "ill", **kw)
        dead = model.cb(10.0, "dead", **kw)
        los_ill = model.length_of_stay_cb(10.0, "ill", **kw)
        los_healthy = model.length_of_stay_cb(10.0, "healthy", **kw)
    return ill, dead, los_ill, los_healthy


def test_coverage_of_occupancy_and_length_of_stay_bounds():
    p_ill, p_dead, los_ill, los_healthy = _truth()
    with process_pool(max_workers=min(os.cpu_count() or 1, 2)) as pool:
        out = list(pool.map(_one, range(REPS)))
    ill = np.array([o[0] for o in out])  # (REPS, 2, 2)
    dead = np.array([o[1] for o in out])
    l_ill = np.array([o[2] for o in out])
    l_healthy = np.array([o[3] for o in out])
    check_coverage(
        ill[:, :, 0],
        ill[:, :, 1],
        p_ill,
        0.95,
        "semi-Markov P(ill) at 3 and 10",
    )
    check_coverage(
        dead[:, :1], dead[:, 1:], p_dead, 0.95, "semi-Markov P(dead) at 10"
    )
    check_coverage(
        l_ill[:, :1],
        l_ill[:, 1:],
        los_ill,
        0.95,
        "semi-Markov time ill by 10",
    )
    check_coverage(
        l_healthy[:, :1],
        l_healthy[:, 1:],
        los_healthy,
        0.95,
        "semi-Markov time healthy by 10",
    )
