"""Coverage of the panel-data multi-state model's bounds.

The truth: an illness-death model with recovery and constant intensities
(healthy -> ill 0.3, healthy -> dead 0.05, ill -> healthy 0.2, ill ->
dead 0.25 per unit of time). ``N = 200`` units start healthy and are
visited at time 0 and then every uniform (0.5, 1.5) time units until a
uniform (4, 8) end of follow-up; a death is recorded at its exact time
(``exact_entry``), and a unit alive at the end has a last visit there
whose state is seen or, with probability 0.3, only known to be "alive"
(``censored``). Between visits the moves are not seen.

Checked over 400 replicates: the 95% Wald bound (``param_cb``, on the log
scale) of each of the four intensities; the delta-method bound
(``cb``, log(-log) scale) on the probability of being ill, and of being
dead, at time 3 for a unit healthy at 0 (elements of ``exp(3 Q)``); and
the bound on the mean stay in "healthy" (``sojourn_cb``). The seed,
replicates, sizes and slack (0.01, the suite's rule) were fixed before
any run.
"""

import os
import warnings

import numpy as np
from scipy.linalg import expm

from surpyval.multistate import PanelMultiState
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

N = 200
REPS = 400
SEED = 26111
STATES = ("healthy", "ill", "dead")
TRANSITIONS = [
    ("healthy", "ill"),
    ("healthy", "dead"),
    ("ill", "healthy"),
    ("ill", "dead"),
]
RATES = np.array([0.3, 0.05, 0.2, 0.25])
Q = np.array([[-0.35, 0.3, 0.05], [0.2, -0.45, 0.25], [0.0, 0.0, 0.0]])
T_QUERY = 3.0


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    x, s, i = [], [], []
    for unit in range(N):
        end = rng.uniform(4.0, 8.0)
        t, state, path = 0.0, 0, [(0.0, 0)]
        while state != 2:
            t += rng.exponential(1.0 / -Q[state, state])
            if t >= end:
                break
            p = np.clip(Q[state], 0.0, None)
            state = int(rng.choice(3, p=p / p.sum()))
            path.append((t, state))
        death = path[-1][0] if path[-1][1] == 2 else None
        visits = [0.0]
        while True:
            nxt = visits[-1] + rng.uniform(0.5, 1.5)
            if nxt >= end:
                break
            visits.append(nxt)
        visits.append(end)
        hidden = rng.uniform() < 0.3
        for k, v in enumerate(visits):
            if death is not None and v >= death:
                x.append(death)
                s.append("dead")
                i.append(unit)
                break
            now = [st for u, st in path if u <= v][-1]
            last = k == len(visits) - 1
            x.append(v)
            s.append("alive" if last and hidden else STATES[now])
            i.append(unit)
    return np.array(x), np.array(s, dtype=object), np.array(i)


def _one(r):
    x, s, i = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelMultiState.fit(
            x,
            s,
            i,
            transitions=TRANSITIONS,
            censored={"alive": ("healthy", "ill")},
            exact_entry="dead",
        )
        wald = np.array([model.param_cb(n) for n in model.parameter_names])
        occ = np.array(
            [
                model.cb(T_QUERY, state, s=0.0, from_state="healthy")
                for state in ("ill", "dead")
            ]
        )
        soj = model.sojourn_cb("healthy")
    return wald, occ, soj


def test_coverage_of_the_panel_model_bounds():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_one, range(REPS)))
    wald = np.array([o[0] for o in out])
    occ = np.array([o[1] for o in out])
    soj = np.array([o[2] for o in out])
    P = expm(T_QUERY * Q)[0]
    check_coverage(
        wald[:, :, 0], wald[:, :, 1], RATES, 0.95, "PanelMultiState intensity"
    )
    check_coverage(
        occ[:, :, 0],
        occ[:, :, 1],
        P[[1, 2]],
        0.95,
        "PanelMultiState P(ill), P(dead) at 3 from healthy",
    )
    check_coverage(
        soj[:, :1], soj[:, 1:], 1 / 0.35, 0.95, "PanelMultiState sojourn"
    )
