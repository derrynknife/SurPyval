"""How fast ``conditional_gaps`` draws a mixture's residual life, against
a single distribution's draw.

A mixture's draws take Newton's steps on its cumulative hazard from a
bound found from its components (``_mixture_start``), where they used to
invert its quantile function (itself a root search on the failure
function). This records, one draw per call as the renewal simulations
make them, the time per draw for

- a single Weibull (the quantile function in closed form, or Newton's
  steps past the quantile limit), and a single log-normal;
- two- and three-component Weibull mixtures and a two-component
  log-normal mixture;

at the ages 0, the median and the 0.999 quantile of each lifetime, and
the ratio of each mixture's time to the single Weibull's at the same
age. Times are the median of 7 repeats of 200 draws, on one core; the
uniforms come from ``default_rng(20261120)`` (log-uniform from 1e-9 to
1, so the far tail is drawn as often as the body). Run it before and
after a change on the same machine::

    python scripts/studies/mixture_conditional_gaps_speed.py

This is a benchmark, not a test: nothing here asserts a time.
"""

import time
from typing import Any

import numpy as np

import surpyval as surv
from surpyval.recurrent.renewal.renewal_model import conditional_gaps

REPEATS = 7
DRAWS = 200


def mixture(dist: Any, params: Any, w: Any) -> surv.MixtureModel:
    model = surv.MixtureModel(dist, m=len(w))
    model.params = np.array(params, dtype=float)
    model.w = np.array(w, dtype=float)
    return model


SINGLE = ("Weibull(2000, 3.5)", surv.Weibull.from_params([2000.0, 3.5]))
CASES = [
    SINGLE,
    ("LogNormal(3, 1.2)", surv.LogNormal.from_params([3.0, 1.2])),
    (
        "Weibull mixture, 2 components",
        mixture(surv.Weibull, [[60, 0.8], [2000, 3.5]], [0.25, 0.75]),
    ),
    (
        "Weibull mixture, 3 components",
        mixture(
            surv.Weibull,
            [[10, 0.5], [300, 1.5], [2000, 4.0]],
            [0.1, 0.3, 0.6],
        ),
    ),
    (
        "LogNormal mixture, 2 components",
        mixture(surv.LogNormal, [[1, 0.5], [3, 1.2]], [0.4, 0.6]),
    ),
]


def per_draw(lifetime: Any, v: float, u: np.ndarray) -> float:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for uk in u:
            conditional_gaps(lifetime, np.array([v]), np.array([uk]))
        times.append((time.perf_counter() - start) / u.size)
    return float(np.median(times))


def main() -> None:
    rng = np.random.default_rng(20261120)
    u = 10.0 ** rng.uniform(-9.0, 0.0, size=DRAWS)
    levels = {"0": 0.0, "median": 0.5, "0.999 quantile": 0.999}
    base = {
        level: per_draw(SINGLE[1], float(SINGLE[1].qf(p)) if p else 0.0, u)
        for level, p in levels.items()
    }
    print("| lifetime | age | time/draw | vs single Weibull |")
    print("|---|---|---|---|")
    for name, lifetime in CASES:
        for level, p in levels.items():
            v = float(lifetime.qf(p)) if p else 0.0
            t = per_draw(lifetime, v, u)
            print(
                "| {} | {} | {:.1f} us | {:.2f} |".format(
                    name, level, t * 1e6, t / base[level]
                )
            )


if __name__ == "__main__":
    main()
