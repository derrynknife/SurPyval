"""How fast ``conditional_gaps`` draws a residual life past the quantile
limit, and how fast a two-Weibull ``MixtureModel`` evaluates ``Hf``.

The renewal simulations draw one failure at a time, so a draw at a large
virtual age (cumulative hazard past ``_QF_HAZARD_LIMIT``) pays for the
inversion of ``H`` once per failure. For each case this records

- the calls of the lifetime's cumulative hazard ``H`` and hazard rate
  ``h`` per draw (counted by wrapping the functions the sampler uses);
- the time per draw, one draw per call as a simulation makes them;
- the time per call of ``MixtureModel.Hf`` and ``MixtureModel.hf`` on
  one point.

Times are the median of 7 repeats of 200 draws (or 2,000 calls), on one
core; the uniforms come from ``default_rng(20261105)``. Run it before and
after a change on the same machine::

    python scripts/studies/conditional_gaps_speed.py
"""

import time
from collections import Counter
from typing import Any

import numpy as np

import surpyval as surv
from surpyval.recurrent.renewal import renewal_model

REPEATS = 7
DRAWS = 200
CALLS = 2000


def mixture() -> surv.MixtureModel:
    model = surv.MixtureModel(surv.Weibull, m=2)
    model.params = np.array([[60.0, 0.8], [2000.0, 3.5]])
    model.w = np.array([0.25, 0.75])
    return model


CASES = [
    (
        "MixtureModel Weibull [[60, 0.8], [2000, 3.5]], w [0.25, 0.75]",
        mixture(),
        5000.0,
    ),
    ("LogNormal(5, 0.8)", surv.LogNormal.from_params([5, 0.8]), 40000.0),
    ("LogNormal(5, 0.8)", surv.LogNormal.from_params([5, 0.8]), 100000.0),
]


def counting(calls: Counter) -> Any:
    """Wrap ``_lifetime_functions`` so that its functions count, in
    ``calls``, the evaluations of ``H`` ("H") and of the hazard rate
    ("h"). Before Newton's method it returned ``(H, qf)``; after it, ``(H,
    qf, both)`` with ``both(x, H=None)`` giving ``H`` and ``h`` (finding
    ``H`` only where it is not passed in)."""
    original = renewal_model._lifetime_functions

    def wrapped(lifetime: Any) -> Any:
        functions = original(lifetime)
        hazard, quantile = functions[:2]

        def counted_hazard(*args: Any, **kwargs: Any) -> Any:
            calls["H"] += 1
            return hazard(*args, **kwargs)

        if len(functions) == 2 or functions[2] is None:
            return (counted_hazard, quantile) + tuple(functions[2:])
        both = functions[2]

        def counted_both(x: Any, H: Any = None) -> Any:
            calls["h"] += 1
            if H is None:
                calls["H"] += 1
            return both(x, H)

        return counted_hazard, quantile, counted_both

    return original, wrapped


def median_time(fn: Any, n: Any) -> Any:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) / n)
    return float(np.median(times))


def main() -> None:
    u = np.random.default_rng(20261105).uniform(size=DRAWS)
    print("| lifetime | v | H(v) | H evals/draw | h evals/draw | time/draw |")
    print("|---|---|---|---|---|---|")
    for name, lifetime, v in CASES:
        calls: Counter = Counter()
        original, wrapped = counting(calls)
        renewal_model._lifetime_functions = wrapped
        try:
            for uk in u:
                renewal_model.conditional_gaps(
                    lifetime, np.array([v]), np.array([uk])
                )
        finally:
            renewal_model._lifetime_functions = original

        def draws() -> Any:
            for uk in u:
                renewal_model.conditional_gaps(
                    lifetime, np.array([v]), np.array([uk])
                )

        per_draw = median_time(draws, DRAWS)
        print(
            "| {} | {:g} | {:.1f} | {:.1f} | {:.1f} | {:.3f} ms |".format(
                name,
                v,
                float(lifetime.Hf(v)),
                calls["H"] / DRAWS,
                calls["h"] / DRAWS,
                per_draw * 1e3,
            )
        )
    model = mixture()
    x = np.array([5000.0])
    for method in ("Hf", "hf"):
        fn = getattr(model, method)

        def run() -> Any:
            for _ in range(CALLS):
                fn(x)

        print(
            "MixtureModel.{} per call: {:.1f} us".format(
                method, median_time(run, CALLS) * 1e6
            )
        )


if __name__ == "__main__":
    main()
