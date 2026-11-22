"""Time ``log_posterior_and_grad``: one value and gradient, as a sampler
asks for it.

The target is about 50 microseconds per gradient for a Weibull with
1,000 rows (exact and right-censored, with a prior on each parameter).
Run from the repository root::

    python scripts/bench_log_posterior.py

It prints microseconds per call (the best of five repeats) for the
hand-written families and, for comparison, for the autograd path (a
Weibull with a limited-failure form) and the public likelihood plus
prior that the grid uses. The data are drawn with a fixed seed.
"""

from __future__ import annotations

import timeit
from typing import Any

import numpy as np

import surpyval as sp
from surpyval.bayes.likelihood import LogLikelihood, log_posterior_and_grad

ROWS = 1000
TARGET_US = 50.0


def _data(dist: Any, params: tuple, rng: Any) -> sp.SurpyvalData:
    x = dist.random(ROWS, *params, random_state=rng)
    c = (rng.uniform(size=ROWS) < 0.3).astype(int)
    return sp.SurpyvalData(x=x, c=c)


def _time(fn: Any, number: int = 2000) -> float:
    best = min(timeit.repeat(fn, number=number, repeat=5))
    return best / number * 1e6


def main() -> None:
    rng = np.random.default_rng(20261120)
    cases: list[tuple[str, Any, tuple, dict[str, Any]]] = [
        ("Weibull", sp.Weibull, (100.0, 2.0), {}),
        ("Exponential", sp.Exponential, (0.01,), {}),
        ("Normal", sp.Normal, (100.0, 20.0), {}),
        ("LogNormal", sp.LogNormal, (4.0, 0.8), {}),
        ("Gamma", sp.Gamma, (2.0, 0.02), {}),
        ("Weibull, lfp (autograd)", sp.Weibull, (100.0, 2.0), {"lfp": True}),
    ]
    print(f"{ROWS} rows, about 30% right-censored; microseconds per call")
    print(f"{'case':<26}{'value+grad':>12}{'public value':>14}")
    for label, dist, params, forms in cases:
        data = _data(dist, params, rng)
        f = log_posterior_and_grad(data, dist, None, **forms)
        model = dist._fit_from_surpyval_data(data, **forms)
        theta = [*np.atleast_1d(model.params)]
        if forms.get("lfp"):
            theta.append(float(model.lfp_p))
        z = f.bound.to_phi(np.array([theta]))[0]
        ll = LogLikelihood(dist, data, **forms)
        bound = f.bound

        def public(z: Any = z, ll: Any = ll, bound: Any = bound) -> Any:
            Z = z[None, :]
            return bound.log_prior(Z) + ll(bound.full_theta(Z))

        fast = _time(lambda f=f, z=z: f(z))
        slow = _time(public, number=500)
        print(f"{label:<26}{fast:>12.1f}{slow:>14.1f}")
        if label == "Weibull":
            verdict = "met" if fast <= TARGET_US else "NOT met"
            weibull = (fast, verdict)
    print(
        f"Weibull target {TARGET_US:.0f} us: {weibull[1]} "
        f"({weibull[0]:.1f} us)"
    )


if __name__ == "__main__":
    main()
