"""Reference values for ``surpyval.bayes.convergence`` from ArviZ.

Run in an environment with ArviZ (0.22.0 was used); SurPyval does not
depend on it. The chains are made by integer arithmetic and IEEE
additions and multiplications only, so they are the same on every
platform; the printed values are stored in
``surpyval/tests/bayes/test_convergence.py``.
"""

from typing import Any

import arviz as az  # type: ignore[import-not-found]
import numpy as np


def chains(
    m: Any, n: Any, phi: Any, shift: Any = 0.0, digits: Any = None
) -> Any:
    """``m`` chains of ``n`` draws of ``x[t] = phi x[t-1] + u[t] - 1/2``,
    ``u`` from a 31-bit linear congruential generator; chain ``k`` is
    moved by ``k * shift`` and, with ``digits``, rounded to make ties."""
    state = 12345
    out = np.empty((m, n))
    for k in range(m):
        x = 0.0
        for t in range(n):
            state = (1103515245 * state + 12345) % 2147483648
            x = phi * x + state / 2147483648 - 0.5
            out[k, t] = x + k * shift
    if digits is not None:
        out = np.round(out, digits)
    return out


CASES = {
    "mixed": (4, 200, 0.5, 0.0, None),
    "sticky": (4, 200, 0.95, 0.0, None),
    "apart": (4, 200, 0.5, 0.3, None),
    "odd_with_ties": (3, 101, 0.7, 0.0, 1),
    "one_chain": (1, 400, 0.3, 0.0, None),
}

if __name__ == "__main__":
    for name, args in CASES.items():
        x = chains(*args)
        print(
            f'    "{name}": ({float(az.rhat(x, method="rank"))!r}, '
            f'{float(az.ess(x, method="bulk"))!r}, '
            f'{float(az.ess(x, method="tail"))!r}, '
            f'{float(az.ess(x, method="mean"))!r}),'
        )
