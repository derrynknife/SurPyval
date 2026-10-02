"""Choosing between optimiser results, and checking the one kept.

A recurrent fit started from a user's ``init`` is also started from the
default, and the better answer kept (#429): from a start far from the
optimum a search can stay where it began, or stop on a plateau, and
report success. A fit whose kept answer is not a verified maximum says
so (principle 13), with ``surpyval.utils.no_maximum.warn_unverified``.
"""

from typing import Any, Callable

import numpy as np

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.univariate.parametric.fitters import (
    at_boundary_maximum,
    is_local_minimum,
    numerical_derivatives,
)


def better_result(res: Any, other: Any) -> Any:
    """The optimiser result with the lower finite objective: ``other``
    only where it is strictly better, so ties keep ``res``."""
    if np.isfinite(other.fun) and not (res.fun <= other.fun):
        return other
    return res


def verified_maximum(
    neg_ll: Callable, mle: Any, bounds: list, n_obs: float
) -> bool:
    """Whether ``mle`` is a verified maximum of the likelihood whose
    negative log is ``neg_ll`` (in the natural parameters, of the given
    ``(lower, upper)`` ``bounds``), per observation (``n_obs``).

    A parameter on a bound of its space where the likelihood is highest
    -- an ARA repair efficiency of 1, a Kijima ``q`` of 0, the ordinary
    renewal process -- is held out of the test (``at_boundary_maximum``:
    the likelihood the same a millionth of the way closer to the bound,
    and not rising ``1e-6`` off it). The others must have a zero gradient
    and a positive-definite Hessian (``is_local_minimum``) in the space the
    searches run in (``unconstraining_maps``), by central differences, as
    the likelihoods are not written for autograd.
    """
    x = np.asarray(mle, dtype=float)
    if not np.all(np.isfinite(x)):
        return False

    def natural(v: Any) -> float:
        return float(neg_ll(np.asarray(v, dtype=float)))

    held = []
    for j, (low, high) in enumerate(bounds):
        for bound, inward in ((low, 1.0), (high, -1.0)):
            if bound is None:
                continue
            toward, away = x.copy(), x.copy()
            toward[j] = bound + (x[j] - bound) * 1e-6
            away[j] = bound + inward * 1e-6
            if at_boundary_maximum(natural, x, toward, away, 1e-6, n_obs):
                held.append(j)
                break
    free = [j for j in range(x.size) if j not in held]
    if not free:
        return True
    to_natural, to_search = unconstraining_maps([bounds[j] for j in free])

    def search(u: Any) -> float:
        full = x.copy()
        full[free] = to_natural(np.asarray(u, dtype=float))
        return natural(full)

    u = to_search(x[free])
    jac, hess = numerical_derivatives(search, u)
    with np.errstate(all="ignore"):
        return is_local_minimum(search, jac, hess, u, obj_scale=n_obs)
