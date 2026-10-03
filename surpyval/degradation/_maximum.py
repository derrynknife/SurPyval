"""The check every degradation likelihood fit makes of its answer (#564).

The process and destructive fits search their likelihood with BFGS (and a
Nelder-Mead fallback) or a bounded scalar search, none of which says
whether it stopped at a maximum. :func:`verified_search` checks the answer
as every other maximum-likelihood fit does (principle 13): a zero gradient
and a positive-definite Hessian of the negative log-likelihood in the space
the fit searched, by central differences (``verify_or_polish(...,
numerical=True)``), polishing an answer that fails with BFGS and warning
(``warn_unverified``) if it still fails. The fitted model records what was
reached as its ``maximum``.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.utils.no_maximum import warn_unverified


def verified_search(
    fun: Callable[[npt.NDArray], Any],
    res: Any,
    n_obs: float,
    what: str,
) -> tuple[npt.NDArray, str]:
    """
    ``res.x``, the minimum of ``fun`` a fit's search found (polished where
    that was not a verified minimum), and its ``maximum`` state:
    ``"verified"``, or ``"unverified"`` after warning that ``what`` did not
    reach a verified maximum.

    ``fun`` is the negative log-likelihood in the space the fit searched,
    over ``n_obs`` observations (the per-observation scale of the gradient
    test), and ``res`` has the point ``x`` and its value ``fun``.
    """
    with np.errstate(all="ignore"):
        res, verified = verify_or_polish(fun, res, n_obs, numerical=True)
    if not verified:
        warn_unverified(what)
    state = "verified" if verified else "unverified"
    return np.asarray(res.x, dtype=float), state
