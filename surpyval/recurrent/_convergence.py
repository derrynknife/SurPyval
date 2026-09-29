"""Choosing between optimiser results, and saying when none converged.

A recurrent fit started from a user's ``init`` is also started from the
default, and the better answer kept (#429): from a start far from the
optimum a search can stay where it began, or stop on a plateau, and
report success. A fit whose kept answer the optimiser did not report as
converged says so (principle 13).
"""

import warnings
from typing import Any

import numpy as np


def better_result(res: Any, other: Any) -> Any:
    """The optimiser result with the lower finite objective: ``other``
    only where it is strictly better, so ties keep ``res``."""
    if np.isfinite(other.fun) and not (res.fun <= other.fun):
        return other
    return res


def warn_unconverged(what: str) -> None:
    """Warn that ``what`` (e.g. ``"The Duane fit"``) did not converge."""
    warnings.warn(
        "{} did not converge: the optimiser stopped without meeting its "
        "convergence test, so the parameters returned may not maximise "
        "the likelihood. Check the fit, or try another `init`.".format(what),
        stacklevel=4,
    )
