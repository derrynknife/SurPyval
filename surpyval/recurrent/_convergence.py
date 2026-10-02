"""Choosing between optimiser results, and checking the one kept.

A recurrent fit started from a user's ``init`` is also started from the
default, and the better answer kept (#429): from a start far from the
optimum a search can stay where it began, or stop on a plateau, and
report success. A fit whose kept answer is not a verified maximum says
so (principle 13), with ``surpyval.utils.no_maximum.warn_unverified``.
"""

from typing import Any

import numpy as np


def better_result(res: Any, other: Any) -> Any:
    """The optimiser result with the lower finite objective: ``other``
    only where it is strictly better, so ties keep ``res``."""
    if np.isfinite(other.fun) and not (res.fun <= other.fun):
        return other
    return res
