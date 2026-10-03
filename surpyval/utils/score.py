"""Harrell's concordance index, under its pre-0.22 name.

The index is :func:`surpyval.metrics.concordance_index` (#512), which
counts the pairs in O(n log n); ``score`` keeps working until v0.23, with a
``DeprecationWarning``, and keeps its old tie convention, Harrell's
(``ties="harrell"``: two events at the same time are a usable pair), where
``concordance_index`` defaults to Therneau's (R's
``survival::concordance``, lifelines: they are not).
"""

import warnings

from numpy.typing import ArrayLike

from surpyval.metrics.concordance import concordance_index
from surpyval.utils.deprecation import REMOVED_IN


def score(
    x: ArrayLike,
    c: ArrayLike,
    scores: ArrayLike,
    tie_tol: float = 1e-8,
) -> float:
    """Harrell's concordance index of risk ``scores`` (deprecated).

    Use :func:`surpyval.metrics.concordance_index`, which this calls with
    ``ties="harrell"``, the convention ``score`` always had: two events at
    the same time are a usable pair. ``concordance_index`` leaves them out
    by default (``ties="therneau"``, as R and lifelines), so the two differ
    on data with tied event times.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.score import score
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     round(score([1.0, 2.0, 3.0], [0, 0, 1], [3.0, 1.0, 2.0]), 4)
    0.6667
    """
    warnings.warn(
        "surpyval.utils.score.score is deprecated and will be removed in "
        f"v{REMOVED_IN}; use surpyval.metrics.concordance_index.",
        DeprecationWarning,
        stacklevel=2,
    )
    return concordance_index(x, c, scores, tie_tol, ties="harrell")
