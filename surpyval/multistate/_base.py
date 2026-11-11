"""What the multi-state models share: arguments, the start of a
prediction, and the bounds."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

VARIANCES = ("greenwood", "aalen")


def check_bound_args(alpha_ci: float, bound: str, bound_type: str) -> None:
    check_alpha_ci(alpha_ci)
    check_option("bound", bound, BOUNDS)
    check_option("bound_type", bound_type, ("exp", "normal"))


def probability_bounds(
    p: npt.NDArray,
    se: npt.NDArray,
    alpha_ci: float,
    bound: str,
    bound_type: str,
) -> npt.NDArray:
    r"""Pointwise bounds on probabilities ``p`` with standard errors
    ``se``: ``[lower, upper]`` rows, or one side.

    ``'exp'`` works on the log(-log) scale, :math:`p^{\exp(\pm z\,
    \sigma / (p \log p))}`, which keeps the bounds in (0, 1); where ``p``
    is 0 or 1, or ``se`` is 0, the bounds are ``p``. ``'normal'`` is
    :math:`p \pm z \sigma`."""
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha_ci / (2.0 if bound == "two-sided" else 1.0))
    p = np.asarray(p, dtype=float)
    se = np.asarray(se, dtype=float)
    if bound_type == "normal":
        lower, upper = p - z * se, p + z * se
    else:
        inside = (se > 0) & (p > 0) & (p < 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_p = np.log(np.where(inside, p, 0.5))
            s = z * se / (np.where(inside, p, 0.5) * np.abs(log_p))
            lower = np.where(inside, np.exp(log_p * np.exp(s)), p)
            upper = np.where(inside, np.exp(log_p * np.exp(-s)), p)
    nan = np.isnan(p)
    lower = np.where(nan, np.nan, lower)
    upper = np.where(nan, np.nan, upper)
    if bound == "lower":
        return lower
    if bound == "upper":
        return upper
    return np.column_stack([lower, upper])


def flat_times(x: Any) -> tuple[npt.NDArray, tuple]:
    """The query as a 1-D float array, and its shape."""
    arr = np.asarray(x, dtype=float)
    return arr.ravel(), arr.shape


def step_index(t: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    """For each query time, the index of the last event time at or
    before it (-1 before the first)."""
    return np.searchsorted(t, x, side="right") - 1
