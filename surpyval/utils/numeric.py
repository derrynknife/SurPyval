"""Rounding and filling helpers for arrays of numbers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ffill_or_zero(values: npt.ArrayLike) -> npt.NDArray:
    """Each ``nan`` replaced by the last value before it that is not, and
    by 0 where there is none: pandas' ``Series(values).ffill().fillna(0)``,
    without importing pandas (#470).

    Examples
    --------
    >>> from surpyval.utils import ffill_or_zero
    >>> ffill_or_zero([float("nan"), 0.2, float("nan"), 0.5, float("nan")])
    array([0. , 0.2, 0.2, 0.5, 0.5])
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    filled = ~np.isnan(values)
    last = np.where(filled, np.arange(values.size), -1)
    np.maximum.accumulate(last, out=last)
    out = np.where(last >= 0, values[np.maximum(last, 0)], 0.0)
    return out


def _round_vals(x: npt.NDArray) -> npt.NDArray:
    """The ticks ``x`` to the fewest significant figures that keep them
    apart (at most 17, a double's full precision, so ticks that are equal
    to begin with do not loop forever)."""
    not_different = True
    i = 1
    while not_different:
        x_ticks = np.array(round_sig(x, i))
        not_different = (np.diff(x_ticks) == 0).any() and i < 17
        i += 1
    return x_ticks


def round_sig(points: npt.NDArray, sig: int = 2) -> list:
    """
    Round each value to ``sig`` significant figures (used for the tick
    labels of probability plots).

    Parameters
    ----------
    points : array or scalar
        The values to round. 0 (which has no leading digit) and the
        non-finite values are returned as they are.
    sig : int, optional
        The number of significant figures. Defaults to 2.

    Returns
    -------
    list or scalar
        The rounded values: a list for an array, a scalar for a scalar.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import round_sig
    >>> round_sig(np.array([1234.5, 0.012345, -0.5678, 0.0]), 2)
    [np.float64(1200.0), np.float64(0.012), np.float64(-0.57), np.float64(0.0)]
    >>> round_sig(0)
    np.int64(0)
    """
    values = np.asarray(points)
    # The decimal place of the leading digit. log10(0) is -inf, and its
    # int() raised an OverflowError, so a probability plot with a tick at
    # exactly 0 failed (#439); 0 and inf / nan keep 0 decimal places,
    # which leaves them unchanged.
    with np.errstate(divide="ignore", invalid="ignore"):
        leading = np.floor(np.log10(np.abs(values.astype(float))))
    places = np.where(np.isfinite(leading), sig - leading - 1, 0)
    output = [
        np.round(p, int(i)) for p, i in zip(values.ravel(), places.ravel())
    ]
    return output[0] if values.ndim == 0 else output
