"""Rounding and filling helpers for arrays of numbers, and a solver for
many bracketed roots at once."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

_MACHINE_EPS = float(np.finfo(float).eps)

# Constants once at the top level of ``surpyval`` (``surpyval.NUM`` ...),
# deprecated there in v0.23 (#613) and kept here, their home, until then.
#: The float type, numpy's float64.
NUM = np.float64
#: The smallest positive normal float, ``numpy.finfo(float).tiny``.
TINIEST = np.finfo(np.float64).tiny
#: The square root of the machine epsilon, a usual finite-difference step.
EPS = np.sqrt(np.finfo(NUM).eps)


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


def solve_bracketed(
    g: Callable,
    lo: npt.NDArray,
    hi: npt.NDArray,
    g_lo: npt.NDArray,
    g_hi: npt.NDArray,
    xtol: float = 0.0,
    rtol: float = 4 * _MACHINE_EPS,
    maxiter: int = 400,
) -> npt.NDArray:
    """
    Solve many bracketed root problems ``g_k(x) = 0`` at once, each with
    ``g_k(lo_k) < 0 < g_k(hi_k)``. ``g(x, sel)`` evaluates the problems
    whose indices are ``sel`` at the points ``x``.

    Each step takes a regula falsi point with the Illinois modification
    (halving the value kept at an end that has not moved for two steps),
    and bisects instead whenever the last two steps together did not halve
    the bracket, so every problem converges at least half as fast as
    bisection, and superlinearly once regula falsi takes hold. No point is
    taken within the tolerance of an end, so the last step closes the
    bracket instead of creeping up on the root. A problem is
    done when its bracket is narrower than ``xtol + rtol * |x|``; the
    midpoint of the final bracket is returned.
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    g_lo = np.array(g_lo, dtype=float)
    g_hi = np.array(g_hi, dtype=float)
    root = lo + 0.5 * (hi - lo)
    moved = np.zeros(lo.size, dtype=int)  # -1 lo moved last, +1 hi did
    bisect = np.zeros(lo.size, dtype=bool)
    # The bracket's width two steps back.
    earlier = hi - lo
    active = np.flatnonzero(
        hi - lo > xtol + rtol * np.maximum(np.abs(lo), np.abs(hi))
    )
    for _ in range(maxiter):
        if not active.size:
            break
        a = active
        lo_a, hi_a, gl, gh = lo[a], hi[a], g_lo[a], g_hi[a]
        width = hi_a - lo_a
        mid = lo_a + 0.5 * width
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            falsi = hi_a - gh * width / (gh - gl)
            x = np.where(bisect[a] | ~np.isfinite(falsi), mid, falsi)
        # Never evaluate within the tolerance of an end (Brent's tolerance
        # step). Once one end has converged, regula falsi lands on it (or
        # past it, by rounding); a point just inside it lets the next sign
        # change close the bracket.
        tol = 0.5 * (xtol + rtol * np.maximum(np.abs(lo_a), np.abs(hi_a)))
        x = np.clip(x, lo_a + tol, hi_a - tol)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gx = np.asarray(g(x, a), dtype=float)
        exact = gx == 0
        below = gx < 0
        # A NaN counts as past the root, so the bracket still shrinks.
        above = ~below & ~exact
        new_lo = np.where(below, x, lo_a)
        new_hi = np.where(above, x, hi_a)
        new_gl = np.where(below, gx, gl)
        new_gh = np.where(above, gx, gh)
        new_gh = np.where(below & (moved[a] == -1), 0.5 * new_gh, new_gh)
        new_gl = np.where(above & (moved[a] == 1), 0.5 * new_gl, new_gl)
        moved[a] = np.where(below, -1, np.where(above, 1, 0))
        lo[a], hi[a], g_lo[a], g_hi[a] = new_lo, new_hi, new_gl, new_gh
        new_width = new_hi - new_lo
        bisect[a] = new_width > 0.5 * earlier[a]
        earlier[a] = width
        root[a] = np.where(exact, x, new_lo + 0.5 * new_width)
        done = exact | (
            new_width
            <= xtol + rtol * np.maximum(np.abs(new_lo), np.abs(new_hi))
        )
        active = a[~done]
    return root
