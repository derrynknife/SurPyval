"""The inverse of the cumulative hazard along a covariate path.

``qf_tvc`` (on :class:`~surpyval.univariate.regression._tvc_evaluation.
TVCEvaluationMixin`) finds the age at which the cumulative hazard along a
path, from a start age, reaches a target. The helpers here are the parts
that know nothing about the model: a vectorised bracketed root finder.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

#: Iterations of the bracketed solver before it gives up (each one is a
#: single vectorised evaluation for every value still converging).
_MAX_ITER = 200


def bracketed_root(
    f: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    lo: npt.NDArray,
    hi: npt.NDArray,
    d_lo: npt.NDArray,
    d_hi: npt.NDArray,
    ftol: npt.NDArray,
    x0: npt.NDArray | None = None,
) -> npt.NDArray:
    """
    Roots of ``f`` between ``lo`` and ``hi``, one per row, by the Illinois
    variant of regula falsi (bisecting where an end is not finite).

    ``f(x, rows)`` gives the residual at ``x`` for those rows; ``d_lo < 0
    <= d_hi`` are the residuals at the ends. A row stops on its own when
    its residual is within ``ftol`` or its bracket is a few ulps wide, so
    one value never moves another. ``x0``, inside each bracket, is a
    first guess tried before the first interpolation.
    """
    lo, hi = lo.astype(float).copy(), hi.astype(float).copy()
    d_lo, d_hi = d_lo.astype(float).copy(), d_hi.astype(float).copy()
    x = hi.copy()
    side = np.zeros(lo.shape, dtype=int)
    active = np.flatnonzero(d_hi > ftol)
    guess = None if x0 is None else np.asarray(x0, dtype=float)
    for _ in range(_MAX_ITER):
        if active.size == 0:
            break
        a, b = lo[active], hi[active]
        fa, fb = d_lo[active], d_hi[active]
        if guess is not None:
            xn, guess = guess[active], None
        else:
            with np.errstate(all="ignore"):
                xn = b - fb * (b - a) / (fb - fa)
        bad = ~np.isfinite(xn) | (xn <= a) | (xn >= b)
        xn = np.where(bad, 0.5 * (a + b), xn)
        fn = np.asarray(f(xn, active), dtype=float)
        x[active] = xn
        above = fn >= 0
        # Illinois: halve the stale end's residual when the same end moves
        # twice running.
        s = side[active]
        stale_lo = above & (s == 1)
        stale_hi = ~above & (s == -1)
        new_lo = np.where(above, a, xn)
        new_hi = np.where(above, xn, b)
        new_flo = np.where(above, np.where(stale_lo, 0.5 * fa, fa), fn)
        new_fhi = np.where(above, fn, np.where(stale_hi, 0.5 * fb, fb))
        lo[active], hi[active] = new_lo, new_hi
        d_lo[active], d_hi[active] = new_flo, new_fhi
        side[active] = np.where(above, 1, -1)
        width = new_hi - new_lo
        done = (
            (np.abs(fn) <= ftol[active])
            | (width <= 4 * np.finfo(float).eps * np.abs(new_hi))
            | np.isnan(fn)
        )
        x[active[np.isnan(fn)]] = np.nan
        active = active[~done]
    return x


def inverse_cubic(
    xs: npt.NDArray, ys: npt.NDArray, k: npt.NDArray, y: npt.NDArray
) -> npt.NDArray:
    """
    A first guess at the ``x`` where ``y`` is reached between columns
    ``k - 1`` and ``k`` of each row of ``xs`` (increasing ``ys``): the
    inverse (``x`` as a function of ``y``) Lagrange cubic through the four
    columns around the bracket, clipped to the bracket.
    """
    rows = np.arange(xs.shape[0])
    m = xs.shape[1]
    first = np.clip(k - 2, 0, m - 4) if m >= 4 else np.zeros_like(k)
    order = min(m, 4)
    cols = first[:, None] + np.arange(order)[None, :]
    X = xs[rows[:, None], cols]
    Y = ys[rows[:, None], cols]
    out = np.zeros(xs.shape[0])
    with np.errstate(all="ignore"):
        for i in range(order):
            term = X[:, i].copy()
            for j in range(order):
                if j != i:
                    term *= (y - Y[:, j]) / (Y[:, i] - Y[:, j])
            out += term
    lo, hi = xs[rows, k - 1], xs[rows, k]
    ok = np.isfinite(out) & (out > lo) & (out < hi)
    return np.where(ok, out, 0.5 * (lo + hi))
