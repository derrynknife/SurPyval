"""Smoothers for the diagnostic plots: a lowess trend and its bootstrap
band (#867). numpy only."""

from __future__ import annotations

from typing import Any

import numpy as np


def lowess(
    x: Any,
    y: Any,
    at: Any,
    frac: float = 2.0 / 3.0,
    iterations: int = 3,
) -> np.ndarray:
    """
    Cleveland's (1979) lowess: at each point of ``at``, the weighted
    straight-line fit to the ``ceil(frac * n)`` nearest points, with
    tricube weights in the distance, refitted ``iterations`` times with
    bisquare robustness weights (six median absolute residuals), as R's
    ``lowess`` (defaults ``f = 2/3``, ``iter = 3``). The residuals that
    set the robustness weights are taken from the fit interpolated
    between the points of ``at``, so the cost is ``len(at) * n`` per
    iteration.

    Returns the smoothed values at ``at``; ``nan`` where fewer than two
    distinct ``x`` are given.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.smooth import lowess
    >>> x = np.linspace(0, 1, 50)
    >>> lowess(x, 2 * x + 1, [0.0, 0.5, 1.0]).round(6)
    array([1., 2., 3.])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    at = np.asarray(at, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = x.size
    if n < 2 or np.unique(x).size < 2:
        return np.full(at.shape, np.nan)
    k = min(n, max(2, int(np.ceil(frac * n))))
    robust = np.ones(n)
    order = np.argsort(at)
    grid = at[order]
    fit = np.empty(grid.size)
    for step in range(iterations + 1):
        for start in range(0, grid.size, 64):
            g = grid[start : start + 64]
            # Signed distances from each point of the grid, so that the
            # local fit is centred where it is evaluated.
            d = x[None, :] - g[:, None]
            dist = np.abs(d)
            h = np.partition(dist, k - 1, axis=1)[:, k - 1]
            h = np.where(h > 0, h, dist.max(axis=1))
            h = np.where(h > 0, h, 1.0)
            u = np.clip(dist / (1.0001 * h[:, None]), 0.0, 1.0)
            w = (1.0 - u**3) ** 3 * robust[None, :]
            sw = w.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                md = (w * d).sum(axis=1) / sw
                my = (w * y).sum(axis=1) / sw
                dc = d - md[:, None]
                var = (w * dc * dc).sum(axis=1)
                cov = (w * dc * (y[None, :] - my[:, None])).sum(axis=1)
                tiny = 1e-12 * (h**2) * sw
                slope = np.where(
                    var > tiny, cov / np.where(var > 0, var, 1), 0
                )
            # The line through (md, my) at distance 0, the grid point.
            fit[start : start + 64] = my - slope * md
        if step == iterations:
            break
        resid = y - np.interp(x, grid, fit)
        mad = np.median(np.abs(resid))
        if mad <= 0:
            break
        r = np.clip(resid / (6.0 * mad), -1.0, 1.0)
        robust = (1.0 - r**2) ** 2
    out = np.empty(at.shape)
    out[order] = fit
    return out


def bootstrap_band(
    x: Any,
    y: Any,
    at: Any,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> np.ndarray:
    """
    A pointwise percentile band for the :func:`lowess` trend: the
    ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles at each point
    of ``at`` of the trends of ``n_boot`` resamples of the ``(x, y)``
    pairs, drawn with replacement from ``rng``. To keep a band of 200
    resamples of a few thousand points to a second or so, each resample's
    trend is computed at (at most) 25 points spanning ``at``, with one
    robustness iteration, and the band is interpolated between them.

    Returns an array of ``[lower, upper]`` rows, one per point of ``at``
    (``nan`` where no resample had a trend there).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.smooth import bootstrap_band
    >>> rng = np.random.default_rng(1)
    >>> x = np.linspace(0, 1, 40)
    >>> y = x + rng.normal(0, 0.1, 40)
    >>> band = bootstrap_band(x, y, [0.5], 50, np.random.default_rng(2))
    >>> bool(band[0, 0] < 0.5 < band[0, 1])
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    at = np.asarray(at, dtype=float)
    coarse = (
        np.linspace(at.min(), at.max(), 25) if at.size > 25 else np.sort(at)
    )
    draws = np.full((n_boot, coarse.size), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        draws[b] = lowess(x[idx], y[idx], coarse, iterations=1)
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    band = np.full((at.size, 2), np.nan)
    has = np.isfinite(draws).any(axis=0)
    if has.any():
        q = np.nanquantile(draws[:, has], [lo, hi], axis=0)
        for k in range(2):
            band[:, k] = np.interp(at, coarse[has], q[k])
    return band
