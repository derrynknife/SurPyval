"""A quantile solver for a positive continuous distribution without a
closed-form inverse that keeps its digits in both tails."""

from __future__ import annotations

from typing import Any

import numpy as onp


def solve_log_tails(
    dist: Any, p: Any, s: Any, params: tuple, start: Any
) -> Any:
    """The time at which ``F = p`` (and ``S = s = 1 - p``, given
    separately so that neither rounds).

    Newton's method in ``y = ln x`` on the log of the smaller tail (``ln
    F`` where ``p <= s``, else ``ln S``), from ``start``, with the slope
    ``x f / F`` (or ``-x f / S``) from the distribution's ``log_df``;
    where it has not reached the target to 1e-12, bisection on ``ln x``
    over 1400 units around the start (which needs only that the tail is
    monotone) and a few more Newton steps.
    """
    p = onp.asarray(p, dtype=float)
    s = onp.asarray(s, dtype=float)
    lower = p <= s
    with onp.errstate(all="ignore"):
        target = onp.log(onp.where(lower, p, s))
        start = onp.asarray(start, dtype=float)
        y = onp.log(onp.where(onp.isfinite(start) & (start > 0), start, 1.0))

        def tail(y: Any) -> Any:
            x = onp.exp(y)
            return onp.where(
                lower, dist.log_ff(x, *params), dist.log_sf(x, *params)
            )

        def newton(y: Any, steps: int) -> Any:
            for _ in range(steps):
                current = tail(y)
                slope = onp.exp(y + dist.log_df(onp.exp(y), *params) - current)
                slope = onp.where(lower, slope, -slope)
                step = (current - target) / slope
                step = onp.clip(
                    onp.where(onp.isfinite(step), step, 0.0), -2.0, 2.0
                )
                y = y - step
                if onp.all(
                    onp.abs(step) <= 4e-16 * onp.maximum(1.0, onp.abs(y))
                ):
                    break
            return y

        y = newton(y, 60)
        missed = ~(
            onp.abs(tail(y) - target)
            <= 1e-12 * onp.maximum(1.0, onp.abs(target))
        )
        if onp.any(missed):
            centre = onp.where(onp.isfinite(y), y, 0.0)
            lo, hi = centre - 700.0, centre + 700.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                value = tail(mid)
                # ln F rises with x, ln S falls
                above = onp.where(lower, value > target, value < target)
                hi = onp.where(above, mid, hi)
                lo = onp.where(above, lo, mid)
            y = onp.where(missed, newton(0.5 * (lo + hi), 4), y)
        out = onp.exp(y)
    out = onp.where(p <= 0, 0.0, onp.where(s <= 0, onp.inf, out))
    return onp.where(onp.isnan(p) | onp.isnan(s), onp.nan, out)
