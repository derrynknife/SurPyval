"""Conditional survival and quantiles of the regression models.

``cs(x, given, Z)`` (#581) is the univariate models' conditional survival
with the covariates of the regression models' ``sf(x, Z)``: the chance a
unit with covariates ``Z`` that has survived to ``given`` survives a
further ``x``. It is taken from the model's own cumulative hazard, as
:math:`\\exp(-(H(given + x \\mid Z) - H(given \\mid Z)))`, which stays
exact in the far tail where the ratio of survival functions underflows to
``0 / 0``.

``quantiles_by_inversion`` is the parametric regression models'
``qf(p, Z)`` (#571): the time at which the model's cumulative hazard
reaches :math:`-\\log(1 - p)`, solved for every probability at once.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.utils.numeric import solve_bracketed

# Expanding a bracket doubles its step: 2000 doublings pass the largest
# float from any start, so a search that has not reached its target by
# then never will (a cure fraction, or a hazard that stops growing).
_MAX_DOUBLINGS = 2000


class ConditionalSurvivalMixin:
    """``cs(x, given, Z)`` for a regression model with ``Hf(x, Z, ...)``.

    The host's ``Hf`` pairs rows of ``Z`` with the times as its ``sf``
    does; any further arguments of ``cs`` (``stratum``, ``group``,
    ``grid``, ...) are passed to it unchanged.
    """

    Hf: Callable[..., Any]

    def cs(
        self,
        x: npt.ArrayLike,
        given: npt.ArrayLike,
        Z: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        r"""
        Conditional survival: the probability that a unit with covariates
        ``Z`` that has survived to ``given`` survives a further ``x``,

        .. math::
            R(x \mid given, Z) = \frac{R(given + x \mid Z)}{R(given \mid Z)}
            = e^{-(H(given + x \mid Z) - H(given \mid Z))},

        the univariate models' ``cs(x, given)`` with the covariates of
        :meth:`sf` (#581).

        Parameters
        ----------
        x : array like or scalar
            The further durations.
        given : array like or scalar
            The ages already survived, broadcast against ``x`` (one age
            per unit and one ``x`` for all, say).
        Z : array like or DataFrame
            The covariates, paired with the times as :meth:`sf` pairs them:
            one row per element of ``x`` / ``given``, or a single row for
            all of them.
        *args, **kwargs
            Any further arguments of this model's :meth:`Hf` (``grid=True``
            for every time for every row; a Cox model's ``stratum``; a
            frailty model's ``group`` or ``frailty``).

        Returns
        -------
        cs : scalar or numpy array
            The conditional survival probabilities.

        Notes
        -----
        The ratio is taken from the model's cumulative hazard, which keeps
        it exact where :math:`R(given)` underflows (a unit far into the
        tail). Where :math:`R(given \mid Z) = 0` the conditional survival
        is undefined and ``nan`` is returned, as for the univariate
        models. A fleet's expected failures over a horizon, built on this,
        are :func:`surpyval.forecast`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)

        Two units, aged 2 and 8, surviving one more unit of time:

        >>> model.cs(1.0, [2.0, 8.0], [[0], [1]]).round(4)
        array([0.9362, 0.6843])
        >>> s = model.sf([3.0, 9.0], [[0], [1]])
        >>> (s / model.sf([2.0, 8.0], [[0], [1]])).round(4)
        array([0.9362, 0.6843])
        """
        xs, gs = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(given, dtype=float)
        )
        end = self.Hf(xs + gs, Z, *args, **kwargs)
        start = self.Hf(gs, Z, *args, **kwargs)
        with np.errstate(invalid="ignore"):
            out = np.exp(np.asarray(start, dtype=float) - end)
        return out[()] if np.ndim(out) == 0 else out


def quantiles_by_inversion(
    Hf: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    p: npt.NDArray,
    support: tuple[float, float],
    start: npt.NDArray,
) -> npt.NDArray:
    """The times at which ``Hf(t, k)``, the cumulative hazard of problem
    ``k``, reaches ``-log(1 - p[k])``, for every ``k`` at once.

    ``p`` is in [0, 1] or ``nan`` (``nan`` gives ``nan``); ``p = 0`` is
    the start of the ``support`` and ``p = 1`` its end. Each search starts
    at ``start[k]`` (a finite guess inside the support, such as the
    baseline's quantile) and steps outward, doubling its step, until it
    brackets the target; a target never reached is ``inf`` (a cure
    fraction, or a hazard that stops growing). The brackets are then
    solved together to ``rtol=1e-12`` (:func:`solve_bracketed`).
    """
    lower, upper = (float(v) for v in support)
    out = np.full(p.shape, np.nan)
    out[p == 0] = lower
    out[p == 1] = upper
    k = np.flatnonzero((p > 0) & (p < 1))
    if not k.size:
        return out
    target = -np.log1p(-p[k])

    def gap(t: npt.NDArray, sel: npt.NDArray) -> npt.NDArray:
        with np.errstate(all="ignore"):
            H = np.asarray(Hf(t, k[sel]), dtype=float)
        return H - target[sel]

    guess = np.clip(np.asarray(start, dtype=float)[k], lower, upper)
    guess = np.where(np.isfinite(guess), guess, max(lower, 0.0))
    g_mid = gap(guess, np.arange(k.size))
    lo, g_lo = guess.copy(), g_mid.copy()
    hi, g_hi = guess.copy(), g_mid.copy()
    # Step down from the guess where it is past the target, up where it
    # is short of it; the start of the support has no hazard, so it ends
    # a downward search.
    _expand(gap, lo, g_lo, np.flatnonzero(g_mid > 0), -1, lower, target)
    _expand(gap, hi, g_hi, np.flatnonzero(g_mid < 0), +1, upper, target)
    # A bracket that ran off the floats: the quantile is past the largest
    # float on that side (or never reached, above).
    found = np.where(lo == -np.inf, -np.inf, np.inf)
    found = np.where(g_lo == 0, lo, np.where(g_hi == 0, hi, found))
    open_ = (g_lo < 0) & (g_hi > 0) & np.isfinite(lo) & np.isfinite(hi)
    if open_.any():
        sel = np.flatnonzero(open_)
        found[sel] = solve_bracketed(
            lambda t, s: gap(t, sel[s]),
            lo[sel],
            hi[sel],
            g_lo[sel],
            g_hi[sel],
            xtol=1e-300,
            rtol=1e-12,
        )
    out[k] = found
    return out


def _expand(
    gap: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    end: npt.NDArray,
    g_end: npt.NDArray,
    active: npt.NDArray,
    direction: int,
    limit: float,
    target: npt.NDArray,
) -> None:
    """Move ``end`` (in place, with its ``g_end``) away from the guess in
    ``direction`` until ``gap`` changes sign or ``end`` reaches ``limit``.
    At a finite ``limit`` (the start or end of the support) the gap is
    known without evaluating: ``-target`` at the start, where nothing has
    happened yet."""
    step = np.maximum(np.abs(end), 1.0)
    for _ in range(_MAX_DOUBLINGS):
        if not active.size:
            return
        with np.errstate(over="ignore", invalid="ignore"):
            moved = end[active] + direction * step[active]
        at_limit = (moved <= limit) if direction < 0 else (moved >= limit)
        moved = np.where(at_limit, limit, moved)
        g = gap(moved, active)
        if direction < 0 and np.isfinite(limit):
            g = np.where(at_limit, -target[active], g)
        end[active], g_end[active] = moved, g
        crossed = (g <= 0) if direction < 0 else (g >= 0)
        stuck = at_limit | ~np.isfinite(moved)
        step[active] *= 2.0
        active = active[~(crossed | stuck)]
