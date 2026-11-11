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
``step_quantiles`` is that of the semi-parametric models, whose
predicted curve is a step function (#662): the first step time at which
the predicted failure probability reaches ``p``, ``nan`` where it never
does, as the non-parametric estimates' ``qf``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.numeric import solve_bracketed
from surpyval.utils.shapes import check_paired_rows, keeps_query_shape
from surpyval.utils.validation import warn_outside_unit_interval

# Expanding a bracket doubles its step: 2000 doublings pass the largest
# float from any start, so a search that has not reached its target by
# then never will (a cure fraction, or a hazard that stops growing).
_MAX_DOUBLINGS = 2000

#: A predicted failure probability within this of ``p`` counts as reaching
#: it in a step curve's quantile, the non-parametric ``qf``'s tolerance:
#: round-off in the curve would otherwise put the quantile a step late.
STEP_QF_TOL = 1e-9


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
        try:
            end = self.Hf(xs + gs, Z, *args, **kwargs)
        except TypeError as error:
            # An argument cs passed on that Hf does not take is cs's
            # mistake to report (#653), not Hf's.
            if "Hf()" not in str(error) or "argument" not in str(error):
                raise
            raise TypeError(str(error).replace("Hf()", "cs()")) from None
        start = self.Hf(gs, Z, *args, **kwargs)
        with np.errstate(invalid="ignore"):
            out = np.exp(np.asarray(start, dtype=float) - end)
        return out[()] if np.ndim(out) == 0 else out


def paired_probabilities(
    p: npt.ArrayLike, rows: npt.NDArray, grid: bool = False
) -> tuple[npt.NDArray, npt.NDArray, tuple[int, int] | None]:
    """The probabilities ``p`` (flat) and covariate ``rows`` of a
    regression model's ``qf(p, Z)``, one problem each: paired as ``sf``
    pairs times with rows (one row for every ``p``, or one ``p`` for
    every row), or, with ``grid``, every ``p`` for every row, with the
    ``(len(rows), len(p))`` shape to give the result. A probability
    outside [0, 1] is ``nan``, with one warning (#611)."""
    u = np.asarray(p, dtype=float).reshape(-1)
    rows = np.asarray(rows, dtype=float)
    shape = None
    if grid:
        shape = (rows.shape[0], u.size)
        u = np.tile(u, shape[0])
        rows = np.repeat(rows, shape[1], axis=0)
    else:
        check_paired_rows(u.size, rows.shape[0])
        size = u.size if rows.shape[0] == 1 else max(u.size, rows.shape[0])
        u = np.broadcast_to(u, (size,))
        rows = np.broadcast_to(rows, (size, rows.shape[1]))
    u = np.where(warn_outside_unit_interval(u), np.nan, u)
    return u, rows, shape


def unique_rows(rows: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """The distinct covariate rows and, for each row, which it is: the
    curves of a step model are evaluated once per distinct row."""
    rows = np.asarray(rows, dtype=float)
    if rows.shape[0] == 0 or rows.shape[1] == 0:
        return rows[:1], np.zeros(rows.shape[0], dtype=int)
    uniq, inverse = np.unique(rows, axis=0, return_inverse=True)
    return uniq, np.ravel(inverse)


def step_quantiles(
    F: npt.NDArray, times: npt.NDArray, p: npt.NDArray
) -> npt.NDArray:
    """The first of ``times`` at which the failure probability ``F[k]``
    of problem ``k`` (the predicted curve at the step times, a row each,
    or one row for all) reaches ``p[k]``: ``nan`` where it never does (a
    curve that ends above ``1 - p``, as right censoring leaves it), and
    for a ``nan`` ``p``. A curve within ``STEP_QF_TOL`` of ``p`` counts
    as reaching it, and ``p = 0`` is the first time the curve rises above
    0, as for the non-parametric ``qf``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.regression._prediction import (
    ...     step_quantiles,
    ... )
    >>> F = np.array([[0.1, 0.5, 0.7], [0.2, 0.3, 0.4]])
    >>> step_quantiles(F, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))
    array([ 2., nan])
    """
    p = np.asarray(p, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full(p.shape, np.nan)
    if not times.size or not p.size:
        return out
    target = np.maximum(p - STEP_QF_TOL, np.finfo(float).tiny)
    with np.errstate(invalid="ignore"):
        hit = np.asarray(F, dtype=float) >= target[:, None]
    found = hit.any(axis=1)
    first = np.argmax(hit, axis=1)
    out = np.where(found, times[first], np.nan)
    return np.where(np.isnan(p), np.nan, out)


def step_draws(
    H: npt.NDArray, times: npt.NDArray, U: npt.NDArray
) -> npt.NDArray:
    """Inverse-transform draws from a step cumulative hazard: for each
    uniform ``U`` (the survival probability the draw is to have), the
    first of ``times`` at which the non-decreasing ``H`` (the predicted
    cumulative hazard at those times) reaches ``-log(U)``. Where it never
    does -- the curve ends above ``U``, as right censoring leaves it --
    the draw is ``inf``: the unit outlives the data. A ``nan`` in ``H``
    (a missing covariate or stratum) gives ``nan`` draws.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.regression._prediction import step_draws
    >>> H = np.array([0.1, 0.1, 0.5])
    >>> step_draws(H, np.array([1.0, 2.0, 3.0]), np.array([0.95, 0.7, 0.5]))
    array([ 1.,  3., inf])
    """
    H = np.asarray(H, dtype=float)
    times = np.asarray(times, dtype=float)
    U = np.asarray(U, dtype=float)
    if np.isnan(H).any():
        return np.full(U.shape, np.nan)
    target = -np.log(U)
    idx = np.searchsorted(H, target, side="left")
    out = np.full(U.shape, np.inf)
    inside = idx < times.size
    out[inside] = times[idx[inside]]
    return out


def quantiles_by_inversion(
    Hf: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    p: npt.NDArray,
    support: tuple[float, float],
    start: npt.NDArray,
    hf: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] | None = None,
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

    That tolerance is on the time, relative to the time itself, so where
    the quantile is far from 0 but close to where the hazard starts to
    accumulate -- an additive hazards model's support starting at -27, a
    small ``p`` there -- the cumulative hazard reached was off by 1e-5 of
    the target (#828). With the hazard ``hf(t, k)`` each answer is then
    taken by Newton's method on the cumulative hazard (a step is kept only
    where it stays in the bracket and brings the cumulative hazard closer
    to its target), which is accurate relative to the target.
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
        if hf is not None:
            found[sel] = _newton_finish(
                gap, hf, k, sel, found[sel], lo[sel], hi[sel]
            )
    out[k] = found
    return out


def _newton_finish(
    gap: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    hf: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    k: npt.NDArray,
    sel: npt.NDArray,
    t: npt.NDArray,
    lo: npt.NDArray,
    hi: npt.NDArray,
    steps: int = 3,
) -> npt.NDArray:
    """Newton's steps on ``gap`` (the cumulative hazard less its target)
    from the bracketed answers ``t`` of the problems ``k[sel]``, with the
    hazard ``hf`` its derivative; a step is kept only where it stays in
    ``[lo, hi]`` and makes ``|gap|`` smaller.

    The bracket is widened by ``1e-10`` of ``|t|`` on each side: the
    cumulative hazard is computed in batches, and a vectorised ``exp`` or
    ``log`` may round the same time differently in a batch of another
    size, so the sign the bracket was found with can flip within a few
    ulps of its ends. An answer at an end of the bracket would then have
    no step that stays in it."""
    t = t.copy()
    slack = 1e-10 * np.abs(t)
    lo, hi = lo - slack, hi + slack
    with np.errstate(all="ignore"):
        g = gap(t, sel)
        for _ in range(steps):
            h = np.asarray(hf(t, k[sel]), dtype=float)
            new = t - g / h
            ok = np.isfinite(new) & (h > 0) & (new >= lo) & (new <= hi)
            if not ok.any():
                break
            g_new = gap(np.where(ok, new, t), sel)
            better = ok & (np.abs(g_new) < np.abs(g))
            if not better.any():
                break
            t = np.where(better, new, t)
            g = np.where(better, g_new, g)
    return t


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


class RestrictedMeanMixin:
    """``rmst``, ``rmst_cb`` and ``rmst_diff`` at covariates ``Z`` for a
    parametric regression model: the area under ``sf(t, Z)`` from 0 to
    ``tau``, by adaptive quadrature, with delta-method (and bootstrap)
    bounds from the fitted parameters' covariance.

    The host provides ``model.sf(x, Zc, *params)`` at centred covariate
    rows, ``distribution.support``, ``_prepare_Z``, ``_n_covariates``,
    ``_centred``, ``_eval_params``, ``_wald_state``, ``_check_inference``
    and ``_refuse_weighted_method``, as the parametric regression models
    do.
    """

    model: Any
    distribution: Any
    data: Any
    _prepare_Z: Callable[..., Any]
    _n_covariates: Callable[..., int]
    _centred: Callable[..., Any]
    _eval_params: Callable[..., Any]
    _wald_state: Callable[..., Any]
    _check_inference: Callable[..., None]
    _refuse_weighted_method: Callable[..., None]

    def _rmst_tau(self, tau: Any, what: str) -> npt.NDArray:
        """The horizons as a flat array: the largest observed time where
        ``tau`` is None; a negative or infinite one is refused."""
        if tau is None:
            if getattr(self, "data", None) is None:
                raise ValueError(
                    f"{what}: this model holds no data to take the default "
                    "tau (the largest observed time) from; pass tau."
                )
            x = np.asarray(self.data.x, dtype=float)
            tau = float(np.max(x[np.isfinite(x)]))
        t = np.atleast_1d(np.asarray(tau, dtype=float)).reshape(-1)
        bad = ~np.isnan(t) & ~((t >= 0) & np.isfinite(t))
        if np.any(bad):
            raise ValueError(
                "{}: tau = {} is not a finite time at or above 0. The "
                "restricted mean is the area under sf(t, Z) from 0 to tau; "
                "for the area to infinity use mean(Z).".format(
                    what, t[bad].tolist()
                )
            )
        return t

    def _rmst_pairs(
        self, tau: Any, Z: Any, what: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The horizons and covariate rows paired, one row each (one
        ``tau`` for every row, or one row for every ``tau``)."""
        from surpyval.utils.shapes import check_paired_rows, covariate_rows

        t = self._rmst_tau(tau, what)
        rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
        check_paired_rows(t.size, rows.shape[0], grid=False)
        n = max(t.size, rows.shape[0])
        t = np.broadcast_to(t, (n,)).copy()
        rows = np.ascontiguousarray(
            np.broadcast_to(np.asarray(rows, float), (n, rows.shape[1]))
        )
        return t, rows

    def _rmst_at(
        self, taus: npt.NDArray, rows_c: npt.NDArray, params: Any
    ) -> npt.NDArray:
        """``int_0^tau_i S(t | row_i) dt`` at the parameters ``params``
        for centred covariate rows ``rows_c``, as ``tau_i int_0^1 S(tau_i
        s | row_i) ds`` by one vector-valued adaptive quadrature (Gauss-
        Kronrod, to 1e-12 relative); ``S = 1`` below the support."""
        from scipy.integrate import quad_vec

        out = np.full(taus.shape, np.nan)
        ok = ~np.isnan(taus) & np.all(np.isfinite(rows_c), axis=1)
        if not ok.any():
            return out
        t_ok, r_ok = taus[ok], rows_c[ok]
        lower = float(self.distribution.support[0])
        inside = lower + 1.0 if np.isfinite(lower) else 0.0

        def integrand(s: float) -> npt.NDArray:
            t = t_ok * s
            below = t < lower
            with np.errstate(all="ignore"):
                S = np.asarray(
                    self.model.sf(np.where(below, inside, t), r_ok, *params),
                    dtype=float,
                )
            return t_ok * np.where(below, 1.0, S)

        with np.errstate(all="ignore"):
            value, _ = quad_vec(
                integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-12, norm="max"
            )
        out[ok] = value
        return out

    def _rmst_wald(
        self,
        taus: npt.NDArray,
        rows: npt.NDArray,
        robust: bool,
        cluster: Any,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The restricted means and their delta-method standard errors."""
        from surpyval.utils.linalg import delta_method_se

        params, center, cov = self._wald_state(robust, cluster, "wald")
        rows_c = np.asarray(self._centred(rows, center), dtype=float)
        value = self._rmst_at(taus, rows_c, params)
        with np.errstate(all="ignore"):
            se = delta_method_se(
                lambda p: self._rmst_at(taus, rows_c, p), params, cov
            )
        return value, np.asarray(se, dtype=float)

    @staticmethod
    def _rmst_logit_cb(
        value: npt.NDArray,
        se: npt.NDArray,
        taus: npt.NDArray,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The Wald bounds on the logit of ``value / tau``, which keep
        them in ``(0, tau)``; at a ``value`` of 0 or ``tau`` (or a zero
        standard error) the bound is the value."""
        from scipy.special import expit, ndtri

        from surpyval.utils.linalg import bound_signs

        alpha, signs = bound_signs(alpha_ci, bound)
        with np.errstate(all="ignore"):
            psi = np.log(value) - np.log(taus - value)
            se_psi = se * (1.0 / value + 1.0 / (taus - value))
            out = taus[:, None] * expit(
                psi[:, None] + signs * ndtri(1.0 - alpha) * se_psi[:, None]
            )
        edge = ~np.isfinite(psi) | (se == 0)
        out = np.where(edge[:, None], value[:, None], out)
        out = np.where(np.isnan(value)[:, None], np.nan, out)
        return out if bound == "two-sided" else out[:, 0]

    def rmst(
        self,
        tau: float | None,
        Z: Any,
        alpha_ci: float = 0.05,
        robust: bool = False,
        cluster: npt.ArrayLike | None = None,
    ) -> dict:
        r"""
        Restricted mean survival time up to ``tau`` of a unit with
        covariates ``Z``, with inference: the univariate models' ``rmst``
        with the covariates of :meth:`sf`.

        It is the expected time such a unit lives within the first
        ``tau`` of its life, the area under its predicted survival curve,

        .. math::
            \mu(\tau \mid Z) = \int_0^\tau S(t \mid Z)\, dt
            = E[\min(T, \tau) \mid Z],

        taken by adaptive quadrature (to 1e-12 relative). The standard
        error is the delta method's, from the gradient of
        :math:`\mu(\tau \mid Z)` in the parameters (central differences)
        and their covariance; the interval is that Wald bound on the
        logit of :math:`\mu/\tau`, so it stays in :math:`(0, \tau)`
        (:meth:`rmst_cb` with ``method="wald"``).

        Parameters
        ----------
        tau : scalar or None
            The horizon; ``None`` takes the largest observed time.
        Z : array like or DataFrame
            One covariate row, or several (a result for each).
        alpha_ci : float, optional
            Significance level for the interval (default 0.05).
        robust : bool, optional
            ``True`` uses the robust (sandwich) covariance, as
            :meth:`cb` does with it. Default ``False``.
        cluster : array_like, optional
            With ``robust=True``, a cluster label for each row given to
            the fit.

        Returns
        -------
        dict
            ``{"rmst", "se", "lower", "upper", "tau"}``, as the univariate
            models' ``rmst``: numbers for one row of ``Z``, arrays (one
            entry per row) for several.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> res = model.rmst(8.0, [1])
        >>> print(round(res["rmst"], 4), round(res["se"], 4))
        4.8975 0.3033
        """
        from surpyval.utils.validation import check_alpha_ci

        check_alpha_ci(alpha_ci)
        if tau is not None and np.size(tau) != 1:
            raise ValueError(
                "rmst takes one tau; rmst_cb bounds several at once."
            )
        self._check_inference()
        self._refuse_weighted_method("wald")
        t, rows = self._rmst_pairs(tau, Z, "rmst")
        value, se = self._rmst_wald(t, rows, robust, cluster)
        cb = self._rmst_logit_cb(value, se, t, alpha_ci, "two-sided")
        out: dict[str, Any] = {
            "rmst": value,
            "se": se,
            "lower": cb[:, 0],
            "upper": cb[:, 1],
            "tau": float(t[0]),
        }
        if value.size == 1:
            return {k: float(np.ravel(v)[0]) for k, v in out.items()}
        return out

    @keeps_query_shape
    def rmst_cb(
        self,
        tau: npt.ArrayLike | None,
        Z: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
        robust: bool = False,
        cluster: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the restricted mean survival time
        :meth:`rmst` at covariates ``Z``, at one horizon or several.

        Parameters
        ----------
        tau : array like, scalar or None
            The horizons, paired with the rows of ``Z`` as :meth:`sf`
            pairs times with them (one row for every ``tau``, or one
            ``tau`` for every row); ``None`` takes the largest observed
            time.
        Z : array like or DataFrame
            The covariates.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        method : {'wald', 'bootstrap'}, optional
            ``'wald'`` (the default) is the delta method on the logit of
            ``rmst / tau``; ``'bootstrap'`` the BCa interval of the RMST
            over the parametric bootstrap refits of :meth:`cb`. There is
            no likelihood-ratio bound on it for a regression model:
            ``'lr'`` is refused.
        n_boot : int, optional
            The number of bootstrap refits (``method='bootstrap'`` only).
            Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the bootstrap (``method='bootstrap'`` only), as for
            :meth:`cb`.
        robust : bool, optional
            ``True`` uses the robust (sandwich) covariance; ``method``
            must be ``"wald"``. Default ``False``.
        cluster : array_like, optional
            With ``robust=True``, a cluster label for each row given to
            the fit.

        Returns
        -------
        numpy array
            The bound(s), one row per pair of ``tau`` and row of ``Z``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.rmst_cb(8.0, [[0], [1]]).round(3)
        array([[5.746, 6.735],
               [4.287, 5.467]])
        """
        from surpyval.utils.validation import (
            BOUNDS,
            check_alpha_ci,
            check_option,
        )

        from ._bootstrap import bound_method
        from ._inference import check_robust_request

        check_alpha_ci(alpha_ci)
        method = bound_method(method)
        if method == "lr":
            raise ValueError(
                "rmst_cb has no likelihood-ratio bound for a regression "
                "model: use method='wald' (the default) or "
                "method='bootstrap'."
            )
        check_robust_request(robust, cluster, method)
        self._refuse_weighted_method(method)
        self._check_inference()
        check_option("bound", bound, BOUNDS)
        t, rows = self._rmst_pairs(tau, Z, "rmst_cb")
        if method == "bootstrap":
            from ._bootstrap import function_bounds, refits

            fits = refits(self, n_boot, random_state)
            fits.warn()

            def value(p: npt.NDArray, center: Any = None) -> npt.NDArray:
                rows_c = np.asarray(self._centred(rows, center), float)
                return self._rmst_at(t, rows_c, p)

            return function_bounds(self, fits, value, None, alpha_ci, bound)
        value_, se = self._rmst_wald(t, rows, robust, cluster)
        return self._rmst_logit_cb(value_, se, t, alpha_ci, bound)

    def rmst_diff(
        self,
        tau: float | None,
        Z_a: Any,
        Z_b: Any,
        alpha_ci: float = 0.05,
        robust: bool = False,
        cluster: npt.ArrayLike | None = None,
    ) -> dict:
        r"""
        Compare the restricted mean survival time of units with
        covariates ``Z_a`` and ``Z_b``: :func:`surpyval.rmst_diff` for two
        covariate rows of one model.

        The two RMSTs come from the same parameters, so they are not
        independent: the standard error of the difference is the delta
        method's on the difference itself,
        :math:`\sqrt{g^\top \Sigma g}` with :math:`g` the gradient of
        :math:`\mu(\tau \mid Z_a) - \mu(\tau \mid Z_b)`.

        Parameters
        ----------
        tau : scalar or None
            The common horizon; ``None`` takes the largest observed time.
        Z_a : array like or DataFrame
            The first covariate row.
        Z_b : array like or DataFrame
            The second covariate row.
        alpha_ci : float, optional
            Significance level for the interval and test (default 0.05).
        robust : bool, optional
            ``True`` uses the robust (sandwich) covariance. Default
            ``False``.
        cluster : array_like, optional
            With ``robust=True``, a cluster label for each row given to
            the fit.

        Returns
        -------
        dict
            ``{"difference", "se", "lower", "upper", "p_value", "ratio",
            "rmst_a", "rmst_b", "tau"}``, as :func:`surpyval.rmst_diff`:
            ``difference`` is ``rmst_a - rmst_b``, ``lower`` and ``upper``
            are ``difference +- z se`` and ``p_value`` is the two-sided
            z-test of ``difference == 0``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> res = model.rmst_diff(8.0, [0], [1])
        >>> print(round(res["difference"], 4), round(res["se"], 4))
        1.3943 0.3375
        """
        from scipy.stats import norm

        from surpyval.utils.linalg import delta_method_se
        from surpyval.utils.validation import check_alpha_ci

        check_alpha_ci(alpha_ci)
        if tau is not None and np.size(tau) != 1:
            raise ValueError("rmst_diff takes one tau.")
        self._check_inference()
        self._refuse_weighted_method("wald")
        t, rows_a = self._rmst_pairs(tau, Z_a, "rmst_diff")
        _, rows_b = self._rmst_pairs(t[0], Z_b, "rmst_diff")
        if rows_a.shape[0] != 1 or rows_b.shape[0] != 1:
            raise ValueError(
                "rmst_diff compares one covariate row with another: pass "
                "one row as Z_a and one as Z_b."
            )
        params, center, cov = self._wald_state(robust, cluster, "wald")
        ra = np.asarray(self._centred(rows_a, center), dtype=float)
        rb = np.asarray(self._centred(rows_b, center), dtype=float)

        def diff_of(p: npt.NDArray) -> npt.NDArray:
            return self._rmst_at(t, ra, p) - self._rmst_at(t, rb, p)

        mu_a = float(self._rmst_at(t, ra, params)[0])
        mu_b = float(self._rmst_at(t, rb, params)[0])
        diff = mu_a - mu_b
        with np.errstate(all="ignore"):
            se = float(np.ravel(delta_method_se(diff_of, params, cov))[0])
        z_crit = norm.ppf(1 - alpha_ci / 2)
        p_value = (
            float(2.0 * norm.sf(abs(diff) / se)) if se > 0 else float("nan")
        )
        return {
            "difference": float(diff),
            "se": se,
            "lower": float(diff - z_crit * se),
            "upper": float(diff + z_crit * se),
            "p_value": p_value,
            "ratio": float(mu_a / mu_b) if mu_b != 0 else float("nan"),
            "rmst_a": mu_a,
            "rmst_b": mu_b,
            "tau": float(t[0]),
        }
