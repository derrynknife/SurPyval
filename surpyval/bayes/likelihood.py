"""The censored and truncated log-likelihood at many parameter points at
once.

This is the likelihood ``fit`` maximises, term for term (observed, right,
left and interval censored rows, counts, left and right truncation, and
the offset, limited-failure and zero-inflated forms), written so that it
takes a whole array of parameter points and returns one log-likelihood
per point. It is built only from each distribution's public functions
(``log_df``, ``log_sf``, ``log_ff``, ``ff``, ``sf``), which broadcast over
parameter arrays, and it is kept separate from the fitters so that
existing fits are unaffected. The tests check that it equals the
fitters' own likelihood at random points for every continuous
distribution and every kind of data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.surpyval_data import SurpyvalData

# The largest number of (point, row) evaluations held at once.
_CHUNK = 2_000_000


def _log1mexp(d: npt.NDArray) -> npt.NDArray:
    """``log(1 - exp(-d))`` for ``d >= 0``, accurate at both ends."""
    with np.errstate(divide="ignore", invalid="ignore"):
        small = d < np.log(2.0)
        safe_small = np.where(small, d, 1.0)
        safe_big = np.where(small, 1.0, d)
        return np.where(
            small,
            np.log(-np.expm1(-safe_small)),
            np.log1p(-np.exp(-safe_big)),
        )


def model_parameter_names(
    dist: Any, offset: bool = False, lfp: bool = False, zi: bool = False
) -> list[str]:
    """The parameter vector of a model, in the order ``Parametric`` uses:
    ``gamma`` (with an offset), the distribution's parameters, ``lfp_p``
    (limited failure population) and ``f0`` (zero inflation).

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.bayes.likelihood import model_parameter_names
    >>> model_parameter_names(sp.Weibull, offset=True, lfp=True)
    ['gamma', 'alpha', 'beta', 'lfp_p']
    """
    names = list(dist.parameter_names)
    if offset:
        names = ["gamma"] + names
    if lfp:
        names.append("lfp_p")
    if zi:
        names.append("f0")
    return names


class LogLikelihood:
    """The log-likelihood of a distribution on fixed data, as a function
    of the parameter vector, evaluated at many points at once.

    Build it with :func:`log_likelihood_fn`. Calling it with an array of
    shape ``(..., k)`` (``k`` the length of :attr:`parameter_names`)
    returns an array of shape ``(...)``.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes import LogLikelihood
    >>> ll = LogLikelihood(sp.Exponential, sp.SurpyvalData(x=[1.0, 2.0]))
    >>> float(ll(np.array([0.5])).round(4))
    -2.8863
    """

    def __init__(
        self,
        dist: Any,
        data: SurpyvalData,
        offset: bool = False,
        lfp: bool = False,
        zi: bool = False,
    ) -> None:
        self.dist = dist
        self.data = data
        self.offset = bool(offset)
        self.lfp = bool(lfp)
        self.zi = bool(zi)
        self.parameter_names = model_parameter_names(dist, offset, lfp, zi)
        self.k = len(self.parameter_names)
        self._lo = float(dist.support[0])
        if not np.isfinite(self._lo):
            self._lo = -np.inf
        d = data
        self._rows = (
            len(d.x_o) + len(d.x_r) + len(d.x_l) + len(d.x_il) + 1
        ) + len(d.tl_unique)

    def __repr__(self) -> str:
        return (
            f"LogLikelihood({self.dist.name}, parameters "
            f"{self.parameter_names})"
        )

    # -- the pieces -----------------------------------------------------
    def _split(self, P: npt.NDArray) -> tuple:
        """(dist params as columns, gamma, lfp_p, f0), each (N, 1)."""
        k_dist = self.dist.k
        j = 0
        if self.offset:
            gamma = P[:, 0:1]
            j = 1
        else:
            gamma = np.zeros((P.shape[0], 1))
        dist_params = [P[:, j + i : j + i + 1] for i in range(k_dist)]
        j += k_dist
        if self.lfp:
            p = P[:, j : j + 1]
            j += 1
        else:
            p = np.ones((P.shape[0], 1))
        f0 = P[:, j : j + 1] if self.zi else np.zeros((P.shape[0], 1))
        return dist_params, gamma, p, f0

    def _censored_x(self, x: npt.NDArray, gamma: npt.NDArray) -> Any:
        xs = x[None, :] - gamma
        if np.isfinite(self._lo):
            xs = np.maximum(xs, self._lo)
        return xs

    def _observed_terms(
        self, x: npt.NDArray, dp: list, gamma: Any, p: Any, f0: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The log density of exact rows at ``x`` (none of them a zero of a
        zero-inflated model), ``(N, m)``, and the log of the mass that can
        fail at a positive time, ``log(p - f0)``, ``(N,)``: a row's term
        is their sum."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = self.dist.log_df(x[None, :] - gamma, *dp)
            mass = np.log(p - f0)[:, 0]
        return ld, mass

    def _observed(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_o, d.n_o
        if len(x) == 0:
            return 0.0
        if self.zi:
            zero = x == 0
            n0 = float(np.sum(n[zero]))
            with np.errstate(divide="ignore"):
                zero_part = n0 * np.log(f0[:, 0]) if n0 else 0.0
            x, n = x[~zero], n[~zero]
        else:
            zero_part = 0.0
        if len(x) == 0:
            return zero_part
        ld, mass = self._observed_terms(x, dp, gamma, p, f0)
        return (ld * n).sum(axis=1) + n.sum() * mass + zero_part

    def _right_terms(
        self, x: npt.NDArray, dp: list, gamma: Any, p: Any, f0: Any
    ) -> npt.NDArray:
        """The log-likelihood of each right-censored row at ``x``."""
        xs = self._censored_x(x, gamma)
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.lfp:
                return np.log1p(-f0) + self.dist.log_sf(xs, *dp)
            return np.log((1 - p) + (p - f0) * self.dist.sf(xs, *dp))

    def _right(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_r, d.n_r
        if len(x) == 0:
            return 0.0
        terms = self._right_terms(x, dp, gamma, p, f0)
        return (terms * n).sum(axis=1)

    def _left_terms(
        self, x: npt.NDArray, dp: list, gamma: Any, p: Any, f0: Any
    ) -> npt.NDArray:
        """The log-likelihood of each left-censored row at ``x``."""
        xs = self._censored_x(x, gamma)
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.zi:
                return self.dist.log_ff(xs, *dp) + np.log(p)
            return np.log(f0 + (p - f0) * self.dist.ff(xs, *dp))

    def _left(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_l, d.n_l
        if len(x) == 0:
            return 0.0
        terms = self._left_terms(x, dp, gamma, p, f0)
        return (terms * n).sum(axis=1)

    def _windows(
        self,
        xl: npt.NDArray,
        xr: npt.NDArray,
        n: npt.NDArray,
        dp: list,
        gamma: Any,
        p: Any,
        f0: Any,
    ) -> Any:
        """Sum of ``n log P(xl < T <= xr)`` over the windows."""
        if len(n) == 0:
            return 0.0
        terms = self._window_terms(xl, xr, dp, gamma, p, f0)
        return (terms * n).sum(axis=1)

    def _window_terms(
        self,
        xl: npt.NDArray,
        xr: npt.NDArray,
        dp: list,
        gamma: Any,
        p: Any,
        f0: Any,
    ) -> npt.NDArray:
        """``log P(xl < T <= xr)`` of each window, ``(N, m)``."""
        dist = self.dist
        lo_finite = np.isfinite(xl)[None, :]
        hi_finite = np.isfinite(xr)[None, :]
        l_shift = xl[None, :] - gamma
        r_shift = xr[None, :] - gamma
        lo_eval = lo_finite & (l_shift > self._lo)
        # stand-ins inside the support where a bound is absent; their
        # values are never used
        inside = self._inside(xl, xr, gamma)
        l_safe = np.where(lo_eval, l_shift, inside)
        r_safe = np.where(hi_finite, r_shift, inside)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.lfp or self.zi:
                upper = np.where(
                    hi_finite,
                    f0 * (xr[None, :] >= 0) + (p - f0) * dist.ff(r_safe, *dp),
                    1.0,
                )
                lower = np.where(
                    lo_finite,
                    f0 * (xl[None, :] > 0)
                    + (p - f0) * np.where(lo_eval, dist.ff(l_safe, *dp), 0.0),
                    0.0,
                )
                terms = np.log(np.maximum(upper - lower, 0.0))
            else:
                log_sl = np.where(lo_eval, dist.log_sf(l_safe, *dp), 0.0)
                log_sr = np.where(hi_finite, dist.log_sf(r_safe, *dp), -np.inf)
                log_fl = np.where(lo_eval, dist.log_ff(l_safe, *dp), -np.inf)
                log_fr = np.where(hi_finite, dist.log_ff(r_safe, *dp), 0.0)
                # upper tail: log S(l) + log(1 - S(r)/S(l))
                tail = log_sl + _log1mexp(np.maximum(log_sl - log_sr, 0.0))
                # elsewhere: log F(r) + log(1 - F(l)/F(r))
                body = log_fr + _log1mexp(np.maximum(log_fr - log_fl, 0.0))
                use_tail = lo_eval & (log_sl < np.log(0.5))
                terms = np.where(use_tail, tail, body)
        return terms

    def _inside(self, xl: npt.NDArray, xr: npt.NDArray, gamma: Any) -> Any:
        """A point inside the support, per parameter point."""
        present = np.concatenate([xl[np.isfinite(xl)], xr[np.isfinite(xr)]])
        if present.size:
            return np.full_like(gamma, float(present.max())) - gamma
        lo, hi = self._lo, float(self.dist.support[1])
        if np.isfinite(lo) and np.isfinite(hi):
            return np.full_like(gamma, 0.5 * (lo + hi))
        if np.isfinite(lo):
            return np.full_like(gamma, lo + 1.0)
        return np.zeros_like(gamma)

    def _evaluate(self, P: npt.NDArray) -> npt.NDArray:
        d = self.data
        dp, gamma, p, f0 = self._split(P)
        out = (
            self._observed(dp, gamma, p, f0)
            + self._right(dp, gamma, p, f0)
            + self._left(dp, gamma, p, f0)
            + self._windows(d.x_il, d.x_ir, d.n_i, dp, gamma, p, f0)
            - self._windows(
                d.tl_unique, d.tr_unique, d.n_t_unique, dp, gamma, p, f0
            )
        )
        out = np.broadcast_to(np.asarray(out, dtype=float), (P.shape[0],))
        return np.where(np.isnan(out), -np.inf, out)

    def __call__(self, params: npt.ArrayLike) -> npt.NDArray:
        P = np.asarray(params, dtype=float)
        if P.shape[-1] != self.k:
            raise ValueError(
                f"params must have {self.k} values in its last axis "
                f"({', '.join(self.parameter_names)}); got shape {P.shape}"
            )
        shape = P.shape[:-1]
        P = P.reshape(-1, self.k)
        step = max(1, _CHUNK // max(self._rows, 1))
        out = np.empty(P.shape[0])
        for start in range(0, P.shape[0], step):
            out[start : start + step] = self._evaluate(P[start : start + step])
        return out.reshape(shape)

    def pointwise(self, params: npt.ArrayLike) -> npt.NDArray:
        """
        Each data row's own log-likelihood at each parameter point.

        Row ``j`` (in the order of the data's ``x``, ``c``, ``n`` and
        ``t``) contributes its censoring kind's term (density, survival,
        failure probability or the probability of its interval) less the
        log probability of its own truncation window, for one unit: the
        row's count is not applied. So ``pointwise(P) @ data.n`` is the
        log-likelihood, to rounding.

        Parameters
        ----------
        params : array like
            Parameter points, shape ``(..., k)``, as for calling the
            likelihood.

        Returns
        -------
        numpy.ndarray
            Shape ``(...) + (rows,)``; ``-inf`` where a row is impossible
            at a point.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> from surpyval.bayes import LogLikelihood
        >>> data = sp.SurpyvalData(x=[1.0, 2.0], c=[0, 1], n=[3, 1])
        >>> ll = LogLikelihood(sp.Exponential, data)
        >>> ll.pointwise(np.array([0.5])).round(4)
        array([-1.1931, -1.    ])
        >>> bool(np.isclose(ll.pointwise(np.array([0.5])) @ data.n,
        ...                 ll(np.array([0.5]))))
        True
        """
        P = np.asarray(params, dtype=float)
        if P.shape[-1] != self.k:
            raise ValueError(
                f"params must have {self.k} values in its last axis "
                f"({', '.join(self.parameter_names)}); got shape {P.shape}"
            )
        shape = P.shape[:-1]
        P = P.reshape(-1, self.k)
        rows = len(self.data.c)
        everything = np.ones(rows, dtype=bool)
        out = np.empty((P.shape[0], rows))
        step = max(1, _CHUNK // max(rows, 1))
        for start in range(0, P.shape[0], step):
            out[start : start + step] = self._pointwise(
                P[start : start + step], everything
            )
        return out.reshape(shape + (rows,))

    def _pointwise(self, P: npt.NDArray, rows: npt.NDArray) -> npt.NDArray:
        """:meth:`pointwise` at the points ``P`` (``(N, k)``) for the rows
        selected by the mask ``rows``, ``(N, rows.sum())``."""
        d = self.data
        dp, gamma, p, f0 = self._split(P)
        column = np.cumsum(rows) - 1
        out = np.zeros((P.shape[0], int(rows.sum())))

        def take(mask: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
            keep = rows[mask]
            return keep, column[mask][keep]

        keep, idx = take(d.mask_o)
        if idx.size:
            x = d.x_o[keep]
            if self.zi:
                zero = x == 0
                with np.errstate(divide="ignore"):
                    out[:, idx[zero]] = np.log(f0)
                x, idx = x[~zero], idx[~zero]
            if idx.size:
                ld, mass = self._observed_terms(x, dp, gamma, p, f0)
                out[:, idx] = ld + mass[:, None]
        keep, idx = take(d.mask_r)
        if idx.size:
            out[:, idx] = self._right_terms(d.x_r[keep], dp, gamma, p, f0)
        keep, idx = take(d.mask_l)
        if idx.size:
            out[:, idx] = self._left_terms(d.x_l[keep], dp, gamma, p, f0)
        keep, idx = take(d.mask_i)
        if idx.size:
            out[:, idx] = self._window_terms(
                d.x_il[keep], d.x_ir[keep], dp, gamma, p, f0
            )
        keep, idx = take(d.truncated_mask)
        if idx.size:
            out[:, idx] -= self._window_terms(
                d.x_tl[keep], d.x_tr[keep], dp, gamma, p, f0
            )
        return np.where(np.isnan(out), -np.inf, out)


def log_likelihood_fn(
    dist: Any,
    x: npt.ArrayLike | SurpyvalData | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    *,
    tl: npt.ArrayLike | float | None = None,
    tr: npt.ArrayLike | float | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    offset: bool = False,
    lfp: bool = False,
    zi: bool = False,
) -> LogLikelihood:
    """
    The exact censored and truncated log-likelihood of ``dist`` on the
    data, as a function that takes many parameter points at once.

    It is the likelihood ``dist.fit`` maximises, for every kind of data
    ``fit`` takes (censoring flags ``c``, counts ``n``, truncation ``t`` or
    ``tl`` / ``tr``, intervals ``xl`` / ``xr``) and for the ``offset``,
    ``lfp`` and ``zi`` forms. Use it to explore the likelihood, or to
    hand SurPyval's likelihood to another sampler.

    Parameters
    ----------
    dist : ParametricFitter
        A continuous SurPyval distribution, e.g. ``surpyval.Weibull``.
    x, c, n, t, tl, tr, xl, xr
        The data, as for ``fit``; ``x`` may be a ``SurpyvalData``.
    offset, lfp, zi : bool, optional
        As for ``fit``: add ``gamma`` (first), ``lfp_p`` and ``f0`` (last)
        to the parameter vector.

    Returns
    -------
    LogLikelihood
        Called with an array of shape ``(..., k)``, in the order of its
        ``parameter_names``, it returns the log-likelihood at each point,
        shape ``(...)``.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes import log_likelihood_fn
    >>> ll = log_likelihood_fn(sp.Weibull, [2.1, 3.4, 4.05, 5.0],
    ...                        c=[0, 0, 0, 1])
    >>> ll.parameter_names
    ['alpha', 'beta']
    >>> ll(np.array([[4.0, 2.0], [5.0, 3.0]])).round(4)
    array([-6.4597, -6.3802])
    """
    if getattr(dist, "discrete", False):
        raise ValueError(
            f"log_likelihood_fn covers continuous distributions; "
            f"{dist.name} is discrete."
        )
    if isinstance(x, SurpyvalData):
        from surpyval.utils.surpyval_data import refuse_multi_state

        refuse_multi_state(x, "log_likelihood_fn")
        data = x
    else:
        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
    from surpyval.utils.weights import (
        BAYES_INSTEAD,
        BAYES_WHY,
        refuse_weighted_data,
    )

    refuse_weighted_data(data, "log_likelihood_fn", BAYES_WHY, BAYES_INSTEAD)
    return LogLikelihood(dist, data, offset=offset, lfp=lfp, zi=zi)


def log_posterior_and_grad(
    data: SurpyvalData,
    dist: Any,
    prior: Any = None,
    *,
    offset: bool = False,
    lfp: bool = False,
    zi: bool = False,
    fixed: dict[str, float] | None = None,
) -> Any:
    """
    The log-posterior of a Bayesian fit and its gradient at one point, for
    a gradient-based sampler: ``f(z) -> (value, gradient)``.

    ``z`` is the free parameters on their unconstrained scale (the log of
    a positive parameter, the logit of a proportion), in the order of
    ``f.free_names``. The value is the posterior ``fit_bayes`` uses (the
    log prior on that scale, its log-Jacobian included, plus this
    module's log-likelihood) to rounding; the data are checked and split
    once, here, so a call does no checking. See :mod:`surpyval.bayes._lean`
    for what a sampler may assume.

    Parameters
    ----------
    data : SurpyvalData
        The data, any censoring and truncation.
    dist : ParametricFitter
        A distribution ``fit_bayes`` takes.
    prior : optional
        What ``fit_bayes`` takes as ``priors`` (a dict, a joint prior), or
        a ``BoundPriors`` already bound to this fit.
    offset, lfp, zi, fixed
        As for ``fit_bayes``.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes.likelihood import log_posterior_and_grad
    >>> data = sp.SurpyvalData(x=[2.1, 3.4, 4.05, 5.0], c=[0, 0, 0, 1])
    >>> f = log_posterior_and_grad(data, sp.Weibull,
    ...                            {"beta": sp.prior.between(1, 5)})
    >>> f.free_names
    ['alpha', 'beta']
    >>> value, gradient = f(np.log([4.0, 2.0]))
    >>> round(value, 4), gradient.round(4)
    (-6.6897, array([1.1716, 1.7445]))
    """
    from surpyval.bayes._lean import LogPosterior
    from surpyval.bayes.fitting import (
        _check_dist,
        _validate_data,
        bayes_data,
        build_bound,
    )
    from surpyval.bayes.prior import BoundPriors

    _check_dist(dist, "auto")
    if getattr(dist, "discrete", False):
        raise ValueError(
            f"log_posterior_and_grad covers the continuous distributions; "
            f"{dist.name} is fitted in closed form (fit_bayes with a Beta "
            "prior)."
        )
    data = bayes_data("log_posterior_and_grad", data)
    _validate_data(dist, data, offset, zi)
    if isinstance(prior, BoundPriors):
        bound = prior
        if (bound.offset, bound.lfp, bound.zi) != (offset, lfp, zi):
            raise ValueError(
                "The BoundPriors were bound with offset, lfp and zi = "
                f"{bound.offset}, {bound.lfp}, {bound.zi}; pass the same "
                "here, or pass the priors themselves."
            )
    else:
        bound = build_bound(dist, data, prior, offset, lfp, zi, fixed)
    if bound.d == 0:
        raise ValueError(
            "Every parameter is fixed, so there is nothing to sample: use "
            f"{dist.name}.from_params(...)."
        )
    return LogPosterior(dist, data, bound)
