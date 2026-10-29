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

    def _observed(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_o, d.n_o
        if len(x) == 0:
            return 0.0
        dist = self.dist
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
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = dist.log_df(x[None, :] - gamma, *dp)
            mass = np.log(p - f0)[:, 0]
        return (ld * n).sum(axis=1) + n.sum() * mass + zero_part

    def _right(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_r, d.n_r
        if len(x) == 0:
            return 0.0
        xs = self._censored_x(x, gamma)
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.lfp:
                terms = np.log1p(-f0) + self.dist.log_sf(xs, *dp)
            else:
                terms = np.log((1 - p) + (p - f0) * self.dist.sf(xs, *dp))
        return (terms * n).sum(axis=1)

    def _left(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        d = self.data
        x, n = d.x_l, d.n_l
        if len(x) == 0:
            return 0.0
        xs = self._censored_x(x, gamma)
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.zi:
                terms = self.dist.log_ff(xs, *dp) + np.log(p)
            else:
                terms = np.log(f0 + (p - f0) * self.dist.ff(xs, *dp))
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
                    f0 * (xl[None, :] >= 0)
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
        return (terms * n).sum(axis=1)

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
        data = x
    else:
        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
    return LogLikelihood(dist, data, offset=offset, lfp=lfp, zi=zi)
