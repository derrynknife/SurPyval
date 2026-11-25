"""Conformal lower bounds on a survival time, for any model.

"We are 90% sure this unit lasts at least ``L``": a lower bound on each new
unit's life from any fitted model's predictions, made honest by checking
the model on data it has not seen. The idea is split conformal prediction
(Vovk, Gammerman and Shafer 2005; Lei, G'Sell, Rinaldo, Tibshirani and
Wasserman 2018), with censoring handled as in Candès, Lei and Ren (2023).

How it works, in words. Set aside a *calibration* set the model was not
fitted to. For each of its units, read the model's own predicted CDF at
the unit's time, ``u_i = F_i(x_i)`` (the predicted chance of failing by
then). If the model were right these would be uniform; whether it is or
not, the calibration set shows how low they actually fall. Take the level
``eta`` that only a share ``alpha`` of them fall below, and give each new
unit the time at which its own predicted CDF reaches ``eta``:
``L = Q(eta | Z)``. A unit lasts at least ``L`` exactly when its ``u`` is
at least ``eta``, which, because new units are like the calibration units,
happens with probability at least ``1 - alpha``. A model that is too
optimistic gets a lower ``eta`` and so lower, honest bounds; a good model
gets ``eta`` near ``alpha`` and bounds near its own ``alpha`` quantiles.

**What is guaranteed.** *Marginal* coverage: over new units drawn like
the calibration units, and over the calibration set itself, at least
``1 - alpha`` of the units outlive their bounds. It holds whatever the
model, right or wrong, and in finite samples. It is not a guarantee for
each covariate value: units of one kind may be covered more often and
another kind less, averaging to ``1 - alpha``. The units must be
exchangeable (drawn from the same population, independently of the
calibration set); a population that drifts breaks it.

**Censoring.** A unit censored at ``x_i`` lasted at least ``x_i``, so its
``u`` at the time it was censored, ``F_i(x_i)``, is at most what it would
have been at its failure. ``method="observed"`` (the default) uses those
values as they are. Every unit's value is then at most its true one, so
the bound can only be lower than it would be with every failure seen: the
guarantee holds exactly, in finite samples, at the price of bounds that
are lower than they need be when many units are censored early (before
their ``alpha`` quantile; censoring later does not matter). This is the
conservative bound of Candès, Lei and Ren (2023), applied with the
observed times. ``method="ipcw"`` instead weights each failure by the
inverse of the chance of its being seen, ``1 / G(x_i-)`` (the Kaplan-Meier
of the censoring times), as if the censored units' failures were filled
in. Its bounds are less conservative, but its coverage is only
approximately ``1 - alpha``, and only when censoring does not depend on
the covariates or the life.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from surpyval.utils.validation import check_option
from surpyval.utils.validation import validate_1d as _as_1d

from .pit import pit_values

__all__ = ["ConformalLowerBound", "conformal_lower_bound"]

_METHODS = ("observed", "ipcw")


def _rows(Z: Any) -> tuple[Any, int]:
    if isinstance(Z, DataFrame):
        return Z, len(Z)
    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, arr.shape[0]


def _invert(cdf: Any, eta: float, m: int) -> npt.NDArray:
    """The smallest ``t >= 0`` with ``cdf(t) >= eta`` for each of ``m``
    rows, by bisection; ``inf`` where the CDF never reaches ``eta``."""
    if eta <= 0:
        return np.zeros(m)
    hi = np.ones(m)
    reached = np.asarray(cdf(hi), dtype=float) >= eta
    with np.errstate(over="ignore"):
        for _ in range(1100):
            if reached.all():
                break
            hi = np.where(reached, hi, hi * 2.0)
            if np.all(np.isinf(hi) | reached):
                break
            reached = np.asarray(cdf(hi), dtype=float) >= eta
    lo = np.zeros(m)
    out = np.where(reached, np.nan, np.inf)
    ok = reached.copy()
    for _ in range(200):
        if not ok.any():
            break
        mid = 0.5 * (lo + hi)
        at = np.asarray(cdf(np.where(ok, mid, 1.0)), dtype=float) >= eta
        hi = np.where(ok & at, mid, hi)
        lo = np.where(ok & ~at, mid, lo)
    out[ok] = hi[ok]
    return out


def _quantiles(model: Any, eta: float, Z: Any) -> npt.NDArray:
    """Each row's predicted ``eta`` quantile, ``inf{t: F(t | Z) >= eta}``."""
    if Z is None:
        qf = getattr(model, "qf", None)
        if qf is not None and 0 < eta < 1:
            try:
                return np.atleast_1d(np.asarray(qf(eta), dtype=float))
            except (TypeError, ValueError, NotImplementedError):
                pass
        return _invert(lambda t: np.asarray(model.ff(t), dtype=float), eta, 1)
    Z_in, m = _rows(Z)
    qf = getattr(model, "qf", None)
    if qf is not None and 0 < eta < 1:
        try:
            out = np.asarray(qf(np.full(m, eta), Z_in), dtype=float).ravel()
            if out.shape[0] == m:
                # A step-function baseline (Cox) gives nan where the CDF
                # never reaches eta; the bound there is inf, as by
                # inversion.
                gap = np.isnan(out)
                if gap.any():
                    rows = np.flatnonzero(gap)
                    Z_gap = (
                        Z_in.iloc[rows]
                        if isinstance(Z_in, DataFrame)
                        else Z_in[rows]
                    )
                    out[gap] = _invert(
                        lambda t: pit_values(model, t, Z_gap), eta, rows.size
                    )
                return out
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            pass

    def cdf(t: npt.NDArray) -> npt.NDArray:
        return pit_values(model, t, Z_in)

    return _invert(cdf, eta, m)


class ConformalLowerBound:
    """
    A model's lower bounds on life, calibrated by split conformal
    prediction. From :func:`conformal_lower_bound`.

    Attributes
    ----------
    model : object
        The fitted model.
    alpha : float
        At least ``1 - alpha`` of new units outlive their bounds
        (marginally, over units and calibration sets).
    method : str
        ``"observed"`` or ``"ipcw"``.
    level : float
        ``eta``: each new unit's bound is the time its predicted CDF
        reaches it. Near ``alpha`` for a well-calibrated model, lower for
        an optimistic one, higher for a pessimistic one.
    n_calibration : int
        The number of calibration units.
    n_censored : int
        How many of them were censored.
    scores : ndarray
        Each calibration unit's predicted CDF at its time.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval.metrics import conformal_lower_bound
    >>> rng = np.random.default_rng(0)
    >>> x = 10 * rng.weibull(2.0, 199)
    >>> model = Weibull.from_params([10.0, 2.0])
    >>> bound = conformal_lower_bound(model, x, alpha=0.1)
    >>> round(bound.level, 4), bound.lower().round(3)
    (0.1006, array([3.257]))
    """

    def __init__(
        self,
        model: Any,
        alpha: float,
        method: str,
        level: float,
        scores: npt.NDArray,
        n_censored: int,
    ) -> None:
        self.model = model
        self.alpha = float(alpha)
        self.method = method
        self.level = float(level)
        self.scores = np.asarray(scores, dtype=float)
        self.n_calibration = int(self.scores.shape[0])
        self.n_censored = int(n_censored)

    def lower(self, Z: npt.ArrayLike | None = None) -> npt.NDArray:
        """Each new unit's lower bound on its life.

        Parameters
        ----------
        Z : array_like or pandas.DataFrame, optional
            The new units' covariates, one row each, for a model with
            covariates. Without them (a model with none) one bound is
            returned.

        Returns
        -------
        ndarray
            The time each unit's predicted CDF reaches ``level``: 0 where
            ``level`` is 0, ``inf`` where the CDF never reaches it.
        """
        return _quantiles(self.model, self.level, Z)

    def __repr__(self) -> str:
        return (
            "Conformal lower bound ({:g}% of new units outlive it, "
            "marginally; method {!r})\n"
            "  calibrated on {} units ({} censored): each bound is the "
            "time the unit's predicted CDF reaches {:.4g}".format(
                100 * (1 - self.alpha),
                self.method,
                self.n_calibration,
                self.n_censored,
                self.level,
            )
        )


def _level(
    scores: npt.NDArray,
    weights: npt.NDArray,
    alpha: float,
    n: int,
) -> float:
    """The largest score level ``eta`` with ``sum_{s_i < eta} p_i + 1 /
    (n + 1) <= alpha``, where the ``p_i`` are the weights scaled to sum
    to ``n / (n + 1)``; 0 if there is none. With unit weights it is the
    ``floor(alpha (n + 1))``-th smallest score."""
    keep = weights > 0
    s = scores[keep]
    w = weights[keep]
    p = w / w.sum() * n / (n + 1.0)
    order = np.argsort(s, kind="mergesort")
    s, p = s[order], p[order]
    # Mass strictly below each distinct score.
    uniq, first = np.unique(s, return_index=True)
    below = np.concatenate([[0.0], np.cumsum(p)])[first]
    ok = below + 1.0 / (n + 1.0) <= alpha * (1 + 1e-12)
    if not ok.any():
        return 0.0
    return float(uniq[ok][-1])


def conformal_lower_bound(
    model: Any,
    x: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    *,
    Z: npt.ArrayLike | None = None,
    alpha: float = 0.1,
    method: str = "observed",
) -> ConformalLowerBound:
    """
    Lower bounds on new units' lives that at least ``1 - alpha`` of them
    outlive, from any fitted model, calibrated on held-out data (split
    conformal prediction).

    The calibration units must not have been used to fit the model: split
    the data once, fit on one part, calibrate on the other. Each
    calibration unit's predicted CDF at its time sets the level ``eta``
    such that a new unit's bound, the time its own predicted CDF reaches
    ``eta``, is outlived by at least ``1 - alpha`` of new units. See the
    module notes for what is and is not guaranteed.

    Parameters
    ----------
    model : object
        A fitted model with a CDF: any model with ``ff(x)`` without ``Z``,
        or ``sf(x, Z)`` paired with the rows of ``Z`` with it (the
        regression families, ``CoxPH``, the ``surpyval.ml`` tree and
        forest, joint models). Its ``qf`` gives the bounds where it has
        one; otherwise they are found from its CDF.
    x : array_like
        The calibration units' times (failures, or when censored).
    c : array_like, optional
        0 for a failure, 1 for right censored (default all 0). Left and
        interval censoring are refused.
    n : array_like, optional
        Each row's count (default 1).
    Z : array_like or pandas.DataFrame, optional
        The calibration units' covariates, for a model with covariates.
    alpha : float, default 0.1
        The share of new units allowed to fail before their bounds: 0.1
        gives 90% lower bounds. Needs at least ``1 / alpha - 1``
        calibration units (9 for 0.1), or every bound is 0.
    method : {"observed", "ipcw"}, default "observed"
        How censored units count. ``"observed"`` uses their predicted CDF
        at their censoring time, which makes the guarantee exact and the
        bounds conservative when many are censored early. ``"ipcw"``
        weights the failures by the inverse of the Kaplan-Meier chance of
        being seen: tighter bounds, approximate coverage, assuming
        censoring that is independent of the covariates and the life.

    Returns
    -------
    ConformalLowerBound
        ``lower(Z)`` gives the new units' bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullPH
    >>> from surpyval.metrics import conformal_lower_bound
    >>> rng = np.random.default_rng(1)
    >>> Z = rng.normal(size=(400, 1))
    >>> t = 10 * rng.weibull(1.5, 400) * np.exp(-0.5 * Z[:, 0] / 1.5)
    >>> fit = WeibullPH.fit(x=t[:200], Z=Z[:200])
    >>> bound = conformal_lower_bound(fit, t[200:], Z=Z[200:], alpha=0.1)
    >>> bound.lower([[-1.0], [0.0], [1.0]]).round(2)
    array([3.38, 2.4 , 1.71])
    """
    check_option("method", method, _METHODS)
    if not (0 < float(alpha) < 1):
        raise ValueError(
            "'alpha' must be strictly between 0 and 1 (0.1 gives 90% lower "
            "bounds); got {!r}".format(alpha)
        )
    x_arr = _as_1d(x, "x")
    N = x_arr.shape[0]
    c_arr = np.zeros(N) if c is None else _as_1d(c, "c")
    if c_arr.shape[0] != N:
        raise ValueError("'c' must have one flag per row of 'x'")
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "'c' must be 0 (failure) or 1 (right censored); left and "
            "interval censoring are not supported: a lower bound needs a "
            "time each unit is known to have lasted"
        )
    n_arr = np.ones(N, dtype=int) if n is None else _as_1d(n, "n")
    if n_arr.shape[0] != N or np.any(n_arr < 0) or np.any(n_arr % 1 != 0):
        raise ValueError("'n' must be one whole, non-negative count per row")
    n_arr = n_arr.astype(int)
    if Z is not None:
        Z_in, m = _rows(Z)
        if m != N:
            raise ValueError(f"'Z' has {m} rows for {N} rows of 'x'")
    else:
        Z_in = None
    u = pit_values(model, x_arr, Z_in)
    if np.any(~np.isfinite(u)):
        raise ValueError(
            "the model's predicted CDF is missing at {} of the {} "
            "calibration rows (a missing covariate gives nan); drop or fix "
            "them".format(int(np.sum(~np.isfinite(u))), N)
        )
    u = np.repeat(u, n_arr)
    cens = np.repeat(c_arr, n_arr)
    total = int(n_arr.sum())
    if total < 1:
        raise ValueError("there are no calibration units")
    if method == "observed":
        w = np.ones(total)
    else:
        from surpyval.metrics.validation import _ipcw

        _, _, g_inv = _ipcw(np.repeat(x_arr, n_arr), cens, None, None)
        w = np.where(cens == 0, g_inv, 0.0)
        if not np.any(w > 0):
            raise ValueError(
                "method='ipcw' needs at least one failure in the "
                "calibration units; use method='observed'"
            )
    if np.floor(alpha * (total + 1)) < 1:
        warnings.warn(
            "{} calibration units are too few for alpha={:g}: at least {} "
            "are needed, so every bound is 0. Use more calibration units "
            "or a larger alpha.".format(
                total, alpha, int(np.ceil(1 / alpha - 1))
            ),
            UserWarning,
            stacklevel=2,
        )
    eta = _level(u, w, float(alpha), total)
    return ConformalLowerBound(
        model, float(alpha), method, eta, u, int(cens.sum())
    )
