"""The coefficient path of a penalised fit, and the penalty an information
criterion chooses: :func:`penalty_path`.

A penalised fit (``fit_penalised``) needs a ``penalizer``. The path fits
the model over a grid of penalties, from the smallest one that sets every
coefficient to 0 down to a small fraction of it, and records each fit's
coefficients, effective degrees of freedom, log-likelihood, AIC and BIC.
The AIC and BIC count the effective degrees of freedom of each fit (Gray
1992; the number of non-zero coefficients for the lasso), so they trade
the gain in likelihood against how much the penalty lets the
coefficients move; the penalty with the smallest criterion is the one
:meth:`PenaltyPath.best` returns.

The choice is a statistic of the data, as the penalty is: the chosen
model still has no standard errors or confidence bounds (choosing the
penalty from the same data is one more reason for that).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

#: The criteria :meth:`PenaltyPath.best` chooses by.
CRITERIA = ("aic", "bic")


class PenaltyPath:
    """
    The fits of :func:`penalty_path`, one per penalty, largest penalty
    first.

    Attributes
    ----------
    penalizers : ndarray
        The penalties, decreasing.
    l1_ratio : float
        The ``l1_ratio`` of every fit.
    models : list
        The fitted models, in the order of ``penalizers``.
    coefficient_names : list of str
        The names of the penalised coefficients.
    coefficients : ndarray
        One row per penalty, one column per penalised coefficient, on the
        covariates' own scale.
    effective_df, neg_ll, aic, bic : ndarray
        Each fit's effective degrees of freedom (``penalty.effective_df``,
        which the criteria count), negative log-likelihood (the partial
        likelihood for a Cox model) and its ``aic()`` and ``bic()``.

    Examples
    --------
    >>> from surpyval import WeibullAFT, penalty_path
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> path = penalty_path(
    ...     WeibullAFT.fit_penalised, df["week"], df[["fin", "age", "prio"]],
    ...     c=1 - df["arrest"], l1_ratio=1, n_penalties=5,
    ... )
    >>> len(path), path.coefficients.shape
    (5, (5, 3))
    >>> round(path.best("bic").penalty.effective_df, 6)  # 2 + 3 coefs
    5.0
    """

    def __init__(
        self, penalizers: npt.NDArray, l1_ratio: float, models: list
    ) -> None:
        self.penalizers = np.asarray(penalizers, dtype=float)
        self.l1_ratio = float(l1_ratio)
        self.models = list(models)
        names, _ = models[0]._penalised_coefficients()
        self.coefficient_names = list(names)
        self.coefficients = np.array(
            [m._penalised_coefficients()[1] for m in models], dtype=float
        )
        self.effective_df = np.array(
            [m.penalty.effective_df for m in models], dtype=float
        )
        self.neg_ll = np.array([m.neg_ll() for m in models], dtype=float)
        self.aic = np.array([m.aic() for m in models], dtype=float)
        self.bic = np.array([m.bic() for m in models], dtype=float)

    def __len__(self) -> int:
        return self.penalizers.size

    def __repr__(self) -> str:
        best = {c: self.best_penalizer(c) for c in CRITERIA}
        return (
            "Penalty path: {} penalties from {:.4g} to {:.4g}, l1_ratio "
            "{:g}; smallest AIC at penalizer {:.4g}, smallest BIC at "
            "penalizer {:.4g}".format(
                len(self),
                self.penalizers[0],
                self.penalizers[-1],
                self.l1_ratio,
                best["aic"],
                best["bic"],
            )
        )

    def table(self) -> pd.DataFrame:
        """
        One row per penalty: the ``penalizer``, the ``effective_df``, the
        number of ``nonzero`` coefficients, ``neg_ll``, ``aic`` and
        ``bic``.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "penalizer": self.penalizers,
                "effective_df": self.effective_df,
                "nonzero": (self.coefficients != 0).sum(axis=1),
                "neg_ll": self.neg_ll,
                "aic": self.aic,
                "bic": self.bic,
            }
        )

    def _index(self, criterion: str) -> int:
        from surpyval.utils.validation import check_option

        check_option("criterion", criterion, CRITERIA)
        values = getattr(self, criterion)
        if not np.any(np.isfinite(values)):
            raise ValueError(
                "No fit on the path has a finite {}; check the fits' "
                "warnings.".format(criterion)
            )
        # The first (largest) penalty among equal values: the simpler fit.
        return int(np.nanargmin(np.where(np.isfinite(values), values, np.nan)))

    def best_penalizer(self, criterion: str = "bic") -> float:
        """The penalty whose fit has the smallest ``criterion``
        (``"aic"`` or ``"bic"``)."""
        return float(self.penalizers[self._index(criterion)])

    def best(self, criterion: str = "bic") -> Any:
        """The fitted model with the smallest ``criterion`` (``"aic"`` or
        ``"bic"``, the default). Like every penalised model it has no
        standard errors or bounds; for inference, refit the unpenalised
        model on the covariates it keeps."""
        return self.models[self._index(criterion)]

    def plot(self, ax: Axes | None = None) -> Axes:
        """The coefficient path: each coefficient against the penalty (on
        a log scale, decreasing to the right), with the penalties AIC and
        BIC choose marked."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for j, name in enumerate(self.coefficient_names):
            ax.plot(self.penalizers, self.coefficients[:, j], label=name)
        for criterion, style in (("aic", ":"), ("bic", "--")):
            ax.axvline(
                self.best_penalizer(criterion),
                color="grey",
                linestyle=style,
                label=f"smallest {criterion.upper()}",
            )
        ax.set_xscale("log")
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("penalizer")
        ax.set_ylabel("coefficient")
        ax.legend(fontsize="small")
        return ax


def _all_zero(model: Any) -> bool:
    """Whether every penalised coefficient (penalty above 0) is 0."""
    values = np.asarray(model._penalised_coefficients()[1], dtype=float)
    pen = np.asarray(model.penalty.penalizer, dtype=float)
    # A model with several coefficients per column (one row per cause)
    # repeats the columns' penalties.
    penalised = np.tile(pen, values.size // max(pen.size, 1)) > 0
    return bool(np.all(values[penalised] == 0))


def _largest_penalty(
    fit: Callable[[float, float], Any], l1_ratio: float
) -> float:
    """The smallest penalty at which the fit with ``l1_ratio`` (at least
    0.001, as glmnet's for a ridge path) sets every coefficient to 0, to
    six digits (relative 1e-6, from above): by doubling from 1, then
    bisection on the log scale."""
    a = max(l1_ratio, 1e-3)
    lo, hi = None, None
    lam = 1.0
    for _ in range(80):
        if _all_zero(fit(lam, a)):
            hi = lam
            if lo is not None:
                break
            lam /= 2
        else:
            lo = lam
            if hi is not None:
                break
            lam *= 2
    if lo is None or hi is None:
        raise ValueError(
            "Could not find a penalty that sets every coefficient to 0 "
            "(searched from {:.3g} to {:.3g}); give `penalizers`.".format(
                2.0**-80, 2.0**80
            )
        )
    while hi / lo > 1 + 1e-6:
        mid = float(np.sqrt(lo * hi))
        if _all_zero(fit(mid, a)):
            hi = mid
        else:
            lo = mid
    return hi


def penalty_path(
    fit_penalised: Callable[..., Any],
    *args: Any,
    penalizers: npt.ArrayLike | None = None,
    l1_ratio: float = 0.0,
    n_penalties: int = 20,
    min_ratio: float = 1e-3,
    penalty_factor: npt.ArrayLike | None = None,
    **kwargs: Any,
) -> PenaltyPath:
    """
    Fit a penalised model over a path of penalties, and choose the penalty
    by AIC or BIC.

    ``fit_penalised`` is any model's penalised fit (``CoxPH.fit_penalised``,
    ``WeibullAFT.fit_penalised``, ``CoxPH.fit_tvc_penalised``,
    ``ProportionalIntensityHPP.fit_penalised``, ...); ``args`` and
    ``kwargs`` are passed to it (the data, ``standardise``, ``tie_method``,
    ...), with each penalty of the path as its ``penalizer``. Every fit
    starts afresh, so each is the same model ``fit_penalised`` gives at that
    penalty.

    Without ``penalizers`` the path is glmnet's: ``n_penalties`` penalties
    evenly spaced on the log scale from the smallest penalty that sets every
    coefficient to 0 (found by search, to six digits; for a ridge
    path, the one for ``l1_ratio=0.001``, as glmnet) down to ``min_ratio``
    times it. A ridge fit at that penalty is small but not 0 (glmnet
    reports its first ridge fit as 0). Columns with a ``penalty_factor``
    of 0 are left out of the search: they are never 0.

    Parameters
    ----------
    fit_penalised : callable
        A penalised fit, such as ``CoxPH.fit_penalised``.
    *args, **kwargs
        The arguments of ``fit_penalised`` other than ``penalizer`` and
        ``l1_ratio``.
    penalizers : array_like, optional
        The penalties, instead of the default path; sorted, largest
        first.
    l1_ratio : float, optional
        The ``l1_ratio`` of every fit: 0 (ridge, the default) to 1 (lasso).
    n_penalties : int, optional
        The number of penalties of the default path. Default 20.
    min_ratio : float, optional
        The smallest penalty of the default path as a fraction of the
        largest. Default 0.001.
    penalty_factor : array_like, optional
        One factor per covariate column, multiplying the penalty for that
        column's coefficient (0 leaves it unpenalised), as glmnet's
        ``penalty.factor``. Default 1 for every column.

    Returns
    -------
    PenaltyPath
        Each fit, with ``table()``, ``best(criterion)``,
        ``best_penalizer(criterion)`` and ``plot()``.

    Examples
    --------
    >>> from surpyval import CoxPH, penalty_path
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> cols = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    >>> path = penalty_path(
    ...     CoxPH.fit_penalised, df["week"], df[cols], c=1 - df["arrest"],
    ...     l1_ratio=1, n_penalties=10,
    ... )
    >>> path.table()[["penalizer", "nonzero"]].round(4).head(3)
       penalizer  nonzero
    0     0.0886        0
    1     0.0411        4
    2     0.0191        6
    >>> round(path.best_penalizer("bic"), 4)
    0.0411
    """
    if "penalizer" in kwargs:
        raise ValueError(
            "penalty_path sets each fit's `penalizer` itself; give the "
            "penalties as `penalizers` (or leave them to the default "
            "path)."
        )
    a = float(l1_ratio)
    factor = None if penalty_factor is None else np.asarray(penalty_factor)
    if factor is not None and (
        factor.ndim != 1
        or not np.all(np.isfinite(factor))
        or np.any(factor < 0)
    ):
        raise ValueError(
            "`penalty_factor` must be one finite value at least 0 per "
            "covariate column; got {!r}.".format(penalty_factor)
        )

    def fit(lam: float, ratio: float) -> Any:
        pen: Any = lam if factor is None else lam * factor
        return fit_penalised(*args, penalizer=pen, l1_ratio=ratio, **kwargs)

    if penalizers is None:
        if int(n_penalties) != n_penalties or n_penalties < 2:
            raise ValueError(
                "`n_penalties` must be a whole number at least 2; got "
                "{!r}.".format(n_penalties)
            )
        if not 0 < min_ratio < 1:
            raise ValueError(
                "`min_ratio` must be between 0 and 1; got {!r}.".format(
                    min_ratio
                )
            )
        with warnings.catch_warnings():
            # The search's fits are not reported; the path's are.
            warnings.simplefilter("ignore")
            top = _largest_penalty(fit, a)
        grid = top * np.geomspace(1.0, min_ratio, int(n_penalties))
    else:
        grid = np.sort(np.atleast_1d(np.asarray(penalizers, dtype=float)))[
            ::-1
        ]
        if grid.size == 0 or not np.all(np.isfinite(grid)) or grid[-1] < 0:
            raise ValueError(
                "`penalizers` must be finite values at least 0; got "
                "{!r}.".format(penalizers)
            )
    models = [fit(float(lam), a) for lam in grid]
    return PenaltyPath(grid, a, models)
