r"""
Aalen's non-parametric additive hazards model.

Each covariate adds to the hazard an amount that may change with time,

.. math::
    h(t \mid Z) = b_0(t) + b_1(t) Z_1 + \dots + b_p(t) Z_p,

with nothing assumed about the shape of any ``b_j(t)``. What the data
determine well are the *cumulative* regression functions
:math:`B_j(t) = \int_0^t b_j(s)\,ds`: at each event time the increments
are a least-squares fit of the events on the covariates of the units at
risk,

.. math::
    dB(t) = (X' Y X)^{-1} X' dN(t),

with ``X`` the rows at risk (a leading column of ones for the baseline
``B_0``), ``Y`` their counts and ``dN`` the events, and the variance of
``B(t)`` is the optional variation
:math:`\sum_{s \le t} (X'YX)^{-1} X' \mathrm{diag}(dN) X (X'YX)^{-1}`.
A slope of ``B_j`` is the covariate's excess hazard per unit at that
time; a ``B_j`` that bends shows an effect that changes. The model
contains the Lin and Ying model (:class:`~surpyval.AdditiveHazards`, a
straight ``B_j``) as the special case of constant effects.

The estimate is R's ``timereg::aalen`` (with ``robust = 0``) and
lifelines' ``AalenAdditiveFitter`` (without penalties). It stops before
the first event time at which ``X' Y X`` is singular: too few units are
at risk to separate the covariates.

References
----------
Aalen, O. O. (1989), "A linear regression model for the analysis of life
times", Statistics in Medicine 8, 907-925.

Martinussen, T. and Scheike, T. H. (2006), Dynamic Regression Models for
Survival Data, Springer, chapter 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

from .._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    warn_aliased,
)
from .._concordance import ConcordanceMixin
from .._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_quantiles,
    unique_rows,
)
from .._summary import coefficient_names, coefficient_table
from ..regression_data import (
    NoLikelihoodMixin,
    check_covariate_width,
    design_matrix_from_df,
    prepare_Z,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd

#: The name of the baseline's column of ``B``.
BASELINE = "baseline"

_BOUNDS = ("two-sided", "lower", "upper")
_ON = ("sf", "R", "ff", "F", "Hf")
_ON_ALIASES = {"R": "sf", "F": "ff"}


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
) -> tuple[npt.NDArray, ...]:
    x_arr, c_arr, n_arr, tl_arr, Z_arr = semi_parametric_inputs(
        x,
        Z,
        c,
        n,
        tl,
        censoring=(
            "Aalen's additive model supports only observed (c=0) and "
            "right-censored (c=1) data; for left or interval censored "
            "data fit a parametric additive hazards model (e.g. "
            "surpyval.WeibullAH)."
        ),
        truncation=(
            "Aalen's additive model takes left truncation (delayed entry) "
            "only: 'tl' must be one entry time per row, not a [tl, tr] "
            "pair."
        ),
    )
    if np.any(x_arr < 0) or np.any(tl_arr[np.isfinite(tl_arr)] < 0):
        raise ValueError(
            "Aalen's additive model integrates the hazard from time 0; "
            "every time 'x' and entry time 'tl' must be non-negative."
        )
    if not np.any(c_arr == 0):
        raise ValueError(
            "Aalen's additive model needs at least one event (c=0); with "
            "every observation censored nothing can be estimated."
        )
    return x_arr, c_arr, n_arr, tl_arr, Z_arr


def _check_penalty(coef_penalty: Any) -> float:
    try:
        value = float(coef_penalty)
    except (TypeError, ValueError):
        value = np.nan
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(
            "'coef_penalty' must be a finite number at or above 0 (0, "
            "the default, is no penalty); got {!r}.".format(coef_penalty)
        )
    return value


def _risk_set_sums(
    x: npt.NDArray,
    tl: npt.NDArray,
    values: npt.NDArray,
    times: npt.NDArray,
) -> npt.NDArray:
    """The sums of ``values`` over the rows at risk at each of ``times``:
    entered before it (``tl < t``) and not yet out (``x >= t``)."""

    def summed(keys: npt.NDArray) -> npt.NDArray:
        # the sum of the values over the rows with key >= each time
        order = np.argsort(keys)
        total = np.concatenate(
            [np.zeros((1,) + values.shape[1:]), np.cumsum(values[order], 0)]
        )
        return total[-1] - total[np.searchsorted(keys[order], times, "left")]

    return summed(x) - summed(tl)


def _estimate(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
    penalty: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, list]:
    """The event times, the increments ``dB`` and the increments of the
    covariance of ``B`` at them, and the inverses ``(X'YX + P)^{-1}``,
    to the first event time at which ``X'YX + P`` is singular."""
    q = X.shape[1]
    events = c == 0
    times = np.unique(x[events])
    XX = n[:, None, None] * X[:, :, None] * X[:, None, :]
    S2 = _risk_set_sums(x, tl, XX, times)
    bucket = np.searchsorted(times, x[events])
    E1 = np.zeros((times.size, q))
    np.add.at(E1, bucket, (n[events])[:, None] * X[events])
    E2 = np.zeros((times.size, q, q))
    np.add.at(E2, bucket, XX[events])
    P = np.diag(penalty)
    dB, dC, inverses = [], [], []
    for k in range(times.size):
        A = S2[k] + P
        if np.linalg.matrix_rank(A) < q:
            break
        inv = np.linalg.inv(A)
        dB.append(inv @ E1[k])
        dC.append(inv @ E2[k] @ inv)
        inverses.append(inv)
    kept = len(dB)
    return (
        times[:kept],
        np.reshape(dB, (kept, q)),
        np.reshape(dC, (kept, q, q)),
        inverses,
    )


def _epanechnikov(u: npt.NDArray) -> npt.NDArray:
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def _default_bandwidth(times: npt.NDArray) -> float:
    """A normal-reference bandwidth on the event times, as the Lin-Ying
    model's smoothed hazard takes (``1.06 sd m^(-1/5)``), on the scale of
    the times where they (nearly) coincide."""
    spread = float(np.std(times)) if times.size > 1 else 0.0
    scale = max(abs(float(np.mean(times))) if times.size else 1.0, 1.0)
    if spread <= 1e-8 * scale:
        spread = scale
    return max(1.06 * spread * max(times.size, 2) ** (-1 / 5), 1e-12)


class AalenAdditiveModel(
    ConditionalSurvivalMixin,
    NoLikelihoodMixin,
    ConcordanceMixin,
    SerialisableMixin,
):
    r"""
    A fitted Aalen additive hazards model, returned by
    :meth:`AalenAdditive.fit <surpyval.AalenAdditive>`.

    The model is ``h(t | Z) = b_0(t) + b(t)'Z`` with every ``b_j`` left
    to the data. ``B`` holds the cumulative regression functions
    :math:`B_j(t) = \int_0^t b_j(s) ds` at the event times ``x``, a
    column per term: the baseline first, then each covariate. A
    covariate's ``B_j(t)`` is the extra cumulative hazard a unit with one
    more unit of it has accumulated by ``t``; its slope is the extra
    hazard at ``t``.

    The predictions follow from ``B``: the cumulative hazard of a unit
    with covariates ``Z`` is ``H(t | Z) = B_0(t) + B(t)'Z`` (see
    :meth:`Hf` for how a falling estimate is handled), its survival
    ``exp(-H)``, and its hazard rate the kernel-smoothed slope
    (:meth:`smoothed_coefficients`).

    Examples
    --------
    On the Rossi recidivism data each covariate adds to the weekly hazard
    of arrest; by week 52 each prior conviction (``prio``) has added
    about 0.037 to the cumulative hazard, and financial aid has taken
    about 0.10 from it:

    >>> from surpyval import AalenAdditive
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> df["censored"] = 1 - df["arrest"]
    >>> model = AalenAdditive.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    ... )
    >>> model.summary()[["coef", "se(coef)", "p"]].round(4)
                 coef  se(coef)       p
    covariate
    baseline   0.6315    0.1167  0.0000
    fin       -0.1037    0.0584  0.0758
    age       -0.0152    0.0041  0.0002
    prio       0.0366    0.0140  0.0091
    >>> model.sf(52, [[0, 25, 3], [1, 25, 3]]).round(3)
    array([0.697, 0.772])
    """

    #: The event times at which ``B`` is estimated.
    x: npt.NDArray
    #: The cumulative regression functions at ``x``, ``(len(x), 1 + p)``:
    #: the baseline's, then each covariate's (``nan`` for an aliased one).
    B: npt.NDArray
    #: The variances of ``B`` (the optional variation), as ``B``.
    B_var: npt.NDArray
    #: The covariance matrix of each row of ``B``, ``(len(x), 1 + p, 1 +
    #: p)``, for the bounds on a unit's prediction.
    _B_cov: npt.NDArray
    #: The ridge penalty the fit took (0 for none).
    coef_penalty: float = 0.0
    #: The aliased covariate columns (#476).
    _aliased: npt.NDArray
    #: The rows fitted ``(x, c, n, tl, Z)``, for ``concordance`` and the
    #: resampling tests; not saved.
    _fit_data: tuple | None = None
    #: Aalen's estimator is a least-squares fit at each event time, not
    #: the maximum of a likelihood.
    maximum: str = "not applicable"
    kind: str
    parameterization: str
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: object = None

    _NO_LIKELIHOOD_WHY = (
        "Aalen's estimator is a least-squares fit at each event time; "
        "there is no likelihood to maximise"
    )

    def __init__(self) -> None:
        self.kind = "Aalen Additive"
        self.parameterization = "Non-Parametric"

    # -- names and covariates ---------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        """The covariates' names, one per column of ``Z``: the columns of
        a formula, ``fit_from_df`` or a DataFrame ``Z``, else ``coef_0``,
        ``coef_1``, ... as in the other regression models."""
        return coefficient_names(self, int(self.B.shape[1]) - 1)

    @property
    def term_names(self) -> list[str]:
        """The names of the columns of ``B``: ``"baseline"``, then
        :attr:`parameter_names`."""
        return [BASELINE] + self.parameter_names

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose effects the data cannot determine
        (#476): a constant column, which the baseline absorbs, or a
        linear combination of the others. Their columns of ``B`` are
        ``nan``, and predictions take them as 0."""
        return np.asarray(self._aliased, dtype=int)

    def _rows(self, Z: Any) -> npt.NDArray:
        """The design rows ``[1, Z]`` of covariates as the predictions
        take them, an aliased column as 0."""
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        p = int(self.B.shape[1]) - 1
        check_covariate_width(Zp, self.parameter_names)
        rows = np.asarray(covariate_rows(Zp, p), dtype=float)
        if self.aliased.size:
            rows = rows.copy()
            rows[:, self.aliased] = 0.0
        return np.column_stack([np.ones(rows.shape[0]), rows])

    def _coefficients(self) -> npt.NDArray:
        """``B`` with an aliased column as 0."""
        return np.where(np.isnan(self.B), 0.0, self.B)

    def __repr__(self) -> str:
        out = (
            "Non-Parametric Regression SurPyval Model"
            + "\n========================================"
            + "\nType                : Additive Hazards"
            + "\nKind                : Aalen"
            + "\nParameterization    : Non-Parametric"
        )
        if self.coef_penalty:
            out += f"\nCoefficient penalty : {self.coef_penalty:g}"
        if self.x.size:
            out += f"\nCumulative effects at t = {self.x[-1]:g}:\n"
            for name, b in zip(self.term_names, self.B[-1]):
                out += f"   {name}  :  {b}\n"
        return out

    # -- the cumulative regression functions ------------------------------

    def _index(self, x: npt.NDArray) -> npt.NDArray:
        """The row of ``B`` in force at each time (right-continuous), -1
        before the first event time."""
        return np.searchsorted(self.x, x, side="right") - 1

    def _check_terms(self, terms: Any) -> npt.NDArray:
        names = self.term_names
        if terms is None:
            return np.arange(len(names))
        out = []
        for t in [terms] if isinstance(terms, (str, int)) else terms:
            if isinstance(t, str) and t in names:
                out.append(names.index(t))
            elif isinstance(t, (int, np.integer)) and 0 <= t < len(names):
                out.append(int(t))
            else:
                raise ValueError(
                    "Unknown term {!r}; the terms are {} (or their "
                    "positions 0 to {}).".format(t, names, len(names) - 1)
                )
        return np.asarray(out, dtype=int)

    def cumulative_coefficients(
        self, x: npt.ArrayLike | None = None, terms: Any = None
    ) -> pd.DataFrame:
        """
        The cumulative regression functions ``B_j(t)`` as a table: one row
        per time (the event times, or ``x``), one column per term
        (:attr:`term_names`). ``B`` is a step function, 0 before the first
        event time and held after the last.

        Parameters
        ----------
        x : array like, optional
            The times. Defaults to the event times :attr:`x`.
        terms : str, int or list, optional
            The terms to include, by name or position. Defaults to all.

        Returns
        -------
        pandas.DataFrame
            Indexed by time.
        """
        import pandas as pd

        t = self.x if x is None else np.atleast_1d(np.asarray(x, float))
        cols = self._check_terms(terms)
        values = self._B_at(t)[:, cols]
        return pd.DataFrame(
            values,
            index=pd.Index(t, name="time"),
            columns=[self.term_names[j] for j in cols],
        )

    def _B_at(self, t: npt.NDArray) -> npt.NDArray:
        idx = self._index(t)
        out = np.where((idx >= 0)[:, None], self.B[np.clip(idx, 0, None)], 0.0)
        if self.B.shape[0] == 0:
            out = np.zeros((t.size, self.B.shape[1]))
        out[np.isnan(t)] = np.nan
        return out

    def _B_var_at(self, t: npt.NDArray) -> npt.NDArray:
        idx = self._index(t)
        if self.B.shape[0] == 0:
            return np.zeros((t.size, self.B.shape[1]))
        out = np.where(
            (idx >= 0)[:, None], self.B_var[np.clip(idx, 0, None)], 0.0
        )
        out[np.isnan(t)] = np.nan
        return out

    def cumulative_coefficients_cb(
        self,
        x: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        terms: Any = None,
    ) -> npt.NDArray:
        """
        Pointwise Wald confidence bounds on the cumulative regression
        functions: ``B_j(t) +/- z se``, with the optional-variation
        standard error (R's ``timereg::aalen`` ``var.cum`` with ``robust
        = 0``; lifelines' ``confidence_intervals_``).

        Parameters
        ----------
        x : array like, optional
            The times. Defaults to the event times :attr:`x`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : ('two-sided', 'lower', 'upper'), optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        terms : str, int or list, optional
            The terms, as :meth:`cumulative_coefficients` takes them.

        Returns
        -------
        numpy array
            ``(len(x), len(terms))``, with a last axis of 2 for two-sided
            bounds.
        """
        check_option("bound", bound, _BOUNDS)
        _check_alpha(alpha_ci)
        t = self.x if x is None else np.atleast_1d(np.asarray(x, float))
        cols = self._check_terms(terms)
        B = self._B_at(t)[:, cols]
        se = np.sqrt(self._B_var_at(t)[:, cols])
        if bound == "two-sided":
            z = norm.isf(alpha_ci / 2)
            return np.stack([B - z * se, B + z * se], axis=-1)
        z = norm.isf(alpha_ci)
        return B - z * se if bound == "lower" else B + z * se

    def smoothed_coefficients(
        self,
        x: npt.ArrayLike,
        bandwidth: float | None = None,
        terms: Any = None,
    ) -> pd.DataFrame:
        r"""
        The regression functions ``b_j(t)`` themselves, the slopes of
        ``B_j``: each covariate's excess hazard per unit at time ``t``.
        They are estimated by smoothing the increments of ``B`` with an
        Epanechnikov kernel (Ramlau-Hansen),

        .. math::
            \hat b_j(t) = \frac{1}{w} \sum_k K\left(\frac{t - t_k}{w}
            \right) \Delta B_j(t_k), \quad K(u) = \tfrac34 (1 - u^2),

        which is lifelines' ``smoothed_hazards_(bandwidth)`` divided by
        the bandwidth ``w`` (lifelines leaves the division out, so the two
        agree at ``w = 1``). Within a bandwidth of either end of the event
        times the kernel runs out of data and the estimate is pulled
        towards 0.

        Parameters
        ----------
        x : array like
            The times.
        bandwidth : float, optional
            The kernel's half-width ``w``, in the units of time. Defaults
            to the normal-reference rule ``1.06 sd m^(-1/5)`` on the ``m``
            event times. A wider bandwidth gives a smoother, more biased
            estimate.
        terms : str, int or list, optional
            The terms, as :meth:`cumulative_coefficients` takes them.

        Returns
        -------
        pandas.DataFrame
            Indexed by time, a column per term.
        """
        import pandas as pd

        t = np.atleast_1d(np.asarray(x, dtype=float))
        cols = self._check_terms(terms)
        values = self._smoothed(t, bandwidth)[:, cols]
        return pd.DataFrame(
            values,
            index=pd.Index(t, name="time"),
            columns=[self.term_names[j] for j in cols],
        )

    def _smoothed(
        self, t: npt.NDArray, bandwidth: float | None
    ) -> npt.NDArray:
        if bandwidth is None:
            bandwidth = _default_bandwidth(self.x)
        else:
            try:
                bw = float(bandwidth)
            except (TypeError, ValueError):
                bw = float("nan")
            if not (np.isfinite(bw) and bw > 0):
                raise ValueError(
                    "'bandwidth' must be a positive number of time units; "
                    "got {!r}.".format(bandwidth)
                )
            bandwidth = bw
        dB = np.diff(
            np.concatenate([np.zeros((1, self.B.shape[1])), self.B]), axis=0
        )
        u = (t[:, None] - self.x[None, :]) / bandwidth
        with np.errstate(invalid="ignore"):
            return _epanechnikov(u) @ dB / bandwidth

    # -- prediction ---------------------------------------------------------

    def _cumulative_hazard(
        self, x: npt.NDArray, Z: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """``(H*, H, rows, x)`` at the times ``x``: the estimate ``H(x | Z) =
        B_0(x) + B(x)'Z``, its running maximum from 0 (floored at 0), the
        cumulative hazard the model predicts with, and the design rows,
        paired with ``x``, and the times, one per row."""
        rows = self._rows(Z)
        check_paired_rows(x.size, rows.shape[0], grid=False)
        if x.size == 1 and rows.shape[0] > 1:
            # One time for every row
            x = np.repeat(x, rows.shape[0])
        rows = np.broadcast_to(rows, (x.size, rows.shape[1]))
        B = self._coefficients()
        idx = self._index(x)
        uniq, which = unique_rows(rows)
        if B.shape[0] == 0 or x.size == 0:
            zero = np.where(np.isnan(x), np.nan, 0.0)
            return zero, zero, rows, x
        grid = B @ uniq.T  # (k, distinct rows)
        peaks = np.maximum.accumulate(np.maximum(grid, 0.0), axis=0)
        at = np.clip(idx, 0, None)
        H = np.where(idx >= 0, grid[at, which], 0.0)
        H_star = np.where(idx >= 0, peaks[at, which], 0.0)
        bad = np.isnan(x) | np.isnan(rows).any(axis=1)
        H = np.where(bad, np.nan, H)
        H_star = np.where(bad, np.nan, H_star)
        # Nothing happens at or before time 0.
        H_star = np.where(x <= 0, 0.0, H_star)
        return H_star, H, rows, x

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The cumulative hazard at ``x`` of a unit with covariates ``Z``
        (one row for every ``x``, or one row per ``x``): the estimate
        ``H(x | Z) = B_0(x) + B(x)'Z``, held at its running maximum from
        time 0.

        Nothing keeps the additive hazard ``b_0(t) + b(t)'Z`` positive, so
        the estimate can fall -- for a covariate row whose hazard the model
        takes below 0, or by chance -- and ``exp(-H)`` would be a survival
        that rises and exceeds 1. As for the Lin-Ying model
        (:meth:`AdditiveHazardsModel.Hf
        <surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazardsModel.Hf>`),
        the prediction is the smallest non-decreasing cumulative hazard at
        or above both the estimate and 0: ``max(0, max_{s <= x} H(s |
        Z))``. It is the estimate wherever the estimate is at its running
        maximum. R's ``timereg`` and lifelines return the estimate itself,
        which is the same wherever it does not fall.

        ``H`` is a step function: 0 before the first event time, a step at
        each event time, and held after the last time ``B`` is estimated
        at (there is no one left at risk to estimate it from).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._cumulative_hazard(x, Z)[0]

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """Survival ``exp(-Hf(x, Z))``, in ``[0, 1]`` and non-increasing
        (see :meth:`Hf`)."""
        return np.exp(-self.Hf(x, Z))

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """Failure probability ``1 - sf(x, Z)``."""
        return -np.expm1(-self.Hf(x, Z))

    @keeps_query_shape
    def hf(
        self, x: npt.ArrayLike, Z: Any, bandwidth: float | None = None
    ) -> npt.NDArray:
        """
        The hazard rate ``b_0(t) + b(t)'Z``, from the kernel-smoothed
        regression functions (:meth:`smoothed_coefficients`, whose
        ``bandwidth`` it takes), floored at 0, and 0 where :meth:`Hf` is
        held: before time 0, after the last event time, and where the
        estimate is below its running maximum.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        H_star, H, rows, x = self._cumulative_hazard(x, Z)
        if self.B.shape[0] == 0:
            return np.where(np.isnan(H_star), np.nan, 0.0)
        b = np.where(np.isnan(self.B[0]), 0.0, 1.0) * self._smoothed(
            x, bandwidth
        )
        b = np.where(np.isnan(b), 0.0, b)
        rate = np.sum(b * rows, axis=1)
        with np.errstate(invalid="ignore"):
            rate = np.where((rate < 0) | (H < H_star), 0.0, rate)
        rate = np.where((x > self.x[-1]) | (x <= 0), 0.0, rate)
        return np.where(np.isnan(H_star), np.nan, rate)

    @keeps_query_shape
    def df(
        self, x: npt.ArrayLike, Z: Any, bandwidth: float | None = None
    ) -> npt.NDArray:
        """Density ``hf(x, Z) * sf(x, Z)``, with the smoothed hazard."""
        return self.hf(x, Z, bandwidth=bandwidth) * self.sf(x, Z)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The quantile function: the first event time at which the predicted
        failure probability ``ff(x, Z)`` reaches ``p``, ``nan`` where it
        never does (the curve is held after the last event time, above
        ``1 - p`` where the data end censored). ``Z`` is paired with ``p``
        as :meth:`sf` pairs it with ``x``. A probability outside [0, 1]
        gives ``nan``, with a warning.
        """
        rows = self._rows(Z)[:, 1:]
        u, rows, _ = paired_probabilities(p, rows)
        if not self.x.size or not u.size:
            return np.full(u.shape, np.nan)
        F = np.vstack([self.ff(self.x, rows[k]) for k in range(rows.shape[0])])
        return step_quantiles(F, self.x, u)

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on a unit's predicted ``sf``, ``ff``
        or ``Hf``. The standard error of ``H(x | Z) = B_0(x) + B(x)'Z`` is
        ``sqrt(X' C(x) X)``, ``X = [1, Z]`` and ``C`` the optional-variation
        covariance of ``B``; the bound is a Wald bound on ``log H`` about
        the predicted ``Hf``, so that ``H`` stays positive and ``sf`` in
        ``(0, 1]``, and ``sf`` and ``ff`` are taken from it. Where the
        predicted ``Hf`` is 0 (before the first event) both bounds are 0.

        Only Wald bounds are available: there is no likelihood for a
        likelihood-ratio bound, and no bootstrap is implemented; for one,
        refit the model to resampled rows.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like
            The covariates, paired with ``x`` as :meth:`sf` pairs them.
        on : ('sf', 'ff', 'Hf'), optional
            The function to bound (``'R'`` and ``'F'`` are aliases of
            ``'sf'`` and ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : ('two-sided', 'lower', 'upper'), optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        method : 'wald', optional
            The only method.

        Returns
        -------
        numpy array
            The bounds, with the shape of ``x`` (and a last axis of 2 for
            two-sided bounds).
        """
        check_option("on", on, _ON)
        on = _ON_ALIASES.get(on, on)
        check_option("bound", bound, _BOUNDS)
        if method != "wald":
            raise ValueError(
                "Aalen's additive model has Wald bounds only "
                "(method='wald'): there is no likelihood for a "
                "likelihood-ratio bound and no bootstrap is implemented; "
                "got method={!r}.".format(method)
            )
        _check_alpha(alpha_ci)
        x_arr = np.asarray(x, dtype=float)
        shape = x_arr.shape
        t = np.atleast_1d(x_arr).ravel()
        H_star, _, rows, t = self._cumulative_hazard(t, Z)
        if t.size != x_arr.size:
            shape = t.shape
        se = self._H_se(t, rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(H_star > 0, se / H_star, 0.0)
        if bound == "two-sided":
            z = norm.isf(alpha_ci / 2)
            lo, hi = H_star * np.exp(-z * ratio), H_star * np.exp(z * ratio)
        else:
            # The side of H that gives the side asked for: sf falls as H
            # rises, so a lower bound on sf is an upper bound on H.
            z = norm.isf(alpha_ci)
            up = (bound == "upper") != (on == "sf")
            lo = hi = H_star * np.exp((z if up else -z) * ratio)
        out = _from_H(lo, hi, on, bound)
        if bound == "two-sided":
            return out.reshape(shape + (2,))
        return out.reshape(shape)

    def _H_se(self, t: npt.NDArray, rows: npt.NDArray) -> npt.NDArray:
        idx = self._index(t)
        if self.B.shape[0] == 0:
            return np.zeros(t.shape)
        C = np.where(np.isnan(self._B_cov), 0.0, self._B_cov)
        C_at = C[np.clip(idx, 0, None)]
        var = np.einsum("ij,ijk,ik->i", rows, C_at, rows)
        var = np.where(idx >= 0, var, 0.0)
        return np.sqrt(np.maximum(var, 0.0))

    # -- summaries and tests ------------------------------------------------

    def summary(
        self, alpha_ci: float = 0.05, at: float | None = None
    ) -> pd.DataFrame:
        """
        The cumulative effects at one time, in the layout of the other
        regression models' :meth:`summary`: for each term (the baseline,
        then each covariate) ``B_j(at)``, its optional-variation standard
        error, a two-sided ``1 - alpha_ci`` Wald interval, and the Wald
        statistic ``z`` and its p-value, a test that the covariate has
        added nothing to the cumulative hazard by ``at``. ``exp(coef)`` is
        ``nan``: an additive effect is not a ratio. For how the effects
        change over time see :meth:`cumulative_coefficients` and
        :meth:`plot`, and for a test that an effect is constant,
        :meth:`constant_effects_test`.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.
        at : float, optional
            The time. Defaults to the last event time at which ``B`` is
            estimated.

        Returns
        -------
        pandas.DataFrame
            One row per term.
        """
        _check_alpha(alpha_ci)
        if at is None:
            at = float(self.x[-1]) if self.x.size else 0.0
        try:
            t = np.atleast_1d(np.asarray(at, dtype=float))
        except (TypeError, ValueError):
            t = np.array([np.nan])
        if t.size != 1 or not np.isfinite(t[0]) or t[0] < 0:
            raise ValueError(
                "'at' must be one finite time at or after 0 (the time "
                f"origin); got {at!r}."
            )
        return coefficient_table(
            self.term_names,
            self._B_at(t)[0],
            np.sqrt(self._B_var_at(t)[0]),
            alpha_ci,
            exp=False,
        )

    def constant_effects_test(
        self, n_sim: int = 1000, random_state: Any = None
    ) -> pd.DataFrame:
        r"""
        Tests that each term's effect is constant in time: that
        ``B_j(t)`` is a straight line through 0, as in the Lin-Ying model
        (:class:`~surpyval.AdditiveHazards`). The statistics are R's
        ``timereg::aalen`` "test for time invariant effects": the largest
        distance of ``B_j`` from the straight line to its last value,

        .. math::
            KS_j = \sup_{t \le \tau} \left|B_j(t) - \tfrac{t}{\tau}
            B_j(\tau)\right|,

        (Kolmogorov-Smirnov) and its integrated square (Cramér-von Mises,
        summed over the event times). Their null distributions are
        simulated by multiplying each unit's martingale residual process
        by a standard normal draw (Martinussen and Scheike 2006, section
        5.3): ``n_sim`` draws, each a pass over the event times, so the
        cost grows with units times event times times ``n_sim``. The
        p-value is the share of draws at least as large as the observed
        statistic; p-values from different seeds differ by about
        ``sqrt(p (1 - p) / n_sim)``.

        The resampled null is a large-sample approximation, and it is too
        narrow with a few hundred units: in simulations of 300 units the
        test rejected a truly constant effect 8% to 16% of the time at the
        5% level, and 5% with 1000 units. With a few hundred units, read a
        p-value near the level as borderline, and look at :meth:`plot`.

        Parameters
        ----------
        n_sim : int, optional
            The number of simulated draws. Default 1000, as ``timereg``.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the draws. ``None`` draws from numpy's global
            generator.

        Returns
        -------
        pandas.DataFrame
            One row per term: ``KS``, ``KS p``, ``CvM`` and ``CvM p``.
        """
        import pandas as pd

        if self._fit_data is None:
            raise ValueError(
                "constant_effects_test() needs the data the model was "
                "fitted to, which a model restored from a dict does not "
                "keep; refit it with AalenAdditive.fit."
            )
        if not (isinstance(n_sim, (int, np.integer)) and n_sim >= 1):
            raise ValueError(
                "'n_sim' must be a whole number of draws, at least 1; got "
                "{!r}.".format(n_sim)
            )
        rng = as_generator(random_state)
        W = self._resampled(int(n_sim), rng)  # (n_sim, k, q)
        B = self._coefficients()
        ks_obs, cvm_obs = _invariance_statistics(self.x, B[None])
        ks_sim, cvm_sim = _invariance_statistics(self.x, W)
        with np.errstate(invalid="ignore"):
            ks_p = np.mean(ks_sim >= ks_obs[0], axis=0)
            cvm_p = np.mean(cvm_sim >= cvm_obs[0], axis=0)
        table = pd.DataFrame(
            {
                "KS": ks_obs[0],
                "KS p": ks_p,
                "CvM": cvm_obs[0],
                "CvM p": cvm_p,
            },
            index=pd.Index(self.term_names, name="term"),
        )
        table.loc[np.isnan(self.B[0]) if self.B.size else [], :] = np.nan
        return table

    def _resampled(self, n_sim: int, rng: np.random.Generator) -> npt.NDArray:
        """``n_sim`` draws of the resampled process ``W(t)`` at the event
        times: each unit's martingale residual process, mapped as the
        increments of ``B`` are, times a standard normal draw (a row of
        ``n`` units carries ``sqrt(n)`` times one)."""
        x, c, n, tl, Z = self._fit_data  # type: ignore[misc]
        X = np.column_stack([np.ones(x.size), Z])
        kept = np.r_[0, 1 + np.setdiff1d(np.arange(Z.shape[1]), self.aliased)]
        X = X[:, kept]
        dB = np.diff(
            np.concatenate([np.zeros((1, kept.size)), self.B[:, kept]]), axis=0
        )
        # The inverses of the fit, recomputed with its penalty.
        penalty = self._penalty_vector(Z)[kept]
        _, _, _, inverses = _estimate(x, c, n, tl, X, penalty)
        G = rng.standard_normal((n_sim, x.size)) * np.sqrt(n)[None, :]
        W = np.zeros((n_sim, self.x.size, self.B.shape[1]))
        running = np.zeros((n_sim, kept.size))
        event = c == 0
        for k, t in enumerate(self.x):
            at_risk = (tl < t) & (x >= t)
            dN = (event & (x == t)).astype(float)
            r = dN - at_risk * (X @ dB[k])
            running = running + ((G * r[None, :]) @ X) @ inverses[k].T
            W[:, k, kept] = running
        return W

    def _penalty_vector(self, Z: npt.NDArray) -> npt.NDArray:
        """The ridge penalty on each term: ``coef_penalty`` times each
        covariate's variance (so the penalty does not depend on its
        units), none on the baseline."""
        x, c, n, tl, _ = self._fit_data  # type: ignore[misc]
        return _penalties(Z, n, self.coef_penalty)

    def plot(
        self,
        terms: Any = None,
        alpha_ci: float = 0.05,
        smoothed: bool = False,
        bandwidth: float | None = None,
        axes: Any = None,
    ) -> Any:
        """
        Plot each term's cumulative regression function ``B_j(t)`` as a
        step with its pointwise ``1 - alpha_ci`` band, one panel per term
        -- a straight line through 0 is a constant effect, a bend an
        effect that changes -- or, with ``smoothed=True``, the smoothed
        slopes ``b_j(t)`` (:meth:`smoothed_coefficients`).

        Parameters
        ----------
        terms : str, int or list, optional
            The terms to draw. Defaults to all.
        alpha_ci : float, optional
            The band's total tail probability (not drawn with
            ``smoothed``). Default 0.05.
        smoothed : bool, optional
            Draw the smoothed ``b_j(t)`` instead. Default False.
        bandwidth : float, optional
            The smoothing bandwidth, as :meth:`smoothed_coefficients`.
        axes : matplotlib Axes or sequence of them, optional
            One axes per term. Without them a new figure is made.

        Returns
        -------
        numpy array of matplotlib Axes
            One per term.
        """
        import matplotlib.pyplot as plt

        from surpyval.plotting import BAND_ALPHA, INK, PALETTE

        cols = self._check_terms(terms)
        if axes is None:
            width = min(len(cols), 3)
            height = int(np.ceil(len(cols) / width))
            fig, grid = plt.subplots(
                height,
                width,
                figsize=(4.0 * width, 3.0 * height),
                squeeze=False,
                constrained_layout=True,
            )
            axs = grid.ravel()
            for ax in axs[len(cols) :]:
                ax.set_visible(False)
            axs = axs[: len(cols)]
        else:
            axs = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
            if axs.size != len(cols):
                raise ValueError(
                    "'axes' must hold one axes per term: {} terms, {} "
                    "axes.".format(len(cols), axs.size)
                )
        t = np.concatenate([[0.0], self.x])
        for ax, j in zip(axs, cols):
            name = self.term_names[j]
            if smoothed:
                grid_t = np.linspace(0.0, t[-1], 200)
                b = self._smoothed(grid_t, bandwidth)[:, j]
                ax.plot(grid_t, b, color=PALETTE[0])
                ax.set_ylabel("b(t), excess hazard per unit")
            else:
                B = np.concatenate([[0.0], self.B[:, j]])
                band = self.cumulative_coefficients_cb(
                    t, alpha_ci=alpha_ci, terms=[j]
                )[:, 0]
                ax.step(t, B, where="post", color=PALETTE[0])
                ax.fill_between(
                    t,
                    band[:, 0],
                    band[:, 1],
                    step="post",
                    color=PALETTE[0],
                    alpha=BAND_ALPHA,
                    linewidth=0,
                )
                ax.set_ylabel("B(t), cumulative effect")
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_title(name)
            ax.set_xlabel("time")
        return axs

    # -- concordance --------------------------------------------------------

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        rows = self._rows(Z)
        last = self._coefficients()[-1] if self.B.shape[0] else 0.0
        return np.broadcast_to(rows @ np.atleast_1d(last), x.shape)

    def _concordance_data(self) -> tuple | None:
        if self._fit_data is None:
            return None
        x, c, n, _, Z = self._fit_data
        return (x, c, n, Z)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-serialisable dict:
        the event times, ``B``, its variances and covariances, the
        penalty, the aliased columns and the covariates' names (or
        formula). Every prediction and bound round-trips exactly; the
        fitted rows, which :meth:`constant_effects_test` and
        :meth:`concordance` without data need, are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict[str, Any] = {
            "model": "AalenAdditiveModel",
            "x": np.asarray(self.x, dtype=float).tolist(),
            "B": np.asarray(self.B, dtype=float).tolist(),
            "B_var": np.asarray(self.B_var, dtype=float).tolist(),
            "B_cov": np.asarray(self._B_cov, dtype=float).tolist(),
            "n_terms": int(self.B.shape[1]),
            "coef_penalty": float(self.coef_penalty),
            "aliased": self.aliased.tolist(),
        }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> AalenAdditiveModel:
        """
        Rebuild an Aalen additive model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "AalenAdditiveModel", "an Aalen additive model"
        )
        q = int(model_dict["n_terms"])
        out = cls()
        out.x = np.array(model_dict["x"], dtype=float)
        k = out.x.size
        out.B = np.array(model_dict["B"], dtype=float).reshape(k, q)
        out.B_var = np.array(model_dict["B_var"], dtype=float).reshape(k, q)
        out._B_cov = np.array(model_dict["B_cov"], dtype=float).reshape(
            k, q, q
        )
        out.coef_penalty = float(model_dict["coef_penalty"])
        out._aliased = np.array(model_dict["aliased"], dtype=int)
        restore_covariate_meta(out, model_dict)
        return out


def _check_alpha(alpha_ci: Any) -> None:
    if not (
        isinstance(alpha_ci, (int, float, np.floating))
        and 0 < float(alpha_ci) < 1
    ):
        raise ValueError(
            "'alpha_ci' must be a probability strictly between 0 and 1 "
            "(0.05 for 95% bounds); got {!r}.".format(alpha_ci)
        )


def _from_H(
    lo: npt.NDArray, hi: npt.NDArray, on: str, bound: str
) -> npt.NDArray:
    """Bounds on ``on`` from bounds ``[lo, hi]`` on ``H``: ``sf`` is
    decreasing in ``H``, so its lower bound is from ``hi``."""
    if on == "Hf":
        if bound == "two-sided":
            return np.stack([lo, hi], axis=-1)
        return lo
    if bound == "two-sided":
        if on == "sf":
            return np.stack([np.exp(-hi), np.exp(-lo)], axis=-1)
        return np.stack([-np.expm1(-lo), -np.expm1(-hi)], axis=-1)
    # One-sided: ``lo`` is the bound on H on the side cb chose.
    return np.exp(-lo) if on == "sf" else -np.expm1(-lo)


def _invariance_statistics(
    times: npt.NDArray, B: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The Kolmogorov-Smirnov and Cramér-von Mises distances of each
    ``B[s, :, j]`` (a step function at ``times``) from the straight line
    to its last value: the supremum over the steps' left and right
    values, and the square summed over the event times."""
    if times.size == 0:
        shape = (B.shape[0], B.shape[2])
        return np.full(shape, np.nan), np.full(shape, np.nan)
    tau = times[-1]
    line = (times / tau)[None, :, None] * B[:, -1:, :]
    right = B - line
    before = np.concatenate([np.zeros_like(B[:, :1]), B[:, :-1]], axis=1)
    left = before - line
    ks = np.maximum(np.abs(right).max(axis=1), np.abs(left).max(axis=1))
    widths = np.diff(np.concatenate([[0.0], times]))
    cvm = np.einsum("skj,k->sj", right**2, widths)
    return ks, cvm


def _penalties(
    Z: npt.NDArray, n: npt.NDArray, coef_penalty: float
) -> npt.NDArray:
    """``coef_penalty`` times each covariate's variance (over the units,
    with ``n - 1`` degrees of freedom), and 0 for the baseline."""
    total = float(n.sum())
    if coef_penalty == 0 or Z.shape[1] == 0:
        return np.zeros(Z.shape[1] + 1)
    mean = (n @ Z) / total
    var = (n @ (Z - mean) ** 2) / max(total - 1.0, 1.0)
    return np.concatenate([[0.0], coef_penalty * var])


def _aliased(Z: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """The columns of ``Z`` the fit cannot separate from the others or
    from the baseline: constant, or a linear combination of the others
    over the data (see :mod:`surpyval.univariate.regression._aliasing`)."""
    if Z.shape[1] == 0:
        return np.array([], dtype=int)
    total = float(n.sum())
    Zc = Z - (n @ Z) / total
    gram = (n[:, None] * Zc).T @ Zc
    return aliased_columns(gram, Z.shape[0], constant_columns(Z))


class AalenAdditive_(FitterRepr):
    r"""
    Aalen's non-parametric additive hazards model: each covariate adds to
    the hazard an amount that may change freely with time,

    .. math::
        h(x \mid Z) = b_0(x) + b(x)' Z.

    The fit estimates the cumulative regression functions :math:`B_j(x)
    = \int_0^x b_j`, a least-squares fit at each event time, with no
    iteration. ``AalenAdditive`` is an instance of this class; its
    ``fit`` returns an
    :class:`~surpyval.univariate.regression.additive_hazards.aalen_additive.AalenAdditiveModel`.
    For effects constant in time, see :class:`~surpyval.AdditiveHazards`
    (Lin and Ying), and for a parametric baseline the ``AH`` family.
    """

    fitter_kind = "non-parametric (Aalen) additive hazards fitter"

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        coef_penalty: float = 0.0,
    ) -> AalenAdditiveModel:
        """
        Fit Aalen's additive hazards model.

        Parameters
        ----------
        x : array like
            The times of the events and censorings, at or above 0.
        Z : array like
            The covariates, one row per observation. Rows with a missing
            or infinite covariate are dropped, with a warning. A constant
            column (which the baseline absorbs) or one that is a linear
            combination of the others is aliased: its column of ``B`` is
            ``nan`` (``model.aliased`` lists it), the others are those of
            the fit without it, predictions take it as 0, and one warning
            names it.
        c : array like, optional
            Censoring flags: 0 observed (an event), 1 right censored.
            Defaults to all observed. Left and interval censored data are
            refused.
        n : array like, optional
            The count of each row. Defaults to 1 each.
        tl : array like, optional
            Each row's entry time (left truncation): it is at risk on
            ``(tl, x]``. Defaults to 0.
        coef_penalty : float, optional
            A ridge penalty on the covariates' increments,
            ``dB = (X'YX + P)^{-1} X'dN`` with ``P`` the penalty times
            each covariate's variance (so the penalty does not depend on
            the covariates' units) and nothing on the baseline. It keeps
            the fit going where few units are at risk, at the price of
            shrinking the effects towards 0. Default 0, no penalty.
            lifelines' ``coef_penalizer`` is the same on standardised
            covariates, but penalises its intercept too.

        Returns
        -------
        AalenAdditiveModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import AalenAdditive
        >>> rng = np.random.default_rng(3)
        >>> Z = rng.binomial(1, 0.5, (300, 1)).astype(float)
        >>> x = rng.exponential(1 / (0.1 + 0.05 * Z[:, 0]))
        >>> c = (x > 15).astype(int)  # follow-up ends at 15
        >>> x = np.minimum(x, 15)
        >>> model = AalenAdditive.fit(x, Z, c=c)
        >>> model.cumulative_coefficients([5, 10]).round(3)
              baseline  coef_0
        time
        5.0      0.562   0.340
        10.0     1.099   0.362

        The truth is ``B_0(t) = 0.1 t`` and ``B_1(t) = 0.05 t``.
        """
        x, c, n, tl, Z = _validate(x, Z, c, n, tl)
        penalty_value = _check_penalty(coef_penalty)
        tl = np.where(np.isfinite(tl), tl, 0.0)
        p = Z.shape[1]
        aliased = _aliased(Z, n)
        if aliased.size:
            warn_aliased(
                aliased,
                "they do not vary beyond a combination of the other "
                "columns (a constant column, which the baseline absorbs, "
                "or a linear combination of the others)",
            )
        kept_cols = np.setdiff1d(np.arange(p), aliased)
        kept = np.r_[0, 1 + kept_cols].astype(int)
        X = np.column_stack([np.ones(x.size), Z])[:, kept]
        penalty = _penalties(Z, n, penalty_value)[kept]
        times, dB, dC, _ = _estimate(x, c, n, tl, X, penalty)
        k = times.size
        B = np.full((k, p + 1), np.nan)
        B[:, kept] = np.cumsum(dB, axis=0)
        C = np.full((k, p + 1, p + 1), np.nan)
        C[np.ix_(np.arange(k), kept, kept)] = np.cumsum(dC, axis=0)
        model = AalenAdditiveModel()
        model.x = times
        model.B = B
        model.B_var = np.diagonal(C, axis1=1, axis2=2).copy()
        model._B_cov = C
        model.coef_penalty = penalty_value
        model._aliased = aliased
        model._fit_data = (x, c, n, tl, Z)
        return model

    @column_arguments("x", "c", "n", "tl")
    @takes_unit
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        tl_col: str | None = None,
        coef_penalty: float = 0.0,
    ) -> AalenAdditiveModel:
        """
        Fit Aalen's additive hazards model from a pandas DataFrame,
        keeping the covariates' names for prediction and the tables (see
        :meth:`fit` for the model). Give the covariates as ``Z_cols`` or a
        ``formula`` (categorical columns get reference-level coding), and
        the entry times, if any, as ``tl_col``.
        """
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col, tl_col=tl_col)
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values
        tl = None if tl_col is None else df[tl_col].values
        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(x, Z, c=c, n=n, tl=tl, coef_penalty=coef_penalty)
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model


AalenAdditive = AalenAdditive_()
