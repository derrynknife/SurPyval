"""
Time-varying coefficients for the Cox model: ``CoxPH.fit(...,
time_interaction=...)``.

A covariate whose effect drifts with time breaks proportional hazards.
The usual remedy is to let its coefficient be a function of time,

.. math::
    \\beta_j(t) = \\beta_j + \\gamma_j\\, g(t),

with ``g(t) = t`` or ``log(t)`` (or any function of time the user
chooses), which is the same as adding the covariate ``Z_j g(t)`` to the
model. That covariate changes with time, so each row is split into
episodes ``(start, stop]`` at the event times inside its follow-up, and
the episodes are fitted as start-stop rows (``CoxPH.fit_tvc``), with
``Z_j g(stop)`` as the extra column. The partial likelihood only reads a
covariate at the event times, and at each event time every unit at risk
is represented by exactly one episode, whose ``stop`` is that event time,
so the fit is exactly R's ``coxph(Surv(time, status) ~ z + tt(z), tt =
function(z, t, ...) z * g(t))`` and the ``survSplit`` form
``coxph(Surv(tstart, time, status) ~ z + z:g(time))``.

The fitted model is a :class:`TimeVaryingCoefficientModel`: its
coefficients are those of the covariates followed by those of the
interactions, its predictions at fixed covariates use :math:`\\beta(t)`
at every baseline time, and :meth:`TimeVaryingCoefficientModel.
time_varying_effect` gives :math:`\\beta_j(t)` with Wald bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.utils import validate_coxph
from surpyval.utils.covariates import coefficient_names as default_names
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.validation import check_alpha_ci

from .._aliasing import covariate_columns, fit_columns
from ..regression_data import check_covariate_width, prepare_Z
from ..semi_parametric_regression_model import SemiParametricRegressionModel

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def _identity(t: npt.NDArray) -> npt.NDArray:
    return t


#: The functions of time ``time_interaction`` takes by name, with the
#: label each interaction coefficient is named by (``karno:log(t)``).
TIME_FUNCTIONS: dict[str, tuple[Callable, str]] = {
    "t": (_identity, "t"),
    "identity": (_identity, "t"),
    "log": (np.log, "log(t)"),
}


@dataclass(frozen=True)
class TimeTerm:
    """One interaction ``Z[:, column] * g(t)`` of a time-varying
    coefficient: the ``column`` of ``Z`` it multiplies, the function of
    time ``g``, the label it is named by (``"log(t)"``), and ``spec``, the
    name ``time_interaction`` gave (``"log"``) or the callable."""

    column: int
    g: Callable
    label: str
    spec: Any

    def values(self, t: npt.ArrayLike) -> npt.NDArray:
        """``g(t)`` as a float array of ``t``'s shape."""
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.asarray(self.g(t), dtype=float)
        if out.shape != t.shape:
            try:
                out = np.broadcast_to(out, t.shape).astype(float)
            except ValueError:
                raise ValueError(
                    "time_interaction: the function of time for column "
                    "{} returned shape {} for {} times; it must take an "
                    "array of times and return one value for each, as "
                    "np.log does.".format(self.column, out.shape, t.size)
                ) from None
        return out


def parse_time_interaction(
    spec: Any, p: int, names: list[str] | None
) -> tuple[TimeTerm, ...]:
    """The terms of ``time_interaction``: a dict ``{column: g}``, the
    column a position in ``Z`` (or, for a ``Z`` with named columns, a
    name) and ``g`` one of ``"t"``, ``"log"`` or a function of time;
    several functions of one column go in a list (``{0: ["t", "log"]}``).
    """
    if not isinstance(spec, Mapping) or not spec:
        raise TypeError(
            "time_interaction must be a dict mapping a column of Z to a "
            "function of time, e.g. time_interaction={{0: 'log'}} (or "
            "{{'age': 'log'}} for a Z with named columns); got "
            "{!r}.".format(spec)
        )
    terms: list[TimeTerm] = []
    for key, gs in spec.items():
        column = _column(key, p, names)
        for g in gs if isinstance(gs, (list, tuple)) else [gs]:
            terms.append(_term(column, g))
    seen = set()
    for term in terms:
        what = (term.column, term.label)
        if what in seen:
            raise ValueError(
                "time_interaction gives column {} the function {} twice; "
                "give each function of a column once.".format(
                    term.column, term.label
                )
            )
        seen.add(what)
    return tuple(terms)


def _column(key: Any, p: int, names: list[str] | None) -> int:
    if isinstance(key, str):
        if names is None or key not in names:
            known = (
                "Z has no column names: use the column's position"
                if names is None
                else f"the columns are {names}"
            )
            raise ValueError(
                "time_interaction names the column {!r}, which is not a "
                "column of Z ({}).".format(key, known)
            )
        return names.index(key)
    if isinstance(key, (bool, np.bool_)) or not isinstance(
        key, (int, np.integer)
    ):
        raise TypeError(
            "time_interaction's keys are columns of Z, by position (an "
            "int) or by name; got {!r}.".format(key)
        )
    if not 0 <= int(key) < p:
        raise ValueError(
            "time_interaction names column {} of Z, which has {} "
            "column(s) (0 to {}).".format(int(key), p, p - 1)
        )
    return int(key)


def _term(column: int, g: Any) -> TimeTerm:
    if isinstance(g, str):
        if g not in TIME_FUNCTIONS:
            raise ValueError(
                "time_interaction's function of time must be one of {} or "
                "a callable g(t); got {!r}.".format(sorted(TIME_FUNCTIONS), g)
            )
        func, label = TIME_FUNCTIONS[g]
        return TimeTerm(column, func, label, g)
    if callable(g):
        name = getattr(g, "__name__", "")
        label = "g(t)" if not name or name == "<lambda>" else f"{name}(t)"
        return TimeTerm(column, g, label, g)
    raise TypeError(
        "time_interaction's function of time must be one of {} or a "
        "callable g(t); got {!r}.".format(sorted(TIME_FUNCTIONS), g)
    )


def interaction_names(
    terms: tuple[TimeTerm, ...], base: list[str]
) -> list[str]:
    """The names of the interaction coefficients, ``<covariate>:<g>``."""
    return [f"{base[t.column]}:{t.label}" for t in terms]


def split_at_event_times(
    x: npt.NDArray,
    c: npt.NDArray,
    tl: npt.NDArray,
    Z: npt.NDArray,
    terms: tuple[TimeTerm, ...],
    event_times: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """The rows ``(tl, x]`` split at the ``event_times`` strictly inside
    them, as R's ``survSplit(cut = event_times)``: ``(row, start, stop, c,
    Z_full)``, ``row`` each episode's row, ``c`` 1 (censored) on every
    episode but a row's last, and ``Z_full`` the row's covariates followed
    by ``Z_j g(stop)`` for each term. A row whose interacting covariates
    are all 0 is not split (its interactions are 0 throughout).

    ``g`` is checked finite at the event times, where the partial
    likelihood reads it; at a ``stop`` that is not an event time the
    episode is in no risk set at an event time, so a non-finite value
    there (``log(0)`` at a censoring time 0) is taken as 0.
    """
    columns = sorted({t.column for t in terms})
    split = np.any(Z[:, columns] != 0, axis=1)
    lo = np.searchsorted(event_times, tl, side="right")
    hi = np.searchsorted(event_times, x, side="left")
    cuts = np.where(split, np.maximum(hi - lo, 0), 0)
    pieces = cuts + 1
    row = np.repeat(np.arange(x.size), pieces)
    first = np.cumsum(pieces) - pieces
    m = np.arange(row.size) - first[row]
    last = m == cuts[row]
    start = np.where(
        m == 0, tl[row], event_times[np.clip(lo[row] + m - 1, 0, None)]
    )
    stop = np.where(
        last,
        x[row],
        event_times[np.clip(lo[row] + m, 0, event_times.size - 1)],
    )
    c_ep = np.where(last, c[row], 1).astype(int)
    Z_rows = Z[row]
    extra = np.empty((row.size, len(terms)))
    at_event = np.isin(stop, event_times)
    for k, term in enumerate(terms):
        g = term.values(stop)
        bad = at_event & ~np.isfinite(g)
        if bad.any():
            t_bad = float(stop[bad][0])
            raise ValueError(
                "time_interaction: {} is not finite at the event time {:g} "
                "(column {}), and the partial likelihood needs it at every "
                "event time. Use a function finite there, such as "
                "lambda t: np.log(t + 1) for a log of a time that can be "
                "0.".format(term.label, t_bad, term.column)
            )
        g = np.where(np.isfinite(g), g, 0.0)
        z = Z_rows[:, term.column]
        extra[:, k] = np.where(z == 0, 0.0, z * g)
    return row, start, stop, c_ep, np.column_stack([Z_rows, extra])


def fit_time_interaction(
    fitter: Any,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    tie_method: str,
    tol: float,
    strata: npt.ArrayLike | None,
    center: bool,
    w: npt.ArrayLike | float | None,
    spec: Any,
) -> TimeVaryingCoefficientModel:
    """``CoxPH.fit(..., time_interaction=spec)``: the rows split at the
    event times (:func:`split_at_event_times`) and fitted by
    ``fitter.fit_tvc``, each row a subject."""
    from .cox_ph import _require_an_event, _stratified_rows

    names = fit_columns()
    Z_in = np.asarray(Z, dtype=float)
    p = 1 if Z_in.ndim < 2 else Z_in.shape[1]
    terms = parse_time_interaction(spec, p, names)
    if w is not None and Z_in.ndim == 2 and not np.isfinite(Z_in).all():
        raise ValueError(
            "time_interaction with weights w: some rows have a missing "
            "covariate. Drop those rows, and their weights, before the "
            "fit."
        )
    labels = None
    if strata is not None:
        positions: list = []
        stratum_labels, validated = _stratified_rows(
            x, Z, c, n, tl, tie_method, strata, rows=positions
        )
        from .cox_ph import _stratified_fit_data

        data = _stratified_fit_data(validated, positions)
        x_a, c_a, n_a, tl_a, Z_a = (
            data["x"],
            data["c"],
            data["n"],
            data["tl"],
            data["Z"],
        )
        labels = np.asarray(stratum_labels)[data["strata"]]
    else:
        x_a, c_a, n_a, tl_a, Z_a = validate_coxph(x, c, n, Z, tl, tie_method)
        _require_an_event(c_a, n_a)
    x_a = np.asarray(x_a, dtype=float)
    c_a = np.asarray(c_a, dtype=int)
    n_a = np.asarray(n_a, dtype=float)
    Z_a = np.atleast_2d(np.asarray(Z_a, dtype=float))
    tl_a = (
        np.full(x_a.shape, -np.inf)
        if tl_a is None
        else np.asarray(tl_a, dtype=float)
    )
    if not np.isfinite(x_a).all():
        raise ValueError(
            "time_interaction needs finite times: a row censored at "
            "infinity cannot be split at the event times. Censor it at "
            "the end of follow-up instead."
        )
    event_times = np.unique(x_a[(c_a == 0) & (n_a > 0)])
    row, start, stop, c_ep, Z_full = split_at_event_times(
        x_a, c_a, tl_a, Z_a, terms, event_times
    )
    base = default_names(p, names)
    full_names = base + interaction_names(terms, base)
    w_ep = None
    if w is not None:
        from surpyval.utils.weights import check_weights

        w_ep = check_weights(w, x_a.size)[row]
    with covariate_columns(full_names):
        model = fitter.fit_tvc(
            row,
            start,
            stop,
            c_ep,
            Z_full,
            n=n_a[row],
            tie_method=tie_method,
            tol=tol,
            center=center,
            strata=None if labels is None else labels[row],
            w=w_ep,
        )
    model.__class__ = TimeVaryingCoefficientModel
    model.time_interaction = terms
    model._n_base = p
    model._unit_Z = Z_a
    model._unit_n = n_a
    if names is not None:
        model.feature_names = list(names)
    return model


@dataclass
class TimeVaryingEffect:
    """A covariate's coefficient over time, :math:`\\beta_j(t) = \\beta_j
    + \\sum_k \\gamma_k g_k(t)`, from
    :meth:`TimeVaryingCoefficientModel.time_varying_effect`, with its
    two-sided Wald bounds.

    Attributes
    ----------
    covariate : str
        The covariate's name.
    x : numpy.ndarray
        The times.
    beta : numpy.ndarray
        :math:`\\beta_j(t)` at each time: the log hazard ratio per unit
        of the covariate at that time (positive means a shorter life).
    se : numpy.ndarray
        Its standard error, :math:`\\sqrt{a' V a}` with :math:`a` the
        weights of the coefficients (1 and :math:`g_k(t)`) and :math:`V`
        their covariance.
    lower : numpy.ndarray
        The lower Wald bound.
    upper : numpy.ndarray
        The upper Wald bound.
    alpha_ci : float
        The bounds' total tail probability.
    """

    covariate: str
    x: npt.NDArray
    beta: npt.NDArray
    se: npt.NDArray
    lower: npt.NDArray
    upper: npt.NDArray
    alpha_ci: float

    @property
    def hazard_ratio(self) -> npt.NDArray:
        """:math:`e^{\\beta_j(t)}`, the hazard ratio per unit of the
        covariate at each time."""
        return np.exp(self.beta)

    def to_frame(self) -> pd.DataFrame:
        """The effect as a DataFrame, one row per time: ``x``, ``beta``,
        ``se``, ``lower``, ``upper`` and ``hazard_ratio``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "x": self.x,
                "beta": self.beta,
                "se": self.se,
                "lower": self.lower,
                "upper": self.upper,
                "hazard_ratio": self.hazard_ratio,
            }
        )

    def to_dict(self) -> dict:
        """The effect as a dict of native Python values, strict JSON (a
        non-finite number is ``None``, recorded under ``"non_finite"``, as
        every SurPyval ``to_dict`` writes it)."""
        from surpyval.serialisation import encode_non_finite, to_native

        out: dict = {"result": type(self).__name__}
        for f in fields(self):
            out[f.name] = to_native(getattr(self, f.name))
        return encode_non_finite(out)

    def plot(self, ax: Axes | None = None) -> Axes:
        """Draw :math:`\\beta_j(t)` against time with its Wald band and
        the line :math:`\\beta = 0` (no effect)."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        (line,) = ax.plot(self.x, self.beta, label=self.covariate)
        ax.fill_between(
            self.x,
            self.lower,
            self.upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel("time")
        ax.set_ylabel(f"coefficient of {self.covariate} (log hazard ratio)")
        return ax


class TimeVaryingCoefficientModel(SemiParametricRegressionModel):
    """
    A Cox model with time-varying coefficients, from ``CoxPH.fit(...,
    time_interaction=...)``: covariate ``j`` acts through

    .. math::
        \\beta_j(t) = \\beta_j + \\sum_k \\gamma_k\\, g_k(t),

    so the hazard is :math:`h_0(t) e^{\\beta(t)' Z}`. ``params`` are the
    covariates' coefficients :math:`\\beta` followed by the interactions'
    :math:`\\gamma` (named ``<covariate>:t``, ``<covariate>:log(t)``,
    ...), with their covariance and Wald p-values; a positive
    :math:`\\gamma` means the covariate's effect on the hazard grows with
    time. :meth:`time_varying_effect` gives :math:`\\beta_j(t)` with Wald
    bounds.

    The model takes the covariates ``Z`` as they were given to the fit
    (one value per column, not the interactions): ``sf(x, Z)``, ``Hf``,
    ``hf``, ``ff``, ``df``, ``qf`` and ``random`` sum the baseline
    hazard's jumps each multiplied by :math:`e^{\\beta(u)' Z}` at its own
    time :math:`u`. ``phi(Z)``, a single hazard ratio, does not exist for
    it and is refused, as are the concordance (the risk ordering changes
    with time), ``diagnose()`` and the time-varying-covariate predictions
    (``sf_tvc``, ``Hf_tvc``, ``predict_tvc``). It is fitted to the rows
    split at the event times, so its residuals are one per episode, and
    its default robust covariance clusters the episodes of each row.
    """

    #: The interactions, one :class:`TimeTerm` each.
    time_interaction: tuple = ()
    #: The number of columns of ``Z``.
    _n_base: int = 0
    #: The rows fitted (before splitting) and their counts, for the
    #: partial-effects plot.
    _unit_Z: npt.NDArray | None = None
    _unit_n: npt.NDArray | None = None

    # -- names ---------------------------------------------------------------

    def _base_names(self) -> list[str]:
        return default_names(
            self._n_base, getattr(self, "feature_names", None)
        )

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``: the covariates, then the
        interactions (``karno:log(t)``)."""
        base = self._base_names()
        return base + interaction_names(self.time_interaction, base)

    def summary(
        self,
        alpha_ci: float | None = 0.05,
        robust: bool = False,
        cluster: npt.ArrayLike | None = None,
    ) -> pd.DataFrame:
        """The coefficient table (see
        :meth:`SemiParametricRegressionModel.summary`), the interactions
        named ``<covariate>:<g>``. The row of an interaction is the Wald
        test of :math:`\\gamma = 0`: of proportional hazards for that
        covariate against the function of time."""
        import pandas as pd

        table = super().summary(alpha_ci, robust, cluster)
        table.index = pd.Index(self.parameter_names, name=table.index.name)
        return table

    def __repr__(self) -> str:
        text = super().__repr__()
        terms = ", ".join(self.parameter_names[self._n_base :])
        line = "\nTime-varying        : {} (rows split at the event times)"
        marker = "\nCoefficients        :"
        return text.replace(marker, line.format(terms) + marker, 1)

    # -- covariates ---------------------------------------------------------

    def _covariate_width(self) -> int:
        return self._n_base

    def _prepare_Z(self, Z: npt.ArrayLike | pd.DataFrame) -> npt.NDArray:
        """The covariates as given to the fit, one value per column of
        ``Z`` (not the interactions), refused at another width."""
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        check_covariate_width(Zp, self._base_names())
        return Zp

    def _log_risk_over_time(
        self, rows: npt.NDArray, times: npt.NDArray
    ) -> npt.NDArray:
        """``beta(u)'Z - beta'center`` for each row (``(m, p)``) at each
        time ``u``: an ``(m, len(times))`` array."""
        coef = self._coef()
        center = self._center()
        p = self._n_base
        out = (rows - center[:p]) @ coef[:p]
        out = np.repeat(out[:, None], times.size, axis=1)
        for k, term in enumerate(self.time_interaction):
            g = term.values(times)
            g = np.where(np.isfinite(g), g, 0.0)
            z = rows[:, term.column][:, None]
            gz = np.where(z == 0, 0.0, z * g[None, :])
            out = out + (gz - center[p + k]) * coef[p + k]
        return out

    def _steps(
        self,
        cumulative: bool,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        stratum: Any,
        grid: bool,
    ) -> npt.NDArray:
        """``hf`` (``cumulative=False``) or ``Hf``: the baseline jumps
        each multiplied by ``exp(beta(u)'Z)`` at its time ``u``."""
        bx, bh0, _ = self._baseline_arrays(stratum)
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        rows = covariate_rows(self._prepare_Z(Z), self._n_base)
        if not grid:
            check_paired_rows(xq.size, rows.shape[0])
        weight = self._times_risk(
            np.asarray(bh0, dtype=float)[None, :],
            self._log_risk_over_time(rows, np.asarray(bx, dtype=float)),
        )
        if cumulative:
            weight = np.cumsum(weight, axis=1)
        idx = np.searchsorted(bx, xq, side="right") - 1
        safe = np.clip(idx, 0, max(len(bx) - 1, 0))
        if grid or rows.shape[0] == 1:
            out = np.where(idx[None, :] >= 0, weight[:, safe], 0.0)
            out = np.where(np.isnan(xq)[None, :], np.nan, out)
            return out if grid else out[0]
        if xq.size == 1:
            out = weight[:, safe[0]] if idx[0] >= 0 else np.zeros(len(rows))
            return np.full(len(rows), np.nan) if np.isnan(xq[0]) else out
        out = np.where(idx >= 0, weight[np.arange(xq.size), safe], 0.0)
        return np.where(np.isnan(xq), np.nan, out)

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """The hazard step at ``x`` for covariates ``Z``: the baseline
        increment at the latest baseline time ``u`` at or before ``x``,
        times :math:`e^{\\beta(u)' Z}`. Arguments as for
        :meth:`SemiParametricRegressionModel.hf`."""
        return self._steps(False, x, Z, stratum, grid)

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """The cumulative hazard at ``x`` for covariates ``Z``,
        :math:`\\sum_{u \\le x} h_0(u) e^{\\beta(u)' Z}`. Arguments as for
        :meth:`SemiParametricRegressionModel.Hf`."""
        return self._steps(True, x, Z, stratum, grid)

    # -- what has no meaning here -------------------------------------------

    def _refuse(self, what: str, instead: str) -> ValueError:
        return ValueError(
            "{} is not available for a Cox model with time-varying "
            "coefficients (fitted with time_interaction): {}".format(
                what, instead
            )
        )

    def phi(self, Z: npt.ArrayLike | pd.DataFrame) -> npt.NDArray:
        """Refused: the hazard ratio of a model with time-varying
        coefficients changes with time. ``time_varying_effect(covariate,
        x).hazard_ratio`` gives it per unit of a covariate at each time,
        and ``hf`` / ``Hf`` the hazard at given covariates."""
        raise self._refuse(
            "phi(Z), a single hazard ratio,",
            "the hazard ratio changes with time. Use "
            "time_varying_effect(covariate, x).hazard_ratio for a "
            "covariate's hazard ratio at each time, or hf / Hf / sf at "
            "the covariates.",
        )

    def concordance(self, *args: Any, **kwargs: Any) -> float:
        """Refused: the units' risk ordering changes with time, so no
        single risk score ranks them."""
        raise self._refuse(
            "concordance",
            "the units' order of risk changes with time, so there is no "
            "single risk score to rank them by. Compare the predicted "
            "survival at a time of interest (sf(t, Z)) with "
            "surpyval.metrics.concordance_index instead.",
        )

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: confidence bounds on the predicted curve of a model
        with time-varying coefficients are not built yet."""
        raise self._refuse(
            "Confidence bounds on the predicted curve (cb)",
            "they are not built yet for time-varying coefficients. "
            "time_varying_effect(covariate) gives each coefficient over "
            "time with Wald bounds; for bounds on the curve, refit "
            "without time_interaction or bootstrap the units.",
        )

    def _refuse_tvc(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse(
            "Prediction along a covariate path",
            "the covariates are fixed and their effects vary with time. "
            "Use sf(x, Z), Hf(x, Z) or hf(x, Z), which apply beta(t) at "
            "each time.",
        )

    sf_tvc = _refuse_tvc
    Hf_tvc = _refuse_tvc
    predict_tvc = _refuse_tvc

    def to_dict(self) -> dict:
        """Refused: saving a model with time-varying coefficients is not
        supported yet (its functions of time may be any callable)."""
        raise self._refuse(
            "Saving (to_dict / to_json)",
            "saving one is not supported yet. Keep the data and refit it, "
            "or pickle the model (with time functions given by name, "
            "'t' or 'log', or defined at module level).",
        )

    # -- partial effects ------------------------------------------------------

    #: The partial-effects plot reads the units' rows, not the episodes.
    _PARTIAL_EFFECTS_UNIT_ROWS = True

    def _partial_effects_width(self) -> int:
        return self._n_base

    def _partial_effects_rows(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray] | str:
        if self._unit_Z is None or self._unit_n is None:
            return "the model does not hold the data it was fitted to"
        return self._unit_Z, self._unit_n

    # -- beta(t) --------------------------------------------------------------

    def time_varying_effect(
        self,
        covariate: Any = None,
        x: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
    ) -> TimeVaryingEffect:
        """
        A covariate's coefficient over time with Wald bounds:

        .. math::
            \\beta_j(t) = \\beta_j + \\sum_k \\gamma_k g_k(t), \\qquad
            \\operatorname{se}(\\beta_j(t)) = \\sqrt{a(t)' V a(t)},

        the sum over the interactions of covariate ``j``, :math:`V` the
        coefficients' covariance and :math:`a(t)` 1 at :math:`\\beta_j`,
        :math:`g_k(t)` at :math:`\\gamma_k` and 0 elsewhere. The bounds
        are :math:`\\beta_j(t) \\pm z_{1 - \\alpha/2}
        \\operatorname{se}`. :math:`\\beta_j(t)` is the log hazard ratio
        per unit of the covariate at time ``t`` (positive means a
        shorter life).

        Parameters
        ----------
        covariate : int or str, optional
            The covariate, by position in ``Z`` or by name. Default: the
            first one with a time interaction.
        x : array_like, optional
            The times. Default: the distinct event times.
        alpha_ci : float, optional
            The bounds' total tail probability. Default 0.05.

        Returns
        -------
        TimeVaryingEffect
            ``beta``, ``se``, ``lower``, ``upper`` (and ``hazard_ratio``)
            at each time, with ``to_frame()``, ``to_dict()`` and
            ``plot()``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import CoxPH
        >>> rng = np.random.default_rng(0)
        >>> z = rng.binomial(1, 0.5, 1000).astype(float)
        >>> e = rng.exponential(size=1000)
        >>> x = np.where(z == 1, np.sqrt(e), e)  # z's hazard ratio is 2t

        The log hazard ratio of ``z`` is :math:`\\log 2 + \\log t`, so
        :math:`\\beta = 0.69` and :math:`\\gamma = 1`:

        >>> model = CoxPH.fit(x, z, time_interaction={0: "log"})
        >>> model.params.round(2)
        array([0.68, 0.94])
        >>> effect = model.time_varying_effect(0, [0.25, 1.0])
        >>> effect.hazard_ratio.round(2)
        array([0.53, 1.97])
        """
        check_alpha_ci(alpha_ci)
        p = self._n_base
        if covariate is None:
            j = self.time_interaction[0].column
        else:
            j = _column(covariate, p, getattr(self, "feature_names", None))
        if x is None:
            x = self._event_times()
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        weights = np.zeros((xq.size, p + len(self.time_interaction)))
        weights[:, j] = 1.0
        for k, term in enumerate(self.time_interaction):
            if term.column == j:
                weights[:, p + k] = term.values(xq)
        beta = weights @ np.asarray(self.params, dtype=float)
        cov = np.asarray(self.covariance(), dtype=float)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
        z = norm.isf(alpha_ci / 2)
        return TimeVaryingEffect(
            covariate=self._base_names()[j],
            x=xq,
            beta=beta,
            se=se,
            lower=beta - z * se,
            upper=beta + z * se,
            alpha_ci=float(alpha_ci),
        )

    def _event_times(self) -> npt.NDArray:
        """The distinct event times (over every stratum)."""
        if self.is_stratified:
            return np.unique(
                np.concatenate(
                    [
                        np.asarray(b["x"])[np.asarray(b["d"]) > 0]
                        for b in self.strata_baselines.values()
                    ]
                )
            )
        return np.asarray(self.x)[np.asarray(self.d) > 0]
