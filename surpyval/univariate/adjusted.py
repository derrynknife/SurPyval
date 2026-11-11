"""Adjusted survival curves for groups that were not assigned at random.

When the groups being compared differ in other ways as well (one supplier's
parts went to the harder duty, one treatment went to the older patients),
the plain per-group Kaplan-Meier curves mix the effect of the group with
the effect of those other differences. An *adjusted* curve answers "what
would the survival curve be if everyone in the sample had been in this
group?", so that the groups are compared on the same mix of covariates.

Two routes are supported:

* **Weighting.** Each unit is weighted by the inverse of the probability
  of being in its own group given its covariates. The weighted
  Kaplan-Meier estimate is ``KaplanMeier.fit(x, c, w=weights)``, and
  ``sf_diff`` and ``rmst_diff`` compare two of them with robust
  variances. :func:`weight_diagnostics` checks the weights that are
  brought to it: the effective sample size, extreme weights, and the
  balance of the covariates before and after weighting.
* **Direct adjustment** (standardisation, g-computation).
  :func:`direct_adjustment` takes a fitted regression model that has the
  group as one of its covariates, predicts every unit's survival curve
  with the group set to each level in turn, and averages the predictions
  over the sample.

Nothing here fits a model: the regression model and the weights are the
caller's.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import check_alpha_ci, check_option

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: The ``covariates`` values of :func:`direct_adjustment`.
COVARIATES = ("sampled", "fixed")

#: The largest number of survival values held at once while averaging.
_CHUNK = 2_000_000


# ---------------------------------------------------------------------------
# The population the predictions are averaged over
# ---------------------------------------------------------------------------


def _fitted_rows(model: Any) -> tuple[Any, Any, Any] | None:
    """The covariate rows the model was fitted to, with their counts and
    weights, or ``None`` where the model does not keep one row per
    unit."""
    data = getattr(model, "_fit_data", None)
    if isinstance(data, dict) and data.get("Z") is not None:
        return data["Z"], data.get("n"), data.get("w")
    data = getattr(model, "data", None)
    Z = getattr(data, "Z", None)
    if Z is None:
        return None
    return Z, getattr(data, "n", None), getattr(data, "w", None)


def _layout(model: Any) -> Any:
    """How the model names its covariates (a formula's variables, named
    columns or positions), as ``plot_partial_effects`` reads them."""
    from surpyval.univariate.regression._partial_effects import Covariates

    if hasattr(model, "_partial_effects_width"):
        return Covariates(model)
    return Covariates(_Named(model))


class _Named:
    """What ``Covariates`` reads of a model, for one that does not draw
    partial effects."""

    def __init__(self, model: Any) -> None:
        self._model_spec = getattr(model, "_model_spec", None)
        self.feature_names = getattr(model, "feature_names", None)
        self.formula = getattr(model, "formula", None)
        self._width = _width(model)

    def _partial_effects_width(self) -> int:
        return self._width


def _width(model: Any) -> int:
    for name in ("_partial_effects_width", "_n_covariates"):
        fn = getattr(model, name, None)
        if fn is not None:
            return int(fn())
    fitted = (
        _fitted_rows(model) if callable(getattr(model, "sf", None)) else None
    )
    if fitted is not None and np.ndim(fitted[0]) == 2:
        return int(np.shape(fitted[0])[1])
    raise TypeError(
        "direct_adjustment needs a fitted regression model whose "
        "predictions take covariates, sf(x, Z); got {}.".format(
            type(model).__name__
        )
    )


def _levels_of(values: npt.NDArray) -> list:
    out = []
    for value in values:
        value = value.item() if hasattr(value, "item") else value
        if value not in out:
            out.append(value)
    try:
        return sorted(out)
    except TypeError:
        return out


def _check_levels(levels: Any, found: list) -> list:
    if levels is None:
        levels = found
    elif isinstance(levels, (str, bytes)) or not np.iterable(levels):
        raise ValueError(
            "'levels' must be a list of the values of the group to "
            "compare, e.g. levels=[0, 1]; got {!r}.".format(levels)
        )
    levels = [v.item() if hasattr(v, "item") else v for v in levels]
    if len(levels) == 0:
        raise ValueError("'levels' is empty: give the group's values.")
    if len(set(map(repr, levels))) != len(levels):
        raise ValueError(
            "'levels' repeats a value: {}. Give each level once.".format(
                levels
            )
        )
    return levels


def _formula_population(
    model: Any, group: Any, levels: Any, Z: Any
) -> tuple[list, list[npt.NDArray]]:
    """The design rows of a formula model at each level: the group is a
    variable of the frame ``Z``, and the formula builds its columns (and
    any interaction with it) from the changed frame."""
    import pandas as pd

    layout = _layout(model)
    name = layout.resolve(group, "group")
    if not isinstance(Z, pd.DataFrame):
        raise ValueError(
            "This model was fitted with a formula, so direct_adjustment "
            "needs the population as a DataFrame of the formula's "
            "variables, to set {!r} in it and let the formula rebuild "
            "its columns: pass Z=df (the frame the model was fitted "
            "to).".format(name)
        )
    if name not in Z.columns:
        raise ValueError(
            "The group {!r} is not a column of Z; its columns are "
            "{}.".format(name, list(Z.columns))
        )
    column = Z[name]
    levels = _check_levels(levels, _levels_of(column.dropna().to_numpy()))
    rows = []
    for level in levels:
        frame = Z.copy()
        if isinstance(column.dtype, pd.CategoricalDtype):
            if level not in column.cat.categories:
                raise ValueError(
                    "The level {!r} is not a category of {!r}; its "
                    "categories are {}.".format(
                        level, name, list(column.cat.categories)
                    )
                )
            frame[name] = pd.Categorical(
                [level] * len(frame), dtype=column.dtype
            )
        else:
            frame[name] = pd.Series(
                [level] * len(frame), index=frame.index, dtype=column.dtype
            )
        rows.append(np.asarray(model._prepare_Z(frame), dtype=float))
    return levels, rows


def _column_population(
    model: Any, group: Any, levels: Any, base: npt.NDArray
) -> tuple[list, list[npt.NDArray]]:
    """The covariate rows at each level for a model fitted on columns:
    the group's column is overwritten with the level."""
    layout = _layout(model)
    name = layout.resolve(group, "group")
    j = name if layout.layout == "positional" else layout.names.index(name)
    levels = _check_levels(levels, _levels_of(np.unique(base[:, j])))
    rows = []
    for level in levels:
        if isinstance(level, (bool, np.bool_)) or not isinstance(
            level, numbers.Real
        ):
            raise ValueError(
                "The level {!r} of the group is not a number, and this "
                "model's covariates are numeric columns: give the "
                "column's values (e.g. levels=[0, 1]).".format(level)
            )
        at = base.copy()
        at[:, j] = float(level)
        rows.append(at)
    return levels, rows


def _population(
    model: Any, group: Any, levels: Any, Z: Any, w: Any
) -> tuple[list, list[npt.NDArray], npt.NDArray, npt.NDArray]:
    """``(levels, rows, n, w)``: the covariate rows of the population at
    each level, with each row's count and weight."""
    from surpyval.utils.shapes import covariate_rows
    from surpyval.utils.weights import check_weights

    width = _width(model)
    counts: Any = None
    weights: Any = None
    if getattr(model, "_model_spec", None) is not None:
        levels, rows = _formula_population(model, group, levels, Z)
    else:
        if Z is None:
            fitted = _fitted_rows(model)
            if fitted is None:
                raise ValueError(
                    "This model does not keep the covariate rows it was "
                    "fitted to (it was built from parameters, restored "
                    "from a dict, or fitted to time-varying covariates), "
                    "so direct_adjustment has no population to average "
                    "over: pass Z, one row of covariates per unit."
                )
            Z, counts, weights = fitted
            base = np.array(Z, dtype=float)
        else:
            base = np.array(
                covariate_rows(model._prepare_Z(Z), width), dtype=float
            )
        if base.ndim != 2 or base.shape[1] != width:
            raise ValueError(
                "Z must have one row per unit and the model's {} "
                "covariate column(s); got shape {}.".format(
                    width, np.shape(base)
                )
            )
        levels, rows = _column_population(model, group, levels, base)
    k = rows[0].shape[0]
    if k == 0:
        raise ValueError("Z has no rows: there is nothing to average over.")
    for at in rows:
        if not np.all(np.isfinite(at)):
            raise ValueError(
                "The population's covariates contain missing or infinite "
                "values. Leave those rows out of Z, or fill them in: a "
                "unit without covariates has no prediction to average."
            )
    n = np.ones(k) if counts is None else np.asarray(counts, dtype=float)
    if w is not None:
        weights = check_weights(w, k)
    w_arr = np.ones(k) if weights is None else np.asarray(weights, float)
    return levels, rows, n, w_arr


# ---------------------------------------------------------------------------
# Engines: one model family's predictions and their variance
# ---------------------------------------------------------------------------


class _Engine:
    """The per-unit predictions of one model family at the population's
    rows, and the variance, from the model's own estimation error, of a
    weighted average of them."""

    #: Why this family has no bounds, or None where it has them.
    refusal: str | None = None
    #: Whether the curves are step functions.
    step = False

    def __init__(self, model: Any, rows: list[npt.NDArray]) -> None:
        self.model = model
        self.rows = rows

    def row_sf(self, t: npt.NDArray) -> npt.NDArray:
        """Survival of every row at every time, ``(levels, rows, t)``."""
        raise NotImplementedError

    def sf_var(
        self, t: npt.NDArray, L: npt.NDArray, omega: npt.NDArray
    ) -> npt.NDArray:
        """The variance at each time of ``sum_g L_g mean_g(t)``, the
        covariates held fixed."""
        raise NotImplementedError

    def row_rmst(self, tau: float) -> npt.NDArray:
        """Restricted mean of every row, ``(levels, rows)``."""
        raise NotImplementedError

    def rmst_var(
        self, tau: float, L: npt.NDArray, omega: npt.NDArray
    ) -> float:
        raise NotImplementedError

    def last_time(self) -> float | None:
        """The largest observed time, where the model knows it."""
        return None

    def check_tau(self, tau: float) -> None:
        """Refuse a horizon the model cannot integrate to."""


def _chunks(n_rows: int, n_times: int) -> list[slice]:
    size = max(1, _CHUNK // max(n_rows, 1))
    return [slice(i, i + size) for i in range(0, n_times, size)]


class _Parametric(_Engine):
    """The parametric regression families: the delta method through the
    covariance of the fitted parameters."""

    def __init__(
        self, model: Any, rows: list[npt.NDArray], robust: bool, cluster: Any
    ) -> None:
        super().__init__(model, rows)
        model._check_inference()
        self.params, center, self.cov = model._wald_state(
            robust, cluster, "wald"
        )
        self.rows_c = [
            np.asarray(model._centred(at, center), dtype=float) for at in rows
        ]
        self.lower = float(model.distribution.support[0])

    def _sf_at(self, params: Any, t: npt.NDArray) -> npt.NDArray:
        inside = self.lower + 1.0 if np.isfinite(self.lower) else 0.0
        out = np.empty((len(self.rows_c), self.rows_c[0].shape[0], t.size))
        below = t < self.lower
        t_in = np.where(below, inside, t)
        for g, at in enumerate(self.rows_c):
            k = at.shape[0]
            for part in _chunks(k, t.size):
                tt = t_in[part]
                with np.errstate(all="ignore"):
                    S = np.asarray(
                        self.model.model.sf(
                            np.tile(tt, k),
                            np.repeat(at, tt.size, axis=0),
                            *params,
                        ),
                        dtype=float,
                    )
                out[g, :, part] = S.reshape(k, tt.size)
        out[:, :, below] = 1.0
        return out

    def row_sf(self, t: npt.NDArray) -> npt.NDArray:
        return self._sf_at(self.params, t)

    def sf_var(
        self, t: npt.NDArray, L: npt.NDArray, omega: npt.NDArray
    ) -> npt.NDArray:
        from surpyval.utils.linalg import delta_method_se

        def value(p: npt.NDArray) -> npt.NDArray:
            return np.einsum("g,gik,i->k", L, self._sf_at(p, t), omega)

        with np.errstate(all="ignore"):
            return (
                np.asarray(
                    delta_method_se(value, self.params, self.cov), float
                )
                ** 2
            )

    def _rmst_at(self, params: Any, tau: float) -> npt.NDArray:
        return np.stack(
            [
                self.model._rmst_at(np.full(at.shape[0], tau), at, params)
                for at in self.rows_c
            ]
        )

    def row_rmst(self, tau: float) -> npt.NDArray:
        return self._rmst_at(self.params, tau)

    def rmst_var(
        self, tau: float, L: npt.NDArray, omega: npt.NDArray
    ) -> float:
        from surpyval.utils.linalg import delta_method_se

        def value(p: npt.NDArray) -> npt.NDArray:
            return np.einsum("g,gi,i->", L, self._rmst_at(p, tau), omega)

        with np.errstate(all="ignore"):
            se = delta_method_se(value, self.params, self.cov)
        return float(np.ravel(se)[0]) ** 2

    def last_time(self) -> float | None:
        data = getattr(self.model, "data", None)
        x = getattr(data, "x", None)
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        return float(np.max(x[np.isfinite(x)]))


class _Cox(_Engine):
    """The Cox model: the baseline's variance and the coefficients', as
    in the bounds of its own predicted curve (``CoxPH.cb``)."""

    step = True

    def __init__(
        self, model: Any, rows: list[npt.NDArray], robust: bool, cluster: Any
    ) -> None:
        super().__init__(model, rows)
        if getattr(model, "is_stratified", False):
            raise ValueError(
                "direct_adjustment does not take a stratified Cox model: "
                "each stratum has its own baseline, and the population's "
                "rows would each need a stratum. Fit the model with the "
                "stratifying variable as a covariate, or use a "
                "parametric regression."
            )
        self.center = np.asarray(model._center(), dtype=float)
        self.eta = [np.asarray(model._log_risk(at), float) for at in rows]
        self.times = np.asarray(model.x, dtype=float)
        self.H0 = np.asarray(model.H0, dtype=float)
        self._robust = (robust, cluster)
        if cluster is not None and not robust:
            raise ValueError(
                "'cluster' was given without robust=True: pass robust=True "
                "for the cluster-robust covariance."
            )
        self._parts: dict | None = None
        self._cov: npt.NDArray | None = None

    def _variance_parts(self) -> tuple[dict, npt.NDArray]:
        if self._parts is None:
            robust, cluster = self._robust
            self._parts = self.model._curve_parts(None)
            cov = (
                self.model.robust_covariance(cluster)
                if robust
                else self.model.covariance()
            )
            cov = np.asarray(cov, dtype=float)
            self._cov = np.where(np.isnan(cov), 0.0, cov)
        assert self._cov is not None
        return self._parts, self._cov

    def _baseline(self, t: npt.NDArray) -> npt.NDArray:
        return np.asarray(
            self.model._baseline_step(self.times, self.H0, t), dtype=float
        )

    def row_sf(self, t: npt.NDArray) -> npt.NDArray:
        H0 = self._baseline(t)
        with np.errstate(over="ignore"):
            return np.stack(
                [
                    np.exp(-np.exp(eta)[:, None] * H0[None, :])
                    for eta in self.eta
                ]
            )

    def _gradients(
        self, at: npt.NDArray, L: npt.NDArray, omega: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """At the event-time positions ``at`` (-1 before the first): the
        derivative of ``sum_g L_g mean_g`` in the baseline's noise
        (``a``), in the coefficients (``c``), and the baseline's variance
        ``V``; the mean falls by ``a dW + c' dbeta``."""
        parts, _ = self._variance_parts()
        ok = at >= 0
        idx = np.maximum(at, 0)
        if parts["times"].size == 0:
            zero = np.zeros(at.size)
            return zero, np.zeros((at.size, self.center.size)), zero
        A = np.where(ok, parts["A"][idx], 0.0)
        V = np.where(ok, parts["V"][idx], 0.0)
        B = np.where(ok[:, None], parts["B"][idx], 0.0)
        a = np.zeros(at.size)
        M = np.zeros((at.size, self.center.size))
        for g, rows in enumerate(self.rows):
            if L[g] == 0:
                continue
            risk = np.exp(self.eta[g])
            for part in _chunks(rows.shape[0], at.size):
                with np.errstate(over="ignore"):
                    S = np.exp(-risk[:, None] * A[None, part])
                weight = (omega * risk)[:, None] * S
                a[part] += L[g] * weight.sum(axis=0)
                M[part] += L[g] * (weight.T @ (rows - self.center))
        c = A[:, None] * M - B * a[:, None]
        return a, c, V

    def sf_var(
        self, t: npt.NDArray, L: npt.NDArray, omega: npt.NDArray
    ) -> npt.NDArray:
        parts, cov = self._variance_parts()
        at = np.searchsorted(parts["times"], t, side="right") - 1
        a, c, V = self._gradients(at, L, omega)
        return a**2 * V + np.einsum("ki,ij,kj->k", c, cov, c)

    def _steps(self, tau: float) -> tuple[npt.NDArray, npt.NDArray, float]:
        """The event times below ``tau`` (as positions), the width of the
        step each starts, and the width before the first."""
        parts, _ = self._variance_parts()
        times = parts["times"]
        at = np.flatnonzero(times < tau)
        edges = np.append(times[at], tau)
        first = float(edges[0]) if at.size else tau
        return at, np.diff(edges), first

    def row_rmst(self, tau: float) -> npt.NDArray:
        at, widths, first = self._steps(tau)
        parts, _ = self._variance_parts()
        A = parts["A"][at]
        out = []
        for eta in self.eta:
            with np.errstate(over="ignore"):
                S = np.exp(-np.exp(eta)[:, None] * A[None, :])
            out.append(first + S @ widths)
        return np.stack(out)

    def rmst_var(
        self, tau: float, L: npt.NDArray, omega: npt.NDArray
    ) -> float:
        parts, cov = self._variance_parts()
        at, widths, _ = self._steps(tau)
        if at.size == 0:
            return 0.0
        a, c, _ = self._gradients(at, L, omega)
        dV = np.diff(parts["V"][at], prepend=0.0)
        if at[0] > 0:
            dV[0] = parts["V"][at[0]]
        tail = np.cumsum((widths * a)[::-1])[::-1]
        C = widths @ c
        return float(np.sum(dV * tail**2) + C @ cov @ C)

    def last_time(self) -> float | None:
        x = np.asarray(self.model._fit_data["x"], dtype=float)
        return float(np.max(x[np.isfinite(x)]))

    def check_tau(self, tau: float) -> None:
        last = self.last_time()
        if last is not None and tau > last:
            raise ValueError(
                "tau = {:g} is past the largest observed time, {:g}: a "
                "Cox model's baseline is not estimated beyond it, so the "
                "restricted mean to tau is not either. Use a tau at or "
                "below {:g}, or a parametric regression, which "
                "extrapolates.".format(tau, last, last)
            )


class _PointOnly(_Engine):
    """Any other model with ``sf(x, Z)``: the averaged curve, without
    bounds."""

    def __init__(self, model: Any, rows: list[npt.NDArray]) -> None:
        super().__init__(model, rows)
        self.refusal = (
            "The adjusted curves of a {} have no bounds: the variance of "
            "an average of its predictions is built for the parametric "
            "regression families (PH, AFT, PO, AH and the accelerated "
            "life models) and CoxPH only. Fit one of those for bounds "
            "and tests, or bootstrap: refit the model to resampled rows "
            "and repeat direct_adjustment.".format(type(model).__name__)
        )

    def row_sf(self, t: npt.NDArray) -> npt.NDArray:
        out = np.empty((len(self.rows), self.rows[0].shape[0], t.size))
        for g, at in enumerate(self.rows):
            for k, time in enumerate(t):
                out[g, :, k] = np.asarray(
                    self.model.sf(np.full(at.shape[0], time), at), float
                )
        return out

    def row_rmst(self, tau: float) -> npt.NDArray:
        raise ValueError(
            "The restricted mean of the adjusted curves of a {} is not "
            "built: it is for the parametric regression families and "
            "CoxPH. Integrate sf(x, level) over a grid of times for the "
            "area itself.".format(type(self.model).__name__)
        )


def _engine(
    model: Any, rows: list[npt.NDArray], robust: bool, cluster: Any
) -> _Engine:
    from surpyval.univariate.regression.parametric_regression_model import (
        ParametricRegressionModel,
    )
    from surpyval.univariate.regression.semi_parametric_regression_model import (  # noqa: E501
        SemiParametricRegressionModel,
    )

    if isinstance(model, ParametricRegressionModel):
        try:
            return _Parametric(model, rows, robust, cluster)
        except ValueError as error:
            # No covariance (built from parameters, or restored without
            # one): the curves still average, without bounds.
            if robust or cluster is not None:
                raise
            point = _PointOnly(model, rows)
            point.refusal = str(error)
            return point
    if (
        isinstance(model, SemiParametricRegressionModel)
        and getattr(model, "_fit_data", None)
        and hasattr(model, "_curve_parts")
    ):
        return _Cox(model, rows, robust, cluster)
    if robust or cluster is not None:
        raise ValueError(
            "robust and cluster apply to the bounds, which a {} has "
            "none of here; leave them out.".format(type(model).__name__)
        )
    if not callable(getattr(model, "sf", None)):
        raise TypeError(
            "direct_adjustment needs a fitted regression model whose "
            "predictions take covariates, sf(x, Z); got {}.".format(
                type(model).__name__
            )
        )
    return _PointOnly(model, rows)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


class AdjustedCurves:
    r"""
    The directly adjusted (standardised) survival curve of each level of
    a group, made by :func:`direct_adjustment`.

    The curve of a level :math:`g` is the average, over the units of the
    population, of the model's predicted survival with the group set to
    :math:`g` and every other covariate left as it is:

    .. math::
        \bar S_g(t) = \sum_i \omega_i\, \hat S(t \mid g, Z_i), \qquad
        \omega_i = \frac{n_i w_i}{\sum_j n_j w_j}.

    It is the survival curve the whole population would have had in
    group :math:`g`, if the model is right and its covariates are the
    ones that matter. Every level is averaged over the same units, so a
    difference between two curves is a difference between the groups and
    not between the units that happened to be in them.

    Attributes
    ----------
    model : the fitted regression model
    group : the covariate that was set
    levels : list
        The values it was set to; the methods take one of them.
    covariates : {"sampled", "fixed"}
        Whether the variances count the population's covariates as a
        sample (see :func:`direct_adjustment`).

    Examples
    --------
    Two suppliers' parts, where supplier 1 went mostly to heavy duty: the
    95% bounds of supplier 1's adjusted survival at 40, and the adjusted
    difference in restricted mean life to 100 (supplier 1 minus 0):

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(0)
    >>> heavy = rng.binomial(1, 0.5, 400)
    >>> supplier = rng.binomial(1, np.where(heavy == 1, 0.8, 0.2))
    >>> x = 100 * np.exp(-0.9 * heavy + 0.2 * supplier)
    >>> x = x * rng.weibull(2.0, 400)
    >>> model = sp.WeibullAFT.fit(x, np.column_stack([supplier, heavy]))
    >>> adjusted = sp.direct_adjustment(model, group=0)
    >>> adjusted.cb(40, 1).round(3)
    array([0.661, 0.743])
    >>> res = adjusted.rmst_diff(100, 1, 0)
    >>> print(round(res["difference"], 1), round(res["se"], 1))
    9.6 2.0
    """

    def __init__(
        self,
        model: Any,
        group: Any,
        levels: list,
        engine: _Engine,
        n: npt.NDArray,
        w: npt.NDArray,
        covariates: str,
    ) -> None:
        self.model = model
        self.group = group
        self.levels = levels
        self.covariates = covariates
        self._engine = engine
        self._n = n
        self._w = w
        self._omega = n * w / np.sum(n * w)
        self.unit = getattr(model, "unit", None)

    def __repr__(self) -> str:
        return (
            "Directly adjusted survival curves\n"
            "  model     : {}\n"
            "  group     : {!r}, levels {}\n"
            "  population: {:g} units ({} rows), covariates {}".format(
                type(self.model).__name__,
                self.group,
                self.levels,
                float(np.sum(self._n)),
                self._n.size,
                self.covariates,
            )
        )

    # -- pieces -----------------------------------------------------------

    def _level(self, level: Any, name: str = "level") -> int:
        for i, known in enumerate(self.levels):
            if type(known) is type(level) or (
                isinstance(known, numbers.Real)
                and isinstance(level, numbers.Real)
            ):
                if known == level:
                    return i
        raise ValueError(
            "{} {!r} is not one of the adjusted levels of {!r}: "
            "{}.".format(name, level, self.group, self.levels)
        )

    def _contrast(self, *signed: tuple[int, float]) -> npt.NDArray:
        L = np.zeros(len(self.levels))
        for i, sign in signed:
            L[i] += sign
        return L

    def _sampling_var(self, values: npt.NDArray) -> npt.NDArray:
        """The variance of the weighted mean of ``values`` (rows first)
        from the units being a sample: ``sum n (w / N)^2 (v - mean)^2``."""
        if self.covariates == "fixed":
            return np.zeros(values.shape[1:])
        mean = np.tensordot(self._omega, values, axes=(0, 0))
        scale = self._n * (self._w / np.sum(self._n * self._w)) ** 2
        return np.tensordot(scale, (values - mean) ** 2, axes=(0, 0))

    def _refuse_bounds(self) -> None:
        if self._engine.refusal is not None:
            raise ValueError(self._engine.refusal)

    @staticmethod
    def _times(x: Any, name: str = "x") -> npt.NDArray:
        try:
            t = np.atleast_1d(np.asarray(x, dtype=float))
        except (TypeError, ValueError):
            raise ValueError(
                f"'{name}' must be a time or an array of times; got {x!r}"
            ) from None
        return t

    def _sf(self, t: npt.NDArray, L: npt.NDArray) -> npt.NDArray:
        rows = self._engine.row_sf(t)
        return np.einsum("g,gik,i->k", L, rows, self._omega)

    def _sf_and_var(
        self, t: npt.NDArray, L: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        self._refuse_bounds()
        rows = np.einsum("g,gik->ik", L, self._engine.row_sf(t))
        value = self._omega @ rows
        var = self._engine.sf_var(t, L, self._omega)
        var = var + self._sampling_var(rows)
        return value, np.where(np.isnan(t), np.nan, var)

    # -- curves -----------------------------------------------------------

    def sf(self, x: npt.ArrayLike, level: Any) -> npt.NDArray:
        """
        The adjusted survival of ``level`` at the times ``x``.

        Parameters
        ----------
        x : array like or scalar
            The times.
        level
            One of :attr:`levels`.

        Returns
        -------
        numpy array
            The adjusted survival, in the shape of ``x``.
        """
        t = self._times(x)
        L = self._contrast((self._level(level), 1.0))
        return self._sf(t.ravel(), L).reshape(np.shape(x))

    def ff(self, x: npt.ArrayLike, level: Any) -> npt.NDArray:
        """The adjusted failure probability of ``level`` at ``x``:
        ``1 - sf(x, level)``."""
        return 1.0 - self.sf(x, level)

    def cb(
        self,
        x: npt.ArrayLike,
        level: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the adjusted survival (or failure
        probability) of ``level`` at the times ``x``.

        They are Wald bounds on the logit of the adjusted survival, which
        keeps them between 0 and 1, from the standard error of
        :meth:`sf_diff`'s kind: the model's estimation error carried
        through the average by the delta method, plus (with
        ``covariates="sampled"``) the spread of the predictions over the
        sampled units.

        Parameters
        ----------
        x : array like or scalar
            The times.
        level
            One of :attr:`levels`.
        on : {"sf", "ff"}, optional
            The function bounded. Default "sf".
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {"two-sided", "lower", "upper"}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.

        Returns
        -------
        numpy array
            The bound(s): the shape of ``x``, with a last axis of 2 for
            two-sided bounds.

        Raises
        ------
        ValueError
            For a model family whose adjusted curves have no variance
            here (anything but the parametric regressions and ``CoxPH``);
            the message says what to use.
        """
        from scipy.special import expit, ndtri

        from surpyval.utils.linalg import bound_signs
        from surpyval.utils.validation import BOUNDS

        check_option("on", on, ("sf", "ff"))
        check_option("bound", bound, BOUNDS)
        check_alpha_ci(alpha_ci)
        t = self._times(x).ravel()
        L = self._contrast((self._level(level), 1.0))
        S, var = self._sf_and_var(t, L)
        value = S if on == "sf" else 1.0 - S
        alpha, signs = bound_signs(alpha_ci, bound)
        with np.errstate(all="ignore"):
            se = np.sqrt(var)
            psi = np.log(value) - np.log1p(-value)
            se_psi = se / (value * (1.0 - value))
            out = expit(
                psi[:, None] + signs * ndtri(1.0 - alpha) * se_psi[:, None]
            )
        edge = ~np.isfinite(psi) | (se == 0)
        out = np.where(edge[:, None], value[:, None], out)
        out = np.where(np.isnan(value + se)[:, None], np.nan, out)
        if bound == "two-sided":
            return out.reshape(np.shape(x) + (2,))
        return out[:, 0].reshape(np.shape(x))

    def sf_diff(
        self,
        x: npt.ArrayLike,
        level_a: Any,
        level_b: Any,
        alpha_ci: float = 0.05,
    ) -> dict:
        r"""
        The adjusted difference in survival between two levels at the
        times ``x``: :math:`\bar S_a(x) - \bar S_b(x)`, with its standard
        error, interval and test.

        The two curves come from one model and one set of units, so they
        are not independent: the standard error is that of the
        difference itself. Its variance has two parts,

        .. math::
            \widehat{\mathrm{Var}} = g^\top \Sigma\, g
            + \sum_i n_i \Big(\frac{w_i}{N}\Big)^2 (d_i - \bar d)^2,

        :math:`g` the gradient of the difference in the model's
        parameters and :math:`\Sigma` their covariance (for a Cox model,
        the baseline's variance and the coefficients', as in its own
        ``cb``), and :math:`d_i` the unit's own predicted difference;
        the second part is left out with ``covariates="fixed"``.

        Parameters
        ----------
        x : array like or scalar
            The time, or several.
        level_a, level_b
            Two of :attr:`levels`.
        alpha_ci : float, optional
            Significance level of the interval (default 0.05).

        Returns
        -------
        dict
            ``{"difference", "se", "lower", "upper", "p_value", "sf_a",
            "sf_b", "x"}``: ``difference`` is ``sf_a - sf_b``, ``lower``
            and ``upper`` are ``difference +- z se`` and ``p_value`` is
            the two-sided z-test of no difference. Numbers for one time,
            arrays for several.
        """
        check_alpha_ci(alpha_ci)
        t = self._times(x).ravel()
        a, b = self._level(level_a, "level_a"), self._level(level_b, "level_b")
        diff, var = self._sf_and_var(t, self._contrast((a, 1.0), (b, -1.0)))
        s_a = self._sf(t, self._contrast((a, 1.0)))
        out = self._difference(diff, var, alpha_ci)
        out.update({"sf_a": s_a, "sf_b": s_a - diff, "x": t})
        return _scalars(out) if np.ndim(x) == 0 else out

    @staticmethod
    def _difference(
        diff: npt.NDArray, var: npt.NDArray, alpha_ci: float
    ) -> dict[str, Any]:
        from scipy.stats import norm

        with np.errstate(all="ignore"):
            se = np.sqrt(var)
            z = norm.ppf(1 - alpha_ci / 2)
            p = np.where(se > 0, 2.0 * norm.sf(np.abs(diff) / se), np.nan)
        return {
            "difference": diff,
            "se": se,
            "lower": diff - z * se,
            "upper": diff + z * se,
            "p_value": p,
        }

    # -- restricted means -------------------------------------------------

    def _tau(self, tau: Any, what: str) -> float:
        if tau is None:
            last = self._engine.last_time()
            if last is None:
                raise ValueError(
                    f"{what}: this model holds no data to take the default "
                    "tau (the largest observed time) from; pass tau."
                )
            return last
        if np.size(tau) != 1:
            raise ValueError(f"{what} takes one tau.")
        tau = float(np.ravel(np.asarray(tau, dtype=float))[0])
        if not (np.isfinite(tau) and tau >= 0):
            raise ValueError(
                "{}: tau = {!r} is not a finite time at or above 0. The "
                "restricted mean is the area under the adjusted survival "
                "curve from 0 to tau.".format(what, tau)
            )
        self._engine.check_tau(tau)
        return tau

    def _rmst_and_var(
        self, tau: float, L: npt.NDArray
    ) -> tuple[float, float, npt.NDArray]:
        rows = self._engine.row_rmst(tau)
        self._refuse_bounds()
        combined = L @ rows
        var = self._engine.rmst_var(tau, L, self._omega)
        var = var + float(self._sampling_var(combined))
        return float(self._omega @ combined), var, rows @ self._omega

    def rmst(
        self, tau: float | None, level: Any, alpha_ci: float = 0.05
    ) -> dict:
        r"""
        The adjusted restricted mean survival time of ``level``: the
        area under its adjusted survival curve from 0 to ``tau``, the
        average time lived within the first ``tau`` if every unit were in
        that group.

        Parameters
        ----------
        tau : scalar or None
            The horizon; ``None`` takes the largest observed time. A Cox
            model refuses a ``tau`` past it (its baseline ends there).
        level
            One of :attr:`levels`.
        alpha_ci : float, optional
            Significance level of the interval (default 0.05).

        Returns
        -------
        dict
            ``{"rmst", "se", "lower", "upper", "tau"}``; the interval is
            a Wald bound on the logit of ``rmst / tau``, so it stays in
            ``(0, tau)``, as the regression models' ``rmst`` is.
        """
        from scipy.special import expit, ndtri

        check_alpha_ci(alpha_ci)
        tau = self._tau(tau, "rmst")
        L = self._contrast((self._level(level), 1.0))
        value, var, _ = self._rmst_and_var(tau, L)
        se = float(np.sqrt(var))
        lower = upper = value
        if 0 < value < tau and se > 0:
            psi = np.log(value) - np.log(tau - value)
            se_psi = se * (1.0 / value + 1.0 / (tau - value))
            z = ndtri(1.0 - alpha_ci / 2)
            lower = float(tau * expit(psi - z * se_psi))
            upper = float(tau * expit(psi + z * se_psi))
        return {
            "rmst": value,
            "se": se,
            "lower": lower,
            "upper": upper,
            "tau": tau,
        }

    def rmst_diff(
        self,
        tau: float | None,
        level_a: Any,
        level_b: Any,
        alpha_ci: float = 0.05,
    ) -> dict:
        """
        The adjusted difference in restricted mean survival time between
        two levels: the extra time lived within the first ``tau`` if
        every unit were in group ``level_a`` rather than ``level_b``.

        The standard error is that of the difference itself, with the
        two parts :meth:`sf_diff` describes.

        Parameters
        ----------
        tau : scalar or None
            The common horizon; ``None`` takes the largest observed time.
        level_a, level_b
            Two of :attr:`levels`.
        alpha_ci : float, optional
            Significance level of the interval and test (default 0.05).

        Returns
        -------
        dict
            ``{"difference", "se", "lower", "upper", "p_value", "ratio",
            "rmst_a", "rmst_b", "tau"}``, as :func:`surpyval.rmst_diff`.
        """
        check_alpha_ci(alpha_ci)
        tau = self._tau(tau, "rmst_diff")
        a, b = self._level(level_a, "level_a"), self._level(level_b, "level_b")
        diff, var, means = self._rmst_and_var(
            tau, self._contrast((a, 1.0), (b, -1.0))
        )
        out = _scalars(
            self._difference(np.array([diff]), np.array([var]), alpha_ci)
        )
        mu_a, mu_b = float(means[a]), float(means[b])
        out.update(
            {
                "ratio": mu_a / mu_b if mu_b != 0 else float("nan"),
                "rmst_a": mu_a,
                "rmst_b": mu_b,
                "tau": tau,
            }
        )
        return out

    # -- plot -------------------------------------------------------------

    def plot(
        self,
        x: npt.ArrayLike | None = None,
        alpha_ci: float | None = 0.05,
        ax: Axes | None = None,
    ) -> Axes:
        """
        Draw the adjusted survival curve of every level, each with its
        two-sided confidence band.

        Parameters
        ----------
        x : array like, optional
            The times to draw at. By default 200 points from 0 to the
            largest observed time (the event times of a Cox model, drawn
            as steps).
        alpha_ci : float or None, optional
            The band's total tail probability (default 0.05); ``None``
            draws no band, and a model without bounds draws none.
        ax : matplotlib axes, optional
            The axes to draw on; the current ones by default.

        Returns
        -------
        matplotlib axes
        """
        import matplotlib.pyplot as plt

        from surpyval import plotting
        from surpyval.utils.units import label_axes

        if alpha_ci is not None:
            check_alpha_ci(alpha_ci)
        if ax is None:
            ax = plt.gcf().gca()
        if x is None:
            last = self._engine.last_time()
            if last is None:
                raise ValueError(
                    "plot: this model holds no data to take the times "
                    "from; pass x."
                )
            if self._engine.step:
                times = np.asarray(self.model.x, dtype=float)
                t = np.unique(np.concatenate([[0.0], times, [last]]))
            else:
                t = np.linspace(0.0, last, 200)
        else:
            t = np.sort(self._times(x).ravel())
        band = alpha_ci is not None and self._engine.refusal is None
        step = self._engine.step
        for level in self.levels:
            color = plotting.next_color(ax)
            label = f"{self.group} = {level}"
            if step:
                (line,) = ax.step(
                    t,
                    self.sf(t, level),
                    where="post",
                    color=color,
                    label=label,
                )
            else:
                (line,) = ax.plot(
                    t, self.sf(t, level), color=color, label=label
                )
            if band and alpha_ci is not None:
                cb = self.cb(t, level, alpha_ci=alpha_ci)
                ax.fill_between(
                    t,
                    cb[:, 0],
                    cb[:, 1],
                    step="post" if step else None,
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0,
                )
        ax.set_ylim(0, 1.02)
        label_axes(ax, self.unit, "Adjusted survival probability")
        plotting.style_axes(ax)
        plotting.style_legend(ax)
        return ax


def _scalars(out: dict[str, Any]) -> dict[str, Any]:
    return {
        k: float(np.ravel(v)[0]) if isinstance(v, np.ndarray) else v
        for k, v in out.items()
    }


def direct_adjustment(
    model: Any,
    group: Any,
    levels: Any = None,
    Z: Any = None,
    w: npt.ArrayLike | None = None,
    covariates: str = "sampled",
    robust: bool = False,
    cluster: npt.ArrayLike | None = None,
) -> AdjustedCurves:
    r"""
    Directly adjusted survival curves from a fitted regression model
    (standardisation, or g-computation): what each group's survival
    curve would be if the whole sample were in that group.

    Fit a regression with the group and the covariates that differ
    between the groups (the confounders). For each level of the group,
    every unit's survival curve is predicted with the group set to that
    level and its other covariates left alone, and the predictions are
    averaged:

    .. math::
        \bar S_g(t) = \sum_i \omega_i\, \hat S(t \mid g, Z_i).

    The curves of the levels are then compared on one and the same mix of
    covariates. The answer is as good as the model: it must contain the
    covariates that drive both the group and the outcome, in the right
    form.

    Parameters
    ----------
    model : a fitted regression model
        A parametric regression (``WeibullPH``, ``LogNormalAFT`` and the
        other PH, AFT, PO, AH and accelerated-life fits) or ``CoxPH``,
        fitted with the group among its covariates. Any other model with
        ``sf(x, Z)`` gives the curves without bounds.
    group : int or str
        The covariate that holds the group: its column position in ``Z``
        for a model fitted on arrays, its column name for one fitted
        with named covariates, or the variable's name for one fitted
        with a formula.
    levels : list, optional
        The values of the group to adjust to. By default each distinct
        value it takes in the population.
    Z : array like or DataFrame, optional
        The population to average over, one row of covariates per unit.
        By default the rows the model was fitted to, with their counts
        ``n`` and weights ``w``. A model fitted with a formula needs the
        DataFrame of the formula's variables here, so that the formula
        can rebuild its columns (and any interaction with the group).
    w : array like, optional
        A positive weight for each row of the population, when it should
        stand for a different population than the sample (for example a
        reference duty mix). By default the fit's own weights, or 1.
    covariates : {"sampled", "fixed"}, optional
        ``"sampled"`` (the default) treats the units as a random sample
        of the population the curve is about, and adds the spread of
        their predictions to the variance. ``"fixed"`` treats the rows
        as the population itself (a reference population, or exactly
        these units), leaving only the model's estimation error.
    robust : bool, optional
        ``True`` uses the model's robust (sandwich) covariance. Default
        ``False``. (A Cox model fitted with weights already reports its
        robust covariance.)
    cluster : array_like, optional
        With ``robust=True``, a cluster label for each row given to the
        fit.

    Returns
    -------
    AdjustedCurves
        With ``sf``, ``ff``, ``cb``, ``sf_diff``, ``rmst``, ``rmst_diff``
        and ``plot``, each taking a level (or two) of :attr:`levels`.

    Raises
    ------
    ValueError
        For a group that is not a covariate of the model, a population
        with missing values, a formula model without the DataFrame
        ``Z``, a stratified Cox model, or a model that holds no rows and
        is given none.

    Examples
    --------
    Two suppliers' parts, where supplier 1 went mostly to heavy duty:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(0)
    >>> heavy = rng.binomial(1, 0.5, 400)
    >>> supplier = rng.binomial(1, np.where(heavy == 1, 0.8, 0.2))
    >>> life = 100 * np.exp(-0.9 * heavy + 0.2 * supplier)
    >>> x = life * rng.weibull(2.0, 400)
    >>> Z = np.column_stack([supplier, heavy])
    >>> model = sp.WeibullAFT.fit(x, Z)
    >>> adjusted = sp.direct_adjustment(model, group=0)
    >>> adjusted.levels
    [0.0, 1.0]
    >>> adjusted.sf([40, 80], 1).round(3)
    array([0.704, 0.36 ])
    >>> res = adjusted.sf_diff(40, 1, 0)
    >>> print(round(res["difference"], 3), round(res["se"], 3))
    0.135 0.031

    Notes
    -----
    The variance treats the model's estimation error and the sampling of
    the units as independent, which they are for a correctly specified
    model (a unit's score has mean zero whatever its covariates).

    References
    ----------
    Makuch, R. W. (1982), "Adjusted survival curve estimation using
    covariates", Journal of Chronic Diseases 35, 437-443.

    Zhang, X., Loberiza, F. R., Klein, J. P. and Zhang, M.-J. (2007), "A
    SAS macro for estimation of direct adjusted survival curves based on
    a stratified Cox regression model", Computer Methods and Programs in
    Biomedicine 88, 95-101.

    Hernan, M. A. and Robins, J. M. (2020), "Causal Inference: What If",
    chapters 13 and 17.
    """
    check_option("covariates", covariates, COVARIATES)
    levels, rows, n, w_arr = _population(model, group, levels, Z, w)
    engine = _engine(model, rows, bool(robust), cluster)
    return AdjustedCurves(model, group, levels, engine, n, w_arr, covariates)


# ---------------------------------------------------------------------------
# Weight diagnostics
# ---------------------------------------------------------------------------


def _weighted_moments(
    Z: npt.NDArray, w: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The weighted mean and variance (weights as frequencies, divisor
    ``sum w``) of each column."""
    total = np.sum(w)
    mean = w @ Z / total
    var = w @ (Z - mean) ** 2 / total
    return mean, var


def weight_diagnostics(
    w: npt.ArrayLike,
    group: npt.ArrayLike,
    Z: Any = None,
    extreme: float = 10.0,
    smd: float = 0.1,
) -> dict:
    r"""
    Check a set of weights before trusting a weighted comparison of
    groups: how much information is left, whether a few units carry it,
    and whether the weights balanced the covariates.

    The weights are the caller's (inverse probability of group
    membership, overlap, matching or survey weights); nothing is fitted
    here. Three things are reported.

    * **Effective sample size** of each group, Kish's
      :math:`(\sum w)^2 / \sum w^2`: the number of equally weighted
      units that would give the same precision. It equals the group's
      size with equal weights and falls as they become uneven.
    * **Extreme weights**: each group's largest weight, the share of the
      group's total weight it carries, and the number of weights more
      than ``extreme`` times the group's mean weight. One unit with a
      large share means the weighted curve is close to that unit's
      history.
    * **Balance**, where the covariates ``Z`` are given: for each
      covariate and each pair of groups, the standardised mean
      difference before and after weighting,

      .. math::
          d = \frac{\bar z_a - \bar z_b}{\sqrt{(s_a^2 + s_b^2)/2}},

      the means weighted (or not) and the standard deviations
      :math:`s` the unweighted ones in both, so the two are on one
      scale; and the ratio of the weighted variances. Weights that work
      bring every :math:`|d|` near 0 (below 0.1 is the usual rule) and
      the variance ratios near 1.

    Parameters
    ----------
    w : array like
        One positive weight per unit.
    group : array like
        Each unit's group (any labels).
    Z : array like or DataFrame, optional
        The covariates the weights were built to balance, one row per
        unit, numeric (code a category as indicator columns). A
        DataFrame's column names label the balance table.
    extreme : float, optional
        A weight above this multiple of its group's mean weight is
        counted as extreme. Default 10.
    smd : float, optional
        The absolute standardised mean difference above which a
        covariate is flagged as still unbalanced after weighting.
        Default 0.1.

    Returns
    -------
    dict
        ``{"groups", "balance", "flags"}``. ``groups`` is a DataFrame
        with a row per group: ``n``, ``sum_w``, ``ess`` (the effective
        sample size), ``ess_ratio`` (``ess / n``), ``max_w``,
        ``max_share`` and ``n_extreme``. ``balance`` is a DataFrame with
        a row per covariate and pair of groups (``covariate``,
        ``group_a``, ``group_b``, ``smd_unweighted``, ``smd_weighted``,
        ``variance_ratio``), or ``None`` without ``Z``. ``flags`` is a
        list of plain sentences, one for each thing worth a look (an
        effective sample size below half the group, extreme weights, a
        covariate still unbalanced); empty when there is none.

    Examples
    --------
    Six units in two groups; the weights make the covariate's mean 1.5
    in both:

    >>> import surpyval as sp
    >>> z = [1, 1, 2, 1, 2, 2]
    >>> group = ["a", "a", "a", "b", "b", "b"]
    >>> w = [1, 1, 2, 2, 1, 1]
    >>> res = sp.weight_diagnostics(w, group, z)
    >>> res["groups"][["n", "ess", "max_share"]].round(3)
           n    ess  max_share
    group
    a      3  2.667        0.5
    b      3  2.667        0.5
    >>> res["balance"][["smd_unweighted", "smd_weighted"]].round(3)
       smd_unweighted  smd_weighted
    0          -0.577           0.0
    >>> res["flags"]
    []

    References
    ----------
    Austin, P. C. and Stuart, E. A. (2015), "Moving towards best
    practice when using inverse probability of treatment weighting
    (IPTW) using the propensity score to estimate causal treatment
    effects in observational studies", Statistics in Medicine 34,
    3661-3679.

    Cole, S. R. and Hernan, M. A. (2008), "Constructing inverse
    probability weights for marginal structural models", American
    Journal of Epidemiology 168, 656-664.
    """
    import pandas as pd

    from surpyval.utils.weights import check_weights, effective_sample_size

    labels = np.asarray(group)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(
            "'group' must be a one-dimensional array with each unit's "
            "group."
        )
    if pd.isna(labels).any():
        raise ValueError(
            "'group' contains missing values: leave out the units "
            "without a group."
        )
    w_arr = check_weights(w, labels.size)
    for name, value in (("extreme", extreme), ("smd", smd)):
        if not (isinstance(value, numbers.Real) and value > 0):
            raise ValueError(f"'{name}' must be a positive number.")
    levels = _levels_of(labels)

    names: list = []
    Z_arr = None
    if Z is not None:
        if isinstance(Z, pd.DataFrame):
            names = [str(c) for c in Z.columns]
        try:
            Z_arr = np.asarray(Z, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "The covariates 'Z' must be numeric: code a category as "
                "indicator (0/1) columns."
            ) from None
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.ndim != 2 or Z_arr.shape[0] != labels.size:
            raise ValueError(
                "'Z' has {} rows for {} units: give one row of "
                "covariates per unit.".format(Z_arr.shape[0], labels.size)
            )
        if not np.all(np.isfinite(Z_arr)):
            raise ValueError(
                "The covariates 'Z' contain missing or infinite values; "
                "balance cannot be measured on them. Leave those units "
                "out, or fill the values in."
            )
        if not names:
            names = [f"Z[:, {j}]" for j in range(Z_arr.shape[1])]

    flags: list[str] = []
    rows = []
    for level in levels:
        mask = labels == level
        wg = w_arr[mask]
        ess = effective_sample_size(np.ones(wg.size), wg)
        n_extreme = int(np.sum(wg > extreme * wg.mean()))
        rows.append(
            {
                "group": level,
                "n": int(wg.size),
                "sum_w": float(wg.sum()),
                "ess": ess,
                "ess_ratio": ess / wg.size,
                "max_w": float(wg.max()),
                "max_share": float(wg.max() / wg.sum()),
                "n_extreme": n_extreme,
            }
        )
        if ess < 0.5 * wg.size:
            flags.append(
                "Group {!r}: the effective sample size is {:.1f} of {} "
                "units; the weights are uneven enough to cost more than "
                "half its information.".format(level, ess, wg.size)
            )
        if n_extreme:
            flags.append(
                "Group {!r}: {} weight(s) are more than {:g} times the "
                "group's mean weight; the largest carries {:.1%} of the "
                "group's total. Look at those units, and consider "
                "stabilised or truncated weights.".format(
                    level, n_extreme, extreme, wg.max() / wg.sum()
                )
            )
    groups = pd.DataFrame(rows).set_index("group")

    balance = None
    if Z_arr is not None:
        moments = {}
        for level in levels:
            mask = labels == level
            ones = np.ones(int(mask.sum()))
            moments[level] = (
                _weighted_moments(Z_arr[mask], ones),
                _weighted_moments(Z_arr[mask], w_arr[mask]),
                int(mask.sum()),
            )
        table = []
        for j, name in enumerate(names):
            for ia, a in enumerate(levels):
                for b in levels[ia + 1 :]:
                    (ma, va), (mwa, vwa), na = moments[a]
                    (mb, vb), (mwb, vwb), nb = moments[b]
                    # sample variances, divisor n - 1
                    sa = va[j] * na / max(na - 1, 1)
                    sb = vb[j] * nb / max(nb - 1, 1)
                    sd = np.sqrt((sa + sb) / 2.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        row = {
                            "covariate": name,
                            "group_a": a,
                            "group_b": b,
                            "smd_unweighted": float((ma[j] - mb[j]) / sd),
                            "smd_weighted": float((mwa[j] - mwb[j]) / sd),
                            "variance_ratio": float(vwa[j] / vwb[j]),
                        }
                    table.append(row)
                    if abs(row["smd_weighted"]) > smd:
                        flags.append(
                            "Covariate {!r} is still unbalanced between "
                            "{!r} and {!r} after weighting: standardised "
                            "mean difference {:.3f} (above {:g}).".format(
                                name, a, b, row["smd_weighted"], smd
                            )
                        )
        balance = pd.DataFrame(table)
    return {"groups": groups, "balance": balance, "flags": flags}
