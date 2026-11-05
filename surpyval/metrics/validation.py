"""Prediction-validation metrics for right-censored survival predictors.

These score a *predicted survival function* against right-censored outcomes,
handling censoring by inverse-probability-of-censoring weighting (IPCW):

* :func:`brier_score` / :func:`integrated_brier_score` -- the time-dependent
  Brier score of Graf et al. (1999): the IPCW-weighted squared error between
  the predicted ``S(t | Z)`` and the survival indicator, and its integral over
  a time grid. The standard scalar summarising calibration and discrimination
  together (lower is better).
* :func:`auc_td` -- Uno's (2007) cumulative/dynamic time-dependent AUC:
  discrimination between subjects who have had the event by ``t`` and those
  still event-free, as a function of the horizon ``t`` (0.5 is chance, 1 is
  perfect).

All three are model-agnostic: they take a matrix of predicted survival
probabilities. :func:`survival_probability` builds that matrix from a fitted
model whose ``sf(x, Z)`` pairs ``x`` with the rows of ``Z`` (the parametric
regression families, ``CoxPH``, ``AdditiveHazards`` and the ``beta.ml``
forest).

The censoring survival ``G`` behind the weights is the reverse Kaplan-Meier
with the *events-first* convention at ties
(:func:`surpyval.utils.ipcw.censoring_survival` with
``ties="events_first"``): an event and a censoring at the same time are
ordered event first, as the data record them, so the event is not at risk of
censoring there. An event at ``x_i`` is weighted by ``1 / G(x_i-)``, the
probability of its being observed, and a subject still event-free at the
horizon ``t`` by ``1 / G(t)``, as in Gerds and Schumacher (2006) and R's
``pec``. Without ties between event and censoring times the results agree
exactly with scikit-survival's ``brier_score``, ``integrated_brier_score``
and ``cumulative_dynamic_auc``. With such ties scikit-survival weights an
event by ``1 / G(x_i)``, which also discounts the censorings at ``x_i`` and
over-weights the event.

References
----------
Graf, E., Schmoor, C., Sauerbrei, W. and Schumacher, M. (1999), "Assessment
and comparison of prognostic classification schemes for survival data",
Statistics in Medicine 18, 2529-2545.

Uno, H., Cai, T., Tian, L. and Wei, L. J. (2007), "Evaluating prediction rules
for t-year survivors with censored regression models", JASA 102, 527-537.

Gerds, T. A. and Schumacher, M. (2006), "Consistent estimation of the expected
Brier score in general survival models with right-censored event times",
Biometrical Journal 48, 1029-1040.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from scipy.stats import norm as _norm

from surpyval.utils import validate_1d as _as_1d
from surpyval.utils.ipcw import (
    censoring_survival,
    ipcw_censoring_influence,
    step_at,
    step_left_limit,
)
from surpyval.utils.validation import alpha_ci_error, check_option

__all__ = [
    "survival_probability",
    "brier_score",
    "integrated_brier_score",
    "auc_td",
    "score_predictions",
    "PredictionScores",
]


def _outcomes(
    x: npt.ArrayLike, c: npt.ArrayLike, xname: str, cname: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """Validate observed times and right-censoring flags of equal length."""
    x_arr = _as_1d(x, xname)
    c_arr = _as_1d(c, cname)
    if c_arr.shape != x_arr.shape:
        raise ValueError(f"'{xname}' and '{cname}' must have the same length")
    if not np.isin(c_arr, (0, 1)).all():
        # A left-censored row (-1) would otherwise be scored as a known
        # survivor past its time; only right censoring is supported.
        raise ValueError(
            "'{}' must be 0 (event) or 1 (right censored); left and "
            "interval censoring are not supported".format(cname)
        )
    return x_arr, c_arr


def _ipcw(
    x: npt.NDArray,
    c: npt.NDArray,
    x_train: npt.ArrayLike | None,
    c_train: npt.ArrayLike | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Events-first censoring survival and the event weights ``1/G(x_i-)``.

    Returns the censoring estimate's times and values (for ``G(t)`` at the
    horizons) and the weight of each evaluation row (only used for the
    events; ``nan`` where ``G(x_i-) = 0``).
    """
    if x_train is None and c_train is None:
        xt, ct = x, c
    elif x_train is not None and c_train is not None:
        xt, ct = _outcomes(x_train, c_train, "x_train", "c_train")
    else:
        raise ValueError("pass both 'x_train' and 'c_train', or neither")
    uniq, g = censoring_survival(xt, ct == 1, ties="events_first")
    # An event at x_i is observed when the censoring time is at least x_i,
    # so its weight is 1/G(x_i-): censorings tied with the event do not count
    # against it. G(x_i-) is positive for an event of the training data; it
    # is 0 only for an evaluation row past the training censoring support,
    # where the weight is not identified: nan, which the callers turn into
    # a nan score rather than silently dropping the row.
    g_xi = step_left_limit(uniq, g, x, before=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(g_xi > 0, 1.0 / g_xi, np.nan)
    return uniq, g, w


def _survival_matrix(
    survival: npt.ArrayLike, n: int, n_times: int, name: str
) -> npt.NDArray:
    """Validate a matrix of predicted survival probabilities."""
    out = np.asarray(survival, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.shape != (n, n_times):
        raise ValueError(
            "'{}' must have shape (n_samples, n_times) = {}".format(
                name, (n, n_times)
            )
        )
    with np.errstate(invalid="ignore"):
        outside = (out < 0) | (out > 1)
    if outside.any():
        raise ValueError(
            "'{}' must hold survival probabilities, in [0, 1]: {} of "
            "its {} values are outside (from {:.4g} to {:.4g}). Pass "
            "S(t | Z), e.g. from survival_probability, not a hazard or a "
            "cumulative hazard".format(
                name,
                int(outside.sum()),
                out.size,
                float(np.nanmin(out)),
                float(np.nanmax(out)),
            )
        )
    return out


def survival_probability(
    model: Any, Z: npt.ArrayLike, times: npt.ArrayLike
) -> npt.NDArray:
    """Predicted survival matrix ``S(t | Z_i)`` from a fitted model.

    Parameters
    ----------
    model : object
        Any fitted model exposing ``sf(x, Z)`` where ``x`` is paired
        element-wise with the rows of ``Z`` (the parametric regression
        families, ``CoxPH``, ``AdditiveHazards``, the ``beta.ml``
        ``SurvivalTree`` and ``RandomSurvivalForest``), or returning an
        ``(n_samples, n_times)`` grid. Models whose ``sf`` takes a single
        covariate vector (``BuckleyJames``) are not supported: build their
        matrix row by row with ``model.sf(times, Z[i])``.
    Z : array_like or pandas.DataFrame
        Covariate matrix, one row per subject. A DataFrame is passed to
        ``model.sf`` as it is, so a model fitted with ``fit_from_df``
        (``Z_cols`` or a ``formula``, string levels included) reads it by
        column name; an array is taken as numbers, one column per
        covariate.
    times : array_like
        Evaluation times.

    Returns
    -------
    survival : ndarray, shape ``(n_samples, n_times)``
        ``survival[i, k]`` is the predicted survival of subject ``i`` at
        ``times[k]``. A subject with a missing covariate, or a missing
        time, gets ``nan`` where the model's ``sf`` gives it.

    Examples
    --------
    A formula fit with a string-valued factor is scored from a DataFrame:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from surpyval import WeibullPH
    >>> from surpyval.metrics import survival_probability
    >>> rng = np.random.default_rng(1)
    >>> g = rng.choice(["a", "b"], 60)
    >>> x = rng.weibull(1.5, 60) * np.where(g == "b", 5.0, 10.0)
    >>> df = pd.DataFrame({"x": x, "g": g})
    >>> model = WeibullPH.fit_from_df(df, x_col="x", formula="g")
    >>> new = pd.DataFrame({"g": ["a", "b"]})
    >>> survival_probability(model, new, [2.0, 5.0]).shape
    (2, 2)
    """
    if isinstance(Z, DataFrame):
        # A DataFrame is the model's to read: formula fits look their
        # columns up by name and code string levels themselves, which a
        # cast to float made impossible.
        Z_in: Any = Z
        n = len(Z)
    else:
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        Z_in = Z_arr
        n = Z_arr.shape[0]
    times = _as_1d(times, "times")
    cols = []
    for t in times:
        out = np.asarray(model.sf(np.full(n, float(t)), Z_in), dtype=float)
        # The regression models, survival trees and forests pair ``x`` with
        # the rows of ``Z`` and return a 1-D vector; a model returning an
        # ``(n_samples, n_times)`` grid instead has, since every requested
        # time here equals ``t``, the same ``S(t | Z_i)`` vector in every
        # column, so take column 0.
        col = out[:, 0] if out.ndim == 2 else out.ravel()
        if col.shape[0] != n:
            raise ValueError(
                "model.sf returned {} values for {} subjects; its sf(x, Z) "
                "convention is not supported".format(col.shape[0], n)
            )
        cols.append(col)
    return np.column_stack(cols)


def brier_score(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    survival: npt.ArrayLike,
    times: npt.ArrayLike,
    x_train: npt.ArrayLike | None = None,
    c_train: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Time-dependent Brier score (Graf et al. 1999).

    At each horizon ``t`` the Brier score is the IPCW-weighted mean squared
    error between the survival indicator ``I(T_i > t)`` and the predicted
    survival ``S(t | Z_i)``:

    .. math::
        BS(t) = \frac1n \sum_i \Big[
            \frac{S(t\mid Z_i)^2\, I(x_i \le t,\ \delta_i=1)}{\hat G(x_i-)}
          + \frac{(1-S(t\mid Z_i))^2\, I(x_i > t)}{\hat G(t)} \Big],

    where :math:`\hat G` is the Kaplan-Meier estimate of the censoring
    survival with the events-first convention at ties (see the module
    notes). Subjects censored before ``t`` contribute nothing (their status
    at ``t`` is unknown), nor do those censored at ``t``; the IPCW weights
    correct for that loss. Lower is better.

    Parameters
    ----------
    x, c : array_like
        Observed times and censoring flags (``0`` event, ``1`` right censored)
        of the evaluation set.
    survival : array_like, shape ``(n_samples, n_times)``
        Predicted survival ``S(times[k] | Z_i)``; see
        :func:`survival_probability`.
    times : array_like
        Horizons at which to score, matching the columns of ``survival``.
    x_train, c_train : array_like, optional
        Data used to estimate the censoring distribution ``G``. Defaults to the
        evaluation ``x`` / ``c``. ``G`` is held at its last value beyond the
        largest training time. If it has reached 0 there, a horizon at which
        an evaluation row needs that zero (an event past the training
        censoring support, or a survivor at such a horizon) scores ``nan``.

    Returns
    -------
    times, bs : ndarray
        The horizons and the Brier score at each.

    Examples
    --------
    An event and a censoring tie at ``t = 3``. The censoring survival is
    ``G = 1`` before 3 and ``1 - 1/2`` from 3 (at risk of censoring at 3:
    the one censored there and the one still under observation). At the
    horizon 3.5 the three events are weighted by ``1/G(x_i-) = 1`` and the
    survivor by ``1/G(3.5) = 2``:

    >>> from surpyval.metrics import brier_score
    >>> x = [1.0, 2.0, 3.0, 3.0, 4.0]
    >>> c = [0, 0, 0, 1, 0]
    >>> S = [[0.8], [0.6], [0.5], [0.5], [0.3]]
    >>> brier_score(x, c, S, [3.5])[1]  # (.64 + .36 + .25 + 2 * .49) / 5
    array([0.446])
    """
    x, c = _outcomes(x, c, "x", "c")
    times = _as_1d(times, "times")
    survival = _survival_matrix(survival, x.size, times.size, "survival")
    uniq, g, w_case = _ipcw(x, c, x_train, c_train)

    n = x.size
    bs = np.empty(times.size)
    for k, t in enumerate(times):
        s = survival[:, k]
        g_t = float(step_at(uniq, g, np.array([t]), before=1.0)[0])
        died = (x <= t) & (c == 0)
        alive = x > t
        if (alive.any() and g_t == 0) or np.isnan(w_case[died]).any():
            # A row that counts needs G where the training censoring
            # estimate has reached 0: the score is not identified (dropping
            # the row, as a zero weight did, biased it towards 0).
            bs[k] = np.nan
            continue
        term = np.zeros(n)
        term[died] = s[died] ** 2 * w_case[died]
        term[alive] = (1.0 - s[alive]) ** 2 / g_t
        bs[k] = term.sum() / n
    return times, bs


def integrated_brier_score(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    survival: npt.ArrayLike,
    times: npt.ArrayLike,
    x_train: npt.ArrayLike | None = None,
    c_train: npt.ArrayLike | None = None,
) -> float:
    """Integrated Brier score: the Brier score averaged over ``times``.

    The trapezoidal integral of :func:`brier_score` over the time grid divided
    by its span, as scikit-survival's ``integrated_brier_score``. A single
    number summarising a survival predictor's accuracy (lower is better); a
    model that predicts the true ``S(t | Z)`` scores below the marginal
    Kaplan-Meier reference. The grid need not be sorted (the columns of
    ``survival`` follow ``times``); a single time, or a grid with no span,
    returns the mean Brier score. Parameters as for :func:`brier_score`.

    Examples
    --------
    A Cox model of the Rossi recidivism data scored over the first 39
    weeks, against the Kaplan-Meier curve that ignores the covariates:

    >>> import numpy as np
    >>> from surpyval import CoxPH, KaplanMeier
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.metrics import (
    ...     integrated_brier_score,
    ...     survival_probability,
    ... )
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> times = [13, 26, 39]
    >>> cox = CoxPH.fit(x, Z, c=c)
    >>> S_cox = survival_probability(cox, Z, times)
    >>> round(integrated_brier_score(x, c, S_cox, times), 4)
    0.0998
    >>> S_km = np.tile(KaplanMeier.fit(x, c).sf(times), (len(x), 1))
    >>> round(integrated_brier_score(x, c, S_km, times), 4)
    0.1038
    """
    times_arr, bs = brier_score(x, c, survival, times, x_train, c_train)
    # Integrate over the grid in time order: an unsorted grid otherwise gave
    # negative panel widths and a span of the wrong sign.
    order = np.argsort(times_arr, kind="stable")
    times_arr, bs = times_arr[order], bs[order]
    if times_arr.size < 2:
        return float(bs.mean())
    span = times_arr[-1] - times_arr[0]
    if span <= 0:
        return float(bs.mean())
    # Trapezoidal integral (np.trapz was removed in NumPy 2.0).
    area = np.sum(np.diff(times_arr) * (bs[:-1] + bs[1:]) / 2.0)
    return float(area / span)


def auc_td(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk: npt.ArrayLike,
    times: npt.ArrayLike,
    x_train: npt.ArrayLike | None = None,
    c_train: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Uno's cumulative/dynamic time-dependent AUC (2007).

    At each horizon ``t`` a *case* is a subject with the event by ``t``
    (``x_i \le t``, ``\delta_i = 1``) and a *control* is a subject still
    event-free (``x_j > t``). The AUC estimates the probability that a case is
    assigned a higher risk than a control, with cases IPCW-weighted by
    ``1 / \hat G(x_i)`` to correct for censoring:

    .. math::
        \widehat{AUC}(t) = \frac{\sum_{i,j} w_i\,
            \big(I(r_i > r_j) + \tfrac12 I(r_i = r_j)\big)\,
            I(\text{case}_i)\, I(\text{control}_j)}
            {\big(\sum_i w_i I(\text{case}_i)\big)\,
             \big(\sum_j I(\text{control}_j)\big)}.

    Parameters
    ----------
    x, c : array_like
        Observed times and censoring flags (``0`` event, ``1`` right censored).
    risk : array_like, shape ``(n_samples, n_times)``
        Risk scores where *higher means earlier event*. For a survival
        predictor use ``1 - survival`` (see :func:`survival_probability`).
        A single column (or 1-D array) is broadcast across all ``times``.
    times : array_like
        Horizons at which to evaluate the AUC.
    x_train, c_train : array_like, optional
        Data used to estimate the censoring distribution ``G``. Defaults to the
        evaluation ``x`` / ``c``.

    Returns
    -------
    times, auc : ndarray
        The horizons and the AUC at each. A horizon with no cases or no
        controls, or with a case whose weight is not identified (``G = 0``,
        see :func:`brier_score`), yields ``nan``.

    Examples
    --------
    At ``t = 1.5`` the one case (risk 0.9) outranks all four controls. At
    ``t = 2.5`` the second case (risk 0.4) outranks only one of the three
    controls (risks 0.7, 0.5, 0.2), so the AUC is ``(3 + 1) / 6``:

    >>> from surpyval.metrics import auc_td
    >>> x = [1.0, 2.0, 3.0, 3.0, 4.0]
    >>> c = [0, 0, 0, 1, 0]
    >>> risk = [0.9, 0.4, 0.7, 0.5, 0.2]
    >>> auc_td(x, c, risk, [1.5, 2.5])[1].round(4)
    array([1.    , 0.6667])
    """
    x, c = _outcomes(x, c, "x", "c")
    times = _as_1d(times, "times")
    risk = np.asarray(risk, dtype=float)
    if risk.ndim == 1:
        risk = risk.reshape(-1, 1)
    if risk.shape[0] != x.size:
        raise ValueError("'risk' must have one row per observation")
    if risk.shape[1] == 1 and times.size > 1:
        risk = np.repeat(risk, times.size, axis=1)
    if risk.shape[1] != times.size:
        raise ValueError(
            "'risk' must have one column per time (or a single column)"
        )
    _, _, w = _ipcw(x, c, x_train, c_train)

    auc = np.full(times.size, np.nan)
    for k, t in enumerate(times):
        r = risk[:, k]
        cases = (x <= t) & (c == 0)
        controls = x > t
        if not cases.any() or not controls.any():
            continue
        if np.isnan(w[cases]).any():
            # A case past the training censoring support: not identified.
            continue
        rc = r[cases]
        wc = w[cases]
        # Each case's controls below it and tied with it, from the sorted
        # controls (a comparison with every control was O(cases x
        # controls): 15 s at 1e5 rows). NaN sorts last, so it is counted
        # as neither, as a comparison counts it; a NaN case counts none.
        rk = np.sort(r[controls])
        below = np.searchsorted(rk, rc, side="left")
        tied = np.searchsorted(rk, rc, side="right") - below
        terms = wc * (below + 0.5 * tied)
        terms[np.isnan(rc)] = 0.0
        # Summed in order, as the loop did
        num = float(np.cumsum(terms)[-1])
        den = wc.sum() * controls.sum()
        if den > 0:
            auc[k] = num / den
    return times, auc


# ---------------------------------------------------------------------------
# Standard errors, intervals and model comparisons.
# ---------------------------------------------------------------------------

_METRICS = ("brier", "auc", "ibs")


class PredictionScores:
    """
    Prediction metrics with standard errors, confidence intervals and
    tests between models, from :func:`score_predictions`.

    Attributes
    ----------
    times : ndarray
        The horizons, as passed.
    models : list of str
        The names of the scored models, in the order given.
    metrics : tuple of str
        The metrics computed: ``"brier"``, ``"auc"`` and/or ``"ibs"``.
    alpha_ci : float
        The intervals are two-sided ``1 - alpha_ci`` intervals.
    n : int
        The number of subjects scored.
    scores : pandas.DataFrame
        One row per metric, model and horizon, with columns ``metric``,
        ``model``, ``time`` (``nan`` for ``"ibs"``, which summarises the
        horizons), ``estimate``, ``se``, ``lower`` and ``upper``. The
        intervals are ``estimate +/- z se``, clipped to ``[0, 1]`` as
        ``riskRegression::Score`` does.
    comparisons : pandas.DataFrame
        One row per metric, pair of models and horizon: ``model`` and
        ``reference`` (every model against each one listed before it),
        ``difference`` (``model`` minus ``reference``), ``se``, ``lower``,
        ``upper`` and ``p_value`` (the two-sided Wald test of no
        difference). Empty with one model.
    influence : dict
        The influence function of each estimate, keyed by
        ``(metric, model)``: an ``(n, n_times)`` array for ``"brier"`` and
        ``"auc"``, a length-``n`` array for ``"ibs"``. Its standard
        deviation over ``sqrt(n)`` is the standard error; the covariance of
        two columns over ``n`` is the estimates' covariance. Not written by
        :meth:`to_dict`.

    Examples
    --------
    :func:`score_predictions` returns one; :meth:`estimate` reads a
    metric's values and :meth:`to_dict` writes strict JSON:

    >>> from surpyval.metrics import score_predictions
    >>> x = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    >>> c = [0, 0, 0, 1, 0, 1]
    >>> S = [[0.8], [0.6], [0.5], [0.5], [0.3], [0.4]]
    >>> result = score_predictions(x, c, S, [3.5], metrics=("brier",))
    >>> result.estimate("brier").round(4)
    array([0.4208])
    >>> sorted(result.to_dict())[:3]
    ['alpha_ci', 'comparisons', 'metrics']
    """

    def __init__(
        self,
        times: npt.NDArray,
        models: list[str],
        metrics: tuple[str, ...],
        alpha_ci: float,
        n: int,
        scores: DataFrame,
        comparisons: DataFrame,
        influence: dict | None = None,
    ) -> None:
        self.times = times
        self.models = models
        self.metrics = metrics
        self.alpha_ci = alpha_ci
        self.n = n
        self.scores = scores
        self.comparisons = comparisons
        self.influence = {} if influence is None else influence

    def _rows(self, metric: str, model: str | None) -> DataFrame:
        check_option("metric", metric, self.metrics)
        if model is None:
            model = self.models[0]
        if model not in self.models:
            raise ValueError(
                "'model' must be one of {}; got {!r}".format(
                    self.models, model
                )
            )
        table = self.scores
        return table[(table["metric"] == metric) & (table["model"] == model)]

    def estimate(self, metric: str, model: str | None = None) -> Any:
        """The estimates of ``metric`` for ``model`` (default the first):
        an array over the horizons, or a float for ``"ibs"``.

        Parameters
        ----------
        metric : str
            ``"brier"``, ``"auc"`` or ``"ibs"``.
        model : str, optional
            The model's name; the first model by default.

        Returns
        -------
        estimate : ndarray or float
            The values the plain functions return.
        """
        values = self._rows(metric, model)["estimate"].to_numpy()
        return float(values[0]) if metric == "ibs" else values

    def se(self, metric: str, model: str | None = None) -> Any:
        """The standard errors of ``metric`` for ``model``; see
        :meth:`estimate`.

        Parameters
        ----------
        metric : str
            ``"brier"``, ``"auc"`` or ``"ibs"``.
        model : str, optional
            The model's name; the first model by default.

        Returns
        -------
        se : ndarray or float
            The standard errors.
        """
        values = self._rows(metric, model)["se"].to_numpy()
        return float(values[0]) if metric == "ibs" else values

    def to_dict(self) -> dict:
        """The result as a dict of native Python values, strict JSON: a
        missing number is written as ``None`` and recorded under
        ``"non_finite"``. The influence functions are not written.

        Returns
        -------
        result : dict
            Settings, ``"scores"`` and ``"comparisons"`` as lists of
            records.
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out = {
            "result": type(self).__name__,
            "times": to_native(np.asarray(self.times, dtype=float)),
            "models": list(self.models),
            "metrics": list(self.metrics),
            "alpha_ci": float(self.alpha_ci),
            "n": int(self.n),
            "scores": [
                {k: to_native(v) for k, v in row.items()}
                for row in self.scores.to_dict("records")
            ],
            "comparisons": [
                {k: to_native(v) for k, v in row.items()}
                for row in self.comparisons.to_dict("records")
            ],
        }
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> PredictionScores:
        """The result :meth:`to_dict` wrote (without influence functions).

        Parameters
        ----------
        result_dict : dict
            A dict written by :meth:`to_dict`.

        Returns
        -------
        result : PredictionScores
            The scores and comparisons restored.
        """
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create a PredictionScores from a dict its to_dict "
                "wrote (with 'result': 'PredictionScores')"
            )
        d = decode_non_finite(dict(result_dict))
        return cls(
            times=np.asarray(d["times"], dtype=float),
            models=list(d["models"]),
            metrics=tuple(d["metrics"]),
            alpha_ci=float(d["alpha_ci"]),
            n=int(d["n"]),
            scores=DataFrame(d["scores"], columns=_SCORE_COLUMNS),
            comparisons=DataFrame(
                d["comparisons"], columns=_COMPARISON_COLUMNS
            ),
        )

    def __repr__(self) -> str:
        level = 100 * (1 - self.alpha_ci)
        out = "Prediction scores ({} subjects, {:g}% intervals)\n".format(
            self.n, level
        )
        out += self.scores.to_string(index=False)
        if len(self.comparisons):
            out += "\n\nComparisons (model minus reference)\n"
            out += self.comparisons.to_string(index=False)
        return out


_SCORE_COLUMNS = [
    "metric",
    "model",
    "time",
    "estimate",
    "se",
    "lower",
    "upper",
]
_COMPARISON_COLUMNS = [
    "metric",
    "model",
    "reference",
    "time",
    "difference",
    "se",
    "lower",
    "upper",
    "p_value",
]


def _brier_influence(
    x: npt.NDArray,
    c: npt.NDArray,
    s: npt.NDArray,
    t: float,
    w_case: npt.NDArray,
    g_t: float,
) -> tuple[float, npt.NDArray]:
    """The Brier score at ``t`` and its influence function.

    With ``r_i`` each subject's IPCW residual (``S^2 / G(x_i-)`` for an
    event by ``t``, ``(1 - S)^2 / G(t)`` for a survivor, 0 otherwise), the
    score is ``mean(r)`` and its influence ``r_k - mean(r)`` plus the
    influence through the censoring estimate, ``sum_j r_j f_k(u_j) / n``
    (:func:`surpyval.utils.ipcw.ipcw_censoring_influence`).
    """
    n = x.size
    died = (x <= t) & (c == 0)
    alive = x > t
    if (alive.any() and g_t == 0) or np.isnan(w_case[died]).any():
        return np.nan, np.full(n, np.nan)
    r = np.zeros(n)
    r[died] = s[died] ** 2 * w_case[died]
    r[alive] = (1.0 - s[alive]) ** 2 / g_t
    bs = r.sum() / n
    influence = r - bs
    influence += ipcw_censoring_influence(
        x, c == 1, r / n, np.where(died, x, t), died
    )
    return float(bs), influence


def _weighted_below(
    values: npt.NDArray, ref: npt.NDArray, ref_w: npt.NDArray
) -> npt.NDArray:
    """For each value, the weight of ``ref`` strictly below it plus half
    the weight tied with it."""
    order = np.argsort(ref, kind="stable")
    r_sorted = ref[order]
    cw = np.concatenate(([0.0], np.cumsum(ref_w[order])))
    lo = np.searchsorted(r_sorted, values, side="left")
    hi = np.searchsorted(r_sorted, values, side="right")
    return cw[lo] + 0.5 * (cw[hi] - cw[lo])


def _auc_influence(
    x: npt.NDArray,
    c: npt.NDArray,
    risk: npt.NDArray,
    t: float,
    w_case: npt.NDArray,
    g_t: float,
) -> tuple[float, npt.NDArray]:
    """Uno's AUC at ``t`` and its influence function (Blanche et al.
    2013): the AUC is ``nu / mu``, ``nu`` the weighted mean over
    (case, control) pairs of ``I(r_i > r_j) + I(r_i = r_j) / 2`` and
    ``mu`` the mean pair weight, each a two-sample U-statistic; the
    censoring estimate enters through the weights, a case's
    ``1 / G(x_i-)`` and a control's ``1 / G(t)``."""
    n = x.size
    cases = (x <= t) & (c == 0)
    controls = x > t
    nan = (np.nan, np.full(n, np.nan))
    if not cases.any() or not controls.any() or g_t == 0:
        return nan
    if np.isnan(w_case[cases]).any() or np.isnan(risk[cases | controls]).any():
        return nan
    w = np.zeros(n)
    w[cases] = w_case[cases]
    w[controls] = 1.0 / g_t
    # a_k: subject k's share of the weighted concordant pairs, b_k its
    # share of all weighted pairs.
    a = np.zeros(n)
    b = np.zeros(n)
    a[cases] = w[cases] * _weighted_below(
        risk[cases], risk[controls], w[controls]
    )
    a[controls] = w[controls] * (
        w[cases].sum() - _weighted_below(risk[controls], risk[cases], w[cases])
    )
    b[cases] = w[cases] * w[controls].sum()
    b[controls] = w[controls] * w[cases].sum()
    pairs = w[cases].sum() * w[controls].sum()
    auc = a[cases].sum() / pairs
    mu = pairs / n**2
    d = a - auc * b
    influence = d / n / mu
    influence += ipcw_censoring_influence(
        x, c == 1, d / n**2 / mu, np.where(cases, x, t), cases
    )
    return float(auc), influence


def _ibs_weights(times: npt.NDArray) -> npt.NDArray:
    """The weights of the Brier scores at ``times`` in the integrated
    Brier score: the trapezoid rule in time order over the span, or the
    plain mean without a span (as :func:`integrated_brier_score`)."""
    k = times.size
    order = np.argsort(times, kind="stable")
    ts = times[order]
    if k < 2 or ts[-1] - ts[0] <= 0:
        return np.full(k, 1.0 / k)
    h = np.diff(ts)
    a_sorted = np.zeros(k)
    a_sorted[:-1] += h / 2
    a_sorted[1:] += h / 2
    a = np.empty(k)
    a[order] = a_sorted / (ts[-1] - ts[0])
    return a


def score_predictions(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    predictions: Any,
    times: npt.ArrayLike,
    metrics: Any = ("brier", "auc", "ibs"),
    alpha_ci: float = 0.05,
) -> PredictionScores:
    r"""Brier score, integrated Brier score and time-dependent AUC with
    standard errors, confidence intervals and tests between models.

    :func:`brier_score`, :func:`integrated_brier_score` and :func:`auc_td`
    give the bare estimates. This gives the same estimates with how
    uncertain they are, and, for two or more models scored on the same
    subjects, whether their difference is more than chance: the
    difference, its interval and a p-value. The comparison is paired (the
    same subjects score both models), so it is much sharper than
    comparing two separate intervals.

    Each estimate is an average over the subjects of an IPCW term, so its
    standard error comes from its *influence function*: how much each
    subject moves the estimate, including through the Kaplan-Meier
    estimate of the censoring distribution behind the weights (the
    ``conservative = FALSE`` variance of ``riskRegression::Score``). The
    standard error is ``sd(IF) / sqrt(n)``; the interval of an estimate is
    ``estimate +/- z se`` clipped to ``[0, 1]``; the difference between
    two models has the standard error ``sd(IF_a - IF_b) / sqrt(n)`` and
    the Wald test ``2 Phi(-|difference| / se)``. For the AUC the influence
    function is that of Blanche, Dartigues and Jacqmin-Gadda (2013), as
    ``timeROC(iid = TRUE)``; for the Brier score that of Gerds and
    Schumacher (2006), as ``Score(se.fit = TRUE)``. The integrated Brier
    score's is the same trapezoid combination of the Brier scores' that
    gives the estimate.

    Parameters
    ----------
    x : array_like
        Observed times of the evaluation set.
    c : array_like
        Censoring flags, ``0`` event, ``1`` right censored.
    predictions : array_like or dict
        Predicted survival ``S(times[k] | Z_i)``, shape
        ``(n_samples, n_times)``, of one model (named ``"model"``), or a
        dict of such matrices keyed by model name. Each model is
        compared with every model before it. The AUC ranks subjects by
        ``1 - S``.
    times : array_like
        Horizons at which to score, matching the columns of each matrix.
    metrics : sequence of str, optional
        Any of ``"brier"``, ``"auc"`` and ``"ibs"`` (all by default).
    alpha_ci : float, optional
        The intervals are two-sided ``1 - alpha_ci`` intervals; default
        0.05.

    Returns
    -------
    PredictionScores
        The estimates, standard errors and intervals (``scores``), the
        comparisons between models (``comparisons``) and the influence
        functions.

    Notes
    -----
    The censoring distribution is estimated from the evaluation data
    themselves (there is no ``x_train``): the standard errors account for
    that estimate, which a censoring distribution from other data would
    need another term for. The estimates equal those of
    :func:`brier_score`, :func:`integrated_brier_score` and
    :func:`auc_td` without ``x_train``. A horizon where an estimate is
    ``nan`` (no cases or controls, or a weight that is not identified)
    has ``nan`` standard error and interval. An AUC with a missing
    prediction among the cases or controls is ``nan`` here, estimate
    included (:func:`auc_td` counts such a pair as neither concordant nor
    tied); drop those subjects to score the rest. A Brier score with a
    missing prediction is ``nan``, as in :func:`brier_score`.

    The influence function of the censoring Kaplan-Meier uses
    ``riskRegression``'s at-risk counts (everyone with ``x >= s``); the
    standard errors agree with ``Score`` and, on data without tied
    event times, with ``timeROC``. With tied event times ``timeROC``'s
    AUC differs from ``Score``'s and SurPyval's.

    References
    ----------
    Blanche, P., Dartigues, J.-F. and Jacqmin-Gadda, H. (2013), "Estimating
    and comparing time-dependent areas under receiver operating
    characteristic curves for censored event times with competing risks",
    Statistics in Medicine 32, 5381-5397.

    Examples
    --------
    Two models of the Rossi recidivism data, the Cox model and the
    Kaplan-Meier curve that ignores the covariates, scored where they
    were fitted. The Cox model's Brier scores are only a little lower,
    and the paired test cannot tell the two apart at 26 weeks; its
    discrimination at 39 weeks is clearly better than the curve's 0.5:

    >>> import numpy as np
    >>> from surpyval import CoxPH, KaplanMeier
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.metrics import score_predictions, survival_probability
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> times = [13, 26, 39]
    >>> S_cox = survival_probability(CoxPH.fit(x, Z, c=c), Z, times)
    >>> S_km = np.tile(KaplanMeier.fit(x, c).sf(times), (len(x), 1))
    >>> result = score_predictions(x, c, {"km": S_km, "cox": S_cox}, times)
    >>> result.estimate("brier", "cox").round(4)
    array([0.0433, 0.1056, 0.1447])
    >>> result.se("brier", "cox").round(4)
    array([0.009 , 0.0116, 0.0114])
    >>> table = result.comparisons.set_index(["metric", "time"])
    >>> round(float(table.loc[("brier", 26.0), "p_value"]), 3)
    0.189
    >>> round(float(table.loc[("auc", 39.0), "difference"]), 3)
    0.128
    >>> round(float(table.loc[("auc", 39.0), "p_value"]), 4)
    0.0003
    """
    x, c = _outcomes(x, c, "x", "c")
    times = _as_1d(times, "times")
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError(
            "'metrics' must name at least one of 'brier', 'auc' or 'ibs'"
        )
    for m in metrics:
        check_option("metrics", m, _METRICS)
    if not (
        isinstance(alpha_ci, (int, float, np.integer, np.floating))
        and 0 < alpha_ci < 1
    ):
        raise alpha_ci_error(alpha_ci)
    if isinstance(predictions, dict):
        if not predictions:
            raise ValueError(
                "'predictions' is an empty dict; pass one survival matrix "
                "or a dict of them keyed by model name"
            )
        named = {str(k): v for k, v in predictions.items()}
        if len(named) != len(predictions):
            raise ValueError("'predictions' has two models with one name")
    else:
        named = {"model": predictions}
    surv = {
        k: _survival_matrix(
            v,
            x.size,
            times.size,
            ("predictions" if len(named) == 1 else f"predictions[{k!r}]"),
        )
        for k, v in named.items()
    }

    uniq, g, w_case = _ipcw(x, c, None, None)
    g_times = step_at(uniq, g, times, before=1.0)
    n = x.size
    z = _norm.ppf(1 - alpha_ci / 2)

    estimates: dict = {}
    influence: dict = {}
    need_brier = "brier" in metrics or "ibs" in metrics
    for name, S in surv.items():
        if need_brier:
            est = np.empty(times.size)
            inf = np.empty((n, times.size))
            for k, t in enumerate(times):
                est[k], inf[:, k] = _brier_influence(
                    x, c, S[:, k], t, w_case, float(g_times[k])
                )
            if "brier" in metrics:
                estimates["brier", name] = est
                influence["brier", name] = inf
            if "ibs" in metrics:
                a = _ibs_weights(times)
                estimates["ibs", name] = np.array([est @ a])
                influence["ibs", name] = inf @ a
        if "auc" in metrics:
            est = np.empty(times.size)
            inf = np.empty((n, times.size))
            for k, t in enumerate(times):
                est[k], inf[:, k] = _auc_influence(
                    x, c, 1.0 - S[:, k], t, w_case, float(g_times[k])
                )
            estimates["auc", name] = est
            influence["auc", name] = inf

    def _se(f: npt.NDArray) -> npt.NDArray:
        f = f.reshape(n, -1)
        if n < 2:
            return np.full(f.shape[1], np.nan)
        return np.std(f, axis=0, ddof=1) / np.sqrt(n)

    score_rows = []
    comparison_rows = []
    names = list(surv)
    for metric in metrics:
        row_times = np.array([np.nan]) if metric == "ibs" else times
        for name in names:
            est = estimates[metric, name]
            se = _se(influence[metric, name])
            se = np.where(np.isnan(est), np.nan, se)
            lower = np.clip(est - z * se, 0.0, 1.0)
            upper = np.clip(est + z * se, 0.0, 1.0)
            for k in range(est.size):
                score_rows.append(
                    [
                        metric,
                        name,
                        row_times[k],
                        est[k],
                        se[k],
                        lower[k],
                        upper[k],
                    ]
                )
        for i, ref in enumerate(names):
            for name in names[i + 1 :]:
                diff = estimates[metric, name] - estimates[metric, ref]
                se = _se(influence[metric, name] - influence[metric, ref])
                se = np.where(np.isnan(diff), np.nan, se)
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = 2 * _norm.sf(np.abs(diff) / se)
                for k in range(diff.size):
                    comparison_rows.append(
                        [
                            metric,
                            name,
                            ref,
                            row_times[k],
                            diff[k],
                            se[k],
                            diff[k] - z * se[k],
                            diff[k] + z * se[k],
                            p[k],
                        ]
                    )
    return PredictionScores(
        times=times,
        models=names,
        metrics=metrics,
        alpha_ci=float(alpha_ci),
        n=n,
        scores=DataFrame(score_rows, columns=_SCORE_COLUMNS),
        comparisons=DataFrame(comparison_rows, columns=_COMPARISON_COLUMNS),
        influence=influence,
    )
