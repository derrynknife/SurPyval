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

from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from surpyval.utils import validate_1d as _as_1d
from surpyval.utils.ipcw import censoring_survival, step_at, step_left_limit

__all__ = [
    "survival_probability",
    "brier_score",
    "integrated_brier_score",
    "auc_td",
]


def _outcomes(
    x: npt.ArrayLike, c: npt.ArrayLike, xname: str, cname: str
) -> "tuple[npt.NDArray, npt.NDArray]":
    """Validate observed times and right-censoring flags of equal length."""
    x_arr = _as_1d(x, xname)
    c_arr = _as_1d(c, cname)
    if c_arr.shape != x_arr.shape:
        raise ValueError(
            "'{}' and '{}' must have the same length".format(xname, cname)
        )
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
    x_train: "npt.ArrayLike | None",
    c_train: "npt.ArrayLike | None",
) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray]":
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


def survival_probability(
    model: Any, Z: npt.ArrayLike, times: npt.ArrayLike
) -> npt.NDArray:
    """Predicted survival matrix ``S(t | Z_i)`` from a fitted model.

    Parameters
    ----------
    model : object
        Any fitted model exposing ``sf(x, Z)`` where ``x`` is paired
        element-wise with the rows of ``Z`` (the parametric regression
        families, ``CoxPH``, ``AdditiveHazards``), or returning an
        ``(n_samples, n_times)`` grid (the ``beta.ml`` ``SurvivalTree`` and
        ``RandomSurvivalForest``). Models whose ``sf`` takes a single
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
        # ``sf`` conventions differ across model families: the regression
        # models pair ``x`` with the rows of ``Z`` and return a 1-D vector,
        # while the ``beta.ml`` forest returns an ``(n_samples, n_times)``
        # grid. Because every requested time here equals ``t``, every column
        # of the grid is the same ``S(t | Z_i)`` vector, so take column 0.
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
    x_train: "npt.ArrayLike | None" = None,
    c_train: "npt.ArrayLike | None" = None,
) -> "tuple[npt.NDArray, npt.NDArray]":
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
    survival = np.asarray(survival, dtype=float)
    if survival.ndim == 1:
        survival = survival.reshape(-1, 1)
    if survival.shape != (x.size, times.size):
        raise ValueError(
            "'survival' must have shape (n_samples, n_times) = {}".format(
                (x.size, times.size)
            )
        )
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
    x_train: "npt.ArrayLike | None" = None,
    c_train: "npt.ArrayLike | None" = None,
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
    x_train: "npt.ArrayLike | None" = None,
    c_train: "npt.ArrayLike | None" = None,
) -> "tuple[npt.NDArray, npt.NDArray]":
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
        rk = r[controls]
        num = 0.0
        for ri, wi in zip(rc, wc):
            num += wi * (
                np.count_nonzero(ri > rk) + 0.5 * np.count_nonzero(ri == rk)
            )
        den = wc.sum() * controls.sum()
        if den > 0:
            auc[k] = num / den
    return times, auc
