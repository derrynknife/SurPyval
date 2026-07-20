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
probabilities. :func:`survival_probability` builds that matrix from any fitted
model exposing ``sf(x, Z)`` (the parametric regression families, ``CoxPH`` and
the ``beta.ml`` forest).

References
----------
Graf, E., Schmoor, C., Sauerbrei, W. and Schumacher, M. (1999), "Assessment
and comparison of prognostic classification schemes for survival data",
Statistics in Medicine 18, 2529-2545.

Uno, H., Cai, T., Tian, L. and Wei, L. J. (2007), "Evaluating prediction rules
for t-year survivors with censored regression models", JASA 102, 527-537.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "survival_probability",
    "brier_score",
    "integrated_brier_score",
    "auc_td",
]


def _as_1d(a: npt.ArrayLike, name: str) -> npt.NDArray:
    arr = np.atleast_1d(np.asarray(a, dtype=float))
    if arr.ndim != 1:
        raise ValueError("'{}' must be one-dimensional".format(name))
    return arr


def _censoring_km(x_train: npt.NDArray, c_train: npt.NDArray):
    """Kaplan-Meier estimate of the *censoring* survival ``G(t) = P(C > t)``.

    Censoring is treated as the event of interest: an originally right-censored
    observation (``c == 1``) is a censoring "event", an originally observed
    event (``c == 0``) is "censored" for ``G``. Returns the sorted unique times
    and the right-continuous step values of ``G`` on them.
    """
    order = np.argsort(x_train)
    xs = x_train[order]
    cs = c_train[order]
    uniq = np.unique(xs)
    g = np.empty(uniq.size)
    surv = 1.0
    for i, t in enumerate(uniq):
        n_cens = np.count_nonzero((xs == t) & (cs == 1))
        n_risk = np.count_nonzero(xs >= t)
        if n_risk > 0:
            surv *= 1.0 - n_cens / n_risk
        g[i] = surv
    return uniq, g


def _g_at(
    uniq: npt.NDArray, g: npt.NDArray, query: npt.NDArray
) -> npt.NDArray:
    """Right-continuous step evaluation of ``G`` at ``query`` (``G(0)=1``)."""
    idx = np.searchsorted(uniq, query, side="right") - 1
    return np.where(idx >= 0, g[np.clip(idx, 0, g.size - 1)], 1.0)


def survival_probability(
    model: Any, Z: npt.ArrayLike, times: npt.ArrayLike
) -> npt.NDArray:
    """Predicted survival matrix ``S(t | Z_i)`` from a fitted model.

    Parameters
    ----------
    model : object
        Any fitted model exposing ``sf(x, Z)`` where ``x`` is paired
        element-wise with the rows of ``Z`` (the parametric regression
        families, ``CoxPH``, the ``beta.ml`` forest).
    Z : array_like
        Covariate matrix, one row per subject.
    times : array_like
        Evaluation times.

    Returns
    -------
    survival : ndarray, shape ``(n_samples, n_times)``
        ``survival[i, k]`` is the predicted survival of subject ``i`` at
        ``times[k]``.
    """
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    n = Z_arr.shape[0]
    times = _as_1d(times, "times")
    cols = []
    for t in times:
        out = np.asarray(model.sf(np.full(n, float(t)), Z_arr), dtype=float)
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
            \frac{S(t\mid Z_i)^2\, I(x_i \le t,\ \delta_i=1)}{\hat G(x_i)}
          + \frac{(1-S(t\mid Z_i))^2\, I(x_i > t)}{\hat G(t)} \Big],

    where :math:`\hat G` is the Kaplan-Meier estimate of the censoring
    survival. Subjects censored before ``t`` contribute nothing (their status
    at ``t`` is unknown); the IPCW weights correct for that loss. Lower is
    better.

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
        evaluation ``x`` / ``c``.

    Returns
    -------
    times, bs : ndarray
        The horizons and the Brier score at each.
    """
    x = _as_1d(x, "x")
    c = _as_1d(c, "c")
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
    xt = x if x_train is None else _as_1d(x_train, "x_train")
    ct = c if c_train is None else _as_1d(c_train, "c_train")

    uniq, g = _censoring_km(xt, ct)
    g_xi = _g_at(uniq, g, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_case = np.where(g_xi > 0, 1.0 / g_xi, 0.0)

    n = x.size
    bs = np.empty(times.size)
    for k, t in enumerate(times):
        s = survival[:, k]
        g_t = float(_g_at(uniq, g, np.array([t]))[0])
        w_ctrl = 1.0 / g_t if g_t > 0 else 0.0
        died = (x <= t) & (c == 0)
        alive = x > t
        term = np.zeros(n)
        term[died] = s[died] ** 2 * w_case[died]
        term[alive] = (1.0 - s[alive]) ** 2 * w_ctrl
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
    by its span. A single number summarising a survival predictor's accuracy
    (lower is better); a model that predicts the true ``S(t | Z)`` scores below
    the marginal Kaplan-Meier reference.
    """
    times_arr, bs = brier_score(x, c, survival, times, x_train, c_train)
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
        controls yields ``nan``.
    """
    x = _as_1d(x, "x")
    c = _as_1d(c, "c")
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
    xt = x if x_train is None else _as_1d(x_train, "x_train")
    ct = c if c_train is None else _as_1d(c_train, "c_train")

    uniq, g = _censoring_km(xt, ct)
    g_xi = _g_at(uniq, g, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(g_xi > 0, 1.0 / g_xi, 0.0)

    auc = np.full(times.size, np.nan)
    for k, t in enumerate(times):
        r = risk[:, k]
        cases = (x <= t) & (c == 0)
        controls = x > t
        if not cases.any() or not controls.any():
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
