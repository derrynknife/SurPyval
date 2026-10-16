"""The calibration of predicted survival, a check ``diagnose()`` shares
between families (#881).

One check, written once: :func:`calibration_check` is the
:class:`~.registry.Check` the Cox family (:mod:`.cox`) and the parametric
proportional hazards and accelerated failure time families
(:mod:`.ph_aft`) list among their checks, and :func:`calibration_advice`
the advice they give when it warns. The computation is
:func:`calibration_curve`, which takes any predicted survival
probabilities and the outcomes they predict.

The method
----------
Austin, Harrell and van Klaveren (2020, *Statistics in Medicine* 39,
2714-2742) at a time ``t0``: each unit's predicted risk of failure by
``t0``, ``p = 1 - S(t0 | Z)``, is set against the *observed* risk of the
units with that prediction, estimated by a Cox model of the observed
times on a restricted cubic spline (three knots, R's ``rms::rcs``) of the
complementary log-log of the prediction, ``log(-log(1 - p))``. The
observed risk is that model's ``1 - S(t0)``, so right-censored units count
as at risk until they are censored, never as survivors, and a unit that
entered late (``tl``) is at risk only after its entry. The curve, observed
against predicted risk, lies on the diagonal for a model that is
calibrated at ``t0``; its summaries are the weighted mean (``ICI``, the
integrated calibration index), median (``E50``) and 90th percentile
(``E90``) of ``|observed - predicted|`` over the units.

The paper describes a second estimator, hazard regression
(``polspline::hare``), which SurPyval does not have; the Cox model with
splines is the one here (lifelines' ``survival_probability_calibration``
uses a third, a parametric spline-baseline model linear in the
complementary log-log). The knots are at the 10th, 50th and 90th
percentiles of the complementary log-log (the 5th smallest and largest
values below 100 units; the interior distinct values when there are five
or fewer), as ``Hmisc::rcspline.eval`` places three; predictions with
fewer than three distinct knots enter linearly (the calibration-slope
model). The calibration model's pointwise band, at the report's level, is
the delta-method one of R's ``survfit`` for a Cox model (the baseline's and
the coefficients' variance, with R's default ``conf.type = "log"``).

The check
---------
``calibration``
    A row for each of three times ``t0``: the lower quartile, the median
    and the upper quartile of the failure times (the first failure time by
    which a quarter, a half and three quarters of the failures had
    happened; with counts ``n``, as the rows written out). Several times,
    because a model can be calibrated at one and not another (a wrong
    baseline is off most where its hazard is furthest from the data's: an
    exponential model of times whose hazard rises is worst early, the
    log-normal model of R's lung data off only early); quartiles of the
    failure times, because the curve needs failures before ``t0`` and
    units still at risk at it, and they rescale with time (principle 6).
    ``statistic`` is the ICI, and ``detail`` has E50, E90, the range of the
    predictions, the share of units outside the band and, for a row that
    warns, how the predictions are off (too high or low on average; too
    extreme or too moderate, by the slope of the observed on the predicted
    risk).

    A screen, never a fail: on the data the model was fitted to, a test of
    calibration has no known null distribution (a fitted model is
    calibrated in the large by construction). It warns when the curve is
    both far from the diagonal and clearly so: the ICI exceeds
    :data:`ICI_LIMIT` (0.02: predictions off by more than two percentage
    points on average) and the predicted risk is outside the curve's
    ``1 - alpha`` pointwise band for more than :data:`OUTSIDE_LIMIT` (10%)
    of the units. In simulations (two normal covariates, Weibull times,
    uniform censoring; 200 runs each at 50, 100, 300 and 1000 units) a
    correctly specified Cox model warned in 4%, 2%, 0% and 0% of reports
    and a Weibull PH model in 12%, 7%, 5% and 2%; a Cox model with its
    coefficients halved in 28%, 79%, 100% and 100%, one fitted to data
    with a 1.5 times longer scale in 76%, 94%, 100% and 100%, an
    exponential PH model of Weibull (shape 2) times in 99% or more, and a
    Cox model missing a quadratic term in 22%, 35%, 87% and 100%.

    Drawn as the curve against the diagonal, with its band and a
    histogram of the predicted risks, at each ``t0``.

What it can see
---------------
The calibration is *apparent*, of the model on the data it was fitted
to, so it shows misspecification, not overfitting (for which a model must
be validated on other data: :func:`calibration_curve` takes the
predictions of any model and any outcomes). The calibration model is a
proportional hazards model of the predictions, with its own baseline. So a
Cox model's curve bends only where its log hazard is not linear in its
linear predictor (a covariate with the wrong form, a missing interaction):
an effect that changes with time is not seen (in the simulations above, a
Cox model with an effect that stops warned in 1-10% of reports), which is
the ``proportional_hazards`` check's to find. A parametric model's curve
also shows a baseline that does not fit.

What it refuses
---------------
The check has no rows, and a note says why, for a model fitted to
left-censored, interval-censored or right-truncated rows (the calibration
model is a Cox model, which takes exact and right-censored times, with
delayed entry), to time-varying covariates (a unit's prediction depends on
its path, not one row), with fewer than :data:`MIN_FAILURES` failures, or
whose predicted survival is the same for every unit. A row whose
calibration model has no verified maximum (predictions that separate the
failures from the survivors) warns, and its detail says so.

``t0`` as an option
-------------------
``diagnose()`` has no family options yet, so the times are the quartiles.
:func:`calibration_check` takes ``times``, a callable of the report's
:class:`~.registry.Context` that returns ``(name, t0)`` pairs; an option
such as ``diagnose(calibration={"t0": [52]})`` would be read there
(``calibration_check(times=...)`` with a callable that returns the
option's times, or :func:`quartile_times` without it), without changing
anything else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import norm

from .registry import AdviceRule, Check, Context, Panel

#: The quantiles of the failure times the check is made at.
QUANTILES = (0.25, 0.5, 0.75)
#: The names of :data:`QUANTILES` in the table.
QUANTILE_NAMES = {0.25: "25%", 0.5: "50%", 0.75: "75%"}
#: The quantiles of the complementary log-log of the predicted risk at
#: which the spline's three knots are (Harrell's default for three).
KNOT_QUANTILES = (0.1, 0.5, 0.9)
#: The ICI above which the check may warn: predictions more than two
#: percentage points off on average.
ICI_LIMIT = 0.02
#: The share of the units whose predicted risk is outside the curve's
#: pointwise band above which the check may warn.
OUTSIDE_LIMIT = 0.10
#: The fewest failures the check is made with.
MIN_FAILURES = 10


# ---------------------------------------------------------------------------
# Weighted quantiles and the spline
# ---------------------------------------------------------------------------
def weighted_quantile(values: Any, weights: Any, q: Any) -> np.ndarray:
    """
    The quantile ``q`` of ``values`` with counts ``weights``: numpy's
    default (linear) quantile of the values each written out ``weights``
    times, without writing them out (principle 5).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.calibration import weighted_quantile
    >>> weighted_quantile([1.0, 2.0, 3.0], [1, 2, 1], [0.5, 0.9])
    array([2. , 2.7])
    >>> np.quantile([1.0, 2.0, 2.0, 3.0], [0.5, 0.9])
    array([2. , 2.7])
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    h = (total - 1.0) * np.asarray(q, dtype=float)
    lo = np.floor(h)
    hi = np.minimum(lo + 1.0, total - 1.0)
    last = v.size - 1
    v_lo = v[np.minimum(np.searchsorted(cw, lo, side="right"), last)]
    v_hi = v[np.minimum(np.searchsorted(cw, hi, side="right"), last)]
    return v_lo + (h - lo) * (v_hi - v_lo)


def lower_quantile(values: Any, weights: Any, q: float) -> float:
    """
    The smallest of ``values`` at which the weighted share of ``values``
    at or below it reaches ``q``: an observed value, as the time a share
    ``q`` of the failures had happened by.

    Examples
    --------
    >>> from surpyval.diagnostics.calibration import lower_quantile
    >>> lower_quantile([5.0, 1.0, 3.0, 2.0], [1, 1, 1, 1], 0.5)
    2.0
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    share = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(share, q - 1e-12)), v.size - 1)])


def rcs_basis(x: Any, knots: Any) -> np.ndarray:
    """
    Harrell's restricted cubic spline basis at ``x`` with ``knots``, as
    R's ``rms::rcs`` (``Hmisc::rcspline.eval`` with ``norm = 2``): the
    columns ``x`` and one cubic term per interior knot, linear beyond the
    outer knots.

    Examples
    --------
    >>> from surpyval.diagnostics.calibration import rcs_basis
    >>> rcs_basis([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]).round(4)
    array([[0.  , 0.  ],
           [1.  , 0.25],
           [2.  , 1.5 ],
           [3.  , 3.  ]])
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    kd = (k[-1] - k[0]) ** (2.0 / 3.0)

    def cube(u: np.ndarray) -> np.ndarray:
        return np.maximum(u / kd, 0.0) ** 3

    cols = [x]
    for j in range(k.size - 2):
        cols.append(
            cube(x - k[j])
            + (
                (k[-2] - k[j]) * cube(x - k[-1])
                - (k[-1] - k[j]) * cube(x - k[-2])
            )
            / (k[-1] - k[-2])
        )
    return np.column_stack(cols)


def _knots(cll: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The spline's knots, as ``Hmisc::rcspline.eval`` places three: the
    10th, 50th and 90th weighted percentiles of ``cll``, the outer ones at
    the 5th smallest and largest values for fewer than 100 units; or, for
    five or fewer distinct values, the distinct values but the smallest
    and largest. The distinct ones (fewer than three: the predictions
    enter linearly)."""
    distinct = np.unique(cll)
    if distinct.size <= 5:
        return distinct[1:-1]
    k = weighted_quantile(cll, n, KNOT_QUANTILES)
    if n.sum() < 100:
        order = np.argsort(cll, kind="stable")
        s, cw = cll[order], np.cumsum(n[order])
        k[0] = s[min(int(np.searchsorted(cw, 5.0)), s.size - 1)]
        k[-1] = s[min(int(np.searchsorted(cw, cw[-1] - 4.0)), s.size - 1)]
    return np.unique(k)


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------
def _cll(risk: np.ndarray) -> np.ndarray:
    return np.log(-np.log1p(-risk))


@dataclass
class CalibrationCurve:
    """
    The smoothed calibration curve of predicted survival at ``t0``
    (:func:`calibration_curve`).

    Attributes
    ----------
    t0 : float
        The time the risks are of.
    predicted, observed : numpy.ndarray
        Each unit's predicted risk of failure by ``t0``, ``1 - S(t0)``,
        and the observed risk of the units with that prediction (the
        calibration model's).
    lower, upper : numpy.ndarray
        The pointwise ``1 - alpha`` band of ``observed``.
    n : numpy.ndarray
        Each unit's count.
    ici, e50, e90 : float
        The weighted mean, median and 90th percentile of ``|observed -
        predicted|``.
    outside : float
        The weighted share of the units whose predicted risk is outside
        the band.
    knots : numpy.ndarray
        The knots of the spline in ``log(-log(1 - predicted))`` (one value,
        the median, when the predictions enter linearly).
    model
        The calibration model, a Cox model fitted with ``CoxPH.fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.diagnostics.calibration import calibration_curve
    >>> rng = np.random.default_rng(1)
    >>> scale = 10 * np.exp(rng.normal(size=200))
    >>> curve = calibration_curve(
    ...     np.exp(-5.0 / scale), rng.exponential(scale), t0=5.0
    ... )
    >>> curve.predicted.shape, curve.observed.shape
    ((200,), (200,))
    """

    t0: float
    predicted: np.ndarray
    observed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n: np.ndarray
    ici: float
    e50: float
    e90: float
    outside: float
    knots: np.ndarray
    model: Any
    alpha: float = 0.05
    _sums: Any = field(default=None, repr=False)

    def at(self, predicted: Any) -> tuple[np.ndarray, ...]:
        """
        The curve at the predicted risks ``predicted``: ``(observed,
        lower, upper)``, the observed risk and its band (to draw the curve
        on a grid).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.diagnostics.calibration import calibration_curve
        >>> rng = np.random.default_rng(1)
        >>> scale = 10 * np.exp(rng.normal(size=200))
        >>> curve = calibration_curve(
        ...     np.exp(-5.0 / scale), rng.exponential(scale), t0=5.0
        ... )
        >>> observed, lower, upper = curve.at([0.2, 0.4])
        >>> bool(np.all(lower <= observed) and np.all(observed <= upper))
        True
        """
        p = np.clip(np.asarray(predicted, dtype=float), 1e-10, 1 - 1e-10)
        return _observed(self, self._design(_cll(p)))

    def _design(self, cll: np.ndarray) -> np.ndarray:
        if self.knots.size >= 3:
            return rcs_basis(cll, self.knots)
        return cll[:, None]


def _observed(curve: CalibrationCurve, B: np.ndarray) -> tuple:
    """The calibration model's risk of failure by ``t0``, ``1 - S(t0 |
    B)``, at the design rows ``B``, and its pointwise band: the
    delta-method variance of the cumulative hazard ``H(t0 | B) = e^(b'B)
    sum_j h0_j``, ``e^(2 b'B) sum_j h0_j^2 / d_j + g' V g`` with ``g =
    e^(b'B) sum_j h0_j (B - Bbar_j)``, over the failure times ``t_j <=
    t0`` (``Bbar_j`` the risk set's weighted mean), on the scale of ``log
    S = -H`` (R's default ``conf.type = "log"``): the survival's band is
    ``S e^(+-z se(H))``, capped at 1."""
    h0, d, bbar = curve._sums
    cal = curve.model
    beta = np.asarray(cal.beta, dtype=float)
    center = getattr(cal, "center", None)
    shift = np.zeros(beta.size) if center is None else np.asarray(center)
    phi = np.exp((B - shift) @ beta)
    H = phi * h0.sum()
    g = phi[:, None] * (h0.sum() * (B - shift) - (h0 @ (bbar - shift)))
    V = np.asarray(cal.covariance(), dtype=float)
    var = phi**2 * np.sum(h0**2 / d) + np.einsum("ij,jk,ik->i", g, V, g)
    se = np.sqrt(np.maximum(var, 0.0))
    z = norm.isf(curve.alpha / 2.0)
    observed = -np.expm1(-H)
    # the risk's band, 1 - S e^(+-z se): S e^(z se) is capped at 1
    lower = np.maximum(-np.expm1(-H + z * se), 0.0)
    upper = -np.expm1(-H - z * se)
    return observed, lower, upper


def _risk_set_means(
    x: np.ndarray,
    tl: np.ndarray,
    w: np.ndarray,
    B: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """The ``w``-weighted mean of the rows of ``B`` at risk (``tl < t <=
    x``) at each of ``times``."""

    def suffix(key: np.ndarray, values: np.ndarray) -> np.ndarray:
        order = np.argsort(key, kind="stable")
        sums = np.cumsum(values[order][::-1], axis=0)[::-1]
        sums = np.vstack([sums, np.zeros((1, values.shape[1]))])
        return sums[np.searchsorted(key[order], times, side="left")]

    values = np.column_stack([w, w[:, None] * B])
    sums = suffix(x, values)
    if np.isfinite(tl).any():
        sums = sums - suffix(tl, values)
    return sums[:, 1:] / sums[:, :1]


def calibration_curve(
    survival: Any,
    x: Any,
    c: Any = None,
    n: Any = None,
    tl: Any = None,
    *,
    t0: float,
    alpha: float = 0.05,
) -> CalibrationCurve:
    """
    The smoothed calibration curve of predicted survival at ``t0``
    (Austin, Harrell and van Klaveren 2020), and its ICI, E50 and E90.

    Each unit's predicted risk, ``1 - survival``, is set against the
    observed risk of the units with that prediction: ``1 - S(t0)`` of a
    Cox model (``CoxPH.fit``) of the observed times on a restricted cubic
    spline, with three knots, of ``log(-log(survival))``. Right-censored
    units are at risk until they are censored, and a unit with an entry
    time ``tl`` only after it. See :mod:`surpyval.diagnostics.calibration`.

    Parameters
    ----------
    survival : array-like
        Each unit's predicted probability of surviving past ``t0``, from
        any model (on the data it was fitted to, or on other data to
        validate it).
    x : array-like
        The observed times.
    c : array-like, optional
        0 for a failure, 1 for a right-censored time (SurPyval's flags);
        all failures by default.
    n : array-like, optional
        Counts; 1 each by default.
    tl : array-like, optional
        Entry (left-truncation) times; ``-inf``, none, by default.
    t0 : float
        The time the risks are of.
    alpha : float, optional
        The band is the pointwise ``1 - alpha`` one. Default 0.05.

    Returns
    -------
    CalibrationCurve

    Examples
    --------
    Predictions from the true model of exponential times are calibrated:

    >>> import numpy as np
    >>> from surpyval.diagnostics.calibration import calibration_curve
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=2000)
    >>> t = rng.exponential(10 * np.exp(-z))
    >>> survival = np.exp(-5.0 / (10 * np.exp(-z)))  # S(5 | z)
    >>> curve = calibration_curve(survival, t, t0=5.0)
    >>> bool(curve.ici < 0.01)
    True

    The same predictions are miscalibrated for times twice as long:

    >>> bool(calibration_curve(survival, 2 * t, t0=5.0).ici > 0.1)
    True
    """
    from surpyval import CoxPH

    S = np.asarray(survival, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    c = np.zeros(x.size, int) if c is None else np.asarray(c).ravel()
    n = np.ones(x.size) if n is None else np.asarray(n, float).ravel()
    tl = (
        np.full(x.size, -np.inf)
        if tl is None
        else np.asarray(tl, dtype=float).ravel()
    )
    if not (S.size == x.size == c.size == n.size == tl.size):
        raise ValueError(
            "'survival', 'x', 'c', 'n' and 'tl' must have one entry per "
            "unit; got {}, {}, {}, {} and {}.".format(
                S.size, x.size, c.size, n.size, tl.size
            )
        )
    if not np.all(np.isin(c, (0, 1))):
        raise ValueError(
            "'c' must be 0 (a failure) or 1 (right censored) for each "
            "unit: the calibration curve is a Cox model of the observed "
            "times, which takes no left- or interval-censored ones."
        )
    if not np.all(np.isfinite(S)) or np.any((S < 0) | (S > 1)):
        raise ValueError("'survival' must be probabilities, each in [0, 1].")
    risk = np.clip(1.0 - S, 1e-10, 1 - 1e-10)
    cll = _cll(risk)
    if np.unique(cll).size < 2:
        raise ValueError(
            "'survival' is the same for every unit, so there is no curve "
            "to draw: the calibration curve sets units with different "
            "predictions against each other."
        )
    knots = _knots(cll, n)
    linear = knots.size < 3
    if linear:
        knots = weighted_quantile(cll, n, [0.5])
    B = cll[:, None] if linear else rcs_basis(cll, knots)
    entry = {"tl": tl} if np.isfinite(tl).any() else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = CoxPH.fit(x, B, c=c, n=n, center=True, **entry)
        if not linear and np.isnan(np.asarray(cal.beta, float)).any():
            # the spline's term is aliased with the predictions'
            linear = True
            knots = weighted_quantile(cll, n, [0.5])
            B = cll[:, None]
            cal = CoxPH.fit(x, B, c=c, n=n, center=True, **entry)
    if getattr(cal, "maximum", "verified") != "verified":
        raise ValueError(
            "The calibration model (a Cox model of the times on the "
            "predictions) has no verified maximum ({}): the predictions "
            "separate the failures from the survivors, or there are too "
            "few failures.".format(cal.maximum)
        )
    bx = np.asarray(cal.x, dtype=float)
    d = np.asarray(cal.d, dtype=float)
    keep = (bx <= t0) & (d > 0)
    beta = np.asarray(cal.beta, dtype=float)
    eta = B @ beta
    w = n * np.exp(eta - eta.max())
    bbar = _risk_set_means(x, tl, w, B, bx[keep])
    curve = CalibrationCurve(
        t0=float(t0),
        predicted=risk,
        observed=np.empty(0),
        lower=np.empty(0),
        upper=np.empty(0),
        n=n,
        ici=float("nan"),
        e50=float("nan"),
        e90=float("nan"),
        outside=float("nan"),
        knots=knots,
        model=cal,
        alpha=float(alpha),
        _sums=(np.asarray(cal.h0, dtype=float)[keep], d[keep], bbar),
    )
    curve.observed, curve.lower, curve.upper = _observed(curve, B)
    diff = np.abs(curve.observed - risk)
    curve.ici = float(np.sum(n * diff) / n.sum())
    e50, e90 = weighted_quantile(diff, n, [0.5, 0.9])
    curve.e50, curve.e90 = float(e50), float(e90)
    out = (risk < curve.lower) | (risk > curve.upper)
    curve.outside = float(np.sum(n * out) / n.sum())
    return curve


# ---------------------------------------------------------------------------
# The model's data and predictions
# ---------------------------------------------------------------------------
@dataclass
class _Outcome:
    """The fitted rows: times, flags, counts, entry times, the covariates
    and the model's survival function ``sf(t, Z)``."""

    x: np.ndarray
    c: np.ndarray
    n: np.ndarray
    tl: np.ndarray
    Z: np.ndarray
    sf: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _is_cox(model: Any) -> bool:
    from surpyval.univariate.regression.semi_parametric_regression_model import (  # noqa: E501
        SemiParametricRegressionModel,
    )

    return isinstance(model, SemiParametricRegressionModel)


def _outcome(ctx: Context) -> "_Outcome | str":
    """The model's fitted rows, or the reason the check cannot be made on
    them."""

    def compute() -> "_Outcome | str":
        m = ctx.model
        if _is_cox(m):
            if getattr(m, "is_tvc", False):
                return (
                    "the model was fitted to time-varying covariates "
                    "(fit_tvc), so a unit's predicted survival depends on "
                    "its covariate path, not on one row"
                )
            data = m._fit_data
            x = np.asarray(data["x"], dtype=float)
            tl = data.get("tl")
            return _Outcome(
                x=x,
                c=np.asarray(data["c"]),
                n=np.asarray(data["n"], dtype=float),
                tl=(
                    np.full(x.size, -np.inf)
                    if tl is None
                    else np.asarray(tl, dtype=float)
                ),
                Z=np.asarray(data["Z"], dtype=float),
                sf=lambda t, Z: m.sf(t, Z),
            )
        data = m.data
        c = np.asarray(data.c)
        odd = int(np.sum(~np.isin(c, (0, 1))))
        if odd:
            return (
                "the model was fitted to {} left- or interval-censored "
                "row{}, and the calibration curve is a Cox model of the "
                "observed times, which takes exact and right-censored "
                "ones".format(odd, "" if odd == 1 else "s")
            )
        tr = np.asarray(data.tr, dtype=float)
        if np.isfinite(tr).any():
            return (
                "the model was fitted to right-truncated rows, and the "
                "calibration curve is a Cox model, which takes delayed "
                "entry but not right truncation"
            )
        Z = np.asarray(data.Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
        return _Outcome(
            x=np.asarray(data.x, dtype=float).ravel(),
            c=c,
            n=np.asarray(data.n, dtype=float),
            tl=np.asarray(data.tl, dtype=float),
            Z=Z,
            sf=lambda t, Z: m.sf(t, Z),
        )

    return ctx.cached("calibration_outcome", compute)


def quartile_times(ctx: Context) -> list[tuple[str, float]]:
    """
    The default times of the check: the lower quartile, median and upper
    quartile of the model's failure times (with counts), each named by its
    share of the failures (``"25%"``, ...); ``[]`` when the check cannot
    be made.

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.diagnostics.calibration import quartile_times
    >>> from surpyval.diagnostics.registry import Context
    >>> df = load_rossi_static()
    >>> model = CoxPH.fit(df["week"], df[["fin", "prio"]], c=1 - df["arrest"])
    >>> quartile_times(Context(model, 0.05, ["fin", "prio"], [0, 1]))
    [('25%', 17.0), ('50%', 28.0), ('75%', 42.0)]
    """
    data = _outcome(ctx)
    if isinstance(data, str):
        return []
    failed = data.c == 0
    if not failed.any():
        return []
    t, w = data.x[failed], data.n[failed]
    return [(QUANTILE_NAMES[q], lower_quantile(t, w, q)) for q in QUANTILES]


def _note(ctx: Context, text: str) -> None:
    if text not in ctx.notes:
        ctx.notes.append(text)


def _curves(
    ctx: Context, times: Callable[[Context], Sequence[tuple[str, float]]]
) -> list[tuple[str, float, "CalibrationCurve | str"]]:
    """``(name, t0, curve)`` at each time, ``curve`` the reason when it
    could not be computed; ``[]``, with a note, when the check cannot be
    made."""

    def compute() -> list:
        data = _outcome(ctx)
        if isinstance(data, str):
            _note(ctx, "Calibration is not checked: {}.".format(data))
            return []
        failures = float(data.n[data.c == 0].sum())
        if failures < MIN_FAILURES:
            _note(
                ctx,
                "Calibration is not checked: the model has {:g} "
                "failure{}, and a smoothed calibration curve needs at "
                "least {}.".format(
                    failures, "" if failures == 1 else "s", MIN_FAILURES
                ),
            )
            return []
        out = []
        for name, t0 in times(ctx):
            with np.errstate(all="ignore"):
                S = np.asarray(
                    data.sf(np.full(data.x.size, float(t0)), data.Z),
                    dtype=float,
                )
            if np.unique(np.round(S, 12)).size < 2:
                _note(
                    ctx,
                    "Calibration is not checked: the model predicts the "
                    "same survival for every unit, so there is no curve "
                    "to draw (the calibration curve sets units with "
                    "different predictions against each other).",
                )
                return []
            try:
                curve: "CalibrationCurve | str" = calibration_curve(
                    S,
                    data.x,
                    data.c,
                    data.n,
                    data.tl,
                    t0=float(t0),
                    alpha=ctx.alpha,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                curve = str(error)
            out.append((name, float(t0), curve))
        return out

    return ctx.cached("calibration_curves", compute)


def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


def _direction(curve: CalibrationCurve) -> str:
    """How the predictions are off: too high or too low on average, and
    too extreme or too moderate (the weighted least-squares slope of the
    observed on the predicted risk)."""
    n, p, o = curve.n, curve.predicted, curve.observed
    mean_gap = float(np.sum(n * (o - p)) / n.sum())
    pm = np.sum(n * p) / n.sum()
    om = np.sum(n * o) / n.sum()
    var = float(np.sum(n * (p - pm) ** 2))
    slope = float(np.sum(n * (p - pm) * (o - om)) / var) if var > 0 else 1.0
    words = []
    if abs(mean_gap) >= curve.ici / 2:
        words.append(
            "the observed risk is {} than predicted on average ({:+.3g})"
            "".format("higher" if mean_gap > 0 else "lower", mean_gap)
        )
    if abs(slope - 1.0) > 0.1:
        words.append(
            "the predictions are too {} (the observed risk rises {} per "
            "unit of predicted risk)".format(
                "extreme" if slope < 1 else "moderate", _fmt(slope)
            )
        )
    return "; ".join(words)


def _rows_of(
    times: Callable[[Context], Sequence[tuple[str, float]]],
) -> Callable[[Context], list[dict]]:
    def rows(ctx: Context) -> list[dict]:
        out = []
        level = "{:g}%".format(100 * (1 - ctx.alpha))
        for name, t0, curve in _curves(ctx, times):
            test = "ICI at t = {} ({} of failures)".format(_fmt(t0), name)
            if isinstance(curve, str):
                out.append(
                    {
                        "test": test,
                        "covariates": (),
                        "verdict": "warn",
                        "detail": "the curve could not be computed: " + curve,
                        "_t0": t0,
                    }
                )
                continue
            off = _off(curve)
            detail = (
                "E50 {}, E90 {}; predicted risk {} to {}; outside the "
                "curve's {} band for {:.0%} of units".format(
                    _fmt(curve.e50),
                    _fmt(curve.e90),
                    _fmt(curve.predicted.min()),
                    _fmt(curve.predicted.max()),
                    level,
                    curve.outside,
                )
            )
            how = _direction(curve) if off else ""
            if how:
                detail += "; " + how
            out.append(
                {
                    "test": test,
                    "covariates": (),
                    "statistic": curve.ici,
                    "verdict": "warn" if off else "ok",
                    "detail": detail,
                    "_t0": t0,
                    "_off": off,
                }
            )
        return out

    return rows


def _panels_of(
    times: Callable[[Context], Sequence[tuple[str, float]]],
) -> Callable[[Context], list[Panel]]:
    def panels(ctx: Context) -> list[Panel]:
        out = []
        for name, t0, curve in _curves(ctx, times):
            if isinstance(curve, str):
                continue

            def draw(
                ax: Any,
                options: Any,
                curve: CalibrationCurve = curve,
                t0: float = t0,
            ) -> None:
                from surpyval.plotting import BAND_ALPHA, INK, PALETTE

                p = curve.predicted
                lo, hi = weighted_quantile(p, curve.n, [0.01, 0.99])
                grid = np.linspace(lo, hi, 80) if hi > lo else np.array([lo])
                observed, lower, upper = curve.at(grid)
                top = float(max(np.max(upper), hi, 1e-3)) * 1.05
                counts, edges = np.histogram(
                    p, bins=20, range=(0.0, top), weights=curve.n
                )
                if counts.max() > 0:
                    ax.bar(
                        edges[:-1],
                        0.15 * top * counts / counts.max(),
                        width=np.diff(edges),
                        align="edge",
                        color=INK["muted"],
                        alpha=0.5,
                        linewidth=0,
                        label="predicted risks",
                    )
                ax.plot(
                    [0, top],
                    [0, top],
                    color=INK["secondary"],
                    ls="--",
                    lw=0.8,
                    label="perfect calibration",
                )
                ax.fill_between(
                    grid,
                    lower,
                    upper,
                    color=PALETTE[0],
                    alpha=BAND_ALPHA,
                    linewidth=0,
                    label="{:g}% band".format(100 * (1 - curve.alpha)),
                )
                ax.plot(grid, observed, color=PALETTE[0], label="observed")
                ax.set_xlim(0, top)
                ax.set_ylim(0, top)
                ax.set_xlabel("predicted P(fail by t = {})".format(_fmt(t0)))
                ax.set_ylabel("observed P(fail by t = {})".format(_fmt(t0)))

            out.append(
                Panel(
                    "calibration at t = {}: ICI {}".format(
                        _fmt(t0), _fmt(curve.ici)
                    ),
                    draw,
                )
            )
        return out

    return panels


# ---------------------------------------------------------------------------
# The check and its advice
# ---------------------------------------------------------------------------
#: The check's name, the table's ``check`` and ``plot(which=...)``'s.
NAME = "calibration"


def calibration_check(
    times: "Callable[[Context], Sequence[tuple[str, float]]] | None" = None,
) -> Check:
    """
    The ``calibration`` check, for a family's ``checks``: a row and a
    panel for each time of ``times(ctx)`` (``(name, t0)`` pairs; by
    default :func:`quartile_times`, the quartiles of the failure times).
    See :mod:`surpyval.diagnostics.calibration`.

    Examples
    --------
    >>> from surpyval.diagnostics.calibration import calibration_check
    >>> check = calibration_check()
    >>> check.name
    'calibration'
    """
    chosen = quartile_times if times is None else times
    return Check(
        NAME,
        "The predicted survival agrees with the observed: at each time, "
        "the units' observed risk of failure by then is their predicted "
        "risk.",
        rows=_rows_of(chosen),
        panels=_panels_of(chosen),
    )


def _warned(ctx: Context) -> list[CalibrationCurve]:
    """The curves of the rows that warned for miscalibration, in order."""
    return [
        curve
        for _, _, curve in ctx.cached("calibration_curves", lambda: [])
        if isinstance(curve, CalibrationCurve) and _off(curve)
    ]


def _off(curve: CalibrationCurve) -> bool:
    """The check's rule: far from the diagonal, and clearly so."""
    return curve.ici > ICI_LIMIT and curve.outside > OUTSIDE_LIMIT


def _opening(ctx: Context) -> str:
    """What the advice says was found: the same for every row that
    warned, so that the report gives the advice once."""
    curves = _warned(ctx)
    if len(curves) == 1:
        (curve,) = curves
        how = _direction(curve)
        return (
            "The predicted survival at t = {t0} is off: the predicted risk "
            "of failure by then is {ici} from the observed on average "
            "(ICI; E90 {e90}), and outside the calibration curve's band "
            "for {out:.0%} of units (see d.plot('calibration')){how}.".format(
                t0=_fmt(curve.t0),
                ici=_fmt(curve.ici),
                e90=_fmt(curve.e90),
                out=curve.outside,
                how=": " + how if how else "",
            )
        )
    each = [
        "at t = {} (ICI {}, E90 {}, outside the band for {:.0%} of "
        "units)".format(_fmt(c.t0), _fmt(c.ici), _fmt(c.e90), c.outside)
        for c in curves
    ]
    return (
        "The predicted survival is off {} (see d.plot('calibration'); "
        "each row's detail says how).".format(
            ", ".join(each[:-1]) + " and " + each[-1]
        )
    )


def _miscalibrated(row: dict, ctx: Context) -> bool:
    return bool(row.get("_off"))


def _cox_rules() -> tuple[AdviceRule, ...]:
    def message(row: dict, ctx: Context) -> str:
        return _opening(ctx) + (
            " On the data it was fitted to, a Cox model's calibration "
            "curve bends only where the log hazard is not linear in the "
            "model's linear predictor: its baseline is estimated from the "
            "data, and the calibration model is itself a proportional "
            "hazards model of the predictions. So look for a covariate "
            "that does not act linearly (the functional_form rows) and "
            "transform it in a formula, CoxPH.fit_from_df(..., "
            'formula="np.log(z) + ..."), or bin it, formula="C(z_band) + '
            '..."; or, for covariates whose effects depend on each other, '
            'add their interaction, formula="a * b + ...".'
        )

    return (
        AdviceRule(
            NAME,
            message,
            ("CoxPH.fit_from_df(formula=)",),
            when=_miscalibrated,
            on=("warn",),
        ),
    )


def _baseline_is(
    name: "str | None", baselines: Sequence[str]
) -> Callable[[Context], bool]:
    """Whether the model's baseline is the built-in ``name`` (``None``:
    one not among ``baselines``)."""

    def test(ctx: Context) -> bool:
        own = ctx.model.distribution.name
        return own == name if name is not None else own not in baselines

    return test


def _parametric_rules(
    kind: str, baselines: Sequence[str], same: Sequence[str]
) -> tuple[AdviceRule, ...]:
    """The advice of the family ``kind`` (``"PH"`` or ``"AFT"``), one
    rule per built-in baseline (and one for any other), so that each
    names the model's own fitters."""
    other_kind = "AFT" if kind == "PH" else "PH"
    rows_of = {"PH": "baseline_distribution", "AFT": "error_distribution"}
    word = {"PH": "baseline", "AFT": "error distribution"}[kind]
    act = (
        "rescale time rather than multiply the hazard"
        if other_kind == "AFT"
        else "multiply the hazard rather than rescale time"
    )
    rules = []
    for name in list(baselines) + [None]:
        if name is None:
            own = "{}(dist)".format(kind)
            other = "{}(dist)".format(other_kind)
            features: tuple[str, ...] = (kind, other_kind, "CoxPH.fit")
        else:
            own = name + kind
            # A Weibull's (an exponential's) PH and AFT models are one
            # model: the other kind named is the log-normal's.
            other = ("LogNormal" if name in same else name) + other_kind
            features = (
                other + ".fit",
                own + ".fit_from_df(formula=)",
                "CoxPH.fit",
            )

        def message(
            row: dict, ctx: Context, own: str = own, other: str = other
        ) -> str:
            return _opening(ctx) + (
                " A parametric model is calibrated when its {word} and the "
                "form of the covariates' effects are right. Compare the "
                "other {word}s (the {rows} rows: each built-in one "
                "refitted, by AIC) and refit with a better one; or fit "
                "{other}.fit, in which the covariates {act}; or let the "
                "data give the baseline, with a Cox model, CoxPH.fit. If "
                "a covariate does not act linearly (the functional_form "
                "rows), transform it in a formula, {own}.fit_from_df(..., "
                'formula="np.log(z) + ...").'.format(
                    word=word,
                    rows=rows_of[kind],
                    other=other,
                    act=act,
                    own=own,
                )
            )

        def when(
            row: dict,
            ctx: Context,
            on_dist: Callable[[Context], bool] = _baseline_is(name, baselines),
        ) -> bool:
            return _miscalibrated(row, ctx) and on_dist(ctx)

        rules.append(
            AdviceRule(NAME, message, features, when=when, on=("warn",))
        )
    return tuple(rules)


def calibration_advice(
    family: str,
    baselines: Sequence[str] = (),
    same: Sequence[str] = (),
) -> tuple[AdviceRule, ...]:
    """
    The advice of the ``calibration`` check, for a family's ``advice``,
    when it warns: SurPyval's own fixes for a miscalibrated model of the
    ``family`` (``"cox"``, ``"PH"`` or ``"AFT"``). For the parametric
    families, ``baselines`` are the built-in baselines (each with its
    ``<name>PH`` and ``<name>AFT`` fitter) and ``same`` those whose PH and
    AFT models are one model, so that the advice names the model's own
    fitters.

    Examples
    --------
    >>> from surpyval.diagnostics.calibration import calibration_advice
    >>> rule, = calibration_advice("cox")
    >>> rule.features
    ('CoxPH.fit_from_df(formula=)',)
    """
    if family == "cox":
        return _cox_rules()
    if family in ("PH", "AFT"):
        return _parametric_rules(family, baselines, same)
    raise ValueError(
        "'family' must be 'cox', 'PH' or 'AFT'; got {!r}.".format(family)
    )
