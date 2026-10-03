"""Forecasting failures from where a fleet is now (#581).

A fitted model answers questions about new units; a fleet in service is
made of units that have already survived to their current ages. Each unit
(or cohort of ``n`` identical units) at age :math:`a_i` fails within the
next :math:`h` with the conditional probability

.. math::
    p_i(h) = 1 - R(h \\mid a_i) = \\frac{F(a_i + h) - F(a_i)}{1 - F(a_i)},

with its covariates :math:`Z_i` for a regression model. The number of
failures over the horizon is a sum of independent Bernoulli variables
(binomial for a cohort): a Poisson-binomial variable with mean
:math:`\\sum_i n_i p_i` and variance :math:`\\sum_i n_i p_i (1 - p_i)`,
whose exact distribution gives the prediction interval.

:func:`forecast` is the one entry point, for the univariate models (a
distribution, a non-parametric estimate) and the regression models (with
a covariate row per unit) alike.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import alpha_ci_error
from surpyval.utils.warnings import caller_stacklevel

__all__ = ["Forecast", "forecast"]

# Elements of the (units x frequencies) characteristic-function matrix
# built at once, so a large fleet is summed in chunks.
_CHUNK = 1_000_000
# The exact distribution costs (units x window) complex logarithms; past
# this many, where the count's variance is large enough for the refined
# normal approximation to be accurate to a fraction of a count at the
# interval's ends, that is used instead.
_EXACT_BUDGET = 4_000_000
_NORMAL_VARIANCE = 100.0


@dataclass
class Forecast:
    """The failures a fleet is expected to have over a horizon.

    Returned by :func:`forecast`. Each horizon :math:`h_k` (time from
    now) has the expected number of failures in :math:`(0, h_k]`, its
    variance and an equal-tailed prediction interval, and the same for
    the failures in each period :math:`(h_{k-1}, h_k]` (with
    :math:`h_0 = 0`).
    """

    #: The horizons, times from now, increasing.
    horizon: npt.NDArray
    #: ``probability[i, k]``: the chance that a unit of row ``i`` fails
    #: within ``horizon[k]``, given it has survived to its age.
    probability: npt.NDArray
    #: The number of units each row stands for.
    n: npt.NDArray
    #: Expected failures in ``(0, horizon[k]]``.
    expected: npt.NDArray
    #: Their variance, :math:`\\sum_i n_i p_i (1 - p_i)`.
    variance: npt.NDArray
    #: The prediction interval of the count in ``(0, horizon[k]]``: the
    #: ``alpha_ci / 2`` and ``1 - alpha_ci / 2`` quantiles of its exact
    #: (Poisson-binomial) distribution.
    lower: npt.NDArray
    upper: npt.NDArray
    #: Expected failures in each period ``(horizon[k - 1], horizon[k]]``.
    period_expected: npt.NDArray
    #: The prediction interval of each period's count.
    period_lower: npt.NDArray
    period_upper: npt.NDArray
    #: The significance level of the intervals.
    alpha_ci: float

    @property
    def unit_expected(self) -> npt.NDArray:
        """``unit_expected[i, k]``: the expected failures of row ``i``
        within ``horizon[k]`` (``n[i] * probability[i, k]``); sort by the
        last column for the units most at risk."""
        return self.n[:, None] * self.probability

    def __repr__(self) -> str:
        level = 100 * (1 - self.alpha_ci)
        head = (
            "Forecast of failures: {:g} units; {:g}% prediction "
            "intervals".format(float(np.sum(self.n)), level)
        )
        columns = [
            ("horizon", self.horizon),
            ("expected", self.expected),
            ("lower", self.lower),
            ("upper", self.upper),
            ("in period", self.period_expected),
            ("period lower", self.period_lower),
            ("period upper", self.period_upper),
        ]
        cells = [
            [name] + ["{:.4g}".format(v) for v in values]
            for name, values in columns
        ]
        widths = [max(len(c) for c in col) for col in cells]
        rows = [
            "  ".join(col[r].rjust(w) for col, w in zip(cells, widths))
            for r in range(len(self.horizon) + 1)
        ]
        return "\n".join([head] + rows)


def forecast(
    model: Any,
    age: npt.ArrayLike,
    horizon: npt.ArrayLike,
    Z: Any = None,
    n: npt.ArrayLike | None = None,
    limit: npt.ArrayLike | None = None,
    alpha_ci: float = 0.05,
) -> Forecast:
    r"""
    Forecast the failures of units in service at their current ages.

    Each unit (or cohort) at age :math:`a_i` that has not failed yet fails
    within the next :math:`h` with probability
    :math:`p_i(h) = 1 - R(a_i + h) / R(a_i)` (the model's conditional
    survival, ``cs``), and the fleet's failures over the horizon are the
    sum: expected :math:`\sum_i n_i p_i`, variance
    :math:`\sum_i n_i p_i (1 - p_i)`, and a prediction interval from
    their exact (Poisson-binomial) distribution. A unit that fails is
    counted once (it is not replaced): for a repairable fleet whose units
    are renewed on failure, the counts are those until each unit's first
    failure.

    Parameters
    ----------
    model : fitted model
        A univariate model (a distribution, a mixture, a non-parametric
        estimate; anything with ``sf(x)``) or a regression model (with
        ``sf(x, Z)``).
    age : array_like
        The current age of each unit or cohort: the time it has survived
        so far, on the model's time scale.
    horizon : scalar or array_like
        The time ahead to forecast over, or several increasing times
        ahead (the end of each period: ``[1, 2, ..., 12]`` months).
    Z : array_like or DataFrame, optional
        The covariates of each unit, for a regression model: one row per
        unit (or a single row for every unit), as its ``sf`` takes them.
        Not taken by a univariate model.
    n : array_like, optional
        The number of units in service at each age (a cohort's
        survivors); 1 each by default.
    limit : scalar or array_like, optional
        An age past which a unit's failures are not counted (the end of
        its warranty, or a planned retirement), one for every unit or one
        each. A unit at or past its limit contributes nothing.
    alpha_ci : float, optional
        The significance level of the prediction intervals (default 0.05,
        a 95% interval).

    Returns
    -------
    Forecast
        The expected failures in ``(0, h]`` for each horizon ``h``, their
        variance and prediction interval, the same for each period, and
        each unit's probability of failing (``probability``,
        ``unit_expected``).

    Raises
    ------
    ValueError
        If an age, horizon or limit is missing or infinite, the horizons
        are not positive and increasing, a count is not a whole number,
        ``Z`` is given to a univariate model or not given to a regression
        model, or ``alpha_ci`` is not strictly between 0 and 1.

    Warns
    -----
    RuntimeWarning
        If the model gives some units a survival of 0 at their age (they
        cannot be in service under it): their probability, and the
        totals, are ``nan``.

    Notes
    -----
    The units are taken to fail independently given their ages (and
    covariates), and the model as known: the interval is that of the
    count, not of the model's estimate, so it is narrower than an
    interval that also carried the uncertainty of the fit. Comparing the
    forecasts of several plausible models shows how much the
    extrapolation depends on the choice.

    The interval is the equal-tailed one of the exact distribution of the
    count (Hong, 2013), so its ends are whole numbers of failures. That
    costs a complex logarithm per unit and count in a window around the
    mean; for a fleet where that passes four million and the count's
    variance is at least 100, the refined normal approximation (Volkova,
    1996) is used instead, which is then within about 1e-4 of the exact
    distribution function (an end can move by one failure).

    Examples
    --------
    A warranty: three monthly shipment cohorts, in service for 3, 2 and 1
    months, with 12-month cover:

    >>> import surpyval as surv
    >>> model = surv.Weibull.from_params([60.0, 1.5])
    >>> result = surv.forecast(
    ...     model, age=[3, 2, 1], n=[950, 1000, 1000],
    ...     horizon=[1, 2, 3], limit=12,
    ... )
    >>> print(result)
    Forecast of failures: 2950 units; 95% prediction intervals
    horizon  expected  lower  upper  in period  period lower  period upper
          1     14.72      8     23      14.72             8            23
          2     32.21     22     44      17.49            10            26
          3     51.98     38     66      19.77            12            29

    A fleet with covariates: each unit's own age and covariate row.

    >>> import numpy as np
    >>> from surpyval import Weibull, WeibullPH
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> ph = WeibullPH.fit(x, Z)
    >>> fleet = surv.forecast(ph, age=[2.0, 5.0, 8.0], Z=[[0], [1], [1]],
    ...                       horizon=1.0)
    >>> fleet.probability.round(4)
    array([[0.0638],
           [0.2393],
           [0.3157]])
    >>> fleet.expected.round(4)
    array([0.6187])
    """
    if not 0 < alpha_ci < 1:
        raise alpha_ci_error(alpha_ci)
    age_arr = _finite(age, "age").reshape(-1)
    h = _finite(horizon, "horizon").reshape(-1)
    if h.size == 0 or np.any(h <= 0) or np.any(np.diff(h) <= 0):
        raise ValueError(
            "horizon must be positive times ahead, increasing; got "
            "{}".format(h.tolist())
        )
    k = age_arr.size
    counts = _counts(n, k)
    end_limit = (
        np.full(k, np.inf)
        if limit is None
        else np.broadcast_to(_finite(limit, "limit"), (k,)).astype(float)
    )
    regression = _takes_covariates(model)
    if regression and Z is None:
        raise ValueError(
            "This is a regression model: give the covariates of each unit "
            "as Z (one row per unit, or a single row for every unit)"
        )
    if not regression and Z is not None:
        raise ValueError(
            "This model takes no covariates, so Z cannot be used; it "
            "forecasts every unit from its age alone"
        )
    prob = np.empty((k, h.size))
    for j, hj in enumerate(h):
        prob[:, j] = _failure_probability(
            model, age_arr, hj, end_limit, Z, regression
        )
    undefined = np.isnan(prob).any(axis=1)
    if undefined.any():
        warnings.warn(
            "{} of the {} units have a survival of 0 at their age under "
            "this model, so their chance of failing is undefined (nan) and "
            "so are the totals. The model says no unit survives to those "
            "ages: check the ages are on the model's time scale, or the "
            "model's fit in its tail.".format(int(undefined.sum()), k),
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )
    period = np.diff(np.column_stack([np.zeros(k), prob]), axis=1)
    level = (alpha_ci / 2, 1 - alpha_ci / 2)
    cum = [_count_summary(prob[:, j], counts, level) for j in range(h.size)]
    per = [_count_summary(period[:, j], counts, level) for j in range(h.size)]
    return Forecast(
        horizon=h,
        probability=prob,
        n=counts,
        expected=np.array([c[0] for c in cum]),
        variance=np.array([c[1] for c in cum]),
        lower=np.array([c[2] for c in cum]),
        upper=np.array([c[3] for c in cum]),
        period_expected=np.array([c[0] for c in per]),
        period_lower=np.array([c[2] for c in per]),
        period_upper=np.array([c[3] for c in per]),
        alpha_ci=float(alpha_ci),
    )


def _finite(values: Any, name: str) -> npt.NDArray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "{} must be finite numbers; got {}".format(
                name, arr.reshape(-1)[~np.isfinite(arr.reshape(-1))][:5]
            )
        )
    return arr


def _counts(n: Any, k: int) -> npt.NDArray:
    if n is None:
        return np.ones(k)
    counts = np.broadcast_to(_finite(n, "n"), (k,)).astype(float)
    if np.any(counts < 0) or np.any(counts != np.round(counts)):
        raise ValueError(
            "n must be whole numbers of units, at least 0; got "
            "{}".format(counts[(counts < 0) | (counts != np.round(counts))])
        )
    return counts


def _takes_covariates(model: Any) -> bool:
    """Whether the model's ``sf`` takes covariates (a regression model)."""
    try:
        params = inspect.signature(model.sf).parameters
    except (TypeError, ValueError, AttributeError):
        raise ValueError(
            "forecast needs a fitted model with an sf; got {}".format(
                type(model).__name__
            )
        ) from None
    return "Z" in params


def _failure_probability(
    model: Any,
    age: npt.NDArray,
    h: float,
    limit: npt.NDArray,
    Z: Any,
    regression: bool,
) -> npt.NDArray:
    """The chance each unit fails in ``(age, min(age + h, limit)]`` given
    it survived to ``age``: ``1 - cs`` where the model has ``cs`` (exact
    in the far tail), else from the ratio of ``sf``."""
    span = np.clip(np.minimum(age + h, limit) - age, 0.0, None)
    args = (Z,) if regression else ()
    with np.errstate(all="ignore"):
        if callable(getattr(model, "cs", None)):
            survive = np.asarray(model.cs(span, age, *args), dtype=float)
        else:
            survive = np.asarray(
                model.sf(age + span, *args), dtype=float
            ) / np.asarray(model.sf(age, *args), dtype=float)
    survive = np.broadcast_to(survive, age.shape)
    # Nothing counted past a limit (span 0): certainly no failure, even
    # where the model's survival at the age is 0.
    return np.where(span > 0, 1.0 - survive, 0.0)


def _count_summary(
    p: npt.NDArray, n: npt.NDArray, level: tuple[float, float]
) -> tuple[float, float, float, float]:
    """``(mean, variance, lower, upper)`` of the number of successes of
    independent ``Binomial(n_i, p_i)`` variables."""
    if np.isnan(p).any():
        return (np.nan,) * 4
    p = np.clip(p, 0.0, 1.0)
    mean = float(np.sum(n * p))
    variance = float(np.sum(n * p * (1 - p)))
    size, _ = _window(n, mean, variance)
    rows = int(np.count_nonzero((p > 0) & (n > 0)))
    if rows * size > _EXACT_BUDGET and variance >= _NORMAL_VARIANCE:
        values, cdf = _refined_normal_cdf(p, n, mean, variance)
    else:
        values, pmf = _poisson_binomial_pmf(p, n, mean, variance)
        cdf = np.cumsum(pmf)
    # The smallest count whose cumulative probability reaches each level.
    at = np.minimum(np.searchsorted(cdf, level), len(values) - 1)
    lower, upper = values[at]
    return mean, variance, float(lower), float(upper)


def _window(n: npt.NDArray, mean: float, variance: float) -> tuple[int, int]:
    """``(size, offset)`` of the counts the distribution is computed at:
    every count from 0 to the number of units, or a window around the
    mean 12 standard deviations and 60 wide on either side, outside which
    the mass is below 1e-30 (Bernstein's inequality)."""
    total = int(round(float(np.sum(n))))
    half = int(np.ceil(12 * np.sqrt(variance) + 60))
    if 2 * half + 1 >= total + 1:
        return total + 1, 0
    size = 2 * half + 1
    return size, int(np.clip(round(mean) - half, 0, total + 1 - size))


def _refined_normal_cdf(
    p: npt.NDArray, n: npt.NDArray, mean: float, variance: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """The counts of the window and the refined normal approximation of
    their cumulative probabilities (Volkova, 1996): the normal with a
    continuity and a skewness correction,
    :math:`\\Phi(x) + \\gamma (1 - x^2) \\phi(x) / 6` at
    :math:`x = (k + 1/2 - \\mu) / \\sigma`."""
    from scipy.special import ndtr

    size, offset = _window(n, mean, variance)
    values = offset + np.arange(size, dtype=float)
    sd = np.sqrt(variance)
    skew = float(np.sum(n * p * (1 - p) * (1 - 2 * p))) / sd**3
    x = (values + 0.5 - mean) / sd
    phi = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    cdf = ndtr(x) + skew * (1 - x**2) * phi / 6
    return values, np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)


def _poisson_binomial_pmf(
    p: npt.NDArray, n: npt.NDArray, mean: float, variance: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """The distribution of the sum of independent ``Binomial(n_i, p_i)``.

    From its characteristic function
    :math:`\\prod_i (1 - p_i + p_i e^{i\\omega})^{n_i}` at the ``M``
    Fourier frequencies (the DFT-CF method; Hong, 2013): exact when ``M``
    is the number of units plus one. A large fleet uses a window of ``M``
    values around the mean instead, wide enough (12 standard deviations
    and 60 on either side) that the mass outside it, which would wrap
    into it, is below 1e-30 by Bernstein's inequality. Returns the counts
    and their probabilities.
    """
    size, offset = _window(n, mean, variance)
    keep = (p > 0) & (n > 0)
    p, n = p[keep], n[keep]
    omega = 2 * np.pi * np.arange(size) / size
    log_cf = np.zeros(size, dtype=complex)
    step = max(1, _CHUNK // max(size, 1))
    for start in range(0, p.size, step):
        pc = p[start : start + step, None]
        nc = n[start : start + step, None]
        z = 1 - pc + pc * np.exp(1j * omega)
        # An exact zero (p = 1/2 at omega = pi) is a factor of 0.
        z = np.where(z == 0, 1e-300, z)
        log_cf += np.sum(nc * np.log(z), 0)
    cf = np.exp(log_cf - 1j * omega * offset)
    pmf = np.clip(np.fft.fft(cf).real / size, 0.0, None)
    pmf /= pmf.sum()
    return offset + np.arange(size, dtype=float), pmf
