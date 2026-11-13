r"""Sample size and power for four comparisons the plain log-rank
calculation of :mod:`surpyval.test_planning.comparison` does not answer:

- **A Cox model's coefficient** (Hsieh and Lavori 2000): the events, units
  and power to detect a hazard ratio per unit of a covariate, continuous
  (with standard deviation ``sd``) or of two levels (with ``proportion``
  of the units at the first), when the covariate is correlated with the
  others in the model (``r_squared``, its squared multiple correlation
  with them, which inflates the variance of its coefficient by
  :math:`1 / (1 - R^2)`).
- **Non-inferiority and equivalence** of a hazard ratio: the events,
  units and power to show that the hazard ratio is below a margin, or
  between two margins, rather than different from 1.
- **The restricted mean survival time** (RMST) up to a horizon
  :math:`\tau`: the units and power to detect a difference in RMST
  between two groups whose survival curves are given, with the variance
  of each group's Kaplan-Meier RMST computed from the curves and the
  censoring of the study (uniform entry over the accrual time, follow-up
  after it, and dropout), not by simulation.
- **A log-rank test when some units are cured** (never fail): the units,
  events and power when each group's survival is a mixture of a cured
  fraction and a latency distribution, so that the hazards are not
  proportional.

The last two use the asymptotic mean and variance of the test statistic
under the alternative, integrated over the study's censoring
distribution. With uniform entry over ``[0, a]`` and the study ending
``f`` after the last entry, a unit is under observation (not yet censored
by the study's end) at time :math:`t` since its entry with probability

.. math::
    G(t) = \min\{1, (a + f - t) / a\}\, e^{-\delta t}, \qquad
    0 \le t \le a + f,

with :math:`\delta` the dropout rate (``dropout_rate``).

References
----------
Hsieh, F. Y. and Lavori, P. W. (2000), "Sample-size calculations for the
Cox proportional hazards regression model with nonbinary covariates",
Controlled Clinical Trials 21, 552-560.

Schoenfeld, D. A. (1983), "Sample-size formula for the proportional-hazards
regression model", Biometrics 39, 499-503.

Schoenfeld, D. (1981), "The asymptotic properties of nonparametric tests
for comparing survival distributions", Biometrika 68, 316-319.

Chow, S.-C., Shao, J. and Wang, H. (2008), "Sample Size Calculations in
Clinical Research", 2nd edition, Chapman and Hall, chapter 7.

Jung, S.-H., Kang, S. J., McCall, L. M. and Blumenstein, B. (2005),
"Sample size computation for two-sample noninferiority log-rank test",
Journal of Biopharmaceutical Statistics 15, 969-979.

Royston, P. and Parmar, M. K. B. (2013), "Restricted mean survival time:
an alternative to the hazard ratio for the design and analysis of
randomized trials with a time-to-event outcome", BMC Medical Research
Methodology 13, 152.

Yung, G. and Liu, Y. (2020), "Sample size and power for the weighted
log-rank test and Kaplan-Meier based tests with allowance for
nonproportional hazards", Biometrics 76, 939-950.

Wang, S., Zhang, J. and Lu, W. (2012), "Sample size calculation for the
proportional hazards cure model", Statistics in Medicine 31, 3959-3971.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from surpyval.test_planning._checks import (
    check_count,
    check_positive,
    check_probability,
    elementwise,
)
from surpyval.test_planning.comparison import (
    _check_hazard_ratio,
    _check_ratio,
    _event_probabilities,
    _z_alpha,
)
from surpyval.utils.validation import check_alpha_ci, check_option

DIRECTIONS = ("increase", "decrease")

# Intervals in each piece of the integration grid (before and after the
# end of the minimum follow-up).
_GRID = 20000

SurvivalCurve = Any


# --- Cox regression: one covariate ------------------------------------------


def _covariate_variance(
    sd: float | None, proportion: float | None, r_squared: float
) -> float:
    """The variance of the covariate not explained by the others,
    ``var * (1 - r_squared)``."""
    if (sd is None) == (proportion is None):
        raise ValueError(
            "Give one of sd (the standard deviation of a continuous "
            "covariate) or proportion (the share of the units at the first "
            "level of a two-level covariate), not both or neither."
        )
    if sd is not None:
        variance = check_positive(sd, "sd") ** 2
    else:
        share = check_probability(proportion, "proportion")
        variance = share * (1.0 - share)
    try:
        inside = bool(0.0 <= r_squared < 1.0)
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise ValueError(
            "r_squared, the squared multiple correlation of the covariate "
            "with the other covariates, must be at least 0 and below 1, "
            f"got {r_squared!r}; use 0 when there are no others."
        )
    return variance * (1.0 - float(r_squared))


@elementwise
def cox_events(
    hazard_ratio: float,
    power: float = 0.8,
    *,
    sd: float | None = None,
    proportion: float | None = None,
    r_squared: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    r"""The events a Cox model needs to detect a covariate's hazard ratio.

    The number of events with which the Wald (or score) test of one
    coefficient of a proportional hazards model, at level ``alpha_ci``,
    detects a hazard ratio ``hazard_ratio`` per unit of the covariate with
    probability ``power``. Hsieh and Lavori's (2000) formula,

    .. math::
        d = \frac{(z_{1 - \alpha / 2} + z_{\text{power}})^2}
                 {\sigma^2 (1 - R^2) (\ln \text{HR})^2},

    with :math:`\sigma` the covariate's standard deviation (``sd``) and
    :math:`R^2` its squared multiple correlation with the other covariates
    in the model (``r_squared``): the correlation inflates the variance of
    the coefficient by :math:`1 / (1 - R^2)`, so a covariate half explained
    by the others needs twice the events. For a covariate of two levels,
    ``proportion`` :math:`\pi` of the units at the first, the variance is
    :math:`\pi (1 - \pi)`, and with ``r_squared=0`` this is Schoenfeld's
    formula, :func:`~surpyval.test_planning.logrank_events`.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, per unit of the covariate (between the
        two levels for a two-level covariate): positive, and not 1. Per
        standard deviation, give ``sd=1``.
    power : float, optional
        The chance of detecting it, by default 0.8.
    sd : float, optional
        Keyword only: the standard deviation of a continuous covariate,
        in its units. Give this or ``proportion``.
    proportion : float, optional
        Keyword only: the share of the units at the first level of a
        two-level covariate. Give this or ``sd``.
    r_squared : float, optional
        Keyword only: the squared multiple correlation of the covariate
        with the other covariates in the model (the :math:`R^2` of a
        regression of it on them), at least 0 and below 1; by default 0.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    int
        The number of events, rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range, or both or neither of ``sd`` and
        ``proportion`` are given.

    Examples
    --------
    A hazard ratio of 1.5 per standard deviation of a measurement takes 48
    events; if the other covariates explain 30% of its variance, 69:

    >>> from surpyval.test_planning import cox_events
    >>> cox_events(1.5, sd=1.0)
    48
    >>> cox_events(1.5, sd=1.0, r_squared=0.3)
    69
    >>> cox_events(0.5, proportion=0.5)  # Schoenfeld's 66
    66
    """
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    power = check_probability(power, "power")
    variance = _covariate_variance(sd, proportion, r_squared)
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    events = z**2 / (variance * np.log(hazard_ratio) ** 2)
    return int(np.ceil(events - 1e-9))


@elementwise
def cox_sample_size(
    hazard_ratio: float,
    event_probability: float,
    power: float = 0.8,
    *,
    sd: float | None = None,
    proportion: float | None = None,
    r_squared: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    """The units a Cox model needs to detect a covariate's hazard ratio.

    The events of :func:`cox_events`, unrounded, divided by the chance
    that a unit has an event during the study, rounded up: R
    ``powerSurvEpi``'s ``ssizeEpiCont`` (continuous covariate) and
    ``ssizeEpi`` (two levels, with no other covariates' adjustment
    beyond :math:`R^2`).

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, per unit of the covariate: positive,
        and not 1.
    event_probability : float
        The chance that a unit has an event during the study.
    power : float, optional
        The chance of detecting it, by default 0.8.
    sd, proportion, r_squared, alpha_ci, alternative
        Keyword only: as for :func:`cox_events`.

    Returns
    -------
    int
        The number of units, rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Hsieh and Lavori's (2000) example: a log hazard ratio of 1.0 per unit
    of a covariate with standard deviation 0.3126, :math:`R^2 = 0.1837`
    with the others, 73.8% of the units having an event, a one-sided test
    at 5% with power 0.806: 107 units.

    >>> import numpy as np
    >>> from surpyval.test_planning import cox_sample_size
    >>> cox_sample_size(np.exp(1.0), 0.738, 0.806, sd=0.3126,
    ...                 r_squared=0.1837, alternative="one-sided")
    107
    """
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    probability = check_probability(event_probability, "event_probability")
    power = check_probability(power, "power")
    variance = _covariate_variance(sd, proportion, r_squared)
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    events = z**2 / (variance * np.log(hazard_ratio) ** 2)
    return int(np.ceil(events / probability - 1e-9))


@elementwise
def cox_power(
    hazard_ratio: float,
    n: int,
    event_probability: float,
    *,
    sd: float | None = None,
    proportion: float | None = None,
    r_squared: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    r"""The chance that a Cox model detects a covariate's hazard ratio,
    given the units.

    The formula of :func:`cox_events` solved for the power, with the
    expected events ``n * event_probability``:
    :math:`\Phi(\sqrt{d \sigma^2 (1 - R^2)} |\ln \text{HR}| -
    z_{1 - \alpha / 2})` (R ``powerSurvEpi``'s ``powerEpiCont``).

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, per unit of the covariate: positive,
        and not 1.
    n : int
        The number of units.
    event_probability : float
        The chance that a unit has an event during the study.
    sd, proportion, r_squared, alpha_ci, alternative
        Keyword only: as for :func:`cox_events`.

    Returns
    -------
    float
        The power.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.test_planning import cox_power
    >>> round(cox_power(np.exp(1.0), 107, 0.738, sd=0.3126,
    ...                 r_squared=0.1837, alternative="one-sided"), 3)
    0.806
    """
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    units = check_count(n, "n")
    probability = check_probability(event_probability, "event_probability")
    variance = _covariate_variance(sd, proportion, r_squared)
    information = units * probability * variance
    z = np.sqrt(information) * abs(np.log(hazard_ratio))
    return float(stats.norm.cdf(z - _z_alpha(alpha_ci, alternative)))


@elementwise
def cox_detectable(
    n: int,
    event_probability: float,
    power: float = 0.8,
    *,
    sd: float | None = None,
    proportion: float | None = None,
    r_squared: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    direction: str = "increase",
) -> float:
    """The smallest hazard ratio per unit of a covariate that a Cox model
    detects with the power given.

    The formula of :func:`cox_events` solved for the hazard ratio.

    Parameters
    ----------
    n : int
        The number of units.
    event_probability : float
        The chance that a unit has an event during the study.
    power : float, optional
        The chance of detecting it, by default 0.8.
    sd, proportion, r_squared, alpha_ci, alternative
        Keyword only: as for :func:`cox_events`.
    direction : str, optional
        Keyword only: ``"increase"`` (the default), the hazard ratio
        above 1, or ``"decrease"``, the one below.

    Returns
    -------
    float
        The hazard ratio per unit of the covariate.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    200 units of which 30% have an event detect a hazard ratio of about
    1.44 per standard deviation:

    >>> from surpyval.test_planning import cox_detectable
    >>> round(cox_detectable(200, 0.3, sd=1.0), 3)
    1.436
    """
    units = check_count(n, "n")
    probability = check_probability(event_probability, "event_probability")
    power = check_probability(power, "power")
    check_option("direction", direction, DIRECTIONS)
    variance = _covariate_variance(sd, proportion, r_squared)
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    log_ratio = z / np.sqrt(units * probability * variance)
    return float(np.exp(log_ratio if direction == "increase" else -log_ratio))


# --- Non-inferiority and equivalence of a hazard ratio ----------------------


def _check_margin_above(margin: float, hazard_ratio: float) -> tuple:
    margin = check_positive(margin, "margin")
    hazard_ratio = check_positive(hazard_ratio, "hazard_ratio")
    if not hazard_ratio < margin:
        raise ValueError(
            f"hazard_ratio ({hazard_ratio!r}) must be below margin "
            f"({margin!r}): a true hazard ratio at or above the margin "
            "cannot be shown to be below it. Lower hazard_ratio or raise "
            "margin."
        )
    return margin, hazard_ratio


def _share(ratio: float) -> float:
    ratio = _check_ratio(ratio)
    return ratio / (1.0 + ratio)


@elementwise
def noninferiority_events(
    margin: float,
    power: float = 0.8,
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
) -> int:
    r"""The events needed to show that a hazard ratio is below a margin
    (non-inferiority).

    To show that a new design, procedure or treatment (the first group)
    is not worse than the standard (the second) by more than a margin:
    the null hypothesis is that the hazard ratio, first group to second,
    is ``margin`` or more, and it is rejected when the upper limit of the
    ``1 - alpha_ci`` confidence interval of the hazard ratio is below
    ``margin`` (a one-sided test at level ``alpha_ci / 2``). With the true
    hazard ratio ``hazard_ratio`` (1, by default: the two are in fact
    equally good), Schoenfeld's formula with the margin in place of 1,

    .. math::
        d = \frac{(z_{1 - \alpha / 2} + z_{\text{power}})^2}
                 {\pi (1 - \pi) (\ln \text{margin} - \ln \text{HR})^2},

    as ``gsDesign::nEvents(hr0=margin)`` and ``rpact``'s
    ``getSampleSizeSurvival(thetaH0=margin)`` give.

    A margin below 1 asks for superiority by a margin: that the new
    group's hazard is at most ``margin`` times the standard's.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    margin : float
        The hazard ratio, first group to second, that would be
        unacceptable (above 1 for non-inferiority, such as 1.3).
    power : float, optional
        The chance of showing it, by default 0.8.
    hazard_ratio : float, optional
        Keyword only: the true hazard ratio, below ``margin``; by default
        1.0.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the interval
        compared with the margin, by default 0.05 (a 95% interval, a
        one-sided test at 2.5%).
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0.

    Returns
    -------
    int
        The number of events, in both groups together, rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``hazard_ratio`` is not below
        ``margin``.

    Examples
    --------
    Showing that a cheaper part raises the hazard by no more than 30%,
    when it is in fact as good:

    >>> from surpyval.test_planning import noninferiority_events
    >>> noninferiority_events(1.3)
    457
    >>> noninferiority_events(1.3, hazard_ratio=0.9)
    233
    """
    margin, hazard_ratio = _check_margin_above(margin, hazard_ratio)
    power = check_probability(power, "power")
    share = _share(ratio)
    z = _z_alpha(alpha_ci, "two-sided") + float(stats.norm.ppf(power))
    distance = np.log(margin) - np.log(hazard_ratio)
    events = z**2 / (share * (1.0 - share) * distance**2)
    return int(np.ceil(events - 1e-9))


def _per_second(
    events: float, ratio: float, first: float, second: float
) -> tuple[int, int]:
    """The units in each group, rounded up, for the expected events
    ``events``."""
    per_second = events / (ratio * first + second)
    return (
        int(np.ceil(ratio * per_second - 1e-9)),
        int(np.ceil(per_second - 1e-9)),
    )


def _check_pair_n(n: Any) -> tuple[int, int]:
    sizes = tuple(np.asarray(n).ravel())
    if len(sizes) != 2:
        raise ValueError(
            f"n must be the units in the two groups, a pair, got {n!r}."
        )
    return check_count(sizes[0], "n"), check_count(sizes[1], "n")


def noninferiority_sample_size(
    margin: float,
    event_probability: float | tuple[float, float],
    power: float = 0.8,
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
) -> tuple[int, int]:
    """The units in each group needed to show that a hazard ratio is below
    a margin.

    The events of :func:`noninferiority_events`, unrounded, shared out in
    the ratio ``ratio`` and divided by each group's chance of an event
    during the study, as :func:`~surpyval.test_planning.logrank_sample_size`
    does.

    Parameters
    ----------
    margin : float
        The hazard ratio, first group to second, that would be
        unacceptable.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair, the first group's then the second's.
    power : float, optional
        The chance of showing it, by default 0.8.
    hazard_ratio, alpha_ci, ratio
        Keyword only: as for :func:`noninferiority_events`.

    Returns
    -------
    tuple of int
        The units in the first group and in the second, each rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import noninferiority_sample_size
    >>> noninferiority_sample_size(1.3, 0.4)
    (571, 571)
    """
    check_alpha_ci(alpha_ci)
    margin, hazard_ratio = _check_margin_above(margin, hazard_ratio)
    power = check_probability(power, "power")
    share = _share(ratio)
    first, second = _event_probabilities(event_probability)
    z = _z_alpha(alpha_ci, "two-sided") + float(stats.norm.ppf(power))
    distance = np.log(margin) - np.log(hazard_ratio)
    events = z**2 / (share * (1.0 - share) * distance**2)
    return _per_second(events, ratio, first, second)


def noninferiority_power(
    margin: float,
    n: tuple[int, int],
    event_probability: float | tuple[float, float],
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
) -> float:
    r"""The chance of showing that a hazard ratio is below a margin, given
    the units in each group.

    :math:`\Phi(\sqrt{d \pi (1 - \pi)} (\ln \text{margin} - \ln
    \text{HR}) - z_{1 - \alpha / 2})`, with :math:`d` the expected events,
    ``n1 * p1 + n2 * p2``, and :math:`\pi = n_1 / (n_1 + n_2)`.

    Parameters
    ----------
    margin : float
        The hazard ratio, first group to second, that would be
        unacceptable.
    n : (int, int)
        The units in the first group and in the second.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair.
    hazard_ratio, alpha_ci
        Keyword only: as for :func:`noninferiority_events`.

    Returns
    -------
    float
        The power.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import noninferiority_power
    >>> round(noninferiority_power(1.3, (571, 571), 0.4), 3)
    0.801
    """
    check_alpha_ci(alpha_ci)
    margin, hazard_ratio = _check_margin_above(margin, hazard_ratio)
    first_n, second_n = _check_pair_n(n)
    first, second = _event_probabilities(event_probability)
    share = first_n / (first_n + second_n)
    events = first_n * first + second_n * second
    distance = np.log(margin) - np.log(hazard_ratio)
    z = np.sqrt(events * share * (1.0 - share)) * distance
    return float(stats.norm.cdf(z - _z_alpha(alpha_ci, "two-sided")))


def _equivalence_margins(margin: Any, hazard_ratio: float) -> tuple:
    """The log margins, lower and upper, and the log hazard ratio."""
    if np.ndim(margin) == 0:
        upper = check_positive(margin, "margin")
        if upper <= 1.0:
            raise ValueError(
                f"margin must be above 1, got {margin!r}: a single margin m "
                "means the interval (1/m, m). Give a pair (lower, upper) "
                "for other margins."
            )
        lower = 1.0 / upper
    else:
        pair = tuple(np.asarray(margin, dtype=float).ravel())
        if len(pair) != 2:
            raise ValueError(
                "margin must be one hazard ratio above 1, or a pair "
                f"(lower, upper), got {margin!r}."
            )
        lower = check_positive(pair[0], "margin")
        upper = check_positive(pair[1], "margin")
    hazard_ratio = check_positive(hazard_ratio, "hazard_ratio")
    if not lower < hazard_ratio < upper:
        raise ValueError(
            f"hazard_ratio ({hazard_ratio!r}) must be strictly between the "
            f"margins ({lower!r}, {upper!r}): a true hazard ratio outside "
            "them cannot be shown to be inside. Widen margin or move "
            "hazard_ratio."
        )
    return np.log(lower), np.log(upper), np.log(hazard_ratio)


def _equivalence_power_at(
    information: float, lower: float, upper: float, theta: float, z: float
) -> float:
    """The chance that the interval ``theta_hat +- z se`` lies inside
    ``(lower, upper)``, with ``se = 1 / sqrt(information)``."""
    root = np.sqrt(information)
    inside = stats.norm.cdf((upper - theta) * root - z) - stats.norm.cdf(
        (lower - theta) * root + z
    )
    return float(max(inside, 0.0))


def _equivalence_events_unrounded(
    lower: float, upper: float, theta: float, z: float, power: float, pq: float
) -> float:
    def gap(events: float) -> float:
        return (
            _equivalence_power_at(events * pq, lower, upper, theta, z) - power
        )

    # Below this many events the interval cannot fit inside the margins.
    start = (2.0 * z / (upper - lower)) ** 2 / pq
    high = 2.0 * start
    while gap(high) < 0.0:
        high *= 2.0
    return float(optimize.brentq(gap, start, high, xtol=1e-10, rtol=1e-14))


def equivalence_events(
    margin: float | tuple[float, float],
    power: float = 0.8,
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
) -> int:
    r"""The events needed to show that a hazard ratio is between two
    margins (equivalence).

    Two one-sided tests: equivalence is shown when the ``1 - alpha_ci``
    confidence interval of the hazard ratio lies inside the margins,
    ``(1 / margin, margin)`` or the pair given. Its power, with
    :math:`\theta` the true log hazard ratio, :math:`L` and :math:`U` the
    log margins and :math:`\text{se} = 1 / \sqrt{d \pi (1 - \pi)}`, is

    .. math::
        \Phi\!\left(\frac{U - \theta}{\text{se}} - z_{1 - \alpha/2}\right)
        - \Phi\!\left(\frac{L - \theta}{\text{se}} + z_{1 - \alpha/2}\right),

    solved for the events :math:`d`. With :math:`\theta = 0` and
    symmetric margins this is
    :math:`d = (z_{1 - \alpha / 2} + z_{1 - \beta / 2})^2 /
    (\pi (1 - \pi) (\ln \text{margin})^2)` (Chow, Shao and Wang 2008).

    The classical two one-sided tests at 5% each compare a 90% interval:
    ``alpha_ci=0.1``.

    Parameters
    ----------
    margin : float or (float, float)
        The equivalence margins of the hazard ratio, first group to
        second: one above 1, for ``(1 / margin, margin)``, or a pair
        ``(lower, upper)``.
    power : float, optional
        The chance of showing it, by default 0.8.
    hazard_ratio : float, optional
        Keyword only: the true hazard ratio, strictly between the
        margins; by default 1.0.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the interval
        compared with the margins, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0.

    Returns
    -------
    int
        The number of events, in both groups together, rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``hazard_ratio`` is not inside
        the margins.

    Examples
    --------
    >>> from surpyval.test_planning import equivalence_events
    >>> equivalence_events(1.25)
    845
    >>> equivalence_events(1.25, alpha_ci=0.1)
    688
    """
    check_alpha_ci(alpha_ci)
    lower, upper, theta = _equivalence_margins(margin, hazard_ratio)
    power = check_probability(power, "power")
    share = _share(ratio)
    z = _z_alpha(alpha_ci, "two-sided")
    events = _equivalence_events_unrounded(
        lower, upper, theta, z, power, share * (1.0 - share)
    )
    return int(np.ceil(events - 1e-9))


def equivalence_sample_size(
    margin: float | tuple[float, float],
    event_probability: float | tuple[float, float],
    power: float = 0.8,
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
) -> tuple[int, int]:
    """The units in each group needed to show that a hazard ratio is
    between two margins.

    The events of :func:`equivalence_events`, unrounded, shared out in the
    ratio ``ratio`` and divided by each group's chance of an event during
    the study.

    Parameters
    ----------
    margin : float or (float, float)
        The equivalence margins: one above 1, or a pair.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair.
    power : float, optional
        The chance of showing it, by default 0.8.
    hazard_ratio, alpha_ci, ratio
        Keyword only: as for :func:`equivalence_events`.

    Returns
    -------
    tuple of int
        The units in the first group and in the second, each rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import equivalence_sample_size
    >>> equivalence_sample_size(1.25, 0.5, alpha_ci=0.1)
    (688, 688)
    """
    check_alpha_ci(alpha_ci)
    lower, upper, theta = _equivalence_margins(margin, hazard_ratio)
    power = check_probability(power, "power")
    share = _share(ratio)
    first, second = _event_probabilities(event_probability)
    z = _z_alpha(alpha_ci, "two-sided")
    events = _equivalence_events_unrounded(
        lower, upper, theta, z, power, share * (1.0 - share)
    )
    return _per_second(events, ratio, first, second)


def equivalence_power(
    margin: float | tuple[float, float],
    n: tuple[int, int],
    event_probability: float | tuple[float, float],
    *,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
) -> float:
    """The chance of showing that a hazard ratio is between two margins,
    given the units in each group.

    The power of :func:`equivalence_events`, with the expected events
    ``n1 * p1 + n2 * p2``.

    Parameters
    ----------
    margin : float or (float, float)
        The equivalence margins: one above 1, or a pair.
    n : (int, int)
        The units in the first group and in the second.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair.
    hazard_ratio, alpha_ci
        Keyword only: as for :func:`equivalence_events`.

    Returns
    -------
    float
        The power.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import equivalence_power
    >>> round(equivalence_power(1.25, (688, 688), 0.5, alpha_ci=0.1), 3)
    0.8
    """
    check_alpha_ci(alpha_ci)
    lower, upper, theta = _equivalence_margins(margin, hazard_ratio)
    first_n, second_n = _check_pair_n(n)
    first, second = _event_probabilities(event_probability)
    share = first_n / (first_n + second_n)
    events = first_n * first + second_n * second
    z = _z_alpha(alpha_ci, "two-sided")
    return _equivalence_power_at(
        events * share * (1.0 - share), lower, upper, theta, z
    )


# --- The study: entry, follow-up and dropout --------------------------------


def _check_study(
    accrual_time: float, follow_up: float, dropout_rate: float
) -> tuple[float, float, float]:
    try:
        accrual_ok = bool(np.isfinite(accrual_time) and accrual_time >= 0.0)
    except (TypeError, ValueError):
        accrual_ok = False
    if not accrual_ok:
        raise ValueError(
            "accrual_time, the time over which units enter (uniformly), "
            f"must be 0 or more, got {accrual_time!r}; use 0 when all "
            "enter at once."
        )
    follow_up = check_positive(follow_up, "follow_up")
    try:
        dropout_ok = bool(np.isfinite(dropout_rate) and dropout_rate >= 0.0)
    except (TypeError, ValueError):
        dropout_ok = False
    if not dropout_ok:
        raise ValueError(
            "dropout_rate, the hazard of a unit leaving the study early, "
            f"must be 0 or more, got {dropout_rate!r}."
        )
    return float(accrual_time), follow_up, float(dropout_rate)


def _grid(accrual: float, follow_up: float, end: float) -> npt.NDArray:
    """Integration points from 0 to ``end``, with a node at the end of
    the minimum follow-up, where the censoring distribution has a kink."""
    if end <= follow_up or accrual == 0.0:
        return np.linspace(0.0, end, 2 * _GRID + 1)
    return np.concatenate(
        [
            np.linspace(0.0, follow_up, _GRID + 1),
            np.linspace(follow_up, end, _GRID + 1)[1:],
        ]
    )


def _uncensored(
    t: npt.NDArray, accrual: float, follow_up: float, dropout: float
) -> npt.NDArray:
    """The chance a unit is still under observation at ``t`` since its
    entry, G(t)."""
    if accrual == 0.0:
        administrative = np.where(t <= follow_up, 1.0, 0.0)
    else:
        administrative = np.clip((accrual + follow_up - t) / accrual, 0, 1)
    return administrative * np.exp(-dropout * t)


def _survival_function(curve: SurvivalCurve, name: str) -> Callable:
    """The survival function of ``curve``: its ``sf`` method, or itself
    if it is callable."""
    if hasattr(curve, "sf") and callable(curve.sf):
        return curve.sf
    if callable(curve):
        return curve
    raise ValueError(
        f"{name} must be a distribution with an sf method (such as "
        "surpyval.Weibull.from_params([alpha, beta])) or a function of "
        f"time returning the survival probability, got {curve!r}."
    )


def _evaluate(sf: Callable, t: npt.NDArray, name: str) -> npt.NDArray:
    values = np.asarray(sf(t), dtype=float).reshape(t.shape)
    bad = ~np.isfinite(values) | (values < -1e-12) | (values > 1 + 1e-12)
    if bad.any():
        raise ValueError(
            f"{name} must give survival probabilities between 0 and 1; "
            f"it gave {values[bad][0]!r} at time {t[bad][0]!r}."
        )
    if np.any(np.diff(values) > 1e-10):
        raise ValueError(
            f"{name} must give a survival curve that never rises; check "
            "its parameters."
        )
    return np.clip(values, 0.0, 1.0)


def _arms(survival: Any) -> tuple[Callable, Callable]:
    if isinstance(survival, (str, bytes)) or not hasattr(survival, "__len__"):
        raise ValueError(
            "survival must be a pair: the first group's survival curve and "
            f"the second's, got {survival!r}."
        )
    if len(survival) != 2:
        raise ValueError(
            "survival must be a pair: the first group's survival curve and "
            f"the second's, got {len(survival)} curves."
        )
    return (
        _survival_function(survival[0], "survival[0]"),
        _survival_function(survival[1], "survival[1]"),
    )


# --- Restricted mean survival time ------------------------------------------


def _rmst_moments(
    sf: Callable,
    tau: float,
    accrual: float,
    follow_up: float,
    dropout: float,
    name: str,
) -> tuple[float, float]:
    r"""The RMST to ``tau`` and the asymptotic variance of ``sqrt(n)``
    times its Kaplan-Meier estimate,
    :math:`\int_0^\tau A(t)^2 dF(t) / (S(t)^2 G(t))`, with
    :math:`A(t) = \int_t^\tau S(u) du`."""
    t = _grid(accrual, follow_up, tau)
    s = _evaluate(sf, t, name)
    widths = np.diff(t)
    pieces = 0.5 * (s[1:] + s[:-1]) * widths  # trapezoid
    rmst = float(pieces.sum())
    # A at the nodes, from tau backwards.
    tail = np.concatenate([np.cumsum(pieces[::-1])[::-1], [0.0]])
    middle = 0.5 * (t[1:] + t[:-1])
    a_mid = 0.5 * (tail[1:] + tail[:-1])
    s_mid = 0.5 * (s[1:] + s[:-1])
    g_mid = _uncensored(middle, accrual, follow_up, dropout)
    dead = -np.diff(s)
    weight = np.zeros_like(dead)
    used = (dead > 0.0) & (s_mid > 0.0) & (g_mid > 0.0)
    weight[used] = a_mid[used] ** 2 / (s_mid[used] ** 2 * g_mid[used])
    return rmst, float(np.sum(weight * dead))


def _rmst_setup(
    survival: Any,
    tau: float,
    accrual_time: float,
    follow_up: float | None,
    dropout_rate: float,
) -> tuple[float, float, float]:
    """The difference in RMST, first group minus second, and the two
    variances."""
    first, second = _arms(survival)
    tau = check_positive(tau, "tau")
    accrual, follow, dropout = _check_study(
        accrual_time, tau if follow_up is None else follow_up, dropout_rate
    )
    if tau > accrual + follow:
        raise ValueError(
            f"tau ({tau!r}) is beyond the end of the study (accrual_time + "
            f"follow_up = {accrual + follow!r}): no unit is observed then. "
            "Lower tau or lengthen follow_up."
        )
    mean_1, var_1 = _rmst_moments(
        first, tau, accrual, follow, dropout, "survival[0]"
    )
    mean_2, var_2 = _rmst_moments(
        second, tau, accrual, follow, dropout, "survival[1]"
    )
    difference = mean_1 - mean_2
    if abs(difference) <= 1e-9 * tau:
        raise ValueError(
            "The two survival curves have the same RMST to tau: no test can "
            "detect a difference of none. Check the curves, or choose "
            "another tau."
        )
    return difference, var_1, var_2


def rmst_sample_size(
    survival: tuple[SurvivalCurve, SurvivalCurve],
    tau: float,
    power: float = 0.8,
    *,
    accrual_time: float = 0.0,
    follow_up: float | None = None,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> tuple[int, int]:
    r"""The units in each group needed to detect a difference in
    restricted mean survival time.

    The restricted mean survival time (RMST) to a horizon :math:`\tau` is
    the expected time without the event up to :math:`\tau`, the area
    under the survival curve: "on average, over the first five years, a
    unit runs 4.1 years without failing". A difference in RMST between
    two groups needs no proportional hazards, and it reads in units of
    time, so it is the comparison to plan when the hazards are expected
    to cross or converge (a delayed effect, a cured fraction), where
    the hazard ratio of the log-rank calculation is not one number.

    The test is of the difference of the two groups' Kaplan-Meier RMSTs
    (:func:`surpyval.rmst_diff`), divided by its standard error. Each
    group's estimate has asymptotic variance :math:`\sigma_j^2 / n_j`,

    .. math::
        \sigma_j^2 = \int_0^\tau
            \frac{\left(\int_t^\tau S_j(u)\,du\right)^2}
                 {S_j(t)^2\, G(t)}\, dF_j(t),

    computed here from the curves given and the study's censoring
    :math:`G` (uniform entry over ``accrual_time``, the study ending
    ``follow_up`` after the last entry, and dropout at the rate
    ``dropout_rate``), so the units in the second group are

    .. math::
        n_2 = \frac{(z_{1 - \alpha / 2} + z_{\text{power}})^2
                    (\sigma_1^2 / r + \sigma_2^2)}{\Delta^2},

    with :math:`r` the ratio and :math:`\Delta` the difference in RMST
    (Royston and Parmar 2013; R ``npsurvSS``'s ``"rmst difference"``
    test).

    Parameters
    ----------
    survival : (curve, curve)
        The survival curves expected in the first group and in the
        second: each a distribution with an ``sf`` method (such as
        ``surpyval.Weibull.from_params([alpha, beta])``, any SurPyval
        distribution, fitted or with given parameters, or a custom one
        such as a piecewise exponential) or a function of time returning
        the survival probability.
    tau : float
        The horizon of the RMST, no later than the end of the study,
        ``accrual_time + follow_up``.
    power : float, optional
        The chance of detecting the difference, by default 0.8.
    accrual_time : float, optional
        Keyword only: the time over which units enter, uniformly; by
        default 0 (all at once).
    follow_up : float, optional
        Keyword only: the time from the last entry to the end of the
        study; by default ``tau``, so that every unit is followed to
        ``tau`` unless it drops out.
    dropout_rate : float, optional
        Keyword only: the hazard of a unit leaving the study early (an
        exponential dropout), by default 0.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    tuple of int
        The units in the first group and in the second, each rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range, ``tau`` is beyond the end of the
        study, or the two curves have the same RMST to ``tau``.

    Examples
    --------
    Exponential lives with means 10 and 6 years, compared over the first
    5 years, units entering over 2 years and followed for 4 more:

    >>> import surpyval
    >>> from surpyval.test_planning import rmst_sample_size
    >>> curves = (surpyval.Exponential.from_params([1 / 10]),
    ...           surpyval.Exponential.from_params([1 / 6]))
    >>> rmst_sample_size(curves, 5.0, accrual_time=2.0, follow_up=4.0)
    (152, 152)
    """
    check_alpha_ci(alpha_ci)
    power = check_probability(power, "power")
    ratio = _check_ratio(ratio)
    difference, var_1, var_2 = _rmst_setup(
        survival, tau, accrual_time, follow_up, dropout_rate
    )
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    per_second = z**2 * (var_1 / ratio + var_2) / difference**2
    return (
        int(np.ceil(ratio * per_second - 1e-9)),
        int(np.ceil(per_second - 1e-9)),
    )


def rmst_power(
    survival: tuple[SurvivalCurve, SurvivalCurve],
    tau: float,
    n: tuple[int, int],
    *,
    accrual_time: float = 0.0,
    follow_up: float | None = None,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    r"""The chance of detecting a difference in restricted mean survival
    time, given the units in each group.

    :math:`\Phi(|\Delta| / \sqrt{\sigma_1^2 / n_1 + \sigma_2^2 / n_2} -
    z_{1 - \alpha / 2})`, with the variances of
    :func:`rmst_sample_size`.

    Parameters
    ----------
    survival : (curve, curve)
        The survival curves expected in the first group and in the
        second, as for :func:`rmst_sample_size`.
    tau : float
        The horizon of the RMST.
    n : (int, int)
        The units in the first group and in the second.
    accrual_time, follow_up, dropout_rate, alpha_ci, alternative
        Keyword only: as for :func:`rmst_sample_size`.

    Returns
    -------
    float
        The power.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.test_planning import rmst_power
    >>> curves = (surpyval.Exponential.from_params([1 / 10]),
    ...           surpyval.Exponential.from_params([1 / 6]))
    >>> round(rmst_power(curves, 5.0, (152, 152), accrual_time=2.0,
    ...                  follow_up=4.0), 3)
    0.8
    """
    check_alpha_ci(alpha_ci)
    first_n, second_n = _check_pair_n(n)
    difference, var_1, var_2 = _rmst_setup(
        survival, tau, accrual_time, follow_up, dropout_rate
    )
    se = np.sqrt(var_1 / first_n + var_2 / second_n)
    z = abs(difference) / se - _z_alpha(alpha_ci, alternative)
    return float(stats.norm.cdf(z))


# --- The log-rank test when some units are cured ----------------------------


def _cure_curves(
    cured: Any, latency: Any, latency_hazard_ratio: float
) -> tuple[Callable, Callable]:
    """The two groups' survival functions,
    ``cured + (1 - cured) * S_latency(t)``."""
    if np.ndim(cured) == 0:
        cure = (float(cured), float(cured))
    else:
        pair = tuple(np.asarray(cured, dtype=float).ravel())
        if len(pair) != 2:
            raise ValueError(
                "cured must be one cured fraction for both groups, or a "
                f"pair, the first group's then the second's, got {cured!r}."
            )
        cure = (float(pair[0]), float(pair[1]))
    for fraction in cure:
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                "cured, the fraction of units that never have the event, "
                f"must be at least 0 and below 1, got {fraction!r}."
            )
    hr = check_positive(latency_hazard_ratio, "latency_hazard_ratio")
    is_pair = isinstance(latency, (tuple, list))
    if is_pair:
        if len(latency) != 2:
            raise ValueError(
                "latency must be one distribution for both groups, or a "
                f"pair, got {len(latency)} distributions."
            )
        if hr != 1.0:
            raise ValueError(
                "latency_hazard_ratio applies to one latency distribution "
                "shared by the groups; with a pair, put the difference in "
                "the pair and leave latency_hazard_ratio at 1."
            )
        latency_1 = _survival_function(latency[0], "latency[0]")
        latency_2 = _survival_function(latency[1], "latency[1]")
        same_latency = latency[0] is latency[1]
    else:
        base = _survival_function(latency, "latency")
        latency_2 = base

        def latency_1(t: Any) -> Any:
            return np.asarray(base(t), dtype=float) ** hr

        same_latency = hr == 1.0
    if same_latency and cure[0] == cure[1]:
        raise ValueError(
            "The two groups have the same cured fraction and the same "
            "latency: no test can detect a difference of none. Give "
            "different cured fractions, a latency_hazard_ratio, or a pair "
            "of latency distributions."
        )

    def sf_1(t: Any) -> Any:
        return cure[0] + (1.0 - cure[0]) * np.asarray(latency_1(t), float)

    def sf_2(t: Any) -> Any:
        return cure[1] + (1.0 - cure[1]) * np.asarray(latency_2(t), float)

    return sf_1, sf_2


def _logrank_moments(
    sf_1: Callable,
    sf_2: Callable,
    share: float,
    accrual: float,
    follow_up: float,
    dropout: float,
) -> tuple[float, float, float, float]:
    r"""Per unit in the study: the mean :math:`\mu` and variance
    :math:`\sigma^2` of the log-rank statistic's numerator under the
    alternative (Schoenfeld 1981), and each group's chance of an event.

    .. math::
        \mu = \int p_1 p_2 G \frac{S_2\, dF_1 - S_1\, dF_2}
                                 {p_1 S_1 + p_2 S_2}, \qquad
        \sigma^2 = \int p_1 p_2 G \frac{S_1 S_2 (p_1\, dF_1 + p_2\, dF_2)}
                                      {(p_1 S_1 + p_2 S_2)^2}.
    """
    end = accrual + follow_up
    t = _grid(accrual, follow_up, end)
    s_1 = _evaluate(sf_1, t, "the first group's survival")
    s_2 = _evaluate(sf_2, t, "the second group's survival")
    df_1, df_2 = -np.diff(s_1), -np.diff(s_2)
    m_1, m_2 = 0.5 * (s_1[1:] + s_1[:-1]), 0.5 * (s_2[1:] + s_2[:-1])
    g = _uncensored(0.5 * (t[1:] + t[:-1]), accrual, follow_up, dropout)
    p_1, p_2 = share, 1.0 - share
    pooled = p_1 * m_1 + p_2 * m_2
    pooled = np.where(pooled > 0.0, pooled, np.inf)
    mean = np.sum(p_1 * p_2 * g * (m_2 * df_1 - m_1 * df_2) / pooled)
    variance = np.sum(
        p_1 * p_2 * g * m_1 * m_2 * (p_1 * df_1 + p_2 * df_2) / pooled**2
    )
    return (
        float(mean),
        float(variance),
        float(np.sum(g * df_1)),
        float(np.sum(g * df_2)),
    )


def _cure_setup(
    cured: Any,
    latency: Any,
    latency_hazard_ratio: float,
    share: float,
    accrual_time: float,
    follow_up: float,
    dropout_rate: float,
) -> tuple[float, float, float, float]:
    sf_1, sf_2 = _cure_curves(cured, latency, latency_hazard_ratio)
    accrual, follow, dropout = _check_study(
        accrual_time, follow_up, dropout_rate
    )
    mean, variance, events_1, events_2 = _logrank_moments(
        sf_1, sf_2, share, accrual, follow, dropout
    )
    if abs(mean) <= 1e-12:
        raise ValueError(
            "Over the study the log-rank test has no power against these "
            "curves (its expected statistic is 0): the hazards cross so "
            "that the differences cancel. Compare restricted mean survival "
            "times (rmst_sample_size) or lengthen follow_up."
        )
    return mean, variance, events_1, events_2


def _cure_total(
    mean: float,
    variance: float,
    alpha_ci: float,
    alternative: str,
    power: float,
) -> float:
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    return z**2 * variance / mean**2


def cure_logrank_sample_size(
    cured: float | tuple[float, float],
    latency: Any,
    power: float = 0.8,
    *,
    latency_hazard_ratio: float = 1.0,
    follow_up: float,
    accrual_time: float = 0.0,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> tuple[int, int]:
    r"""The units in each group a log-rank test needs when some units are
    cured.

    When a fraction of the units never have the event (cured, or immune:
    a defect-free unit that never fails of this mode, a patient cured by
    treatment), each group's survival levels off at its cured fraction,
    :math:`S_j(t) = \pi_j + (1 - \pi_j) S_{u,j}(t)`, with
    :math:`S_{u,j}` the latency distribution, the survival of the units
    that are not cured. The hazards are then not proportional: a
    difference in the cured fraction gives a hazard ratio that drifts
    towards 1 as the uncured fail, so Schoenfeld's events for a single
    hazard ratio (:func:`~surpyval.test_planning.logrank_sample_size`)
    are wrong, too few or too many depending on the follow-up.

    The units here come from the asymptotic mean :math:`\mu` and variance
    :math:`\sigma^2` of the log-rank statistic under the alternative given
    (Schoenfeld 1981), integrated over the study's censoring:
    :math:`n = (z_{1 - \alpha / 2} + z_{\text{power}})^2 \sigma^2 /
    \mu^2`, in all, shared in the ratio ``ratio``. A difference in the
    cured fraction, in the latency (``latency_hazard_ratio``, or a pair of
    latency distributions), or both, may be given.

    Parameters
    ----------
    cured : float or (float, float)
        The cured fraction, at least 0 and below 1: one for both groups,
        or a pair, the first group's then the second's.
    latency : distribution or (distribution, distribution)
        The survival of the units not cured: a distribution with an
        ``sf`` method (``surpyval.Weibull.from_params([alpha, beta])``,
        say) or a function of time, one for both groups or a pair.
    power : float, optional
        The chance of detecting the difference, by default 0.8.
    latency_hazard_ratio : float, optional
        Keyword only: with one latency distribution, the hazard ratio of
        the first group's uncured units to the second's (proportional
        hazards among the uncured, as in the proportional hazards cure
        model); by default 1.
    follow_up : float
        Keyword only: the time from the last entry to the end of the
        study.
    accrual_time : float, optional
        Keyword only: the time over which units enter, uniformly; by
        default 0 (all at once).
    dropout_rate : float, optional
        Keyword only: the hazard of a unit leaving the study early, by
        default 0.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    tuple of int
        The units in the first group and in the second, each rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range, the groups do not differ, or the
        log-rank test has no power against them.

    Examples
    --------
    A process change expected to raise the fraction of units that never
    fail of a mode from 30% to 45%, the others failing as a Weibull with
    scale 2 years and shape 1.2; units enter over a year and are followed
    for 3 more:

    >>> import surpyval
    >>> from surpyval.test_planning import cure_logrank_sample_size
    >>> latency = surpyval.Weibull.from_params([2.0, 1.2])
    >>> cure_logrank_sample_size((0.45, 0.30), latency, accrual_time=1.0,
    ...                          follow_up=3.0)
    (247, 247)
    """
    check_alpha_ci(alpha_ci)
    power = check_probability(power, "power")
    ratio = _check_ratio(ratio)
    share = ratio / (1.0 + ratio)
    mean, variance, _, _ = _cure_setup(
        cured,
        latency,
        latency_hazard_ratio,
        share,
        accrual_time,
        follow_up,
        dropout_rate,
    )
    total = _cure_total(mean, variance, alpha_ci, alternative, power)
    return (
        int(np.ceil(share * total - 1e-9)),
        int(np.ceil((1.0 - share) * total - 1e-9)),
    )


def cure_logrank_events(
    cured: float | tuple[float, float],
    latency: Any,
    power: float = 0.8,
    *,
    latency_hazard_ratio: float = 1.0,
    follow_up: float,
    accrual_time: float = 0.0,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """The events a log-rank test needs when some units are cured.

    The expected events, in both groups, at the units of
    :func:`cure_logrank_sample_size` (unrounded): the number to wait for
    in a study that runs until a count of events. Unlike Schoenfeld's,
    it depends on the follow-up, as the hazard ratio drifts with time.

    Parameters
    ----------
    cured, latency, power
        As for :func:`cure_logrank_sample_size`.
    latency_hazard_ratio, follow_up, accrual_time, dropout_rate
        Keyword only: as for :func:`cure_logrank_sample_size`.
    alpha_ci, ratio, alternative
        Keyword only: as for :func:`cure_logrank_sample_size`.

    Returns
    -------
    int
        The number of events, rounded up.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.test_planning import cure_logrank_events
    >>> latency = surpyval.Weibull.from_params([2.0, 1.2])
    >>> cure_logrank_events((0.45, 0.30), latency, accrual_time=1.0,
    ...                     follow_up=3.0)
    265
    """
    check_alpha_ci(alpha_ci)
    power = check_probability(power, "power")
    ratio = _check_ratio(ratio)
    share = ratio / (1.0 + ratio)
    mean, variance, events_1, events_2 = _cure_setup(
        cured,
        latency,
        latency_hazard_ratio,
        share,
        accrual_time,
        follow_up,
        dropout_rate,
    )
    total = _cure_total(mean, variance, alpha_ci, alternative, power)
    events = total * (share * events_1 + (1.0 - share) * events_2)
    return int(np.ceil(events - 1e-9))


def cure_logrank_power(
    cured: float | tuple[float, float],
    latency: Any,
    n: tuple[int, int],
    *,
    latency_hazard_ratio: float = 1.0,
    follow_up: float,
    accrual_time: float = 0.0,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    r"""The chance that a log-rank test detects a difference when some
    units are cured, given the units in each group.

    :math:`\Phi(\sqrt{n} |\mu| / \sigma - z_{1 - \alpha / 2})`, with the
    mean and variance of :func:`cure_logrank_sample_size` at the groups'
    shares of the ``n1 + n2`` units.

    Parameters
    ----------
    cured, latency
        As for :func:`cure_logrank_sample_size`.
    n : (int, int)
        The units in the first group and in the second.
    latency_hazard_ratio, follow_up, accrual_time, dropout_rate
        Keyword only: as for :func:`cure_logrank_sample_size`.
    alpha_ci, alternative
        Keyword only: as for :func:`cure_logrank_sample_size`.

    Returns
    -------
    float
        The power.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.test_planning import cure_logrank_power
    >>> latency = surpyval.Weibull.from_params([2.0, 1.2])
    >>> round(cure_logrank_power((0.45, 0.30), latency, (247, 247),
    ...                          accrual_time=1.0, follow_up=3.0), 3)
    0.8
    """
    check_alpha_ci(alpha_ci)
    first_n, second_n = _check_pair_n(n)
    total = first_n + second_n
    mean, variance, _, _ = _cure_setup(
        cured,
        latency,
        latency_hazard_ratio,
        first_n / total,
        accrual_time,
        follow_up,
        dropout_rate,
    )
    z = np.sqrt(total) * abs(mean) / np.sqrt(variance)
    return float(stats.norm.cdf(z - _z_alpha(alpha_ci, alternative)))
