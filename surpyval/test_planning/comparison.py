"""Power and sample size for comparing two groups' survival (#880): how
many events, and how many units in each group, a log-rank (or Cox) test
needs to detect a hazard ratio, under proportional hazards.

The events needed are Schoenfeld's (1983) by default, from the
asymptotic distribution of the log-rank statistic; Freedman's (1982)
formula, lifelines' ``sample_size_necessary_under_cph``, is the option
``method="freedman"``. Both count events, not units: the units follow
from each group's chance of having an event during the study.
"""

import numpy as np
from scipy import stats

from surpyval.test_planning._checks import (
    check_count,
    check_positive,
    check_probability,
    elementwise,
)
from surpyval.utils.validation import check_alpha_ci, check_option

METHODS = ("schoenfeld", "freedman")
ALTERNATIVES = ("two-sided", "one-sided")


def _z_alpha(alpha_ci: float, alternative: str) -> float:
    check_option("alternative", alternative, ALTERNATIVES)
    tail = alpha_ci / 2.0 if alternative == "two-sided" else alpha_ci
    return float(stats.norm.isf(tail))


def _check_hazard_ratio(hazard_ratio: float) -> float:
    hazard_ratio = check_positive(hazard_ratio, "hazard_ratio")
    if hazard_ratio == 1.0:
        raise ValueError(
            "hazard_ratio must differ from 1: no test can detect a "
            "difference of none."
        )
    return hazard_ratio


def _check_ratio(ratio: float) -> float:
    return check_positive(ratio, "ratio")


def _events(hazard_ratio: float, z: float, ratio: float, method: str) -> float:
    """The events, unrounded, for a sum ``z`` of the two normal
    quantiles."""
    check_option("method", method, METHODS)
    if method == "schoenfeld":
        share = ratio / (1.0 + ratio)
        return z**2 / (share * (1.0 - share) * np.log(hazard_ratio) ** 2)
    return (
        z**2
        * (ratio * hazard_ratio + 1.0) ** 2
        / (ratio * (hazard_ratio - 1.0) ** 2)
    )


def _event_probabilities(
    event_probability: float | tuple[float, float],
) -> tuple[float, float]:
    """The two groups' chances of an event, from one for both or a
    pair."""
    if np.ndim(event_probability) == 0:
        both = check_probability(event_probability, "event_probability")
        return both, both
    pair = tuple(np.asarray(event_probability, dtype=float).ravel())
    if len(pair) != 2:
        raise ValueError(
            "event_probability must be one probability, or a pair for the "
            f"two groups, got {event_probability!r}."
        )
    return (
        check_probability(pair[0], "event_probability"),
        check_probability(pair[1], "event_probability"),
    )


@elementwise
def logrank_events(
    hazard_ratio: float,
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    method: str = "schoenfeld",
    alternative: str = "two-sided",
) -> int:
    r"""The events a log-rank test needs to detect a hazard ratio.

    The number of events (failures, deaths), in both groups together,
    with which a log-rank test at level ``alpha_ci`` detects a hazard
    ratio ``hazard_ratio`` between the groups with probability
    ``power``, under proportional hazards. Schoenfeld's formula, with
    :math:`\pi = r / (1 + r)` the share of the units in the first group,

    .. math::
        d = \frac{(z_{1 - \alpha / 2} + z_{\text{power}})^2}
                 {\pi (1 - \pi) (\ln \text{HR})^2},

    is the default; ``method="freedman"`` uses Freedman's,
    :math:`d = (z_{1 - \alpha / 2} + z_{\text{power}})^2 (r \text{HR} +
    1)^2 / (r (\text{HR} - 1)^2)`. With equal groups Freedman's asks for
    a few more events; with unequal ones it can ask for fewer, and then
    falls a little short of the power (Schoenfeld's holds it better).

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second:
        positive, and not 1.
    power : float, optional
        The chance of detecting it, by default 0.8.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0 (equal groups).
    method : str, optional
        Keyword only: ``"schoenfeld"`` (the default) or ``"freedman"``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

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
    Detecting a halving of the hazard with 80% power at the 5% level
    takes 66 events (88 for 90% power), with equal groups:

    >>> from surpyval.test_planning import logrank_events
    >>> logrank_events(0.5)
    66
    >>> logrank_events(0.5, 0.9)
    88
    >>> logrank_events(0.5, method="freedman")
    71
    """
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    power = check_probability(power, "power")
    ratio = _check_ratio(ratio)
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    return int(np.ceil(_events(hazard_ratio, z, ratio, method) - 1e-9))


def logrank_sample_size(
    hazard_ratio: float,
    event_probability: float | tuple[float, float],
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    method: str = "schoenfeld",
    alternative: str = "two-sided",
) -> tuple[int, int]:
    """The units in each group a log-rank test needs to detect a hazard
    ratio.

    The events of :func:`logrank_events`, shared out in the ratio
    ``ratio`` and divided by each group's chance of having an event
    during the study: with ``n2`` units in the second group and
    ``ratio * n2`` in the first, the expected events are
    ``ratio * n2 * p1 + n2 * p2``. The same as lifelines'
    ``sample_size_necessary_under_cph`` with ``method="freedman"``.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second:
        positive, and not 1.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair, the first group's then the second's.
    power : float, optional
        The chance of detecting it, by default 0.8.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0 (equal groups).
    method : str, optional
        Keyword only: ``"schoenfeld"`` (the default) or ``"freedman"``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

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
    A design expected to cut the failure rate by 30%, where 25% of the
    new units and 35% of the old would fail during the study:

    >>> from surpyval.test_planning import logrank_sample_size
    >>> logrank_sample_size(0.7, (0.25, 0.35))
    (412, 412)
    >>> logrank_sample_size(0.7, (0.25, 0.35), method="freedman")
    (421, 421)
    """
    check_alpha_ci(alpha_ci)
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    power = check_probability(power, "power")
    ratio = _check_ratio(ratio)
    first, second = _event_probabilities(event_probability)
    z = _z_alpha(alpha_ci, alternative) + float(stats.norm.ppf(power))
    events = _events(hazard_ratio, z, ratio, method)
    per_second = events / (ratio * first + second)
    return (
        int(np.ceil(ratio * per_second - 1e-9)),
        int(np.ceil(per_second - 1e-9)),
    )


def logrank_power(
    hazard_ratio: float,
    n: tuple[int, int],
    event_probability: float | tuple[float, float],
    *,
    alpha_ci: float = 0.05,
    method: str = "schoenfeld",
    alternative: str = "two-sided",
) -> float:
    """The chance that a log-rank test detects a hazard ratio, given the
    units in each group.

    The power of the comparison planned: the expected events,
    ``n1 * p1 + n2 * p2``, put back in the formula of
    :func:`logrank_events` and solved for the power. lifelines'
    ``power_under_cph`` with ``method="freedman"``.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second:
        positive, and not 1.
    n : (int, int)
        The units in the first group and in the second.
    event_probability : float or (float, float)
        The chance that a unit has an event during the study: one for
        both groups, or a pair, the first group's then the second's.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    method : str, optional
        Keyword only: ``"schoenfeld"`` (the default) or ``"freedman"``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

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
    With 200 units in each group the comparison above has about a 50%
    chance of detecting the improvement:

    >>> from surpyval.test_planning import logrank_power
    >>> round(logrank_power(0.7, (200, 200), (0.25, 0.35)), 3)
    0.497
    """
    check_alpha_ci(alpha_ci)
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    sizes = tuple(np.asarray(n).ravel())
    if len(sizes) != 2:
        raise ValueError(
            f"n must be the units in the two groups, a pair, got {n!r}."
        )
    first_n = check_count(sizes[0], "n")
    second_n = check_count(sizes[1], "n")
    first, second = _event_probabilities(event_probability)
    ratio = first_n / second_n
    events = first_n * first + second_n * second
    one = _events(hazard_ratio, 1.0, ratio, method)  # events per z ** 2
    z = np.sqrt(events / one) - _z_alpha(alpha_ci, alternative)
    return float(stats.norm.cdf(z))
