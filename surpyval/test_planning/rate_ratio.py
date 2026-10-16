r"""Power and exposure for comparing two Poisson rates: the failure rate
before an intervention (a design change, a new procedure, a corrective
action) and the rate after it.

The test planned is the exact conditional test of the ratio of two
Poisson rates (Przyborowski and Wilenski 1940; Cox and Lewis 1966,
section 9.4), the test :func:`surpyval.recurrent.intervention_test` runs
under a homogeneous Poisson process. With :math:`E_b` and :math:`E_a` the
exposure (operating time) before and after, and :math:`\rho` the rate
after over the rate before, the number of failures after, given the
:math:`N` failures in all, is binomial,

.. math::
    N_a \mid N \sim \text{Bin}\!\left(N,\
    \pi(\rho) = \frac{\rho E_a}{E_b + \rho E_a}\right),

so the test is exact whatever the rates, and :math:`\rho = 1` is tested
by the binomial tail at :math:`\pi(1) = E_a / (E_b + E_a)`. Its power has
no closed form, but it is a finite sum: the failures in all are Poisson,
with mean :math:`\lambda_b (E_b + \rho E_a)`, and for each :math:`N` the
test rejects for the counts in a binomial tail, so

.. math::
    \text{power} = \sum_{N} \Pr(N)\,
    \Pr\{N_a \le c_N \mid N, \pi(\rho)\},

with :math:`c_N` the largest count the test rejects at (Gu, Ng, Tang and
Schucany 2008 give the same sum). It is computed exactly here, not by
the normal approximation, which is poor for the few failures a corrective
action is usually judged on.

References
----------
Przyborowski, J. and Wilenski, H. (1940), "Homogeneity of results in
testing samples from Poisson series", Biometrika 31, 313-323.

Cox, D. R. and Lewis, P. A. W. (1966), "The Statistical Analysis of
Series of Events", Methuen, section 9.4.

Gu, K., Ng, H. K. T., Tang, M. L. and Schucany, W. R. (2008), "Testing
the ratio of two Poisson rates", Biometrical Journal 50, 283-298.

Lancaster, H. O. (1961), "Significance tests in discrete distributions",
Journal of the American Statistical Association 56, 223-234 (the mid-p
value).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from surpyval.test_planning._checks import (
    check_positive,
    check_probability,
    elementwise,
)
from surpyval.utils.validation import check_option

METHODS = ("exact", "mid-p")
ALTERNATIVES = ("two-sided", "one-sided")
DIRECTIONS = ("decrease", "increase")

# The Poisson tail left out of the power's sum over the total count.
_TAIL = 1e-14
# The largest exposure multiple searched for by rate_ratio_exposure.
_MAX_MULTIPLE = 1e6


def _tail_level(alpha_ci: float, alternative: str) -> float:
    """The probability in the tail the test rejects in."""
    check_option("alternative", alternative, ALTERNATIVES)
    return alpha_ci / 2.0 if alternative == "two-sided" else alpha_ci


def _lower_p(
    k: npt.NDArray, n: npt.NDArray, p: float, midp: bool
) -> npt.NDArray:
    """The lower-tail p-value of ``k`` successes in ``n`` trials:
    :math:`\\Pr(X \\le k)`, or with ``midp`` :math:`\\Pr(X < k) +
    \\Pr(X = k) / 2`."""
    lower = stats.binom.cdf(k, n, p)
    if midp:
        lower = lower - 0.5 * stats.binom.pmf(k, n, p)
    return lower


def critical_counts(
    n: npt.NDArray, p0: float, tail: float, midp: bool
) -> npt.NDArray:
    """For each total count ``n``, the largest count ``c`` of failures
    after the intervention whose lower-tail p-value is below ``tail``
    (the test rejects for every count up to it); -1 where none is."""
    n = np.asarray(n, dtype=float)
    c = stats.binom.ppf(tail, n, p0) - 1.0
    c = np.where(np.isnan(c), -1.0, c)
    # binom.ppf is the smallest k with cdf(k) >= tail; guard its rounding.
    down = (c >= 0) & (_lower_p(c, n, p0, False) >= tail)
    c = np.where(down, c - 1, c)
    if midp:
        # The mid-p value of k lies between the exact ones of k - 1 and k,
        # so its critical count is the exact one or one more.
        up = (c + 1 <= n) & (_lower_p(c + 1, n, p0, True) < tail)
        c = np.where(up, c + 1, c)
    return c


def expected_power(
    expected_before: float,
    expected_after: float,
    ratio: float,
    tail: float,
    midp: bool,
) -> float:
    """The power of the conditional test from the expected failures:
    ``expected_before`` before the intervention, and ``expected_after``
    after it were the rate unchanged. The chance of rejecting in the
    direction of ``ratio`` (a decrease for a ratio below 1)."""
    if ratio > 1.0:
        # An increase after is a decrease before: swap the periods.
        return expected_power(
            ratio * expected_after,
            ratio * expected_before,
            1.0 / ratio,
            tail,
            midp,
        )
    p0 = expected_after / (expected_before + expected_after)
    p1 = ratio * expected_after / (expected_before + ratio * expected_after)
    mean = expected_before + ratio * expected_after
    lo = stats.poisson.ppf(_TAIL, mean)
    hi = stats.poisson.isf(_TAIL, mean)
    n = np.arange(max(0.0, lo), hi + 1.0)
    weight = stats.poisson.pmf(n, mean)
    c = critical_counts(n, p0, tail, midp)
    reject = np.where(c >= 0, stats.binom.cdf(c, n, p1), 0.0)
    return float(min(1.0, np.sum(weight * reject)))


def _check_ratio(ratio: float) -> float:
    ratio = check_positive(ratio, "ratio")
    if ratio == 1.0:
        raise ValueError(
            "ratio must differ from 1: no test can detect a change of none."
        )
    return ratio


def _check_method(method: str) -> bool:
    check_option("method", method, METHODS)
    return method == "mid-p"


@elementwise
def rate_ratio_power(
    ratio: float,
    rate: float,
    exposure_before: float,
    exposure_after: float,
    *,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    method: str = "exact",
) -> float:
    r"""The chance that the exact conditional test detects a change in a
    failure rate.

    The power of the test of :func:`surpyval.recurrent.intervention_test`
    (under a homogeneous Poisson process): the chance that, with the rate
    ``rate`` before an intervention over ``exposure_before``, and
    ``ratio`` times it after over ``exposure_after``, the test finds the
    change at level ``alpha_ci``, in its direction. It is the exact sum
    over the Poisson count of failures given in the module notes, not a
    normal approximation.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    ratio : float
        The rate after the intervention over the rate before: below 1 for
        an improvement (0.5 halves the rate), above 1 for a worsening.
    rate : float
        The failure rate before the intervention, failures per unit of
        exposure.
    exposure_before, exposure_after : float
        The exposure (operating time, cycles, ... in the unit of
        ``rate``) before and after the intervention.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"`` (a
        test in the direction of ``ratio``).
    method : str, optional
        Keyword only: ``"exact"`` (the default) for the exact conditional
        test, or ``"mid-p"`` for its mid-p version.

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
    Thirty failures were seen in 10000 hours before a design change.
    Had the change halved the rate, 5000 hours after it would detect that
    about a third of the time, and 20000 hours about 70% of the time:

    >>> from surpyval.test_planning import rate_ratio_power
    >>> round(rate_ratio_power(0.5, 30 / 10000, 10000, 5000), 3)
    0.356
    >>> round(rate_ratio_power(0.5, 30 / 10000, 10000, 20000), 3)
    0.712
    >>> rate_ratio_power([0.3, 0.7], 30 / 10000, 10000, 20000).round(3)
    array([0.973, 0.276])

    The mid-p test is more powerful, at the price of a level that holds
    only on average:

    >>> round(
    ...     rate_ratio_power(0.5, 30 / 10000, 10000, 5000, method="mid-p"), 3
    ... )
    0.431
    """
    ratio = _check_ratio(ratio)
    rate = check_positive(rate, "rate")
    exposure_before = check_positive(exposure_before, "exposure_before")
    exposure_after = check_positive(exposure_after, "exposure_after")
    midp = _check_method(method)
    tail = _tail_level(alpha_ci, alternative)
    return expected_power(
        rate * exposure_before, rate * exposure_after, ratio, tail, midp
    )


def detectable_ratio(
    expected_before: float,
    expected_after: float,
    power: float,
    tail: float,
    midp: bool,
    direction: str,
) -> float:
    """The ratio nearest 1 that the test detects with ``power``, in
    ``direction``, from the expected failures; ``nan`` for a decrease no
    ratio reaches."""

    def short(u: float) -> float:
        ratio = float(np.exp(u))
        return (
            expected_power(expected_before, expected_after, ratio, tail, midp)
            - power
        )

    if direction == "decrease":
        far = np.log(1e-9)
        if short(far) < 0.0:
            return float("nan")  # not even removing every failure
        near = -1e-9
    else:
        near, far = 1e-9, 1.0
        while short(far) < 0.0:
            far *= 2.0
    if short(near) >= 0.0:
        return float(np.exp(near))
    u = optimize.brentq(short, far, near, xtol=1e-10)
    # brentq's root may sit a hair on the short side of the target.
    step = -1e-9 if direction == "decrease" else 1e-9
    while short(u) < 0.0:
        u += step
        step *= 2.0
    return float(np.exp(u))


@elementwise
def rate_ratio_detectable(
    rate: float,
    exposure_before: float,
    exposure_after: float,
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    method: str = "exact",
    direction: str = "decrease",
) -> float:
    """The smallest change in a failure rate that the exact conditional
    test detects with a given power.

    The ratio, nearest 1, of the rate after an intervention to the rate
    before that :func:`rate_ratio_power` detects with probability
    ``power``: for a decrease, the smallest reduction the exposure can
    reliably show is ``1 - ratio``. Found by root-finding on the exact
    power.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    rate : float
        The failure rate before the intervention, failures per unit of
        exposure.
    exposure_before, exposure_after : float
        The exposure before and after the intervention.
    power : float, optional
        The chance of detecting the change, by default 0.8.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.
    method : str, optional
        Keyword only: ``"exact"`` (the default) or ``"mid-p"``.
    direction : str, optional
        Keyword only: ``"decrease"`` (the default) for the smallest
        reduction, or ``"increase"`` for the smallest worsening.

    Returns
    -------
    float
        The ratio: below 1 for a decrease, above 1 for an increase. For a
        decrease, ``nan`` when even a rate of zero after the
        intervention would be detected with less than ``power``: the
        failures before the intervention are too few, or the exposure
        after it too short, to show any improvement reliably.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    With 30 failures in 10000 hours before, and 5000 hours after, only a
    reduction of about 74% would be detected reliably; 20000 hours after
    would detect one of about 55%:

    >>> from surpyval.test_planning import rate_ratio_detectable
    >>> round(rate_ratio_detectable(30 / 10000, 10000, 5000), 3)
    0.263
    >>> round(rate_ratio_detectable(30 / 10000, 10000, 20000), 3)
    0.455

    Three failures before can never show an improvement reliably:

    >>> rate_ratio_detectable(3 / 10000, 10000, 20000)
    nan
    """
    rate = check_positive(rate, "rate")
    exposure_before = check_positive(exposure_before, "exposure_before")
    exposure_after = check_positive(exposure_after, "exposure_after")
    power = check_probability(power, "power")
    midp = _check_method(method)
    check_option("direction", direction, DIRECTIONS)
    tail = _tail_level(alpha_ci, alternative)
    return detectable_ratio(
        rate * exposure_before,
        rate * exposure_after,
        power,
        tail,
        midp,
        direction,
    )


def limit_power(
    expected_before: float, ratio: float, tail: float, midp: bool
) -> float:
    """The power of the conditional test as the exposure after the
    intervention grows without end. The rate after is then known exactly,
    and the test is of the failures before, Poisson with mean
    ``expected_before``, against the Poisson count the rate after would
    give over the exposure before, of mean ``ratio * expected_before``."""
    null_mean = ratio * expected_before
    hi = stats.poisson.isf(_TAIL, max(null_mean, expected_before))
    k = np.arange(0.0, hi + 1.0)
    point = stats.poisson.pmf(k, null_mean)
    if ratio < 1.0:
        # A decrease: too many failures before for the rate after.
        p_value = stats.poisson.sf(k - 1, null_mean)
    else:
        p_value = stats.poisson.cdf(k, null_mean)
    if midp:
        p_value = p_value - 0.5 * point
    reject = p_value < tail
    return float(np.sum(stats.poisson.pmf(k[reject], expected_before)))


def exposure_multiple(
    expected_before: float,
    ratio: float,
    power: float,
    tail: float,
    midp: bool,
) -> float:
    """The exposure after the intervention, as a multiple of the exposure
    before, at which the test detects ``ratio`` with ``power``; ``inf``
    when no exposure after does (the failures before are too few)."""

    def short(log_multiple: float) -> float:
        multiple = float(np.exp(log_multiple))
        return (
            expected_power(
                expected_before,
                expected_before * multiple,
                ratio,
                tail,
                midp,
            )
            - power
        )

    if limit_power(expected_before, ratio, tail, midp) < power:
        return float("inf")
    hi = 0.0
    while short(hi) < 0.0:
        hi += np.log(4.0)
        if hi > np.log(_MAX_MULTIPLE):
            return float("inf")
    lo = hi - np.log(4.0)
    while short(lo) >= 0.0:
        lo -= np.log(4.0)
        if lo < np.log(1e-12):
            return float(np.exp(lo))
    root = optimize.brentq(short, lo, hi, xtol=1e-10)
    step = 1e-9
    while short(root) < 0.0:
        root += step
        step *= 2.0
    return float(np.exp(root))


@elementwise
def rate_ratio_exposure(
    ratio: float,
    rate: float,
    exposure_before: float,
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    method: str = "exact",
) -> float:
    """The exposure after an intervention that the exact conditional test
    needs to detect a change in a failure rate.

    The exposure after the intervention (in all, not in addition to any
    already seen) at which :func:`rate_ratio_power` reaches ``power`` for
    the change ``ratio``, the exposure before being fixed. Found by
    root-finding on the exact power; the power of a discrete test is not
    quite monotone in the exposure, so the answer is the exposure at
    which the search found it reach ``power``.

    The failures before the intervention cap what any exposure after it
    can show: even an exposure without end only pins down the rate after
    exactly, leaving the rate before as uncertain as its failures make
    it. When that is not enough, the answer is ``inf``, and only more
    exposure before (or a larger change) can help.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    ratio : float
        The change to detect: the rate after the intervention over the
        rate before.
    rate : float
        The failure rate before the intervention.
    exposure_before : float
        The exposure before the intervention.
    power : float, optional
        The chance of detecting the change, by default 0.8.
    alpha_ci : float, optional
        Keyword only: the significance level of the test, by default
        0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.
    method : str, optional
        Keyword only: ``"exact"`` (the default) or ``"mid-p"``.

    Returns
    -------
    float
        The exposure after the intervention, in the unit of
        ``exposure_before``; ``inf`` when no exposure is enough.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    To detect a halving of the rate after 30 failures in 10000 hours
    takes about 41000 hours after the change, and a one-sided test
    about 19000:

    >>> from surpyval.test_planning import rate_ratio_exposure
    >>> round(rate_ratio_exposure(0.5, 30 / 10000, 10000), -2)
    41200.0
    >>> round(
    ...     rate_ratio_exposure(
    ...         0.5, 30 / 10000, 10000, alternative="one-sided"
    ...     ),
    ...     -2,
    ... )
    19200.0

    Twelve failures before can never show a halving with 80% power, as
    the rate before is too uncertain:

    >>> rate_ratio_exposure(0.5, 12 / 10000, 10000)
    inf
    """
    ratio = _check_ratio(ratio)
    rate = check_positive(rate, "rate")
    exposure_before = check_positive(exposure_before, "exposure_before")
    power = check_probability(power, "power")
    midp = _check_method(method)
    tail = _tail_level(alpha_ci, alternative)
    multiple = exposure_multiple(
        rate * exposure_before, ratio, power, tail, midp
    )
    return multiple * exposure_before
