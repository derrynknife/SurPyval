"""Reliability demonstration tests (#862, from RePyability's
``demonstration``, RePyability#129).

A demonstration test shows, at a confidence level, that a design meets a
reliability target: test ``n`` units for a time, allow ``failures`` of
them, and if no more fail, the design has demonstrated the target.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from surpyval.test_planning._checks import (
    check_count,
    check_failures,
    check_flag,
    check_positive,
    check_probability,
    elementwise,
)
from surpyval.test_planning._serialise import dataclass_dict
from surpyval.utils.validation import check_alpha_ci


@dataclass(frozen=True, repr=False)
class DemonstrationPlan:
    """The result of ``demonstration_plan`` and
    ``mtbf_demonstration_plan``.

    A demonstration test that keeps both of its risks: a design no better
    than the target passes it with a chance of at most the consumer's
    risk, and a good design fails it with a chance of at most the
    producer's.

    Attributes
    ----------
    n : int or None
        The number of units to test (an attribute test); None for a test
        of a constant failure rate.
    test_multiple : float or None
        How many missions long each unit's test is (an attribute test);
        None for a test of a constant failure rate.
    test_time : float or None
        The total unit time to test (a test of a constant failure rate);
        None for an attribute test.
    failures : int
        The failures the test allows: it passes with at most so many.
    consumer_risk : float
        The chance that a design at the target passes.
    producer_risk : float
        The chance that a design at the good level fails.

    Examples
    --------
    >>> from surpyval.test_planning import demonstration_plan
    >>> plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1)
    >>> plan
    Demonstration Plan
    ==================
    Units            : 128
    Test length      : 1 mission
    Failures allowed : 8
    Consumer's risk  : 0.0971274
    Producer's risk  : 0.191852
    >>> plan.n, plan.failures
    (128, 8)
    """

    n: int | None
    test_multiple: float | None
    test_time: float | None
    failures: int
    consumer_risk: float
    producer_risk: float

    def to_dict(self) -> dict:
        """The plan as a dict of native Python values, strict JSON:
        ``n``, ``test_multiple``, ``test_time``, ``failures``,
        ``consumer_risk`` and ``producer_risk``, under ``"result":
        "DemonstrationPlan"``. An infinite or missing number is ``None``,
        recorded under ``"non_finite"``.

        Examples
        --------
        >>> import json
        >>> from surpyval.test_planning import demonstration_plan
        >>> d = demonstration_plan(0.9, 0.95, alpha_ci=0.1).to_dict()
        >>> d["n"], d["failures"], d["test_time"]
        (128, 8, None)
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        """
        return dataclass_dict(self)

    def __repr__(self) -> str:
        header = "Demonstration Plan"
        lines = [header, "=" * len(header)]
        if self.n is not None:
            multiple = self.test_multiple
            missions = "mission" if multiple == 1.0 else "missions"
            lines += [
                f"Units            : {self.n}",
                f"Test length      : {multiple:.6g} {missions}",
            ]
        else:
            lines.append(f"Total test time  : {self.test_time:.6g}")
        lines += [
            f"Failures allowed : {self.failures}",
            f"Consumer's risk  : {self.consumer_risk:.6g}",
            f"Producer's risk  : {self.producer_risk:.6g}",
        ]
        return "\n".join(lines)


@elementwise
def demonstration_sample_size(
    reliability: float,
    failures: int = 0,
    *,
    alpha_ci: float = 0.05,
    test_multiple: float = 1.0,
    shape: float | None = None,
) -> int:
    """The fewest units whose test demonstrates a reliability.

    The smallest ``n`` for which ``n`` units tested, with at most
    ``failures`` of them failing, demonstrate ``reliability`` over a
    mission at the confidence level ``1 - alpha_ci``: the chance that so
    few fail, were the reliability only the target, is at most
    ``alpha_ci``. With no failures this is the success run,
    ``n = ln(alpha_ci) / ln(reliability)`` rounded up, the inverse of
    :func:`surpyval.success_run`. Testing each unit for ``test_multiple``
    missions, with the lifetime's Weibull ``shape`` known, needs fewer
    units (the extended success run, or Weibayes).

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape, one plan for each element.

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    failures : int, optional
        The failures the test allows, by default 0.
    alpha_ci : float, optional
        Keyword only: the consumer's risk, the chance that a design no
        better than the target passes, by default 0.05 (95% confidence).
    test_multiple : float, optional
        How many missions long each unit's test is, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple``
        is not 1.

    Returns
    -------
    int
        The number of units to test.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    59 units with no failures demonstrate 95% reliability at 95%
    confidence, and allowing one failure takes 93:

    >>> from surpyval.test_planning import demonstration_sample_size
    >>> demonstration_sample_size(0.95)
    59
    >>> demonstration_sample_size(0.95, failures=1)
    93

    Testing each unit for two missions, for a wear-out shape of 2, needs a
    quarter of the units:

    >>> demonstration_sample_size(0.95, test_multiple=2.0, shape=2.0)
    15
    """
    check_probability(reliability, "reliability")
    failures = check_failures(failures)
    per_test = _test_reliability(reliability, test_multiple, shape)

    def passes_too_often(n: int) -> bool:
        return bool(stats.binom.cdf(failures, n, 1.0 - per_test) > alpha_ci)

    low, high = failures + 1, 2 * failures + 2
    while passes_too_often(high):
        low, high = high + 1, 2 * high
    while low < high:
        middle = (low + high) // 2
        if passes_too_often(middle):
            low = middle + 1
        else:
            high = middle
    return int(low)


@elementwise
def demonstrated_reliability(
    n: int,
    failures: int = 0,
    *,
    alpha_ci: float = 0.05,
    test_multiple: float = 1.0,
    shape: float | None = None,
) -> float:
    """The reliability a finished test demonstrated.

    The lower one-sided ``1 - alpha_ci`` confidence bound on the
    reliability over a mission, from ``n`` units tested with ``failures``
    of them failing: the Clopper-Pearson bound, the smallest reliability
    for which so few failures have a chance of at least ``alpha_ci``.
    With no failures it is ``alpha_ci ** (1 / n)``,
    :func:`surpyval.success_run`. A test of ``test_multiple`` missions per
    unit, with the lifetime's Weibull ``shape`` known, demonstrates that
    bound to the power ``1 / test_multiple ** shape`` over one mission.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    n : int
        The number of units tested, at least 1.
    failures : int, optional
        How many of them failed, from 0 to ``n``, by default 0.
    alpha_ci : float, optional
        Keyword only: the significance level of the one-sided bound, by
        default 0.05 (95% confidence).
    test_multiple : float, optional
        How many missions long each unit's test was, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple``
        is not 1.

    Returns
    -------
    float
        The demonstrated reliability over one mission.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    >>> from surpyval.test_planning import demonstrated_reliability
    >>> round(demonstrated_reliability(59), 4)
    0.9505
    >>> round(demonstrated_reliability(93, failures=1), 5)
    0.95001
    """
    n = check_count(n)
    failures = check_failures(failures)
    if failures > n:
        raise ValueError(f"failures must be at most n ({n}), got {failures}.")
    exponent = _test_exponent(test_multiple, shape)
    if failures == n:
        return 0.0
    upper = stats.beta.ppf(1.0 - alpha_ci, failures + 1, n - failures)
    return float((1.0 - upper) ** (1.0 / exponent))


@elementwise
def demonstration_test_multiple(
    reliability: float,
    n: int,
    failures: int = 0,
    *,
    shape: float,
    alpha_ci: float = 0.05,
) -> float:
    """How many missions long each unit's test must be to demonstrate a
    reliability with ``n`` units.

    The shortest test, as a multiple of the mission time, for which ``n``
    units with at most ``failures`` failing demonstrate ``reliability``
    at the confidence level ``1 - alpha_ci``, the lifetime's Weibull
    ``shape`` known (the extended success run, or Weibayes). Below 1, a
    test shorter than a mission is enough.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    n : int
        The number of units to test, more than ``failures``.
    failures : int, optional
        The failures the test allows, by default 0.
    shape : float
        Keyword only: the Weibull shape of the lifetime.
    alpha_ci : float, optional
        Keyword only: the consumer's risk, by default 0.05 (95%
        confidence).

    Returns
    -------
    float
        The test length, in missions.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Twenty units, for a wear-out shape of 2, each run about 1.7 missions
    to demonstrate 95% reliability at 95% confidence (with a test of one
    mission, 59 units would be needed):

    >>> from surpyval.test_planning import demonstration_test_multiple
    >>> round(demonstration_test_multiple(0.95, 20, shape=2.0), 3)
    1.709
    """
    check_probability(reliability, "reliability")
    n = check_count(n)
    failures = check_failures(failures)
    if failures >= n:
        raise ValueError(
            f"n must be more than the failures allowed ({failures}), got {n}."
        )
    _check_shape(shape)
    per_test = demonstrated_reliability(n, failures, alpha_ci=alpha_ci)
    return float((np.log(per_test) / np.log(reliability)) ** (1.0 / shape))


@elementwise
def demonstration_pass_probability(
    reliability: float,
    n: int,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: float | None = None,
) -> float:
    """The chance that a design of a given true reliability passes a test.

    The probability that at most ``failures`` of ``n`` units fail when
    each survives a mission with probability ``reliability`` (the test's
    operating characteristic). At the target reliability it is at most
    ``alpha_ci`` for a planned test, the consumer's risk; one minus it at
    a good design's reliability is the producer's risk, the chance of
    failing a design that should pass.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    reliability : float
        The design's true reliability over one mission, in (0, 1).
    n : int
        The number of units tested, at least 1.
    failures : int, optional
        The failures the test allows, by default 0.
    test_multiple : float, optional
        How many missions long each unit's test is, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple``
        is not 1.

    Returns
    -------
    float
        The probability of passing.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    The 59-unit, no-failure test passes a design at the 95% target about
    one time in twenty, and one of 99% reliability five times in nine:

    >>> from surpyval.test_planning import demonstration_pass_probability
    >>> round(demonstration_pass_probability(0.95, 59), 3)
    0.048
    >>> round(demonstration_pass_probability(0.99, 59), 3)
    0.553
    """
    check_probability(reliability, "reliability")
    n = check_count(n)
    failures = check_failures(failures)
    per_test = _test_reliability(reliability, test_multiple, shape)
    return float(stats.binom.cdf(failures, n, 1.0 - per_test))


@elementwise
def mtbf_test_time(
    mtbf: float,
    failures: int = 0,
    *,
    alpha_ci: float = 0.05,
    failure_terminated: bool = False,
) -> float:
    """The total test time that demonstrates an MTBF, at a constant
    failure rate.

    The unit time (summed over the units, failed ones repaired or
    replaced) a time-terminated test needs, with at most ``failures``
    failures, to demonstrate ``mtbf`` at the confidence level
    ``1 - alpha_ci``: ``mtbf * chi2.ppf(1 - alpha_ci, 2 * failures + 2) /
    2``. A failure-terminated test, which runs until its ``failures``-th
    failure, demonstrates ``mtbf`` if that failure comes after
    ``mtbf * chi2.ppf(1 - alpha_ci, 2 * failures) / 2``: the time of a
    time-terminated test that allows one failure fewer.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    mtbf : float
        The mean time between failures to demonstrate, positive.
    failures : int, optional
        The failures the test allows, by default 0; with
        ``failure_terminated``, the failure it stops at, from 1.
    alpha_ci : float, optional
        Keyword only: the consumer's risk, by default 0.05 (95%
        confidence).
    failure_terminated : bool, optional
        Keyword only: whether the test stops at its ``failures``-th
        failure rather than at a time, by default False.

    Returns
    -------
    float
        The total test time, in the MTBF's units.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    With no failures, about 3 MTBFs of testing demonstrate the MTBF at 95%
    confidence:

    >>> from surpyval.test_planning import mtbf_test_time
    >>> round(mtbf_test_time(1000.0), 1)
    2995.7
    >>> round(mtbf_test_time(1000.0, failures=2), 1)
    6295.8
    >>> round(mtbf_test_time(1000.0, 3, failure_terminated=True), 1)
    6295.8
    """
    check_positive(mtbf, "mtbf")
    failures = check_failures(failures)
    freedom = _degrees_of_freedom(failures, failure_terminated)
    return float(mtbf * stats.chi2.ppf(1.0 - alpha_ci, freedom) / 2.0)


@elementwise
def demonstrated_mtbf(
    test_time: float,
    failures: int = 0,
    *,
    alpha_ci: float = 0.05,
    failure_terminated: bool = False,
) -> float:
    """The MTBF a finished test demonstrated, at a constant failure rate.

    The lower one-sided ``1 - alpha_ci`` confidence bound on the MTBF
    from a total test time with ``failures`` failures. A test stopped at
    its time (time-terminated) gives ``2 * test_time / chi2.ppf(1 -
    alpha_ci, 2 * failures + 2)``; one stopped at its ``failures``-th
    failure (``failure_terminated``), with ``2 * failures`` degrees of
    freedom instead.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    test_time : float
        The total unit time tested, positive: with
        ``failure_terminated``, up to the last failure.
    failures : int, optional
        The failures seen, by default 0 (at least 1 with
        ``failure_terminated``).
    alpha_ci : float, optional
        Keyword only: the significance level of the one-sided bound, by
        default 0.05 (95% confidence).
    failure_terminated : bool, optional
        Keyword only: whether the test stopped at its last failure rather
        than at a time, by default False.

    Returns
    -------
    float
        The demonstrated MTBF, in the test time's units.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import demonstrated_mtbf
    >>> round(demonstrated_mtbf(2995.7), 1)
    1000.0

    Three failures, the test stopped at the third, 6,295.8 hours in all:

    >>> round(demonstrated_mtbf(6295.8, 3, failure_terminated=True), 1)
    1000.0
    """
    check_positive(test_time, "test_time")
    failures = check_failures(failures)
    freedom = _degrees_of_freedom(failures, failure_terminated)
    return float(2.0 * test_time / stats.chi2.ppf(1.0 - alpha_ci, freedom))


@elementwise
def mtbf_pass_probability(
    mtbf: float,
    test_time: float,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The chance that a design of a given true MTBF passes a test, at a
    constant failure rate.

    The probability of at most ``failures`` failures in ``test_time``
    when failures come at rate ``1 / mtbf``: the Poisson distribution's,
    the test's operating characteristic. A failure-terminated test passes
    if its ``failures``-th failure comes after ``test_time``: at most one
    fewer by then.

    Any argument may be an array: the answer is then an array of the
    arguments' broadcast shape.

    Parameters
    ----------
    mtbf : float
        The design's true mean time between failures, positive.
    test_time : float
        The total unit time tested, positive: with ``failure_terminated``,
        the time the last failure must come after.
    failures : int, optional
        The failures the test allows, by default 0; with
        ``failure_terminated``, the failure it stops at, from 1.
    failure_terminated : bool, optional
        Keyword only: whether the test stops at its ``failures``-th
        failure, by default False.

    Returns
    -------
    float
        The probability of passing.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    A test planned to demonstrate an MTBF of 1000 passes a design at that
    MTBF one time in twenty, and one with three times the MTBF only a
    little more than one time in three:

    >>> from surpyval.test_planning import (
    ...     mtbf_pass_probability,
    ...     mtbf_test_time,
    ... )
    >>> time = mtbf_test_time(1000.0)
    >>> round(mtbf_pass_probability(1000.0, time), 3)
    0.05
    >>> round(mtbf_pass_probability(3000.0, time), 3)
    0.368
    """
    check_positive(mtbf, "mtbf")
    check_positive(test_time, "test_time")
    failures = check_failures(failures)
    allowed = _degrees_of_freedom(failures, failure_terminated) // 2 - 1
    return float(stats.poisson.cdf(allowed, test_time / mtbf))


def demonstration_plan(
    reliability: float,
    good_reliability: float,
    *,
    alpha_ci: float = 0.05,
    producer_risk: float = 0.2,
    n: int | None = None,
    test_multiple: float = 1.0,
    shape: float | None = None,
    max_failures: int = 1000,
) -> DemonstrationPlan:
    """The smallest attribute test that demonstrates a reliability and
    that a good design passes.

    A plan's consumer's risk is the chance that a design of the target
    ``reliability`` passes it, at most ``alpha_ci``; its producer's risk
    is the chance that a design of ``good_reliability``, which should
    pass, fails it, at most ``producer_risk``. A test allowing no
    failures keeps the first with the fewest units, but often fails a
    good design: allowing failures, with more units, keeps both. For each
    number of failures allowed, from none, the fewest units that keep the
    consumer's risk (:func:`demonstration_sample_size`) pass the good
    design most often, and the first of those plans that keeps the
    producer's risk tests the fewest units of all.

    Given ``n``, the test length is searched instead, with the lifetime's
    Weibull ``shape``: the shortest test of ``n`` units that keeps both
    risks (each failure allowed takes a longer test, as
    :func:`demonstration_test_multiple` gives it).

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    good_reliability : float
        The reliability of a design that should pass, more than
        ``reliability`` and less than 1.
    alpha_ci : float, optional
        Keyword only: the most consumer's risk, by default 0.05 (95%
        confidence).
    producer_risk : float, optional
        Keyword only: the most chance of failing a good design, in
        (0, 1), by default 0.2.
    n : int, optional
        Keyword only: the number of units to test, to search the test
        length instead of the number of units (with ``shape``).
    test_multiple : float, optional
        Keyword only, without ``n``: how many missions long each unit's
        test is, by default 1.0 (another length needs ``shape``).
    shape : float, optional
        Keyword only: the Weibull shape of the lifetime, needed for a
        test longer or shorter than a mission, and with ``n``.
    max_failures : int, optional
        Keyword only: the most failures a plan may allow, by default
        1000.

    Returns
    -------
    DemonstrationPlan
        The plan: ``n``, ``test_multiple`` and ``failures``, and its
        risks.

    Raises
    ------
    ValueError
        If an argument is out of range, ``good_reliability`` is no more
        than ``reliability``, ``n`` is given without ``shape``, or no plan
        within ``max_failures`` (or, given ``n``, fewer failures than
        units) keeps both risks.

    Examples
    --------
    Demonstrating 90% reliability at 90% confidence, so that a design of
    95% passes at least 80% of the time: a test of 22 units with no
    failures would pass that design only a third of the time, and the
    plan tests 128, allowing 8 to fail:

    >>> from surpyval.test_planning import (
    ...     demonstration_pass_probability,
    ...     demonstration_plan,
    ... )
    >>> round(demonstration_pass_probability(0.95, 22), 2)
    0.32
    >>> plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1)
    >>> plan.n, plan.failures
    (128, 8)
    >>> round(plan.consumer_risk, 3), round(plan.producer_risk, 3)
    (0.097, 0.192)
    """
    check_probability(reliability, "reliability")
    check_probability(good_reliability, "good_reliability")
    check_alpha_ci(alpha_ci)
    check_probability(producer_risk, "producer_risk")
    if good_reliability <= reliability:
        raise ValueError(
            "good_reliability must be more than the reliability to "
            f"demonstrate ({reliability!r}), got {good_reliability!r}: no "
            "test can pass a design no better than the target more often "
            "than the target."
        )
    most = check_failures(max_failures)
    if n is not None:
        n = check_count(n)
        if shape is None:
            raise ValueError(
                "Given n, the test length is searched: give the lifetime's "
                "Weibull shape."
            )
        _check_shape(shape)
        for failures in range(min(most, n - 1) + 1):
            multiple = demonstration_test_multiple(
                reliability, n, failures, shape=shape, alpha_ci=alpha_ci
            )
            plan = _attribute_plan(
                reliability, good_reliability, n, failures, multiple, shape
            )
            if plan.producer_risk <= producer_risk:
                return plan
        raise ValueError(
            f"No test of {n} units keeps both risks: test more units, or "
            "allow a larger producer's risk."
        )
    _test_exponent(test_multiple, shape)
    for failures in range(most + 1):
        count = demonstration_sample_size(
            reliability,
            failures,
            alpha_ci=alpha_ci,
            test_multiple=test_multiple,
            shape=shape,
        )
        plan = _attribute_plan(
            reliability,
            good_reliability,
            count,
            failures,
            test_multiple,
            shape,
        )
        if plan.producer_risk <= producer_risk:
            return plan
    raise ValueError(
        f"No plan allowing at most {most} failures keeps both risks: the "
        "good reliability is too close to the target (raise max_failures, "
        "or allow larger risks)."
    )


def _attribute_plan(
    reliability: float,
    good_reliability: float,
    n: int,
    failures: int,
    test_multiple: float,
    shape: float | None,
) -> DemonstrationPlan:
    """An attribute test's plan, with its risks."""
    return DemonstrationPlan(
        n=int(n),
        test_multiple=float(test_multiple),
        test_time=None,
        failures=int(failures),
        consumer_risk=demonstration_pass_probability(
            reliability, n, failures, test_multiple=test_multiple, shape=shape
        ),
        producer_risk=1.0
        - demonstration_pass_probability(
            good_reliability,
            n,
            failures,
            test_multiple=test_multiple,
            shape=shape,
        ),
    )


def mtbf_demonstration_plan(
    mtbf: float,
    good_mtbf: float,
    *,
    alpha_ci: float = 0.05,
    producer_risk: float = 0.2,
    max_failures: int = 1000,
) -> DemonstrationPlan:
    """The shortest constant-failure-rate test that demonstrates an MTBF
    and that a good design passes.

    The total test time and the failures it allows, for a time-terminated
    test, such that a design of the target ``mtbf`` passes with a chance
    of at most ``alpha_ci`` (the consumer's risk) and one of ``good_mtbf``
    fails with a chance of at most ``producer_risk``: the fixed-length
    test plans of MIL-HDBK-781, for a discrimination ratio
    ``good_mtbf / mtbf``. For each number of failures allowed, from none,
    the shortest test that keeps the consumer's risk
    (:func:`mtbf_test_time`) passes the good design most often, and the
    first that keeps the producer's risk is the shortest of all. Both
    risks are kept, where some of the handbook's plans exceed one a
    little for a shorter test; those that keep both are these (plans
    XI-D, XV-D and XVII-D).

    Parameters
    ----------
    mtbf : float
        The MTBF to demonstrate, positive.
    good_mtbf : float
        The MTBF of a design that should pass, more than ``mtbf``.
    alpha_ci : float, optional
        Keyword only: the most consumer's risk, by default 0.05 (95%
        confidence).
    producer_risk : float, optional
        Keyword only: the most chance of failing a good design, in
        (0, 1), by default 0.2.
    max_failures : int, optional
        Keyword only: the most failures a plan may allow, by default
        1000.

    Returns
    -------
    DemonstrationPlan
        The plan: ``test_time`` and ``failures``, and its risks.

    Raises
    ------
    ValueError
        If an argument is out of range, ``good_mtbf`` is no more than
        ``mtbf``, or no plan within ``max_failures`` keeps both risks.

    Examples
    --------
    Demonstrating an MTBF of 1000 hours at 90% confidence, so that a
    design of 2000 passes at least 80% of the time (a discrimination
    ratio of 2): the test runs about 14,200 unit hours, and passes with
    at most 9 failures:

    >>> from surpyval.test_planning import mtbf_demonstration_plan
    >>> plan = mtbf_demonstration_plan(1000.0, 2000.0, alpha_ci=0.1)
    >>> round(plan.test_time), plan.failures
    (14206, 9)
    >>> round(plan.consumer_risk, 3), round(plan.producer_risk, 3)
    (0.1, 0.18)
    """
    check_positive(mtbf, "mtbf")
    check_positive(good_mtbf, "good_mtbf")
    check_alpha_ci(alpha_ci)
    check_probability(producer_risk, "producer_risk")
    if good_mtbf <= mtbf:
        raise ValueError(
            f"good_mtbf must be more than the MTBF to demonstrate "
            f"({mtbf!r}), got {good_mtbf!r}."
        )
    most = check_failures(max_failures)
    for failures in range(most + 1):
        time = mtbf_test_time(mtbf, failures, alpha_ci=alpha_ci)
        missed = 1.0 - mtbf_pass_probability(good_mtbf, time, failures)
        if missed <= producer_risk:
            return DemonstrationPlan(
                n=None,
                test_multiple=None,
                test_time=float(time),
                failures=failures,
                consumer_risk=mtbf_pass_probability(mtbf, time, failures),
                producer_risk=float(missed),
            )
    raise ValueError(
        f"No plan allowing at most {most} failures keeps both risks: the "
        "good MTBF is too close to the target (raise max_failures, or "
        "allow larger risks)."
    )


def _test_reliability(
    reliability: float, test_multiple: float, shape: float | None
) -> float:
    """A unit's reliability over the whole test."""
    return float(reliability ** _test_exponent(test_multiple, shape))


def _test_exponent(test_multiple: float, shape: float | None) -> float:
    """The power a mission's reliability is raised to over a test of
    ``test_multiple`` missions: ``test_multiple ** shape``."""
    try:
        positive = bool(np.isfinite(test_multiple) and test_multiple > 0.0)
    except (TypeError, ValueError):
        positive = False
    if not positive:
        raise ValueError(
            f"test_multiple must be positive, got {test_multiple!r}."
        )
    if test_multiple == 1.0:
        return 1.0
    if shape is None:
        raise ValueError(
            "A test longer or shorter than a mission needs the lifetime's "
            "Weibull shape: give shape."
        )
    _check_shape(shape)
    return float(test_multiple**shape)


def _check_shape(shape: float) -> None:
    check_positive(shape, "shape")


def _degrees_of_freedom(failures: int, failure_terminated: bool) -> int:
    """The chi-squared degrees of freedom of a test with ``failures``
    failures: ``2 * failures + 2`` if it stopped at its time, ``2 *
    failures`` at its last failure."""
    if not check_flag(failure_terminated, "failure_terminated"):
        return 2 * failures + 2
    if failures < 1:
        raise ValueError(
            "A failure-terminated test stops at a failure: failures must be "
            "at least 1."
        )
    return 2 * failures
