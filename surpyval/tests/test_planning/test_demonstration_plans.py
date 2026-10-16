"""Demonstration test plans that keep both risks (#862, ported from
RePyability#184): the consumer's, that a design at the target passes, and
the producer's, that a good design fails.

The reference is every plan tried in turn: the fewest units (or the least
test time), and of those the fewest failures allowed, that keep both.
"""

import pytest
from scipy import stats

from surpyval.test_planning import (
    DemonstrationPlan,
    demonstration_pass_probability,
    demonstration_plan,
    demonstration_test_multiple,
    mtbf_demonstration_plan,
)


def risks(target, good, n, failures, per_test=1.0):
    """A binomial plan's consumer's and producer's risks, the reliabilities
    over the test ``per_test`` powers of the mission's."""
    consumer = stats.binom.cdf(failures, n, 1 - target**per_test)
    producer = 1 - stats.binom.cdf(failures, n, 1 - good**per_test)
    return consumer, producer


def tried(target, good, alpha_ci, producer_risk, most=400):
    for n in range(1, most):
        for failures in range(n):
            consumer, producer = risks(target, good, n, failures)
            if consumer <= alpha_ci and producer <= producer_risk:
                return n, failures
    raise AssertionError("no plan")


@pytest.mark.parametrize(
    "target, good, alpha_ci, producer_risk",
    [
        (0.9, 0.95, 0.1, 0.2),
        (0.9, 0.99, 0.05, 0.1),
        (0.8, 0.9, 0.2, 0.2),
        (0.95, 0.99, 0.1, 0.3),
        (0.5, 0.8, 0.1, 0.1),
    ],
)
def test_the_fewest_units_that_keep_both_risks(
    target, good, alpha_ci, producer_risk
):
    plan = demonstration_plan(
        target, good, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    assert isinstance(plan, DemonstrationPlan)
    assert (plan.n, plan.failures) == tried(
        target, good, alpha_ci, producer_risk
    )
    consumer, producer = risks(target, good, plan.n, plan.failures)
    assert plan.consumer_risk == pytest.approx(consumer, rel=1e-12)
    assert plan.producer_risk == pytest.approx(producer, rel=1e-12)
    assert plan.consumer_risk <= alpha_ci
    assert plan.producer_risk <= producer_risk
    assert plan.test_multiple == 1.0 and plan.test_time is None


def test_a_zero_failure_plan_fails_a_good_design():
    # The success run keeps the consumer's risk with the fewest units, but
    # fails a design 50% better two times in three.
    assert demonstration_pass_probability(0.95, 22) < 0.35
    plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1)
    assert plan.failures > 0
    assert 1 - plan.producer_risk >= 0.8


def test_longer_tests_are_the_same_plans_at_powers_of_the_reliability():
    # A test of k missions, for a Weibull shape b, tests reliability
    # R ** (k ** b): the plan for that is the plan for the powers.
    longer = demonstration_plan(
        0.95, 0.98, alpha_ci=0.1, test_multiple=2.0, shape=1.5
    )
    power = 2.0**1.5
    plain = demonstration_plan(0.95**power, 0.98**power, alpha_ci=0.1)
    assert (longer.n, longer.failures) == (plain.n, plain.failures)
    assert longer.test_multiple == 2.0
    assert longer.consumer_risk == pytest.approx(plain.consumer_risk, 1e-12)


@pytest.mark.parametrize("n, shape", [(20, 2.0), (40, 1.5), (10, 3.0)])
def test_given_the_units_the_shortest_test(n, shape):
    plan = demonstration_plan(0.9, 0.97, alpha_ci=0.1, n=n, shape=shape)
    assert plan.n == n
    per_test = plan.test_multiple**shape
    consumer, producer = risks(0.9, 0.97, n, plan.failures, per_test)
    assert consumer == pytest.approx(0.1, rel=1e-6)  # on the boundary
    assert producer <= 0.2
    # Allowing a failure fewer, the shortest test that keeps the
    # consumer's risk fails the good design too often.
    if plan.failures:
        shorter = demonstration_test_multiple(
            0.9, n, plan.failures - 1, shape=shape, alpha_ci=0.1
        )
        assert shorter < plan.test_multiple
        _, missed = risks(0.9, 0.97, n, plan.failures - 1, shorter**shape)
        assert missed > 0.2


@pytest.mark.parametrize(
    "producer_risk, consumer_risk, ratio, duration, failures",
    [
        # MIL-HDBK-781A's fixed-length plans that keep both risks: XI-D,
        # XV-D and XVII-D (durations in multiples of the MTBF to
        # demonstrate).
        (0.2, 0.2, 1.5, 21.5, 17),
        (0.1, 0.1, 3.0, 9.3, 5),
        (0.2, 0.2, 3.0, 4.3, 2),
    ],
)
def test_the_handbooks_plans(
    producer_risk, consumer_risk, ratio, duration, failures
):
    plan = mtbf_demonstration_plan(
        100.0,
        ratio * 100.0,
        alpha_ci=consumer_risk,
        producer_risk=producer_risk,
    )
    assert plan.failures == failures
    # The handbook gives the durations to 0.1, rounded up at times.
    assert plan.test_time / 100.0 == pytest.approx(duration, abs=0.1)
    assert plan.n is None and plan.test_multiple is None


@pytest.mark.parametrize("ratio", [1.25, 1.5, 2.0, 3.0, 5.0])
@pytest.mark.parametrize("alpha_ci, producer_risk", [(0.1, 0.1), (0.2, 0.3)])
def test_the_shortest_mtbf_test_that_keeps_both_risks(
    ratio, alpha_ci, producer_risk
):
    plan = mtbf_demonstration_plan(
        1.0, ratio, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    # Each number of failures in turn, at its shortest test.
    for failures in range(plan.failures + 1):
        time = stats.chi2.ppf(1 - alpha_ci, 2 * failures + 2) / 2
        missed = 1 - stats.poisson.cdf(failures, time / ratio)
        if failures < plan.failures:
            assert missed > producer_risk
        else:
            assert plan.test_time == pytest.approx(time, rel=1e-12)
            assert plan.producer_risk == pytest.approx(missed, rel=1e-12)
    assert plan.consumer_risk == pytest.approx(alpha_ci, rel=1e-9)


def test_the_plans_print_what_to_run():
    attribute = repr(demonstration_plan(0.9, 0.95, alpha_ci=0.1))
    assert "Units            : 128" in attribute
    assert "Failures allowed : 8" in attribute
    mtbf = repr(mtbf_demonstration_plan(1000.0, 2000.0, alpha_ci=0.1))
    assert "Total test time  : 14206" in mtbf
    assert "Units" not in mtbf


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: demonstration_plan(0.9, 0.9), "more than the reliability"),
        (lambda: demonstration_plan(0.9, 0.8), "more than the reliability"),
        (
            lambda: demonstration_plan(0.9, 0.95, producer_risk=1.0),
            "between",
        ),
        (lambda: demonstration_plan(0.9, 0.95, alpha_ci=0.0), "alpha_ci"),
        (lambda: demonstration_plan(0.9, 0.95, n=10), "give the lifetime"),
        (
            lambda: demonstration_plan(0.9, 0.95, test_multiple=2.0),
            "shape",
        ),
        (
            lambda: demonstration_plan(0.9, 0.91, max_failures=3),
            "at most 3 failures",
        ),
        (
            lambda: demonstration_plan(0.9, 0.95, n=3, shape=2.0),
            "No test of 3 units",
        ),
        (lambda: mtbf_demonstration_plan(10.0, 10.0), "more than the MTBF"),
        (lambda: mtbf_demonstration_plan(-1.0, 10.0), "positive"),
        (
            lambda: mtbf_demonstration_plan(10.0, 10.5, max_failures=5),
            "at most 5 failures",
        ),
    ],
)
def test_plans_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call()
