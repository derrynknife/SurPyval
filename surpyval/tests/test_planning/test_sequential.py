"""Sequential demonstration tests (#865): Wald's SPRT for an MTBF and for
a pass/fail reliability.

The oracles: the likelihood ratio written out here, a simulation of the
test run failure by failure (or unit by unit), Wald's inequality for the
untruncated test, and the fixed-length test it is truncated at.
"""

from typing import Any

import numpy as np
import pytest

from surpyval.test_planning import (
    SequentialDemonstrationTest,
    SequentialMTBFTest,
    demonstration_plan,
    mtbf_demonstration_plan,
    sequential_demonstration_test,
    sequential_mtbf_test,
)

# Wrong types, passed on purpose to the checks below.
ONE: Any = 1


def log_ratio_mtbf(failures, time, target, good):
    """The log likelihood ratio of the target MTBF to the good one."""
    return failures * np.log(good / target) - time * (1 / target - 1 / good)


def log_ratio_attribute(n, failures, target, good):
    return failures * np.log((1 - target) / (1 - good)) + (
        n - failures
    ) * np.log(target / good)


def run_mtbf(test, mtbf, reps, rng):
    """The share of simulated designs of ``mtbf`` that pass, and their
    mean test time: failures drawn one after another, the test asked at
    each and as time accumulates."""
    passed, total = 0, 0.0
    for _ in range(reps):
        time, failures = 0.0, 0
        while True:
            arrival = time + rng.exponential(mtbf)
            accept = test.accept_time(failures)
            if accept <= arrival:
                passed += 1
                total += accept
                break
            time, failures = arrival, failures + 1
            if test.decide(failures, time) == "reject":
                total += time
                break
    return passed / reps, total / reps


def run_attribute(test, reliability, reps, rng):
    passed, units = 0, 0
    for _ in range(reps):
        n = failures = 0
        while True:
            n += 1
            failures += rng.random() > reliability
            decision = test.decide(n, failures)
            if decision != "continue":
                passed += decision == "accept"
                units += n
                break
    return passed / reps, units / reps


@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("mtbf", [100.0, 140.0, 200.0])
def test_the_mtbf_tests_risks_are_its_simulated_ones(truncate, mtbf):
    test = sequential_mtbf_test(
        100.0, 200.0, alpha_ci=0.1, producer_risk=0.1, truncate=truncate
    )
    reps = 4000
    passed, mean_time = run_mtbf(test, mtbf, reps, np.random.default_rng(7))
    chance = test.pass_probability(mtbf)
    assert passed == pytest.approx(chance, abs=4 * np.sqrt(0.25 / reps))
    assert mean_time == pytest.approx(test.expected_test_time(mtbf), rel=0.05)


@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("reliability", [0.9, 0.94, 0.97])
def test_the_attribute_tests_risks_are_its_simulated_ones(
    truncate, reliability
):
    test = sequential_demonstration_test(
        0.9, 0.97, alpha_ci=0.1, producer_risk=0.1, truncate=truncate
    )
    reps = 4000
    passed, units = run_attribute(
        test, reliability, reps, np.random.default_rng(11)
    )
    chance = test.pass_probability(reliability)
    assert passed == pytest.approx(chance, abs=4 * np.sqrt(0.25 / reps))
    assert units == pytest.approx(test.expected_units(reliability), rel=0.05)


def test_the_attribute_oc_is_every_path_summed():
    # A small truncated test, every sequence of passes and failures
    # enumerated: the chance of those it accepts.
    test = sequential_demonstration_test(
        0.7, 0.95, alpha_ci=0.2, producer_risk=0.2
    )
    reliability = 0.85

    def accepted(n, failures, chance):
        decision = test.decide(n, failures) if n else "continue"
        if decision != "continue":
            return chance if decision == "accept" else 0.0
        return accepted(n + 1, failures, chance * reliability) + accepted(
            n + 1, failures + 1, chance * (1 - reliability)
        )

    assert test.max_units is not None and test.max_units < 25
    expected = accepted(0, 0, 1.0)
    assert test.pass_probability(reliability) == pytest.approx(expected)


def test_the_boundaries_are_walds():
    alpha_ci, producer_risk = 0.1, 0.2
    test = sequential_mtbf_test(
        50.0,
        150.0,
        alpha_ci=alpha_ci,
        producer_risk=producer_risk,
        truncate=False,
    )
    accept = np.log(alpha_ci / (1 - producer_risk))
    reject = np.log((1 - alpha_ci) / producer_risk)
    for failures in range(8):
        at = test.accept_time(failures)
        ratio = log_ratio_mtbf(failures, at, 50.0, 150.0)
        assert ratio == pytest.approx(accept)
        by = test.reject_time(failures)
        if by > 0:
            ratio = log_ratio_mtbf(failures, by, 50.0, 150.0)
            assert ratio == pytest.approx(reject)
    attribute = sequential_demonstration_test(
        0.8,
        0.95,
        alpha_ci=alpha_ci,
        producer_risk=producer_risk,
        truncate=False,
    )
    for n in range(1, 80):
        for failures in range(n + 1):
            ratio = log_ratio_attribute(n, failures, 0.8, 0.95)
            decision = attribute.decide(n, failures)
            if ratio >= reject:
                assert decision == "reject"
            elif ratio <= accept:
                assert decision == "accept"
            else:
                assert decision == "continue"


@pytest.mark.parametrize("ratio", [1.5, 2.0, 3.0])
def test_untruncated_tests_keep_walds_inequality(ratio):
    # The two error chances of an SPRT add to no more than the two risks
    # its boundaries were set for (Wald).
    test = sequential_mtbf_test(
        1.0, ratio, alpha_ci=0.1, producer_risk=0.2, truncate=False
    )
    assert test.consumer_risk + test.producer_risk <= 0.3 + 1e-9
    attribute = sequential_demonstration_test(
        0.9, 1 - 0.1 / ratio, alpha_ci=0.1, producer_risk=0.2, truncate=False
    )
    total = attribute.consumer_risk + attribute.producer_risk
    assert total <= 0.3 + 1e-9


@pytest.mark.parametrize(
    "mtbf, good, alpha_ci, producer_risk",
    [(100.0, 200.0, 0.1, 0.1), (100.0, 150.0, 0.2, 0.2), (10, 30, 0.1, 0.1)],
)
def test_a_sequential_test_decides_sooner_than_the_fixed_one(
    mtbf, good, alpha_ci, producer_risk
):
    fixed = mtbf_demonstration_plan(
        mtbf, good, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    test = sequential_mtbf_test(
        mtbf, good, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    assert test.max_time == fixed.test_time
    assert test.max_failures == fixed.failures + 1
    for design in (mtbf, good):
        assert test.expected_test_time(design) < fixed.test_time
    plan = demonstration_plan(
        0.9, 0.97, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    attribute = sequential_demonstration_test(
        0.9, 0.97, alpha_ci=alpha_ci, producer_risk=producer_risk
    )
    assert attribute.max_units == plan.n
    for design in (0.9, 0.97):
        assert attribute.expected_units(design) < plan.n


def test_a_failure_that_cannot_reject_never_does():
    # The first failures cannot reject however soon they come: at test
    # time 0 the test has only started.
    test = sequential_mtbf_test(100.0, 200.0)
    assert test.reject_time(0) == test.reject_time(1) == 0.0
    assert test.decide(0, 0.0) == "continue"
    assert test.decide(1, 0.0) == "continue"


def test_a_truncated_test_stops_at_the_fixed_test():
    test = sequential_mtbf_test(100.0, 300.0, alpha_ci=0.1, producer_risk=0.1)
    assert test.decide(test.max_failures, 1e9) == "reject"
    assert test.decide(test.max_failures - 1, test.max_time) == "accept"
    assert test.accept_time(50) == test.max_time
    attribute = sequential_demonstration_test(
        0.9, 0.99, alpha_ci=0.1, producer_risk=0.1
    )
    n = attribute.max_units
    assert attribute.decide(n, attribute.max_failures - 1) == "accept"
    assert attribute.decide(n - 1, attribute.max_failures) == "reject"


def test_better_designs_pass_more_often():
    test = sequential_mtbf_test(100.0, 200.0)
    chances = test.pass_probability(np.linspace(60.0, 400.0, 9))
    assert chances.shape == (9,)
    assert np.all(np.diff(chances) > 0)
    attribute = sequential_demonstration_test(0.9, 0.97)
    chances = attribute.pass_probability([[0.85, 0.9], [0.95, 0.99]])
    assert chances.shape == (2, 2)
    assert np.all(np.diff(chances.ravel()) > 0)


def test_the_tests_print_their_risks():
    text = repr(sequential_mtbf_test(100.0, 200.0))
    assert text.startswith("Sequential MTBF Test")
    assert "Truncated at" in text and "Consumer's risk" in text
    text = repr(sequential_demonstration_test(0.9, 0.97, truncate=False))
    assert "Truncated" not in text
    assert isinstance(sequential_mtbf_test(1.0, 2.0), SequentialMTBFTest)
    assert isinstance(
        sequential_demonstration_test(0.5, 0.9), SequentialDemonstrationTest
    )


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: sequential_mtbf_test(100.0, 100.0), "more than the MTBF"),
        (lambda: sequential_mtbf_test(-1.0, 100.0), "positive"),
        (
            lambda: sequential_mtbf_test(
                1.0, 2.0, alpha_ci=0.5, producer_risk=0.5
            ),
            "add to less than 1",
        ),
        (
            lambda: sequential_mtbf_test(1.0, 2.0, truncate=ONE),
            "True or False",
        ),
        (
            lambda: sequential_demonstration_test(0.9, 0.9),
            "more than the reliability",
        ),
        (
            lambda: sequential_demonstration_test(0.9, 0.95, producer_risk=0),
            "between",
        ),
        (lambda: sequential_mtbf_test(1.0, 2.0).decide(-1, 1.0), "failures"),
        (lambda: sequential_mtbf_test(1.0, 2.0).decide(1, -1.0), "test_time"),
        (
            lambda: sequential_demonstration_test(0.9, 0.97).decide(3, 4),
            "at most n",
        ),
        (
            lambda: sequential_mtbf_test(1.0, 2.0).pass_probability(0.0),
            "mtbf",
        ),
        (
            lambda: sequential_demonstration_test(0.9, 0.97).expected_units(
                1.0
            ),
            "reliability",
        ),
    ],
)
def test_sequential_tests_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call()
