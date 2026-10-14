"""Reliability demonstration test planning (#862, ported from RePyability's
``demonstration``, RePyability#129): each plan against its forward
function, textbook values, and the special cases that reduce to simpler
plans."""

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats

import surpyval
from surpyval.test_planning import (
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_sample_size,
    demonstration_test_multiple,
    mtbf_pass_probability,
    mtbf_test_time,
)

# Wrong types, passed on purpose to the checks below.
HALF: Any = 1.5
YES: Any = "yes"


@pytest.mark.parametrize(
    "reliability, confidence, failures, n",
    [
        (0.95, 0.95, 0, 59),
        (0.95, 0.95, 1, 93),
        (0.90, 0.90, 0, 22),
        (0.99, 0.90, 0, 230),
        (0.999, 0.90, 0, 2302),
        (0.90, 0.90, 2, 52),
    ],
)
def test_sample_sizes_match_textbook_values(
    reliability, confidence, failures, n
):
    planned = demonstration_sample_size(
        reliability, failures, alpha_ci=1 - confidence
    )
    assert planned == n


@pytest.mark.parametrize("reliability", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
@pytest.mark.parametrize("alpha_ci", [0.5, 0.2, 0.1, 0.05, 0.01])
@pytest.mark.parametrize("failures", [0, 1, 3])
def test_the_planned_test_is_the_smallest_that_demonstrates(
    reliability, alpha_ci, failures
):
    n = demonstration_sample_size(reliability, failures, alpha_ci=alpha_ci)
    shown = demonstrated_reliability(n, failures, alpha_ci=alpha_ci)
    assert shown >= reliability
    if n - 1 > failures:
        fewer = demonstrated_reliability(n - 1, failures, alpha_ci=alpha_ci)
        assert fewer < reliability
    # The planned test passes a design at the target no more often than
    # alpha_ci, and one unit fewer would pass it more often.
    risk = demonstration_pass_probability(reliability, n, failures)
    assert risk <= alpha_ci + 1e-12
    if n - 1 > failures:
        fewer = demonstration_pass_probability(reliability, n - 1, failures)
        assert fewer > alpha_ci


@pytest.mark.parametrize("n", [1, 5, 59, 1000])
@pytest.mark.parametrize("alpha_ci", [0.4, 0.1, 0.05])
def test_no_failures_is_success_run(n, alpha_ci):
    expected = float(surpyval.success_run(n, alpha_ci=alpha_ci))
    shown = demonstrated_reliability(n, alpha_ci=alpha_ci)
    assert shown == pytest.approx(expected, rel=1e-12)


def test_the_bound_with_failures_is_clopper_pearson():
    # The chance of so few failures, at the bound, is exactly alpha_ci.
    n, failures, alpha_ci = 40, 3, 0.1
    bound = demonstrated_reliability(n, failures, alpha_ci=alpha_ci)
    chance = stats.binom.cdf(failures, n, 1.0 - bound)
    assert chance == pytest.approx(alpha_ci, rel=1e-9)
    assert demonstrated_reliability(10, 10, alpha_ci=0.1) == 0.0


def test_the_bound_is_bernoullis_exact_bound():
    # The same Clopper-Pearson bound the Bernoulli fit gives its p.
    n, failures = 30, 2
    x = np.r_[np.zeros(n - failures), np.ones(failures)]
    p_upper = surpyval.Bernoulli.fit(x).param_cb("p", 0.05, bound="upper")
    shown = demonstrated_reliability(n, failures)
    assert shown == pytest.approx(1.0 - p_upper.item(), rel=1e-9)


def test_a_one_mission_test_ignores_the_shape():
    assert demonstration_sample_size(
        0.95, alpha_ci=0.1, test_multiple=1.0, shape=3.0
    ) == demonstration_sample_size(0.95, alpha_ci=0.1)
    assert demonstrated_reliability(
        30, 1, alpha_ci=0.1, test_multiple=1.0, shape=0.5
    ) == demonstrated_reliability(30, 1, alpha_ci=0.1)


@pytest.mark.parametrize("k, beta", [(2.0, 2.0), (1.5, 3.0), (0.5, 1.0)])
def test_weibayes_trades_units_for_test_time(k, beta):
    reliability, alpha_ci = 0.95, 0.1
    n = demonstration_sample_size(
        reliability, alpha_ci=alpha_ci, test_multiple=k, shape=beta
    )
    closed_form = math.log(alpha_ci) / (k**beta * math.log(reliability))
    assert n == math.ceil(closed_form)
    shown = demonstrated_reliability(
        n, alpha_ci=alpha_ci, test_multiple=k, shape=beta
    )
    run = float(surpyval.success_run(n, alpha_ci=alpha_ci))
    assert shown == pytest.approx(run ** (1 / k**beta))
    assert shown >= reliability


@pytest.mark.parametrize("failures", [0, 2])
def test_the_test_multiple_demonstrates_exactly_the_target(failures):
    reliability, n, alpha_ci, beta = 0.97, 25, 0.1, 2.5
    k = demonstration_test_multiple(
        reliability, n, failures, shape=beta, alpha_ci=alpha_ci
    )
    shown = demonstrated_reliability(
        n, failures, alpha_ci=alpha_ci, test_multiple=k, shape=beta
    )
    assert shown == pytest.approx(reliability, rel=1e-12)
    # Enough units for a one-mission test need less than a mission each.
    many = (
        demonstration_sample_size(reliability, failures, alpha_ci=alpha_ci)
        + 50
    )
    shorter = demonstration_test_multiple(
        reliability, many, failures, shape=beta, alpha_ci=alpha_ci
    )
    assert shorter < 1.0


@pytest.mark.parametrize(
    "alpha_ci, failures, multiple",
    [(0.1, 0, 2.303), (0.1, 1, 3.890), (0.1, 2, 5.322), (0.2, 0, 1.609)],
)
def test_mtbf_test_times_match_textbook_values(alpha_ci, failures, multiple):
    time = mtbf_test_time(1000.0, failures, alpha_ci=alpha_ci)
    assert time == pytest.approx(1000.0 * multiple, abs=0.5)


@pytest.mark.parametrize("failures", [0, 1, 5])
@pytest.mark.parametrize("alpha_ci", [0.4, 0.1, 0.01])
def test_the_mtbf_plan_and_bound_are_inverses(failures, alpha_ci):
    time = mtbf_test_time(250.0, failures, alpha_ci=alpha_ci)
    shown = demonstrated_mtbf(time, failures, alpha_ci=alpha_ci)
    assert shown == pytest.approx(250.0, rel=1e-12)
    # At the target MTBF the test passes exactly alpha_ci of the time (the
    # Poisson-chi-squared duality).
    assert mtbf_pass_probability(250.0, time, failures) == pytest.approx(
        alpha_ci, rel=1e-9
    )


def test_a_failure_terminated_test_is_one_failure_fewer():
    stopped = mtbf_test_time(100.0, 3, failure_terminated=True)
    assert stopped == mtbf_test_time(100.0, 2)
    assert mtbf_pass_probability(
        100.0, stopped, 3, failure_terminated=True
    ) == mtbf_pass_probability(100.0, stopped, 2)


def test_better_designs_pass_more_often():
    reliabilities = np.linspace(0.9, 0.999, 12)
    chances = demonstration_pass_probability(reliabilities, 40, 1)
    assert np.all(np.diff(chances) > 0)
    mtbfs = np.linspace(100.0, 5000.0, 12)
    chances = mtbf_pass_probability(mtbfs, 3000.0, 2)
    assert np.all(np.diff(chances) > 0)


def test_arrays_answer_element_by_element():
    sizes = demonstration_sample_size(
        np.array([[0.9], [0.95]]), [0, 1], alpha_ci=0.05
    )
    assert sizes.shape == (2, 2)
    assert sizes[1, 0] == 59 and sizes[1, 1] == 93
    levels = mtbf_test_time(1000.0, alpha_ci=np.array([0.05, 0.1]))
    assert levels == pytest.approx(
        [mtbf_test_time(1000.0), mtbf_test_time(1000.0, alpha_ci=0.1)]
    )


def test_alpha_ci_given_as_a_confidence_is_warned_of():
    with pytest.warns(UserWarning, match="significance level"):
        demonstration_sample_size(0.9, alpha_ci=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        demonstration_sample_size(0.9, alpha_ci=0.05)


@pytest.mark.parametrize(
    "call",
    [
        lambda: demonstration_sample_size(1.0),
        lambda: demonstration_sample_size(0.9, alpha_ci=1.2),
        lambda: demonstration_sample_size(0.9, failures=-1),
        lambda: demonstration_sample_size(0.9, failures=HALF),
        lambda: demonstration_sample_size(0.9, test_multiple=2.0),
        lambda: demonstration_sample_size(0.9, test_multiple=0.0, shape=2.0),
        lambda: demonstration_sample_size(0.9, test_multiple=2.0, shape=0.0),
        lambda: demonstrated_reliability(0),
        lambda: demonstrated_reliability(5, failures=6),
        lambda: demonstration_test_multiple(0.9, 3, 3, shape=2.0),
        lambda: demonstration_pass_probability(0.0, 10),
        lambda: mtbf_test_time(-1.0),
        lambda: mtbf_test_time(10.0, 0, failure_terminated=True),
        lambda: mtbf_test_time(10.0, 1, failure_terminated=YES),
        lambda: demonstrated_mtbf(0.0),
        lambda: mtbf_pass_probability(100.0, -5.0),
    ],
)
def test_invalid_plans_are_refused(call):
    with pytest.raises(ValueError):
        call()
