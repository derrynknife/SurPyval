"""Power and sample size for comparing two groups under proportional
hazards (#880): published values, lifelines' formulas written out, the
plans' consistency with each other, and a simulation of the log-rank test
the plan is for."""

from typing import Any

import numpy as np
import pytest
from scipy import stats

import surpyval
from surpyval.test_planning import (
    logrank_events,
    logrank_power,
    logrank_sample_size,
)

# Wrong types, passed on purpose to the checks below.
THREE: Any = (0.2, 0.3, 0.4)
ONE_GROUP: Any = (10,)


@pytest.mark.parametrize(
    "hazard_ratio, power, events",
    [
        # Schoenfeld (1983), equal groups, two-sided 5%: the commonly
        # quoted 66 events for a hazard ratio of 2 (or 1/2) at 80% power,
        # and 88 at 90%.
        (0.5, 0.8, 66),
        (2.0, 0.8, 66),
        (0.5, 0.9, 88),
        (2.0, 0.9, 88),
        (0.75, 0.8, 380),
    ],
)
def test_schoenfelds_events(hazard_ratio, power, events):
    assert logrank_events(hazard_ratio, power) == events


def freedman_lifelines(power, ratio, p_exp, p_con, hazard_ratio, alpha):
    """lifelines' ``sample_size_necessary_under_cph``, written out."""
    z = stats.norm.ppf
    m = (
        1.0
        / ratio
        * ((ratio * hazard_ratio + 1.0) / (hazard_ratio - 1.0)) ** 2
        * (z(1.0 - alpha / 2.0) + z(power)) ** 2
    )
    n_exp = m * ratio / (ratio * p_exp + p_con)
    n_con = m / (ratio * p_exp + p_con)
    return int(np.ceil(n_exp)), int(np.ceil(n_con))


def power_lifelines(n_exp, n_con, p_exp, p_con, hazard_ratio, alpha):
    """lifelines' ``power_under_cph``, written out."""
    m = n_exp * p_exp + n_con * p_con
    k = n_exp / n_con
    return stats.norm.cdf(
        np.sqrt(k * m) * abs(hazard_ratio - 1) / (k * hazard_ratio + 1)
        - stats.norm.ppf(1 - alpha / 2.0)
    )


def test_freedman_is_lifelines_documented_example():
    # lifelines' documentation: (421, 421).
    planned = logrank_sample_size(0.7, (0.25, 0.35), method="freedman")
    assert planned == (421, 421)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("hazard_ratio", [0.4, 0.7, 1.5, 3.0])
@pytest.mark.parametrize("power, alpha", [(0.8, 0.05), (0.9, 0.01)])
def test_freedman_is_lifelines(ratio, hazard_ratio, power, alpha):
    p_exp, p_con = 0.3, 0.45
    planned = logrank_sample_size(
        hazard_ratio,
        (p_exp, p_con),
        power,
        alpha_ci=alpha,
        ratio=ratio,
        method="freedman",
    )
    expected = freedman_lifelines(
        power, ratio, p_exp, p_con, hazard_ratio, alpha
    )
    # Equal but for an exact integer, which lifelines rounds up one more.
    assert np.abs(np.subtract(planned, expected)).max() <= 1
    chance = logrank_power(
        hazard_ratio,
        planned,
        (p_exp, p_con),
        alpha_ci=alpha,
        method="freedman",
    )
    assert chance == pytest.approx(
        power_lifelines(*planned, p_exp, p_con, hazard_ratio, alpha),
        rel=1e-12,
    )


@pytest.mark.parametrize("method", ["schoenfeld", "freedman"])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
@pytest.mark.parametrize("alternative", ["two-sided", "one-sided"])
def test_the_planned_units_reach_the_power(method, ratio, alternative):
    options = dict(method=method, alternative=alternative)
    first, second = logrank_sample_size(
        0.6, (0.4, 0.5), 0.85, ratio=ratio, **options
    )
    assert first == pytest.approx(ratio * second, abs=1)
    reached = logrank_power(0.6, (first, second), (0.4, 0.5), **options)
    assert reached >= 0.85 - 1e-9
    fewer = logrank_power(0.6, (first - 1, second - 1), (0.4, 0.5), **options)
    assert fewer < 0.85


def test_schoenfeld_takes_fewer_events_than_freedman():
    for hazard_ratio in (0.3, 0.6, 0.9, 1.2, 2.5):
        assert logrank_events(hazard_ratio) <= logrank_events(
            hazard_ratio, method="freedman"
        )


def test_equal_groups_need_fewest_events():
    events = logrank_events(0.6, ratio=np.array([0.25, 0.5, 1.0, 2.0, 4.0]))
    assert events.argmin() == 2
    assert events[0] == events[-1]  # 1:4 and 4:1 alike


def test_one_sided_tests_need_fewer_events():
    assert logrank_events(0.5, alternative="one-sided") < logrank_events(0.5)


def test_the_planned_test_detects_the_ratio_as_often_as_planned():
    # Complete exponential data, equal groups, as many units as the
    # events planned: the log-rank test rejects at 5% about 80% of the
    # time.
    events = logrank_events(0.5, 0.8)
    per_group = int(np.ceil(events / 2))
    rng = np.random.default_rng(3)
    Z = np.r_[np.zeros(per_group), np.ones(per_group)]
    reps, rejected = 400, 0
    for _ in range(reps):
        x = np.r_[
            rng.exponential(1.0, per_group), rng.exponential(2.0, per_group)
        ]
        rejected += surpyval.logrank(x, Z).p_value < 0.05
    assert rejected / reps == pytest.approx(0.8, abs=0.07)


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: logrank_events(1.0), "differ from 1"),
        (lambda: logrank_events(-0.5), "positive"),
        (lambda: logrank_events(0.5, 1.0), "power"),
        (lambda: logrank_events(0.5, ratio=0.0), "ratio"),
        (lambda: logrank_events(0.5, method="cox"), "'method'"),
        (lambda: logrank_events(0.5, alternative="less"), "'alternative'"),
        (lambda: logrank_events(0.5, alpha_ci=1.5), "alpha_ci"),
        (lambda: logrank_sample_size(0.5, 1.2), "event_probability"),
        (lambda: logrank_sample_size(0.5, THREE), "pair"),
        (lambda: logrank_power(0.5, ONE_GROUP, 0.3), "pair"),
        (lambda: logrank_power(0.5, (10, 0), 0.3), "n must"),
    ],
)
def test_comparisons_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call()
