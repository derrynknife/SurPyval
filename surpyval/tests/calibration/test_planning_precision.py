"""The test plans against simulation (#863, #864, #865, #880).

A plan's precision is a large-sample approximation: simulating the
planned test and fitting each simulated data set must give estimates
whose spread is the planned one (to 12% on the log scale for the life
tests, 15% for the accelerated tests, whose fits extrapolate further; the
large-sample spread is a little narrow when the quantity lies beyond the
test's end and the failures are few, as for the Weibull median below),
and that
fall within the planned factor about ``1 - alpha_ci`` of the time. The
sequential tests' exact operating characteristics must match their
simulated pass rates and test times, and the log-rank sample size must
give the planned power.
"""

import numpy as np
import pytest

import surpyval
from surpyval import life_models
from surpyval.test_planning import (
    accelerated_test_plan,
    life_test_precision,
    logrank_sample_size,
    sequential_demonstration_test,
    sequential_mtbf_test,
)
from surpyval.tests.test_planning.test_sequential import (
    run_attribute,
    run_mtbf,
)


@pytest.mark.parametrize(
    "distribution, params, n, options",
    [
        (surpyval.Weibull, (1000.0, 2.0), 50, dict(censor_time=500.0)),
        (surpyval.Weibull, (1000.0, 2.0), 50, dict(failures=20)),
        (surpyval.Weibull, (1000.0, 3.0), 80, dict(censor_time=700.0, p=0.5)),
        (surpyval.LogNormal, (6.0, 0.6), 40, dict(censor_time=400.0)),
        (surpyval.LogNormal, (6.0, 0.6), 40, dict(quantity="mean")),
        (
            surpyval.Weibull,
            (1000.0, 1.5),
            100,
            dict(quantity="mean", censor_time=1500.0, alpha_ci=0.1),
        ),
    ],
)
def test_life_test_precision_is_the_simulated_one(
    distribution, params, n, options
):
    plan = life_test_precision(distribution, params, n, **options)
    check = plan.simulate(1000, random_state=0)
    assert np.log(check.precision) == pytest.approx(
        np.log(plan.precision), rel=0.12
    )
    assert check.within == pytest.approx(1 - plan.alpha_ci, abs=0.025)


@pytest.mark.parametrize(
    "distribution, model, life_params, shape, stresses, levels",
    [
        (
            surpyval.Weibull,
            life_models.Power,
            {"a": 1e6, "n": -2.0},
            2.0,
            dict(use_stress=10.0, high_stress=60.0, censor_time=1000.0),
            2,
        ),
        (
            surpyval.LogNormal,
            life_models.Arrhenius,
            {"a": 6000.0, "b": 6.2e-4},
            0.6,
            dict(use_stress=330.0, high_stress=420.0, censor_time=3000.0),
            3,
        ),
    ],
)
def test_accelerated_test_precision_is_the_simulated_one(
    distribution, model, life_params, shape, stresses, levels
):
    plan = accelerated_test_plan(
        distribution,
        model,
        life_params,
        shape,
        n=80,
        levels=levels,
        **stresses,
    )
    check = plan.simulate(400, random_state=1)
    assert check.unusable == 0
    assert np.log(check.precision) == pytest.approx(
        np.log(plan.precision), rel=0.15
    )
    assert check.within >= 0.9


@pytest.mark.parametrize("truncate", [True, False])
def test_sequential_tests_pass_as_often_as_computed(truncate):
    rng = np.random.default_rng(21)
    reps = 20000
    test = sequential_mtbf_test(
        100.0, 250.0, alpha_ci=0.1, producer_risk=0.2, truncate=truncate
    )
    for mtbf in (100.0, 160.0, 250.0):
        passed, mean_time = run_mtbf(test, mtbf, reps, rng)
        assert passed == pytest.approx(
            test.pass_probability(mtbf), abs=4 * np.sqrt(0.25 / reps)
        )
        assert mean_time == pytest.approx(
            test.expected_test_time(mtbf), rel=0.02
        )
    attribute = sequential_demonstration_test(
        0.85, 0.95, alpha_ci=0.1, producer_risk=0.2, truncate=truncate
    )
    for reliability in (0.85, 0.9, 0.95):
        passed, units = run_attribute(attribute, reliability, reps, rng)
        assert passed == pytest.approx(
            attribute.pass_probability(reliability),
            abs=4 * np.sqrt(0.25 / reps),
        )
        assert units == pytest.approx(
            attribute.expected_units(reliability), rel=0.02
        )


@pytest.mark.parametrize("method", ["schoenfeld", "freedman"])
def test_the_planned_sample_size_has_the_planned_power(method):
    # Two groups of 2:1, exponential lifetimes with a hazard ratio of
    # 0.6, every unit followed for a fixed time: the log-rank test of the
    # planned units rejects at 5% about 80% of the time with Schoenfeld's
    # formula. Freedman's asks for fewer events with unequal groups and
    # falls a little short (about 77%), as is known of it.
    rng = np.random.default_rng(31)
    follow_up = 1.0
    rate = -np.log(1 - 0.5)  # half the second group fails
    p1, p2 = 1 - np.exp(-0.6 * rate), 0.5
    n1, n2 = logrank_sample_size(0.6, (p1, p2), 0.8, ratio=2.0, method=method)
    Z = np.r_[np.zeros(n1), np.ones(n2)]
    reps, rejected = 2000, 0
    for _ in range(reps):
        t = np.r_[
            rng.exponential(1 / (0.6 * rate), n1),
            rng.exponential(1 / rate, n2),
        ]
        c = (t > follow_up).astype(int)
        x = np.minimum(t, follow_up)
        rejected += surpyval.logrank(x, Z, c=c).p_value < 0.05
    low = 0.77 if method == "schoenfeld" else 0.74
    assert low <= rejected / reps <= 0.86
