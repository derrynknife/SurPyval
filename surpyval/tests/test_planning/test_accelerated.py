"""Planning an accelerated life test (#864).

The oracles: the variance of the estimated quantile from the curvature of
the test's expected log-likelihood, found by finite differences (the
planning code multiplies analytic information matrices instead); the
plan's optimality, checked against plans that move its lowest stress or
its allocation; the textbook limits (an uncensored test is best run at
the use stress); and, briefly, the test simulated and fitted (the full
simulation study is in ``tests/calibration``).
"""

from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

import surpyval
from surpyval import life_models
from surpyval.test_planning import (
    AcceleratedTestPlan,
    accelerated_test_plan,
    accelerated_test_precision,
)

VOLTAGE: dict[str, Any] = dict(
    distribution=surpyval.Weibull,
    life_model=life_models.Power,
    life_params={"a": 1e6, "n": -2.0},
    shape=2.0,
)
THERMAL: dict[str, Any] = dict(
    distribution=surpyval.LogNormal,
    life_model=life_models.Arrhenius,
    life_params={"a": 6000.0, "b": 6.2e-4},
    shape=0.6,
)


def expected_log_likelihood(family, zeta, mu, sigma):
    """One unit's expected log-likelihood at (mu, sigma) on the
    standardised scale, the truth being (0, 1), censored at zeta."""
    dist = stats.gumbel_l if family == "sev" else stats.norm

    def failed(y):
        return (dist.logpdf((y - mu) / sigma) - np.log(sigma)) * dist.pdf(y)

    value, _ = integrate.quad(failed, -40, zeta, limit=200, epsabs=1e-13)
    return value + dist.sf(zeta) * dist.logsf((zeta - mu) / sigma)


def finite_difference_variance(case, stresses, shares, use, censor, p):
    """The per-unit variance of the log quantile at ``use``: the inverse
    of minus the expected log-likelihood's Hessian, by finite
    differences in the life parameters (relative steps) and the shape."""
    model = case["life_model"]
    family = "sev" if case["distribution"] is surpyval.Weibull else "normal"
    values = np.array(list(case["life_params"].values()))
    sigma0 = 1 / case["shape"] if family == "sev" else case["shape"]

    def log_life(theta, stress):
        return float(np.log(model.phi(np.array([stress]), *theta)[0]))

    truth = [log_life(values, s) for s in stresses]
    zetas = [(np.log(censor) - m) / sigma0 for m in truth]
    scale = np.append(np.abs(values), sigma0)

    def ell(step):
        theta = values + step[:-1] * scale[:-1]
        sigma = sigma0 + step[-1] * scale[-1]
        return sum(
            w
            * expected_log_likelihood(
                family,
                z,
                (log_life(theta, s) - m) / sigma0,
                sigma / sigma0,
            )
            for s, w, z, m in zip(stresses, shares, zetas, truth)
        )

    k, h = len(scale), 1e-4
    hessian = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            hessian[i, j] = hessian[j, i] = (
                ell(e_i + e_j)
                - ell(e_i - e_j)
                - ell(-e_i + e_j)
                + ell(-e_i - e_j)
            ) / (4 * h * h)
    # The steps were in units of the scaled parameters (the
    # standardised log-likelihood differs from the log-likelihood only by
    # a constant, -log sigma0).
    hessian /= np.outer(scale, scale)
    gradient = np.empty(k)
    for i in range(k - 1):
        step = np.eye(k - 1)[i] * 1e-6 * scale[i]
        gradient[i] = (
            log_life(values + step, use) - log_life(values - step, use)
        ) / (2e-6 * scale[i])
    gradient[-1] = (
        np.log(-np.log1p(-p)) if family == "sev" else stats.norm.ppf(p)
    )
    return float(gradient @ np.linalg.solve(-hessian, gradient))


@pytest.mark.parametrize(
    "case, stresses, units, use, censor",
    [
        (VOLTAGE, (30.0, 45.0, 60.0), (20, 10, 10), 10.0, 1000.0),
        (VOLTAGE, (25.0, 60.0), (30, 10), 10.0, 500.0),
        (THERMAL, (380.0, 400.0, 420.0), (24, 12, 12), 330.0, 3000.0),
    ],
)
def test_the_variance_is_the_expected_curvatures(
    case, stresses, units, use, censor
):
    plan = accelerated_test_precision(
        **case,
        use_stress=use,
        stresses=stresses,
        units=units,
        censor_time=censor,
    )
    n = sum(units)
    shares = [u / n for u in units]
    expected = finite_difference_variance(
        case, stresses, shares, use, censor, 0.1
    )
    assert plan.standard_error**2 * n == pytest.approx(expected, rel=1e-3)


def test_the_plan_estimates_the_quantile_at_use():
    plan = accelerated_test_precision(
        **VOLTAGE,
        use_stress=10.0,
        stresses=(40.0, 60.0),
        units=(20, 20),
        censor_time=800.0,
        p=0.05,
    )
    life = 1e6 * 10.0**-2.0
    assert plan.estimate == pytest.approx(
        float(surpyval.Weibull.qf(0.05, life, 2.0))
    )
    chance = surpyval.Weibull.ff(800.0, 1e6 * 40.0**-2.0, 2.0)
    assert plan.expected_failures[0] == pytest.approx(20 * chance)


def standard_error(case, stresses, units, **common):
    return accelerated_test_precision(
        **case, stresses=stresses, units=units, **common
    ).standard_error


@pytest.mark.parametrize("case", ["voltage", "thermal"])
def test_the_optimum_plan_beats_its_neighbours(case):
    if case == "voltage":
        case = VOLTAGE
        common = dict(use_stress=10.0, censor_time=1000.0)
        high = 60.0
    else:
        case = THERMAL
        common = dict(use_stress=330.0, censor_time=3000.0)
        high = 420.0
    plan = accelerated_test_plan(**case, high_stress=high, n=1000, **common)
    assert plan.stresses[-1] == high
    low, units = plan.stresses[0], plan.units
    best = plan.standard_error
    for factor in (0.97, 1.03):
        moved = standard_error(case, (low * factor, high), units, **common)
        assert moved > best
    for shift in (-30, 30):
        shared = (units[0] + shift, units[1] - shift)
        assert standard_error(case, plan.stresses, shared, **common) > best


def test_an_uncensored_test_is_best_run_at_the_use_stress():
    # With every unit run to failure there is nothing to extrapolate
    # over: the optimum's low stress falls to the use stress.
    plan = accelerated_test_plan(
        **VOLTAGE,
        use_stress=10.0,
        high_stress=60.0,
        n=100,
        censor_time=1e12,
    )
    assert plan.stresses[0] == pytest.approx(10.0, rel=0.02)
    assert plan.fractions[1] < 0.01 and plan.units == (99, 1)


def test_compromise_levels_are_equally_spaced_in_log_life():
    plan = accelerated_test_plan(
        **VOLTAGE,
        use_stress=10.0,
        high_stress=60.0,
        n=70,
        censor_time=1000.0,
        levels=4,
        middle_fraction=0.3,
    )
    # The power law's log life is linear in log V.
    assert np.diff(np.diff(np.log(plan.stresses))) == pytest.approx(
        [0.0, 0.0], abs=1e-9
    )
    assert plan.fractions[1] == pytest.approx(0.15)
    assert plan.fractions[2] == pytest.approx(0.15)
    assert sum(plan.units) == 70
    thermal = accelerated_test_plan(
        **THERMAL,
        use_stress=330.0,
        high_stress=420.0,
        n=60,
        censor_time=3000.0,
        levels=3,
    )
    # Arrhenius' log life is linear in 1 / T.
    inverse = 1.0 / np.array(thermal.stresses)
    assert inverse[0] - inverse[1] == pytest.approx(inverse[1] - inverse[2])


def test_a_fixed_allocation_is_kept():
    plan = accelerated_test_plan(
        **VOLTAGE,
        use_stress=10.0,
        high_stress=60.0,
        n=70,
        censor_time=1000.0,
        levels=3,
        allocation=(4 / 7, 2 / 7, 1 / 7),
    )
    assert plan.units == (40, 20, 10)
    assert plan.fractions == pytest.approx((4 / 7, 2 / 7, 1 / 7))


def test_a_compromise_costs_precision_but_not_much():
    common = dict(use_stress=10.0, high_stress=60.0, n=50, censor_time=1000.0)
    optimum = accelerated_test_plan(**VOLTAGE, **common)
    compromise = accelerated_test_plan(**VOLTAGE, levels=3, **common)
    assert optimum.precision < compromise.precision
    assert np.log(compromise.precision) < 1.3 * np.log(optimum.precision)


def test_the_life_parameters_may_be_given_in_order():
    by_name = accelerated_test_precision(
        **VOLTAGE,
        use_stress=10.0,
        stresses=(30.0, 60.0),
        units=(10, 10),
        censor_time=1000.0,
    )
    case = dict(VOLTAGE, life_params=(1e6, -2.0))
    in_order = accelerated_test_precision(
        **case,
        use_stress=10.0,
        stresses=(60.0, 30.0),  # in any order: sorted
        units=(10, 10),
        censor_time=1000.0,
    )
    assert in_order.standard_error == by_name.standard_error
    assert in_order.stresses == (30.0, 60.0)


def test_simulate_fits_the_accelerated_life_model():
    plan = accelerated_test_precision(
        **VOLTAGE,
        use_stress=10.0,
        stresses=(30.0, 60.0),
        units=(30, 15),
        censor_time=1000.0,
    )
    check = plan.simulate(25, random_state=5)
    assert check.n_sim == 25 and check.unusable == 0
    assert 1.0 < check.precision < plan.precision**2


def test_the_plan_prints_the_test():
    plan = accelerated_test_plan(
        **THERMAL,
        use_stress=330.0,
        high_stress=420.0,
        n=40,
        censor_time=3000.0,
    )
    assert isinstance(plan, AcceleratedTestPlan)
    text = repr(plan)
    assert "Exponential(a=6000, b=0.00062), sigma=0.6" in text
    assert "B10 at 330" in text


@pytest.mark.parametrize(
    "call, match",
    [
        (
            lambda: accelerated_test_plan(
                surpyval.Gamma,
                life_models.Power,
                (1, 1),
                1.0,
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
            ),
            "Weibull",
        ),
        (
            lambda: accelerated_test_plan(
                surpyval.Weibull,
                life_models.DualPower,
                (1, 1, 1),
                1.0,
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
            ),
            "single-stress",
        ),
        (
            lambda: accelerated_test_plan(
                **dict(VOLTAGE, life_params={"a": 1.0}),
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
            ),
            "name the life model",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=-1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
            ),
            "positive stresses",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=2.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
            ),
            "shorter",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=20.0,
                high_stress=10.0,
                n=10,
                censor_time=1.0,
            ),
            "shorter",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
                levels=3,
                allocation=(0.5, 0.5),
            ),
            "each of the 3",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
                levels=3,
                middle_fraction=1.0,
            ),
            "middle_fraction",
        ),
        (
            lambda: accelerated_test_plan(
                **VOLTAGE,
                use_stress=1.0,
                high_stress=2.0,
                n=10,
                censor_time=1.0,
                levels=1,
            ),
            "levels",
        ),
        (
            lambda: accelerated_test_precision(
                **VOLTAGE,
                use_stress=1.0,
                stresses=(2.0, 3.0),
                units=(10,),
                censor_time=1.0,
            ),
            "same length",
        ),
        (
            lambda: accelerated_test_precision(
                **VOLTAGE,
                use_stress=1.0,
                stresses=(2.0, 3.0),
                units=(10, 0),
                censor_time=1.0,
            ),
            "as many stresses",
        ),
    ],
)
def test_plans_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call()
