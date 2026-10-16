"""Planning a life test for estimation precision (#863).

The oracles: the complete-data information of the smallest extreme value
and normal distributions in closed form; the information of a censored
test as the curvature of its expected log-likelihood, found by finite
differences (the planning code integrates the scores instead); and the
test simulated and fitted.
"""

from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

import surpyval
from surpyval.test_planning import (
    LifeTestPlan,
)
from surpyval.test_planning import _information as info
from surpyval.test_planning import (
    life_test_length,
    life_test_precision,
    life_test_sample_size,
)

WEIBULL = (surpyval.Weibull, (1000.0, 2.0))
LOGNORMAL = (surpyval.LogNormal, (6.0, 0.6))

# Wrong types, passed on purpose to the checks below.
ONE: Any = 1


def test_complete_information_in_closed_form():
    gamma = np.euler_gamma
    sev = info.unit_information("sev", np.inf)
    expected = [[1.0, 1 - gamma], [1 - gamma, np.pi**2 / 6 + (1 - gamma) ** 2]]
    np.testing.assert_allclose(sev, expected, rtol=1e-9)
    normal = info.unit_information("normal", np.inf)
    np.testing.assert_allclose(normal, [[1, 0], [0, 2]], atol=1e-12)


def expected_log_likelihood(family, zeta, mu, sigma):
    """One unit's expected log-likelihood at (mu, sigma), the truth being
    (0, 1) and the unit censored at the standardised time zeta."""
    if family == "sev":
        log_pdf = stats.gumbel_l.logpdf
        log_sf = stats.gumbel_l.logsf
        density = stats.gumbel_l.pdf
        survives = stats.gumbel_l.sf(zeta)
    else:
        log_pdf, log_sf = stats.norm.logpdf, stats.norm.logsf
        density, survives = stats.norm.pdf, stats.norm.sf(zeta)

    def failed(y):
        return (log_pdf((y - mu) / sigma) - np.log(sigma)) * density(y)

    value, _ = integrate.quad(failed, -40, zeta, limit=200, epsabs=1e-13)
    return value + survives * log_sf((zeta - mu) / sigma)


@pytest.mark.parametrize("family", ["sev", "normal"])
@pytest.mark.parametrize("zeta", [-2.0, -0.5, 0.0, 1.0])
def test_censored_information_is_the_expected_curvature(family, zeta):
    h = 1e-3

    def ell(a, b):
        return expected_log_likelihood(family, zeta, a, 1.0 + b)

    hessian = np.empty((2, 2))
    hessian[0, 0] = (ell(h, 0) - 2 * ell(0, 0) + ell(-h, 0)) / h**2
    hessian[1, 1] = (ell(0, h) - 2 * ell(0, 0) + ell(0, -h)) / h**2
    hessian[0, 1] = hessian[1, 0] = (
        ell(h, h) - ell(h, -h) - ell(-h, h) + ell(-h, -h)
    ) / (4 * h**2)
    np.testing.assert_allclose(
        info.unit_information(family, zeta), -hessian, atol=2e-4
    )


def test_simulate_fits_the_planned_test():
    # A quick check of the simulation itself; the study of the
    # large-sample precision against simulation is in tests/calibration.
    plan = life_test_precision(*WEIBULL, 40, censor_time=600.0)
    check = plan.simulate(40, random_state=4)
    assert check.n_sim == 40 and check.unusable == 0
    assert 0.8 <= check.within <= 1.0
    assert np.log(check.precision) == pytest.approx(
        np.log(plan.precision), rel=0.4
    )
    again = plan.simulate(40, random_state=4)
    assert again == check
    few = life_test_precision(*WEIBULL, 5, censor_time=50.0).simulate(
        10, random_state=1
    )
    assert few.unusable > 0


def test_the_estimate_is_the_planned_quantity():
    plan = life_test_precision(*WEIBULL, 20)
    assert plan.estimate == pytest.approx(
        float(surpyval.Weibull.qf(0.1, *WEIBULL[1]))
    )
    mean = life_test_precision(*WEIBULL, 20, quantity="mean")
    assert mean.estimate == pytest.approx(
        float(surpyval.Weibull.mean(*WEIBULL[1]))
    )
    median = life_test_precision(*LOGNORMAL, 20, p=0.5)
    assert median.estimate == pytest.approx(np.exp(6.0))


def test_the_variance_falls_as_one_over_n():
    small = life_test_precision(*WEIBULL, 25, censor_time=400.0)
    large = life_test_precision(*WEIBULL, 100, censor_time=400.0)
    assert large.standard_error == pytest.approx(small.standard_error / 2)
    assert large.expected_failures == pytest.approx(
        4 * small.expected_failures
    )


def test_a_longer_test_is_more_precise():
    times = [200.0, 400.0, 800.0, 1600.0, 3200.0]
    errors = [
        life_test_precision(*WEIBULL, 30, censor_time=t).standard_error
        for t in times
    ]
    assert np.all(np.diff(errors) < 0)
    complete = life_test_precision(*WEIBULL, 30)
    assert complete.standard_error < errors[-1]
    assert life_test_precision(
        *WEIBULL, 30, censor_time=1e9
    ).standard_error == pytest.approx(complete.standard_error)
    every = life_test_precision(*WEIBULL, 30, failures=30)
    assert every.standard_error == complete.standard_error


def test_the_sample_size_is_the_fewest_that_reach_the_precision():
    for options in (
        dict(censor_time=600.0),
        dict(failure_fraction=0.4),
        dict(),
        dict(quantity="mean", alpha_ci=0.1),
    ):
        plan = life_test_sample_size(*WEIBULL, 1.25, **options)
        assert plan.precision <= 1.25
        options.pop("failure_fraction", None)
        if "censor_time" not in options and plan.failures is not None:
            options["failures"] = int(np.ceil(0.4 * (plan.n - 1)))
        fewer = life_test_precision(*WEIBULL, plan.n - 1, **options)
        assert fewer.precision > 1.25 - 0.01


def test_the_length_reaches_the_precision_exactly():
    plan = life_test_length(*LOGNORMAL, 60, 1.3)
    assert plan.precision == pytest.approx(1.3, rel=1e-9)
    shorter = life_test_precision(
        *LOGNORMAL, 60, censor_time=0.99 * plan.censor_time
    )
    assert shorter.precision > 1.3
    stopped = life_test_length(*LOGNORMAL, 60, 1.3, failure_censored=True)
    assert stopped.precision <= 1.3
    earlier = life_test_precision(
        *LOGNORMAL, 60, failures=stopped.failures - 1
    )
    assert earlier.precision > 1.3


def test_the_plan_prints_the_test():
    plan = life_test_precision(*WEIBULL, 12, failures=6)
    assert isinstance(plan, LifeTestPlan)
    text = repr(plan)
    assert "failure-censored at failure 6" in text
    assert "B10" in text and "95% interval" in text
    assert "mean" in repr(life_test_precision(*WEIBULL, 12, quantity="mean"))
    assert "every unit" in repr(life_test_precision(*WEIBULL, 12))


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: life_test_precision(surpyval.Gamma, (1, 2), 10), "Weibull"),
        (lambda: life_test_precision(surpyval.Weibull, (1.0,), 10), "two"),
        (
            lambda: life_test_precision(surpyval.Weibull, (-1.0, 2.0), 10),
            "positive",
        ),
        (
            lambda: life_test_precision(surpyval.LogNormal, (1.0, 0.0), 10),
            "sigma",
        ),
        (lambda: life_test_precision(*WEIBULL, 0), "n must"),
        (
            lambda: life_test_precision(
                *WEIBULL, 10, censor_time=5.0, failures=3
            ),
            "not both",
        ),
        (lambda: life_test_precision(*WEIBULL, 10, failures=11), "at most"),
        (lambda: life_test_precision(*WEIBULL, 10, p=1.0), "p must"),
        (lambda: life_test_precision(*WEIBULL, 10, p=None), "give p"),
        (
            lambda: life_test_precision(*WEIBULL, 10, quantity="median"),
            "'quantity'",
        ),
        (lambda: life_test_sample_size(*WEIBULL, 1.0), "more than 1"),
        (
            lambda: life_test_sample_size(*WEIBULL, 1.2, failure_fraction=0),
            "failure_fraction",
        ),
        (lambda: life_test_length(*WEIBULL, 10, 1.05), "Not even"),
        (
            lambda: life_test_length(*WEIBULL, 10, 3.0, failure_censored=ONE),
            "True or False",
        ),
    ],
)
def test_plans_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call()
