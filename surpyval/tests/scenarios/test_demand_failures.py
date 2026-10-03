"""Card: the probability of failure on demand.

Persona: a reliability engineer with start-up records for an inverter
fleet -- 3 failures in 1200 starts, and a second unit type with none in
1200 -- and monthly batches of starts of different sizes. Questions
(Meeker and Escobar ch. 3.3; IEC 61508-6 annex D; Clopper and Pearson,
1934):

1. What is the probability of failure on demand, with a confidence
   interval (Clopper-Pearson, the standard for proportions)?
2. With no failures, what upper bound has been demonstrated?
3. From monthly batches of different sizes, the same.
"""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist

import surpyval as sp

FAILURES, DEMANDS = 3, 1200


def _clopper_pearson(k, n, alpha_ci):
    lower = beta_dist.ppf(alpha_ci / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta_dist.ppf(1 - alpha_ci / 2, k + 1, n - k)
    return lower, upper


def test_point_estimate():
    model = sp.Bernoulli.fit(x=[1, 0], n=[FAILURES, DEMANDS - FAILURES])
    (p,) = model.params
    assert model.parameter_names == ["p"]
    assert p == pytest.approx(FAILURES / DEMANDS)
    assert model.sf(0) == pytest.approx(FAILURES / DEMANDS)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#580: model.p is the limited-failure-population fraction (1), "
        "not the Bernoulli parameter named p (0.0025)"
    ),
)
def test_the_attribute_p_is_the_parameter_p():
    model = sp.Bernoulli.fit(x=[1, 0], n=[FAILURES, DEMANDS - FAILURES])
    assert model.p == pytest.approx(FAILURES / DEMANDS)


def test_success_run_bound():
    # After n successes, the lower 90% bound on reliability is 0.1**(1/n).
    assert sp.success_run(DEMANDS, 0.9) == pytest.approx(0.1 ** (1 / DEMANDS))


@pytest.mark.xfail(
    strict=True,
    reason="#580: Bernoulli and Binomial fits give no bounds on p",
)
def test_interval_on_the_demand_failure_probability():
    model = sp.Bernoulli.fit(x=[1, 0], n=[FAILURES, DEMANDS - FAILURES])
    lower, upper = sorted(np.ravel(model.param_cb("p", alpha_ci=0.1)))
    exact = _clopper_pearson(FAILURES, DEMANDS, 0.1)
    # Wald or exact: either brackets the estimate within the exact band's
    # order of magnitude.
    assert lower < FAILURES / DEMANDS < upper
    assert upper == pytest.approx(exact[1], rel=0.5)


@pytest.mark.xfail(
    strict=True,
    reason="#580: no bound on p with zero failures",
)
def test_zero_failure_upper_bound():
    model = sp.Bernoulli.fit(x=[0], n=[DEMANDS])
    upper = max(np.ravel(model.param_cb("p", alpha_ci=0.1, bound="upper")))
    assert upper == pytest.approx(1 - 0.1 ** (1 / DEMANDS), rel=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="#580: success_run takes confidence/alpha, not alpha_ci",
)
def test_success_run_takes_alpha_ci():
    assert sp.success_run(DEMANDS, alpha_ci=0.1) == pytest.approx(
        0.1 ** (1 / DEMANDS)
    )


@pytest.mark.xfail(
    strict=True,
    reason="#580: Binomial.fit takes one n_trials for every row",
)
def test_batches_of_different_sizes():
    failures = np.array([1, 0, 2, 0])
    trials = np.array([100, 120, 95, 110])
    model = sp.Binomial.fit(x=failures, n_trials=trials)
    assert model.params[-1] == pytest.approx(failures.sum() / trials.sum())
