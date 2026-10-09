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


def test_the_attribute_p_is_the_parameter_p():
    # #580: model.p was the limited-failure-population fraction (1); it is
    # the Bernoulli parameter named p (that fraction is lfp_p).
    model = sp.Bernoulli.fit(x=[1, 0], n=[FAILURES, DEMANDS - FAILURES])
    assert model.p == pytest.approx(FAILURES / DEMANDS)


def test_success_run_bound():
    # After n successes, the lower 90% bound on reliability is 0.1**(1/n).
    # (#580: the level is alpha_ci, as for every bound; confidence and
    # alpha are gone.)
    assert sp.success_run(DEMANDS, alpha_ci=0.1) == pytest.approx(
        0.1 ** (1 / DEMANDS)
    )


def test_interval_on_the_demand_failure_probability():
    # #580: the bounds on p are exact (Clopper-Pearson) by default.
    model = sp.Bernoulli.fit(x=[1, 0], n=[FAILURES, DEMANDS - FAILURES])
    bounds = np.ravel(model.param_cb("p", alpha_ci=0.1))
    np.testing.assert_allclose(
        bounds, _clopper_pearson(FAILURES, DEMANDS, 0.1), rtol=1e-6
    )


def test_zero_failure_upper_bound():
    # #580: with no failures the exact upper bound on p is still there, and
    # it is the complement of the success-run bound.
    model = sp.Bernoulli.fit(x=[0], n=[DEMANDS])
    upper = max(np.ravel(model.param_cb("p", alpha_ci=0.1, bound="upper")))
    assert upper == pytest.approx(1 - 0.1 ** (1 / DEMANDS), rel=1e-6)
    assert upper == pytest.approx(
        1 - sp.success_run(DEMANDS, alpha_ci=0.1), rel=1e-6
    )


def test_batches_of_different_sizes():
    # #580: Binomial.fit takes the number of trials of each batch; the MLE
    # of p is the pooled proportion.
    failures = np.array([1, 0, 2, 0])
    trials = np.array([100, 120, 95, 110])
    model = sp.Binomial.fit(x=failures, n_trials=trials)
    assert model.params[-1] == pytest.approx(failures.sum() / trials.sum())
