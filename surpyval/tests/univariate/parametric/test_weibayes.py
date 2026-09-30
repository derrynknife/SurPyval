"""Weibayes: the zero-failure bound on a Weibull scale of known shape (#493).

``Weibull.fit`` refuses data with no failure, fixed shape or not (the
likelihood has no maximum); its message now points at ``weibayes``, which
gives the standard bound (Nelson, 1985; Abernethy, The New Weibull
Handbook, ch. 6): alpha_L = (2 sum(n x^beta) / chi2(1 - alpha_ci; 2r + 2))
** (1 / beta).
"""

import numpy as np
import pytest
from scipy.stats import chi2

import surpyval as sp


def test_the_issues_weibull_zero_failure_case():
    # 10 units, 500 h, no failure, beta = 2: (2 * 2.5e6 / 5.991) ** 0.5
    model = sp.weibayes([500] * 10, c=[1] * 10, beta=2)
    assert model.params[0] == pytest.approx(913.5209, rel=1e-6)
    assert model.params[1] == 2
    # the bound on R(500) is the success-run bound for ten passes
    assert model.sf(500) == pytest.approx(sp.success_run(10), rel=1e-12)


def test_the_issues_exponential_zero_failure_case():
    # 5 units, 1000 h, no failure: MTBF_L = 2T / chi2(0.6; 2) at 60 %
    model = sp.weibayes([1000] * 5, c=[1] * 5, alpha_ci=0.4)
    assert model.params[0] == pytest.approx(10_000 / chi2.ppf(0.6, 2))
    assert model.params[0] == pytest.approx(5456.783, rel=1e-6)


def test_at_63_percent_it_is_the_one_assumed_failure_estimate():
    # Abernethy's Weibayes line: the fixed-shape MLE with r = 1
    x = np.array([200.0, 350, 350, 800])
    model = sp.weibayes(x, c=[1] * 4, beta=1.7, alpha_ci=np.exp(-1))
    assert model.params[0] == pytest.approx(np.sum(x**1.7) ** (1 / 1.7))


def test_with_failures_it_uses_2r_plus_2_degrees_of_freedom():
    x = [300, 500, 800, 900, 1000]
    c = [0, 1, 1, 0, 1]
    n = [1, 2, 1, 1, 3]
    model = sp.weibayes(x, c, n, beta=1.5, alpha_ci=0.1)
    total = np.sum(np.array(n) * np.array(x, float) ** 1.5)
    want = (2 * total / chi2.ppf(0.9, 2 * 2 + 2)) ** (1 / 1.5)
    assert model.params[0] == pytest.approx(want, rel=1e-12)
    # below the fixed-shape maximum-likelihood scale (sum / r) ** (1/beta)
    mle = sp.Weibull.fit(x, c, n, fixed={"beta": 1.5})
    assert model.params[0] < mle.params[0]


def test_coverage_of_a_time_terminated_test():
    # Type I censoring at tau: the bound holds at least 95 % of the time
    # (97.1 % over 10 000 runs: 2r + 2 degrees of freedom are conservative)
    rng = np.random.default_rng(4)
    eta, beta, tau, units = 1000.0, 2.0, 1000.0, 8
    covered = 0
    reps = 4000
    for _ in range(reps):
        t = eta * rng.weibull(beta, units)
        c = (t > tau).astype(int)
        x = np.minimum(t, tau)
        covered += sp.weibayes(x, c, beta=beta).params[0] <= eta
    assert covered / reps >= 0.95 - 3 * np.sqrt(0.95 * 0.05 / reps)


def test_fit_still_refuses_and_points_at_weibayes():
    for kwargs in ({}, {"fixed": {"beta": 2}}):
        with pytest.raises(ValueError, match="surpyval.weibayes"):
            sp.Weibull.fit([500] * 10, c=[1] * 10, **kwargs)
    with pytest.raises(ValueError, match="only right censored"):
        sp.Exponential.fit([1000] * 5, c=[1] * 5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"x": [1, 2], "c": [0, -1]}, "left and interval"),
        ({"x": [1, 2], "beta": 0}, "'beta'"),
        ({"x": [1, 2], "beta": np.nan}, "'beta'"),
        ({"x": [1, 2], "alpha_ci": 1.0}, "'alpha_ci'"),
        ({"x": [0, 2], "c": [1, 1]}, "positive"),
    ],
)
def test_invalid_input_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sp.weibayes(**kwargs)


def test_a_large_shape_does_not_overflow():
    model = sp.weibayes([1e6] * 3, c=[1] * 3, beta=80)
    assert np.isfinite(model.params[0]) and model.params[0] > 1e6
