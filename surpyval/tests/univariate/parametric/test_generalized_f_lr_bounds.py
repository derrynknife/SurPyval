"""The GeneralizedF's likelihood-ratio bounds at the edges of its space.

At n = 200 from GeneralizedF(2, 0.5, 0.5, 1), about 30% censored, the
likelihood-ratio bounds on the parameters covered 0.90-0.93: every miss
was a side reported as not found (``nan``), where an independent profile
(Nelder-Mead from several starts) put the truth inside. Two causes:

- a fit at the generalised gamma edge (P near 0), where the likelihood is
  flat in log P: the profiles' searches, scaled by a Wald standard error
  of log P in the thousands and started from P near 0, stayed there;
- the limit as P runs to infinity (sigma to 0, Q to plus or minus
  infinity), the log time an asymmetric Laplace, which on some samples
  fits better than any member: the profiles running to it rose above the
  fit's likelihood, which the searches refuse, and stopped.

Each case below is one of those samples (the coverage study's data, as
``calibration/test_coverage_new_families.py`` draws them).
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from surpyval import GeneralizedF

TRUE = [2.0, 0.5, 0.5, 1.0]
CRIT = 3.841458820694124  # chi-squared(1) at 0.95


def _sample(seed, n=200, c_max=25.0):
    rng = np.random.default_rng(seed)
    true = GeneralizedF.from_params(TRUE)
    t = true.qf(rng.uniform(size=n))
    cens = rng.uniform(0, c_max, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _fit(seed):
    x, c = _sample(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GeneralizedF.fit(x, c=c), x, c


def _ll(p, x, c):
    mu, sigma, Q, P = p
    if sigma <= 0 or P <= 0:
        return -np.inf
    with np.errstate(all="ignore"):
        v = np.sum(GeneralizedF.log_df(x[c == 0], mu, sigma, Q, P))
        v += np.sum(GeneralizedF.log_sf(x[c == 1], mu, sigma, Q, P))
    return v if np.isfinite(v) else -np.inf


def _profile_ll(x, c, idx, value, starts):
    """An independent profile: Nelder-Mead over the other parameters (log
    sigma and log P), from each start, the best kept."""
    free = [j for j in range(4) if j != idx]
    logs = {1, 3}

    def to_p(u):
        p = np.empty(4)
        p[idx] = value
        for k, j in enumerate(free):
            p[j] = np.exp(u[k]) if j in logs else u[k]
        return p

    best = -np.inf
    for s in starts:
        u0 = [np.log(s[j]) if j in logs else s[j] for j in free]
        res = minimize(
            lambda u: -_ll(to_p(u), x, c),
            u0,
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000},
        )
        best = max(best, -res.fun)
    return best


def _asymmetric_laplace_ll(mu, a, b, x, c):
    """The log-likelihood of the limit as P runs to infinity, in closed
    form: ln x is asymmetric Laplace about mu, with the rate a below it
    and b above it."""
    y = np.log(x)
    below = y < mu
    log_f = (
        np.log(a * b / (a + b))
        + np.where(below, a * (y - mu), -b * (y - mu))
        - y
    )
    sf = np.where(
        below,
        1.0 - b / (a + b) * np.exp(a * (y - mu)),
        a / (a + b) * np.exp(-b * (y - mu)),
    )
    return np.sum(log_f[c == 0]) + np.sum(np.log(sf[c == 1]))


def test_the_ridge_tends_to_the_asymmetric_laplace():
    x, c = _sample(3185372835)
    mu, a, b = 2.0, 1.7, 2.5
    want = _asymmetric_laplace_ll(mu, a, b, x, c)
    for step in (1e-6, 1e-8, GeneralizedF._RIDGE_C):
        p = GeneralizedF._ridge_params(mu, a, b, step)
        assert p[1] < 1e-2 and p[3] > 1e4 and p[2] > 0  # b > a: Q > 0
        assert _ll(p, x, c) == pytest.approx(want, abs=1e-5)
    p = GeneralizedF._ridge_params(mu, a, b, GeneralizedF._RIDGE_C)
    assert _ll(p, x, c) == pytest.approx(want, abs=1e-8)


def test_a_limit_that_fits_better_than_the_fit_is_the_reference():
    # The fit stops at a local maximum (P = 1.8); the limit as P runs to
    # infinity is 0.285 higher in log-likelihood, found independently.
    model, x, c = _fit(3185372835)
    assert 1 < model.params[3] < 3
    limits = model._lr_family_limits()
    nll_fit = -_ll(np.asarray(model.params), x, c)
    assert limits[(3, 1.0)] == pytest.approx(nll_fit - 0.2855, abs=2e-3)
    assert model._lr_reference_nll() == limits[(3, 1.0)]
    # b > a there, so it is Q's upper side that runs to it
    assert limits[(2, 1.0)] == limits[(3, 1.0)] < limits[(2, -1.0)]


def test_sigma_runs_to_zero_on_the_way_to_a_better_limit():
    model, x, c = _fit(3185372835)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # was nan, with a warning
        lower, upper = model.param_cb("sigma", method="lr")
    assert lower == 0.0
    assert np.isfinite(upper) and upper > model.params[1]
    # the upper bound is where the independent profile reaches crit
    ref = -model._lr_reference_nll()
    starts = [model.params, TRUE]
    dev = 2.0 * (ref - _profile_ll(x, c, 1, upper, starts))
    assert dev == pytest.approx(CRIT, abs=0.02)


def test_bound_found_from_the_generalised_gamma_edge():
    # The fit is at the generalised gamma edge (P = 4e-8); sigma's lower
    # bound was nan, its profile deviance overstated (4.67 for 2.99 at
    # sigma = 0.45) by searches that did not leave P near 0.
    model, x, c = _fit(388395554)
    assert model.params[3] < 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower = model.param_cb("sigma", method="lr", bound="lower")[0]
    # one-sided at 95%: the chi-squared(1) value at 0.90
    crit = 2.705543454095404
    ref = -model._lr_reference_nll()
    starts = [model.params, [*model.params[:3], 1.0], TRUE]
    dev = 2.0 * (ref - _profile_ll(x, c, 1, lower, starts))
    assert dev == pytest.approx(crit, abs=0.02)


def test_p_runs_to_infinity_where_the_limit_is_the_supremum():
    # A fit at the generalised gamma edge, where the limit at the other
    # end, P to infinity, is 0.36 higher in log-likelihood: P's interval
    # is the whole of its space (its upper bound was nan, and then a
    # crossing read in noise at 1.1e15).
    model, x, c = _fit(997154490)
    assert model.params[3] < 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(model.param_cb("P", method="lr")) == [0.0, np.inf]
