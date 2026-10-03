"""Targeted review of ``parametric/distributions/expo_weibull.py`` (#399).

Each test pins a bug found by reading the module adversarially (all
fixed under #436; they were strict expected failures until then).
"""

import warnings

import numpy as np
from autograd import grad
from autograd import numpy as anp

from surpyval import ExpoWeibull, Weibull


def test_lower_tail_density_is_finite():
    # t = (x / alpha) ** beta = 1e-20
    x, alpha, beta, mu = 1e-4, 10.0, 4.0, 0.5
    t = (x / alpha) ** beta
    expected = (
        np.log(beta * mu)
        + (beta - 1) * np.log(x)
        - beta * np.log(alpha)
        + (mu - 1) * np.log(-np.expm1(-t))
        - t
    )
    np.testing.assert_allclose(
        ExpoWeibull.log_df(x, alpha, beta, mu), expected, rtol=1e-10
    )
    np.testing.assert_allclose(
        ExpoWeibull.df(x, alpha, beta, mu), np.exp(expected), rtol=1e-10
    )


def test_lower_tail_cdf_is_accurate():
    alpha, beta, mu = 10.0, 4.0, 2.0
    x = np.array([1e-3, 1e-4])
    t = (x / alpha) ** beta
    expected = np.power(-np.expm1(-t), mu)
    np.testing.assert_allclose(
        ExpoWeibull.ff(x, alpha, beta, mu), expected, rtol=1e-10
    )
    np.testing.assert_allclose(
        ExpoWeibull.log_ff(x, alpha, beta, mu), np.log(expected), rtol=1e-10
    )


def test_far_right_tail_matches_weibull_at_mu_one():
    x = np.array([100.0, 300.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = (
            ExpoWeibull.hf(x, 10, 3, 1.0),
            ExpoWeibull.Hf(x, 10, 3, 1.0),
            ExpoWeibull.log_sf(x, 10, 3, 1.0),
        )
    want = (
        Weibull.hf(x, 10, 3),
        Weibull.Hf(x, 10, 3),
        Weibull.log_sf(x, 10, 3),
    )
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-8)


def test_moment_accepts_array_parameters():
    alpha = np.array([3.0, 4.0])
    got = ExpoWeibull.mean(alpha, 4.0, 1.2)
    want = [ExpoWeibull.mean(a, 4.0, 1.2) for a in alpha]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_qf_near_one_with_a_large_mu_is_finite():
    # u^(1/mu) rounds to 1 at mu = 500, and the direct -log1p(-u^(1/mu))
    # was inf; the true quantile is 4.2951408668e-5 (mpmath).
    got = ExpoWeibull.qf(0.9999999999999999, 1e-6, 1.0, 500.0)
    np.testing.assert_allclose(got, 4.2951408668099291e-5, rtol=1e-8)


def test_601_qf_is_finite_where_its_scale_factor_overflows():
    # alpha * exp(log(t) / beta) was 2.2e-308 * inf, though the quantile
    # is 19.4: the likelihood-ratio searches reach such parameters at the
    # end of alpha's coordinate, and read the quantile as inf there.
    alpha, beta, mu = np.finfo(float).tiny, 0.0076, 3e95
    t = -np.log(-np.expm1(np.log(0.95) / mu))
    want = np.exp(np.log(alpha) + np.log(t) / beta)
    np.testing.assert_allclose(
        ExpoWeibull.qf(0.95, alpha, beta, mu), want, rtol=1e-12
    )
    assert 19 < want < 20


def test_values_at_zero_are_the_limits():
    # beta * mu below, at and above 1: the density at 0 is inf, 1 / alpha
    # and 0; the rest are those of an empty CDF.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dfs = [ExpoWeibull.df(0.0, 10.0, 2.0, mu) for mu in (0.4, 0.5, 0.6)]
        rest = [
            getattr(ExpoWeibull, fn)(0.0, 10.0, 2.0, 1.3)
            for fn in ("sf", "ff", "Hf", "log_sf", "log_ff")
        ]
    np.testing.assert_allclose(dfs, [np.inf, 0.1, 0.0])
    np.testing.assert_array_equal(rest, [1.0, 0.0, 0.0, 0.0, -np.inf])


def test_log_forms_differentiate_with_autograd():
    # The fits differentiate the log-likelihood with autograd; the
    # branch-wise forms must give finite gradients that match finite
    # differences, in the lower tail, the centre and the right tail.
    x = np.array([1e-4, 1.0, 5.0, 20.0, 50.0])
    p0 = np.array([10.0, 2.0, 1.3])

    def ll(p):
        return (
            anp.sum(ExpoWeibull.log_df(x, p[0], p[1], p[2]))
            + anp.sum(ExpoWeibull.log_sf(x, p[0], p[1], p[2]))
            + anp.sum(ExpoWeibull.log_ff(x, p[0], p[1], p[2]))
            + anp.sum(ExpoWeibull.hf(x, p[0], p[1], p[2]))
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = grad(ll)(p0)
    fd = []
    for i in range(3):
        h = np.zeros(3)
        h[i] = 1e-6 * p0[i]
        fd.append((ll(p0 + h) - ll(p0 - h)) / (2 * h[i]))
    np.testing.assert_allclose(got, fd, rtol=1e-6)


def test_fit_with_a_tiny_value_is_at_least_as_good_as_weibull():
    # An ExpoWeibull with mu = 1 is a Weibull, so its best likelihood is
    # never worse. With one value at 1e-4 the old likelihood was NaN along
    # the way: the MLE failed and returned its start, neg_ll 124.97,
    # against the Weibull's 95.38 (now 90.97).
    rng = np.random.default_rng(1)
    x = np.r_[10 * rng.weibull(3, 30), 1e-4]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ExpoWeibull.fit(x)
        bounds = [model.param_cb(name) for name in ("alpha", "beta", "mu")]
    assert model.neg_ll() <= Weibull.fit(x).neg_ll()
    assert np.isfinite(bounds).all()
