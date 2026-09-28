"""Targeted review of ``parametric/distributions/expo_weibull.py`` (#399).

Each test pins a bug found by reading the module adversarially. They are
strict expected failures until the bug is fixed.
"""

import warnings

import numpy as np
import pytest

from surpyval import ExpoWeibull, Weibull


@pytest.mark.xfail(
    strict=True,
    reason="#436: ExpoWeibull df/log_df use 1 - exp(-t), which is 0 for "
    "t < 1e-16: log_df(1e-4, 10, 4, 0.5) is +inf, the true value -13.12",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="#436: ExpoWeibull ff/log_ff lose the lower tail: ff(1e-3, 10, "
    "4, 2) is 23% high (1.23e-32 for 1e-32) and ff(1e-4, ...) is 0 for "
    "1e-40",
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


@pytest.mark.xfail(
    strict=True,
    reason="#436: ExpoWeibull's far right tail underflows: with mu = 1 "
    "(a Weibull) at x = 100, alpha = 10, beta = 3, hf is nan (Weibull 30), "
    "Hf inf (1000) and log_sf -inf (-1000), with raw numpy warnings",
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


@pytest.mark.xfail(
    strict=True,
    reason="#436: ExpoWeibull.moment/mean raise TypeError for an array "
    "parameter, which their docstrings allow (Weibull.mean broadcasts)",
)
def test_moment_accepts_array_parameters():
    alpha = np.array([3.0, 4.0])
    got = ExpoWeibull.mean(alpha, 4.0, 1.2)
    want = [ExpoWeibull.mean(a, 4.0, 1.2) for a in alpha]
    np.testing.assert_allclose(got, want, rtol=1e-10)
