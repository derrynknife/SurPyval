"""Bounds on the conditional survival and the chance of failing in the
next window, ``cs_cb``, of the univariate and regression models."""

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm

import surpyval as sp


@pytest.fixture(scope="module")
def weibull():
    np.random.seed(1)
    x = sp.Weibull.random(30, 10, 3)
    return x, sp.Weibull.fit(x)


def test_at_age_zero_it_is_cb(weibull):
    _, model = weibull
    x = [3.0, 5.0, 8.0]
    for method in ("wald", "lr"):
        np.testing.assert_allclose(
            model.cs_cb(x, 0.0, method=method),
            model.cb(x, method=method),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            model.cs_cb(x, 0.0, on="ff", method=method),
            model.cb(x, on="ff", method=method),
            rtol=1e-9,
        )


def test_wald_is_the_delta_method_on_log_window_hazard(weibull):
    # Delta H = ((g + x)^b - g^b) / a^b; its gradient in (a, b) by hand.
    _, model = weibull
    a, b = model.params
    g, w = 8.0, 2.0
    u, v = (g + w) ** b, g**b
    dH = (u - v) / a**b
    d_a = -b * dH / a
    d_b = (u * np.log(g + w) - v * np.log(g)) / a**b - dH * np.log(a)
    grad = np.array([d_a, d_b])
    se_log = np.sqrt(grad @ model.covariance() @ grad) / dH
    zq = norm.ppf(0.975)
    H_cb = dH * np.exp([-zq * se_log, zq * se_log])
    np.testing.assert_allclose(model.cs_cb(w, g, on="Hf"), H_cb, rtol=1e-6)
    np.testing.assert_allclose(
        model.cs_cb(w, g, on="ff"), -np.expm1(-H_cb), rtol=1e-6
    )
    np.testing.assert_allclose(
        model.cs_cb(w, g), np.exp(-H_cb[::-1]), rtol=1e-6
    )
    assert model.cs_cb(w, g)[0] < model.cs(w, g) < model.cs_cb(w, g)[1]


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_lr_is_the_profile_likelihood_interval(weibull, alpha):
    # Reparameterise by d = Delta H over (8, 10]: a = (((10)^b - 8^b) /
    # d)^(1/b); profile the log-likelihood over b and solve the deviance
    # for chi2_1(1 - alpha) on each side, independently of the searches.
    x, model = weibull
    g, w = 8.0, 2.0

    def ll(a, b):
        return np.sum(sp.Weibull.log_df(x, a, b))

    ll_hat = ll(*model.params)

    def deviance(d):
        def neg(b):
            a = (((g + w) ** b - g**b) / d) ** (1 / b)
            return -ll(a, b)

        found = minimize_scalar(
            neg, bounds=(0.5, 10), method="bounded", options={"xatol": 1e-12}
        )
        return 2 * (ll_hat + found.fun) - chi2.ppf(1 - alpha, 1)

    a_hat, b_hat = model.params
    d_hat = ((g + w) ** b_hat - g**b_hat) / a_hat**b_hat
    lower = brentq(deviance, 1e-3, d_hat, xtol=1e-12)
    upper = brentq(deviance, d_hat, 10.0, xtol=1e-12)
    # 95%: Delta H in [0.3526, 0.7859]
    np.testing.assert_allclose(
        model.cs_cb(w, g, on="Hf", alpha_ci=alpha, method="lr"),
        [lower, upper],
        rtol=1e-5,
    )


def test_an_exponential_window_does_not_depend_on_age():
    np.random.seed(2)
    model = sp.Exponential.fit(sp.Exponential.random(40, 0.1))
    for method in ("wald", "lr"):
        out = model.cs_cb(3.0, [0.0, 5.0, 20.0], method=method)
        np.testing.assert_allclose(out, np.tile(out[0], (3, 1)), rtol=1e-9)
        np.testing.assert_allclose(
            out[0], model.cb(3.0, method=method), rtol=1e-9
        )


def test_shapes_sides_and_edges(weibull):
    _, model = weibull
    assert model.cs_cb([1, 2, 3], [8, 8, 8]).shape == (3, 2)
    assert model.cs_cb([1, 2], [[4], [8]]).shape == (2, 2, 2)
    two = model.cs_cb(2, 8, alpha_ci=0.1)
    assert model.cs_cb(2, 8, bound="lower") == pytest.approx(two[0])
    assert model.cs_cb(2, 8, bound="upper") == pytest.approx(two[1])
    np.testing.assert_array_equal(model.cs_cb(0.0, 8.0), [1.0, 1.0])
    with pytest.raises(ValueError, match="at or above 0"):
        model.cs_cb(-1.0, 8.0)
    with pytest.raises(ValueError, match="no bootstrap"):
        model.cs_cb(1.0, 8.0, method="bootstrap")
    with pytest.raises(ValueError, match="on"):
        model.cs_cb(1.0, 8.0, on="hf")


def test_an_offset_and_a_limited_failure_population():
    np.random.seed(4)
    x = sp.Weibull.random(40, 10, 2) + 5
    model = sp.Weibull.fit(x, offset=True)
    with pytest.warns(UserWarning, match="offset"):
        wald = model.cs_cb(2.0, 8.0)
    lr = model.cs_cb(2.0, 8.0, method="lr")
    for out in (wald, lr):
        assert out[0] < model.cs(2.0, 8.0) < out[1]
    # A window wholly before the offset: nothing can fail in it
    np.testing.assert_array_equal(model.cs_cb(1.0, 2.0, method="lr"), 1.0)
    t = np.concatenate([sp.Weibull.random(40, 10, 2), np.full(20, 30.0)])
    c = np.r_[np.zeros(40), np.ones(20)]
    lfp = sp.Weibull.fit(t, c=c, lfp=True)
    for method in ("wald", "lr"):
        out = lfp.cs_cb(5.0, 10.0, on="ff", method=method)
        assert out[0] < 1 - lfp.cs(5.0, 10.0) < out[1]


# -- regression ------------------------------------------------------------


@pytest.fixture(scope="module")
def weibull_ph():
    np.random.seed(1)
    Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    x = sp.Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    return sp.WeibullPH.fit(x, Z)


def test_regression_at_age_zero_it_is_cb(weibull_ph):
    model = weibull_ph
    for method in ("wald", "lr"):
        np.testing.assert_allclose(
            model.cs_cb(5.0, 0.0, [1], method=method),
            model.cb(5.0, [1], method=method),
            rtol=1e-9,
        )


def test_regression_wald_by_hand(weibull_ph):
    # Delta H = exp(beta z) ((g + x)^k - g^k) / a^k for the Weibull PH:
    # its gradient in (a, k, beta) by hand.
    model = weibull_ph
    a, k, beta = model.params
    g, w, z = 8.0, 1.0, 1.0
    u, v = (g + w) ** k, g**k
    dH = np.exp(beta * z) * (u - v) / a**k
    grad = np.array(
        [
            -k * dH / a,
            np.exp(beta * z) * ((u * np.log(g + w) - v * np.log(g)) / a**k)
            - dH * np.log(a),
            z * dH,
        ]
    )
    se_log = np.sqrt(grad @ model.covariance() @ grad) / dH
    zq = norm.ppf(0.975)
    H_cb = dH * np.exp([-zq * se_log, zq * se_log])
    # 1 - cs = 0.3157, 95% [0.2372, 0.4122]
    np.testing.assert_allclose(
        model.cs_cb(w, g, [z], on="ff"), -np.expm1(-H_cb), rtol=1e-5
    )


def test_regression_exponential_window_does_not_depend_on_age():
    np.random.seed(3)
    Z = np.random.binomial(1, 0.5, 80).reshape(-1, 1)
    x = sp.Exponential.random(80, 0.1) * np.exp(-0.5 * Z[:, 0])
    model = sp.ExponentialPH.fit(x, Z)
    for method in ("wald", "lr"):
        out = model.cs_cb(3.0, [0.0, 5.0, 20.0], [1], method=method)
        np.testing.assert_allclose(out, np.tile(out[0], (3, 1)), rtol=1e-6)


def test_regression_pairs_bootstrap_and_sides(weibull_ph):
    model = weibull_ph
    out = model.cs_cb([1.0, 2.0], [8.0, 4.0], [[1], [0]])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[1], model.cs_cb(2.0, 4.0, [0]), rtol=1e-12)
    assert model.cs_cb(1.0, 8.0, [[0], [1]]).shape == (2, 2)
    two = model.cs_cb(1.0, 8.0, [1], alpha_ci=0.1, on="ff")
    lower = model.cs_cb(1.0, 8.0, [1], bound="lower", on="ff")
    assert float(np.ravel(lower)[0]) == pytest.approx(two[..., 0].item())
    boot = model.cs_cb(
        1.0, 8.0, [1], on="ff", method="bootstrap", n_boot=40, random_state=1
    )
    point = 1 - model.cs(1.0, 8.0, [1])
    assert np.ravel(boot)[0] < point < np.ravel(boot)[1]
    with pytest.raises(ValueError, match="at or above 0"):
        model.cs_cb(-1.0, 8.0, [1])
