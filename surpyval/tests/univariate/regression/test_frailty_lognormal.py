"""The log-normal shared frailty (#343): ``Frailty(dist, family="lognormal")``.

The frailty is ``u = exp(w)`` with ``w ~ N(0, theta)`` (R's frailtypack,
coxme and ``survival::frailty(dist = "gaussian")``); each group's integral
is by adaptive Gauss-Hermite quadrature.
"""

import json
import warnings

import autograd
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

import surpyval as sp
from surpyval import Exponential, Frailty, Weibull
from surpyval.datasets import load_kidney
from surpyval.tests.reference._data import reference
from surpyval.univariate.regression.frailty.families import (
    N_NODES,
    kendall_tau,
    lognormal_log_integral,
    lognormal_mode,
)


def _quad_log_integral(D, H, theta):
    """``log int exp(D w - H e^w) phi(w; 0, theta) dw`` by scipy's adaptive
    quadrature, centred on the mode and broken at the integrand's edge."""
    mode, _ = lognormal_mode(np.array(D, float), np.array(H, float), theta)
    mode = float(mode)

    def g(w):
        return (
            D * w
            - H * np.exp(w)
            - w * w / (2 * theta)
            - 0.5 * np.log(2 * np.pi * theta)
        )

    top = g(mode)
    half = 40 * np.sqrt(theta)
    points = [mode] + ([-np.log(H)] if abs(-np.log(H) - mode) < half else [])
    val, _ = quad(
        lambda w: np.exp(g(w) - top),
        mode - half,
        mode + half,
        epsabs=0,
        epsrel=1e-13,
        limit=1000,
        points=sorted(points),
    )
    return top + np.log(val)


def _simulate(seed, family="lognormal", G=60, per=5, theta=0.5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), per)
    if family == "lognormal":
        u = np.exp(rng.normal(0.0, np.sqrt(theta), G))
    else:
        u = rng.gamma(1 / theta, theta, G)
    Z = rng.normal(size=(G * per, 1))
    # Weibull(10, 1.5) baseline: H(t) = (t / 10)^1.5 times the multiplier
    H = rng.exponential(size=G * per) / (np.exp(0.7 * Z[:, 0]) * u[g])
    t = 10.0 * H ** (1 / 1.5)
    c = (t > 25.0).astype(int)
    return np.minimum(t, 25.0), c, Z, g


def _kidney():
    df = load_kidney()
    Z = np.column_stack([df["age"], (df["sex"] == 2).astype(float)])
    return df["time"].values, 1 - df["status"].values, Z, df["id"].values


@pytest.mark.parametrize(
    "theta, tol",
    [(0.01, 1e-11), (0.5, 1e-11), (1.0, 1e-9), (2.0, 1e-7), (5.0, 3e-5)],
)
def test_quadrature_against_adaptive_reference(theta, tol):
    # The accuracy check that chose the number of nodes: the log-integral
    # against scipy's adaptive quadrature, from no events to 200 in a
    # group and cumulative hazards from 1e-4 to 1000.
    D = np.array([0.0, 1.0, 2.0, 5.0, 30.0, 200.0])
    H = np.array([1e-4, 0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 1e3])
    DD, HH = np.meshgrid(D, H, indexing="ij")
    got = lognormal_log_integral(DD, HH, theta)
    ref = np.vectorize(_quad_log_integral)(DD, HH, theta)
    assert N_NODES == 30
    assert np.max(np.abs(got - ref)) < tol


def test_plain_gauss_hermite_would_fail_where_adaptive_does_not():
    # Why the rule is adaptive: with 200 events the integrand sits far out
    # in the prior's tail, where 100 fixed nodes miss it by 57 nats.
    from scipy.special import logsumexp, roots_hermite

    z, wt = roots_hermite(100)
    D, H, theta = 200.0, 1e-4, 0.5
    w = np.sqrt(2 * theta) * z
    plain = logsumexp(np.log(wt) + D * w - H * np.exp(w)) - 0.5 * np.log(np.pi)
    ref = _quad_log_integral(D, H, theta)
    assert abs(plain - ref) > 10
    assert abs(lognormal_log_integral(D, H, theta) - ref) < 1e-10


def test_small_cumulative_hazard_keeps_relative_accuracy():
    # The marginal cumulative hazard -log I(0, s) for a small s is formed
    # without cancellation: against 1 - E[exp(-s u)] integrated directly.
    theta = 0.5
    for s in (1e-10, 1e-6, 1e-3):
        one_minus = quad(
            lambda w: -np.expm1(-s * np.exp(w))
            * np.exp(-w * w / (2 * theta))
            / np.sqrt(2 * np.pi * theta),
            -40,
            40,
            epsabs=0,
            epsrel=1e-13,
            limit=400,
        )[0]
        ref = -np.log1p(-one_minus)
        got = -lognormal_log_integral(0.0, s, theta)
        assert got == pytest.approx(ref, rel=1e-11)


def test_derivatives_match_finite_differences():
    # The fit differentiates the quadrature with autograd; with the nodes
    # placed from the values its derivatives are those of the rule.
    D = np.array([0.0, 1.0, 3.0, 10.0])
    H = np.array([0.05, 0.7, 2.0, 8.0])
    for theta in (0.05, 0.5, 2.0):

        def f(v):
            return np.sum(lognormal_log_integral(D, v[:4], v[4]))

        v = np.r_[H, theta]
        grad = autograd.grad(f)(v)
        step = 1e-6
        fd = [
            (f(v + step * e) - f(v - step * e)) / (2 * step) for e in np.eye(5)
        ]
        np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_no_frailty_limit_is_minus_H():
    D = np.array([0.0, 2.0, 5.0])
    H = np.array([0.3, 1.7, 4.0])
    np.testing.assert_array_equal(lognormal_log_integral(D, H, 1e-40), -H)
    np.testing.assert_allclose(
        lognormal_log_integral(D, H, 1e-9), -H, rtol=0, atol=1e-7
    )


def test_kidney_matches_lme4_adaptive_quadrature():
    # Reference: R lme4 glmer (nAGQ = 25) of the equivalent Poisson model,
    # profiled over the Weibull shape (scripts/reference/
    # reference_r_frailty.R).
    x, c, Z, g = _kidney()
    for name, dist in (("weibull", Weibull), ("exponential", Exponential)):
        ref = reference("r_frailty", "kidney_lognormal_" + name)["values"]
        m = Frailty(dist, family="lognormal").fit(x, Z=Z, c=c, groups=g)
        np.testing.assert_allclose(m.beta, ref["beta"], rtol=0, atol=2e-5)
        assert m.theta == pytest.approx(ref["theta"], rel=2e-5)
        np.testing.assert_allclose(m.dist_params, ref["dist_params"], 2e-5)
        assert -m.neg_ll() == pytest.approx(ref["loglik"], abs=1e-6)


def test_fit_is_a_maximum_of_an_independent_likelihood():
    # The fitted likelihood, recomputed group by group with scipy's
    # quadrature, is the fit's own value and is not improved by a step in
    # any parameter.
    x, c, Z, g = _simulate(3)
    m = Frailty(Weibull, family="lognormal").fit(x, Z=Z, c=c, groups=g)

    def loglik(alpha, shape, beta, theta):
        eta = np.exp(beta * Z[:, 0])
        H0 = Weibull.Hf(x, alpha, shape)
        h0 = Weibull.hf(x, alpha, shape)
        total = np.sum(np.log(h0[c == 0] * eta[c == 0]))
        for k in np.unique(g):
            i = g == k
            D = float(np.sum(c[i] == 0))
            total += _quad_log_integral(
                D, float(np.sum(eta[i] * H0[i])), theta
            )
        return total

    best = loglik(*m.params)
    assert best == pytest.approx(-m.neg_ll(), abs=1e-8)
    for j in range(4):
        for sign in (-1, 1):
            p = m.params.copy()
            p[j] *= 1 + sign * 1e-3
            assert loglik(*p) < best


def test_posterior_frailty_and_marginal_functions():
    x, c, Z, g = _simulate(4)
    m = Frailty(Weibull, family="lognormal").fit(x, Z=Z, c=c, groups=g)
    theta = m.theta
    # each group's posterior mean, E[u | data] = I(D + 1, H) / I(D, H)
    eta = np.exp(m.beta[0] * Z[:, 0])
    H0 = Weibull.Hf(x, *m.dist_params)
    for k in (0, 7, 31):
        i = g == k
        D, H = float(np.sum(c[i] == 0)), float(np.sum(eta[i] * H0[i]))
        post = np.exp(
            _quad_log_integral(D + 1, H, theta)
            - _quad_log_integral(D, H, theta)
        )
        assert m.frailties[str(k)] == pytest.approx(post, rel=1e-9)
    # the marginal survival is the frailty's Laplace transform
    t = np.array([2.0, 8.0, 20.0])
    s = np.exp(0.4 * m.beta[0]) * Weibull.Hf(t, *m.dist_params)
    ref = np.exp([_quad_log_integral(0.0, v, theta) for v in s])
    np.testing.assert_allclose(m.sf(t, [0.4]), ref, rtol=1e-10)
    # and the marginal hazard is eta h0 times the survivors' mean frailty
    h = m.hf(t, [0.4])
    mean_u = np.exp(
        [
            _quad_log_integral(1.0, v, theta)
            - _quad_log_integral(0.0, v, theta)
            for v in s
        ]
    )
    h_ref = np.exp(0.4 * m.beta[0]) * Weibull.hf(t, *m.dist_params) * mean_u
    np.testing.assert_allclose(h, h_ref, rtol=1e-10)
    # conditional on a group: its posterior mean frailty
    u0 = m.frailties["0"]
    np.testing.assert_allclose(
        m.sf(t, [0.4], group=0), np.exp(-u0 * s), rtol=1e-12
    )


def test_recovers_frailty_variance_for_each_family():
    # Each family recovers the variance of the data simulated from it
    # (about two standard errors), and the two fits agree on beta.
    for family in ("lognormal", "gamma"):
        x, c, Z, g = _simulate(11, family=family, G=150, per=5)
        m = Frailty(Weibull, family=family).fit(x, Z=Z, c=c, groups=g)
        se = dict(zip(m.parameter_names, m.standard_errors()))
        assert abs(m.theta - 0.5) < 2.5 * se["theta"], (family, m.theta)
        assert abs(m.beta[0] - 0.7) < 2.5 * se["beta_0"], (family, m.beta)


def test_aic_prefers_the_family_the_data_came_from():
    # Over 20 data sets from each family, AIC picks the generating family
    # more often than not (200 replications: 75% for log-normal data,
    # in the study reported on #343).
    right = {"lognormal": 0, "gamma": 0}
    for family in right:
        for seed in range(20):
            x, c, Z, g = _simulate(100 + seed, family=family, G=100)
            aic = {
                f: Frailty(Weibull, family=f).fit(x, Z=Z, c=c, groups=g).aic()
                for f in right
            }
            right[family] += min(aic, key=aic.get) == family
    assert right["lognormal"] >= 12 and right["gamma"] >= 12, right


def test_no_frailty_in_the_data_reduces_to_the_ph_fit():
    rng = np.random.default_rng(1)
    n = 300
    Z = rng.normal(size=(n, 1))
    groups = rng.integers(0, 30, n)
    x = 10.0 * rng.weibull(1.5, n) * np.exp(-0.8 * Z[:, 0] / 1.5)
    c = (x > 12).astype(int)
    x = np.minimum(x, 12.0)
    ph = sp.WeibullPH.fit(x=x, Z=Z, c=c)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr = Frailty(Weibull, family="lognormal").fit(
            x, Z=Z, c=c, groups=groups
        )
    assert fr.theta < 1e-3
    assert fr.neg_ll() == pytest.approx(ph.neg_ll(), abs=1e-4)
    assert fr.beta[0] == pytest.approx(ph.params[-1], rel=1e-3)
    assert np.all(np.isfinite(list(fr.frailties.values())))


def test_variance_measures_compare_the_families():
    m = sp.FrailtyModel()
    m.family, m.theta = "lognormal", 0.4
    assert m.frailty_variance == pytest.approx(np.expm1(0.4))
    m.family = "gamma"
    assert m.frailty_variance == 0.4
    assert m.kendall_tau == pytest.approx(0.4 / 2.4)


def test_kendall_tau_of_the_lognormal_frailty():
    # tau = 4 int s L(s) L''(s) ds - 1, with the Laplace transform L and
    # its second derivative integrated directly by scipy (Hougaard 2000).
    theta = 0.7

    def moment(k, s):
        return quad(
            lambda w: np.exp(k * w - s * np.exp(w) - w * w / (2 * theta))
            / np.sqrt(2 * np.pi * theta),
            -30,
            30,
            epsabs=0,
            epsrel=1e-12,
            limit=400,
            points=[np.clip(-np.log(s), -29.0, 29.0)],
        )[0]

    total = quad(
        lambda v: np.exp(2 * v) * moment(0, np.exp(v)) * moment(2, np.exp(v)),
        -40,
        40,
        limit=400,
    )[0]
    assert kendall_tau("lognormal", theta) == pytest.approx(
        4 * total - 1, abs=1e-8
    )
    assert kendall_tau("lognormal", 0.0) == 0.0


def test_summary_serialisation_and_data_frame():
    x, c, Z, g = _simulate(5)
    fitter = Frailty(Weibull, family="lognormal")
    m = fitter.fit(x, Z=Z, c=c, groups=g)
    assert m.family == "lognormal"
    assert list(m.summary().index)[-1] == ("frailty", "theta")
    assert "lognormal (log u ~ N(0, theta))" in repr(m)
    restored = sp.from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.family == "lognormal"
    t = np.array([3.0, 9.0])
    np.testing.assert_array_equal(m.sf(t, [0.2]), restored.sf(t, [0.2]))
    np.testing.assert_array_equal(m.hf(t, [0.2]), restored.hf(t, [0.2]))
    df = pd.DataFrame({"x": x, "c": c, "z": Z[:, 0], "g": g})
    from_df = fitter.fit_from_df(
        df, x_col="x", c_col="c", group_col="g", Z_cols="z"
    )
    np.testing.assert_allclose(from_df.params, m.params, rtol=1e-6)
    assert from_df.family == "lognormal"


def test_unknown_family_raises_a_value_error():
    with pytest.raises(ValueError, match="'family' must be one of"):
        Frailty(Weibull, family="weibull")


def test_617_a_huge_cumulative_hazard_does_not_overflow():
    # The likelihood-ratio searches evaluate the likelihood far out (a
    # group's H of 1e200): the check for a negligible theta squared it,
    # an OverflowError for a Python float.
    D = np.array([0.0, 2.0])
    with np.errstate(all="ignore"):
        value = lognormal_log_integral(D, np.full(2, 1e200), 0.5)
    assert np.all(np.isfinite(value)) and np.all(value < -1e5)
    # Still the no-frailty limit -H for a negligible theta.
    assert lognormal_log_integral(D, np.full(2, 3.0), 1e-20).tolist() == [
        -3.0,
        -3.0,
    ]
