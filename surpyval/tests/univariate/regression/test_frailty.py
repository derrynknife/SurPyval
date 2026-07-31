"""Tests for the shared-frailty proportional-hazards model."""

import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

import surpyval as surv
from surpyval import ExponentialFrailty, Frailty, Weibull, WeibullFrailty
from surpyval.univariate.regression.frailty.frailty_model import (
    FrailtyModel,
)


def _sim(seed=7, G=80, per=6, alpha=12.0, shape=1.8, beta=0.9, theta=0.6):
    rng = np.random.default_rng(seed)
    groups, x, c, Z = [], [], [], []
    for g in range(G):
        u = rng.gamma(1.0 / theta, theta)
        for _ in range(per):
            z = rng.normal(0, 1)
            eta = np.exp(beta * z) * u
            t = alpha * (-np.log(rng.uniform()) / eta) ** (1.0 / shape)
            obs = min(t, 30.0)
            groups.append(g)
            x.append(obs)
            c.append(0 if t <= 30.0 else 1)
            Z.append(z)
    return (
        np.array(x),
        np.array(c),
        np.array(Z).reshape(-1, 1),
        np.array(groups),
    )


def test_marginal_likelihood_matches_numerical_integration():
    # Gold standard: the closed-form gamma-frailty group likelihood must equal
    # brute-force integration of the conditional likelihood over the frailty.
    rng = np.random.default_rng(0)
    fitter = WeibullFrailty
    dp = np.array([9.0, 1.8])
    theta = 0.7
    for _ in range(5):
        m = rng.integers(2, 6)
        x = np.abs(rng.weibull(2.0, m)) * 10 + 0.5
        c = rng.integers(0, 2, m)
        c[0] = 0  # ensure an event
        eta = np.exp(rng.normal(0, 0.5, m))

        H0 = Weibull.Hf(x, *dp)
        h0 = Weibull.hf(x, *dp)
        event = c == 0
        D = int(event.sum())
        H = float(np.sum(eta * H0))
        it = 1.0 / theta
        closed = (
            np.sum(np.log(h0[event]) + np.log(eta[event]))
            - it * np.log(theta)
            - gammaln(it)
            + gammaln(D + it)
            - (D + it) * np.log(H + it)
        )

        prod_event = np.prod(h0[event] * eta[event])
        logconst = it * np.log(theta) + gammaln(it)

        def integrand(u):
            cond = (u**D) * prod_event * np.exp(-u * H)
            fu = np.exp((it - 1) * np.log(u) - u / theta - logconst)
            return cond * fu

        val, _ = quad(integrand, 0, np.inf, limit=200)
        assert closed == pytest.approx(np.log(val), abs=1e-6)
    assert fitter is WeibullFrailty  # keep the fixture referenced


def test_recovers_known_parameters():
    x, c, Z, groups = _sim()
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    assert m.dist_params[0] == pytest.approx(12.0, rel=0.2)
    assert m.dist_params[1] == pytest.approx(1.8, rel=0.2)
    assert m.beta[0] == pytest.approx(0.9, abs=0.2)
    assert m.theta == pytest.approx(0.6, abs=0.2)
    assert m.n_groups == 80
    # theta CI excludes 0 for this clearly-heterogeneous data
    lo, hi = m.param_cb("theta")
    assert 0 < lo < m.theta < hi


def test_marginal_sf_is_gamma_laplace_transform():
    x, c, Z, groups = _sim(seed=2)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    t = np.array([4.0, 9.0, 15.0])
    z = np.array([0.3])
    eta = np.exp(z @ m.beta)
    H0 = Weibull.Hf(t, *m.dist_params)
    expected = (1.0 + m.theta * eta * H0) ** (-1.0 / m.theta)
    assert np.allclose(m.sf(t, z), expected)


def test_posterior_frailty_mean_is_about_one():
    x, c, Z, groups = _sim(seed=3)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    vals = np.array(list(m.frailties.values()))
    assert vals.mean() == pytest.approx(1.0, abs=0.1)


def test_conditional_orders_by_frailty():
    x, c, Z, groups = _sim(seed=4)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    t = np.array([8.0])
    z = np.array([0.0])
    # a higher frailty means a higher hazard, hence lower survival
    assert m.sf(t, z, frailty=2.0) < m.sf(t, z, frailty=0.5)


def test_no_covariate_frailty():
    rng = np.random.default_rng(5)
    G, per, theta = 60, 8, 0.5
    groups, x, c = [], [], []
    for g in range(G):
        u = rng.gamma(1.0 / theta, theta)
        for _ in range(per):
            t = 10.0 * (-np.log(rng.uniform()) / u) ** (1.0 / 2.0)
            groups.append(g)
            x.append(min(t, 25.0))
            c.append(0 if t <= 25.0 else 1)
    m = WeibullFrailty.fit(
        x=np.array(x), c=np.array(c), groups=np.array(groups)
    )
    assert m.beta.size == 0
    assert m.theta == pytest.approx(0.5, abs=0.2)
    assert m.sf(np.array([8.0])).shape == (1,)


def test_fit_from_df_formula_round_trips():
    rng = np.random.default_rng(6)
    n = 400
    import pandas as pd

    df = pd.DataFrame(
        {
            "t": rng.weibull(1.8, n) * 10 + 0.5,
            "c": rng.integers(0, 2, n),
            "load": rng.uniform(1, 5, n),
            "site": rng.choice(["A", "B", "C"], n),
            "unit": rng.integers(0, 50, n),
        }
    )
    m = WeibullFrailty.fit_from_df(
        df, x_col="t", c_col="c", group_col="unit", formula="load + site"
    )
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    assert type(restored).__name__ == "FrailtyModel"
    raw = pd.DataFrame({"load": [3.0, 2.0], "site": ["B", "A"]})
    t = np.array([5.0, 10.0])
    assert np.allclose(m.sf(t, raw), restored.sf(t, raw))
    # a known group's conditional prediction round-trips too
    g = m.group_labels[0]
    assert np.allclose(
        m.sf(t, raw.iloc[[0]], group=g),
        restored.sf(t, raw.iloc[[0]], group=g),
    )


def test_serialisation_round_trip_arrays():
    x, c, Z, groups = _sim(seed=8, G=40, per=5)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    t = np.array([5.0, 12.0])
    z = np.array([0.4])
    assert np.allclose(m.sf(t, z), restored.sf(t, z))
    assert restored.theta == pytest.approx(m.theta)
    assert restored.frailties == m.frailties


def test_guard_single_group():
    x, c, Z, groups = _sim(seed=9, G=1, per=20)
    with pytest.raises(ValueError, match="two groups"):
        WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)


def test_guard_unsupported_censoring():
    with pytest.raises(ValueError, match="right-censored"):
        WeibullFrailty.fit(
            x=np.array([1.0, 2.0, 3.0, 4.0]),
            c=np.array([-1, 0, 0, 1]),
            groups=np.array([0, 0, 1, 1]),
        )


def test_guard_unknown_family():
    with pytest.raises(NotImplementedError):
        Frailty(Weibull, family="lognormal")


def test_exponential_frailty_available():
    x, c, Z, groups = _sim(seed=10, G=50, per=5)
    m = ExponentialFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    assert m.dist.name == "Exponential"
    assert m.theta > 0


def test_theta_zero_gives_ph_limit_not_nan():
    # theta -> 0 is the no-frailty PH limit: Hf = eta * H0. Dividing by a
    # zero theta (frailty-free data, or a restored model) gave NaN (#262).
    m = FrailtyModel.__new__(FrailtyModel)
    m.dist = Weibull
    m.dist_params = np.array([10.0, 3.0])
    m.beta = np.array([])
    m.theta = 0.0
    m.family = "gamma"
    m._frailties = {}
    m.feature_names = None
    m.formula = None
    m._model_spec = None

    x = np.array([2.0, 5.0, 10.0])
    assert np.allclose(m.Hf(x), Weibull.Hf(x, 10.0, 3.0))
    assert np.all(np.isfinite(m.sf(x)))
