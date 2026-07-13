"""Lin & Ying (1994) semi-parametric additive hazards model.

Validation strategy (no R in CI): the closed-form estimator is checked
against a brute-force numerical integration of the estimating equation, its
point estimates against simulation from a known additive model, and its
sandwich standard errors against the empirical spread of repeated fits.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import AdditiveHazards


def _simulate(N, seed, lambda0=0.5, beta=(0.30, -0.15), tau=6.0):
    # h(t | Z) = lambda0 + beta'Z is constant in t, so survival is
    # exponential with rate lambda0 + beta'Z; recovering beta validates the
    # estimator against a known truth. Covariates are kept small so the
    # hazard stays positive.
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta)
    Z = rng.uniform(-1, 1, size=(N, beta.size))
    rate = lambda0 + Z @ beta
    Z, rate = Z[rate > 0], rate[rate > 0]
    x = rng.exponential(1.0 / rate)
    c = (x > tau).astype(int)
    x = np.minimum(x, tau)
    return x, c, Z


def test_closed_form_matches_brute_force_integration():
    # A (a time integral over the risk sets) and b (a sum over events) must
    # match a direct numerical evaluation of the Lin-Ying estimating
    # equation.
    x = np.array([2.0, 5.0, 1.0, 8.0, 4.0])
    c = np.array([0, 0, 1, 0, 0])
    Z = np.array([[0.5], [1.0], [0.2], [-0.3], [0.8]])
    model = AdditiveHazards.fit(x, Z, c=c)

    grid = np.linspace(0, x.max(), 200_000)
    dt = grid[1] - grid[0]
    A_bf = 0.0
    for t in grid:
        at_risk = x >= t
        if at_risk.sum() == 0:
            continue
        Zt = Z[at_risk]
        centered = Zt - Zt.mean(0)
        A_bf += (centered[:, :, None] * centered[:, None, :]).sum(0) * dt
    b_bf = np.zeros(1)
    for i in np.where(c == 0)[0]:
        b_bf += Z[i] - Z[x >= x[i]].mean(0)

    assert np.allclose(model._A, A_bf, atol=1e-3)
    assert np.allclose(model._b, b_bf, atol=1e-4)
    assert np.allclose(model.beta, np.linalg.solve(A_bf, b_bf), atol=1e-3)


def test_estimating_equation_is_solved():
    x, c, Z = _simulate(500, 0)
    model = AdditiveHazards.fit(x, Z, c=c)
    # beta = A^-1 b, so A beta must reproduce b exactly.
    assert np.allclose(model._A @ model.beta, model._b)


def test_recovers_known_beta():
    x, c, Z = _simulate(20000, 1)
    model = AdditiveHazards.fit(x, Z, c=c)
    assert np.allclose(model.beta, [0.30, -0.15], atol=0.02)
    assert np.all(np.abs(model.beta - [0.30, -0.15]) < 3 * model.se)


def test_sandwich_se_matches_empirical_spread():
    # The Lin-Ying sandwich SE should track the actual sampling spread of
    # the estimate across repeated samples.
    ests, ses = [], []
    for s in range(80):
        x, c, Z = _simulate(1500, 100 + s)
        model = AdditiveHazards.fit(x, Z, c=c)
        ests.append(model.beta)
        ses.append(model.se)
    empirical_sd = np.std(ests, axis=0)
    mean_se = np.mean(ses, axis=0)
    assert np.allclose(empirical_sd, mean_se, rtol=0.2)


def test_p_values_shape_and_significance():
    x, c, Z = _simulate(20000, 2)
    model = AdditiveHazards.fit(x, Z, c=c)
    assert model.p_values.shape == (2,)
    # Both effects are real and the sample is large, so both are significant.
    assert np.all(model.p_values < 0.05)
    assert model.covariance().shape == (2, 2)
    assert np.allclose(model.standard_errors(), np.sqrt(np.diag(model.cov)))


def test_counts_equivalent_to_repeated_rows():
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    c = np.array([0, 0, 1, 0, 0])
    Z = np.array([[0.4], [0.7], [0.7], [-0.2], [0.9]])
    m_rep = AdditiveHazards.fit(
        np.repeat(x, 2), np.repeat(Z, 2, axis=0), c=np.repeat(c, 2)
    )
    m_cnt = AdditiveHazards.fit(x, Z, c=c, n=np.full(5, 2))
    assert np.allclose(m_rep.beta, m_cnt.beta)
    assert np.allclose(m_rep.H0, m_cnt.H0)


def test_baseline_survival_matches_exponential():
    # With covariate effects removed (Z = 0) the fitted survival should be
    # the constant-baseline exponential the data were generated from.
    x, c, Z = _simulate(20000, 3)
    model = AdditiveHazards.fit(x, Z, c=c)
    t = np.array([0.5, 1.0, 2.0, 3.0])
    sf0 = model.sf(t, np.array([[0.0, 0.0]]))
    assert np.allclose(sf0, np.exp(-0.5 * t), atol=0.03)


def test_prediction_methods_are_consistent():
    x, c, Z = _simulate(2000, 4)
    model = AdditiveHazards.fit(x, Z, c=c)
    t = np.array([1.0, 2.0, 3.0])
    Z0 = np.array([[0.2, -0.1]])
    assert np.allclose(model.ff(t, Z0), 1 - model.sf(t, Z0))
    assert np.allclose(model.df(t, Z0), model.hf(t, Z0) * model.sf(t, Z0))
    # H(t | Z) = H0(t) + t * beta'Z is linear in t beyond the baseline step.
    assert model.Hf(t, Z0).shape == (3,)


def test_fit_from_df_retains_feature_names():
    rng = np.random.default_rng(5)
    Z = rng.uniform(-1, 1, size=(500, 2))
    rate = 0.5 + Z @ np.array([0.3, -0.15])
    x = rng.exponential(1 / rate)
    df = pd.DataFrame({"time": x, "event": 1, "age": Z[:, 0], "dose": Z[:, 1]})
    # surpyval censoring convention: 0 = observed event, 1 = censored.
    df["c"] = 0
    model = AdditiveHazards.fit_from_df(
        df, x_col="time", Z_cols=["age", "dose"], c_col="c"
    )
    assert model.feature_names == ["age", "dose"]
    # Prediction accepts a DataFrame and selects the right columns.
    pred = model.sf([1.0, 2.0], df[["age", "dose"]].iloc[[0]])
    assert pred.shape == (2,)


def test_rejects_interval_and_left_censoring():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="right-censored"):
        AdditiveHazards.fit(x, np.array([[0.5], [0.7]]), c=np.array([2, 2]))
