"""Stratified Cox proportional-hazards models (#214).

Stratification fits a separate baseline hazard per stratum while sharing the
coefficient vector, summing the partial likelihood within strata. The tests
pin the three properties that make it correct:

* it recovers the true coefficient when a strong per-stratum baseline is
  *confounded* with the covariate -- the case where an ordinary Cox fit is
  badly biased;
* a single stratum reduces exactly to an ordinary Cox fit; and
* the stratified partial log-likelihood factorises into the sum of the
  per-stratum ordinary Cox log-likelihoods.
"""

import numpy as np
import pytest

import surpyval as sp


def _confounded(seed, n=600, true_beta=0.8):
    """Baseline scale differs by stratum and the covariate is correlated
    with the stratum, so an unstratified fit is confounded."""
    rng = np.random.default_rng(seed)
    strat = rng.integers(0, 3, n)
    Z = rng.normal(strat.astype(float), 1.0).reshape(-1, 1)
    base_scale = np.array([1.0, 6.0, 30.0])[strat]
    lam = np.exp(true_beta * Z[:, 0]) / base_scale
    t = rng.exponential(1.0 / lam)
    cens = rng.exponential(np.median(1.0 / lam) * 2)
    x = np.minimum(t, cens)
    c = (cens < t).astype(int)
    return x, Z, c, strat


def test_stratified_recovers_beta_when_pooled_is_biased():
    true_beta = 0.8
    strat_est, pool_est = [], []
    for s in range(25):
        x, Z, c, strat = _confounded(s, true_beta=true_beta)
        strat_est.append(sp.CoxPH.fit(x=x, Z=Z, c=c, strata=strat).beta[0])
        pool_est.append(sp.CoxPH.fit(x=x, Z=Z, c=c).beta[0])
    strat_est = np.array(strat_est)
    pool_est = np.array(pool_est)
    # Stratified is close to the truth; pooled is badly biased towards zero.
    assert abs(strat_est.mean() - true_beta) < 0.1
    assert pool_est.mean() < 0.4  # confounding attenuates the pooled estimate
    # And the stratified estimator is meaningfully closer to the truth.
    assert abs(strat_est.mean() - true_beta) < abs(pool_est.mean() - true_beta)


def test_single_stratum_equals_ordinary_cox():
    rng = np.random.default_rng(1)
    n = 300
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.5 * Z[:, 0] - 0.3 * Z[:, 1]
    x = rng.exponential(1 / np.exp(lin))
    c = (rng.random(n) < 0.2).astype(int)
    m_plain = sp.CoxPH.fit(x=x, Z=Z, c=c)
    m_strat = sp.CoxPH.fit(x=x, Z=Z, c=c, strata=np.ones(n))
    np.testing.assert_allclose(m_strat.beta, m_plain.beta, atol=1e-4)
    assert m_strat.is_stratified


def test_partial_likelihood_factorises_over_strata():
    rng = np.random.default_rng(2)
    n = 300
    Z = rng.normal(0, 1, (n, 2))
    x = rng.exponential(1 / np.exp(0.4 * Z[:, 0]))
    c = (rng.random(n) < 0.2).astype(int)
    strata = (Z[:, 0] > 0).astype(int)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, strata=strata)
    total = 0.0
    for s in (0, 1):
        mask = strata == s
        sub = sp.CoxPH.fit(x=x[mask], Z=Z[mask], c=c[mask])
        total += float(sub.neg_ll(m.beta))
    assert float(m.neg_ll(m.beta)) == pytest.approx(total, abs=1e-6)


def test_per_stratum_baselines_differ_and_predict():
    rng = np.random.default_rng(3)
    n = 400
    strat = rng.integers(0, 2, n)
    Z = rng.normal(0, 1, (n, 1))
    base = np.array([2.0, 10.0])[strat]
    lam = np.exp(0.6 * Z[:, 0]) / base
    t = rng.exponential(1 / lam)
    cens = rng.exponential(8)
    x = np.minimum(t, cens)
    c = (cens < t).astype(int)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, strata=strat)

    assert set(m.strata_labels) == {0, 1}
    # Stratum 1 has a much longer baseline scale, so at a fixed covariate its
    # survival is higher.
    s0 = float(m.sf(2.0, np.array([[0.0]]), stratum=0).ravel()[0])
    s1 = float(m.sf(2.0, np.array([[0.0]]), stratum=1).ravel()[0])
    assert 0.0 < s0 < s1 < 1.0


def test_prediction_requires_stratum():
    x, Z, c, strat = _confounded(4)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, strata=strat)
    with pytest.raises(ValueError, match="stratified Cox model"):
        m.sf(1.0, np.array([[0.0]]))
    with pytest.raises(ValueError, match="unknown stratum"):
        m.Hf(1.0, np.array([[0.0]]), stratum=99)


def test_stratum_argument_rejected_on_unstratified_model():
    x, Z, c, _ = _confounded(5)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="not stratified"):
        m.sf(1.0, np.array([[0.0]]), stratum=0)


def test_diagnostics_not_available_for_stratified():
    x, Z, c, strat = _confounded(6)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, strata=strat)
    for method in ("compute_residuals", "check_ph", "robust_summary"):
        with pytest.raises(NotImplementedError):
            getattr(m, method)()
    with pytest.raises(NotImplementedError):
        m.to_dict()


def test_fit_from_df_strata_col():
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 300
    strat = rng.integers(0, 2, n)
    z = rng.normal(strat.astype(float), 1.0)
    base = np.array([2.0, 12.0])[strat]
    lam = np.exp(0.7 * z) / base
    t = rng.exponential(1 / lam)
    cens = rng.exponential(10)
    df = pd.DataFrame(
        {
            "time": np.minimum(t, cens),
            "event": (cens < t).astype(int),
            "z": z,
            "site": strat,
        }
    )
    m = sp.CoxPH.fit_from_df(
        df, x_col="time", Z_cols=["z"], c_col="event", strata_col="site"
    )
    assert m.is_stratified
    assert set(m.strata_labels) == {0, 1}
    assert abs(m.beta[0] - 0.7) < 0.2
