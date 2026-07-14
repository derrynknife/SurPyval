"""Buckley-James semi-parametric AFT regression.

The model is ``log T = gamma'Z + eps`` with an unspecified error distribution,
fit under right-censoring by the Buckley-James imputation iteration.
Coefficients are reported in surpyval's accelerated-failure convention (the
negative of the textbook ``gamma``), so a positive coefficient shortens life --
matching ``WeibullAFT``/``LogNormalAFT``.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import BuckleyJames, LogNormalAFT


def _aft_data(N, seed, gamma=(0.8, -0.5), mu=1.5, sigma=0.6, cens_mu=2.2):
    """``log T = mu + gamma'Z + eps`` with normal errors and independent
    right-censoring. Returns the surpyval-convention coefficients ``-gamma``.
    """
    rng = np.random.default_rng(seed)
    gamma = np.asarray(gamma, dtype=float)
    Z = rng.uniform(-1, 1, size=(N, gamma.size))
    Y = mu + Z @ gamma + rng.normal(0, sigma, N)
    T = np.exp(Y)
    C = np.exp(rng.normal(cens_mu, 0.8, N))
    c = (T > C).astype(int)
    x = np.minimum(T, C)
    return x, Z, c, -gamma


def test_no_censoring_equals_ols():
    # With no censoring the imputation is a no-op, so BJ is exactly the
    # least-squares slope of log(x) on Z (negated for the AFT sign).
    rng = np.random.default_rng(0)
    N = 500
    Z = rng.uniform(-1, 1, size=(N, 2))
    Y = 2.0 + Z @ [0.7, -0.4] + rng.normal(0, 0.5, N)
    x = np.exp(Y)
    m = BuckleyJames.fit(x, Z)
    X = np.column_stack([np.ones(N), Z])
    ols_slope = np.linalg.lstsq(X, Y, rcond=None)[0][1:]
    assert np.allclose(m.beta, -ols_slope, atol=1e-6)
    assert m.converged and m.n_iter == 1


def test_recovers_coefficients_under_censoring():
    x, Z, c, beta = _aft_data(6000, 1)
    m = BuckleyJames.fit(x, Z, c=c)
    assert m.converged
    assert np.allclose(m.beta, beta, atol=0.06)


def test_agrees_with_lognormal_aft():
    # For normal errors the semi-parametric BJ should track the parametric
    # LogNormal AFT in both sign and magnitude.
    x, Z, c, beta = _aft_data(6000, 2)
    bj = BuckleyJames.fit(x, Z, c=c)
    la = LogNormalAFT.fit(x=x, Z=Z, c=c)
    la_beta = la.params[la.k_dist :]
    assert np.allclose(bj.beta, la_beta, atol=0.06)


def test_positive_coefficient_shortens_life():
    # Direct convention check: a larger positive coefficient must lower
    # survival at fixed time (accelerated failure).
    x, Z, c, beta = _aft_data(4000, 3)
    m = BuckleyJames.fit(x, Z, c=c)
    # beta[0] is negative here (gamma[0] positive), so raising covariate 0
    # should *raise* survival. Compare two covariate vectors.
    t = np.array([5.0])
    hi_cov = m.sf(t, [1.0, 0.0])
    lo_cov = m.sf(t, [-1.0, 0.0])
    # sign of the effect follows sign of beta[0]
    if m.beta[0] < 0:
        assert hi_cov > lo_cov
    else:
        assert hi_cov < lo_cov


def test_sf_is_monotone_and_bounded():
    x, Z, c, beta = _aft_data(3000, 4)
    m = BuckleyJames.fit(x, Z, c=c)
    t = np.linspace(0.5, 60.0, 80)
    sf = m.sf(t, [0.3, -0.2])
    assert np.all(sf >= 0) and np.all(sf <= 1)
    assert np.all(np.diff(sf) <= 1e-12)
    assert np.allclose(m.ff(t, [0.3, -0.2]), 1 - sf)


def test_counts_equivalent_to_repeated_rows():
    x, Z, c, beta = _aft_data(1200, 5)
    m_rep = BuckleyJames.fit(
        np.repeat(x, 2), np.repeat(Z, 2, axis=0), c=np.repeat(c, 2)
    )
    m_cnt = BuckleyJames.fit(x, Z, c=c, n=np.full(x.size, 2))
    assert np.allclose(m_rep.beta, m_cnt.beta, atol=1e-4)


def test_single_covariate():
    x, Z, c, beta = _aft_data(2000, 6, gamma=(0.7,))
    m = BuckleyJames.fit(x, Z[:, 0], c=c)
    assert m.beta.shape == (1,)
    assert np.allclose(m.beta, beta, atol=0.08)


def test_bootstrap_ci_brackets_estimate():
    x, Z, c, beta = _aft_data(2000, 7)
    m = BuckleyJames.fit(x, Z, c=c)
    ci = m.bootstrap_ci(n_boot=120, seed=0)
    assert ci.shape == (2, 2)
    assert np.all(ci[:, 0] <= m.beta) and np.all(m.beta <= ci[:, 1])
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_rejects_non_right_censoring():
    x, Z, c, beta = _aft_data(200, 8)
    bad = c.copy()
    bad[:3] = -1  # left-censored
    with pytest.raises(ValueError, match="only observed"):
        BuckleyJames.fit(x, Z, c=bad)


def test_rejects_non_positive_times():
    rng = np.random.default_rng(9)
    Z = rng.uniform(-1, 1, size=(50, 2))
    x = rng.uniform(-1, 1, size=50)  # some non-positive
    with pytest.raises(ValueError, match="must be positive"):
        BuckleyJames.fit(x, Z)


# --- fit_from_df ----------------------------------------------------------


def test_fit_from_df_matches_array_fit():
    x, Z, c, beta = _aft_data(3000, 10)
    df = pd.DataFrame({"t": x, "c": c, "age": Z[:, 0], "dose": Z[:, 1]})
    m_df = BuckleyJames.fit_from_df(
        df, x_col="t", Z_cols=["age", "dose"], c_col="c"
    )
    m_arr = BuckleyJames.fit(x, Z, c=c)
    assert m_df.feature_names == ["age", "dose"]
    assert np.allclose(m_df.beta, m_arr.beta)
    pred = m_df.sf([2.0, 5.0], df[["age", "dose"]].iloc[[0]])
    assert pred.shape == (2,)


def test_fit_from_df_formula():
    x, Z, c, beta = _aft_data(3000, 11)
    df = pd.DataFrame({"t": x, "c": c, "age": Z[:, 0], "dose": Z[:, 1]})
    m = BuckleyJames.fit_from_df(
        df, x_col="t", formula="age + dose", c_col="c"
    )
    assert "age" in m.feature_names and "dose" in m.feature_names
    assert np.allclose(m.beta, beta, atol=0.08)
