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

import surpyval as sp
from surpyval import BuckleyJames, LogNormalAFT
from surpyval.tests._helpers import counted_regression_data


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
    ci = m.bootstrap_ci(n_boot=120, random_state=0)
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


def test_648_refuses_data_with_no_event():
    # It reported converged=True with the slope of the censoring times.
    x, Z, c, beta = _aft_data(50, 8)
    with pytest.raises(ValueError, match=r"needs at least one event \(c=0\)"):
        BuckleyJames.fit(x, Z, c=np.ones(50))


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


# ---------------------------------------------------------------------------
# Counts are frequency weights in the bootstrap.
# ---------------------------------------------------------------------------


def test_buckley_james_bootstrap_equals_expanded_data():
    x, Z, n, c = counted_regression_data()
    a = BuckleyJames.fit(x, Z, c=c, n=n)
    b = BuckleyJames.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    )
    np.testing.assert_allclose(
        a.bootstrap_ci(random_state=3, n_boot=50),
        b.bootstrap_ci(random_state=3, n_boot=50),
    )


# ---------------------------------------------------------------------------
# #426: one covariate row per time.
# ---------------------------------------------------------------------------


def _buckley_james():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(100, 2))
    t = np.exp(2.0 - 0.5 * Z[:, 0] + 0.2 * Z[:, 1] + rng.normal(0, 0.5, 100))
    c = (t > 12).astype(int)
    return sp.BuckleyJames.fit(np.minimum(t, 12), Z, c=c)


def test_buckley_james_pairs_one_row_per_time():
    # It used to raise numpy's bare matmul error.
    model = _buckley_james()
    x = np.array([5.0, 10.0, 7.0])
    Z = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5]])
    alone = [model.sf(x[k : k + 1], Z[k])[0] for k in range(3)]
    np.testing.assert_allclose(model.sf(x, Z), alone, rtol=1e-15)
    np.testing.assert_allclose(model.ff(x, Z), 1 - np.array(alone))
    # One row is still used at every time.
    np.testing.assert_allclose(
        model.sf(x, Z[:1]), model.sf(x, Z[0]), rtol=1e-15
    )


def test_buckley_james_refuses_a_mismatched_z():
    model = _buckley_james()
    with pytest.raises(ValueError, match="3 covariate rows but there are 2"):
        model.sf([5.0, 10.0], np.zeros((3, 2)))
    # The width named as every regression names it (#657)
    with pytest.raises(
        ValueError, match=r"has 2 covariates \(coef_0, coef_1\)"
    ):
        model.sf([5.0, 10.0], [0.0, 1.0, 2.0])


def test_746_Hf_before_the_first_time_is_plus_zero():
    # sf is 1 before the first residual time; -log(1) was -0.0 (#746).
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Z = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1.0])
    model = BuckleyJames.fit(x, Z)
    H = model.Hf([0.0, 0.5], np.array([0.0]))
    assert np.all(H == 0) and not np.any(np.signbit(H))
