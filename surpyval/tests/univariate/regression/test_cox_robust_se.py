"""Cluster-robust (sandwich) standard errors for Cox models (#215).

The Lin-Wei robust variance is validated two ways: on independent data it
agrees with the model-based standard errors, and on data with exactly
replicated clusters it inflates by the theoretically exact factor
``sqrt(cluster_size)`` relative to naively treating the rows as independent.
"""

import numpy as np
import pytest

import surpyval as sp


def _cox(seed=0, n=300):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.7 * Z[:, 0] - 0.4 * Z[:, 1]
    x = (rng.exponential(1.0, n) / np.exp(lin)) ** (1 / 1.4)
    c = (rng.random(n) < 0.2).astype(int)
    return sp.CoxPH.fit(x=x, Z=Z, c=c), Z, x, c


def test_robust_se_matches_model_based_on_independent_data():
    m, Z, x, c = _cox()
    model_se = np.sqrt(np.diag(np.linalg.inv(m.jac(m.beta)[1])))
    robust_se = m.robust_summary()["se"]
    # On independent, well-specified data the two agree closely.
    np.testing.assert_allclose(robust_se, model_se, rtol=0.25)


def test_robust_covariance_is_symmetric_psd():
    m, *_ = _cox(seed=1)
    cov = m.robust_covariance()
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)


def test_cluster_robust_inflates_by_sqrt_cluster_size():
    # Replicate every subject `reps` times into one cluster. Ignoring the
    # clustering underestimates the SE by exactly sqrt(reps); the
    # cluster-robust SE recovers it.
    m, Z, x, c = _cox(seed=2, n=200)
    reps = 4
    Zc = np.repeat(Z, reps, axis=0)
    xc = np.repeat(x, reps)
    cc = np.repeat(c, reps)
    clid = np.repeat(np.arange(len(x)), reps)
    mc = sp.CoxPH.fit(x=xc, Z=Zc, c=cc)

    se_ignore = mc.robust_summary()["se"]
    se_cluster = mc.robust_summary(cluster=clid)["se"]
    np.testing.assert_allclose(
        se_cluster / se_ignore, np.sqrt(reps), rtol=0.02
    )


def test_robust_summary_fields_and_pvalues():
    m, *_ = _cox(seed=3)
    out = m.robust_summary()
    assert out["se"].shape == m.beta.shape
    assert out["covariance"].shape == (m.beta.size, m.beta.size)
    assert np.all((out["p_value"] >= 0) & (out["p_value"] <= 1))
    # Both coefficients are strong here, so both should be significant.
    assert np.all(out["p_value"] < 0.05)


def test_cluster_length_mismatch_raises():
    m, Z, x, c = _cox(seed=4)
    with pytest.raises(ValueError, match="one label per observation"):
        m.robust_covariance(cluster=np.arange(len(x) - 1))


def test_robust_summary_carries_covariate_names():
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame(
        {
            "time": rng.exponential(5.0, n),
            "temp": rng.normal(0, 1, n),
            "volt": rng.normal(0, 1, n),
        }
    )
    m = sp.CoxPH.fit_from_df(df, x_col="time", Z_cols=["temp", "volt"])
    assert m.robust_summary()["covariate"] == ["temp", "volt"]
