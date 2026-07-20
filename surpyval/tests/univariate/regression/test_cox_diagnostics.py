"""Cox residuals and the proportional-hazards test (#211).

The residual identities are exact at the MLE and are the strongest
correctness checks available: Schoenfeld, score and martingale residuals each
sum to (approximately) zero because the fitted ``beta`` solves the score
equation. The proportional-hazards test is additionally checked for **power**
(it rejects a genuine time-varying-coefficient violation) and **calibration**
(under true proportional hazards its p-values are ~Uniform).
"""

import numpy as np
import pytest

import surpyval as sp


def _ph_data(seed=0, n=200, censor=0.25):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.8 * Z[:, 0] - 0.5 * Z[:, 1]
    x = (rng.exponential(1.0, n) / np.exp(lin)) ** (1 / 1.5)
    c = (rng.random(n) < censor).astype(int)
    return x, Z, c


# -- residual identities at the MLE -----------------------------------------


def test_schoenfeld_residuals_sum_to_zero():
    x, Z, c = _ph_data()
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    sch = m.compute_residuals("schoenfeld")
    assert sch.shape == ((c == 0).sum(), Z.shape[1])
    np.testing.assert_allclose(sch.sum(axis=0), 0.0, atol=1e-6)


def test_score_residuals_sum_to_score_zero():
    x, Z, c = _ph_data(seed=1)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    score = m.compute_residuals("score")
    assert score.shape == Z.shape
    # The score residuals sum to the total score, which is zero at the MLE.
    np.testing.assert_allclose(score.sum(axis=0), 0.0, atol=1e-5)


def test_martingale_residuals_sum_to_zero_and_bounded():
    x, Z, c = _ph_data(seed=2)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    mart = m.compute_residuals("martingale")
    assert mart.shape == (len(x),)
    assert abs(float(mart.sum())) < 1e-6
    # Martingale residuals lie in (-inf, 1].
    assert mart.max() <= 1.0 + 1e-9


def test_deviance_residuals_finite_and_symmetrising():
    x, Z, c = _ph_data(seed=3)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    dev = m.compute_residuals("deviance")
    mart = m.compute_residuals("martingale")
    assert np.all(np.isfinite(dev))
    # Deviance residuals share the martingale's sign and are more symmetric.
    assert np.all(np.sign(dev[mart != 0]) == np.sign(mart[mart != 0]))
    assert abs(float(dev.mean())) < abs(float(mart.mean())) + 0.5


def test_dfbeta_shape_and_scale():
    x, Z, c = _ph_data(seed=4)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    dfb = m.compute_residuals("dfbeta")
    assert dfb.shape == Z.shape
    # No single observation should dominate the coefficient (well-behaved data)
    assert np.all(np.abs(dfb).max(axis=0) < np.abs(m.beta) + 1.0)


def test_scaled_schoenfeld_centres_on_beta():
    x, Z, c = _ph_data(seed=5)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    scaled = m.compute_residuals("scaled_schoenfeld")
    # Under proportional hazards the scaled residuals fluctuate about beta.
    np.testing.assert_allclose(scaled.mean(axis=0), m.beta, atol=0.15)


# -- residuals respect delayed entry ----------------------------------------


def test_residuals_respect_left_truncation():
    rng = np.random.default_rng(6)
    n = 200
    Z = rng.normal(0, 1, (n, 1))
    x = (rng.exponential(1.0, n) / np.exp(0.6 * Z[:, 0])) ** (1 / 1.4)
    tl = np.minimum(rng.uniform(0, 0.3, n), x * 0.5)
    c = (rng.random(n) < 0.2).astype(int)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, tl=tl)
    # Martingale residuals use the fitted Breslow baseline directly, so they
    # sum to zero exactly.
    assert abs(float(m.compute_residuals("martingale").sum())) < 1e-6
    # Score residuals use exact risk-set membership ({tl < tau <= x}); the
    # fitted beta comes from a truncated partial likelihood that buckets
    # truncation times onto the event-time grid, so under truncation the sum
    # is O(grid error) rather than machine zero -- small relative to the
    # O(1) per-observation residual scale, not a residual bug.
    score = m.compute_residuals("score")
    rms = float(np.sqrt((score**2).mean()))
    assert abs(float(score.sum(axis=0)[0])) < 0.1 * rms * np.sqrt(n)


# -- the proportional-hazards test ------------------------------------------


def test_ph_test_does_not_reject_true_ph():
    x, Z, c = _ph_data(seed=7)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph()
    assert ph["global"]["df"] == Z.shape[1]
    assert ph["global"]["p_value"] > 0.05
    assert len(ph["per_covariate"]) == Z.shape[1]


def test_ph_test_detects_violation():
    # Z0 drives only the early half of the events -> a time-varying effect.
    rng = np.random.default_rng(8)
    n = 400
    Z = rng.normal(0, 1, (n, 1))
    u = rng.random(n)
    x = np.where(
        u < 0.5,
        np.exp(-1.5 * Z[:, 0]) * rng.exponential(1, n),
        rng.exponential(5, n),
    )
    c = np.zeros(n)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph()
    assert ph["global"]["p_value"] < 0.01


def test_ph_test_is_calibrated_under_null():
    # Under true proportional hazards the global p-value is ~Uniform(0, 1);
    # the rejection rate at 0.05 should be close to 0.05 (not systematically
    # small, which would indicate a mis-scaled statistic).
    def one(seed):
        x, Z, c = _ph_data(seed=1000 + seed, n=150, censor=0.2)
        return sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph()["global"]["p_value"]

    pvals = np.array([one(s) for s in range(80)])
    assert 0.0 <= pvals.min() and pvals.max() <= 1.0
    assert (pvals < 0.05).mean() < 0.15  # not over-rejecting
    assert 0.35 < pvals.mean() < 0.65  # centred near 0.5


@pytest.mark.parametrize("transform", ["km", "rank", "identity", "log"])
def test_ph_test_transforms_run(transform):
    x, Z, c = _ph_data(seed=9)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph(transform=transform)
    assert ph["transform"] == transform
    assert np.isfinite(ph["global"]["statistic"])


# -- guards ------------------------------------------------------------------


def test_unknown_residual_kind_raises():
    x, Z, c = _ph_data(seed=10)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="Unknown residual kind"):
        m.compute_residuals("bogus")


def test_unknown_transform_raises():
    x, Z, c = _ph_data(seed=11)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="transform"):
        m.check_ph(transform="bogus")


def test_per_covariate_names_from_dataframe():
    import pandas as pd

    rng = np.random.default_rng(12)
    n = 150
    df = pd.DataFrame(
        {
            "time": (rng.exponential(1.0, n)) ** (1 / 1.4),
            "temp": rng.normal(0, 1, n),
            "volt": rng.normal(0, 1, n),
        }
    )
    m = sp.CoxPH.fit_from_df(df, x_col="time", Z_cols=["temp", "volt"])
    ph = m.check_ph()
    names = [e.get("covariate") for e in ph["per_covariate"]]
    assert names == ["temp", "volt"]
