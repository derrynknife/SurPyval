"""Prediction-validation metrics: Brier / IBS and time-dependent AUC (#212).

The metrics are validated against properties with known answers rather than
"it runs":

* with no censoring the Brier score is exactly the mean squared error between
  the survival indicator and the prediction;
* a well-specified model has a lower integrated Brier score than the marginal
  Kaplan-Meier reference, and a useless (constant) predictor is worse still;
* the time-dependent AUC is ~1 for a near-perfect risk ordering and ~0.5 for a
  random one.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.metrics import (
    auc_td,
    brier_score,
    integrated_brier_score,
    survival_probability,
)


def _cox_data(seed, n=600):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 1.2 * Z[:, 0] - 0.8 * Z[:, 1]
    t = rng.exponential(1.0 / np.exp(lin))
    cens = rng.exponential(np.median(t) * 3)
    x = np.minimum(t, cens)
    c = (cens < t).astype(int)
    return x, c, Z


def test_brier_reduces_to_mse_without_censoring():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.exponential(5, n)
    c = np.zeros(n, int)
    times = np.array([2.0, 4.0, 6.0])
    s = np.full((n, times.size), 0.5)
    _, bs = brier_score(x, c, s, times)
    manual = np.array(
        [np.mean(((x > t).astype(float) - 0.5) ** 2) for t in times]
    )
    np.testing.assert_allclose(bs, manual, atol=1e-9)


def test_well_specified_model_beats_marginal_km():
    xtr, ctr, Ztr = _cox_data(1)
    xte, cte, Zte = _cox_data(2)
    m = sp.CoxPH.fit(x=xtr, Z=Ztr, c=ctr)
    times = np.quantile(xte[cte == 0], [0.2, 0.4, 0.6, 0.8])

    s_cox = survival_probability(m, Zte, times)
    ibs_cox = integrated_brier_score(
        xte, cte, s_cox, times, x_train=xtr, c_train=ctr
    )

    km = sp.KaplanMeier.fit(xtr, ctr)
    s_km = np.tile(np.array([km.sf([t])[0] for t in times]), (len(xte), 1))
    ibs_km = integrated_brier_score(
        xte, cte, s_km, times, x_train=xtr, c_train=ctr
    )
    assert ibs_cox < ibs_km


def test_constant_predictor_is_worse_than_the_model():
    xtr, ctr, Ztr = _cox_data(1)
    xte, cte, Zte = _cox_data(2)
    m = sp.CoxPH.fit(x=xtr, Z=Ztr, c=ctr)
    times = np.quantile(xte[cte == 0], [0.3, 0.5, 0.7])
    s_good = survival_probability(m, Zte, times)
    s_flat = np.full_like(s_good, 0.5)
    ibs_good = integrated_brier_score(
        xte, cte, s_good, times, x_train=xtr, c_train=ctr
    )
    ibs_flat = integrated_brier_score(
        xte, cte, s_flat, times, x_train=xtr, c_train=ctr
    )
    assert ibs_good < ibs_flat


def test_ibs_single_time_is_the_brier_score():
    xte, cte, Zte = _cox_data(2, n=200)
    m = sp.CoxPH.fit(x=xte, Z=Zte, c=cte)
    t = np.array([np.median(xte)])
    s = survival_probability(m, Zte, t)
    _, bs = brier_score(xte, cte, s, t)
    ibs = integrated_brier_score(xte, cte, s, t)
    assert ibs == pytest.approx(float(bs[0]))


def test_auc_perfect_vs_random():
    # Near-deterministic ordering: event time decreases in z, so the risk
    # score z ranks the events almost perfectly (AUC ~ 1). A random score is
    # ~0.5.
    rng = np.random.default_rng(4)
    n = 400
    z = rng.normal(0, 1, n)
    t = 10.0 - 1.5 * z + rng.normal(0, 0.05, n)
    c = np.zeros(n, int)
    times = np.quantile(t, [0.3, 0.5, 0.7])
    _, auc_perfect = auc_td(t, c, z.reshape(-1, 1), times)
    _, auc_random = auc_td(t, c, rng.normal(0, 1, (n, 1)), times)
    assert np.nanmean(auc_perfect) > 0.97
    assert 0.4 < np.nanmean(auc_random) < 0.6


def test_auc_recovers_fitted_cox_discrimination():
    xte, cte, Zte = _cox_data(3)
    m = sp.CoxPH.fit(x=xte, Z=Zte, c=cte)
    times = np.quantile(xte[cte == 0], [0.3, 0.5, 0.7])
    risk = 1.0 - survival_probability(m, Zte, times)
    _, auc = auc_td(xte, cte, risk, times)
    # Strongly-predictive covariates: clearly better than chance everywhere.
    assert np.all(auc[~np.isnan(auc)] > 0.7)


def test_auc_single_risk_column_broadcasts():
    xte, cte, Zte = _cox_data(5, n=300)
    risk = (1.2 * Zte[:, 0] - 0.8 * Zte[:, 1]).reshape(-1, 1)
    times = np.quantile(xte[cte == 0], [0.4, 0.6])
    t_out, auc = auc_td(xte, cte, risk, times)
    assert auc.shape == t_out.shape == (2,)
    assert np.all(auc > 0.6)


def test_survival_probability_shape_and_values():
    xte, cte, Zte = _cox_data(6, n=150)
    m = sp.CoxPH.fit(x=xte, Z=Zte, c=cte)
    times = np.array([1.0, 2.0, 3.0])
    s = survival_probability(m, Zte, times)
    assert s.shape == (150, 3)
    # matches model.sf column by column
    for k, t in enumerate(times):
        expected = np.asarray(m.sf(np.full(150, t), Zte), dtype=float).ravel()
        np.testing.assert_allclose(s[:, k], expected)


def test_metrics_work_with_beta_ml_forest():
    # The metrics are model-agnostic: they must also accept the beta.ml forest,
    # whose sf returns a grid rather than a paired vector.
    import warnings

    from surpyval.beta.ml import RandomSurvivalForest

    rng = np.random.default_rng(11)
    n = 150
    Z = rng.normal(0, 1, (n, 2))
    lin = 1.0 * Z[:, 0] - 0.6 * Z[:, 1]
    t = rng.exponential(1 / np.exp(lin))
    cens = rng.exponential(np.median(t) * 3)
    x = np.minimum(t, cens)
    c = (cens < t).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = RandomSurvivalForest.fit(x=x, Z=Z, c=c, n_trees=4)
    times = np.quantile(x[c == 0], [0.4, 0.6])
    s = survival_probability(f, Z, times)
    assert s.shape == (n, 2)
    assert np.all((s >= 0) & (s <= 1))
    ibs = integrated_brier_score(x, c, s, times)
    assert 0.0 <= ibs <= 0.25
    _, auc = auc_td(x, c, 1.0 - s, times)
    assert np.nanmean(auc) > 0.6  # informative covariates


def test_brier_shape_validation():
    x = np.array([1.0, 2.0, 3.0])
    c = np.zeros(3, int)
    times = np.array([1.0, 2.0])
    bad = np.ones((3, 3))  # wrong number of time columns
    with pytest.raises(ValueError, match="n_samples, n_times"):
        brier_score(x, c, bad, times)
