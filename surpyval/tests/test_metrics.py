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

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.metrics import (
    auc_td,
    brier_score,
    integrated_brier_score,
    survival_probability,
)
from surpyval.utils.ipcw import censoring_survival


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


# -- the AUC's pairs counted from the sorted controls (performance sweep) --


def _auc_by_pairs(x, c, risk, times, w):
    # Every case against every control, as auc_td counted them, O(n^2)
    auc = np.full(times.size, np.nan)
    for k, t in enumerate(times):
        r = risk[:, k]
        cases = (x <= t) & (c == 0)
        controls = x > t
        if not cases.any() or not controls.any():
            continue
        rk = r[controls]
        num = 0.0
        for ri, wi in zip(r[cases], w[cases]):
            num += wi * (
                np.count_nonzero(ri > rk) + 0.5 * np.count_nonzero(ri == rk)
            )
        auc[k] = num / (w[cases].sum() * controls.sum())
    return auc


def test_auc_counts_the_pairs_as_every_comparison_did(monkeypatch):
    # Ties in the risk and in the times, and missing risks: the same AUC,
    # to the bit, as comparing each case with every control, which took
    # 15 s at 1e5 rows and 20 horizons.
    from surpyval.metrics import validation

    rng = np.random.default_rng(3)
    n = 700
    x = np.round(rng.exponential(5, n)) + 0.5
    c = rng.choice([0, 1], n)
    risk = np.round(rng.normal(size=(n, 4)), 1)
    risk[rng.uniform(size=risk.shape) < 0.05] = np.nan
    times = np.quantile(x, [0.2, 0.5, 0.8, 0.95])
    _, _, w = validation._ipcw(x, c, None, None)
    expected = _auc_by_pairs(x, c, risk, times, w)

    calls = []
    count_nonzero = np.count_nonzero

    def counting(*args, **kwargs):
        calls.append(1)
        return count_nonzero(*args, **kwargs)

    monkeypatch.setattr(np, "count_nonzero", counting)
    got = auc_td(x, c, risk, times)[1]
    assert len(calls) < 10
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# The censoring convention at ties and the #290 review.
#
# * #365: the censoring survival behind the IPCW weights counted an event as
#   still at risk of being censored at its own time. The metrics now use the
#   events-first reverse Kaplan-Meier and weight an event by ``1 / G(x_i-)``
#   (Gerds and Schumacher 2006, R's ``pec``). On a *population* data set --
#   every combination of covariate, event time and censoring time once, so
#   that the Kaplan-Meier estimates are exact -- the IPCW Brier score and AUC
#   must then equal their true values exactly, ties and all.
# * Without ties between event and censoring times the metrics agree with
#   scikit-survival to rounding.
# * The integrated Brier score integrates in time order whatever the order of
#   the grid; the flags must be right-censoring flags of the right length;
#   a score that needs the censoring survival where the training estimate has
#   reached 0 is ``nan`` rather than silently biased.
# ---------------------------------------------------------------------------


# Event time given a binary covariate z, and an independent censoring time,
# each uniform on a few integers so that events and censorings tie.
T_GIVEN_Z = {0: [2.0, 3.0, 4.0], 1: [1.0, 2.0, 3.0]}


C_VALUES = [1.0, 2.0, 3.0, 4.0]


PRED = {0: 0.7, 1: 0.4}  # predicted survival, constant in t


def _population():
    z, x, c = [], [], []
    for zi, ts in T_GIVEN_Z.items():
        for t, cens in itertools.product(ts, C_VALUES):
            z.append(zi)
            x.append(min(t, cens))
            c.append(0 if t <= cens else 1)  # a tie is recorded as an event
    return np.array(z), np.array(x), np.array(c)


def _true_brier(t):
    pairs = [(z, T) for z, ts in T_GIVEN_Z.items() for T in ts]
    return np.mean([(float(T > t) - PRED[z]) ** 2 for z, T in pairs])


def _true_auc(t):
    pairs = [(z, T) for z, ts in T_GIVEN_Z.items() for T in ts]
    cases = [z for z, T in pairs if T <= t]
    controls = [z for z, T in pairs if T > t]
    return np.mean(
        [float(i > j) + 0.5 * float(i == j) for i in cases for j in controls]
    )


def test_censoring_survival_tie_conventions():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    cens = np.array([False, False, True, False])
    # Default (Fine-Gray / cmprsk::crr): the event at 2 is at risk of
    # censoring at 2, so the step there is 1 - 1/3. Unchanged.
    _, g = censoring_survival(x, cens)
    np.testing.assert_allclose(g, [1.0, 2 / 3, 2 / 3])
    # Events first (prodlim reverse = TRUE, scikit-survival): 1 - 1/2.
    _, g = censoring_survival(x, cens, ties="events_first")
    np.testing.assert_allclose(g, [1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="ties"):
        censoring_survival(x, cens, ties="censored_first")


@pytest.mark.parametrize("t", [1.0, 2.0, 2.5, 3.0])
def test_brier_exact_on_population_with_ties(t):
    # At t = 2 the truth is 0.2250. The old code (events at risk of
    # censoring, 1/G(x_i)) gave 0.2330; the events-first estimate with
    # 1/G(x_i) (scikit-survival's weights) gives 0.2881.
    z, x, c = _population()
    S = np.array([PRED[zi] for zi in z]).reshape(-1, 1)
    _, bs = brier_score(x, c, S, [t])
    assert bs[0] == pytest.approx(_true_brier(t), abs=1e-12)


@pytest.mark.parametrize("t", [1.0, 2.0, 2.5, 3.0])
def test_auc_exact_on_population_with_ties(t):
    # Binary risk score: tied scores count one half. At t = 2 the truth is
    # 2/3; the old code gave 0.6703, scikit-survival's weights 0.6603.
    z, x, c = _population()
    _, auc = auc_td(x, c, z.astype(float), [t])
    assert auc[0] == pytest.approx(_true_auc(t), abs=1e-12)


def test_tied_event_and_censoring_hand_example():
    # Weights at the tie at 3: G(3-) = 1 for the event, G(3.5) = 1/2 for
    # the survivor (at risk of censoring at 3: the censored row and x = 4).
    x = [1.0, 2.0, 3.0, 3.0, 4.0]
    c = [0, 0, 0, 1, 0]
    S = [[0.8], [0.6], [0.5], [0.5], [0.3]]
    _, bs = brier_score(x, c, S, [3.5])
    assert bs[0] == pytest.approx((0.64 + 0.36 + 0.25 + 2 * 0.49) / 5)


def _sksurv_data(seed, n):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.8 * z))
    cens = rng.exponential(1.5, n)
    return np.minimum(t, cens), (cens < t).astype(int), z


def test_agrees_with_sksurv_without_event_censoring_ties():
    sm = pytest.importorskip("sksurv.metrics")
    from sksurv.util import Surv

    xtr, ctr, _ = _sksurv_data(1, 200)
    xte, cte, zte = _sksurv_data(2, 150)
    # Round the test times so events tie with each other (not with
    # censorings of the training data, which stay continuous).
    ev = cte == 0
    xte[ev] = np.round(xte[ev], 1) + 1e-9
    times = np.quantile(xte[ev], [0.2, 0.4, 0.6, 0.8])
    S = np.exp(-np.outer(np.exp(0.8 * zte), times))
    ytr = Surv.from_arrays(ctr == 0, xtr)
    yte = Surv.from_arrays(cte == 0, xte)

    _, bs = brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    np.testing.assert_allclose(bs, sm.brier_score(ytr, yte, S, times)[1])
    ibs = integrated_brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    assert ibs == pytest.approx(sm.integrated_brier_score(ytr, yte, S, times))
    # Tied risk scores (rounded) count one half in both.
    risk = np.round(1.0 - S, 1)
    _, auc = auc_td(xte, cte, risk, times, x_train=xtr, c_train=ctr)
    ref, _ = sm.cumulative_dynamic_auc(ytr, yte, risk, times)
    np.testing.assert_allclose(auc, ref)


def test_ibs_does_not_depend_on_grid_order():
    # Old code: 0.1908 for the shuffled grid against 0.1949 sorted.
    rng = np.random.default_rng(3)
    x = rng.exponential(5, 50)
    c = (rng.uniform(size=50) < 0.3).astype(int)
    times = np.array([1.0, 2.0, 3.0, 4.0])
    S = np.exp(-np.outer(np.ones(50), times) / 5)
    p = [2, 0, 3, 1]
    assert integrated_brier_score(x, c, S[:, p], times[p]) == pytest.approx(
        integrated_brier_score(x, c, S, times)
    )


def test_flags_must_be_right_censoring_of_matching_length():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    S = np.full((4, 1), 0.5)
    # A left-censored row was scored as a known survivor past its time.
    with pytest.raises(ValueError, match="right censored"):
        brier_score(x, [0, -1, 0, 1], S, [1.5])
    with pytest.raises(ValueError, match="right censored"):
        auc_td(x, [0, 2, 0, 1], S, [1.5])
    # A length-one c broadcast silently to 'every row an event'.
    with pytest.raises(ValueError, match="same length"):
        brier_score(x, [0], S, [1.5])
    with pytest.raises(ValueError, match="both"):
        brier_score(x, [0, 0, 0, 1], S, [1.5], x_train=x)


def test_nan_where_training_censoring_survival_is_zero():
    # The training censoring estimate reaches 0 at 5 (its last row is
    # censored). A survivor at the horizon 6, or an event after 5, needs
    # 1/G = 1/0: the score is not identified. A zero weight used to drop
    # those rows and bias the score towards 0.
    xtr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ctr = np.array([0, 1, 0, 0, 1])
    xte = np.array([1.5, 2.5, 5.5, 7.0, 8.0])
    cte = np.array([0, 1, 0, 0, 1])
    S = np.full((5, 2), 0.5)
    times = [2.0, 6.0]
    _, bs = brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    assert np.isfinite(bs[0]) and np.isnan(bs[1])
    _, auc = auc_td(xte, cte, xte[::-1], times, x_train=xtr, c_train=ctr)
    assert np.isfinite(auc[0]) and np.isnan(auc[1])


# ---------------------------------------------------------------------------
# ``survival_probability`` with a DataFrame of covariates (#375
# item 8c): a formula model over string-valued factors is
# scored, a DataFrame being passed to ``model.sf``.
# ---------------------------------------------------------------------------


TIMES = np.array([2.0, 5.0, 8.0])


def _df(seed=0, n=150):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], size=n)
    eff = 0.5 * z + np.where(g == "b", 0.6, np.where(g == "c", -0.6, 0.0))
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    cens = rng.uniform(5, 30, n)
    return pd.DataFrame(
        {
            "x": np.minimum(t, cens),
            "c": (cens < t).astype(int),
            "z": z,
            "g": g,
        }
    )


def _by_time(model, Z, times):
    # The matrix built directly from the model, one time at a time
    n = len(Z)
    return np.column_stack([model.sf(np.full(n, t), Z) for t in times])


@pytest.mark.parametrize("fitter", ["WeibullPH", "CoxPH", "AdditiveHazards"])
def test_formula_fit_with_string_levels_is_scored(fitter):
    train, test = _df(0), _df(1, n=40)
    model = getattr(sp, fitter).fit_from_df(
        train, x_col="x", c_col="c", formula="z + g"
    )
    Z = test[["z", "g"]]
    # Old code: ValueError, could not convert string to float
    S = survival_probability(model, Z, TIMES)
    assert S.shape == (40, TIMES.size)
    np.testing.assert_allclose(S, _by_time(model, Z, TIMES))
    ibs = integrated_brier_score(test["x"], test["c"], S, TIMES)
    assert np.isfinite(ibs)


def test_formula_fit_missing_level_scores_nan_row():
    train = _df(0)
    model = sp.WeibullPH.fit_from_df(
        train, x_col="x", c_col="c", formula="z + g"
    )
    Z = pd.DataFrame(
        {"z": [0.1, 0.2, np.nan], "g": pd.Series([None, "b", "c"])}
    )
    S = survival_probability(model, Z, TIMES)
    assert np.isnan(S[[0, 2]]).all()
    np.testing.assert_allclose(S[1], _by_time(model, Z.iloc[[1]], TIMES)[0])


def test_named_column_fit_reads_dataframe_by_name():
    train, test = _df(0), _df(1, n=30)
    train["w"] = 1.0
    test["w"] = 1.0
    model = sp.WeibullPH.fit_from_df(
        train, x_col="x", c_col="c", Z_cols=["z", "w"]
    )
    # Columns in another order, and an extra one: read by name
    S = survival_probability(model, test[["g", "w", "z"]], TIMES)
    np.testing.assert_allclose(
        S, survival_probability(model, test[["z", "w"]].values, TIMES)
    )


def test_arrays_unchanged():
    df = _df(0)
    Z = df[["z"]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullPH.fit(df["x"].values, Z, df["c"].values)
    S = survival_probability(model, Z[:10], TIMES)
    np.testing.assert_allclose(S, _by_time(model, Z[:10], TIMES))
    # A 1-D array is one covariate, and a list works too
    np.testing.assert_allclose(
        survival_probability(model, Z[:10, 0], TIMES), S
    )
    np.testing.assert_allclose(
        survival_probability(model, Z[:10].tolist(), TIMES), S
    )
