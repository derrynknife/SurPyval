"""The calibration of predicted survival, the check ``diagnose()`` shares
between the Cox and the parametric PH and AFT families (#881).

The curve, its ICI, E50 and E90 and its band are compared with R's (the
model, its predictions and the calibration model all R's: coxph or survreg,
and coxph on rcs(log(-log(1 - p)), knots) with survfit, stored by
``scripts/reference/reference_r_calibration.R``), checked on a case
computed by hand, and shown to separate calibrated from miscalibrated
models.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis, calibration
from surpyval.diagnostics.calibration import (
    ICI_LIMIT,
    OUTSIDE_LIMIT,
    calibration_advice,
    calibration_curve,
    lower_quantile,
    rcs_basis,
    weighted_quantile,
)
from surpyval.tests.reference._data import fixture, values

FITTED = dict(rtol=1e-6, atol=1e-8)


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _rows(d):
    return d.table[d.table["check"] == "calibration"]


def _curves(d):
    """The ``(name, t0, curve)`` of each row of the report ``d``."""
    return d._context.cached("calibration_curves", lambda: None)


# ---------------------------------------------------------------------------
# Against R
# ---------------------------------------------------------------------------
def _lung():
    lung = fixture("lung")
    keep = np.isfinite(lung["ph_ecog"])
    x, c = lung["time"][keep], lung["c"][keep]
    Z = np.column_stack([lung["age"], lung["sex"], lung["ph_ecog"]])[keep]
    return x, c, Z


def _case(name):
    if name.startswith("lung"):
        x, c, Z = _lung()
        fitter = {
            "lung_cox": sp.CoxPH,
            "lung_weibull": sp.WeibullPH,
            "lung_lognormal": sp.LogNormalAFT,
        }[name]
        return _quiet(fitter.fit, x, Z, c=c)
    if name == "ph_violation_cox":
        pv = fixture("ph_violation")
        Z = np.column_stack([pv["z1"], pv["z2"], pv["z3"]])
        return sp.CoxPH.fit(pv["x"], Z, c=pv["c"], tl=pv["tl"])
    lt = fixture("left_truncation")
    return sp.CoxPH.fit(lt["x"], lt["z"][:, None], c=lt["c"], tl=lt["tl"])


CASES = [
    "lung_cox",
    "lung_weibull",
    "lung_lognormal",
    "ph_violation_cox",
    "left_truncation_cox",
]


@pytest.mark.parametrize("name", CASES)
def test_the_curve_and_its_summaries_match_r(name):
    model = _case(name)
    d = model.diagnose()
    r = values("r_calibration", name)
    # survreg's fit (rel.tolerance 1e-12) leaves the parametric
    # predictions good to about 1e-6
    tol = FITTED if name.endswith("cox") else dict(rtol=1e-4, atol=1e-6)
    curves = _curves(d)
    assert [t0 for _, t0, _ in curves] == list(r["t0"])
    for k, (_, t0, curve) in enumerate(curves, 1):
        assert bool(r["linear_%d" % k]) == (curve.knots.size == 1)
        assert_allclose(curve.knots, r["knots_%d" % k], **tol)
        if not r["linear_%d" % k]:
            # the knots are the ones R's own rule places
            assert_allclose(
                curve.knots, r["knots_rcspline_eval_%d" % k], **tol
            )
        assert_allclose(curve.model.beta, r["coef_%d" % k], **tol)
        assert_allclose(curve.ici, r["ici_%d" % k], **tol)
        assert_allclose(curve.e50, r["e50_%d" % k], **tol)
        assert_allclose(curve.e90, r["e90_%d" % k], **tol)
        assert_allclose(
            [curve.predicted.min(), curve.predicted.max()],
            r["predicted_range_%d" % k],
            **tol,
        )
        observed, lower, upper = curve.at(r["grid_%d" % k])
        assert_allclose(observed, r["grid_observed_%d" % k], **tol)
        # survfit's variance has Efron's terms for tied failures, the
        # band here Breslow's: equal without ties, close with them
        band = 2e-4 if name.startswith("lung") else 5e-3
        assert_allclose(lower, r["grid_lower_%d" % k], atol=band)
        assert_allclose(upper, r["grid_upper_%d" % k], atol=band)
    # the table's statistic is the ICI
    assert_allclose(
        _rows(d)["statistic"], [r["ici_%d" % k] for k in (1, 2, 3)], **tol
    )


@pytest.mark.parametrize("name", CASES)
def test_austin_et_al_s_own_code_gives_the_same_ici(name):
    # rms::cph(rcs(cll, 3)) with survest, the paper's code: its baseline
    # survival is computed otherwise, so the ICI agrees to about 1e-5
    d = _case(name).diagnose()
    r = values("r_calibration", name)
    for k, (_, _, curve) in enumerate(_curves(d), 1):
        assert abs(curve.ici - r["ici_rms_%d" % k]) < 2e-5


def test_hazard_regression_is_close_but_not_equal():
    # Austin et al.'s other estimator, hare, is not in SurPyval; on the
    # lung data it agrees about how well each model is calibrated
    for name in ("lung_cox", "lung_weibull", "lung_lognormal"):
        r = values("r_calibration", name)
        d = _case(name).diagnose()
        for k, (_, _, curve) in enumerate(_curves(d), 1):
            assert abs(curve.ici - r["ici_hare_%d" % k]) < 0.035


def test_rcs_basis_is_rms_rcs():
    # Hmisc::rcspline.eval(c(0.5, 1.7, 2.5, 4), knots = c(0, 1.2, 3.5),
    # inclx = TRUE)
    assert_allclose(
        rcs_basis([0.5, 1.7, 2.5, 4.0], [0.0, 1.2, 3.5]),
        [
            [0.5, 0.01020408],
            [1.7, 0.38553327],
            [2.5, 1.00259095],
            [4.0, 2.50285714],
        ],
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# By hand
# ---------------------------------------------------------------------------
def test_a_case_computed_by_hand():
    # Ten failure times, the same in two groups predicted 0.2 and 0.6 at
    # t0 = 5. The groups' times are the same, so the calibration model's
    # coefficient is 0 and the observed risk of both is 1 - exp(-H) with
    # Efron's cumulative hazard for the pairs of tied failures,
    # H = sum over t <= 5 of 1/r + 1/(r - 1), r = 20, 18, ..., 12.
    x = np.tile(np.arange(1.0, 11.0), 2)
    survival = np.repeat([0.8, 0.4], 10)
    curve = calibration_curve(survival, x, t0=5.0)
    r = np.array([20.0, 18.0, 16.0, 14.0, 12.0])
    observed = 1 - np.exp(-np.sum(1 / r + 1 / (r - 1)))
    assert_allclose(observed, 0.4876623523221763)
    # two distinct predictions: they enter linearly
    assert curve.knots.size == 1
    assert_allclose(curve.model.beta, [0.0], atol=1e-12)
    assert_allclose(curve.observed, observed)
    gaps = np.repeat([observed - 0.2, 0.6 - observed], 10)
    assert_allclose(curve.ici, 0.2)  # the mean of the two gaps
    assert_allclose(curve.e50, 0.2)  # their mean, the median of 20
    assert_allclose(curve.e90, np.quantile(gaps, 0.9))
    assert_allclose(curve.e90, observed - 0.2)
    # the band holds the observed risk
    assert np.all(curve.lower < observed) and np.all(observed < curve.upper)


def test_a_cox_model_with_one_binary_covariate_is_calibrated_on_its_data():
    # With two distinct predictions the calibration model is the model
    # itself: observed equals predicted, ICI 0
    lt = fixture("left_truncation")
    d = _case("left_truncation_cox").diagnose()
    for _, _, curve in _curves(d):
        assert_allclose(curve.observed, curve.predicted, atol=1e-12)
        assert curve.ici < 1e-12
    assert (_rows(d)["verdict"] == "ok").all()
    assert lt["x"].size == 60


def test_weighted_quantile_is_numpy_s_of_the_rows_written_out():
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(size=12)
        w = rng.integers(1, 5, size=12)
        q = rng.uniform(size=5)
        assert_allclose(
            weighted_quantile(v, w, q), np.quantile(np.repeat(v, w), q)
        )
    assert lower_quantile([3.0, 1.0, 2.0], [1, 1, 2], 0.5) == 2.0


# ---------------------------------------------------------------------------
# Censoring and entry
# ---------------------------------------------------------------------------
def _exponential(rng, n, censor):
    z = rng.normal(size=n)
    scale = 10 * np.exp(-0.8 * z)
    t = rng.exponential(scale)
    cens = rng.exponential(censor, size=n)
    x = np.minimum(t, cens)
    return x, (t > cens).astype(int), z, scale


def test_censored_units_are_not_counted_as_survivors():
    # The true model's predictions, under heavy censoring (about 60%):
    # the calibration curve finds them calibrated, while counting the
    # censored units as survivors past t0 would put the observed risk far
    # below the predicted
    rng = np.random.default_rng(7)
    x, c, z, scale = _exponential(rng, 4000, censor=6.0)
    assert c.mean() > 0.5
    t0 = 5.0
    survival = np.exp(-t0 / scale)
    curve = calibration_curve(survival, x, c, t0=t0)
    assert curve.ici < 0.01
    naive = np.mean((x <= t0) & (c == 0))
    assert np.mean(1 - survival) - naive > 0.1


def test_delayed_entry_is_respected():
    # Units seen only if they survive to their entry time: with the entry
    # times the true model is calibrated; without, the curve sees too few
    # early failures
    rng = np.random.default_rng(8)
    n = 6000
    z = rng.normal(size=n)
    scale = 10 * np.exp(-0.8 * z)
    t = rng.exponential(scale)
    tl = rng.uniform(0, 8, size=n)
    seen = t > tl
    x, z, scale, tl = t[seen], z[seen], scale[seen], tl[seen]
    t0 = 6.0
    survival = np.exp(-t0 / scale)
    with_entry = calibration_curve(survival, x, tl=tl, t0=t0)
    without = calibration_curve(survival, x, t0=t0)
    assert with_entry.ici < 0.015
    assert without.ici > 3 * with_entry.ici


def test_a_cox_model_with_delayed_entry_is_diagnosed():
    d = _case("ph_violation_cox").diagnose()
    assert len(_rows(d)) == 3
    assert not any("Calibration is not checked" in n for n in d.notes)


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------
def _weibull_ph(rng, n, z, beta=0.6, shape=1.5):
    return 10 * (rng.exponential(size=n) / np.exp(beta * z)) ** (1 / shape)


def test_interval_and_left_censoring_are_refused_with_a_note():
    rng = np.random.default_rng(12)
    n = 150
    z = rng.binomial(1, 0.5, n).astype(float)
    t = _weibull_ph(rng, n, z)
    lo = np.floor(t)
    x = np.column_stack([lo, lo + 1])
    c = np.full(n, 2)
    c[lo == 0] = -1
    x[lo == 0] = 1
    for fitter in (sp.WeibullPH, sp.LogNormalAFT):
        d = _quiet(fitter.fit, x, z[:, None], c=c).diagnose()
        assert _rows(d).empty
        (note,) = [n for n in d.notes if n.startswith("Calibration is not")]
        assert "{} left- or interval-censored rows".format(n) in note
        assert "exact and right-censored" in note


def test_right_truncation_is_refused_with_a_note():
    rng = np.random.default_rng(12)
    z = rng.normal(size=300)
    t = _weibull_ph(rng, 300, z)
    seen = t < 15
    d = _quiet(sp.WeibullPH.fit, t[seen], z[seen, None], tr=15.0).diagnose()
    assert _rows(d).empty
    assert any(
        n.startswith("Calibration is not checked") and "right-truncated" in n
        for n in d.notes
    )


def test_time_varying_covariates_are_refused_with_a_note():
    i = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9]
    xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, 4]
    xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8, 2, 5, 4, 4, 9]
    c = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    Z = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    d = _quiet(sp.CoxPH.fit_tvc, i, xl, xr, c, Z).diagnose()
    assert _rows(d).empty
    assert any("time-varying covariates (fit_tvc)" in n for n in d.notes)


def test_too_few_failures_are_refused_with_a_note():
    rng = np.random.default_rng(2)
    z = rng.normal(size=40)
    t = _weibull_ph(rng, 40, z)
    c = (np.arange(40) >= 8).astype(int)  # 8 failures
    d = _quiet(sp.CoxPH.fit, t, z[:, None], c=c).diagnose()
    assert _rows(d).empty
    assert any(
        "8 failures" in n
        and "at least {}".format(calibration.MIN_FAILURES) in n
        for n in d.notes
    )


def test_calibration_curve_refuses_what_it_cannot_take():
    x = np.arange(1.0, 21.0)
    s = np.linspace(0.2, 0.9, 20)
    with pytest.raises(ValueError, match="left- or interval-censored"):
        calibration_curve(s, x, c=np.full(20, 2), t0=5.0)
    with pytest.raises(ValueError, match=r"each in \[0, 1\]"):
        calibration_curve(s + 1, x, t0=5.0)
    with pytest.raises(ValueError, match="one entry per unit"):
        calibration_curve(s[:-1], x, t0=5.0)
    with pytest.raises(ValueError, match="the same for every unit"):
        calibration_curve(np.full(20, 0.5), x, t0=5.0)


def test_separating_predictions_warn_that_the_curve_could_not_be_computed():
    # Predictions that order the failures perfectly, with the survivors at
    # the bottom: the calibration model has no finite maximum
    x = np.r_[np.arange(1.0, 21.0), np.full(10, 30.0)]
    c = np.r_[np.zeros(20), np.ones(10)].astype(int)
    s = np.r_[np.linspace(0.01, 0.5, 20), np.full(10, 0.99)]
    with pytest.raises(ValueError, match="no verified maximum"):
        calibration_curve(s, x, c, t0=10.0)


# ---------------------------------------------------------------------------
# In the reports
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rossi():
    from surpyval.datasets import load_rossi_static

    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    return df


@pytest.mark.parametrize(
    "fitter, family",
    [
        (sp.CoxPH, "Cox proportional hazards"),
        (sp.WeibullPH, "Parametric proportional hazards"),
        (sp.WeibullAFT, "Accelerated failure time"),
    ],
)
def test_the_check_is_in_each_family_s_report(rossi, fitter, family):
    model = _quiet(
        fitter.fit_from_df,
        rossi,
        x_col="week",
        c_col="censored",
        Z_cols=["fin", "age", "prio"],
    )
    d = model.diagnose()
    assert d.family == family
    rows = _rows(d)
    assert rows["test"].tolist() == [
        "ICI at t = 17 (25% of failures)",
        "ICI at t = 28 (50% of failures)",
        "ICI at t = 42 (75% of failures)",
    ]
    assert (rows["covariates"] == ()).all()
    assert rows["p"].isna().all() and rows["df"].isna().all()
    assert set(rows["verdict"]) <= {"ok", "warn"}
    assert rows["detail"].str.startswith("E50 ").all()
    # the rows come last, after the family's own checks
    assert d.table["check"].iloc[-1] == "calibration"
    # columns= chooses covariates; the check is of the whole model
    sub = model.diagnose(columns=["age"])
    pd.testing.assert_frame_equal(
        _rows(sub).reset_index(drop=True), rows.reset_index(drop=True)
    )


def test_to_dict_round_trips(rossi):
    model = sp.CoxPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    )
    d = model.diagnose()
    out = d.to_dict()
    text = json.dumps(out, allow_nan=False)
    back = Diagnosis.from_dict(json.loads(text))
    assert back.to_dict() == out
    pd.testing.assert_frame_equal(_rows(back), _rows(d))
    (row,) = [r for r in out["table"] if r["test"].startswith("ICI at t = 28")]
    assert row["check"] == "calibration" and row["covariates"] == []
    assert row["df"] is None and row["p"] is None
    assert_allclose(row["statistic"], _curves(d)[1][2].ici)


def test_plot_draws_the_curves_under_agg(rossi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = sp.CoxPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    )
    d = model.diagnose()
    fig = d.plot("calibration")
    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == 3
    assert titles[0].startswith("calibration at t = 17: ICI ")
    ax = fig.axes[1]
    assert ax.get_xlabel() == "predicted P(fail by t = 28)"
    assert ax.get_ylabel() == "observed P(fail by t = 28)"
    labels = ax.get_legend_handles_labels()[1]
    assert {"observed", "perfect calibration", "95% band"} <= set(labels)
    assert "predicted risks" in labels  # the histogram
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibrated and miscalibrated models
# ---------------------------------------------------------------------------
def _data(rng, n, quad=0.0, shape=1.5):
    Z = rng.normal(size=(n, 2))
    eta = Z @ np.array([0.7, -0.5]) + quad * Z[:, 0] ** 2
    t = 10 * (rng.exponential(size=n) / np.exp(eta)) ** (1 / shape)
    cens = rng.uniform(0, 30, n)
    return np.minimum(t, cens), Z, (t > cens).astype(int)


@pytest.mark.parametrize("fitter", [sp.CoxPH, sp.WeibullPH, sp.WeibullAFT])
def test_a_correctly_specified_model_does_not_warn(fitter):
    x, Z, c = _data(np.random.default_rng(31), 500)
    d = _quiet(fitter.fit, x, Z, c=c).diagnose()
    rows = _rows(d)
    assert (rows["verdict"] == "ok").all(), rows
    assert (rows["statistic"] < 0.03).all()
    assert not [a for a in d.advice if a.check == "calibration"]


def test_ici_grows_with_miscalibration():
    # The true model's coefficients, halved and doubled, at the median
    # failure time: ICI is small for the true ones and grows the further
    # they are from them
    x, Z, c = _data(np.random.default_rng(5), 1000)
    t0 = lower_quantile(x[c == 0], np.ones(int((c == 0).sum())), 0.5)
    H0 = (t0 / 10) ** 1.5
    eta = Z @ np.array([0.7, -0.5])
    ici = {
        k: calibration_curve(np.exp(-H0 * np.exp(k * eta)), x, c, t0=t0).ici
        for k in (0.25, 0.5, 1.0, 2.0)
    }
    assert ici[1.0] < 0.01
    assert ici[1.0] < ici[0.5] < ici[0.25]
    assert ici[2.0] > 0.05


def test_a_model_fitted_to_other_data_warns():
    # Fitted to times 1.5 times as long, predicting these
    rng = np.random.default_rng(6)
    x, Z, c = _data(rng, 400)
    xo, Zo, co = _data(rng, 400)
    other = sp.CoxPH.fit(1.5 * xo, Zo, c=co)
    t0 = lower_quantile(x[c == 0], np.ones(int((c == 0).sum())), 0.5)
    curve = calibration_curve(other.sf(np.full(400, t0), Z), x, c, t0=t0)
    assert curve.ici > 0.05 and curve.outside > 0.5


def test_a_wrong_baseline_warns_and_is_advised():
    # An exponential PH model of Weibull (shape 2) times
    x, Z, c = _data(np.random.default_rng(9), 400, shape=2.0)
    d = _quiet(sp.ExponentialPH.fit, x, Z, c=c).diagnose()
    rows = _rows(d)
    assert (rows["verdict"] == "warn").any()
    warned = rows[rows["verdict"] == "warn"].iloc[0]
    assert warned["statistic"] > ICI_LIMIT
    assert "outside the curve's 95% band" in warned["detail"]
    # several rows warn, and the advice is given once, naming each time
    assert (rows["verdict"] == "warn").sum() >= 2
    (advice,) = [a for a in d.advice if a.check == "calibration"]
    assert advice.count("at t = ") == (rows["verdict"] == "warn").sum()
    assert advice.features == (
        "LogNormalAFT.fit",
        "ExponentialPH.fit_from_df(formula=)",
        "CoxPH.fit",
    )
    assert "baseline_distribution rows" in advice
    assert "CoxPH.fit" in advice and "d.plot('calibration')" in advice


def test_a_missing_quadratic_term_warns_a_cox_model():
    x, Z, c = _data(np.random.default_rng(10), 1000, quad=0.5)
    d = sp.CoxPH.fit(x, Z, c=c).diagnose()
    assert (_rows(d)["verdict"] == "warn").any()
    (advice,) = [a for a in d.advice if a.check == "calibration"]
    assert advice.features == ("CoxPH.fit_from_df(formula=)",)
    assert "functional_form rows" in advice


def test_the_lung_log_normal_model_warns_early():
    d = _case("lung_lognormal").diagnose()
    assert _rows(d)["verdict"].tolist() == ["warn", "ok", "ok"]
    detail = _rows(d)["detail"].iloc[0]
    assert "the observed risk is lower than predicted" in detail
    assert "too extreme" in detail
    (advice,) = [a for a in d.advice if a.check == "calibration"]
    assert advice.startswith("The predicted survival at t = 135 is off")
    assert advice.features == (
        "LogNormalPH.fit",
        "LogNormalAFT.fit_from_df(formula=)",
        "CoxPH.fit",
    )
    # the Weibull, which the AIC screen prefers, is calibrated
    assert (_rows(_case("lung_weibull").diagnose())["verdict"] == "ok").all()


def test_the_rule_needs_both_a_large_ici_and_a_clear_one():
    assert ICI_LIMIT == 0.02 and OUTSIDE_LIMIT == 0.1
    # a small sample's noisy curve: ICI above the limit, but within the
    # band, so no warning
    x, Z, c = _data(np.random.default_rng(4), 60)
    d = _quiet(sp.WeibullPH.fit, x, Z, c=c).diagnose()
    rows = _rows(d)
    curves = _curves(d)
    for (_, row), (_, _, curve) in zip(rows.iterrows(), curves):
        off = curve.ici > ICI_LIMIT and curve.outside > OUTSIDE_LIMIT
        assert row["verdict"] == ("warn" if off else "ok")


def test_advice_names_the_model_s_own_fitters():
    rules = calibration_advice("PH", ("Weibull", "Gamma"), ("Weibull",))
    assert [r.features for r in rules] == [
        ("LogNormalAFT.fit", "WeibullPH.fit_from_df(formula=)", "CoxPH.fit"),
        ("GammaAFT.fit", "GammaPH.fit_from_df(formula=)", "CoxPH.fit"),
        ("PH", "AFT", "CoxPH.fit"),
    ]
    rules = calibration_advice("AFT", ("LogNormal",), ())
    assert rules[0].features == (
        "LogNormalPH.fit",
        "LogNormalAFT.fit_from_df(formula=)",
        "CoxPH.fit",
    )
    with pytest.raises(ValueError, match="'cox', 'PH' or 'AFT'"):
        calibration_advice("frailty")


# ---------------------------------------------------------------------------
# Principles
# ---------------------------------------------------------------------------
def test_counts_equal_repetition_and_order_does_not_matter():
    rng = np.random.default_rng(11)
    x, Z, c = _data(rng, 120)
    x = np.round(x, 1)
    n = rng.integers(1, 4, size=120)
    t0 = 5.0
    s = np.exp(-((t0 / 10) ** 1.5) * np.exp(Z @ np.array([0.7, -0.5])))
    counted = calibration_curve(s, x, c, n, t0=t0)
    written = calibration_curve(
        np.repeat(s, n), np.repeat(x, n), np.repeat(c, n), t0=t0
    )
    for key in ("ici", "e50", "e90", "outside"):
        assert_allclose(getattr(counted, key), getattr(written, key), **FITTED)
    assert_allclose(counted.knots, written.knots, **FITTED)
    order = rng.permutation(120)
    shuffled = calibration_curve(s[order], x[order], c[order], n[order], t0=t0)
    assert_allclose(shuffled.ici, counted.ici, **FITTED)


def test_rescaling_time_rescales_t0_and_nothing_else(rossi):
    def fit(scale):
        df = rossi.assign(week=rossi["week"] * scale)
        return sp.CoxPH.fit_from_df(
            df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
        ).diagnose()

    one, many = _curves(fit(1.0)), _curves(fit(7.0))
    for (_, t1, a), (_, t7, b) in zip(one, many):
        assert_allclose(t7, 7 * t1)
        assert_allclose(b.ici, a.ici, **FITTED)
        assert_allclose(b.e90, a.e90, **FITTED)
