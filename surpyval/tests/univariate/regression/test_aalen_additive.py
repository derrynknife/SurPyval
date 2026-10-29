"""Aalen's additive hazards model, ``surpyval.AalenAdditive``.

References:

- R ``timereg`` 2.0.5, ``aalen(..., robust = 0)``: the cumulative
  coefficients ``cum`` and their variances ``var.cum`` on the additive and
  veteran fixtures, stored in ``tests/diagnostics/data/r_odds_additive.json``
  by ``scripts/reference/reference_r_odds_additive.R``.
- lifelines 0.30.3, ``AalenAdditiveFitter()`` on the Rossi recidivism data
  (``week``, ``arrest``; ``fin``, ``age``, ``prio``): its
  ``cumulative_hazards_``, the intercept's ``cumulative_variance_``,
  ``predict_survival_function`` and ``smoothed_hazards_(bandwidth)``,
  pasted below. (Its covariates' ``cumulative_variance_`` are rescaled by
  each covariate's standard deviation rather than its square, so the
  variances are checked against timereg instead.)

The estimator is closed form, so the agreement is to rounding.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.diagnostics.odds_additive import aalen_cumulative_coefficients
from surpyval.tests.reference._data import fixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXACT = dict(rtol=1e-8, atol=1e-12)
R_DATA = (
    Path(__file__).parents[2] / "diagnostics" / "data" / "r_odds_additive.json"
)


def _timereg(ref_id):
    with open(R_DATA) as f:
        entry = json.load(f)["references"][ref_id]
    return {
        k: np.array(v, dtype=float) if isinstance(v, list) else v
        for k, v in entry["values"].items()
    }


def _rossi():
    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    return df


def _rossi_model(**kw):
    return sp.AalenAdditive.fit_from_df(
        _rossi(),
        x_col="week",
        c_col="censored",
        Z_cols=["fin", "age", "prio"],
        **kw,
    )


def _veteran():
    """R's veteran data with the ties broken as the R script breaks them
    (timereg's aalen takes one event per time)."""
    d = fixture("veteran")
    rows = np.arange(d["time"].size)
    order = np.lexsort((rows, -d["status"], d["time"]))
    d = {k: v[order] for k, v in d.items()}
    t = d["time"]
    rank = np.zeros(t.size)
    for k in range(1, t.size):
        rank[k] = rank[k - 1] + 1 if t[k] == t[k - 1] else 0
    Z = np.column_stack([d["trt"], d["karno"], d["age"], d["celltype"]])
    return t + 0.001 * rank, Z, 1 - d["status"]


# -- against R and lifelines -------------------------------------------------
def test_cumulative_coefficients_match_timereg_additive():
    d = fixture("additive")
    model = sp.AalenAdditive.fit(d["x"], np.c_[d["z1"], d["z2"]], c=d["c"])
    ref = _timereg("aalen_additive")
    assert_allclose(model.x, ref["time"], rtol=0, atol=0)
    assert_allclose(model.B, ref["cum"], **EXACT)
    assert_allclose(model.B_var, ref["var_cum"], **EXACT)


def test_cumulative_coefficients_match_timereg_veteran():
    x, Z, c = _veteran()
    model = sp.AalenAdditive.fit(x, Z, c=c)
    ref = _timereg("aalen_veteran")
    k = model.x.size
    assert_allclose(model.x, ref["time"][:k], rtol=0, atol=0)
    assert_allclose(model.B, ref["cum"][:k], **EXACT)
    assert_allclose(model.B_var, ref["var_cum"][:k], **EXACT)
    # Past the first risk set too small to separate the covariates
    # timereg holds the estimate, and the model holds its last value too.
    assert k < len(ref["time"])
    later = ref["time"][k:]
    assert_allclose(
        model.cumulative_coefficients(later).to_numpy(),
        np.repeat(model.B[-1:], later.size, axis=0),
    )


def test_unpenalised_fit_is_the_diagnostics_estimator():
    df = _rossi()
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    tl = np.where(np.arange(len(df)) % 3 == 0, 2.0, 0.0)
    for entry in (None, tl):
        ref = aalen_cumulative_coefficients(
            df["week"], Z, c=df["censored"], tl=entry
        )
        model = sp.AalenAdditive.fit(df["week"], Z, c=df["censored"], tl=entry)
        assert_array_equal(model.x, ref["time"])
        assert_array_equal(model.B, ref["cum"])
        assert_array_equal(model.B_var, ref["var"])


# lifelines 0.30.3: AalenAdditiveFitter().fit(rossi, "week", "arrest"),
# columns (fin, age, prio, Intercept).
LIFELINES_B = {
    10: [-0.014986737956609975, 0.0009114060357683264, 0.004194715030605972,
         0.0077321363501417426],
    30: [-0.07488579568062667, -0.0018950659686595814, 0.023218513884787684,
         0.16577081704620644],
    52: [-0.10368542515662754, -0.015193115152489434, 0.03656310897070047,
         0.6314756844907388],
}  # fmt: skip
LIFELINES_INTERCEPT_VAR = {
    10: 0.0021216040120597183,
    30: 0.005997841661848516,
    52: 0.013620354292592257,
}
# predict_survival_function at week 52 of (fin, age, prio) = (0, 25, 3)
# and (1, 25, 3).
LIFELINES_SF_52 = [0.6967440637358252, 0.7728643681491365]
# smoothed_hazards_(bandwidth) at weeks 10 and 30, without the division by
# the bandwidth.
LIFELINES_SMOOTHED = {
    1: [[-0.003523087500643298, -5.107554787080106e-05, 0.0023450061251923202,
         -0.002160037126953991],
        [0.00044355635687668094, -0.0005750728674913773,
         -0.0008553043558481107, 0.020367355358756917]],
    5: [[0.009272909495165876, -0.00024165176369160383, 0.006999523755733271,
         0.0044701436770612744],
        [-0.022470269114608504, -0.001770680711206017, 0.0016127763232882692,
         0.07224904202607024]],
}  # fmt: skip
# lifelines' column order -> the model's (baseline first)
_ORDER = ["fin", "age", "prio", "baseline"]


def test_match_lifelines_on_rossi():
    model = _rossi_model()
    B = model.cumulative_coefficients([10, 30, 52])[_ORDER].to_numpy()
    assert_allclose(B, [LIFELINES_B[t] for t in (10, 30, 52)], rtol=1e-10)
    var = model.B_var[np.searchsorted(model.x, [10, 30, 52]), 0]
    assert_allclose(
        var, [LIFELINES_INTERCEPT_VAR[t] for t in (10, 30, 52)], rtol=1e-10
    )
    # lifelines predicts with the estimate H = B'[1, Z] itself; the model
    # with its running maximum, which is the same where H does not fall
    # (the first row) and above it where it does (the second dips).
    rows = np.array([[1.0, 0, 25, 3], [1.0, 1, 25, 3]])
    raw = rows @ model.cumulative_coefficients([52]).to_numpy()[0]
    assert_allclose(np.exp(-raw), LIFELINES_SF_52, rtol=1e-10)
    sf = model.sf(52, rows[:, 1:])
    assert_allclose(sf[0], LIFELINES_SF_52[0], rtol=1e-10)
    assert sf[1] < LIFELINES_SF_52[1]
    for bw, ref in LIFELINES_SMOOTHED.items():
        b = model.smoothed_coefficients([10, 30], bandwidth=bw)[_ORDER]
        assert_allclose(b.to_numpy() * bw, ref, rtol=1e-10, atol=1e-15)


# -- the estimator ---------------------------------------------------------
def test_ridge_penalty_is_the_closed_form():
    df = _rossi()
    x, c = df["week"].to_numpy(float), df["censored"].to_numpy()
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    lam = 50.0
    model = sp.AalenAdditive.fit(x, Z, c=c, coef_penalty=lam)
    # By hand: at each event time (X'X + P)^{-1} X'dN over the risk set,
    # P = lam * diag(0, var(Z_j)).
    X = np.c_[np.ones(x.size), Z]
    P = np.diag(np.r_[0.0, lam * Z.var(axis=0, ddof=1)])
    B = np.zeros(4)
    for k, t in enumerate(np.unique(x[c == 0])):
        risk = x >= t
        dN = ((x == t) & (c == 0)).astype(float)
        A = X[risk].T @ X[risk] + P
        B = B + np.linalg.solve(A, X.T @ dN)
        assert_allclose(model.B[k], B, rtol=1e-10, atol=1e-14)
    assert model.coef_penalty == lam
    # Shrinks every covariate's effect, not the baseline's sign.
    plain = _rossi_model()
    assert np.all(np.abs(model.B[-1, 1:]) < np.abs(plain.B[-1, 1:]))


def test_penalty_does_not_depend_on_units():
    df = _rossi()
    a = _rossi_model(coef_penalty=10.0)
    df["age"] = df["age"] / 10
    b = sp.AalenAdditive.fit_from_df(
        df,
        x_col="week",
        c_col="censored",
        Z_cols=["fin", "age", "prio"],
        coef_penalty=10.0,
    )
    assert_allclose(b.B[:, 2], a.B[:, 2] * 10, rtol=1e-9)
    assert_allclose(b.sf([20, 40], [0, 2.5, 3]), a.sf([20, 40], [0, 25, 3]))


def test_delayed_entry_splits_exactly():
    """A unit followed on (0, x] is the same as one censored at a and one
    entering at a and leaving at x: the risk sets are the same."""
    df = _rossi()
    x, c = df["week"].to_numpy(float), df["censored"].to_numpy()
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    a = np.where(x > 20, 20.0, 0.0)
    split = x > 20
    x2 = np.r_[x[~split], np.full(split.sum(), 20.0), x[split]]
    c2 = np.r_[c[~split], np.ones(split.sum(), int), c[split]]
    tl2 = np.r_[np.zeros((~split).sum() + split.sum()), a[split]]
    Z2 = np.r_[Z[~split], Z[split], Z[split]]
    whole = sp.AalenAdditive.fit(x, Z, c=c)
    pieces = sp.AalenAdditive.fit(x2, Z2, c=c2, tl=tl2)
    assert_array_equal(whole.x, pieces.x)
    assert_allclose(pieces.B, whole.B, rtol=1e-10, atol=1e-14)
    assert_allclose(pieces.B_var, whole.B_var, rtol=1e-10, atol=1e-16)


# -- prediction ------------------------------------------------------------
def test_cumulative_hazard_is_the_running_maximum():
    model = _rossi_model()
    # A row the model gives a negative hazard at times: many priors offset
    # by a large age.
    z = [1, 44, 0]
    raw = model.B @ np.r_[1.0, z]
    assert np.any(np.diff(raw) < 0)
    H = model.Hf(model.x, z)
    assert_allclose(H, np.maximum.accumulate(np.maximum(raw, 0.0)))
    s = model.sf(np.linspace(0, 60, 61), z)
    assert np.all(np.diff(s) <= 0) and np.all((s >= 0) & (s <= 1))
    # 0 before the first event, held after the last
    assert model.Hf(0.5, z) == 0.0
    assert model.Hf(1e3, z) == model.Hf(model.x[-1], z)
    assert model.hf(1e3, z) == 0.0 and model.hf(0.0, z) == 0.0


def test_quantile_inverts_the_curve():
    model = _rossi_model()
    z = [0, 25, 3]
    q = model.qf([0.05, 0.1, 0.2], z)
    F = model.ff(q, z)
    assert np.all(F >= [0.05, 0.1, 0.2])
    before = model.ff(np.nextafter(q, 0), z)
    assert np.all(before < [0.05, 0.1, 0.2])
    # never reached: the curve ends above 1 - 0.9
    assert np.isnan(model.qf(0.9, z))


def test_hazard_is_the_smoothed_slope():
    model = _rossi_model()
    z = np.array([0, 25, 3])
    b = model.smoothed_coefficients([20.0, 30.0], bandwidth=4).to_numpy()
    want = np.maximum(b @ np.r_[1.0, z], 0.0)
    assert_allclose(model.hf([20.0, 30.0], z, bandwidth=4), want)
    assert_allclose(
        model.df([20.0, 30.0], z, bandwidth=4),
        want * model.sf([20.0, 30.0], z),
    )


def test_bounds_are_wald_on_log_cumulative_hazard():
    model = _rossi_model()
    z = np.array([0, 25, 3])
    t = np.array([10.0, 30.0, 52.0])
    H = model.Hf(t, z)
    X = np.r_[1.0, z]
    C = model._B_cov[np.searchsorted(model.x, t, side="right") - 1]
    se = np.sqrt(np.einsum("j,ijk,k->i", X, C, X))
    z975 = norm.isf(0.025)
    want = np.c_[H * np.exp(-z975 * se / H), H * np.exp(z975 * se / H)]
    assert_allclose(model.cb(t, z, on="Hf"), want, rtol=1e-12)
    assert_allclose(model.cb(t, z), np.exp(-want[:, ::-1]), rtol=1e-12)
    assert_allclose(model.cb(t, z, on="R"), model.cb(t, z, on="sf"))
    # At Z = 0 the standard error is the baseline's.
    assert_allclose(
        model._H_se(t, np.tile([1.0, 0, 0, 0], (3, 1))),
        np.sqrt(model.B_var[np.searchsorted(model.x, t, "right") - 1, 0]),
    )
    # A one-sided bound at alpha is the two-sided one at 2 alpha.
    two = model.cb(t, z, alpha_ci=0.1)
    assert_allclose(model.cb(t, z, bound="lower", alpha_ci=0.05), two[:, 0])
    assert_allclose(model.cb(t, z, bound="upper", alpha_ci=0.05), two[:, 1])
    two_ff = model.cb(t, z, on="ff", alpha_ci=0.1)
    assert_allclose(
        model.cb(t, z, on="ff", bound="lower", alpha_ci=0.05), two_ff[:, 0]
    )


def test_cumulative_coefficient_bounds_and_summary():
    model = _rossi_model()
    band = model.cumulative_coefficients_cb([52.0])
    se = np.sqrt(model.B_var[-1])
    z975 = norm.isf(0.025)
    assert_allclose(band[0, :, 0], model.B[-1] - z975 * se)
    table = model.summary()
    assert list(table.index) == ["baseline", "fin", "age", "prio"]
    assert_allclose(table["coef"], model.B[-1])
    assert_allclose(table["se(coef)"], se)
    assert_allclose(table["coef lower 95%"], band[0, :, 0])
    assert table["exp(coef)"].isna().all()
    early = model.summary(at=10.0)
    assert_allclose(
        early["coef"], model.cumulative_coefficients([10.0]).iloc[0]
    )


def test_constant_effects_test_is_seeded_and_finds_a_changing_effect():
    """Veteran: the Karnofsky score's effect fades over follow-up (its
    period trend test against timereg rejects it too)."""
    x, Z, c = _veteran()
    model = sp.AalenAdditive.fit(
        x, pd.DataFrame(Z, columns=["trt", "karno", "age", "cell"]), c=c
    )
    a = model.constant_effects_test(n_sim=500, random_state=3)
    b = model.constant_effects_test(n_sim=500, random_state=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["karno", "KS p"] < 0.01
    # The observed statistics are deterministic: the distance from the
    # straight line to the last value.
    B = model.B[:, 2]
    line = model.x / model.x[-1] * B[-1]
    before = np.r_[0.0, B[:-1]]
    ks = max(np.abs(B - line).max(), np.abs(before - line).max())
    assert_allclose(a.loc["karno", "KS"], ks)


def test_plot_draws_the_steps_and_bands():
    model = _rossi_model()
    axes = model.plot(terms=["fin", "prio"])
    assert len(axes) == 2
    line = axes[1].lines[0]
    assert_allclose(line.get_xdata(), np.r_[0.0, model.x])
    assert_allclose(line.get_ydata(), np.r_[0.0, model.B[:, 3]])
    assert axes[1].get_title() == "prio"
    smooth = model.plot(terms="age", smoothed=True, bandwidth=4)
    t = smooth[0].lines[0].get_xdata()
    assert_allclose(
        smooth[0].lines[0].get_ydata(),
        model.smoothed_coefficients(t, bandwidth=4)["age"],
    )
    plt.close("all")


def test_saved_model_keeps_predictions_and_refuses_the_test():
    model = _rossi_model()
    back = sp.from_json(model.to_json())
    z = [0, 25, 3]
    assert_allclose(back.cb([10, 52], z), model.cb([10, 52], z))
    pd.testing.assert_frame_equal(back.summary(), model.summary())
    with pytest.raises(ValueError, match="refit it"):
        back.constant_effects_test()


# -- refusals ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"c": [0, 2, 0, 1]}, "right-censored"),
        ({"x": [-1, 2, 3, 4]}, "non-negative"),
        ({"tl": [0, 3, 0, 0]}, "strictly less than"),
        ({"c": [1, 1, 1, 1]}, "at least one event"),
        ({"coef_penalty": -1}, "coef_penalty"),
    ],
)
def test_refusals_say_what_to_do(kwargs, match):
    data = {"x": [1, 2, 3, 4], "Z": [[0], [1], [0], [1]], "c": [0, 0, 1, 0]}
    if "c" in kwargs and 2 in kwargs["c"]:
        data["x"] = [[1, 1], [2, 2.5], [3, 3], [4, 4]]
    data.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sp.AalenAdditive.fit(**data)


def test_prediction_refusals():
    model = _rossi_model()
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(10, [0, 25, 3], method="lr")
    with pytest.raises(ValueError, match="Unknown term"):
        model.cumulative_coefficients(terms="wexp")
    with pytest.raises(ValueError, match="bandwidth"):
        model.smoothed_coefficients([10], bandwidth=0)
    with pytest.raises(ValueError, match="n_sim"):
        model.constant_effects_test(n_sim=0)
