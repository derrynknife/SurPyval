"""The proportional odds and additive hazards families of ``diagnose()``
(#870): their statistics against R (``timereg`` and ``survival``, stored in
``data/r_odds_additive.json`` by
``scripts/reference/reference_r_odds_additive.R``), the checks on data
built to pass or to fail them, the advice each failure calls for, the
refusals, and the options of the fits they support.

Tolerances: Lin and Ying's estimator and Aalen's are closed form, so the
statistics agree with timereg's to rounding (``rtol=1e-8``); the
Kaplan-Meier estimate is exact.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics import odds_additive as oa
from surpyval.diagnostics import registry
from surpyval.tests.reference._data import fixture

EXACT = dict(rtol=1e-8, atol=1e-12)
DATA = Path(__file__).parent / "data" / "r_odds_additive.json"


def _reference(ref_id):
    with open(DATA) as f:
        entry = json.load(f)["references"][ref_id]
    return {
        k: np.array(v, dtype=float) if isinstance(v, list) else v
        for k, v in entry["values"].items()
    }


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _rows(d, verdict=None):
    table = d.table
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _row(d, name):
    return d.table[d.table["covariates"] == (name,)].iloc[0]


# -- R references -------------------------------------------------------------
def _additive():
    d = fixture("additive")
    Z = pd.DataFrame({"z1": d["z1"], "z2": d["z2"]})
    return sp.AdditiveHazards.fit(d["x"], Z, c=d["c"]), ["z1", "z2"]


def _veteran():
    """R's veteran data with the tied times broken as the R script breaks
    them (timereg's aalen takes one event per time)."""
    d = fixture("veteran")
    rows = np.arange(d["time"].size)
    order = np.lexsort((rows, -d["status"], d["time"]))
    d = {k: v[order] for k, v in d.items()}
    t = d["time"]
    rank = np.zeros(t.size)
    for k in range(1, t.size):
        rank[k] = rank[k - 1] + 1 if t[k] == t[k - 1] else 0
    names = ["trt", "karno", "age", "cell"]
    Z = pd.DataFrame(
        {
            "trt": d["trt"],
            "karno": d["karno"],
            "age": d["age"],
            "cell": d["celltype"],
        }
    )
    model = sp.AdditiveHazards.fit(t + 0.001 * rank, Z, c=1 - d["status"])
    return model, names


CASES = {"additive": _additive, "veteran": _veteran}


@pytest.mark.parametrize("case", sorted(CASES))
def test_lin_ying_trend_tests_match_timereg(case):
    model, names = CASES[case]()
    d = model.diagnose()
    for name in names:
        ref = _reference("trend_{}_{}".format(case, name))
        row = _row(d, name)
        assert row["df"] == int(ref["df"])
        assert_allclose(row["statistic"], ref["statistic"], **EXACT)
    ref = _reference("trend_{}_global".format(case))
    glob = d.table[d.table["test"].str.contains("GLOBAL")].iloc[0]
    assert glob["df"] == int(ref["df"])
    assert_allclose(glob["statistic"], ref["statistic"], **EXACT)
    # The cuts are R's (the event-time quartiles)
    cuts = ", ".join("{:.3g}".format(v) for v in np.atleast_1d(ref["cuts"]))
    assert any(cuts in note for note in d.notes), (cuts, d.notes)
    assert_allclose(oa._cuts(d._context), ref["cuts"], rtol=0, atol=0)


def test_the_veteran_karno_effect_is_not_constant():
    # The textbook case: the Karnofsky score's effect fades
    model, _ = _veteran()
    d = model.diagnose()
    karno = _row(d, "karno")
    assert karno["verdict"] == "fail"
    assert karno["detail"] == "its coefficient rises with time"
    assert _row(d, "cell")["df"] == 3  # four levels: three indicators
    assert _row(d, "cell")["test"].endswith("trends in the group effects")


@pytest.mark.parametrize("case", sorted(CASES))
def test_aalen_cumulative_coefficients_match_timereg(case):
    model, names = CASES[case]()
    data = model._fit_data
    out = oa.aalen_cumulative_coefficients(
        data[0], data[3], c=data[1], n=data[2]
    )
    ref = _reference("aalen_" + case)
    k = out["time"].size
    assert_allclose(out["time"], ref["time"][:k], rtol=0, atol=0)
    assert_allclose(out["cum"], ref["cum"][:k], **EXACT)
    assert_allclose(out["var"], ref["var_cum"][:k], **EXACT)
    # Past the first risk set too small to separate the covariates
    # timereg holds the estimate; SurPyval stops there.
    assert_allclose(
        ref["cum"][k:], ref["cum"][[k - 1] * (len(ref["time"]) - k)]
    )
    assert k == len(ref["time"]) or case == "veteran"


def test_kaplan_meier_log_odds_match_survfit():
    d = fixture("lung")
    ref = _reference("km_lung_sex")
    for sex in (1.0, 2.0):
        rows = d["sex"] == sex
        t, S = oa._kaplan_meier(
            d["time"][rows],
            d["c"][rows].astype(int),
            np.ones(rows.sum()),
            np.full(rows.sum(), -np.inf),
        )
        mine = ref["sex"] == sex
        assert_allclose(t, ref["time"][mine], rtol=0, atol=0)
        assert_allclose(S, ref["surv"][mine], rtol=1e-12)


def test_the_split_null_fit_is_the_fit():
    # The episode-splitting identity the likelihood-ratio tests rest on:
    # the model refitted on the rows split at the cuts is the model.
    rossi = sp.datasets.load_rossi_static()
    x, c = rossi["week"].values, 1 - rossi["arrest"].values
    Z = rossi[["fin", "age", "prio"]].values.astype(float)
    for fitter in (sp.ProportionalOdds, sp.WeibullPO, sp.LogNormalPO):
        model = _quiet(fitter.fit, x, Z, c=c)
        ctx = registry.Context(model, 0.05, ["a", "b", "c"], [0, 1, 2])
        refit = oa._fit_split(ctx, oa._covariates_split(ctx))
        assert_allclose(
            oa._coefficients(refit), oa._coefficients(model), rtol=1e-5
        )
        if fitter is sp.ProportionalOdds:
            assert_allclose(
                refit.log_likelihood, model.log_likelihood, rtol=1e-12
            )
        else:
            assert_allclose(refit._neg_ll, model._neg_ll, rtol=1e-9)


# -- data that meet, and break, each assumption -------------------------------
def _po_times(rng, eta, shape=1.5):
    """Times whose survival odds are exp(eta) times a Weibull baseline's
    (scale 1, ``shape``)."""
    u = rng.uniform(size=eta.size)
    phi = np.exp(eta)
    S0 = u / (phi + u - u * phi)
    return (-np.log(S0)) ** (1 / shape)


def _frame(t, cens, **covariates):
    return pd.DataFrame(
        dict(x=np.minimum(t, cens), c=(t > cens).astype(int), **covariates)
    )


def _fit(fitter, df, Z_cols, **kwargs):
    return _quiet(
        fitter.fit_from_df, df, x_col="x", c_col="c", Z_cols=Z_cols, **kwargs
    )


@pytest.fixture(scope="module")
def po_data():
    rng = np.random.default_rng(5)
    n = 300
    z, w = rng.binomial(1, 0.5, n), rng.normal(size=n)
    t = _po_times(rng, 0.7 * z + 0.5 * w)
    return _frame(t, rng.uniform(0, 3, n), z=z, w=w)


@pytest.fixture(scope="module")
def ah_data():
    rng = np.random.default_rng(7)
    n = 300
    z, w = rng.binomial(1, 0.5, n), rng.uniform(0, 2, n)
    t = rng.exponential(1 / (0.5 + 0.4 * z + 0.3 * w))
    return _frame(t, rng.uniform(0, 3, n), z=z, w=w)


@pytest.mark.parametrize("fitter", [sp.ProportionalOdds, sp.WeibullPO])
def test_proportional_odds_data_pass(po_data, fitter):
    d = _fit(fitter, po_data, ["z", "w"]).diagnose()
    assert d.ok, d
    assert not d.advice
    assert d.family == "Proportional odds"
    assert set(d.table["check"]) == {"proportional_odds"}
    assert _row(d, "z")["test"].endswith("parallel log-odds across groups")
    assert _row(d, "w")["test"].endswith("of a coefficient trend")
    assert d.table["detail"].iloc[0] == (
        "no evidence that its effect changes with time"
    )


@pytest.mark.parametrize("fitter", [sp.AdditiveHazards, sp.WeibullAH])
def test_additive_hazards_data_pass(ah_data, fitter):
    d = _fit(fitter, ah_data, ["z", "w"]).diagnose()
    assert d.ok, d
    assert not d.advice
    assert d.family == "Additive hazards"
    assert set(d.table["check"]) == {"constant_effects"}


def _log_logistic(rng, shape):
    """Log-logistic times whose odds of failure are ``t ** shape``: the
    log-odds are lines in log t of slope ``shape``, parallel only for a
    common shape."""
    u = rng.uniform(size=shape.size)
    return ((1 - u) / u) ** (1 / shape)


@pytest.fixture(scope="module")
def crossing():
    """z changes the slope of the log-odds (not proportional odds); w acts
    through proportional odds."""
    rng = np.random.default_rng(11)
    n = 300
    z, w = rng.binomial(1, 0.5, n), rng.normal(size=n)
    t = _log_logistic(rng, np.where(z == 1, 2.5, 1.0)) * np.exp(0.3 * w)
    return _frame(t, rng.uniform(0, 4, n), z=z, w=w)


def test_non_parallel_log_odds_are_advised_ph_or_aft(crossing):
    d = _fit(sp.ProportionalOdds, crossing, ["z", "w"]).diagnose()
    z = _row(d, "z")
    assert z["verdict"] == "fail"
    assert z["detail"].startswith(
        "its groups' log-odds of failure are not parallel"
    )
    (advice,) = [a for a in d.advice if a.covariates == ("z",)]
    assert "CoxPH.fit" in advice and "WeibullAFT.fit" in advice
    assert advice.features == (
        "CoxPH.fit",
        "WeibullAFT.fit",
        "LogNormalAFT.fit",
    )
    assert advice.check == "proportional_odds"


@pytest.mark.parametrize(
    "fitter, named",
    [
        (sp.WeibullPO, ("WeibullPH.fit", "WeibullAFT.fit")),
        (sp.LogNormalPO, ("LogNormalPH.fit", "LogNormalAFT.fit")),
        (sp.PO(sp.LogLogistic), ("PH(LogLogistic).fit", "AFT(LogLogistic)")),
    ],
)
def test_a_parametric_failure_names_the_same_baseline(crossing, fitter, named):
    d = _fit(fitter, crossing, ["z", "w"]).diagnose()
    assert _row(d, "z")["verdict"] == "fail"
    (advice,) = [a for a in d.advice if a.covariates == ("z",)]
    for feature in named:
        assert feature in advice
    if fitter is sp.WeibullPO:
        assert advice.features == (
            "WeibullPH.fit",
            "WeibullAFT.fit",
            "CoxPH.fit",
        )
    # ... and the fits it names take the same data
    ph = named[0].split(".fit")[0]
    refit = sp.PH(sp.LogLogistic) if "(" in ph else getattr(sp, ph)
    _fit(refit, crossing, ["z", "w"])


def _reversing(rng, n):
    """A continuous covariate whose log hazard ratio is +1 before t = 0.5
    and -1 after."""
    w = rng.normal(size=n)
    e = rng.exponential(size=n)
    early, late = np.exp(w), np.exp(-w)
    t = np.where(e < 0.5 * early, e / early, 0.5 + (e - 0.5 * early) / late)
    return w, t


def test_a_continuous_trend_fails_with_its_direction():
    rng = np.random.default_rng(3)
    w, t = _reversing(rng, 500)
    df = _frame(t, np.full(500, 4.0), w=w)
    d = _fit(sp.ProportionalOdds, df, ["w"]).diagnose()
    row = _row(d, "w")
    assert row["verdict"] == "fail"
    # the hazard ratio falls, so the survival-odds coefficient rises
    assert row["detail"] == "its coefficient rises with time"
    assert any("its coefficient trends over follow-up" in a for a in d.advice)


def test_a_categorical_covariate_is_tested_across_its_groups():
    rng = np.random.default_rng(11)
    stage = rng.integers(0, 3, 400)  # three groups
    t = _log_logistic(rng, 1.0 + stage)
    df = _frame(t, rng.uniform(0, 4, 400), stage=stage)
    d = _fit(sp.ProportionalOdds, df, ["stage"]).diagnose()
    row = _row(d, "stage")
    assert row["df"] == 2 and row["verdict"] == "fail"
    assert row["detail"].startswith("its groups' log-odds of failure")
    assert "the gaps between its groups change with time" in row["detail"]


def _early_effect(rng, n):
    """An additive effect of z, 1.0 on the hazard before t = 0.5 and 0
    after, on a baseline hazard of 0.5."""
    z = rng.binomial(1, 0.5, n).astype(float)
    e = rng.exponential(size=n)
    early = 0.5 + 1.0 * z
    t = np.where(e < 0.5 * early, e / early, 0.5 + (e - 0.5 * early) / 0.5)
    return z, t


def test_a_time_varying_additive_effect_is_advised_fit_tvc():
    rng = np.random.default_rng(2)
    z, t = _early_effect(rng, 400)
    w = rng.uniform(0, 2, 400)
    df = _frame(t, np.full(400, 3.0), z=z, w=w)
    d = _fit(sp.AdditiveHazards, df, ["z", "w"]).diagnose()
    row = _row(d, "z")
    assert row["verdict"] == "fail"
    assert row["detail"] == "its coefficient falls with time"
    (advice,) = [a for a in d.advice if a.covariates == ("z",)]
    assert "WeibullAH.fit_tvc" in advice and "ExponentialAH.fit_tvc" in advice
    assert advice.features[0] == "WeibullAH.fit_tvc"
    # The fix the advice names: z in its own column in each period
    cut = 0.5
    one = df["x"] <= cut
    i = np.r_[np.arange(len(df)), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(len(df)), np.full((~one).sum(), cut)]
    xr = np.r_[np.where(one, df["x"], cut), df["x"][~one]]
    c = np.r_[np.where(one, df["c"], 1), df["c"][~one]]
    zz = np.r_[df["z"], df["z"][~one]]
    period = np.r_[np.zeros(len(df)), np.ones((~one).sum())]
    Z = np.column_stack(
        [zz * (period == 0), zz * (period == 1), np.r_[df["w"], df["w"][~one]]]
    )
    tv = _quiet(sp.WeibullAH.fit_tvc, i, xl, xr, c, Z)
    early, late = tv.params[2], tv.params[3]
    assert early > 0.5 and abs(late) < 0.3


def test_a_parametric_additive_failure_names_its_fitter():
    rng = np.random.default_rng(2)
    z, t = _early_effect(rng, 400)
    df = _frame(t, np.full(400, 3.0), z=z)
    d = _fit(sp.ExponentialAH, df, ["z"]).diagnose()
    assert _row(d, "z")["verdict"] == "fail"
    (advice,) = [a for a in d.advice if a.covariates == ("z",)]
    assert "ExponentialAH.fit_tvc" in advice
    assert advice.features == (
        "ExponentialAH.fit_tvc",
        "ExponentialAH.fit_tvc_from_df",
    )


@pytest.mark.parametrize(
    "model_fixture, fitter, check",
    [
        ("po_data", sp.ProportionalOdds, "proportional_odds"),
        ("ah_data", sp.AdditiveHazards, "constant_effects"),
    ],
)
def test_a_joint_failure_alone_has_its_own_advice(
    request, monkeypatch, model_fixture, fitter, check
):
    df = request.getfixturevalue(model_fixture)
    model = _fit(fitter, df, ["z", "w"])

    def fake(ctx, key, base):
        p = 0.01 if key == "global" else 0.3
        return {"statistic": 1.0, "df": base.shape[1], "p": p, "theta": [1]}

    monkeypatch.setattr(oa, "_trend", fake)
    d = model.diagnose()
    (advice,) = d.advice
    assert "GLOBAL" in advice and "no single covariate" in advice
    assert advice.covariates == ("z", "w") and advice.check == check


def test_no_joint_advice_when_a_covariate_fails(crossing):
    d = _fit(sp.ProportionalOdds, crossing, ["z", "w"]).diagnose()
    assert (
        not d.table[d.table["test"].str.contains("GLOBAL")]["verdict"]
        .eq("ok")
        .all()
    )
    assert not any("GLOBAL" in a for a in d.advice)


def test_a_test_that_cannot_be_computed_warns():
    # Every event at one time: follow-up cannot be split into periods
    rng = np.random.default_rng(1)
    n = 30
    z = rng.normal(size=n)
    x = np.where(np.arange(n) < 12, 1.0, 2.0)
    c = (np.arange(n) >= 12).astype(int)
    d = sp.AdditiveHazards.fit(x, z[:, None], c=c).diagnose()
    assert d.table["p"].isna().all()
    assert (d.table["verdict"] == "warn").all() and d.ok
    assert d.table["detail"].iloc[0] == (
        "the test could not be computed: too few distinct event times "
        "to split follow-up into periods"
    )
    assert any("too few distinct event times" in n for n in d.notes)


def test_a_refit_that_fails_warns(monkeypatch, po_data):
    model = _fit(sp.ProportionalOdds, po_data, ["z", "w"])

    def fails(ctx, design):
        raise oa._NotComputed(
            "the model with the added terms has no " "finite maximum"
        )

    monkeypatch.setattr(oa, "_fit_split", fails)
    d = model.diagnose()
    assert (d.table["verdict"] == "warn").all() and d.ok
    assert d.table["detail"].iloc[0].endswith("has no finite maximum")


# -- what the fits support ---------------------------------------------------
@pytest.mark.parametrize(
    "fitter", [sp.ProportionalOdds, sp.AdditiveHazards, sp.WeibullPO]
)
def test_counts_equal_repetition(fitter):
    rng = np.random.default_rng(3)
    m = 60
    z, w = rng.normal(size=m), rng.integers(0, 2, m)
    x = np.ceil(rng.exponential(1 / np.exp(0.5 * z + 0.3 * w)) * 5) / 5
    c = (rng.uniform(size=m) < 0.2).astype(int)
    n = rng.integers(1, 4, m)
    Z = np.column_stack([z, w])
    counted = _quiet(fitter.fit, x, Z, c=c, n=n).diagnose()
    written = _quiet(
        fitter.fit, np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    ).diagnose()
    columns = ["check", "test", "covariates", "df", "verdict"]
    pd.testing.assert_frame_equal(
        counted.table[columns], written.table[columns]
    )
    assert_allclose(
        counted.table["statistic"], written.table["statistic"], rtol=1e-5
    )


@pytest.mark.parametrize(
    "fitter", [sp.ProportionalOdds, sp.AdditiveHazards, sp.WeibullPO]
)
def test_the_unit_of_time_does_not_matter(po_data, fitter):
    Z = po_data[["z", "w"]].values
    hours = _quiet(fitter.fit, po_data["x"], Z, c=po_data["c"]).diagnose()
    minutes = _quiet(
        fitter.fit, 60 * po_data["x"], Z, c=po_data["c"]
    ).diagnose()
    assert_allclose(
        hours.table["statistic"], minutes.table["statistic"], rtol=1e-4
    )


def test_delayed_entry_is_supported():
    rng = np.random.default_rng(8)
    n = 200
    z = rng.normal(size=n)
    tl = np.where(rng.uniform(size=n) < 0.3, rng.uniform(0, 0.5, n), 0.0)
    t = tl + _po_times(rng, 0.5 * z)
    df = _frame(t, np.full(n, 3.0), z=z)
    df["tl"] = tl
    for fitter in (sp.ProportionalOdds, sp.WeibullPO):
        d = _fit(fitter, df, ["z"], tl_col="tl").diagnose()
        assert np.isfinite(d.table["statistic"]).all()


def test_time_varying_covariates_fixed_parameters_and_centring():
    rossi = sp.datasets.load_rossi_static()
    x, c = rossi["week"].values, 1 - rossi["arrest"].values
    Z = rossi[["fin", "age"]].values.astype(float)
    fixed = _quiet(sp.WeibullPO.fit, x, Z, c=c, fixed={"beta": 1.5})
    d = fixed.diagnose()
    assert np.isfinite(d.table["p"]).all()
    refit = oa._fit_split(d._context, oa._covariates_split(d._context))
    assert refit.params[1] == 1.5
    # A fixed coefficient stays fixed in the refits, under its new name
    held = _quiet(sp.WeibullPO.fit, x, Z, c=c, fixed={"coef_0": 0.3})
    d = held.diagnose()
    refit = oa._fit_split(d._context, oa._covariates_split(d._context))
    assert refit.params[2] == 0.3 and np.isfinite(d.table["p"]).all()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close(d.plot())
    centred = _quiet(sp.WeibullPO.fit, x, Z, c=c, center=True).diagnose()
    assert np.isfinite(centred.table["p"]).all()
    # A time-varying-covariate fit: each interval row split again
    i = np.r_[np.arange(x.size), np.arange(x.size)]
    half = x / 2
    tv = _quiet(
        sp.WeibullPO.fit_tvc,
        i,
        np.r_[np.zeros(x.size), half],
        np.r_[half, x],
        np.r_[np.ones(x.size), c],
        np.r_[Z, Z],
    )
    same = sp.WeibullPO.fit(x, Z, c=c).diagnose()
    assert_allclose(
        tv.diagnose().table["statistic"], same.table["statistic"], rtol=1e-3
    )


def test_covariate_names_come_from_the_fit(po_data):
    d = _fit(sp.WeibullPO, po_data, ["z", "w"]).diagnose()
    assert d.columns == ["z", "w"]
    Z = po_data[["z", "w"]].values
    d = _quiet(sp.WeibullPO.fit, po_data["x"], Z, c=po_data["c"]).diagnose()
    assert d.columns == ["coef_0", "coef_1"]
    d = _quiet(
        sp.AdditiveHazards.fit, po_data["x"], Z, c=po_data["c"]
    ).diagnose()
    assert d.columns == ["coef_0", "coef_1"]


def test_columns_and_alpha(crossing):
    model = _fit(sp.ProportionalOdds, crossing, ["z", "w"])
    d = model.diagnose(columns=["w"])
    assert set(d.table["covariates"]) == {("w",), ("z", "w")}
    full = model.diagnose()
    assert _row(d, "w")["statistic"] == _row(full, "w")["statistic"]
    p = _row(full, "z")["p"]
    assert model.diagnose(alpha=p / 3).table.iloc[0]["verdict"] == "ok"
    assert model.diagnose(alpha=p * 0.75).table.iloc[0]["verdict"] == "warn"


def test_lin_ying_test_has_its_stated_size():
    # Principle 17: under the model the 5% test rejects about 5% of the
    # time (Lin and Ying's fit is closed form, so this is quick).
    rng = np.random.default_rng(1)
    rejected = []
    for _ in range(300):
        n = 150
        z, w = rng.binomial(1, 0.5, n), rng.uniform(0, 2, n)
        t = rng.exponential(1 / (0.5 + 0.3 * z + 0.2 * w))
        cens = rng.uniform(0, 3, n)
        model = sp.AdditiveHazards.fit(
            np.minimum(t, cens), np.column_stack([z, w]), c=(t > cens) * 1
        )
        rejected.append(model.diagnose().table["p"].values < 0.05)
    rate = np.mean(rejected, axis=0)
    assert np.all((rate > 0.02) & (rate < 0.09)), rate


def test_the_report_round_trips(crossing):
    d = _fit(sp.ProportionalOdds, crossing, ["z", "w"]).diagnose()
    out = d.to_dict()
    back = Diagnosis.from_dict(json.loads(json.dumps(out, allow_nan=False)))
    assert back.to_dict() == out
    pd.testing.assert_frame_equal(back.table, d.table)
    assert back.advice == d.advice
    assert [a.features for a in back.advice] == [a.features for a in d.advice]


# -- refusals -----------------------------------------------------------------
@pytest.mark.parametrize(
    "fitter",
    [sp.ProportionalOdds, sp.AdditiveHazards, sp.WeibullPO, sp.WeibullAH],
)
def test_a_restored_model_refuses(po_data, fitter):
    model = _fit(fitter, po_data, ["z", "w"])
    restored = sp.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="from_dict"):
        restored.diagnose()


@pytest.mark.parametrize(
    "fitter", [sp.ProportionalOdds, sp.AdditiveHazards, sp.WeibullPO]
)
def test_an_aliased_fit_refuses(fitter):
    rng = np.random.default_rng(1)
    z = rng.normal(size=60)
    model = _quiet(
        fitter.fit, rng.exponential(size=60), np.column_stack([z, 2 * z])
    )
    with pytest.raises(ValueError, match="aliased.*Refit without"):
        model.diagnose()


def test_a_fit_with_no_finite_maximum_refuses():
    x = np.arange(1.0, 11.0)
    z = (x <= 5).astype(float)  # every early failure has z = 1
    model = _quiet(sp.ProportionalOdds.fit, x, z[:, None])
    assert model.maximum == "no finite maximum"
    with pytest.raises(ValueError, match="finite maximum"):
        model.diagnose()


def test_censoring_intervals_and_right_truncation_refuse():
    rng = np.random.default_rng(4)
    n = 80
    z = rng.binomial(1, 0.5, n)
    t = _po_times(rng, 0.5 * z)
    xl = np.column_stack([np.floor(t * 4) / 4, np.floor(t * 4) / 4 + 0.25])
    model = _quiet(sp.WeibullPO.fit, xl, z, c=np.full(n, 2))
    with pytest.raises(NotImplementedError, match="interval.*#870"):
        model.diagnose()
    model = _quiet(sp.WeibullPO.fit, t, z, tr=np.full(n, 10.0))
    with pytest.raises(NotImplementedError, match="right-truncated.*#870"):
        model.diagnose()


def test_real_fits_are_no_longer_pending():
    families = [f.name for f in registry.families()]
    assert "Proportional odds" in families and "Additive hazards" in families
    assert not any(p.issue == 870 for p in registry.PENDING)


# -- plots -------------------------------------------------------------------
def test_po_plot_draws_log_odds_and_period_coefficients(crossing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _fit(sp.WeibullPO, crossing, ["z", "w"]).diagnose()
    fig = d.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0].startswith("z: log-odds by group, p = ")
    assert titles[1].startswith("w: coefficient by period, p = ")
    assert len(fig.axes[0].get_lines()) == 2  # a curve per group
    assert fig.axes[0].get_xscale() == "log"
    plt.close(fig)
    fig = d.plot("proportional_odds")
    assert len(fig.axes) == 2
    plt.close(fig)


@pytest.mark.parametrize("fitter", [sp.AdditiveHazards, sp.WeibullAH])
def test_ah_plot_draws_cumulative_coefficients(ah_data, fitter):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _fit(fitter, ah_data, ["z", "w"]).diagnose()
    fig = d.plot(n_boot=0)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        t
        for t in titles
        if t.startswith(("z: cumulative effect", "w: cumulative effect"))
    ]
    assert len(titles) == 2
    plt.close(fig)
