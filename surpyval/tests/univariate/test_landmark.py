"""Landmark data, the landmark Cox models and their dynamic predictions.

The comparison with R (``dynpred``'s landmark data, ``coxph(... +
cluster(id))``'s supermodels and ``survfit``'s predictions) is in
``tests/reference/test_landmark.py``; the coverage of the robust
intervals is in ``tests/calibration/test_coverage_landmark.py``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from surpyval import (
    CoxPH,
    ItemLifetime,
    KaplanMeier,
    LandmarkCox,
    Measurements,
    StepSchedule,
    SurpyvalData,
    landmark,
    landmark_kaplan_meier,
)


def _switching(n=150, seed=3):
    """Units with a fixed z1 and a covariate that switches 0 -> 1."""
    rng = np.random.default_rng(seed)
    items = []
    for u in range(n):
        z1 = float(rng.uniform() < 0.5)
        # a fifth of the units start switched
        v = 0.0 if rng.uniform() < 0.2 else round(rng.exponential(3.0), 2)
        v += 0.01 if v else 0.0
        rate = 0.2 * np.exp(0.5 * z1)
        t = rng.exponential(1 / rate)
        if t > v:
            t = v + rng.exponential(1 / (rate * np.exp(0.8)))
        cens = rng.uniform(4, 10)
        time = round(float(min(t, cens)), 2) + 0.005
        outcome = {"observed": time} if t <= cens else {"right_censored": time}
        items.append(
            ItemLifetime(
                u,
                **outcome,
                covariates={"z1": z1},
                time_varying_covariates=(
                    StepSchedule.from_changepoints([0, v], [0, 1])
                    if v
                    else StepSchedule.from_changepoints([0], [1])
                ),
            )
        )
    return SurpyvalData.from_items(items)


LIFE = _switching()


# -- the landmark data --------------------------------------------------------


def test_landmark_keeps_units_at_risk_and_cuts_at_the_horizon():
    life = SurpyvalData(
        x=[1.5, 6, 4, 9],
        c=[0, 0, 1, 0],
        Z=[[1], [0], [1], [0]],
        i=["a", "b", "c", "d"],
        covariate_names=["z"],
    )
    data = landmark(life, at=2, horizon=3)
    assert data.i == ["b", "c", "d"]
    assert_allclose(data.x, [5, 4, 5])
    # b fails at 6, after the horizon 5: censored there
    assert_allclose(data.c, [1, 1, 1])
    assert_allclose(data.tl, 2)
    assert data.covariate_names == ["z"]
    assert data.attributes[0] == {"i": "b", "L": 2.0}


def test_landmark_change_at_the_landmark_is_known():
    schedule = StepSchedule.from_changepoints([0, 2], [0, 1])
    life = SurpyvalData.from_items(
        [ItemLifetime("a", observed=5, time_varying_covariates=schedule)]
    )
    assert landmark(life, at=2, horizon=3).Z[0, 0] == 1
    assert landmark(life, at=1.99, horizon=3).Z[0, 0] == 0


def test_landmark_late_entry_is_not_at_risk_before_it():
    life = SurpyvalData(x=[5, 6], c=[0, 0], tl=[0, 3], Z=[[0], [1]])
    assert len(landmark(life, at=2, horizon=3).x) == 1
    assert len(landmark(life, at=3, horizon=3).x) == 2


def test_stacked_adds_the_landmark_and_interactions():
    life = SurpyvalData(x=[1.5, 6, 4], c=[0, 0, 1], Z=[[1], [0], [1]])
    data = landmark(life, at=[0, 2], horizon=3, interactions=True)
    assert data.covariate_names == ["Z0", "L", "Z0*L"]
    assert data.i is None
    assert_allclose(data.Z[:, 1], [0, 0, 0, 2, 2])
    assert_allclose(data.Z[:, 2], data.Z[:, 0] * data.Z[:, 1])
    assert [a["i"] for a in data.attributes] == [0, 1, 2, 1, 2]


def test_stacked_data_fits_an_ordinary_cox_model():
    # the supermodel by hand: CoxPH on the stacked data, clustered on
    # the unit, is LandmarkCox's with the matching basis
    data = landmark(LIFE, at=[0, 1, 2], horizon=3)
    units = [a["i"] for a in data.attributes]
    lm = [a["L"] for a in data.attributes]
    cox = CoxPH.fit(
        data.x, data.Z[:, :2], data.c, data.n, tl=data.tl, strata=lm
    )
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3, degree=0)
    assert_allclose(model.params, cox.params, rtol=1e-10)
    robust = np.sqrt(np.diag(cox.robust_covariance(cluster=units)))
    assert_allclose(model.standard_errors(), robust, rtol=1e-10)


def test_measurements_last_reading_at_or_before_the_landmark():
    m = Measurements(
        {"wear": {"a": {0: 1.0, 2: 3.0}, "b": {1: 2.0, 3: 9.0}, "c": {4: 5}}}
    )
    life = SurpyvalData(x=[5, 6, 7], c=[0, 0, 1], i=["a", "b", "c"], m=m)
    with pytest.warns(UserWarning, match="1 at L = 2"):
        data = landmark(life, at=2, horizon=3)
    assert data.i == ["a", "b"]
    assert data.covariate_names == ["wear"]
    assert_allclose(data.Z[:, 0], [3.0, 2.0])


def test_one_unnamed_measurement_is_named_m():
    life = SurpyvalData(
        x=[5, 6], c=[0, 1], i=["a", "b"], m={"a": {0: 1.0}, "b": {1: 4.0}}
    )
    data = landmark(life, at=2, horizon=3)
    assert data.covariate_names == ["m"]
    assert_allclose(data.Z[:, 0], [1.0, 4.0])


def test_stretches_are_time_varying_covariates():
    # A unit's rows, its covariates fixed in each, are the same data as a
    # step schedule
    rows = SurpyvalData(
        x=[2, 5, 3, 1, 6],
        c=[1, 0, 1, 1, 1],
        Z=[[0], [1], [0], [0], [1]],
        i=["a", "a", "b", "c", "c"],
    )
    steps = SurpyvalData.from_items(
        [
            ItemLifetime(
                "a",
                observed=5,
                time_varying_covariates=StepSchedule.from_changepoints(
                    [0, 2], [0, 1]
                ),
            ),
            ItemLifetime(
                "b",
                right_censored=3,
                time_varying_covariates=StepSchedule.from_changepoints(
                    [0], [0]
                ),
            ),
            ItemLifetime(
                "c",
                right_censored=6,
                time_varying_covariates=StepSchedule.from_changepoints(
                    [0, 1], [0, 1]
                ),
            ),
        ]
    )
    for L in (0.5, 1, 2, 2.5):
        a = landmark(rows, at=L, horizon=3)
        b = landmark(steps, at=L, horizon=3)
        assert a.i == b.i
        assert_allclose(a.Z, b.Z)
        assert_allclose(a.x, b.x)
        assert_allclose(a.c, b.c)
    # a change at the landmark is known there
    assert_allclose(landmark(rows, at=2, horizon=3).Z[:, 0], [1, 0, 1])


def test_repeated_events_and_gaps_are_refused():
    events = SurpyvalData(x=[2, 5], c=[0, 0], a=[0, 1], i=["a", "a"])
    with pytest.raises(ValueError, match="recurrent"):
        landmark(events, at=1, horizon=3)
    gap = SurpyvalData(x=[2, 5], c=[1, 0], tl=[0, 3], i=["a", "a"])
    with pytest.raises(ValueError, match="without a break"):
        landmark(gap, at=1, horizon=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(at=[], horizon=1), "'at'"),
        (dict(at=[1, 1], horizon=1), "repeated"),
        (dict(at=1, horizon=0), "'horizon'"),
        (dict(at=1, horizon="a"), "'horizon'"),
        (dict(at=100, horizon=1), "No unit is at risk"),
    ],
)
def test_landmark_refusals(kwargs, match):
    life = SurpyvalData(x=[1, 2, 3], c=[0, 0, 1], Z=[[0], [1], [0]])
    with pytest.raises(ValueError, match=match):
        landmark(life, **kwargs)


def test_landmark_refuses_what_it_cannot_take():
    with pytest.raises(TypeError, match="SurpyvalData"):
        landmark([1, 2], at=1, horizon=1)
    with pytest.raises(ValueError, match="left or interval"):
        landmark(SurpyvalData(x=[1, 2], c=[-1, 0]), at=1, horizon=1)
    with pytest.raises(ValueError, match="right truncation"):
        landmark(SurpyvalData(x=[1, 2], tr=[5, 5]), at=1, horizon=1)
    with pytest.raises(ValueError, match="event types"):
        landmark(
            SurpyvalData(x=[1, 2], c=[0, 0], e=["a", "b"]), at=1, horizon=1
        )


# -- the models ---------------------------------------------------------------


def test_single_landmark_is_the_cox_model_of_its_landmark_data():
    model = LandmarkCox.fit(LIFE, at=2, horizon=3)
    assert model.degree == 0
    data = landmark(LIFE, at=2, horizon=3)
    cox = CoxPH.fit(data.x, data.Z, data.c, data.n, tl=data.tl)
    assert_allclose(model.params, cox.params, rtol=1e-10)
    Z = [[1.0, 0.0], [0.0, 1.0]]
    assert_allclose(model.predict(2, Z), cox.ff(5, Z), rtol=1e-10)
    # every unit once: the predicted curve's bounds are close to the
    # model-based ones (the robust variance is another estimate)
    ours = model.cb(5, Z, s=2, on="ff", bound_type="log")
    theirs = 1 - cox.cb(5, Z, bound_type="log")[:, ::-1]
    assert_allclose(ours, theirs, rtol=0.15)


@pytest.mark.parametrize("baseline", ["stratified", "pooled"])
def test_robust_prediction_se_is_the_jackknife_derivative(baseline):
    # The standard error is the infinitesimal jackknife: each unit's
    # influence is the derivative of log H with respect to its count,
    # here by finite differences (counts scaled by 10^5, one unit's
    # raised by 1), with Breslow's ties, for which it is exact.
    model = LandmarkCox.fit(
        LIFE, at=[0, 1, 2], horizon=3, baseline=baseline, tie_method="breslow"
    )
    st, X = model._stack, model._X[:, model._kept]
    s = 1.0 if baseline == "stratified" else 0.5
    H, Xq, tc, _, stratum, s = model._Hf([s + 3], [[1.0, 1.0]], s)
    se = model._log_H_se(Xq, tc, stratum, s)[0]
    strata = st["lm"] if baseline == "stratified" else None

    def log_H(n):
        cox = CoxPH.fit(
            x=st["x"],
            Z=X,
            c=st["c"],
            n=n,
            tl=st["L"],
            tie_method="breslow",
            strata=strata,
        )
        args = () if stratum is None else (stratum,)
        return np.log(cox.Hf([s + 3], Xq, *args) - cox.Hf([s], Xq, *args))[0]

    K = 100000
    n0 = (st["n"] * K).astype(int)
    base = log_H(n0)
    infl = []
    for u in range(len(model._lives.ids)):
        n = n0.copy()
        n[st["unit"] == u] += 1
        infl.append((log_H(n) - base) * K)
    assert_allclose(np.sqrt(np.sum(np.square(infl))), se, rtol=1e-4)


def test_supermodel_robust_differs_from_model_based():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2, 3], horizon=3)
    assert not np.allclose(
        model.standard_errors(), model.standard_errors(robust=False)
    )
    table = model.summary()
    assert list(table.index) == model.parameter_names
    assert_allclose(table["se(coef)"], model.standard_errors())


def test_predict_shapes_and_table():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3, degree=1)
    assert np.shape(model.predict(1, [1, 0])) == ()
    assert model.predict([0, 1], [1, 0]).shape == (2,)
    assert model.predict([0, 1, 2], [[1, 0], [0, 1]]).shape == (3, 2)
    profiles = pd.DataFrame(
        {"z1": [0, 1], "time_varying_0": [1, 1]}, index=["low", "high"]
    )
    table = model.table(profiles)
    assert list(table.columns) == [
        "landmark",
        "profile",
        "probability",
        "lower",
        "upper",
    ]
    assert len(table) == 6
    assert set(table["profile"]) == {"low", "high"}
    assert (table["lower"] <= table["probability"]).all()
    assert (table["probability"] <= table["upper"]).all()
    p = model.predict(2, profiles.to_numpy(dtype=float))
    assert_allclose(table["probability"].to_numpy()[-2:], p)
    # a shorter window is within the fitted one
    assert (
        model.predict(1, [1, 0], window=1) < model.predict(1, [1, 0])
    ).all()
    with pytest.raises(ValueError, match="at most the fitted window"):
        model.predict(1, [1, 0], window=4)


def test_cb_default_is_sf_and_ff_is_its_complement():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3)
    sf = model.cb(4.0, [1, 0], s=1)
    ff = model.cb(4.0, [1, 0], s=1, on="ff")
    assert_allclose(ff, 1 - sf[::-1])


def test_dynamic_curve_from_the_landmark():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3)
    x = np.array([0.5, 1.0, 2.0, 4.0])
    assert_allclose(model.ff(x[:2], [1, 0], s=1), 0.0)
    assert_allclose(model.sf(x, [1, 0], s=1) + model.ff(x, [1, 0], s=1), 1)
    with pytest.warns(UserWarning, match="past the horizon"):
        late = model.ff([3.0, 5.0], [1, 0], s=1)
    assert np.isfinite(late[0]) and np.isnan(late[1])


def test_stratified_predicts_only_at_its_landmarks():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3)
    with pytest.raises(ValueError, match="baseline='pooled'"):
        model.predict(1.5, [1, 0])
    pooled = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3, baseline="pooled")
    assert np.isfinite(pooled.predict(1.5, [1, 0]))
    with pytest.raises(ValueError, match="from s = 0 to 2"):
        pooled.predict(2.5, [1, 0])


def test_beta_and_its_bounds():
    model = LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3, degree=1)
    b = model.beta([0, 2])
    assert b.shape == (2, 2)
    assert_allclose(b[0], model.params[:2])
    assert_allclose(b[1], model.params[:2] + model.params[2:4])
    cb = model.beta_cb([0, 2])
    assert cb.shape == (2, 2, 2)
    se0 = model.standard_errors()[:2]
    assert_allclose(cb[0, :, 1] - b[0], 1.959963984540054 * se0, rtol=1e-9)
    lower = model.beta_cb(1, alpha_ci=0.1, bound="lower")
    assert_allclose(lower, model.beta_cb(1, alpha_ci=0.2)[..., 0])


def test_varying_and_basis():
    fixed = LandmarkCox.fit(
        LIFE, at=[0, 1, 2], horizon=3, degree=1, varying=["time_varying_0"]
    )
    assert fixed.parameter_names == [
        "z1",
        "time_varying_0",
        "time_varying_0*u",
    ]
    own = LandmarkCox.fit(
        LIFE,
        at=[0, 1, 2],
        horizon=3,
        basis=lambda s: s / 2.0,
        varying=["time_varying_0"],
    )
    # the same model in another basis: the same predictions
    assert_allclose(
        own.predict([0, 1, 2], [[1, 0], [0, 1]]),
        fixed.predict([0, 1, 2], [[1, 0], [0, 1]]),
        rtol=1e-8,
    )
    with pytest.raises(ValueError, match="basis of its own"):
        own.to_dict()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(degree=3), "degree at most 2"),
        (dict(degree=1.5), "whole number"),
        (dict(baseline="smooth"), "baseline"),
        (dict(tie_method="exact"), "tie_method"),
        (dict(varying=["w"]), "not among the covariates"),
        (dict(degree=1, basis=lambda s: s), "not both"),
    ],
)
def test_fit_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3, **kwargs)


def test_fit_needs_covariates():
    with pytest.raises(ValueError, match="landmark_kaplan_meier"):
        LandmarkCox.fit(SurpyvalData(x=[1, 2, 3]), at=0, horizon=2)


def test_to_dict_round_trip():
    model = LandmarkCox.fit(
        LIFE, at=[0, 1, 2], horizon=3, baseline="pooled", degree=1
    )
    text = json.dumps(model.to_dict(), allow_nan=False)
    back = LandmarkCox.from_dict(json.loads(text))
    assert_allclose(back.params, model.params)
    Z = [[1, 0], [0, 1]]
    assert_allclose(back.cb(4.0, Z, s=1.5), model.cb(4.0, Z, s=1.5))
    assert "Landmark Cox model (supermodel)" in repr(back)


def test_counts_are_independent_units():
    # a unit counted twice is two units, for the estimates and the
    # robust errors alike
    life = SurpyvalData(
        x=[2, 3, 5, 6, 7, 8, 4, 9],
        c=[0, 0, 0, 1, 0, 0, 0, 1],
        Z=[[0], [1], [0], [1], [1], [0], [1], [0]],
    )
    double = SurpyvalData(x=life.x, c=life.c, n=[2] * 8, Z=life.Z)
    twice = SurpyvalData(
        x=np.repeat(life.x, 2),
        c=np.repeat(life.c, 2),
        Z=np.repeat(life.Z, 2, 0),
    )
    a = LandmarkCox.fit(double, at=[0, 1, 2], horizon=5, degree=0)
    b = LandmarkCox.fit(twice, at=[0, 1, 2], horizon=5, degree=0)
    assert_allclose(a.params, b.params, rtol=1e-10)
    assert_allclose(a.standard_errors(), b.standard_errors(), rtol=1e-10)
    assert_allclose(a.cb(6, [1], s=1), b.cb(6, [1], s=1), rtol=1e-10)


# -- the landmark Kaplan-Meier ------------------------------------------------


def test_landmark_kaplan_meier_by_group():
    km = landmark_kaplan_meier(LIFE, at=2, horizon=3, by="time_varying_0")
    assert sorted(km) == [0.0, 1.0]
    data = landmark(LIFE, at=2, horizon=3)
    keep = data.Z[:, 1] == 1
    direct = KaplanMeier.fit(
        x=data.x[keep], c=data.c[keep], n=data.n[keep], tl=data.tl[keep]
    )
    t = [2.5, 3.5, 4.5]
    assert_allclose(km[1.0].sf(t), direct.sf(t))
    assert_allclose(km[1.0].cb(t), direct.cb(t))
    every = landmark_kaplan_meier(LIFE, at=2, horizon=3)
    assert list(every) == [None]
    with pytest.raises(ValueError, match="one landmark"):
        landmark_kaplan_meier(LIFE, at=[1, 2], horizon=3)
    with pytest.raises(ValueError, match="not among"):
        landmark_kaplan_meier(LIFE, at=2, horizon=3, by="w")


def test_no_warnings_on_a_plain_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LandmarkCox.fit(LIFE, at=[0, 1, 2], horizon=3)


def test_missing_covariate_is_refused():
    life = SurpyvalData(x=[1, 2, 3], c=[0, 0, 1], Z=[[0], [np.nan], [1]])
    with pytest.raises(ValueError, match="1 row"):
        landmark(life, at=0, horizon=2)


def test_weights_are_refused():
    with pytest.raises(ValueError, match="does not take weights"):
        LandmarkCox.fit(LIFE, at=1, horizon=3, w=[1.0])
    with pytest.raises(TypeError, match="unexpected"):
        LandmarkCox.fit(LIFE, at=1, horizon=3, windows=3)
    with pytest.raises(ValueError, match="weights"):
        landmark(SurpyvalData(x=[1, 2], w=[1.0, 2.0]), at=0, horizon=1)


def test_fit_from_df_is_fit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "t": rng.exponential(5, 60).round(2) + 0.01,
            "z": rng.integers(0, 2, 60),
            "u": rng.normal(size=60),
        }
    )
    a = LandmarkCox.fit_from_df(
        df, x_col="t", Z_cols=["z", "u"], at=[0, 1], horizon=4
    )
    b = LandmarkCox.fit(
        SurpyvalData(x=df["t"], Z=df[["z", "u"]].to_numpy(dtype=float)),
        at=[0, 1],
        horizon=4,
    )
    assert a.feature_names == ["z", "u"]
    assert_allclose(a.params, b.params)


def test_aliased_covariate_warns_once_naming_it():
    life = SurpyvalData(
        x=LIFE.x, c=LIFE.c, Z=np.c_[LIFE.Z, LIFE.Z[:, :1]], i=LIFE.i
    )
    with pytest.warns(UserWarning, match=r"^Covariate column\(s\) 1 of Z"):
        model = LandmarkCox.fit(life, at=[0, 1, 2], horizon=3, degree=1)
    assert model.aliased.tolist() == [1]
    assert np.isnan(model.params[1])
