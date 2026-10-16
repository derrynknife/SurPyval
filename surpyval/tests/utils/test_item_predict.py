"""A model fitted on item records predicts from an item (its fixed and
time-varying covariates lined up as fitted)."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    CovariatePath,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)

RNG = np.random.default_rng(2)
LIFE = RNG.weibull(2, 80) * 900
SITE = (np.arange(80) % 2).astype(float)
STRESS = StepSchedule.from_changepoints([0, 400], [[0.0], [1.0]])
ITEMS = [
    ItemLifetime(k, observed=t, covariates=[s], time_varying_covariates=STRESS)
    for k, (t, s) in enumerate(zip(LIFE, SITE))
]
TIMES = [200.0, 500.0, 800.0]


@pytest.mark.parametrize("fitter", [sp.WeibullPH, sp.CoxPH])
def test_predicting_from_an_item_lines_its_covariates_up_as_fitted(fitter):
    model = fitter.fit(SurpyvalData.from_items(ITEMS))
    by_hand = StepSchedule.from_changepoints(
        [0, 400], [[1.0, 0.0], [1.0, 1.0]]
    )
    item = ITEMS[1]  # site 1
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, item), model.sf_tvc(TIMES, by_hand)
    )


def test_an_item_with_fixed_covariates_only_is_a_constant_schedule():
    items = [
        ItemLifetime(k, observed=t, covariates=[s])
        for k, (t, s) in enumerate(zip(LIFE, SITE))
    ]
    model = sp.WeibullPH.fit(SurpyvalData.from_items(items))
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, items[1]), model.sf(np.array(TIMES), [1.0])
    )


def test_an_item_on_a_continuous_path_with_fixed_covariates():
    model = sp.WeibullPH.fit(SurpyvalData.from_items(ITEMS))
    ramp = CovariatePath.from_points([0, 600], [0.0, 1.0])
    item = ITEMS[1].clone(time_varying_covariates=ramp)
    by_hand = CovariatePath.from_points([0, 600], [[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        model.sf_tvc(TIMES, item), model.sf_tvc(TIMES, by_hand), rtol=1e-8
    )


def test_predicting_from_an_item_without_covariates_or_with_xl():
    model = sp.WeibullPH.fit(SurpyvalData.from_items(ITEMS))
    with pytest.raises(ValueError, match="no covariates to predict with"):
        model.sf_tvc(TIMES, ItemLifetime("Z", observed=5))
    with pytest.raises(ValueError, match="xl must not be given"):
        model.sf_tvc(TIMES, ITEMS[0], xl=[0.0])


# -- #940: records as Z, and predictions as a DataFrame --------------------

NAMED = [
    ItemLifetime(
        f"P{k}",
        **{"right_censored" if t > 900 else "observed": t},
        covariates={"site": s, "load": 0.5},
    )
    for k, (t, s) in enumerate(zip(LIFE, SITE))
]


@pytest.mark.parametrize("fitter", [sp.WeibullPH, sp.CoxPH, sp.WeibullAFT])
def test_issue_940_a_record_stands_as_Z(fitter):
    data = SurpyvalData.from_items(NAMED)
    model = fitter.fit(data)
    item = NAMED[1]
    np.testing.assert_allclose(
        model.sf(TIMES, item), model.sf(TIMES, item.covariates)
    )
    np.testing.assert_allclose(
        model.sf(TIMES, item), model.sf(TIMES, {"load": 0.5, "site": 1.0})
    )
    np.testing.assert_allclose(
        model.Hf(500.0, item), model.Hf(500.0, [1.0, 0.5])
    )
    # Several records, or a dataset: one row each, paired with the times
    np.testing.assert_allclose(
        model.sf([200.0, 500.0], NAMED[:2]),
        model.sf([200.0, 500.0], [[0.0, 0.5], [1.0, 0.5]]),
    )
    np.testing.assert_allclose(
        model.ff([200.0, 500.0], data.select(ids=["P0", "P1"])),
        model.ff([200.0, 500.0], [[0.0, 0.5], [1.0, 0.5]]),
    )
    # By name: a record naming the covariates in another order
    swapped = ItemLifetime(
        "S", observed=1, covariates={"load": 0.5, "site": 1.0}
    )
    np.testing.assert_allclose(model.sf(TIMES, swapped), model.sf(TIMES, item))
    with pytest.raises(ValueError, match="'Q' have no covariates"):
        model.sf(TIMES, ItemLifetime("Q", observed=1))


def test_issue_940_predict_gives_a_tidy_frame():
    data = SurpyvalData.from_items(NAMED)
    model = sp.WeibullPH.fit(data)
    frame = model.predict(data.select(ids=["P0", "P1"]), times=[200, 500])
    assert frame.index.tolist() == ["P0", "P0", "P1", "P1"]
    assert frame.columns.tolist() == ["time", "sf"]
    np.testing.assert_allclose(
        frame.loc["P1", "sf"].values, model.sf([200.0, 500.0], NAMED[1])
    )
    several = model.predict(NAMED[:2], times=[200], what=["sf", "hf", "mean"])
    assert several.columns.tolist() == ["time", "sf", "hf", "mean"]
    assert several.loc["P0", "mean"] == pytest.approx(model.mean(NAMED[0]))
    quantiles = model.predict(NAMED[0], p=[0.1, 0.5], what="qf")
    assert quantiles.columns.tolist() == ["p", "qf"]
    np.testing.assert_allclose(
        quantiles["qf"].values, model.qf([0.1, 0.5], NAMED[0])
    )
    # Without covariates: the same prediction for every item
    plain = sp.Weibull.fit(data.without("covariates"))
    both = plain.predict(data.select(ids=["P0", "P1"]), times=[200.0])
    assert both["sf"].tolist() == [plain.sf(200.0)] * 2
    km = sp.KaplanMeier.fit(data.without("covariates"))
    assert km.predict(NAMED[:1], times=[200.0])["sf"].iloc[0] == km.sf(200.0)


def test_issue_940_predict_given_each_items_age():
    data = SurpyvalData.from_items(NAMED)
    model = sp.WeibullPH.fit(data)
    running = [item for item in NAMED if item.outcome == "right_censored"][0]
    failed = [item for item in NAMED if item.outcome == "observed"][0]
    frame = model.predict(
        [running, failed],
        times=[500.0, 1200.0],
        what=["sf", "Hf"],
        given="age",
    )
    assert frame.columns.tolist() == ["time", "age", "sf", "Hf"]
    age = running.right_censored
    assert frame.loc[running.id, "age"].tolist() == [age, age]
    expected = model.sf(1200.0, running) / model.sf(age, running)
    assert frame.loc[running.id, "sf"].tolist()[1] == pytest.approx(expected)
    assert frame.loc[running.id, "sf"].tolist()[0] == 1.0  # before its age
    assert frame.loc[running.id, "Hf"].tolist()[1] == pytest.approx(
        model.Hf(1200.0, running) - model.Hf(age, running)
    )
    assert np.isnan(frame.loc[failed.id, "sf"]).all()
    # A history's current age is the end of its observation
    history = sp.ItemHistory(
        "M", [100], right_censored=300, covariates={"site": 1.0, "load": 0.5}
    )
    frame = model.predict(
        sp.EventHistoryData.from_items([history]), times=[400.0], given="age"
    )
    assert frame["age"].iloc[0] == 300.0
    assert frame["sf"].iloc[0] == pytest.approx(
        model.sf(400.0, history) / model.sf(300.0, history)
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"what": "cdf"}, "what must be among"),
        ({}, "times= is needed for sf"),
        ({"what": "qf"}, "p= is needed for qf"),
        ({"times": [1], "given": "now"}, "given takes None"),
        ({"what": "mean", "given": "age"}, "qf and mean are not given"),
    ],
)
def test_issue_940_predict_checks_its_arguments(kwargs, match):
    model = sp.Weibull.from_params([100, 2])
    with pytest.raises(ValueError, match=match):
        model.predict(NAMED[:1], **kwargs)
    with pytest.raises(TypeError, match="predict takes a SurpyvalData"):
        model.predict([1, 2], times=[1])
