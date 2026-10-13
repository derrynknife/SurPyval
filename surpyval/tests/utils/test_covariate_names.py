"""Covariates by name: on the records, the datasets, their reshaping and
JSON, a fit's coefficients, and prediction from an item."""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)


def _lives(k=40, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for j in range(k):
        age, arm = float(rng.uniform(40, 70)), float(j % 2)
        t = float(rng.weibull(2) * 100 * np.exp(-0.5 * arm))
        # The names come in either order: the data takes the first item's
        cov = (
            {"arm": arm, "age": age}
            if j % 3 == 0
            else {"age": age, "arm": arm}
        )
        if t < 120:
            out.append(ItemLifetime(j, observed=t, covariates=cov))
        else:
            out.append(ItemLifetime(j, right_censored=120, covariates=cov))
    return out


LIVES = _lives()
DATA = SurpyvalData.from_items(LIVES)


# Records


def test_a_record_names_its_covariates():
    item = ItemLifetime("A", observed=5, covariates={"age": 54, "arm": 1})
    assert item.covariate_names == ("age", "arm")
    assert item.covariates.tolist() == [54.0, 1.0]
    assert item == ItemLifetime(
        "A", observed=5, covariates={"arm": 1, "age": 54}
    )
    assert "covariates={'age': 54.0, 'arm': 1.0}" in repr(item)
    assert item.clone("B").covariate_names == ("age", "arm")
    series = ItemLifetime("A", observed=5, covariates=pd.Series({"age": 54}))
    assert series.covariate_names == ("age",)
    history = ItemHistory("H", [3], right_censored=9, covariates={"load": 2})
    assert history.covariate_names == ("load",)


def test_unnamed_covariates_have_no_names():
    assert (
        ItemLifetime("A", observed=5, covariates=[1, 2]).covariate_names
        is None
    )


@pytest.mark.parametrize(
    "bad, match",
    [
        ({}, "empty dict"),
        ({1: 2.0}, "must be text"),
        ({"a": [1, 2]}, "one number"),
        ({"a": "x"}, "must be numbers"),
    ],
)
def test_bad_named_covariates_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        ItemLifetime("A", observed=5, covariates=bad)


# Datasets


def test_from_items_keeps_the_names_in_the_first_items_order():
    assert DATA.covariate_names == ["arm", "age"]
    k = 1  # given as {"age": ..., "arm": ...}
    assert DATA.item(k) == LIVES[k]
    assert DATA.Z[k].tolist() == [
        LIVES[k].covariates[1],
        LIVES[k].covariates[0],
    ]
    assert SurpyvalData.from_items(
        DATA.to_items(), counts=DATA.n
    ).covariate_names == [
        "arm",
        "age",
    ]


def test_names_for_some_items_or_different_names_are_refused():
    with pytest.raises(ValueError, match="named for some items but not"):
        SurpyvalData.from_items(
            [
                ItemLifetime(1, observed=2, covariates={"a": 1}),
                ItemLifetime(2, observed=3, covariates=[1]),
            ]
        )
    with pytest.raises(ValueError, match="needs the same covariates"):
        SurpyvalData.from_items(
            [
                ItemLifetime(1, observed=2, covariates={"a": 1}),
                ItemLifetime(2, observed=3, covariates={"b": 1}),
            ]
        )


def test_names_given_with_arrays_and_dataframes():
    data = SurpyvalData(
        [1, 2, 3],
        c=[0, 0, 1],
        Z=[[1, 2], [3, 4], [5, 6]],
        covariate_names=["load", "temp"],
    )
    assert data.covariate_names == ["load", "temp"]
    frame = SurpyvalData([1, 2], Z=pd.DataFrame({"load": [1.0, 2.0]}))
    assert frame.covariate_names == ["load"]
    plain = SurpyvalData([1, 2], Z=[1.0, 2.0])
    assert plain.covariate_names is None
    from_obs = SurpyvalData.from_observations(
        [1, 2],
        [5],
        observed_covariates=[[1, 2], [3, 4]],
        right_censored_covariates=[[0, 1]],
        covariate_names=["a", "b"],
    )
    assert from_obs.item(2).covariates.tolist() == [0.0, 1.0]
    assert from_obs.item(2).covariate_names == ("a", "b")


@pytest.mark.parametrize(
    "names, Z, match",
    [
        (["a"], [[1, 2]], "1 covariate_names for 2"),
        (["a", "a"], [[1, 2]], "more than once"),
        (["a", 3], [[1, 2]], "must be text"),
        (["a"], None, "without covariates"),
    ],
)
def test_bad_covariate_names_are_refused(names, Z, match):
    with pytest.raises(ValueError, match=match):
        SurpyvalData([1], Z=Z, covariate_names=names)


def test_names_survive_slicing_json_combining_and_summary():
    assert DATA[:3].covariate_names == ["arm", "age"]
    back = SurpyvalData.from_json(DATA.to_json())
    assert back.covariate_names == ["arm", "age"] and back.item(
        0
    ) == DATA.item(0)
    assert SurpyvalData.combine(DATA[:3], DATA[3:6]).covariate_names == [
        "arm",
        "age",
    ]
    assert "covariates: 2 columns (arm, age)" in DATA.summary()
    other = SurpyvalData([1.0], Z=[[1.0, 2.0]], covariate_names=["x", "y"])
    plain = SurpyvalData([2.0], Z=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="different names"):
        SurpyvalData.combine(other, plain)


def test_with_covariates_by_name():
    arm = DATA.with_covariates(["arm"])
    assert arm.covariate_names == ["arm"]
    assert arm.item(1).covariates.tolist() == [LIVES[1].covariates[1]]
    swapped = DATA.with_covariates(["age", "arm"])
    assert swapped.covariate_names == ["age", "arm"]
    assert DATA.with_covariates("age").covariate_names == ["age"]
    assert DATA.without("covariates").covariate_names is None
    with pytest.raises(ValueError, match=r"no covariates \['weight'\]"):
        DATA.with_covariates(["weight"])


def test_histories_name_their_covariates():
    data = EventHistoryData.from_histories(
        observed={"M1": [1, 2], "M2": [3]},
        right_censored={"M1": 5, "M2": 6},
        covariates={
            "M1": {"load": 1, "temp": 2},
            "M2": {"temp": 3, "load": 0},
        },
    )
    assert data.covariate_names == ["load", "temp"]
    assert data.item("M2").covariates.tolist() == [0.0, 3.0]
    assert data.to_histories()["covariates"]["M2"] == {
        "load": 0.0,
        "temp": 3.0,
    }
    assert data[:2].covariate_names == ["load", "temp"]
    assert "covariates: 2 columns (load, temp)" in data.summary()


# Fitting and prediction


def test_fits_name_their_coefficients():
    cox = sp.CoxPH.fit(DATA)
    assert cox.feature_names == ["arm", "age"]
    assert sp.WeibullPH.fit(DATA).parameter_names == [
        "alpha",
        "beta",
        "arm",
        "age",
    ]
    # The same fit as from the arrays, the names aside
    plain = sp.CoxPH.fit(DATA.x, DATA.Z, c=DATA.c, n=DATA.n)
    assert np.allclose(cox.beta, plain.beta)
    # A model fitted with names reads a DataFrame by them
    by_name = cox.sf(50, Z=pd.DataFrame({"age": [50.0], "arm": [1.0]}))
    assert np.allclose(by_name, cox.sf(50, Z=[[1.0, 50.0]]))


def test_recurrent_fits_name_their_coefficients():
    rng = np.random.default_rng(1)
    histories = [
        ItemHistory(
            k,
            sorted(rng.uniform(0, 100, 3 + k % 3).tolist()),
            right_censored=100,
            covariates={"load": float(k % 3)},
        )
        for k in range(30)
    ]
    model = sp.ProportionalIntensityHPP.fit(
        EventHistoryData.from_items(histories)
    )
    assert model.parameter_names == ["lambda", "load"]


STRESS = StepSchedule.from_changepoints([0, 30], [[0.0], [1.0]])


def test_time_varying_fits_name_the_fixed_columns():
    data = SurpyvalData.from_items(
        [item.clone(time_varying_covariates=STRESS) for item in LIVES]
    )
    model = sp.CoxPH.fit(data)
    assert model.feature_names == ["arm", "age", "time_varying_0"]
    one = ItemLifetime(
        "new",
        right_censored=10,
        covariates={"age": 50.0, "arm": 1.0},
        time_varying_covariates=STRESS,
    )
    other_order = ItemLifetime(
        "new",
        right_censored=10,
        covariates={"arm": 1.0, "age": 50.0},
        time_varying_covariates=STRESS,
    )
    assert np.allclose(
        model.sf_tvc([20, 60], one), model.sf_tvc([20, 60], other_order)
    )
    wrong = one.clone(covariates={"age": 50.0, "dose": 1.0})
    with pytest.raises(ValueError, match="fitted on"):
        model.sf_tvc([20], wrong)
