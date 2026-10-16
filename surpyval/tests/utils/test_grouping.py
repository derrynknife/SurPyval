"""Grouping data (#938): attributes on the records, carried through the
datasets; groupby by attribute or covariate; select by name."""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

TRIAL = SurpyvalData.from_items(
    [
        ItemLifetime(
            k, observed=t, covariates={"age": a}, attributes={"arm": arm}
        )
        for k, (t, a, arm) in enumerate(
            zip(
                [5, 7, 9, 4, 6, 8, 10, 3],
                [50, 60, 70, 55, 65, 45, 75, 40],
                ["drug", "placebo"] * 4,
            )
        )
    ]
)


def test_attributes_are_kept_and_read_back():
    item = ItemLifetime("A", observed=5, attributes={"site": "N", "batch": 7})
    assert item.attributes == {"site": "N", "batch": 7}
    assert item.attribute("site") == "N" and item.attribute("colour") is None
    assert ItemLifetime("A", observed=5).attribute("site") is None
    assert item.clone("B").attributes == {"site": "N", "batch": 7}
    assert ItemLifetime.from_dict(item.to_dict()) == item
    series = pd.Series({"site": "N", "batch": np.int64(7)})
    assert ItemLifetime("A", observed=5, attributes=series) == item
    with pytest.raises(ValueError, match="attribute names must be text"):
        ItemLifetime("A", observed=5, attributes={1: "x"})
    with pytest.raises(ValueError, match="must be a value the data can group"):
        ItemLifetime("A", observed=5, attributes={"site": ["N"]})
    with pytest.raises(ValueError, match="must be a dict"):
        ItemLifetime("A", observed=5, attributes="N")


def test_a_fit_does_not_refuse_attributes():
    np.testing.assert_allclose(
        sp.Weibull.fit(TRIAL.without("covariates")).params,
        sp.Weibull.fit([5, 7, 9, 4, 6, 8, 10, 3]).params,
    )
    sp.WeibullPH.fit(TRIAL)  # covariates fitted, attributes carried


def test_groupby_an_attribute_and_a_covariate():
    arms = TRIAL.groupby("arm")
    assert list(arms) == ["drug", "placebo"]
    assert arms["drug"].ids == [0, 2, 4, 6]
    assert arms["drug"].attributes == [{"arm": "drug"}] * 4
    # One line per group, as the docs promise
    km = {
        arm: sp.KaplanMeier.fit(part.without("covariates"))
        for arm, part in arms.items()
    }
    assert km["drug"].sf(6) == pytest.approx(0.5)
    by_age = TRIAL.groupby("age")
    assert list(by_age) == [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    with pytest.raises(ValueError, match="No attribute or covariate 'site'"):
        TRIAL.groupby("site")
    mixed = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=5, attributes={"site": "N"}),
            ItemLifetime("B", observed=7),
            ItemLifetime("C", observed=8, attributes={"site": None}),
        ]
    )
    with pytest.raises(ValueError, match="Items 'B', 'C' have no attribute"):
        mixed.groupby("site")
    assert mixed.select(where={"site": "N"}).ids == ["A"]


def test_select_by_name():
    assert TRIAL.select(where={"arm": "placebo"}).ids == [1, 3, 5, 7]
    assert TRIAL.select(where={"arm": "placebo", "age": 40}).ids == [7]
    assert TRIAL.select(ids=[1, 2, 3], where={"arm": "placebo"}).ids == [1, 3]
    with pytest.raises(ValueError, match="No items are left"):
        TRIAL.select(where={"arm": "x"})


def test_attributes_travel_through_reshapes_slices_json_and_frames():
    data = TRIAL.censor_at(6).without("covariates").enter_at(1)
    assert data.attributes == TRIAL.attributes
    assert data.item(3).attributes == {"arm": "placebo"}
    assert TRIAL[2:4].attributes == [{"arm": "drug"}, {"arm": "placebo"}]
    assert (
        SurpyvalData.from_json(TRIAL.to_json()).attributes == TRIAL.attributes
    )
    frame = TRIAL.to_dataframe()
    assert frame["arm"].tolist()[:2] == ["drug", "placebo"]
    again = SurpyvalData.from_dataframe(
        frame,
        i_col="id",
        x_col="x",
        c_col="c",
        Z_cols=["age"],
        attribute_cols=["arm"],
    )
    assert again.to_items() == TRIAL.to_items()
    assert TRIAL.without("attributes").attributes is None
    assert TRIAL.attribute_names == ["arm"]
    assert "attributes: arm" in TRIAL.summary()
    assert "'drug'" in repr(TRIAL)


def test_histories_group_and_select_too():
    data = EventHistoryData.from_items(
        [
            ItemHistory(
                "M2", [5], right_censored=9, attributes={"plant": "S"}
            ),
            ItemHistory("M1", right_censored=4, attributes={"plant": "N"}),
            ItemHistory(
                "M3", [2, 3], right_censored=4, attributes={"plant": "N"}
            ),
        ]
    )
    plants = data.groupby("plant")
    assert plants["N"].ids == ["M1", "M3"] and plants["S"].ids == ["M2"]
    assert plants["N"].attributes == [{"plant": "N"}] * 2
    assert data.select(where={"plant": "S"}).to_items() == [data.item("M2")]
    assert data.item("M3").attribute("plant") == "N"
    assert (
        EventHistoryData.from_json(data.to_json()).attributes
        == data.attributes
    )
    assert data.to_dataframe()["plant"].tolist() == [
        "N",
        "S",
        "S",
        "N",
        "N",
        "N",
    ]
    assert data.censor_at(3.5).attributes == data.attributes
    log = pd.DataFrame(
        {"m": ["A", "A", "B"], "t": [1, 2, 3], "plant": ["N", "N", "S"]}
    )
    from_log = EventHistoryData.from_dataframe(
        log, i_col="m", x_col="t", attribute_cols=["plant"]
    )
    assert from_log.attributes == [{"plant": "N"}, {"plant": "S"}]
    with pytest.raises(ValueError, match="the attributes differs between"):
        EventHistoryData.from_dataframe(
            log.assign(plant=["N", "S", "S"]),
            i_col="m",
            x_col="t",
            attribute_cols=["plant"],
        )
