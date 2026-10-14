"""Covariates (#929): one column once, covariates not recorded follow
the missing-value rule, a record's covariate by name, and the round trips
of a dataset bring back everything it holds."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

NAMED = SurpyvalData.from_items(
    [ItemLifetime("a", observed=5, covariates={"p": 1.0, "q": 2.0})]
)
UNNAMED = SurpyvalData.from_items(
    [ItemLifetime("a", observed=5, covariates=[1.0, 2.0])]
)


@pytest.mark.parametrize(
    "data, columns",
    [
        (NAMED, ["p", "p"]),
        (UNNAMED, [0, 0]),
        (UNNAMED, [0, -2]),
        (NAMED, ["p", 0]),
    ],
)
def test_a_covariate_picked_twice_is_refused(data, columns):
    with pytest.raises(ValueError, match="picked more than once"):
        data.with_covariates(columns)


def test_covariates_in_another_order_are_still_fine():
    assert NAMED.with_covariates(["q", "p"]).covariate_names == ["q", "p"]


# Covariates not recorded


def test_records_take_a_covariate_not_recorded():
    item = ItemLifetime("x", observed=2, covariates={"age": np.nan, "arm": 1})
    assert np.isnan(item.covariate("age")) and item.covariate("arm") == 1.0
    assert item == item.clone()
    assert item == ItemLifetime(
        "x", observed=2, covariates={"age": None, "arm": 1}
    )
    assert hash(item) == hash(item.clone())
    with pytest.raises(ValueError, match="NaN, or None, for one not recorded"):
        ItemLifetime("x", observed=2, covariates=[np.inf])


def test_a_dataset_with_a_missing_covariate_reshapes_and_fits():
    data = SurpyvalData(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Z=[[1.0], [np.nan], [2.0], [1.0], [3.0], [2.0]],
    )
    assert data.without("covariates").Z is None
    assert np.isnan(data.item(1).covariates[0])
    assert len(data.censor_at(4)) == 6
    with pytest.warns(UserWarning, match="Dropped 1 of 6 rows"):
        sp.WeibullPH.fit(data)


def test_histories_keep_a_missing_covariate_and_recurrent_fits_refuse_it():
    data = EventHistoryData.from_items(
        [
            ItemHistory(
                "A", [1.0, 2.0], right_censored=5, covariates={"load": np.nan}
            ),
            ItemHistory(
                "B", [3.0], right_censored=5, covariates={"load": 2.0}
            ),
        ]
    )
    assert np.isnan(data.item("A").covariate("load"))
    assert data.without("covariates").Z is None
    with pytest.raises(ValueError, match="item\\(s\\) A have one missing"):
        sp.ProportionalIntensityHPP.fit(data)
    assert sp.ProportionalIntensityHPP.fit(data.select(ids=["B"])) is not None


# A covariate by name


def test_covariate_by_name():
    item = NAMED.item("a")
    assert item.covariate("q") == 2.0
    with pytest.raises(ValueError, match="no covariate 'r'; its covariates"):
        item.covariate("r")
    with pytest.raises(ValueError, match="have no names"):
        UNNAMED.item("a").covariate("p")
    with pytest.raises(ValueError, match="has no covariates"):
        ItemLifetime("z", observed=1).covariate("p")
    history = ItemHistory(
        "H", [1.0], right_censored=4, covariates={"load": 3.0}
    )
    assert history.covariate("load") == 3.0


def test_select_by_a_named_covariate():
    data = SurpyvalData.from_items(
        [
            ItemLifetime(
                "P1", observed=5, covariates={"site": 1, "load": 0.8}
            ),
            ItemLifetime(
                "P2", observed=6, covariates={"load": 0.5, "site": 0}
            ),
        ]
    )
    assert data.select(where=lambda item: item.covariate("site") == 1).ids == [
        "P1"
    ]


# Round trips bring back the same data


def _rich():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SurpyvalData.from_items(
            [
                ItemLifetime("b", right_censored=9, covariates={"age": 2.0}),
                ItemLifetime(
                    "a",
                    observed=5,
                    covariates={"age": 1.0},
                    event_type="x",
                    left_truncation=1,
                ),
                ItemLifetime(
                    "c",
                    interval_censored=[2, 4],
                    covariates={"age": np.nan},
                    event_type="y",
                ),
                ItemLifetime("d", left_censored=3, covariates={"age": 4.0}),
            ],
            counts=[1, 2, 3, 1],
        )


def _same(a, b):
    assert sorted(map(repr, a.ids)) == sorted(map(repr, b.ids))
    assert a.covariate_names == b.covariate_names
    for item in a.ids:
        assert a.item(item) == b.item(item)
        k_a, k_b = a.ids.index(item), b.ids.index(item)
        assert a.n[k_a] == b.n[k_b]


@pytest.mark.parametrize(
    "cycle",
    [
        lambda d: SurpyvalData.from_observations(**d.to_observations()),
        lambda d: SurpyvalData.from_items(d.to_items(), counts=d.n),
        lambda d: SurpyvalData.from_json(d.to_json()),
    ],
    ids=["observations", "items", "json"],
)
def test_round_trips_bring_back_the_same_data(cycle):
    data = _rich()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same(data, cycle(data))


def test_ids_are_given_for_every_kind_or_none():
    with pytest.raises(ValueError, match="for every kind with entries"):
        SurpyvalData.from_observations(
            observed=[1, 2], right_censored=[3], observed_ids=["a", "b"]
        )
    with pytest.raises(ValueError, match="one id per entry"):
        SurpyvalData.from_observations(observed=[1, 2], observed_ids=["a"])
