"""The dict and JSON forms of every data class (#935): round trips
through text, a file path and surpyval.from_json / from_dict."""

import os

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
    handle_xicn,
)

STEPS = StepSchedule.from_changepoints([0, 3], [[0.0], [1.0]])

LIVES = [
    ItemLifetime(
        "A",
        observed=5,
        event_type="seal",
        covariates={"load": 1.0, "site": 2.0},
        left_truncation=1,
        time_varying_covariates={0: [0.0], 3: [1.0]},
    ),
    ItemLifetime(
        "B",
        interval_censored=[4, 8],
        right_truncation=20,
        covariates={"load": 0.5, "site": None},
        time_varying_covariates=STEPS,
    ),
    ItemLifetime(
        "C",
        right_censored=9,
        covariates={"load": 0.1, "site": 1.0},
        time_varying_covariates={-np.inf: [0.0]},
    ),
]
GAP = ItemLifetime(("C", 2), observed=4, event_type=("bearing", 1))
HISTORIES = [
    ItemHistory(
        "M1",
        observed=[420, 1310],
        observed_event_types=["seal", ("bearing", 1)],
        interval_censored=[[2000, 2100]],
        interval_censored_counts=2,
        interval_censored_event_types="seal",
        right_censored=2600,
        left_truncation=100,
        covariates={"load": 1.0},
        initial_state="ok",
        observed_states={1000: "worn"},
        interval_censored_states=[[1500, 1800, "bad"]],
        time_varying_covariates={0: [1.0], 500: [2.0]},
    ),
    ItemHistory(
        "M2",
        left_censored=300,
        left_censored_counts=2,
        left_censored_event_types="seal",
        right_censored=2900,
        covariates={"load": 0.0},
        initial_state="ok",
        time_varying_covariates={0: [0.0]},
    ),
]


def _same_lifetimes(a, b):
    assert a.to_items() == b.to_items()
    assert a.n.tolist() == b.n.tolist()
    assert a.covariate_names == b.covariate_names


@pytest.mark.parametrize("record", [*LIVES, GAP, *HISTORIES])
def test_records_round_trip(record, tmp_path):
    cls = type(record)
    assert cls.from_dict(record.to_dict()) == record
    assert cls.from_json(record.to_json()) == record
    path = os.path.join(tmp_path, "record.json")
    record.to_json(path)
    assert cls.from_json(path) == record
    assert sp.from_dict(record.to_dict()) == record
    assert sp.from_json(path) == record
    assert record.to_dict()["class"] == cls.__name__


def test_record_dicts_are_plain_and_strict():
    import json

    d = GAP.to_dict()
    assert d["id"] == {"tuple": ["C", 2]}
    assert d["event_type"] == {"tuple": ["bearing", 1]}
    d = LIVES[2].to_dict()
    json.dumps(d, allow_nan=False)  # no Infinity: the -inf age is encoded
    assert d["non_finite"]
    d = HISTORIES[0].to_dict()
    assert d["observed_event_types"] == ["seal", {"tuple": ["bearing", 1]}]
    assert d["observed_states"] == [[1000.0, "worn"]]
    assert d["time_varying_covariates"] == {
        "ages": [0.0, 500.0],
        "values": [[1.0], [2.0]],
    }


def test_lifetime_data_round_trips(tmp_path):
    with pytest.warns(UserWarning, match="masked"):
        data = SurpyvalData.from_items(LIVES, counts=[2, 1, 3])
    _same_lifetimes(SurpyvalData.from_dict(data.to_dict()), data)
    _same_lifetimes(SurpyvalData.from_json(data.to_json()), data)
    path = os.path.join(tmp_path, "data.json")
    data.to_json(path)
    _same_lifetimes(SurpyvalData.from_json(path), data)
    _same_lifetimes(sp.from_json(path), data)
    _same_lifetimes(sp.from_dict(data.to_dict()), data)
    assert data.to_dict()["class"] == "SurpyvalData"
    # Plain data, with no ids
    plain = SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1], tl=0.5)
    again = sp.from_json(plain.to_json())
    assert again.ids is None and again.n.tolist() == [1, 2, 1]
    assert again.tl.tolist() == [0.5, 0.5, 0.5]


def test_event_history_data_round_trips(tmp_path):
    data = EventHistoryData.from_items(HISTORIES)
    for again in (
        EventHistoryData.from_dict(data.to_dict()),
        EventHistoryData.from_json(data.to_json()),
        sp.from_dict(data.to_dict()),
    ):
        assert again.to_items() == data.to_items()
        assert again.covariate_names == ["load"]
    path = os.path.join(tmp_path, "history.json")
    data.to_json(path)
    assert sp.from_json(path).to_items() == data.to_items()
    numbered = EventHistoryData.from_histories(
        observed={1: [1, 2], 2: [3]}, right_censored={1: 5, 2: 5}
    )
    assert sp.from_json(numbered.to_json()).ids == [1, 2]


def test_windows_and_paths_are_refused():
    gapped = handle_xicn(
        [1, 2, 25], [1, 1, 1], windows={1: [(0, 10), (20, 40)]}
    )
    with pytest.raises(ValueError, match="observation windows"):
        gapped.to_dict()
    path = sp.CovariatePath.from_points([0.0, 5.0], [[0.0], [1.0]])
    item = ItemLifetime("A", observed=5, time_varying_covariates=path)
    with pytest.raises(ValueError, match="CovariatePath .* no dict or JSON"):
        item.to_dict()


def test_the_wrong_reader_names_the_right_one():
    with pytest.raises(ValueError, match="written by ItemHistory, not"):
        ItemLifetime.from_dict(HISTORIES[1].to_dict())
    with pytest.raises(ValueError, match="written by SurpyvalData, not"):
        EventHistoryData.from_dict(SurpyvalData([1, 2]).to_dict())
    with pytest.raises(ValueError, match="needs 'x'"):
        SurpyvalData.from_dict({"class": "SurpyvalData"})
    with pytest.raises(ValueError, match="tagged by its 'class'"):
        sp.from_dict({"x": [1]})
    with pytest.raises(FileNotFoundError, match="No such file"):
        SurpyvalData.from_json("d.json")
