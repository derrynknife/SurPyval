"""Records stay fixed through pickling and copying, a schedule given to a
record is the record's own, and a dataset's arrays cannot be changed in
place (#927)."""

import copy
import pickle

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)

READ_ONLY = "read-only"


@pytest.mark.parametrize(
    "clone", [None, copy.deepcopy], ids=["pickle", "deepcopy"]
)
def test_records_stay_fixed_when_copied(clone):
    def again(record):
        if clone is None:
            return pickle.loads(pickle.dumps(record))
        return clone(record)

    history = again(ItemHistory("A", observed=[1, 2], right_censored=5))
    with pytest.raises(ValueError, match=READ_ONLY):
        history.observed[0] = 99
    with pytest.raises(AttributeError, match="fixed once made"):
        history.right_censored = 9
    lifetime = again(ItemLifetime("B", interval_censored=[1, 3]))
    with pytest.raises(ValueError, match=READ_ONLY):
        lifetime.value[1] = 0.5
    assert lifetime == ItemLifetime("B", interval_censored=[1, 3])


def test_a_schedule_given_to_a_record_is_its_own():
    schedule = StepSchedule.from_changepoints([0, 500], [[0.0], [1.0]])
    record = ItemLifetime("a", observed=900, time_varying_covariates=schedule)
    seen = {record}
    before = hash(record)
    schedule.Z[1, 0] = 7.0
    assert record.time_varying_covariates.Z.tolist() == [[0.0], [1.0]]
    assert hash(record) == before and record in seen
    with pytest.raises(ValueError, match=READ_ONLY):
        record.time_varying_covariates.Z[0, 0] = 3.0
    unpickled = pickle.loads(pickle.dumps(record))
    with pytest.raises(ValueError, match=READ_ONLY):
        unpickled.time_varying_covariates.Z[0, 0] = 3.0


def test_dataset_arrays_cannot_be_changed_in_place():
    data = SurpyvalData([1.0, 2.0, 3.0], [0, 0, 0])
    for name in ("x", "c", "n", "t", "x_o"):
        with pytest.raises(ValueError, match=READ_ONLY):
            getattr(data, name)[0] = 1
    history = EventHistoryData.from_items(
        [ItemHistory("A", [1.0, 2.0], right_censored=5)]
    )
    for name in ("x", "i", "c", "n", "tl"):
        with pytest.raises(ValueError, match=READ_ONLY):
            getattr(history, name)[0] = 1
    for again in (pickle.loads(pickle.dumps(data)), copy.deepcopy(history)):
        with pytest.raises(ValueError, match=READ_ONLY):
            again.c[0] = 1


def test_the_callers_arrays_stay_theirs():
    Z = np.array([[1.0], [2.0], [3.0]])
    data = SurpyvalData([1.0, 2.0, 3.0], Z=Z)
    Z[0, 0] = 5.0  # the caller's array is still writeable
    assert data.Z[0, 0] == 1.0 and data.Z_o[0, 0] == 1.0
    x = np.array([1.0, 2.0, 3.0])
    history = EventHistoryData(x, [1, 1, 1], [0, 0, 1], 1)
    x[0] = 9.0
    assert history.x[0] == 1.0


def test_a_dataset_fits_the_same_after_pickling():
    data = SurpyvalData([1.0, 2.0, 3.0, 5.0], [0, 0, 1, 0])
    again = pickle.loads(pickle.dumps(data))
    assert np.allclose(
        sp.Weibull.fit(data).params, sp.Weibull.fit(again).params
    )
