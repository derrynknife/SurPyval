"""Marker readings on lifetime records and datasets (#891): an
``ItemLifetime``'s ``markers={age: value}``, checked against its
observation, carried by ``SurpyvalData`` through every reshape, slice and
JSON, and refused by every fit that does not take them."""

import json
import pickle

import numpy as np
import pytest

import surpyval as sp
from surpyval import ItemLifetime, SurpyvalData


def _items():
    return [
        ItemLifetime(
            "A", observed=50, markers={20: 2.0, 0: 1.0}, covariates={"z": 1.0}
        ),
        ItemLifetime(
            "B",
            right_censored=80,
            left_truncation=10,
            markers={10: 1.5, 40: 2.5, 70: 3.0},
            covariates={"z": 0.0},
        ),
        ItemLifetime("C", right_censored=90, covariates={"z": 0.0}),
        ItemLifetime(
            "D",
            interval_censored=[30, 60],
            markers={0: 1.2, 45: 2.2, 60: 2.4},
            covariates={"z": 1.0},
        ),
    ]


# -- the record ---------------------------------------------------------------


def test_markers_are_kept_in_order_of_age_and_shown():
    item = _items()[0]
    assert item.markers == {0.0: 1.0, 20.0: 2.0}
    assert list(item.markers) == [0.0, 20.0]
    assert "markers={0.0: 1.0, 20.0: 2.0}" in repr(item)
    assert ItemLifetime("A", observed=5).markers is None


def test_no_readings_is_no_markers():
    assert ItemLifetime("A", observed=5, markers={}) == ItemLifetime(
        "A", observed=5
    )


def test_equality_and_hash_follow_the_readings():
    a = ItemLifetime("A", observed=50, markers={0: 1.0, 20: 2.0})
    b = ItemLifetime("A", observed=50, markers={20: 2.0, 0.0: 1})
    c = ItemLifetime("A", observed=50, markers={0: 1.0, 20: 2.5})
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2


def test_record_is_fixed_and_pickles():
    item = _items()[1]
    with pytest.raises(ValueError):
        item._marker_ages[0] = 3.0
    item.markers[99.0] = 1.0  # a copy: the record is unchanged
    assert 99.0 not in item.markers
    again = pickle.loads(pickle.dumps(item))
    assert again == item and again.markers == item.markers


def test_clone_and_updates_carry_markers():
    item = _items()[1]
    assert item.clone("B2").markers == item.markers
    longer = item.extend(100).update_markers(95, 3.4)
    assert longer.markers == {10.0: 1.5, 40.0: 2.5, 70.0: 3.0, 95.0: 3.4}
    assert longer.update_markers(40, 9.0).markers[40.0] == 9.0
    failed = item.fail(85)
    assert failed.markers == item.markers


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(right_censored=90, markers={100: 1.0}), "after its"),
        (dict(right_censored=90, markers={100: 1.0}), r"extend\(100\)"),
        (dict(observed=90, markers={95: 1.0}), "observed age, 90"),
        (dict(right_censored=90, markers={-1: 1.0}), "age 0"),
        (
            dict(right_censored=90, left_truncation=20, markers={5: 1.0}),
            "left_truncation, 20",
        ),
        (dict(right_censored=90, markers={5: np.nan}), "finite"),
        (dict(right_censored=90, markers={5: np.inf}), "finite"),
        (dict(right_censored=90, markers={np.inf: 1.0}), "finite"),
        (dict(right_censored=90, markers={5: [1, 2]}), "one finite number"),
        (dict(right_censored=90, markers={5: "high"}), "one finite number"),
        (dict(right_censored=90, markers=[1.0, 2.0]), "dict"),
        (dict(right_censored=90, markers={"5": 1.0, 5: 2.0}), "two readings"),
    ],
)
def test_readings_outside_observation_or_not_numbers_are_refused(
    kwargs, match
):
    with pytest.raises(ValueError, match=match) as caught:
        ItemLifetime("P1", **kwargs)
    assert "'P1'" in str(caught.value)


def test_a_reading_after_the_end_is_refused_by_the_update():
    item = ItemLifetime("P1", right_censored=90, markers={0: 1.0})
    with pytest.raises(ValueError, match=r"extend\(120\)"):
        item.update_markers(120, 2.0)
    with pytest.raises(ValueError, match="after its right_censored age"):
        item.clone(right_censored=50, markers={60: 1.0})


def test_readings_at_the_ends_of_observation_are_kept():
    item = ItemLifetime(
        "P1", observed=90, left_truncation=10, markers={10: 1.0, 90: 2.0}
    )
    assert item.markers == {10.0: 1.0, 90.0: 2.0}
    # the upper end of an interval is its end
    item = ItemLifetime("P2", interval_censored=[30, 60], markers={60: 2.0})
    assert item.markers == {60.0: 2.0}


# -- the dataset --------------------------------------------------------------


def test_from_items_keeps_markers_by_item():
    data = SurpyvalData.from_items(_items())
    assert data.markers == {
        "A": {0.0: 1.0, 20.0: 2.0},
        "B": {10.0: 1.5, 40.0: 2.5, 70.0: 3.0},
        "C": {},
        "D": {0.0: 1.2, 45.0: 2.2, 60.0: 2.4},
    }
    assert data.to_items() == _items()
    assert data.item("B") == _items()[1]
    assert (
        SurpyvalData.from_items([ItemLifetime(1, observed=4)]).markers is None
    )


def test_summary_and_repr_count_the_readings():
    data = SurpyvalData.from_items(_items())
    assert "  markers: 8 readings on 3 items" in data.summary()
    assert "markers" in repr(data).splitlines()[1]  # a column per row
    assert "3 readings" in repr(data)
    assert "markers" not in repr(data.without("markers"))


def test_slicing_keeps_each_items_readings():
    data = SurpyvalData.from_items(_items())
    assert data[1:3].to_items() == _items()[1:3]
    assert data[[3, 0]].to_items() == [_items()[3], _items()[0]]
    assert data[2].markers == {"C": {}}


def test_json_round_trip_keeps_markers():
    data = SurpyvalData.from_items(_items())
    text = data.to_json()
    json.loads(text)  # strict JSON
    again = SurpyvalData.from_json(text)
    assert again.markers == data.markers
    assert again.to_items() == _items()


def test_without_markers_and_select():
    data = SurpyvalData.from_items(_items())
    plain = data.without("markers")
    assert plain.markers is None
    assert all(item.markers is None for item in plain.to_items())
    picked = data.select(where=lambda item: item.covariate("z") == 1)
    assert picked.markers == {
        "A": {0.0: 1.0, 20.0: 2.0},
        "D": {0.0: 1.2, 45.0: 2.2, 60.0: 2.4},
    }
    assert data.with_covariates(["z"]).markers == data.markers


def test_censor_at_drops_readings_after_the_new_end():
    data = SurpyvalData.from_items(_items()).censor_at(40)
    assert data.markers == {
        "A": {0.0: 1.0, 20.0: 2.0},
        "B": {10.0: 1.5, 40.0: 2.5},
        "C": {},
        # (30, 60] crosses 40: right censored at 30, its last good look
        "D": {0.0: 1.2},
    }
    assert data.item("D").right_censored == 30.0


def test_enter_at_drops_readings_before_the_new_start():
    data = SurpyvalData.from_items(_items()[:3]).enter_at(15)
    assert data.markers == {
        "A": {20.0: 2.0},
        "B": {40.0: 2.5, 70.0: 3.0},
        "C": {},
    }


def test_for_event_type_keeps_readings_inside_the_new_window():
    items = [
        ItemLifetime("A", observed=50, event_type="x", markers={0: 1.0}),
        ItemLifetime(
            "B",
            interval_censored=[30, 60],
            event_type="y",
            markers={0: 1.0, 45: 2.0},
        ),
    ]
    data = SurpyvalData.from_items(items).for_event_type("x")
    # B failed of another mode within (30, 60]: right censored at 30
    assert data.markers == {"A": {0.0: 1.0}, "B": {0.0: 1.0}}


def test_items_without_readings_alone_make_no_markers():
    data = SurpyvalData.from_items(_items()).censor_at(5)
    # A, D (read at 0) keep a reading; B entered at 10, so it is left out
    assert set(data.markers) == {"A", "C", "D"}


# -- every other fit refuses them --------------------------------------------


_REFUSED = 'This fit takes no markers, which the SurpyvalData holds; use data.without\\("markers"\\)'  # noqa: E501


def _plain():
    return SurpyvalData.from_items(
        [
            ItemLifetime(k, observed=float(10 + k), markers={0: float(k)})
            for k in range(8)
        ]
        + [ItemLifetime(99, right_censored=30.0)]
    )


@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.Weibull.fit(d),
        lambda d: sp.Weibull.fit_from_surpyval_data(d),
        lambda d: sp.KaplanMeier.fit(d),
        lambda d: sp.fit_best(d),
    ],
    ids=["Weibull", "fit_from_surpyval_data", "KaplanMeier", "fit_best"],
)
def test_other_fits_refuse_markers_naming_the_fix(fit):
    with pytest.raises(ValueError, match=_REFUSED):
        fit(_plain())
    # and fit the data without them
    fit(_plain().without("markers"))


def test_a_regression_refuses_markers_with_covariates_too():
    data = SurpyvalData.from_items(_items()[:3])
    with pytest.raises(ValueError, match=_REFUSED):
        sp.WeibullPH.fit(data)
    with pytest.raises(ValueError, match=_REFUSED):
        sp.CoxPH.fit(data)


def test_markers_are_refused_before_time_varying_covariates_are_fitted():
    from surpyval import StepSchedule

    steps = StepSchedule.from_changepoints([0, 20], [[0.0], [1.0]])
    items = [
        ItemLifetime(
            k,
            observed=float(10 + 3 * k),
            time_varying_covariates=steps,
            markers={0: 1.0},
        )
        for k in range(8)
    ]
    with pytest.raises(ValueError, match=_REFUSED):
        sp.WeibullPH.fit(SurpyvalData.from_items(items))
