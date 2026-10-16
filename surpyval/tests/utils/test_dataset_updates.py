"""Updating a dataset with new records (#939): update (replace by id or
add), drop, and extend_running, on both datasets."""

import numpy as np
import pytest

from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

FLEET = SurpyvalData.from_items(
    [
        ItemLifetime("CV-1", right_censored=500, covariates={"load": 1.0}),
        ItemLifetime("CV-2", right_censored=700, covariates={"load": 2.0}),
        ItemLifetime("CV-3", observed=300, covariates={"load": 3.0}),
    ],
    counts=[1, 3, 2],
    unit="hours",
)


def test_the_monthly_refresh_is_two_lines():
    failed = FLEET.item("CV-1").fail(560, "seal")
    with pytest.warns(UserWarning, match="masked"):
        refreshed = FLEET.update(
            [
                failed,
                ItemLifetime(
                    "CV-4", right_censored=60, covariates={"load": 1.5}
                ),
            ],
            counts={"CV-4": 5},
        ).extend_running(760)
    assert refreshed.ids == ["CV-1", "CV-2", "CV-3", "CV-4"]
    assert refreshed.c.tolist() == [0, 1, 0, 1]
    assert refreshed.x.tolist() == [560.0, 760.0, 300.0, 760.0]
    assert refreshed.n.tolist() == [1, 3, 2, 5]  # counts kept and given
    assert refreshed.covariate_names == ["load"] and refreshed.unit == "hour"
    assert list(refreshed.e) == ["seal", None, None, None]
    assert FLEET.item("CV-1").right_censored == 500  # the old data as it was


def test_update_checks_its_records():
    with pytest.raises(ValueError, match="update needs item ids"):
        SurpyvalData([1, 2]).update(ItemLifetime(0, observed=1))
    with pytest.raises(TypeError, match="takes ItemLifetime records; got"):
        FLEET.update([ItemHistory("A", observed=[1])])
    with pytest.raises(ValueError, match="'CV-9' is given twice"):
        FLEET.update([ItemLifetime("CV-9", observed=1, covariates=1)] * 2)
    with pytest.raises(ValueError, match="'counts' names items not given"):
        FLEET.update(FLEET.item("CV-1").extend(600), counts={"CV-2": 4})
    with pytest.raises(ValueError, match="must be given for every item"):
        FLEET.update(ItemLifetime("CV-9", observed=1))  # no covariates
    with pytest.raises(ValueError, match="different units"):
        FLEET.update(
            ItemLifetime("CV-9", observed=1, covariates=1, unit="days")
        )
    one = FLEET.update(FLEET.item("CV-2").extend(800), counts={"CV-2": 7})
    assert one.n.tolist() == [1, 7, 2] and one.x[1] == 800


def test_drop():
    assert FLEET.drop("CV-1").ids == ["CV-2", "CV-3"]
    assert FLEET.drop(["CV-1", "CV-3"]).n.tolist() == [3]
    with pytest.raises(KeyError, match="No items 'X', 'Y' in the data"):
        FLEET.drop(["X", "CV-1", "Y"])
    with pytest.raises(ValueError, match="No items are left"):
        FLEET.drop(FLEET.ids)
    with pytest.raises(ValueError, match="drop needs item ids"):
        SurpyvalData([1, 2]).drop(0)


def test_extend_running_by_age_dict_and_function():
    assert FLEET.extend_running(600).x.tolist() == [600.0, 700.0, 300.0]
    assert FLEET.extend_running({"CV-1": 550}).x.tolist() == [
        550.0,
        700.0,
        300.0,
    ]
    seen = {"CV-1": 650, "CV-2": 650, "CV-3": 650}
    assert FLEET.extend_running(lambda item: seen[item.id]).x.tolist() == [
        650.0,
        700.0,
        300.0,
    ]
    with pytest.raises(ValueError, match="'CV-3' is observed, not still"):
        FLEET.extend_running({"CV-3": 900})
    with pytest.raises(ValueError, match="takes one age"):
        FLEET.extend_running("900")
    # The same data as built from the extended arrays
    extended = FLEET.extend_running(600)
    built = SurpyvalData(
        [600, 700, 300],
        c=[1, 1, 0],
        n=[1, 3, 2],
        Z=[[1.0], [2.0], [3.0]],
        ids=["CV-1", "CV-2", "CV-3"],
    )
    for a, b in zip(extended.to_xcnt(), built.to_xcnt()):
        np.testing.assert_array_equal(a, b)


def test_histories_update_drop_and_extend():
    data = EventHistoryData.from_items(
        [
            ItemHistory("M1", [1, 2], right_censored=3, attributes={"p": "N"}),
            ItemHistory("M2", [1.5], attributes={"p": "S"}),
        ]
    )
    updated = data.update(
        [
            data.item("M2").add_failure(4),
            ItemHistory("M3", right_censored=1, attributes={"p": "N"}),
        ]
    )
    assert updated.ids == ["M1", "M2", "M3"]
    assert updated.item("M2").observed.tolist() == [1.5, 4.0]
    assert updated.attributes == [{"p": "N"}, {"p": "S"}, {"p": "N"}]
    assert data.drop("M1").ids == ["M2"]
    on = data.extend_running(5)
    assert [h.right_censored for h in on.to_items()] == [5.0, 5.0]
    assert data.extend_running({"M1": 2.5}).item("M1").right_censored == 3.0
    with pytest.raises(ValueError, match="before its end"):
        data.item("M1").extend(2)
