"""EventHistoryData rows (#925): one row stays a row, the constructor
checks its arrays, an empty selection is refused clearly, and the repr is
a readable table."""

import numpy as np
import pytest

from surpyval import EventHistoryData, ItemHistory, SurpyvalData

COUNTED = EventHistoryData.from_items(
    [
        ItemHistory(
            "A",
            interval_censored=[[3, 4]],
            right_censored=6,
            covariates={"load": 1.0, "size": 2.0},
        )
    ]
)


def test_one_row_keeps_its_interval_and_covariates():
    row = COUNTED[0]
    assert row.x.tolist() == [[3.0, 4.0]]
    assert row.Z.tolist() == [[1.0, 2.0]]
    assert row.covariate_names == ["load", "size"]
    assert len(row) == 1
    last = COUNTED[-1].to_items()[0]
    assert last.covariates.tolist() == [1.0, 2.0]


def test_rows_by_list_and_mask_as_by_slice():
    assert COUNTED[[0, 1]].x.tolist() == COUNTED[0:2].x.tolist()
    mask = np.array([True, False])
    assert COUNTED[mask].x.tolist() == [[3.0, 4.0]]


@pytest.mark.parametrize(
    "args, match",
    [
        (([1, 2, 3], [1, 1], [0, 0], [1, 1]), "x and i must have the same"),
        (([1, 2], [1, 1], [7, 0], [1, 1]), "must be one of -1"),
        (([1, 2], [1, 1], [0, 0], [-1, 1]), "strictly positive"),
        ((["a", "b"], [1, 1], [0, 0], [1, 1]), "'x' must be numbers"),
        (([1, 2], [1, 1], [True, False], [1, 1]), "True/False 'c'"),
        (([1.0, np.inf], [1, 1], [0, 1], [1, 1]), "finite"),
        (([], [], [], []), "at least one row"),
    ],
)
def test_the_constructor_checks_its_arrays(args, match):
    with pytest.raises(ValueError, match=match):
        EventHistoryData(*args)


def test_the_constructor_takes_one_value_for_every_row():
    data = EventHistoryData([1.0, 2.0, 3.0], 1, [0, 0, 1], 1, tl=0.5)
    assert data.i.tolist() == [1, 1, 1] and data.n.tolist() == [1, 1, 1]
    assert data.tl.tolist() == [0.5, 0.5, 0.5]


def test_an_empty_selection_is_refused_clearly():
    with pytest.raises(ValueError, match="selects no rows"):
        COUNTED[5:5]
    with pytest.raises(ValueError, match="selects no rows"):
        SurpyvalData([1.0, 2.0, 3.0])[5:5]


def test_repr_is_a_table_of_the_first_rows():
    text = repr(COUNTED)
    assert text.splitlines()[0] == "EventHistoryData: 1 item, 2 rows"
    assert "load" in text and "[3.0, 4.0]" in text
    many = EventHistoryData.from_items(
        [
            ItemHistory(k, [1.0, 2.0], right_censored=10, left_truncation=0.5)
            for k in range(8)
        ]
    )
    lines = repr(many).splitlines()
    assert lines[1].split() == ["i", "x", "c", "n", "tl"]
    assert lines[-1].startswith("  ... 14 more rows")
