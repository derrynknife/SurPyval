"""The observation notation (surpyval.utils.notation): O, RC, LC, IC with
counts and truncation windows, text parsing, and the arrays it hands on."""

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.utils.notation import (
    IC,
    LC,
    RC,
    O,
    Observations,
    Truncation,
)


def test_each_kind_has_its_censoring_code():
    data = Observations([O(10), RC(20), LC(3), IC(5, 10)])
    np.testing.assert_array_equal(data.xcnt()["c"], [0, 1, -1, 2])


def test_counts_multiply_and_compose():
    assert (3 * O(10)).n == 3
    assert (O(10) * 3).n == 3
    assert (2 * (3 * O(10))).n == 6
    with pytest.raises(ValueError, match="positive"):
        0 * O(10)


def test_operators_do_not_change_the_original():
    base = O(10)
    _ = 3 * base
    _ = base / Truncation(0, 50)
    assert base.n == 1 and base.truncation is None


def test_one_entry_per_observation_with_truncation_on_its_own_row():
    # Truncated and untruncated rows mixed: each row keeps its own window.
    data = Observations(
        [
            O(10),
            RC(20) / Truncation(5, 30),
            IC(5, 10),
            O(12) / Truncation(2, None),
        ]
    )
    a = data.xcnt()
    assert {len(v) for v in a.values()} == {4}
    np.testing.assert_array_equal(a["tl"], [-np.inf, 5, -np.inf, 2])
    np.testing.assert_array_equal(a["tr"], [np.inf, 30, np.inf, np.inf])


def test_a_point_lies_in_its_window_and_the_left_edge_is_outside():
    with pytest.raises(ValueError, match="outside its observation window"):
        O(10) / Truncation(10, 50)
    with pytest.raises(ValueError, match="outside"):
        O(60) / Truncation(10, 50)
    assert (O(50) / Truncation(10, 50)).truncation == Truncation(10, 50)


def test_an_interval_needs_some_part_inside_its_window():
    assert (IC(5, 15) / Truncation(10, 50)).truncation.left == 10
    with pytest.raises(ValueError, match="outside"):
        IC(1, 10) / Truncation(10, 50)


def test_a_window_for_the_whole_set_keeps_any_own_window():
    data = Observations([O(10), O(20) / Truncation(15, 40)]) / Truncation(
        0, 100
    )
    a = data.xcnt()
    np.testing.assert_array_equal(a["tl"], [0, 15])
    np.testing.assert_array_equal(a["tr"], [100, 40])


def test_text_reads_every_form():
    data = Observations.parse("3*10, 20+, <3, [5, 10]\n2×7; 1.5e1")
    a = data.xcnt()
    np.testing.assert_array_equal(a["c"], [0, 1, -1, 2, 0, 0])
    np.testing.assert_array_equal(a["n"], [3, 1, 1, 1, 2, 1])
    np.testing.assert_array_equal(a["x"][:, 0], [10, 20, 3, 5, 7, 15])
    np.testing.assert_array_equal(a["x"][3], [5, 10])


def test_a_trailing_minus_is_left_censored_too():
    a = Observations.parse("15-, <15").xcnt()
    np.testing.assert_array_equal(a["c"], [-1, -1])
    np.testing.assert_array_equal(a["x"][:, 0], [15, 15])


def test_text_that_cant_be_read_says_how_to_write_it():
    with pytest.raises(ValueError, match=r"20\+ \(still running at 20\)"):
        Observations.parse("10, twenty")


def test_bad_inputs_are_refused_with_what_to_do():
    with pytest.raises(ValueError, match="below its end"):
        IC(10, 5)
    with pytest.raises(ValueError, match="below the right edge"):
        Truncation(5, 5)
    with pytest.raises(ValueError, match="not NaN"):
        O(float("nan"))
    with pytest.raises(ValueError, match="parse"):
        Observations([10])
    with pytest.raises(ValueError, match="no observations"):
        Observations().xcnt()


def test_joining_slicing_and_counting():
    data = Observations([3 * O(10)]) + RC(20) + Observations([IC(1, 2)])
    assert len(data) == 3 and data.units == 5
    assert isinstance(data[:2], Observations) and len(data[:2]) == 2
    assert data[1] == RC(20)


def test_the_printout_reads_as_words():
    text = repr(Observations([3 * O(10), IC(5, 10) / Truncation(0, 40)]))
    assert "2 rows (4 units)" in text
    assert "3 × failed at 10" in text
    assert "failed between 5 and 10, observed in (0, 40]" in text


def test_a_fit_from_the_notation_matches_the_same_data_as_arrays():
    data = Observations(
        [3 * O(10), RC(20), LC(3), IC(5, 10) / Truncation(0, 40), O(14), O(8)]
    )
    from_notation = Weibull.fit(data.surpyval_data())
    from_arrays = Weibull.fit(
        x=[10, 20, 3, [5, 10], 14, 8],
        c=[0, 1, -1, 2, 0, 0],
        n=[3, 1, 1, 1, 1, 1],
        tl=[-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf],
        tr=[np.inf, np.inf, np.inf, 40, np.inf, np.inf],
    )
    np.testing.assert_allclose(
        from_notation.params, from_arrays.params, rtol=1e-9
    )


def test_no_truncation_columns_when_nothing_is_truncated():
    data = Observations([O(10), RC(20), O(5)]).surpyval_data()
    assert not np.isfinite(data.t).any()
