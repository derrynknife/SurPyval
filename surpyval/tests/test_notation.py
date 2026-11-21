"""The observation notation (surpyval.utils.notation): O, RC, LC, IC with
counts and truncation windows, text parsing, and the arrays it hands on."""

import numpy as np
import pytest

from surpyval import KaplanMeier, SurpyvalData, Weibull
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
    with pytest.raises(ValueError, match="whole number"):
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


def test_an_interval_lies_inside_its_window_as_surpyval_data_says():
    # tl <= lower and upper <= tr, the rule SurpyvalData checks
    assert (IC(10, 15) / Truncation(10, 50)).truncation.left == 10
    assert (IC(20, 50) / Truncation(10, 50)).truncation.right == 50
    with pytest.raises(ValueError, match="cannot start below its own left"):
        IC(5, 15) / Truncation(10, 50)
    with pytest.raises(ValueError, match="outside"):
        IC(20, 60) / Truncation(10, 50)
    # SurpyvalData gives the same reason for the same row
    with pytest.raises(ValueError, match="cannot start below its own left"):
        SurpyvalData(x=[[5, 15]], c=[2], tl=[10])


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


# --- The rules agree with SurpyvalData's ---------------------------------


def test_a_count_must_be_a_whole_number():
    with pytest.raises(ValueError, match="whole number.*w= to SurpyvalData"):
        2.5 * O(10)
    with pytest.raises(ValueError, match="whole number"):
        Observations.parse("2.5*10")
    assert (O(10) * 3.0).n == 3 and isinstance((O(10) * 3.0).n, int)
    # SurpyvalData refuses the same count
    with pytest.raises(ValueError, match="integer"):
        SurpyvalData([10], n=[2.5])


def test_each_point_kind_keeps_surpyval_datas_window_rule():
    # observed and left censored: (tl, tr]; right censored: (tl, tr)
    for kind in (O, LC):
        assert (kind(50) / Truncation(10, 50)).truncation is not None
        with pytest.raises(ValueError, match="no observation window"):
            kind(10) / Truncation(10, 50)
    with pytest.raises(ValueError, match="strictly less than its right"):
        RC(50) / Truncation(10, 50)
    for c in (0, 1, -1):
        with pytest.raises(ValueError):
            SurpyvalData([10], c=[c], tl=[10])
    with pytest.raises(ValueError, match="strictly less than its right"):
        SurpyvalData([50], c=[1], tr=[50])


def test_an_open_interval_is_one_sided_censoring_everywhere():
    inf = float("inf")
    assert IC(5, inf) == RC(5) and IC(-inf, 5) == LC(5)
    obs = Observations([IC(5, inf), IC(-inf, 3)])
    np.testing.assert_array_equal(obs.xcnt()["c"], [1, -1])
    data = obs.surpyval_data()
    assert sorted(data.c.tolist()) == [-1, 1]
    # as SurpyvalData reads the same rows
    same = SurpyvalData(x=[[5, inf], [-inf, 3]], c=[2, 2])
    np.testing.assert_array_equal(data.c, same.c)
    np.testing.assert_array_equal(data.x, same.x)
    assert Observations.parse("[5, inf], [-inf, 3]") == obs
    with pytest.raises(ValueError, match="no information"):
        IC(-inf, inf)


def test_an_infinite_point_is_refused_when_built():
    inf = float("inf")
    for kind in (O, RC, LC):
        with pytest.raises(ValueError, match="must be finite"):
            kind(inf)
        with pytest.raises(ValueError, match="must be finite"):
            kind(-inf)
    with pytest.raises(ValueError, match="must be finite"):
        Observations.parse("inf")


def test_the_printout_counts_one_unit_in_the_singular():
    assert repr(Observations([O(1)])).startswith(
        "Observations, 1 row (1 unit):"
    )
    assert repr(Observations()) == "Observations, 0 rows (0 units)"


# --- Text written and read back ------------------------------------------


def test_a_window_per_entry_in_text():
    obs = Observations.parse("12 | (2, 50], 3*20+ | (2, inf], <4 | (-inf, 9]")
    a = obs.xcnt()
    np.testing.assert_array_equal(a["tl"], [2, 2, -np.inf])
    np.testing.assert_array_equal(a["tr"], [50, np.inf, 9])
    np.testing.assert_array_equal(a["n"], [1, 3, 1])
    with pytest.raises(ValueError, match=r"window in '12 \| 2'"):
        Observations.parse("12 | 2, 50")
    with pytest.raises(ValueError, match="outside its observation window"):
        Observations.parse("60 | (2, 50]")


@pytest.mark.parametrize(
    "text",
    [
        "10, 20+, <3, [5, 10]",
        "3*10, 2*20+ | (2, inf], [4, 9] | (4, 40], <3 | (-inf, 40]",
        "0.1, 1e+20, 0.30000000000000004+, 12345.678901234567",
        "",
    ],
)
def test_text_to_notation_to_text_is_the_same(text):
    obs = Observations.parse(text)
    assert obs.text() == text
    assert Observations.parse(obs.text()) == obs


def test_text_is_written_in_one_form_however_it_was_typed():
    obs = Observations.parse("3×10; 15-\n[5, inf]")
    assert obs.text() == "3*10, <15, 5+"


def test_equality_compares_observations_in_order_and_the_unit():
    a = Observations.parse("10, 20+")
    assert a == Observations([O(10), RC(20)])
    assert a != Observations([RC(20), O(10)])
    assert a != Observations.parse("10, 20+", unit="hours")
    assert a != "10, 20+"
    with pytest.raises(TypeError):
        hash(a)


def test_the_unit_is_kept_and_joined():
    a = Observations.parse("10", unit="hours")
    assert (a + O(20)).unit == "hours"
    assert (a + Observations.parse("5")).unit == "hours"
    assert a.surpyval_data().unit == "hours"
    assert "times in hours" in repr(a)
    with pytest.raises(ValueError, match="convert one set"):
        a + Observations.parse("5", unit="km")
    with pytest.raises(ValueError, match="leave unit= out"):
        a.surpyval_data(unit="km")


# --- SurpyvalData from and to the notation -------------------------------


def _same_data(a, b):
    for name in ("x", "c", "n", "t"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.unit == b.unit


def test_data_to_notation_to_data_is_the_same():
    inf = float("inf")
    datasets = [
        SurpyvalData([10, 20, 20, 3, 5], c=[0, 1, 1, -1, 0]),
        SurpyvalData(
            x=[[1, 4], 6, 9, [2, 8]],
            c=[2, 0, 1, 2],
            n=[3, 1, 2, 1],
            tl=[0, 1, 1, -inf],
            tr=[10, 50, 60, 9],
        ),
        SurpyvalData([600, 700, 1000], c=[0, 0, 1], tl=500, unit="hours"),
    ]
    for data in datasets:
        back = SurpyvalData.from_notation(data.to_notation())
        _same_data(back, data)
        _same_data(
            SurpyvalData.from_notation(
                data.to_notation().text(), unit=data.unit
            ),
            data,
        )


def test_from_notation_reads_text_or_observations():
    from_text = SurpyvalData.from_notation("3*10, 20+, [5, 10]", unit="h")
    from_obs = SurpyvalData.from_notation(
        Observations([3 * O(10), RC(20), IC(5, 10)]), unit="h"
    )
    _same_data(from_text, from_obs)
    assert from_text.unit == "h"
    with pytest.raises(TypeError, match="text"):
        SurpyvalData.from_notation([10, 20])


def test_to_notation_refuses_what_it_would_drop():
    cases = {
        "covariates": SurpyvalData([1, 2], Z=[[1], [2]]),
        "event types": SurpyvalData([1, 2], e=["a", "b"]),
        "unit ids": SurpyvalData([1, 2], i=["A", "B"]),
        "weights": SurpyvalData([1, 2], w=[1, 2]),
    }
    for what, data in cases.items():
        with pytest.raises(ValueError, match=f"{what}.*to_observations"):
            data.to_notation()


# --- Every fitter takes an Observations ----------------------------------


def test_a_fitter_takes_observations_as_it_takes_surpyval_data():
    obs = Observations.parse("10, 20+, 5, 7, 30, [8, 12], 14")
    np.testing.assert_allclose(
        Weibull.fit(obs).params,
        Weibull.fit(obs.surpyval_data()).params,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        KaplanMeier.fit(Observations.parse("10, 20+, 5, 7, 30")).R,
        KaplanMeier.fit([10, 20, 5, 7, 30], c=[0, 1, 0, 0, 0]).R,
    )
    with pytest.raises(TypeError, match="leave out 'c'"):
        Weibull.fit(obs, c=[0] * 7)


def test_the_unit_reaches_the_model():
    model = Weibull.fit(Observations.parse("10, 20+, 5, 7, 30", unit="hours"))
    assert model.unit == "hours"


def test_the_other_entry_points_take_observations():
    from surpyval import life_table
    from surpyval.recurrent import JelinskiMoranda

    obs = Observations.parse("3, 5, 4+, 8, 9+")
    pd_testing = pytest.importorskip("pandas").testing
    pd_testing.assert_frame_equal(
        life_table(obs, intervals=[0, 4, 8, 12]),
        life_table(
            [3, 5, 4, 8, 9], c=[0, 0, 1, 0, 1], intervals=[0, 4, 8, 12]
        ),
    )
    times = np.cumsum([3, 5, 4, 8, 5, 8, 8, 6, 11, 12, 11, 18, 21])
    text = ", ".join(
        f"{t:g}" + ("+" if k == 12 else "") for k, t in enumerate(times)
    )
    model = JelinskiMoranda.fit(Observations.parse(text))
    same = JelinskiMoranda.fit(times, c=[0] * 12 + [1])
    assert model.N == same.N and model.phi == same.phi
