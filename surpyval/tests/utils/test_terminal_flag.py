"""``a``, the terminal flag, and several rows per unit without states.

The baseline values (``terminal_flag_baseline.json``) were recorded with
the code before ``a`` was added: plain life data, step time-varying
covariates from records and data with states, as reprs, summaries,
dictionaries and fitted parameters. The fitted parameters were recorded
on one machine; another platform's floating point can differ in the
last bits, so they are compared to 1e-12 (``_same_fit``). Whether the
flag changes a fit is checked exactly, against the same fit without it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import ItemLifetime, StepSchedule, SurpyvalData
from surpyval.utils.warnings import MaskedCauseWarning

BASELINE = json.loads(
    (Path(__file__).parent / "terminal_flag_baseline.json").read_text()
)


def _tvc():
    """Single-event life data with time-varying covariates: A ran under
    load 0 to 4, then load 1 until it failed at 9; B was still running
    at 6; C ran to 3, then to 8 under load 1, still running."""
    return SurpyvalData(
        x=[4, 9, 6, 3, 8],
        tl=[0, 4, 0, 0, 3],
        c=[1, 0, 1, 1, 1],
        i=["A", "A", "B", "C", "C"],
        Z=[[0.0], [1.0], [0.0], [0.0], [1.0]],
    )


def _recurrent():
    """Repairs at 2 and 5 then scrapped at 9 (unit 1); a repair at 4,
    still running at 7 (unit 2)."""
    return SurpyvalData(
        x=[2, 5, 9, 4, 7], c=[0, 0, 0, 0, 1], i=[1, 1, 1, 2, 2]
    )


def _same_fit(params, recorded):
    """A fit's parameters against those recorded before ``a`` existed,
    to within the platforms' rounding."""
    np.testing.assert_allclose(params, recorded, rtol=1e-12, atol=0)


# -- the four patterns -------------------------------------------------------


def test_stretches_then_a_terminal_event():
    data = _tvc()
    assert data.a.tolist() == [0, 1, 0, 0, 0]
    assert data.a_inferred.tolist() == [False, True, False, False, False]
    assert data.n_items == 3


def test_repeated_events_then_more_rows():
    data = SurpyvalData(x=[2, 5, 7], c=[0, 0, 1], i=["u", "u", "u"])
    assert data.a.tolist() == [0, 0, 0]
    assert data.a_inferred.tolist() == [True, True, False]


def test_repeated_events_ending_in_a_terminal_event():
    data = _recurrent()
    assert data.a.tolist() == [0, 0, 1, 0, 0]
    assert data.a_inferred.tolist() == [True, True, True, True, False]


def test_last_row_still_running():
    data = SurpyvalData(x=[3, 8], c=[1, 1], tl=[0, 3], i=["v", "v"])
    assert data.a.tolist() == [0, 0]
    assert not data.a_inferred.any()
    assert data.summary().splitlines()[2] == (
        "  units: 1 (0 ended for good, 1 still running)"
    )


def test_explicit_a_wins_and_blank_cells_are_inferred():
    data = SurpyvalData(x=[2, 5, 9], c=[0, 0, 0], i=[1, 1, 1], a=[0, None, 0])
    assert data.a.tolist() == [0, 0, 0]
    assert data.a_inferred.tolist() == [False, True, False]
    single = SurpyvalData(x=[2, 5], c=[0, 1], a=[0, 0])
    assert single.a.tolist() == [0, 0]
    assert single.a_inferred.tolist() == [False, False]
    filled = SurpyvalData(x=[2, 5], c=[0, 1], a=[np.nan, None])
    assert filled.a.tolist() == [1, 0]
    assert filled.a_inferred.tolist() == [True, False]


def test_simultaneous_events_are_taken_as_the_recurrent_handler_takes_them():
    data = SurpyvalData(x=[4, 4, 6], c=[0, 0, 1], i=[1, 1, 1])
    assert data.a.tolist() == [0, 0, 0]
    # An end of record at an event's time
    data = SurpyvalData(x=[4, 4], c=[0, 1], i=[1, 1])
    assert data.a.tolist() == [0, 0]


def test_e_stays_optional():
    data = SurpyvalData(x=[2, 5, 9], c=[0, 0, 0], i=[1, 1, 1])
    assert data.e is None and data.a.tolist() == [0, 0, 1]
    # Without states, e keeps its warnings: an event with no type is a
    # masked cause
    with pytest.warns(MaskedCauseWarning, match="rows 0"):
        data = SurpyvalData(
            x=[2, 5, 9], c=[0, 0, 0], i=[1, 1, 1], e=[None, "seal", "crack"]
        )
    assert data.e.tolist() == [None, "seal", "crack"]
    assert data.a.tolist() == [0, 0, 1]


# -- refusals ----------------------------------------------------------------


def test_a_row_after_a_terminal_event_is_refused():
    with pytest.raises(
        ValueError,
        match=r"^1 row comes after a row with a = 1, which ended its "
        r"unit's history for good \(row 1 after row 0 of unit 1\)",
    ):
        SurpyvalData(x=[2, 5], c=[0, 0], i=[1, 1], a=[1, 0])


def test_rows_out_of_order_or_overlapping_are_refused():
    with pytest.raises(
        ValueError,
        match=r"^1 row overlaps the previous row of its unit or comes "
        r"before it \(row 1 \(x = 2\) after row 0 \(x = 5\) of unit 1\)",
    ):
        SurpyvalData(x=[5, 2], c=[0, 0], i=[1, 1])
    with pytest.raises(ValueError, match=r"row 1 \(x = 9, tl = 3\) after"):
        SurpyvalData(x=[4, 9], tl=[0, 3], c=[1, 0], i=[1, 1])
    # A stretch still running at 4 and a row at 4 is no stretch at all
    with pytest.raises(ValueError, match="overlaps the previous row"):
        SurpyvalData(x=[4, 4], c=[1, 0], i=[1, 1])
    with pytest.raises(ValueError, match=r"^2 rows overlap"):
        SurpyvalData(x=[5, 2, 5, 2], c=[0, 0, 0, 0], i=[1, 1, 2, 2])


def test_a_terminal_flag_on_a_running_row_is_refused():
    with pytest.raises(ValueError, match=r"^Row 1 has a = 1 .* and c = 1"):
        SurpyvalData(x=[2, 5], c=[0, 1], i=[1, 1], a=[0, 1])


def test_a_holds_flags_only():
    with pytest.raises(ValueError, match="'a' marks the row"):
        SurpyvalData(x=[2, 5], c=[0, 0], a=[2, 0])
    with pytest.raises(ValueError, match="one terminal flag per row"):
        SurpyvalData(x=[2, 5], a=1)


# -- regression guards: nothing changes without a and repeats ----------------


def _plain():
    return SurpyvalData(
        x=[1, 2, 3, 4, 5, 6],
        c=[0, 0, 1, 0, 1, 0],
        i=list("abcdef"),
        Z=[[1.0], [2.0], [1.0], [2.0], [1.0], [2.0]],
    )


def test_plain_life_data_is_unchanged():
    data = _plain()
    assert data.a is None and data.a_inferred is None
    assert repr(data) == BASELINE["plain_repr"]
    assert data.summary() == BASELINE["plain_summary"]
    assert data.to_dict() == BASELINE["plain_dict"]
    plain = SurpyvalData(
        x=[1, 2, 3, 4, 5, 6], c=[0, 0, 1, 0, 1, 0], i=list("abcdef")
    )
    _same_fit(sp.Weibull.fit(plain).params, BASELINE["weibull"])
    _same_fit(sp.WeibullPH.fit(data).params, BASELINE["weibullph"])


def test_states_data_is_unchanged():
    data = SurpyvalData(
        x=[3, 5, 7, 4],
        tl=[0, 3, 0, 0],
        s=["well", "ill", "well", "well"],
        e=["ill", "dead", None, "dead"],
        i=["A", "A", "B", "C"],
    )
    assert data.a is None
    assert repr(data) == BASELINE["s_repr"]
    assert data.summary() == BASELINE["s_summary"]
    assert data.to_dict() == BASELINE["s_dict"]


def test_states_data_takes_a_checked_only():
    data = SurpyvalData(
        x=[3, 5, 7, 4],
        tl=[0, 3, 0, 0],
        s=["well", "ill", "well", "well"],
        e=["ill", "dead", None, "dead"],
        i=["A", "A", "B", "C"],
        a=[0, 1, 0, 1],
    )
    assert data.a.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="comes after a row with a = 1"):
        SurpyvalData(
            x=[3, 5],
            tl=[0, 3],
            s=["well", "ill"],
            e=["ill", "dead"],
            i=["A", "A"],
            a=[1, 1],
        )


def test_tvc_records_are_unchanged():
    steps = StepSchedule.from_changepoints([0, 4.0], [[0.0], [1.0]])
    items = [
        ItemLifetime(
            k,
            **{"right_censored" if ck else "observed": xk},
            time_varying_covariates=steps,
        )
        for k, (xk, ck) in enumerate(
            zip([3.0, 5, 6, 8, 9, 2.5], [0, 0, 1, 0, 0, 1])
        )
    ]
    data = SurpyvalData.from_items(items)
    assert data.a is None
    assert repr(data) == BASELINE["tvc_repr"]
    assert data.summary() == BASELINE["tvc_summary"]
    assert data.to_dict() == BASELINE["tvc_dict"]
    _same_fit(sp.WeibullPH.fit(data).params, BASELINE["tvc_fit"])


def test_all_terminal_single_rows_fit_bit_identically():
    x, c = [1.0, 2, 3, 4, 5, 6], [0, 0, 1, 0, 1, 0]
    flagged = SurpyvalData(x=x, c=c, a=[1, 1, 0, 1, 0, 1])
    unflagged = SurpyvalData(x=x, c=c)
    assert (
        sp.Weibull.fit(flagged).params.tolist()
        == sp.Weibull.fit(unflagged).params.tolist()
    )
    _same_fit(sp.Weibull.fit(flagged).params, BASELINE["weibull"])


# -- carried along ---------------------------------------------------------


def test_indexing_and_reshapes_carry_a():
    data = _recurrent()
    assert data[[2, 3]].a.tolist() == [1, 0]
    assert data[[2, 3]].a_inferred.tolist() == [True, True]
    assert data.select(ids=[2]).a.tolist() == [0, 0]
    assert data.drop(ids=[1]).a.tolist() == [0, 0]
    # A row censored by a reshape has no terminal event
    cut = data.censor_at(8)
    assert cut.c.tolist() == [0, 0, 1, 0, 1]
    assert cut.a.tolist() == [0, 0, 0, 0, 0]
    assert cut.ids == [1, 1, 1, 2, 2]


def test_dataframes_carry_a():
    data = _recurrent()
    df = data.to_dataframe()
    assert df["a"].tolist() == [0, 0, 1, 0, 0]
    again = SurpyvalData.from_dataframe(
        df, x_col="x", c_col="c", i_col="id", a_col="a"
    )
    assert again.a.tolist() == data.a.tolist()
    frame = pd.DataFrame(
        {
            "unit": [1, 1, 2],
            "t": [2.0, 5, 4],
            "c": [0, 0, 0],
            "a": [0, np.nan, np.nan],
        }
    )
    read = SurpyvalData.from_dataframe(
        frame, x_col="t", c_col="c", i_col="unit", a_col="a"
    )
    assert read.a.tolist() == [0, 1, 1]
    assert read.a_inferred.tolist() == [False, True, True]


def test_dict_and_json_carry_a_with_the_schema_stamp():
    data = _recurrent()
    d = data.to_dict()
    assert d["a"] == [0, 0, 1, 0, 0]
    assert d["a_inferred"] == [True, True, True, True, False]
    assert d["schema"] == 4
    again = SurpyvalData.from_json(data.to_json())
    assert again.a.tolist() == data.a.tolist()
    assert again.a_inferred.tolist() == data.a_inferred.tolist()
    assert again.ids == data.ids


def test_repr_and_summary():
    data = _recurrent()
    assert repr(data).splitlines()[:3] == [
        "SurpyvalData: 2 items, 5 rows",
        "  id    x  c  n  a",
        "   1  2.0  0  1  0",
    ]
    assert data.summary().splitlines()[2] == (
        "  units: 2 (1 ended for good, 1 still running)"
    )


def test_records_refuse_several_rows_per_unit():
    data = _recurrent()
    for call in (data.to_items, lambda: list(data.records())):
        with pytest.raises(ValueError, match="several rows"):
            call()
    with pytest.raises(ValueError, match="several rows"):
        data.item(1)
    with pytest.raises(ValueError, match="terminal flags a"):
        data.update(ItemLifetime(1, observed=3.0))
    # Single rows whose a says what c does: records as before
    single = SurpyvalData(x=[2, 5], c=[0, 1], i=["p", "q"], a=[1, 0])
    assert single.item("p") == ItemLifetime("p", observed=2.0)


# -- fits ------------------------------------------------------------------


def test_fits_refuse_several_rows_per_unit_saying_what_they_are():
    with pytest.raises(
        ValueError,
        match=r"^This fit takes no covariates, which the SurpyvalData "
        r"holds, and takes one row per unit \(units 'A', 'C' have several: "
        r"stretches with the covariates fixed in each; the model's fit_tvc",
    ):
        sp.Weibull.fit(_tvc())
    with pytest.raises(ValueError, match=r"^This fit takes one row per unit"):
        sp.WeibullPH.fit(_tvc())
    with pytest.raises(
        ValueError,
        match=r"^This fit takes one row per unit \(units 1, 2 have several: "
        r"repeated events, a = 0 where the unit carried on; the recurrent "
        r"models in surpyval.recurrent take them\)\.$",
    ):
        sp.Weibull.fit(_recurrent())
    for call in (
        sp.KaplanMeier.fit,
        sp.Weibull.fit_from_surpyval_data,
        lambda d: d.to_xrd(),
        lambda d: sp.life_table(d, intervals=[0, 5, 10]),
    ):
        with pytest.raises(ValueError, match="takes one row per unit"):
            call(_recurrent())


def test_a_fit_with_an_a_argument_is_given_the_flags_and_units():
    from surpyval.utils.surpyval_data import accepts_surpyval_data

    seen = {}

    class Fitter:
        @accepts_surpyval_data
        def fit(self, x, c=None, n=None, i=None, a=None):
            seen.update(x=x, c=c, i=i, a=a)
            return None

    Fitter().fit(_recurrent())
    assert seen["a"].tolist() == [0, 0, 1, 0, 0]
    assert seen["i"] == [1, 1, 1, 2, 2]
