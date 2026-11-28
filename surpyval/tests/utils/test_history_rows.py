"""Histories of repeated events as a SurpyvalData with units ``i``: built
from ItemHistory records and from the ``{item: ages}`` dicts, reshaped as
first events, gaps and dicts, and converted from an EventHistoryData. Each
is checked against the EventHistoryData route it stands in for, against
hand-worked values, and by the fits giving the same numbers."""

import math
import warnings

import numpy as np
import pytest

import surpyval
import surpyval.recurrent as R
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData
from surpyval.utils.warnings import MaskedCauseWarning

H = ItemHistory

CASES = {
    "plain": [
        H("A", [5, 9], right_censored=12),
        H("B", [4], right_censored=10),
        H("C", right_censored=8),
    ],
    "no end of observation": [H("A", [5, 9]), H("B", [4], right_censored=10)],
    "late entry": [
        H("A", [5, 9], right_censored=12, left_truncation=2),
        H("B", [4], right_censored=10),
    ],
    "right truncation": [
        H("A", [5, 9], right_truncation=20),
        H("B", [4], right_truncation=20),
    ],
    "counts at inspections": [
        H(
            "A",
            [5],
            interval_censored=[[6, 8], [1, 3]],
            interval_censored_counts=[2, 1],
            right_censored=12,
        ),
        H(
            "B",
            [4],
            left_censored=3,
            left_censored_counts=2,
            right_censored=10,
        ),
    ],
    "event types": [
        H("A", [5, 9], right_censored=12, observed_event_types=["x", "y"]),
        H("B", [4], right_censored=10, observed_event_types="x"),
    ],
    "named covariates": [
        H("A", [5, 9], right_censored=12, covariates={"t": 1.0, "p": 2.0}),
        H("B", [4], right_censored=10, covariates={"t": 0.0, "p": 3.0}),
    ],
    "unnamed covariates": [
        H("A", [5, 9], right_censored=12, covariates=[1.0, 2.0]),
        H("B", [4], right_censored=10, covariates=[0.0, 3.0]),
    ],
    "attributes": [
        H("A", [5, 9], right_censored=12, attributes={"site": "north"}),
        H("B", [4], right_censored=10),
    ],
    "two failures at one age": [
        H("A", [5, 5, 9], right_censored=12),
        H("B", [4], right_censored=10),
    ],
    "unit": [
        H("A", [5, 9], right_censored=12, unit="h"),
        H("B", [4], right_censored=10, unit="h"),
    ],
    "number ids": [
        H(1, [4], right_censored=10),
        H(2, [5, 9], right_censored=12),
    ],
    "one row each": [H("A", [5]), H("B", right_censored=10)],
    "time-varying covariates": [
        H(
            "A",
            [5, 9],
            right_censored=12,
            time_varying_covariates={0: [1.0], 6: [2.0]},
        ),
        H(
            "B",
            [4],
            right_censored=10,
            time_varying_covariates={0: [1.0]},
        ),
    ],
}


def _outcome(call):
    """What a call gives, or the refusal it raises."""
    try:
        return call()
    except (TypeError, ValueError) as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize("name", CASES)
def test_rows_are_the_histories(name):
    """``SurpyvalData.from_items(histories)`` is the data the
    EventHistoryData of the same records converts to, reads back as that
    EventHistoryData, and reshapes as it does."""
    histories = CASES[name]
    history = EventHistoryData.from_items(histories)
    rows = SurpyvalData.from_items(histories)
    assert rows.i == [
        h.id for h in histories for _ in range(len(h._rows()[0]))
    ]
    assert history.to_surpyval_data() == rows
    assert rows._event_history("test") == history
    for reshape in ("to_first_events", "to_gaps"):
        expected = _outcome(getattr(history, reshape))
        got = _outcome(getattr(rows, reshape))
        if isinstance(expected, tuple):
            # The same refusal
            assert got == expected
        else:
            assert got == expected
    expected = _outcome(history.to_histories)
    got = _outcome(rows.to_histories)
    assert repr(got) == repr(expected)


def test_rows_by_hand():
    """One row per event in time order, the end of observation last, the
    late entry on the first row only, counts at inspections on one row."""
    data = SurpyvalData.from_items(
        [
            H(
                "A",
                [5],
                interval_censored=[[6, 8], [1, 3]],
                interval_censored_counts=[2, 1],
                right_censored=12,
                left_truncation=0.5,
            ),
            H("B", [4, 4], right_censored=10),
        ]
    )
    assert data.x.tolist() == [
        [1, 3],
        [5, 5],
        [6, 8],
        [12, 12],
        [4, 4],
        [4, 4],
        [10, 10],
    ]
    assert data.c.tolist() == [2, 0, 2, 1, 0, 0, 1]
    assert data.n.tolist() == [1, 1, 2, 1, 1, 1, 1]
    assert data.i == ["A"] * 4 + ["B"] * 3
    assert data.tl.tolist() == [0.5] + [-np.inf] * 6
    assert data.a.tolist() == [0] * 7
    assert data.n_items == 2


def test_from_histories_by_hand():
    data = SurpyvalData.from_histories(
        observed={"M1": [420, 1310, 2050], "M2": [980, 2400]},
        right_censored={"M1": 2600, "M2": 2900, "M3": 3100},
        left_truncation={"M2": 100},
        covariates={"M1": {"z": 1}, "M2": {"z": 0}, "M3": {"z": 2}},
        unit="hours",
    )
    assert data.x.tolist() == [420, 1310, 2050, 2600, 980, 2400, 2900, 3100]
    assert data.c.tolist() == [0, 0, 0, 1, 0, 0, 1, 1]
    assert data.i == ["M1"] * 4 + ["M2"] * 3 + ["M3"]
    assert data.tl.tolist() == [-np.inf] * 4 + [100] + [-np.inf] * 3
    assert data.Z[:, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 2]
    assert data.covariate_names == ["z"]
    assert data.unit == "hours"
    # The times between failures are the same histories
    again = SurpyvalData.from_histories(
        observed={"M1": [420, 890, 740], "M2": [880, 1420]},
        right_censored={"M1": 550, "M2": 500, "M3": 3100},
        left_truncation={"M2": 100},
        covariates={"M1": {"z": 1}, "M2": {"z": 0}, "M3": {"z": 2}},
        unit="hours",
        gaps=True,
    )
    assert again == data
    # And the dicts come back
    back = data.to_histories()
    assert {k: v.tolist() for k, v in back["observed"].items()} == {
        "M1": [420, 1310, 2050],
        "M2": [980, 2400],
        "M3": [],
    }
    assert back["right_censored"] == {"M1": 2600, "M2": 2900, "M3": 3100}
    assert back["left_truncation"] == {"M2": 100}
    assert SurpyvalData.from_histories(**back, unit="hours") == data


def test_from_histories_is_the_event_history_datas():
    arguments = dict(
        observed={1: [1, 2], 2: [1, 2, 3, 4]},
        right_censored={1: 5, 2: 5},
        right_truncation=9,
    )
    assert EventHistoryData.from_histories(
        **arguments
    ).to_surpyval_data() == SurpyvalData.from_histories(**arguments)
    with pytest.raises(TypeError, match=r"SurpyvalData.from_histories\(\)"):
        SurpyvalData.from_histories(failures={1: [2]})
    with pytest.raises(ValueError, match="Give each item's failures"):
        SurpyvalData.from_histories()


def test_first_events_and_gaps_by_hand():
    """A fails at 5 and 9 and is watched to 12 (from 2); B fails at 4,
    watched to 10; C never fails, watched to 8. First failures: 5, 4 and
    8+. Gaps: A 5 (seen from 2), 4, 3+; B 4, 6+; C 8+."""
    data = SurpyvalData(
        [5, 9, 12, 4, 10, 8],
        [0, 0, 1, 0, 1, 1],
        i=["A", "A", "A", "B", "B", "C"],
        tl=[2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    )
    first = data.to_first_events()
    assert first.x.tolist() == [5, 4, 8]
    assert first.c.tolist() == [0, 0, 1]
    assert first.tl.tolist() == [2, -np.inf, -np.inf]
    assert first.ids == ["A", "B", "C"]
    gaps = data.to_gaps()
    assert gaps.x.tolist() == [5, 4, 3, 4, 6, 8]
    assert gaps.c.tolist() == [0, 0, 1, 0, 1, 1]
    assert gaps.tl.tolist() == [2] + [-np.inf] * 5
    assert gaps.ids == [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)] + [
        ("C", 1)
    ]


def test_terminal_event_ends_the_history():
    """A unit ended by a terminal event (``a = 1``, no ``c = 1`` row) is
    read as watched until that event, as a recurrent fit reads it."""
    data = SurpyvalData(
        [5, 9, 4, 10], [0, 0, 0, 1], i=["A", "A", "B", "B"], a=[0, 1, 0, 0]
    )
    assert data.to_first_events().x.tolist() == [5, 4]
    gaps = data.to_gaps()
    assert gaps.x.tolist() == [5, 4, 4, 6]
    assert gaps.c.tolist() == [0, 0, 0, 1]


def test_crow_amsaa_closed_form():
    """Two systems watched to T = 12 with failures at 5, 9 and 4: the
    power-law MLE is beta = N / sum(ln(T / t)) = 3 / ln(12^3 / 180) and,
    with the intensity (beta / alpha) (t / alpha)^(beta - 1),
    alpha = T / (N / 2)^(1 / beta)."""
    histories = [
        H("A", [5, 9], right_censored=12),
        H("B", [4], right_censored=12),
    ]
    beta = 3 / math.log(12**3 / (5 * 9 * 4))
    alpha = 12 / 1.5 ** (1 / beta)
    for data in (
        SurpyvalData.from_items(histories),
        SurpyvalData.from_histories(
            observed={"A": [5, 9], "B": [4]},
            right_censored={"A": 12, "B": 12},
        ),
        EventHistoryData.from_items(histories).to_surpyval_data(),
    ):
        model = R.CrowAMSAA.fit(data)
        assert model.params == pytest.approx([alpha, beta], rel=1e-6)
    # Homogeneous: 3 failures in 24 unit-hours
    assert R.HPP.fit(SurpyvalData.from_items(histories)).params[
        0
    ] == pytest.approx(3 / 24, rel=1e-9)


FITS: list = [
    ("plain", R.CrowAMSAA, {}),
    ("late entry", R.CrowAMSAA, {}),
    ("no end of observation", R.HPP, {}),
    ("counts at inspections", R.CrowAMSAA, {}),
    ("two failures at one age", R.Duane, {}),
]


@pytest.mark.parametrize("name, fitter, kwargs", FITS)
def test_fit_of_the_rows_is_the_fit_of_the_history(name, fitter, kwargs):
    history = EventHistoryData.from_items(CASES[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = fitter.fit(history, **kwargs)
        got = fitter.fit(history.to_surpyval_data(), **kwargs)
    np.testing.assert_array_equal(got.params, expected.params)


def test_mcf_of_the_rows_is_the_mcf_of_the_history():
    history = EventHistoryData.from_items(CASES["late entry"])
    expected = R.NonParametricCounting.fit(history)
    got = R.NonParametricCounting.fit(history.to_surpyval_data())
    np.testing.assert_array_equal(got.x, expected.x)
    np.testing.assert_array_equal(got.mcf_hat, expected.mcf_hat)


def test_cause_specific_and_covariate_fits_agree():
    typed = EventHistoryData.from_items(CASES["event types"])
    expected = R.CauseSpecificMCF.fit(typed)
    got = R.CauseSpecificMCF.fit(typed.to_surpyval_data())
    assert sorted(got.models) == sorted(expected.models)
    for cause in expected.models:
        np.testing.assert_array_equal(
            got.models[cause].mcf_hat, expected.models[cause].mcf_hat
        )
    rng = np.random.default_rng(0)
    histories = []
    for k in range(30):
        z = float(k % 2)
        ages = np.cumsum(rng.exponential(1 / (0.5 * math.exp(0.7 * z)), 6))
        histories.append(
            H(k, ages[ages < 8], right_censored=8, covariates={"z": z})
        )
    history = EventHistoryData.from_items(histories)
    expected = R.ProportionalIntensityHPP.fit(history)
    got = R.ProportionalIntensityHPP.fit(history.to_surpyval_data())
    np.testing.assert_array_equal(got.params, expected.params)


def test_observation_windows_become_gaps():
    """An item watched over (0, 3] and (5, 9]: each stretch starts with
    its ``tl`` and ends with a ``c = 1`` row, and the fit of the rows is
    the fit of the arrays with ``windows``."""
    arrays = dict(x=[1, 2, 7, 8], i=[1, 1, 1, 2], c=[0, 0, 0, 0])
    windows = {1: [(0, 3), (5, 9)], 2: [(0, 10)]}
    history = surpyval.handle_xicn(**arrays, windows=windows)
    rows = history.to_surpyval_data()
    assert rows.x.tolist() == [1, 2, 3, 7, 9, 8, 10]
    assert rows.c.tolist() == [0, 0, 1, 0, 1, 0, 1]
    assert rows.i == [1, 1, 1, 1, 1, 2, 2]
    assert rows.tl.tolist() == [0, -np.inf, -np.inf, 5, -np.inf, 0, -np.inf]
    assert rows._event_history("test") == history
    expected = R.CrowAMSAA.fit(**arrays, windows=windows)
    np.testing.assert_array_equal(
        R.CrowAMSAA.fit(rows).params, expected.params
    )
    # A window with no failure in it is still a stretch of watching, and
    # so are the windows of an item that never failed
    quiet = surpyval.handle_xicn(
        [1], [1], [0], windows={1: [(0, 3), (5, 9)], 2: [(0, 4)]}
    )
    rows = quiet.to_surpyval_data()
    assert rows.x.tolist() == [1, 3, 9, 4]
    assert rows.c.tolist() == [0, 1, 1, 1]
    assert rows.i == [1, 1, 1, 2]
    assert rows.tl.tolist() == [0, -np.inf, 5, 0]
    assert rows._event_history("test") == quiet
    # One failure in 3 + 4 + 4 hours of watching
    assert R.HPP.fit(rows).params[0] == pytest.approx(1 / 11, rel=1e-9)


def test_simulated_data_as_rows():
    model = R.CrowAMSAA.fit(SurpyvalData.from_items(CASES["plain"]))
    simulated = model.count_terminated_simulation_data(
        5, items=4, random_state=3
    )
    rows = simulated.to_surpyval_data()
    assert isinstance(rows, SurpyvalData)
    assert rows.n_items == 4
    np.testing.assert_array_equal(
        R.CrowAMSAA.fit(rows).params, R.CrowAMSAA.fit(simulated).params
    )
    assert model.data.to_surpyval_data() == SurpyvalData.from_items(
        CASES["plain"]
    )


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match="has no units.*i=i"):
        SurpyvalData([1, 2, 3]).to_gaps()
    with pytest.raises(ValueError, match="weights .build the data without w"):
        SurpyvalData([1, 2, 3], i=[1, 1, 2], w=[1, 2, 1]).to_first_events()
    varying = SurpyvalData(
        [1, 2, 3], [0, 1, 1], i=[1, 1, 2], Z=[[0.0], [1.0], [0.0]]
    )
    with pytest.raises(ValueError, match="covariates of units 1 change"):
        varying.to_gaps()
    with pytest.raises(TypeError, match="a list holds records of one kind"):
        SurpyvalData.from_items([H("A", [5]), ItemLifetime("B", observed=3)])
    with pytest.raises(ValueError, match="leave counts out"):
        SurpyvalData.from_items([H("A", [5])], counts=[2])
    states = EventHistoryData.from_items(
        [
            H(
                "A",
                [5],
                right_censored=9,
                initial_state="up",
                observed_states={6: "down"},
            )
        ]
    )
    with pytest.raises(ValueError, match="changes of state .items 'A'."):
        states.to_surpyval_data()
    with pytest.raises(ValueError, match="changes of state"):
        SurpyvalData.from_items(states.to_items())


def test_masked_causes_warn_once_by_item():
    histories = [
        H("A", [5, 9], right_censored=12, observed_event_types=["x", None]),
        H("B", [4], right_censored=10, observed_event_types="x"),
    ]
    with pytest.warns(MaskedCauseWarning, match="items 'A'") as seen:
        rows = SurpyvalData.from_items(histories)
    assert len(seen) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = EventHistoryData.from_items(histories)
    with warnings.catch_warnings():
        # The data warned when it was built: converting it does not again
        warnings.simplefilter("error")
        assert history.to_surpyval_data() == rows


def test_event_log_from_a_dataframe():
    """A log with one row per failure and each unit's end of observation
    (``ends=`` or ``end_col=``) is the EventHistoryData's reading of it,
    as rows; a unit named only in ``ends`` never failed."""
    import pandas as pd

    log = pd.DataFrame(
        {
            "machine": ["M1", "M1", "M2"],
            "hours": [1310, 420, 980],
            "part": ["bearing", "seal", "seal"],
            "end": [2600, 2600, 2900],
        }
    )
    columns = dict(i_col="machine", x_col="hours", e_col="part")
    ends = {"M1": 2600, "M2": 2900, "M3": 3100}
    data = SurpyvalData.from_dataframe(log, **columns, ends=ends)
    assert data.i == ["M1", "M1", "M1", "M2", "M2", "M3"]
    assert data.x.tolist() == [420, 1310, 2600, 980, 2900, 3100]
    assert data.c.tolist() == [0, 0, 1, 0, 1, 1]
    assert data.e.tolist() == ["seal", "bearing", None, "seal", None, None]
    history = EventHistoryData.from_dataframe(log, **columns, ends=ends)
    assert history.to_surpyval_data() == data
    by_column = SurpyvalData.from_dataframe(log, **columns, end_col="end")
    assert by_column.x.tolist() == [420, 1310, 2600, 980, 2900]
    # 3 failures in 2600 + 2900 + 3100 hours
    assert R.HPP.fit(data.without("event_types")).params[0] == pytest.approx(
        3 / 8600, rel=1e-9
    )
    with pytest.raises(ValueError, match="name the column of unit ids, i_col"):
        SurpyvalData.from_dataframe(log, x_col="hours", ends=ends)
    with pytest.raises(ValueError, match="no place for w_col.*c = 1"):
        SurpyvalData.from_dataframe(log, **columns, ends=ends, w_col="end")


def test_from_items_takes_any_run_of_histories():
    histories = CASES["plain"]
    expected = SurpyvalData.from_items(histories)
    assert SurpyvalData.from_items(tuple(histories)) == expected
    assert SurpyvalData.from_items(h for h in histories) == expected
    with pytest.raises(TypeError, match=r"got one ItemHistory: give \[item\]"):
        SurpyvalData.from_items(histories[0])


def test_failures_before_a_first_look_only():
    """Left censored counts without intervals: the rows keep one age per
    row and read back as the same history."""
    histories = [
        H("A", [6], left_censored=3, left_censored_counts=2, right_censored=9),
        H("B", right_censored=7),
    ]
    rows = SurpyvalData.from_items(histories)
    assert rows.c.tolist() == [-1, 0, 1, 1]
    assert rows.n.tolist() == [2, 1, 1, 1]
    history = EventHistoryData.from_items(histories)
    assert rows._event_history("test") == history
    assert history.to_surpyval_data() == rows
    # 3 failures in 9 + 7 hours
    assert R.HPP.fit(history).params[0] == pytest.approx(3 / 16, rel=1e-9)
    assert R.HPP.fit(rows).params[0] == pytest.approx(3 / 16, rel=1e-9)


def _round_trip(history, fitter=None, same=True):
    """The history as rows reads back as the same history, and the fit of
    either has the same numbers."""
    rows = history.to_surpyval_data()
    if same:
        assert rows._event_history("test") == history
    if fitter is not None:
        np.testing.assert_array_equal(
            fitter.fit(rows).params, fitter.fit(history).params
        )
    return rows


def test_inspections_that_found_nothing_hold_as_rows():
    """Item 1 was looked at over (0, 3] and (3, 6] and found 2 events then
    none; item 2 found none over (0, 2] and 3 over (2, 6]. The counts of 0
    are information (the item was looked at and failed no time), kept as
    rows with n = 0."""
    arrays = dict(
        x=[[0, 3], [3, 6], [6, 6], [0, 2], [2, 6], [6, 6]],
        i=[1, 1, 1, 2, 2, 2],
        c=[2, 2, 1, 2, 2, 1],
        n=[2, 0, 1, 0, 3, 1],
    )
    history = surpyval.handle_xicn(**arrays)
    rows = _round_trip(history, R.CrowAMSAA)
    assert rows.n.tolist() == [2, 0, 1, 0, 3, 1]
    assert rows.c.tolist() == [2, 2, 1, 2, 2, 1]
    assert rows.i == [1, 1, 1, 2, 2, 2]
    assert rows.n_items == 2
    # The fit of the rows is the fit of the arrays
    np.testing.assert_array_equal(
        R.CrowAMSAA.fit(rows).params, R.CrowAMSAA.fit(**arrays).params
    )
    # And the zero counts change the fit: they are real information
    without = surpyval.handle_xicn(
        x=[[0, 3], [6, 6], [2, 6], [6, 6]],
        i=[1, 1, 2, 2],
        c=[2, 1, 2, 1],
        n=[2, 1, 3, 1],
    )
    assert not np.allclose(
        R.CrowAMSAA.fit(rows).params, R.CrowAMSAA.fit(without).params
    )


def test_zero_counts_with_late_entry_covariates_and_types():
    arrays = dict(
        x=[[1, 3], [3, 6], [6, 6], [0, 2], [2, 6], [6, 6]],
        i=["a", "a", "a", "b", "b", "b"],
        c=[2, 2, 1, 2, 2, 1],
        n=[0, 2, 1, 1, 0, 1],
        tl=[1, 1, 1, 0, 0, 0],
        Z=[[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]],
    )
    history = surpyval.handle_xicn(**arrays)
    rows = _round_trip(history)
    assert rows.tl.tolist() == [1, -np.inf, -np.inf, 0, -np.inf, -np.inf]
    assert rows.Z.ravel().tolist() == [1, 1, 1, 2, 2, 2]
    assert rows.n.tolist() == [0, 2, 1, 1, 0, 1]


def test_a_count_of_zero_stays_an_error_for_lifetimes():
    # No units: a lifetime row, a count of 0 is nothing
    with pytest.raises(ValueError, match="can't be 0 or less"):
        SurpyvalData([[0, 3], [1, 4]], c=[2, 2], n=[0, 1])
    # With units, only on a counted row (an inspection)
    with pytest.raises(ValueError, match="can't be 0 or less"):
        SurpyvalData([1, 2, 3], c=[0, 0, 1], n=[0, 1, 1], i=[1, 1, 1])
    with pytest.raises(ValueError, match="can't be 0 or less"):
        SurpyvalData([1, 2, 3], c=[0, 0, 1], n=[1, 1, 0], i=[1, 1, 1])
    with pytest.raises(ValueError, match="can't be 0 or less"):
        SurpyvalData([1], c=[0], n=[-1], i=[1])


def test_counts_before_a_first_look_given_as_one_age_each():
    """Failures before a first look, given as ``x = age, c = -1`` (the way
    the fits are given them), come back as the same ages."""
    arrays = dict(
        x=[3, 4, 8, 2, 7],
        i=[1, 1, 1, 2, 2],
        c=[-1, 0, 1, -1, 1],
        n=[2, 1, 1, 1, 1],
    )
    history = surpyval.handle_xicn(**arrays)
    assert history.x.ndim == 1
    # (The rows read back as [age, age] pairs, the same ages)
    rows = _round_trip(history, R.CrowAMSAA, same=False)
    assert rows.c.tolist() == [-1, 0, 1, -1, 1]
    assert rows.n.tolist() == [2, 1, 1, 1, 1]
    assert rows.x.tolist() == [3, 4, 8, 2, 7]
    np.testing.assert_array_equal(
        R.CrowAMSAA.fit(rows).params, R.CrowAMSAA.fit(**arrays).params
    )
    # Given as [age, age] pairs they stay pairs
    pairs = surpyval.handle_xicn(
        **{**arrays, "x": [[v, v] for v in arrays["x"]]}
    )
    assert pairs.to_surpyval_data()._event_history("test") == pairs


def test_a_unit_with_repeated_events_is_one_item():
    """Counts at inspections are events, not units: one unit that had 2
    then 3 failures between inspections is one item."""
    rows = SurpyvalData(
        [[0, 3], [3, 6], [6, 6]], c=[2, 2, 1], n=[2, 3, 1], i=["u"] * 3
    )
    assert rows.n_items == 1
    assert "1 item," in repr(rows)
    assert "units: 1 " in rows.summary()
    # A unit with one row is counted by its n (identical items)
    assert SurpyvalData([1, 2], c=[0, 0], n=[2, 5], i=["a", "b"]).n_items == 7
    two = SurpyvalData(
        [[0, 3], [3, 3], [0, 4], [4, 4]],
        c=[2, 1, 2, 1],
        n=[4, 1, 2, 1],
        i=["u", "u", "v", "v"],
    )
    assert two.n_items == 2


def test_watched_stretch_with_no_failure_as_the_fits_build_it():
    arrays = dict(x=[1, 6], i=[1, 1], c=[0, 0])
    windows = {1: [(0, 3), (5, 9)], 2: [(0, 4)], 3: [(2, 8)]}
    history = surpyval.handle_xicn(**arrays, windows=windows)
    rows = _round_trip(history)
    assert rows.i == [1, 1, 1, 1, 2, 3]
    assert rows.c.tolist() == [0, 1, 0, 1, 1, 1]
    assert rows.tl.tolist() == [0, -np.inf, 5, -np.inf, 0, 2]
    np.testing.assert_array_equal(
        R.CrowAMSAA.fit(rows).params,
        R.CrowAMSAA.fit(**arrays, windows=windows).params,
    )


def test_an_event_at_the_right_truncation_age_converts():
    arrays = dict(x=[5, 10, 3, 7], i=[1, 1, 2, 2], c=[0, 0, 0, 0], tr=10)
    history = surpyval.handle_xicn(**arrays)
    rows = _round_trip(history, R.CrowAMSAA)
    assert rows.x.tolist() == [5, 10, 3, 7]
    assert rows.tr.tolist() == [10.0] * 4
