"""DataFrames in and out of the data classes (#936): from_dataframe on
both datasets, to_dataframe, and the round trips."""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

FLEET = pd.DataFrame(
    {
        "unit": ["P1", "P2", "P3", "P4"],
        "hours": [820, 2000, 1100, 1700],
        "failed": [True, False, True, False],
        "mode": ["seal", None, "bearing", None],
        "load": [0.8, 0.4, 0.9, 0.6],
        "site": ["N", "N", "S", "S"],
        "installed": [100, 0, 0, 50],
    }
)


def test_one_row_per_item_with_a_status_column():
    data = SurpyvalData.from_dataframe(
        FLEET,
        i_col="unit",
        x_col="hours",
        event_col="failed",
        e_col="mode",
        Z_cols=["load"],
        tl_col="installed",
    )
    assert data.ids == ["P1", "P2", "P3", "P4"]
    assert data.c.tolist() == [0, 1, 0, 1]
    assert list(data.e) == ["seal", None, "bearing", None]
    assert data.covariate_names == ["load"]
    assert data.tl.tolist() == [100.0, 0.0, 0.0, 50.0]
    assert data.item("P1") == ItemLifetime(
        "P1",
        observed=820,
        event_type="seal",
        covariates={"load": 0.8},
        left_truncation=100,
    )
    # The same data as the fitter's own DataFrame entry point
    np.testing.assert_allclose(
        sp.WeibullPH.fit(data.without("event_types")).params,
        sp.WeibullPH.fit_from_df(
            FLEET.assign(c=1 - FLEET.failed),
            x_col="hours",
            c_col="c",
            Z_cols=["load"],
            tl_col="installed",
        ).params,
    )


def test_c_col_and_counts_and_a_number_for_truncation():
    df = pd.DataFrame({"t": [5, 9, 12], "c": [0, 1, 2], "k": [1, 40, 2]})
    with pytest.raises(ValueError, match="Censoring value must only be"):
        SurpyvalData.from_dataframe(df, x_col="t", c_col="c", n_col="k")
    df["c"] = [0, 1, 0]
    data = SurpyvalData.from_dataframe(
        df, x_col="t", c_col="c", n_col="k", tl_col=1.0
    )
    assert data.n.tolist() == [1, 40, 2] and data.tl.tolist() == [1.0] * 3


def test_last_ok_and_first_failed_with_blanks():
    df = pd.DataFrame(
        {
            "last_ok": [1000, 2400, None, 500],
            "first_failed": [1500, None, 800, 500],
        }
    )
    data = SurpyvalData.from_dataframe(
        df, xl_col="last_ok", xr_col="first_failed"
    )
    # Sorted by time, as data without ids is
    assert data.c.tolist() == [0, -1, 2, 1]
    # Flags given with the bounds must agree with them
    with pytest.raises(ValueError, match="not interval censored but has"):
        SurpyvalData.from_dataframe(
            df.assign(c=0), xl_col="last_ok", xr_col="first_failed", c_col="c"
        )
    blank = df.assign(last_ok=[1000, None, None, 500])
    with pytest.raises(ValueError, match="Row 1 have both .* blank"):
        SurpyvalData.from_dataframe(
            blank, xl_col="last_ok", xr_col="first_failed"
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"x_col": "hours", "c_col": "failed", "event_col": "failed"},
            "not both",
        ),
        ({"x_col": "hours", "event_col": "mode"}, "has a blank cell"),
        ({"x_col": "hours", "event_col": "site"}, "must hold 1 or True"),
        ({}, "Give the times"),
        ({"x_col": "hours", "Z_cols": ["site"]}, "'site' is not numeric"),
        ({"x_col": "hours", "Z_cols": ["nope"]}, "not a column"),
        ({"x_col": "nope"}, "x_col='nope' is not a column"),
    ],
)
def test_frame_inputs_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SurpyvalData.from_dataframe(FLEET, **kwargs)


def test_blank_times_and_dates_are_refused():
    df = FLEET.assign(hours=[820, None, 1100, 1700])
    with pytest.raises(ValueError, match="blank on row 1: every item needs"):
        SurpyvalData.from_dataframe(df, x_col="hours")
    dated = FLEET.assign(when=pd.to_datetime(["2024-01-01"] * 4))
    with pytest.raises(ValueError, match="holds dates"):
        SurpyvalData.from_dataframe(dated, x_col="when")


def test_lifetime_round_trip_through_a_frame():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=5, event_type="x", covariates={"z": 1}),
            ItemLifetime("B", interval_censored=[2, 4], covariates={"z": 2}),
            ItemLifetime(
                "C", right_censored=9, covariates={"z": 3}, right_truncation=20
            ),
        ],
        counts=[1, 2, 3],
    )
    frame = data.to_dataframe()
    assert list(frame.columns) == ["id", "xl", "xr", "c", "n", "tr", "e", "z"]
    with pytest.warns(UserWarning, match="masked"):
        again = SurpyvalData.from_dataframe(
            frame,
            i_col="id",
            xl_col="xl",
            xr_col="xr",
            c_col="c",
            n_col="n",
            tr_col="tr",
            e_col="e",
            Z_cols=["z"],
        )
    assert again.to_items() == data.to_items()
    assert again.n.tolist() == [1, 2, 3]
    plain = SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1])
    assert list(plain.to_dataframe().columns) == ["x", "c", "n"]
    assert plain.to_dataframe()["n"].tolist() == [1, 2, 1]
    steps = SurpyvalData.from_items(
        [ItemLifetime("A", observed=5, time_varying_covariates={0: [1.0]})]
    )
    with pytest.raises(ValueError, match="time-varying covariates, which a"):
        steps.to_dataframe()


LOG = pd.DataFrame(
    {
        "machine": ["M1", "M1", "M1", "M2", "M2"],
        "hours": [420, 1310, 2050, 980, 2400],
        "part": ["seal", "bearing", "seal", "seal", None],
        "load": [1.0, 1.0, 1.0, 0.5, 0.5],
        "plant": ["N", "N", "N", "S", "S"],
    }
)


def test_a_log_with_ends_per_item():
    with pytest.raises(ValueError, match="'M3' is in ends= but has no row"):
        EventHistoryData.from_dataframe(
            LOG,
            i_col="machine",
            x_col="hours",
            Z_cols=["load"],
            ends={"M1": 2600, "M2": 2900, "M3": 3100},
        )
    with pytest.warns(UserWarning, match="masked"):
        data = EventHistoryData.from_dataframe(
            LOG,
            i_col="machine",
            x_col="hours",
            e_col="part",
            Z_cols=["load"],
            ends={"M1": 2600, "M2": 2900},
        )
    assert data.ids == ["M1", "M2"]
    with_m3 = EventHistoryData.from_dataframe(
        LOG,
        i_col="machine",
        x_col="hours",
        ends={"M1": 2600, "M2": 2900, "M3": 3100},
    )
    assert with_m3.ids == ["M1", "M2", "M3"]
    assert with_m3.item("M3") == ItemHistory("M3", right_censored=3100)
    assert data.item("M1").right_censored == 2600
    assert list(data.item("M1").observed_event_types) == [
        "seal",
        "bearing",
        "seal",
    ]
    assert data.covariate_names == ["load"]
    # The same data as the fitter's own DataFrame entry point
    log = LOG.assign(c=0)
    ends = pd.DataFrame(
        {"machine": ["M1", "M2"], "hours": [2600, 2900], "c": 1}
    )
    both = pd.concat([log[["machine", "hours", "c"]], ends])
    from_frame = EventHistoryData.from_dataframe(
        LOG.iloc[:, :2],
        i_col="machine",
        x_col="hours",
        ends={"M1": 2600, "M2": 2900},
    )
    np.testing.assert_allclose(
        sp.recurrent.CrowAMSAA.fit(from_frame).params,
        sp.recurrent.CrowAMSAA.fit_from_df(
            both, x_col="hours", i_col="machine", c_col="c"
        ).params,
    )


def test_a_log_with_end_rows_counts_and_an_end_column():
    log = pd.DataFrame(
        {
            "i": ["A", "A", "A", "B", "B"],
            "lo": [5, 10, 12, 0, 8],
            "hi": [5, 12, 12, 8, 8],
            "c": [0, 2, 1, 2, 1],
            "n": [1, 3, 1, 2, 1],
        }
    )
    data = EventHistoryData.from_dataframe(
        log, i_col="i", xl_col="lo", xr_col="hi", c_col="c", n_col="n"
    )
    assert data.item("A") == ItemHistory(
        "A",
        observed=[5],
        interval_censored=[[10, 12]],
        interval_censored_counts=3,
        right_censored=12,
    )
    assert data.item("B").left_censored is None
    with_end = pd.DataFrame(
        {"i": ["A", "A", "B"], "t": [5, 9, 4], "end": [12, None, 10]}
    )
    data = EventHistoryData.from_dataframe(
        with_end, i_col="i", x_col="t", end_col="end"
    )
    assert data.item("A").right_censored == 12
    assert data.item("B").right_censored == 10


@pytest.mark.parametrize(
    "frame, kwargs, match",
    [
        (
            LOG.assign(load=[1.0, 2.0, 1.0, 0.5, 0.5]),
            {"x_col": "hours", "Z_cols": ["load"]},
            "Item 'M1': the covariates differs between its rows",
        ),
        (
            LOG.assign(c=[0, 1, 1, 0, 0]),
            {"x_col": "hours", "c_col": "c"},
            "Item 'M1' has 2 end-of-observation rows",
        ),
        (
            LOG.assign(c=[0, 0, 7, 0, 0]),
            {"x_col": "hours", "c_col": "c"},
            "c_col='c' must hold 0",
        ),
        (
            LOG.assign(end=[2600, 2600, 2600, 2900, 2900]),
            {"x_col": "hours", "end_col": "end", "ends": {"M1": 2700}},
            "ends= says 2700; give one of them",
        ),
        (LOG, {}, "Give the ages"),
        (
            LOG.assign(hours=[420, None, 2050, 980, 2400]),
            {"x_col": "hours"},
            "blank on row 1: every event needs its age",
        ),
    ],
)
def test_log_inputs_are_checked(frame, kwargs, match):
    with pytest.raises(ValueError, match=match):
        EventHistoryData.from_dataframe(frame, i_col="machine", **kwargs)


def test_history_round_trip_through_a_frame():
    data = EventHistoryData.from_items(
        [
            ItemHistory(
                "A",
                observed=[5, 9],
                observed_event_types=["x", "y"],
                interval_censored=[[10, 12]],
                interval_censored_counts=2,
                interval_censored_event_types="x",
                right_censored=12,
                left_truncation=1,
                covariates={"load": 1.0},
            ),
            ItemHistory("B", right_censored=4, covariates={"load": 0.5}),
        ]
    )
    frame = data.to_dataframe()
    assert list(frame.columns) == [
        "i",
        "xl",
        "xr",
        "c",
        "n",
        "tl",
        "e",
        "load",
    ]
    again = EventHistoryData.from_dataframe(
        frame,
        i_col="i",
        xl_col="xl",
        xr_col="xr",
        c_col="c",
        n_col="n",
        tl_col="tl",
        e_col="e",
        Z_cols=["load"],
    )
    assert again.to_items() == data.to_items()
    with_states = EventHistoryData.from_items(
        [ItemHistory("A", right_censored=4, initial_state="ok")]
    )
    with pytest.raises(ValueError, match="states, which a table"):
        with_states.to_dataframe()
