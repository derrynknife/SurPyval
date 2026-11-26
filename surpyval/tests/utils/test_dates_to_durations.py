"""``surpyval.utils.dates_to_durations``: dates to durations and flags."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.utils import dates_to_durations

START = pd.to_datetime(
    ["2023-01-01", "2023-01-05", "2023-02-01", "2023-02-10"]
)
END = pd.to_datetime(["2023-03-01", None, "2023-04-11", "2023-06-01"])


def test_missing_end_is_censored_at_the_latest_date_seen():
    x, c = dates_to_durations(START, END)
    # latest date is 2023-06-01
    np.testing.assert_array_equal(x, [59, 147, 69, 111])
    np.testing.assert_array_equal(c, [0, 1, 0, 0])


def test_cutoff_censors_ends_after_it():
    # matches lifelines.datetimes_to_durations(fill_date="2023-05-01"):
    # T = [59, 116, 69, 80], E = [1, 0, 1, 0]
    x, c = dates_to_durations(START, END, cutoff=pd.Timestamp("2023-05-01"))
    np.testing.assert_array_equal(x, [59, 116, 69, 80])
    np.testing.assert_array_equal(c, [0, 1, 0, 1])


@pytest.mark.parametrize(
    "unit, per_day",
    [
        ("D", 1),
        ("days", 1),
        ("h", 24),
        ("hours", 24),
        ("min", 1440),
        ("s", 86400),
        ("W", 1 / 7),
        ("weeks", 1 / 7),
        ("Y", 1 / 365.25),
        ("years", 1 / 365.25),
    ],
)
def test_units(unit, per_day):
    x, _ = dates_to_durations(START, END, unit=unit)
    np.testing.assert_allclose(x, np.array([59, 147, 69, 111]) * per_day)


def test_part_of_a_unit_is_kept():
    x, _ = dates_to_durations(
        pd.to_datetime(["2023-01-01 12:00"]),
        pd.to_datetime(["2023-01-03 06:00"]),
    )
    assert x[0] == 1.75


def test_round_trip_from_known_durations():
    rng = np.random.default_rng(1)
    days = rng.integers(1, 400, size=50)
    start = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 300, size=50), unit="D"
    )
    flags = rng.integers(0, 2, size=50)
    end = start + pd.to_timedelta(days, unit="D")
    # a unit with c = 1 has no end date: it is censored at the cutoff
    cutoff = end.max()
    ended = pd.Series(end).where(flags == 0)
    x, c = dates_to_durations(start, ended, cutoff=cutoff)
    np.testing.assert_array_equal(c, flags)
    np.testing.assert_array_equal(x[flags == 0], days[flags == 0])
    np.testing.assert_array_equal(
        x[flags == 1], ((cutoff - start) / pd.Timedelta(days=1))[flags == 1]
    )
    # and the fit takes them
    sp.Weibull.fit(x, c=c)


def test_event_false_censors_at_the_end_date():
    x, c = dates_to_durations(START, END, event=[1, 1, 0, 1])
    np.testing.assert_array_equal(c, [0, 1, 1, 0])
    np.testing.assert_array_equal(x, [59, 147, 69, 111])


def test_no_end_means_all_running():
    x, c = dates_to_durations(START)
    np.testing.assert_array_equal(c, [1, 1, 1, 1])
    assert x[-1] == 0


def test_inputs_may_be_lists_arrays_or_datetimes():
    a = dates_to_durations(START, END)
    for s, e in [
        (list(START), list(END)),
        (START.values, END.values),
        (pd.Series(START), pd.Series(END)),
        (
            [dt.datetime(2023, 1, 1)],
            [dt.datetime(2023, 3, 1)],
        ),
    ][:3]:
        for got, want in zip(dates_to_durations(s, e), a):
            np.testing.assert_array_equal(got, want)
    x, c = dates_to_durations(
        [dt.datetime(2023, 1, 1)], [dt.datetime(2023, 3, 1)]
    )
    assert x[0] == 59 and c[0] == 0


def test_aware_dates_are_compared_in_utc():
    # 2023-03-12 is the US spring-forward day: 23 hours elapse in a "day"
    s = pd.to_datetime(["2023-03-12 00:00"]).tz_localize("America/New_York")
    e = pd.to_datetime(["2023-03-13 00:00"]).tz_localize("America/New_York")
    x, _ = dates_to_durations(s, e, unit="h")
    assert x[0] == 23
    # a zone is no different from UTC for the same instants
    x2, _ = dates_to_durations(
        s.tz_convert("UTC"), e.tz_convert("UTC"), unit="h"
    )
    assert x2[0] == 23


def test_a_mix_of_aware_and_naive_is_refused():
    with pytest.raises(ValueError, match="mix time-zone aware and naive"):
        dates_to_durations(START, END.tz_localize("UTC"))
    with pytest.raises(ValueError, match="mix time-zone aware and naive"):
        dates_to_durations(
            START, END, cutoff=pd.Timestamp("2024-01-01", tz="UTC")
        )
    with pytest.raises(ValueError, match="mixes time-zone aware and naive"):
        dates_to_durations(
            [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02", tz="UTC")]
        )


def test_end_before_start_is_refused_with_a_count():
    end = pd.to_datetime(["2022-12-01", None, "2023-01-01", "2023-06-01"])
    with pytest.raises(ValueError, match=r"2 unit\(s\) end before they start"):
        dates_to_durations(START, end)


def test_start_after_cutoff_and_missing_start_are_refused():
    with pytest.raises(
        ValueError, match=r"3 unit\(s\) start after the cutoff"
    ):
        dates_to_durations(START, END, cutoff=pd.Timestamp("2023-01-02"))
    with pytest.raises(ValueError, match="missing date"):
        dates_to_durations([pd.NaT, pd.Timestamp("2023-01-01")])


def test_text_numbers_and_bad_units_are_refused():
    with pytest.raises(ValueError, match="must hold dates"):
        dates_to_durations(["2023-01-01"])
    with pytest.raises(ValueError, match="must hold dates"):
        dates_to_durations([1.0, 2.0])
    with pytest.raises(ValueError, match="Months have no fixed length"):
        dates_to_durations(START, unit="months")
    with pytest.raises(ValueError, match="one end date"):
        dates_to_durations(START, END[:2])


def test_the_date_refusals_point_to_the_helper():
    with pytest.raises(ValueError, match="dates_to_durations"):
        sp.Weibull.fit(START)
    with pytest.raises(ValueError, match="dates_to_durations"):
        sp.Weibull.fit_from_df(pd.DataFrame({"x": START}), x_col="x")
