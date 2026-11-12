"""``x``, ``xl`` and ``xr`` given together: each row an exact time in
``x`` or a range in ``xl`` and ``xr``, read row by row."""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import SurpyvalData
from surpyval.utils.data_formats import exact_or_range, xcnt_handler

inf, nan = np.inf, np.nan


def _same(a, b):
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.c, b.c)
    assert np.array_equal(a.n, b.n)
    assert np.array_equal(a.t, b.t)


# -- the readings of a row -------------------------------------------------


@pytest.mark.parametrize(
    "x, xl, xr, c",
    [
        (3.0, nan, nan, 0),  # exact
        (nan, 5.0, inf, 1),  # right censored at 5
        (nan, -inf, 5.0, -1),  # left censored at 5
        (nan, 2.0, 4.0, 2),  # interval
        (nan, 4.0, 4.0, 0),  # equal ends: exact
        (nan, 0.0, 4.0, 2),  # (0, v) stays an interval
    ],
)
def test_each_reading(x, xl, xr, c):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = SurpyvalData(x=[x], xl=[xl], xr=[xr])
    assert data.c.tolist() == [c]
    # The same as the row given as a pure xl / xr range
    exact = np.isfinite(x)
    pure = SurpyvalData(xl=[x if exact else xl], xr=[x if exact else xr])
    _same(data, pure)


@pytest.mark.parametrize("blank", [None, nan, pd.NA])
def test_every_kind_of_blank(blank):
    data = SurpyvalData(
        x=[3, blank, blank],
        xl=[blank, 5, 2],
        xr=[blank, inf, 4],
        group_and_sort=False,
    )
    assert data.c.tolist() == [0, 1, 2]
    assert data.x.tolist() == [[3, 3], [5, 5], [2, 4]]


def test_infinite_ends_and_c_default_match_the_handler():
    x, c, keep = exact_or_range([nan, nan, 1], [-inf, 2, nan], [3, inf, nan])
    assert keep is None and c is None
    assert x.tolist() == [[-inf, 3], [2, inf], [1, 1]]


def test_mixed_equals_pure_arrays_and_fit_bit_identical():
    rng = np.random.default_rng(3)
    t = np.round(rng.weibull(1.7, 40) * 10, 3)
    exact = t[:20]
    right = t[20:30]
    lo = t[30:]
    hi = lo + 1.5
    x = np.r_[exact, [nan] * 20]
    xl = np.r_[[nan] * 20, right, lo]
    xr = np.r_[[nan] * 20, [inf] * 10, hi]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = SurpyvalData(x=x, xl=xl, xr=xr)
    pure = SurpyvalData(xl=np.r_[exact, right, lo], xr=np.r_[exact, xr[20:]])
    _same(mixed, pure)
    a = sp.Weibull.fit(mixed)
    b = sp.Weibull.fit(pure)
    assert a.params.tolist() == b.params.tolist()
    xm, cm, nm, tm = xcnt_handler(x=x, xl=xl, xr=xr)
    xp, cp, np_, tp = xcnt_handler(
        xl=np.r_[exact, right, lo], xr=np.r_[exact, xr[20:]]
    )
    assert np.array_equal(xm, xp) and np.array_equal(cm, cp)
    assert np.array_equal(nm, np_) and np.array_equal(tm, tp)


# -- rows giving both: left out, loudly ------------------------------------


def test_rows_giving_both_are_left_out_of_every_per_row_array():
    with pytest.warns(UserWarning) as caught:
        data = SurpyvalData(
            x=[3, nan, 7, nan, 8],
            xl=[nan, 5, 1, 2, nan],
            xr=[nan, inf, 9, 4, 12],
            c=[0, 1, 0, 2, 0],
            n=[1, 2, 3, 4, 5],
            tl=[0, 0, 0.5, 0, 0.25],
            Z=[[1.0], [2.0], [3.0], [4.0], [5.0]],
            e=["a", None, "c", "d", "e"],
            i=["u", "v", "w", "y", "z"],
            w=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
    message = str(caught[0].message)
    assert message.startswith("2 rows (rows 2, 4) were left out")
    assert "both an exact time in 'x' and a range in 'xl' / 'xr'" in message
    assert "Give each row either 'x' or 'xl' and 'xr'" in message
    assert data.ids == ["u", "v", "y"]
    assert data.n.tolist() == [1, 2, 4]
    assert data.c.tolist() == [0, 1, 2]
    assert data.tl.tolist() == [0, 0, 0]
    assert data.Z[:, 0].tolist() == [1.0, 2.0, 4.0]
    assert data.e.tolist() == ["a", None, "d"]
    assert data.w.tolist() == [1.0, 2.0, 4.0]


def test_one_end_given_with_x_is_both_too():
    with pytest.warns(UserWarning, match=r"^1 row \(row 0\) was left out"):
        data = SurpyvalData(x=[3, 4], xl=[1, nan])
    assert data.x.tolist() == [4.0]


def test_the_warning_names_the_first_few():
    x = [1.0] * 8 + [2.0]
    xl = [0.5] * 8 + [nan]
    xr = [1.5] * 8 + [nan]
    with pytest.warns(
        UserWarning, match=r"8 rows \(rows 0, 1, 2, 3, 4 and 3 more\)"
    ):
        SurpyvalData(x=x, xl=xl, xr=xr)


def test_every_row_left_out_is_refused():
    with pytest.raises(ValueError, match=r"Every row \(2\) gives both"):
        SurpyvalData(x=[1, 2], xl=[0, 1], xr=[2, 3])


# -- refusals ---------------------------------------------------------------


def test_a_row_with_nothing_is_refused():
    with pytest.raises(ValueError, match=r"Row 1 has no time: 'x', 'xl'"):
        SurpyvalData(x=[1, nan], xl=[nan, nan], xr=[nan, None])


def test_a_range_with_one_end_blank_is_refused():
    with pytest.raises(ValueError, match=r"Rows 1, 2 have a range with one"):
        SurpyvalData(x=[1, nan, nan], xl=[nan, 2, nan], xr=[nan, nan, 4])


def test_c_that_disagrees_with_the_range_is_refused():
    with pytest.raises(ValueError) as caught:
        SurpyvalData(
            x=[1, nan, nan], xl=[nan, 2, -inf], xr=[nan, inf, 4], c=[0, -1, 1]
        )
    message = str(caught.value)
    assert message.startswith("Rows 1, 2 have a flag in 'c' that disagrees")
    assert "row 1: c = -1, but [2, inf] reads as right censored" in message
    assert "row 2: c = 1, but [-inf, 4] reads as left censored" in message


def test_c_agreeing_blank_or_two_on_a_range_is_taken():
    data = SurpyvalData(
        x=[1, nan, nan, nan],
        xl=[nan, 2, -inf, 3],
        xr=[nan, inf, 4, 6],
        c=[0, None, 2, 2],
        group_and_sort=False,
    )
    assert data.c.tolist() == [0, 1, -1, 2]


def test_c_on_equal_ends_is_the_users_as_today():
    data = SurpyvalData(x=[1, nan], xl=[nan, 4], xr=[nan, 4], c=[0, 1])
    pure = SurpyvalData(xl=[1, 4], xr=[1, 4], c=[0, 1])
    _same(data, pure)


def test_blank_c_on_an_exact_row_and_two_on_one_are_refused():
    with pytest.raises(
        ValueError, match="Row 0 has an exact time in 'x' and a blank 'c'"
    ):
        SurpyvalData(x=[1, nan], xl=[nan, 2], xr=[nan, 3], c=[nan, 2])
    with pytest.raises(ValueError, match=r"Row 0 has c = 2 \(interval"):
        SurpyvalData(x=[1, nan], xl=[nan, 2], xr=[nan, 3], c=[2, 2])


def test_lengths_must_match():
    with pytest.raises(ValueError, match="'xl' has 1 rows and 'x' 2"):
        SurpyvalData(x=[1, nan], xl=[2], xr=[3, 4])


# -- only x, or only xl / xr: as before ----------------------------------


def test_only_x_and_only_xl_xr_are_unchanged():
    with pytest.raises(ValueError, match="Variable 'x' cannot contain NaN"):
        SurpyvalData(xl=[1, nan], xr=[2, 3])
    with pytest.raises(ValueError, match="Variable 'x' cannot contain NaN"):
        SurpyvalData(x=[1, nan])
    with pytest.raises(ValueError, match="Must use either 'x' or both"):
        SurpyvalData(xl=[1, 2])
    with pytest.raises(ValueError, match="Censor flag indicates not interval"):
        SurpyvalData(xl=[1, 2], xr=[1, inf], c=[0, 1])
    data = SurpyvalData(xl=[1, 2, 3], xr=[1, inf, 5], group_and_sort=False)
    assert data.c.tolist() == [0, 1, 2]


# -- DataFrames -----------------------------------------------------------


def _frame():
    return pd.DataFrame(
        {
            "x": [3.0, nan, nan, 7.0, 4.0, 6.0, 8.0],
            "xl": [nan, 5.0, 2.0, 1.0, nan, nan, nan],
            "xr": [nan, nan, 4.0, 9.0, nan, nan, nan],
            "c": [0, nan, nan, 0, 1, 0, 0],
            "load": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.5],
            "unit": list("abcdefg"),
        }
    )


def test_from_dataframe_reads_blank_ends_as_infinite_as_it_does_alone():
    df = _frame()
    with pytest.warns(UserWarning, match=r"1 row \(row 3\) was left out"):
        data = SurpyvalData.from_dataframe(
            df, x_col="x", xl_col="xl", xr_col="xr", c_col="c", i_col="unit"
        )
    assert data.ids == ["a", "b", "c", "e", "f", "g"]
    assert data.c.tolist() == [0, 1, 2, 1, 0, 0]
    rest = df.drop(index=3)
    pure = SurpyvalData.from_dataframe(
        rest.assign(
            xl=rest.x.fillna(rest.xl),
            xr=rest.x.fillna(rest.xr),
            c=[0, 1, 2, 1, 0, 0],
        ),
        xl_col="xl",
        xr_col="xr",
        c_col="c",
        i_col="unit",
    )
    _same(data, pure)


def test_fit_from_df_with_the_three_columns():
    df = _frame()
    df.loc[1, "xr"] = inf  # fit_from_df takes no blank end
    with pytest.warns(UserWarning, match="left out"):
        model = sp.Weibull.fit_from_df(
            df, x_col="x", xl_col="xl", xr_col="xr", c_col="c"
        )
    rest = df.drop(index=3)
    pure = sp.Weibull.fit_from_df(
        rest.assign(
            xl=rest.x.fillna(rest.xl),
            xr=rest.x.fillna(rest.xr),
            c=[0, 2, 2, 1, 0, 0],
        ),
        xl_col="xl",
        xr_col="xr",
        c_col="c",
    )
    assert model.params.tolist() == pure.params.tolist()
    with (
        pytest.warns(UserWarning, match="left out"),
        pytest.raises(ValueError, match="a range with one end blank"),
    ):
        sp.Weibull.fit_from_df(
            _frame(), x_col="x", xl_col="xl", xr_col="xr", c_col="c"
        )


def test_regression_fit_from_df_with_the_three_columns():
    df = _frame()
    df.loc[1, "xr"] = inf
    with pytest.warns(UserWarning, match="left out"):
        model = sp.WeibullPH.fit_from_df(
            df, x_col="x", xl_col="xl", xr_col="xr", c_col="c", Z_cols="load"
        )
    rest = df.drop(index=3)
    pure = sp.WeibullPH.fit_from_df(
        rest.assign(
            xl=rest.x.fillna(rest.xl),
            xr=rest.x.fillna(rest.xr),
            c=[0, 2, 2, 1, 0, 0],
        ),
        xl_col="xl",
        xr_col="xr",
        c_col="c",
        Z_cols="load",
    )
    assert model.params.tolist() == pure.params.tolist()
