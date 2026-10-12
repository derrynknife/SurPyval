"""The data utilities on SurpyvalData: from_observations, from_xrd,
to_observations and to_xcnt; fitters taking a SurpyvalData as ``x``; and
the dated deprecation of the top-level data-format functions they
replace."""

import datetime
from pathlib import Path

import numpy as np
import pytest

import surpyval as sp
from surpyval import SurpyvalData
from surpyval.utils import data_formats
from surpyval.utils.deprecation import (
    DATA_FUNCTIONS_DEPRECATED_ON,
    DATA_FUNCTIONS_REMOVED_AFTER,
    DATED_DATA_FUNCTIONS,
)

X = sp.Weibull.random(40, 10, 2, random_state=1)
C = (X > 14).astype(int)
Z = np.r_[np.zeros(20), np.ones(20)]


# -- constructors and converters ------------------------------------------


def test_from_observations_counts_and_sorts():
    data = SurpyvalData.from_observations(
        observed=[1, 3, 3, 7], right_censored=[5, 9], left_censored=[0.5]
    )
    np.testing.assert_array_equal(data.x, [0.5, 1, 3, 5, 7, 9])
    np.testing.assert_array_equal(data.c, [-1, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(data.n, [1, 1, 2, 1, 1, 1])


def test_from_observations_matches_the_old_function():
    f, s, lc, i = [1, 3, 3, 7], [5, 9], [0.5], [[2, 4], [6, 8]]
    data = SurpyvalData.from_observations(
        observed=f, right_censored=s, left_censored=lc, interval_censored=i
    )
    for got, want in zip(
        data.to_xcnt(), data_formats.fsli_to_xcnt(f, s, lc, i)
    ):
        np.testing.assert_array_equal(got, want)


def test_from_observations_takes_a_shared_truncation():
    data = SurpyvalData.from_observations(
        observed=[2, 4], left_truncation=0.5, right_truncation=9
    )
    np.testing.assert_array_equal(data.tl, [0.5, 0.5])
    np.testing.assert_array_equal(data.tr, [9, 9])
    with pytest.raises(ValueError, match="'left_truncation' must be one"):
        SurpyvalData.from_observations(observed=[2, 4], left_truncation=[0, 1])
    with pytest.raises(TypeError):
        SurpyvalData.from_observations(observed=[2, 4], tl=0.5)


def test_from_observations_takes_a_single_time_as_a_number():
    data = SurpyvalData.from_observations(
        observed=12, right_censored=30.0, left_censored=np.float64(8)
    )
    np.testing.assert_array_equal(data.x, [8, 12, 30])
    np.testing.assert_array_equal(data.c, [-1, 0, 1])


def test_from_observations_needs_data():
    with pytest.raises(ValueError):
        SurpyvalData.from_observations()


def test_from_xrd_matches_the_old_function():
    x, r, d = [1, 2, 3], [5, 3, 1], [1, 1, 1]
    data = SurpyvalData.from_xrd(x, r, d)
    for got, want in zip(data.to_xcnt(), data_formats.xrd_to_xcnt(x, r, d)):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(data.to_xrd()[1], r)


def test_to_observations_round_trips():
    data = SurpyvalData.from_observations(
        observed=[1, 3, 3, 7],
        right_censored=[5, 9],
        left_censored=[0.5],
        interval_censored=[[2, 4], [2, 4], [6, 8]],
    )
    lists = data.to_observations()
    # each value once, with its count
    assert lists["observed"].tolist() == [1, 3, 7]
    assert lists["observed_counts"].tolist() == [1, 2, 1]
    assert lists["interval_censored"].tolist() == [[2, 4], [6, 8]]
    assert lists["interval_censored_counts"].tolist() == [2, 1]
    again = SurpyvalData.from_observations(**lists)
    for a, b in zip(again.to_xcnt(), data.to_xcnt()):
        np.testing.assert_array_equal(a, b)


def test_each_kind_takes_its_own_truncation():
    data = SurpyvalData.from_observations(
        observed=[600, 700, 900],
        observed_left_truncation=[500, 0, 650],
        right_censored=[1000, 1200],
        right_censored_left_truncation=500,
        interval_censored=[[300, 400]],
        interval_censored_right_truncation=800,
    )
    by_hand = SurpyvalData(
        [600, 700, 900, 1000, 1200, [300, 400]],
        c=[0, 0, 0, 1, 1, 2],
        tl=[500, 0, 650, 500, 500, -np.inf],
        tr=[np.inf] * 5 + [800],
    )
    for got, want in zip(data.to_xcnt(), by_hand.to_xcnt()):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(
        sp.Weibull.fit(data).params, sp.Weibull.fit(by_hand).params
    )


def test_a_kinds_own_bound_needs_its_list_and_the_right_length():
    with pytest.raises(ValueError, match="one per item \\(2"):
        SurpyvalData.from_observations(
            observed=[600, 700], observed_left_truncation=[1, 2, 3]
        )
    with pytest.raises(ValueError, match="without 'right_censored'"):
        SurpyvalData.from_observations(
            observed=[600], right_censored_left_truncation=500
        )
    with pytest.raises(ValueError, match="use -inf"):
        SurpyvalData.from_observations(
            observed=[600, 700], observed_left_truncation=[1, np.nan]
        )


def test_shared_and_own_bound_on_one_side_are_refused():
    with pytest.raises(ValueError, match="give one"):
        SurpyvalData.from_observations(
            observed=[600],
            left_truncation=0,
            observed_left_truncation=500,
        )
    # the other side, or another kind, is no clash
    data = SurpyvalData.from_observations(
        observed=[600],
        right_censored=[1000],
        left_truncation=100,
        observed_right_truncation=800,
    )
    np.testing.assert_array_equal(data.t, [[100, 800], [100, np.inf]])


def test_to_observations_round_trips_truncated_data():
    data = SurpyvalData(
        [600, 600, 700, 1000, [300, 400]],
        c=[0, 0, 0, 1, 2],
        n=[2, 1, 1, 3, 1],
        tl=[500, 0, 0, 500, 100],
    )
    lists = data.to_observations()
    assert lists["observed_left_truncation"].tolist() == [0, 500, 0]
    assert lists["observed_counts"].tolist() == [1, 2, 1]
    assert "observed_right_truncation" not in lists
    again = SurpyvalData.from_observations(**lists)
    for a, b in zip(again.to_xcnt(), data.to_xcnt()):
        np.testing.assert_array_equal(a, b)


def test_to_xrd_gives_copies():
    data = SurpyvalData([1, 2, 3], [0, 1, 0])
    x, r, d = data.to_xrd()
    r *= 10
    d[:] = 0
    np.testing.assert_array_equal(data.to_xrd()[1], [3, 2, 1])
    np.testing.assert_array_equal(data.to_xrd()[2], [1, 0, 1])


@pytest.mark.parametrize(
    "kind", ["observed", "right_censored", "left_censored"]
)
def test_infinite_single_times_are_refused_by_name(kind):
    with pytest.raises(ValueError, match=f"'{kind}' times must be finite"):
        SurpyvalData.from_observations(**{kind: [1.0, np.inf]})


def test_pandas_lists_and_bounds_must_share_an_index():
    import pandas as pd

    times = pd.Series([600.0, 700.0], index=[1, 0])
    with pytest.raises(ValueError, match="different indexes"):
        SurpyvalData.from_observations(
            observed=times,
            observed_left_truncation=pd.Series([500.0, 0.0], index=[0, 1]),
        )
    data = SurpyvalData.from_observations(
        observed=times,
        observed_left_truncation=pd.Series([500.0, 0.0], index=[1, 0]),
    )
    np.testing.assert_array_equal(data.tl, [500, 0])


def test_to_xcnt_gives_copies():
    data = SurpyvalData([3, 1, 3], [0, 1, 0])
    x, c, n, t = data.to_xcnt()
    x[0] = 99.0
    assert data.x[0] == 1.0


# -- fit(data) ------------------------------------------------------------


def test_parametric_and_nonparametric_fits_take_data():
    data = SurpyvalData(X, C)
    np.testing.assert_allclose(
        sp.Weibull.fit(data).params, sp.Weibull.fit(X, C).params
    )
    np.testing.assert_allclose(
        sp.KaplanMeier.fit(data).sf([5, 10]),
        sp.KaplanMeier.fit(X, C).sf([5, 10]),
    )
    assert sp.Weibull.fit(data, unit="hours").unit == "hour"


def test_mixture_and_fit_best_take_data():
    data = SurpyvalData(X, C)
    assert sp.MixtureModel.fit(data, dist=sp.Weibull).params.shape == (2, 2)
    assert sp.MixtureModel(sp.Weibull).fit(data).params.shape == (2, 2)
    np.testing.assert_allclose(
        sp.fit_best(data).params, sp.fit_best(X, C).params
    )


def test_regressions_take_covariates_from_the_data():
    data = SurpyvalData(X, C, Z=Z)
    np.testing.assert_allclose(
        sp.WeibullPH.fit(data).params, sp.WeibullPH.fit(X, Z, C).params
    )
    np.testing.assert_allclose(
        sp.CoxPH.fit(data).params, sp.CoxPH.fit(X, Z, c=C).params
    )


def test_truncated_interval_data_fits_as_from_its_arrays():
    data = SurpyvalData.from_observations(
        observed=[2, 4, 5],
        interval_censored=[[1, 3], [4, 6]],
        left_truncation=0.5,
    )
    np.testing.assert_allclose(
        sp.Weibull.fit(data).params,
        sp.Weibull.fit(data.x, data.c, data.n, data.t).params,
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.Weibull.fit(d, c=C),
        lambda d: sp.Weibull.fit(d, tl=0.0),
        lambda d: sp.KaplanMeier.fit(d, n=np.ones(40)),
    ],
)
def test_data_and_separate_data_arguments_are_refused(call):
    with pytest.raises(TypeError, match="holds the data"):
        call(SurpyvalData(X, C))


def test_a_regression_given_data_without_covariates_says_so():
    with pytest.raises(TypeError, match="holds none"):
        sp.WeibullPH.fit(SurpyvalData(X, C))


@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.recurrent.CrowAMSAA.fit(d),
        lambda d: sp.recurrent.NonParametricCounting.fit(d),
    ],
)
def test_a_fit_taking_arrays_only_names_the_data(fit):
    with pytest.raises(TypeError, match="does not take"):
        fit(SurpyvalData(np.cumsum(X[:10])))


def test_covariates_in_both_places_are_refused():
    with pytest.raises(TypeError, match="'Z'"):
        sp.WeibullPH.fit(SurpyvalData(X, C, Z=Z), Z=Z)


def test_fit_from_surpyval_data_is_kept():
    data = SurpyvalData(X, C)
    np.testing.assert_allclose(
        sp.Weibull.fit_from_surpyval_data(data).params,
        sp.Weibull.fit(data).params,
    )


# -- the dated deprecation --------------------------------------------------


@pytest.mark.parametrize("name", sorted(DATED_DATA_FUNCTIONS))
def test_old_top_level_names_warn_with_their_date_and_replacement(name):
    with pytest.warns(FutureWarning) as caught:
        function = getattr(sp, name)
    message = str(caught[0].message)
    assert DATA_FUNCTIONS_REMOVED_AFTER.isoformat() in message
    assert "SurpyvalData" in message
    assert function is getattr(data_formats, name)
    # Listed, so that a star import still gets it until the removal date
    assert name in sp.__all__


def test_a_star_import_still_gets_the_old_names_with_the_warning():
    namespace: dict = {}
    with pytest.warns(FutureWarning) as caught:
        exec("from surpyval import *", namespace)
    assert any("fs_to_xcnt" in str(w.message) for w in caught)
    assert namespace["fs_to_xcnt"] is data_formats.fs_to_xcnt


def test_the_warning_shows_in_code_outside_main(tmp_path):
    # A DeprecationWarning would be hidden here by Python's default
    # filters; the removal is on a date, so the warning must show.
    import subprocess
    import sys

    (tmp_path / "usermod.py").write_text(
        "import surpyval\n"
        "def go():\n"
        "    return surpyval.fs_to_xcnt([1, 2], [3])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", "import usermod; usermod.go()"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    assert "FutureWarning" in out.stderr and "fs_to_xcnt" in out.stderr


def test_the_window_is_six_months():
    assert (
        DATA_FUNCTIONS_REMOVED_AFTER - DATA_FUNCTIONS_DEPRECATED_ON
    ).days in (
        181,
        182,
        183,
        184,
    )


def test_importing_surpyval_does_not_warn():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-W",
            "error::FutureWarning",
            "-c",
            "import surpyval, surpyval.recurrent, surpyval.degradation",
        ],
        check=True,
    )


@pytest.mark.skipif(
    not (Path(sp.__file__).resolve().parent.parent / ".git").exists(),
    reason="a reminder for SurPyval's own repository: an installed copy "
    "keeps working past the date",
)
def test_dated_removal_is_not_overdue():
    # Fails from the removal date on: the first release on or after it
    # removes the top-level names (the functions stay internal in
    # surpyval.utils.data_formats). Only in the repository: the installed
    # test suite (#661) must not start failing for users on a date.
    assert datetime.date.today() < DATA_FUNCTIONS_REMOVED_AFTER, (
        "The top-level data-format functions deprecated on "
        f"{DATA_FUNCTIONS_DEPRECATED_ON} are due for removal: remove "
        "DATED_DATA_FUNCTIONS and its branch in surpyval.__getattr__."
    )


# -- counts and covariates per kind ----------------------------------------


def test_counts_per_kind_read_a_life_table():
    data = SurpyvalData.from_observations(
        interval_censored=[[0, 100], [100, 200], [200, 300], [300, 400]],
        interval_censored_counts=[3, 0, 9, 4],
        right_censored=[400],
        right_censored_counts=12,
    )
    by_hand = SurpyvalData(
        [[0, 100], [200, 300], [300, 400], [400, 400]],
        c=[2, 2, 2, 1],
        n=[3, 9, 4, 12],
    )
    for got, want in zip(data.to_xcnt(), by_hand.to_xcnt()):
        np.testing.assert_array_equal(got, want)


def test_counts_equal_repeating_the_values():
    counted = SurpyvalData.from_observations(
        observed=[10, 20],
        observed_counts=[2, 3],
        right_censored=30,
        right_censored_counts=4,
    )
    repeated = SurpyvalData.from_observations(
        observed=[10, 10, 20, 20, 20], right_censored=[30] * 4
    )
    for got, want in zip(counted.to_xcnt(), repeated.to_xcnt()):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "counts, match",
    [([1, 2, 3], "one per item"), ([1, -1], "0 or more"), ([1, 0.5], "whole")],
)
def test_bad_counts_are_refused(counts, match):
    with pytest.raises(ValueError, match=match):
        SurpyvalData.from_observations(observed=[1, 2], observed_counts=counts)


def test_every_count_zero_is_no_data():
    with pytest.raises(ValueError, match="count above 0"):
        SurpyvalData.from_observations(observed=[1, 2], observed_counts=0)


def test_covariates_per_kind_fit_as_the_arrays():
    zo, zr = Z[C == 0], Z[C == 1]
    data = SurpyvalData.from_observations(
        observed=X[C == 0],
        observed_covariates=zo,
        right_censored=X[C == 1],
        right_censored_covariates=zr,
    )
    by_hand = sp.WeibullPH.fit(
        np.r_[X[C == 0], X[C == 1]],
        np.r_[zo, zr],
        np.r_[np.zeros((C == 0).sum()), np.ones((C == 1).sum())],
    )
    np.testing.assert_allclose(sp.WeibullPH.fit(data).params, by_hand.params)


def test_covariates_with_counts_drop_the_zero_counted_rows():
    data = SurpyvalData.from_observations(
        observed=[10, 20, 30],
        observed_counts=[1, 0, 2],
        observed_covariates=[[1, 0], [0, 1], [1, 1]],
    )
    np.testing.assert_array_equal(data.Z, [[1, 0], [1, 1]])
    np.testing.assert_array_equal(data.n, [1, 2])


def test_covariates_needed_for_every_kind_with_items():
    with pytest.raises(ValueError, match="'right_censored_covariates' too"):
        SurpyvalData.from_observations(
            observed=[1, 2], observed_covariates=[0, 1], right_censored=[3]
        )
    # an empty kind needs none
    data = SurpyvalData.from_observations(
        observed=[1, 2], observed_covariates=[0, 1], right_censored=[]
    )
    assert data.Z.shape == (2, 1)


def test_covariates_need_the_same_columns_and_one_row_per_item():
    with pytest.raises(ValueError, match="same columns"):
        SurpyvalData.from_observations(
            observed=[1, 2],
            observed_covariates=[[0, 1], [1, 0]],
            right_censored=[3],
            right_censored_covariates=[1],
        )
    with pytest.raises(ValueError, match=r"\[\[z1, z2"):
        SurpyvalData.from_observations(
            observed=[1], observed_covariates=[1, 0, 2]
        )


def test_a_kinds_values_need_its_list():
    for name in ("right_censored_counts", "right_censored_covariates"):
        with pytest.raises(ValueError, match="without 'right_censored'"):
            SurpyvalData.from_observations(observed=[1], **{name: 1})


def test_to_observations_round_trips_counts_truncation_and_covariates():
    data = SurpyvalData.from_observations(
        observed=[10, 20, 20],
        observed_counts=[1, 2, 3],
        observed_left_truncation=[0, 5, 0],
        observed_covariates=[[1, 0], [0, 1], [1, 1]],
        interval_censored=[[30, 40]],
        interval_censored_covariates=[[0, 0]],
    )
    lists = data.to_observations()
    assert lists["observed_covariates"].tolist() == [[1, 0], [0, 1], [1, 1]]
    again = SurpyvalData.from_observations(**lists)
    for a, b in zip(again.to_xcnt(), data.to_xcnt()):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(again.Z, data.Z)
