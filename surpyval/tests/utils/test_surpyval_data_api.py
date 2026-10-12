"""The data utilities on SurpyvalData: from_observations, from_xrd,
to_observations and to_xcnt; fitters taking a SurpyvalData as ``x``; and
the dated deprecation of the top-level data-format functions they
replace."""

import datetime

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
    data = SurpyvalData.from_observations(observed=[2, 4], tl=0.5, tr=9)
    np.testing.assert_array_equal(data.tl, [0.5, 0.5])
    np.testing.assert_array_equal(data.tr, [9, 9])
    with pytest.raises(ValueError, match="one bound shared"):
        SurpyvalData.from_observations(observed=[2, 4], tl=[0, 1])


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
    assert lists["observed"].tolist() == [1, 3, 3, 7]
    assert lists["interval_censored"].tolist() == [[2, 4], [2, 4], [6, 8]]
    again = SurpyvalData.from_observations(**lists)
    for a, b in zip(again.to_xcnt(), data.to_xcnt()):
        np.testing.assert_array_equal(a, b)


def test_to_observations_refuses_truncated_data():
    with pytest.raises(ValueError, match="truncation"):
        SurpyvalData([1, 2, 3], tl=0.5).to_observations()


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
        observed=[2, 4, 5], interval_censored=[[1, 3], [4, 6]], tl=0.5
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
    with pytest.warns(DeprecationWarning) as caught:
        function = getattr(sp, name)
    message = str(caught[0].message)
    assert DATA_FUNCTIONS_REMOVED_AFTER.isoformat() in message
    assert "SurpyvalData" in message
    assert function is getattr(data_formats, name)
    assert name not in sp.__all__


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
            "-c",
            "import surpyval, surpyval.recurrent, surpyval.degradation",
        ],
        check=True,
    )


def test_dated_removal_is_not_overdue():
    # Fails from the removal date on: the first release on or after it
    # removes the top-level names (the functions stay internal in
    # surpyval.utils.data_formats).
    assert datetime.date.today() < DATA_FUNCTIONS_REMOVED_AFTER, (
        "The top-level data-format functions deprecated on "
        f"{DATA_FUNCTIONS_DEPRECATED_ON} are due for removal: remove "
        "DATED_DATA_FUNCTIONS and its branch in surpyval.__getattr__."
    )
