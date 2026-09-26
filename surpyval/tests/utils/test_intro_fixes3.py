"""Regression tests for the third review of the data layer (handlers and
converters in ``surpyval.utils``)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as surv
from surpyval import (
    KaplanMeier,
    NelsonAalen,
    SurpyvalData,
    Turnbull,
    Weibull,
    fsl_to_xcnt,
    fsli_handler,
    xcn_to_fs,
    xcnt_handler,
    xcnt_to_xrd,
    xrd_handler,
    xrd_to_xcnt,
)
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)

nan = np.nan
inf = np.inf


# --- NaN truncation bounds -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tl=[0, nan, 0, 0]),
        dict(tr=[9, 9, nan, 9]),
        dict(t=[[0, 9], [nan, 9], [0, 9], [0, 9]]),
        dict(tl=nan),
    ],
)
def test_nan_truncation_bound_is_refused(kwargs):
    # KM divided by zero, Weibull read NaN as "no truncation" and
    # Turnbull gave a third answer; all now refuse it.
    for fit in (KaplanMeier.fit, Weibull.fit, Turnbull.fit):
        with pytest.raises(ValueError, match="must not contain NaN"):
            fit([1, 2, 3, 4], **kwargs)
    with pytest.raises(ValueError, match="must not contain NaN"):
        xcnt_handler([1, 2, 3, 4], **kwargs)


def test_nan_and_inf_counts_are_refused():
    with pytest.raises(ValueError, match="integer values"):
        xcnt_handler([1, 2], n=[1, nan])
    with pytest.raises(ValueError, match="integer values"):
        xcnt_handler([1, 2], n=[1, inf])


def test_nan_in_fsli_and_fsl_is_refused():
    with pytest.raises(ValueError, match="'f' cannot contain NaN"):
        fsli_handler(f=[1, nan])
    with pytest.raises(ValueError, match="'i' cannot contain NaN"):
        fsli_handler(i=[[1, 2], [nan, 3]])
    with pytest.raises(ValueError, match="'s' cannot contain NaN"):
        fsl_to_xcnt([1, 2], [nan])


def test_nan_in_two_column_x_names_the_nan():
    # NaN used to fail the ordering check first, with a message about
    # left intervals instead.
    with pytest.raises(ValueError, match="cannot contain NaN"):
        xcnt_handler([[1, 2], [nan, 3]])


# --- xrd input --------------------------------------------------------------


def test_unsorted_xrd_is_sorted_with_its_rows():
    unsorted = KaplanMeier.from_xrd([3, 1, 2], [2, 5, 4], [1, 1, 1])
    ordered = KaplanMeier.from_xrd([1, 2, 3], [5, 4, 2], [1, 1, 1])
    np.testing.assert_allclose(unsorted.sf([1, 2, 3]), ordered.sf([1, 2, 3]))
    np.testing.assert_allclose(ordered.sf([1, 2, 3]), [0.8, 0.6, 0.3])

    x, r, d = xrd_handler([3, 1, 2], [2, 5, 4], [1, 0, 1])
    np.testing.assert_array_equal(x, [1, 2, 3])
    np.testing.assert_array_equal(r, [5, 4, 2])
    np.testing.assert_array_equal(d, [0, 1, 1])


def test_xrd_to_xcnt_validates_and_sorts():
    got = xrd_to_xcnt([2, 1], [3, 4], [1, 1])
    expected = xrd_to_xcnt([1, 2], [4, 3], [1, 1])
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="more deaths"):
        xrd_to_xcnt([1, 2], [2, 1], [3, 1])


def test_repeated_xrd_time_is_refused():
    with pytest.raises(ValueError, match="repeated times"):
        xrd_handler([1, 2, 2], [5, 4, 3], [1, 1, 1])
    with pytest.raises(ValueError, match="repeated times"):
        xrd_to_xcnt([1, 1], [5, 4], [1, 1])


def test_xrd_integer_valued_float_counts_are_accepted():
    x, r, d = xrd_handler([1.0, 2.0], [5.0, 4.0], [1.0, 2.0])
    assert r.dtype.kind == "i" and d.dtype.kind == "i"
    np.testing.assert_array_equal(r, [5, 4])
    with pytest.raises(ValueError, match="'r' must be an array of integers"):
        xrd_handler([1, 2], [5.5, 4], [1, 1])
    with pytest.raises(ValueError, match="'d' must be an array of integers"):
        xrd_handler([1, 2], [5, 4], [1, nan])


def test_xrd_zero_at_risk_is_not_called_negative():
    with pytest.raises(ValueError, match="must be positive") as err:
        xrd_handler([1, 2], [5, 0], [1, 0])
    assert "negative" not in str(err.value)


def test_xrd_empty_and_nan_are_refused():
    with pytest.raises(ValueError, match="empty"):
        xrd_handler([], [], [])
    with pytest.raises(ValueError, match="NaN"):
        xrd_handler([1, nan], [2, 1], [1, 1])


# --- two-column x with no interval rows ------------------------------------

x1 = np.array([2, 3, 5, 7, 8, 11, 13, 17, 19, 23, 29, 31.0])
c1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
x2 = np.c_[x1, x1]
q = np.array([4.0, 10.0, 20.0])


def test_two_column_points_become_one_column():
    x, c, n, t = xcnt_handler(x2, c1)
    assert x.ndim == 1
    x_ref, c_ref, _, _ = xcnt_handler(x1, c1)
    np.testing.assert_array_equal(x, x_ref)
    np.testing.assert_array_equal(c, c_ref)
    # One-sided infinite rows are points too once converted.
    x, c, _, _ = xcnt_handler(xl=[1, 2, 5], xr=[1, 2, inf])
    assert x.ndim == 1
    np.testing.assert_array_equal(c, [0, 0, 1])
    # A real interval keeps the two columns.
    x, _, _, _ = xcnt_handler([[1, 1], [2, 3]])
    assert x.ndim == 2


@pytest.mark.parametrize(
    "fit",
    [
        KaplanMeier.fit,
        NelsonAalen.fit,
        surv.FlemingHarrington.fit,
        lambda **k: Weibull.fit(how="MPP", **k),
        lambda **k: Weibull.fit(how="MPS", **k),
        lambda **k: Weibull.fit(how="MSE", **k),
    ],
)
def test_fitters_accept_two_column_points(fit):
    # These crashed with "object too deep for desired array".
    np.testing.assert_allclose(
        fit(x=x2, c=c1).sf(q), fit(x=x1, c=c1).sf(q), rtol=1e-10
    )


def test_mom_and_converters_accept_two_column_points():
    np.testing.assert_allclose(
        Weibull.fit(x=x2, how="MOM").params,
        Weibull.fit(x=x1, how="MOM").params,
    )
    for a, b in zip(xcnt_to_xrd(x2, c1), xcnt_to_xrd(x1, c1)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        SurpyvalData(xl=x1, xr=x1, c=c1).to_xrd(),
        SurpyvalData(x1, c1).to_xrd(),
    ):
        np.testing.assert_array_equal(a, b)


def test_fit_from_df_with_equal_left_and_right_columns():
    df = pd.DataFrame({"xl": x1, "xr": x1, "c": c1})
    model = Weibull.fit_from_df(df, xl="xl", xr="xr", c="c", how="MPP")
    np.testing.assert_allclose(
        model.params, Weibull.fit(x1, c1, how="MPP").params
    )


def test_regression_and_competing_risks_accept_two_column_points():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(len(x1), 1))
    for F in (surv.BuckleyJames, surv.AdditiveHazards):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = F.fit(x=x2, Z=Z, c=c1)
            b = F.fit(x=x1, Z=Z, c=c1)
        np.testing.assert_allclose(a.params, b.params)

    e = np.where(c1 == 1, None, np.where(np.arange(12) % 2, "a", "b"))
    e = e.tolist()
    np.testing.assert_allclose(
        CompetingRisks.fit(x2, e, c1).cif(q, "a"),
        CompetingRisks.fit(x1, e, c1).cif(q, "a"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(
            FineGray.fit(x2, Z, e, c1, cause="a").beta,
            FineGray.fit(x1, Z, e, c1, cause="a").beta,
        )
        CompetingRisksProportionalHazards.fit(x2, Z, e, c1)
        ParametricCompetingRisks.fit(x2, e, c1)


# --- truncation bounds: one- and two-column paths agree ---------------------


def test_right_censored_at_right_truncation_is_refused():
    # Censored at tr leaves no room for the event (tr < X <= tr); the
    # fit returned the optimiser's start with a stream of warnings.
    with pytest.raises(ValueError, match="right censored value must be"):
        Weibull.fit([1, 2, 3, 4, 5], c=[0, 0, 0, 0, 1], tr=5)
    with pytest.raises(ValueError, match="right censored value must be"):
        xcnt_handler(xl=[1, 2, 5], xr=[1, 2, inf], tr=5)
    # Observed at tr is fine, as is censored below it.
    xcnt_handler([1, 2, 5], c=[0, 0, 0], tr=5)
    xcnt_handler([1, 2, 4.9], c=[0, 0, 1], tr=5)


def test_one_sided_two_column_rows_with_truncation():
    # The infinite end used to be compared with the truncation bound.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Weibull.fit(xl=[1, 2, 3, 5.0], xr=[1, 2, 3, inf], tr=10)
        b = Weibull.fit([1, 2, 3, 5.0], c=[0, 0, 0, 1], tr=10)
        np.testing.assert_allclose(a.params, b.params)
        a = Weibull.fit(xl=[1, 2, 3, -inf], xr=[1, 2, 3, 5.0], tl=0.5)
        b = Weibull.fit([1, 2, 3, 5.0], c=[0, 0, 0, -1], tl=0.5)
        np.testing.assert_allclose(a.params, b.params)


def test_two_column_point_at_left_truncation_is_refused_like_one_column():
    with pytest.raises(ValueError, match="strictly less"):
        xcnt_handler([1, 2, 3], tl=1)
    with pytest.raises(ValueError, match="strictly less"):
        xcnt_handler([[1, 1], [2, 3]], tl=1)
    # An interval (xl, xr] may start at its truncation time.
    x, c, _, _ = xcnt_handler([[1, 2], [2, 3]], tl=1)
    np.testing.assert_array_equal(c, [2, 2])
    with pytest.raises(ValueError, match="left truncated values"):
        xcnt_handler([[1, 2], [2, 3]], tl=1.5)


# --- pairs, empty and scalar input -----------------------------------------


@pytest.mark.parametrize(
    "pair",
    [(2, 3), np.array([2, 3]), [2, 3]],
    ids=["tuple", "ndarray", "list"],
)
def test_any_pair_in_a_list_is_an_interval_row(pair):
    x, c, n, t = xcnt_handler([1, pair, 4])
    np.testing.assert_array_equal(x, [[1, 1], [2, 3], [4, 4]])
    np.testing.assert_array_equal(c, [0, 2, 0])
    Weibull.fit([1, pair, 4, 5])


def test_bad_elements_give_a_clear_error():
    with pytest.raises(ValueError, match="no more than length 2"):
        xcnt_handler([1, (2, 3, 4)])
    with pytest.raises(ValueError, match="no more than length 2"):
        xcnt_handler([1, []])


def test_empty_x_is_refused():
    for fit in (KaplanMeier.fit, Weibull.fit, Turnbull.fit):
        with pytest.raises(ValueError, match="'x' is empty"):
            fit([])
    with pytest.raises(ValueError, match="'x' is empty"):
        SurpyvalData([])


def test_scalar_x_and_zero_dimensional_bounds():
    x, c, n, t = xcnt_handler(5)
    np.testing.assert_array_equal(x, [5.0])
    np.testing.assert_array_equal(t, [[-inf, inf]])
    x, c, n, t = xcnt_handler(np.float64(5), c=1, n=2, tl=np.array(1.0))
    np.testing.assert_array_equal(c, [1])
    np.testing.assert_array_equal(n, [2])
    np.testing.assert_array_equal(t, [[1.0, inf]])
    a = Weibull.fit([1, 2, 3.0], tl=np.array(0.5), tr=np.array(9.0))
    b = Weibull.fit([1, 2, 3.0], tl=0.5, tr=9.0)
    np.testing.assert_allclose(a.params, b.params)


# --- xcn_to_fs --------------------------------------------------------------


def test_xcn_to_fs_validates():
    with pytest.raises(ValueError, match="whole numbers"):
        xcn_to_fs([1, 2], [0, 0], [1.7, 1])
    with pytest.raises(ValueError, match="whole numbers"):
        xcn_to_fs([1, 2], [0, 0], [-1, 1])
    with pytest.raises(ValueError, match="'n' must be the same length"):
        xcn_to_fs([1, 2], [0, 0], [1])
    with pytest.raises(ValueError, match="'c' must be the same length"):
        xcn_to_fs([1, 2, 3], [0, 0])
    with pytest.raises(ValueError, match="equal ends"):
        xcn_to_fs([[1, 2], [3, 3]], [0, 0])


def test_xcn_to_fs_takes_the_handlers_two_column_output():
    f, s = xcn_to_fs(*xcnt_handler([1, [2, 3], 4, 4], [0, 2, 1, 1])[:3])
    np.testing.assert_array_equal(f, [1])
    np.testing.assert_array_equal(s, [4, 4])
    f, s = xcn_to_fs([1.0, 2.0], [0, 1], [2.0, 1.0])
    np.testing.assert_array_equal(f, [1, 1])
    np.testing.assert_array_equal(s, [2])


# --- fsli_handler messages --------------------------------------------------


def test_fsli_handler_messages():
    with pytest.raises(ValueError, match="must be two-dimensional"):
        fsli_handler(i=[1, 2])
    with pytest.raises(ValueError) as err:
        fsli_handler(i=[[2, 1]])
    assert "  " not in str(err.value)
