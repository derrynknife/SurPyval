"""Regression tests for the non-parametric fixes of #390, #391, #408, #417,
#420 and #425."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import NonParametric
from surpyval.univariate.nonparametric import (
    fleming_harrington,
    kaplan_meier,
    nelson_aalen,
)

ESTIMATORS = (kaplan_meier, nelson_aalen, fleming_harrington)


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _no_warnings(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


# -- #425: a step with no one at risk and no events ---------------------------


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_no_one_at_risk_and_no_events_changes_nothing(estimator):
    # 0 / 0 is no change, as R's survfit carries the estimate: the
    # Kaplan-Meier and Nelson-Aalen used to drop to 0 (with a raw "invalid
    # value" warning), where the Fleming-Harrington kept its value.
    r, d = np.array([3.0, 0.0]), np.array([1.0, 0.0])
    R = _no_warnings(estimator, r, d)
    assert R[1] == R[0] and 0 < R[0] < 1


def test_the_three_estimators_agree_on_an_empty_step():
    r, d = np.array([3.0, 0.0, 2.0]), np.array([1.0, 0.0, 1.0])
    np.testing.assert_allclose(kaplan_meier(r, d), [2 / 3, 2 / 3, 1 / 3])
    H = np.cumsum([1 / 3, 0.0, 1 / 2])
    np.testing.assert_allclose(nelson_aalen(r, d), np.exp(-H))
    np.testing.assert_allclose(fleming_harrington(r, d), np.exp(-H))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_events_with_no_one_at_risk_are_refused(estimator):
    with pytest.raises(ValueError, match="no one at risk"):
        estimator(np.array([3.0, 0.0]), np.array([1.0, 1.0]))


def test_fleming_harrington_all_at_risk_failing_stays_above_zero():
    # Documented: the tie ladder is 1/3 + 1/2 + 1.
    np.testing.assert_allclose(
        fleming_harrington([3], [3]), np.exp(-(1 / 3 + 1 / 2 + 1))
    )


# -- #391: Turnbull with every row right truncated ----------------------------
# The Kaplan-Meier option reaches 0 where all the mass is placed; the
# default Fleming-Harrington one never reaches 0, and must agree with the
# same data without the right truncation.


def _km_turnbull(**kw):
    return _quiet(sp.Turnbull.fit, turnbull_estimator="Kaplan-Meier", **kw)


def test_turnbull_failure_at_its_right_truncation_time():
    # One failure at 1, observable up to 1: all the mass is at 1, so
    # sf(1) is 0. Turnbull gave 1 (ladder x = [1], R = [1]).
    model = _km_turnbull(x=[1.0], c=[0], tr=[1.0])
    assert model.sf([1.0])[0] == 0.0
    fh = _quiet(sp.Turnbull.fit, x=[1.0], c=[0], tr=[1.0])
    untruncated = _quiet(sp.Turnbull.fit, x=[1.0], c=[0])
    np.testing.assert_allclose(fh.sf([0.5, 1.0]), untruncated.sf([0.5, 1.0]))


def test_turnbull_two_failures_right_truncated_at_the_last():
    # Failures at 1 and 2, both observable up to 2: sf(2) is 0. Turnbull
    # gave 0.5; with the second row untruncated (tr = inf) it gave 0.
    x, c = [1.0, 2.0], [0, 0]
    model = _km_turnbull(x=x, c=c, tr=[2.0, 2.0])
    np.testing.assert_allclose(model.sf([1.0, 2.0]), [0.5, 0.0])
    fh = _quiet(sp.Turnbull.fit, x=x, c=c, tr=[2.0, 2.0])
    one = _quiet(sp.Turnbull.fit, x=x, c=c, tr=[2.0, np.inf])
    np.testing.assert_allclose(fh.sf([1.0, 2.0]), one.sf([1.0, 2.0]))


def test_turnbull_right_censored_inside_its_window():
    # Censored at 1 and observable up to 2: the event is in (1, 2], so
    # sf(2) is 0. Turnbull gave 1.
    model = _km_turnbull(x=[1.0], c=[1], tr=[2.0])
    np.testing.assert_allclose(model.sf([1.0, 2.0]), [1.0, 0.0])


def test_turnbull_left_censored_at_its_right_truncation_time():
    # Left censored at 1, observable up to 1: the ladder was empty and sf
    # raised IndexError (index -1 of an empty R).
    model = _km_turnbull(x=[1.0], c=[-1], tr=[1.0])
    np.testing.assert_allclose(model.sf([0.5, 1.0]), [1.0, 0.0])


def test_turnbull_right_truncated_matches_the_kaplan_meier():
    # Exact data, every row observable up to a time past the last failure
    # (so the truncation excludes nothing): the Turnbull-KM is the
    # Kaplan-Meier, to its last failure.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    model = _km_turnbull(x=x, c=np.zeros(4, int), tr=np.full(4, 5.0))
    km = sp.KaplanMeier.fit(x)
    np.testing.assert_allclose(model.sf(x), km.sf(x), atol=1e-8)


# -- #408: df where the estimate reaches zero ---------------------------------


def test_df_is_the_step_probability_where_the_estimate_reaches_zero():
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0])
    df = _no_warnings(model.df, [1.5, 2.5, 3.5])
    # Each step of 1 takes 1/3; past 3 the last jump is carried, as hf
    # carries the infinite jump to zero.
    np.testing.assert_allclose(df, [1 / 3, 1 / 3, 1 / 3], rtol=1e-12)
    np.testing.assert_allclose(_no_warnings(model.df, 3.5), 1 / 3)
    assert np.all(np.isposinf(model.hf([2.5, 3.5])))


@pytest.mark.parametrize("interp", ["step", "linear", "cubic"])
def test_df_is_finite_and_a_probability(interp):
    model = sp.KaplanMeier.fit([1, 2, 3, 4, 5])
    df = _no_warnings(model.df, [1.0, 2.0, 3.0, 4.0, 5.0], interp=interp)
    assert np.all(np.isfinite(df)) and np.all((df >= 0) & (df <= 1))


def test_df_is_the_drop_in_sf_over_the_step_hf_differences():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.array([0, 1, 0, 0, 1, 0, 0, 1])
    model = sp.KaplanMeier.fit(x, c=c)
    q = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5])
    sf, hf, df = model.sf(q), model.hf(q), model.df(q)
    # 0.5 repeats 1.5; 2.5 has no failure since 1.5 and takes 1.5's step.
    before = np.array([1.0, 1.0, 1.0, sf[2], sf[3], sf[4]])
    after = np.array([sf[1], sf[1], sf[1], sf[3], sf[4], sf[5]])
    np.testing.assert_allclose(df, before - after, rtol=1e-12)
    # The same step as hf: df = S (1 - exp(-hf)).
    np.testing.assert_allclose(df, before * -np.expm1(-hf), rtol=1e-12)


def test_df_with_a_support_is_zero_before_the_first_value():
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0]).set_support(0, 10)
    df = _no_warnings(model.df, [-1.0, 0.5, 2.5, 5.0])
    # From 0.5 (sf 1) to 2.5 (sf 1/3), then to 5 (sf 0).
    np.testing.assert_allclose(df, [np.nan, 0.0, 2 / 3, 1 / 3])


# -- #417: cubic bounds close onto sf -----------------------------------------


def _km_ending_at_zero():
    # The estimate reaches 0 at the last time, where the variance is
    # undefined (the registry's Kaplan-Meier fixture).
    from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted

    return fitted(CASE_BY_NAME["KaplanMeier"])


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
def test_cubic_bounds_close_onto_sf(bound_type):
    model = _km_ending_at_zero()
    x = np.array([2.0, 6.0, 10.0, 13.0])
    sf = model.sf(x, interp="cubic")
    cb = _no_warnings(
        model.cb, x, interp="cubic", alpha_ci=1 - 1e-6, bound_type=bound_type
    )
    # 13 is between the last two times with a variance: the upper bound
    # there was 0.1873 against sf 0.1948.
    np.testing.assert_allclose(cb[:, 0], sf, rtol=1e-5)
    np.testing.assert_allclose(cb[:, 1], sf, rtol=1e-5)


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
def test_cubic_bounds_hold_sf_between_them(bound_type):
    model = _km_ending_at_zero()
    x = np.linspace(model.x[0], model.x[-1], 201)
    sf = model.sf(x, interp="cubic")
    cb = model.cb(x, interp="cubic", bound_type=bound_type)
    assert np.all(cb[:, 0] <= sf + 1e-12) and np.all(sf <= cb[:, 1] + 1e-12)
    if bound_type == "exp":
        assert np.all((cb >= 0) & (cb <= 1))


def test_linear_bounds_are_unchanged():
    # For the linear kind the estimate plus the interpolated distance is
    # the interpolated bound itself.
    model = _km_ending_at_zero()
    x = np.linspace(model.x[0], model.x[-2], 50)
    got = model.R_cb(x, interp="linear")
    knots = model.R_cb(model.x)
    np.testing.assert_allclose(got[:, 0], np.interp(x, model.x, knots[:, 0]))
    np.testing.assert_allclose(got[:, 1], np.interp(x, model.x, knots[:, 1]))


# -- #420: band critical value at a large alpha_ci ----------------------------


def test_critical_value_at_alpha_ci_near_one_is_bounded():
    # 1 - 1e-6 used to ask for a 158 TiB grid; the critical value of the
    # equal precision band over a narrow range is small but not zero.
    c = NonParametric._band_critical_value(0.3, 0.31, 1 - 1e-6, True)
    assert 0 < c < 0.1


def test_critical_value_decreases_with_alpha_ci():
    c = [
        NonParametric._band_critical_value(0.3, 0.31, alpha, False)
        for alpha in (0.05, 0.5, 0.99)
    ]
    assert c[0] > c[1] > c[2] > 0


@pytest.mark.parametrize("alpha_ci", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_band_refuses_alpha_ci_outside_the_open_unit_interval(alpha_ci):
    model = sp.KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model.band(alpha_ci=alpha_ci)


# -- #390: the equal precision band's range -----------------------------------


def _km_ci_data():
    rng = np.random.default_rng(3)
    n = 60
    t = sp.Weibull.from_params([10, 1.5]).qf(rng.uniform(size=n))
    cens = rng.uniform(0, 25, n)
    return np.round(np.minimum(t, cens), 3), (cens < t).astype(int)


def test_equal_precision_band_starts_at_a_of_one_tenth():
    x, c = _km_ci_data()
    model = sp.KaplanMeier.fit(x, c=c)
    n = len(x)
    a = n * model.greenwood / (1 + n * model.greenwood)
    band = model.band(method="nair")
    defined = np.isfinite(band[:, 0])
    first = np.flatnonzero(defined)[0]
    assert a[first] >= 0.1 and a[first - 1] < 0.1
    # The Hall-Wellner band still starts at the first event.
    assert np.isfinite(model.band()[np.flatnonzero(model.d)[0]]).all()


def test_equal_precision_band_matches_km_ci():
    # R's km.ci (0.5-6), method="logep", tl = 2.267 (where a reaches 0.1)
    # and tu = 13.77 (its default, the last event before the estimate
    # reaches 0). km.ci interpolates its critical value in Klein and
    # Moeschberger's table, so the bands agree to about 0.003.
    x, c = _km_ci_data()
    model = sp.KaplanMeier.fit(x, c=c)
    t = np.array([2.267, 2.442, 2.925, 3.117, 12.644, 13.011, 13.77])
    km_ci = np.array(
        [
            [0.6715, 0.9698],
            [0.6505, 0.9608],
            [0.6297, 0.9512],
            [0.6092, 0.9410],
            [0.0248, 0.3974],
            [0.0117, 0.3593],
            [0.0036, 0.3184],
        ]
    )
    np.testing.assert_allclose(model.band(t, method="nair"), km_ci, atol=3e-3)


def test_equal_precision_band_refused_below_one_tenth():
    # One failure among 50: a is 1 / 50 / (1 + 1 / 49) ~ 0.02.
    x = np.arange(1.0, 51.0)
    c = np.ones(50, dtype=int)
    c[0] = 0
    model = sp.KaplanMeier.fit(x, c=c)
    with pytest.raises(ValueError, match="hall-wellner"):
        model.band(method="nair")
    assert np.isfinite(model.band()[0]).all()
