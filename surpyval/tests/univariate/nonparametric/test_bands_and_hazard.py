import numpy as np
import pytest
from scipy.stats import kstwobign, norm

import surpyval
import surpyval as sp
from surpyval import KaplanMeier
from surpyval.tests._helpers import (
    TURNBULL_MIXED_CENSORING,
    fit_turnbull_quietly,
    small_kaplan_meier,
)
from surpyval.univariate.nonparametric.nonparametric import NonParametric


def _censored_model():
    x = np.array([4.0, 7, 9, 13, 16, 21, 28, 33, 41, 50])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    return surpyval.KaplanMeier.fit(x, c=c)


def test_hall_wellner_critical_value_approaches_kolmogorov():
    # Over the whole [0, 1] interval the supremum of |Brownian bridge|
    # has the Kolmogorov distribution; its 0.95 quantile is ~1.358.
    cv = NonParametric._band_critical_value(
        1e-4, 1 - 1e-4, 0.05, standardized=False
    )
    assert np.isclose(cv, 1.358, atol=0.001)


def test_band_contains_pointwise_bounds():
    model = _censored_model()
    pw = model.cb(model.x)
    for method in ["hall-wellner", "nair"]:
        # On the pointwise bounds' (log(-log)) scale: the default arcsine
        # band (#390) is on another, so it need not contain them.
        band = model.band(method=method, bound_type="exp")
        finite = np.isfinite(band[:, 0]) & np.isfinite(pw[:, 0])
        assert np.all(band[finite, 0] <= pw[finite, 0] + 1e-9)
        assert np.all(band[finite, 1] >= pw[finite, 1] - 1e-9)


def test_band_reproducible_with_fixed_seed():
    model = _censored_model()
    assert np.allclose(model.band(), model.band(), equal_nan=True)


def test_band_normal_bound_type_shape():
    model = _censored_model()
    band = model.band(bound_type="normal")
    assert band.shape == (model.x.size, 2)


def test_band_invalid_args():
    model = _censored_model()
    with pytest.raises(ValueError):
        model.band(method="nope")
    with pytest.raises(ValueError):
        model.band(bound_type="regular")


def test_band_fit_from_ecdf_raises():
    model = NonParametric.fit_from_ecdf(
        np.array([1.0, 2, 3]), np.array([0.9, 0.5, 0.1])
    )
    with pytest.raises(ValueError, match="variance"):
        model.band()


def test_band_simultaneous_coverage():
    # The whole-curve coverage of the band should be close to nominal,
    # and markedly better than the pointwise bounds which are not
    # designed for simultaneous coverage.
    rng = np.random.default_rng(3)
    n, reps = 50, 150
    pw_cover = 0
    band_cover = 0
    for _ in range(reps):
        x = rng.exponential(1.0, n)
        cens = rng.exponential(2.0, n)
        obs = np.minimum(x, cens)
        c = (x > cens).astype(int)
        model = surpyval.KaplanMeier.fit(obs, c=c)
        grid = model.x
        S_true = np.exp(-grid)
        pw = model.cb(grid)
        band = model.band()
        f = np.isfinite(band[:, 0])
        pw_cover += np.all((pw[f, 0] <= S_true[f]) & (S_true[f] <= pw[f, 1]))
        band_cover += np.all(
            (band[f, 0] <= S_true[f]) & (S_true[f] <= band[f, 1])
        )
    assert band_cover / reps > 0.88
    assert band_cover / reps > pw_cover / reps


def test_smoothed_hazard_recovers_constant():
    rng = np.random.default_rng(4)
    x = rng.exponential(1.0, 2000)
    model = surpyval.NelsonAalen.fit(x)
    h = model.smoothed_hf([0.3, 0.6, 1.0, 1.5], bandwidth=0.5)
    assert np.allclose(h, 1.0, atol=0.15)


def test_smoothed_hazard_recovers_linear_weibull():
    # Weibull with shape 2 has hazard h(t) = 2t.
    x = surpyval.Weibull.random(4000, 1.0, 2.0)
    model = surpyval.NelsonAalen.fit(x)
    t = np.array([0.5, 1.0, 1.5])
    h = model.smoothed_hf(t, bandwidth=0.3)
    assert np.allclose(h, 2 * t, rtol=0.15)


def test_smoothed_hazard_nan_outside_range():
    model = surpyval.NelsonAalen.fit(np.array([1.0, 2, 3, 4, 5]))
    h = model.smoothed_hf([-1.0, 1e6])
    assert np.isnan(h).all()


def test_smoothed_hazard_bad_bandwidth():
    model = surpyval.NelsonAalen.fit(np.array([1.0, 2, 3]))
    with pytest.raises(ValueError):
        model.smoothed_hf([2.0], bandwidth=-1)


# ---------------------------------------------------------------------------
# #420: the band's critical value at a large ``alpha_ci``;
# #390: the equal precision band starts at the first event and
# the arcsine band is Klein and Moeschberger's.
# ---------------------------------------------------------------------------


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


def _censored_weibull(seed, n):
    true = sp.Weibull.from_params([10.0, 1.5])
    rng = np.random.default_rng(seed)
    t = true.qf(rng.uniform(size=n))
    cens = rng.uniform(0, 25, n)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    return true, sp.KaplanMeier.fit(x, c=c)


def test_equal_precision_band_covers_a_from_a_tenth_to_nine_tenths():
    # #390: the equal precision band's standardized boundary is unbounded
    # as a = N sigma^2 / (1 + N sigma^2) nears 0 or 1, and there the
    # estimate rests on a few failures or a few at risk; over the first to
    # the last event it covered 0.87-0.89 for 0.95 (log(-log) scale) and
    # 0.93 (arcsine). By default it now covers 0.1 <= a <= 0.9, NaN
    # outside; the Hall-Wellner band still covers every event.
    _, model = _censored_weibull(3, 60)
    N = 60.0
    a = N * model.greenwood / (1 + N * model.greenwood)
    with np.errstate(all="ignore"):
        valid = np.isfinite(a) & (a > 0) & (model.R > 0) & (model.R < 1)
    inside = valid & (a >= 0.1) & (a <= 0.9)
    assert inside.any() and (valid & ~inside).any()
    for bound_type in ("arcsine", "exp", "normal"):
        nair = model.band(method="nair", bound_type=bound_type)
        assert np.isfinite(nair[inside]).all()
        assert np.isnan(nair[~inside]).all()
        hw = model.band(bound_type=bound_type)
        assert np.isfinite(hw[valid]).all()
    # Its critical value is that of the range it covers.
    crit = NonParametric._band_critical_value(
        a[inside].min(), a[inside].max(), 0.05, True
    )
    half = crit * np.sqrt(model.greenwood[inside])
    R = model.R[inside]
    np.testing.assert_allclose(
        model.band(method="nair", bound_type="normal")[inside],
        np.c_[R - half * R, R + half * R],
        rtol=1e-12,
    )


def test_band_x_range_sets_the_times_it_covers():
    # x_range = (t_L, t_U): the band over those times, its critical value
    # from a at both ends, NaN outside; for the equal precision band it
    # can reach the first event, as Klein and Moeschberger's t_L can.
    _, model = _censored_weibull(3, 60)
    first, last = model.x[0], model.x[-1]
    t_l, t_u = np.quantile(model.x, [0.3, 0.6])
    q = np.array([first, t_l, (t_l + t_u) / 2, t_u, last])
    for method in ("hall-wellner", "nair"):
        band = model.band(q, method=method, x_range=(t_l, t_u))
        assert np.isnan(band[[0, 4]]).all()
        assert np.isfinite(band[1:4]).all()
        # A narrower range needs a smaller critical value.
        wide = model.band(q, method=method, x_range=(first, last))
        assert np.all(band[1:4, 0] > wide[1:4, 0])
        assert np.isfinite(wide[0]).all()
    for bad in [(5.0, 1.0), (1.0,), "ab", 3.0]:
        with pytest.raises(ValueError, match="'x_range' must be"):
            model.band(x_range=bad)
    with pytest.raises(ValueError, match="no observations in x_range"):
        model.band(x_range=(-5.0, -1.0))


def test_equal_precision_band_with_no_a_in_range_says_so():
    # One failure among 30: a = 1/30 at most, below the default range.
    x = np.arange(1.0, 31.0)
    c = np.ones(30, int)
    c[0] = 0
    model = sp.KaplanMeier.fit(x, c=c)
    with pytest.raises(ValueError, match="x_range, or use the Hall-Wellner"):
        model.band(method="nair")
    assert np.isfinite(model.band(method="nair", x_range=(1, 30))[0]).all()


def test_equal_precision_band_old_default_missed_the_first_event():
    # #390: on the log(-log) scale over the first to the last event (the
    # default until v0.22) the band pulled the cumulative hazard at the
    # first event down by a factor e^-c (c ~ 3.2, to 4% of 1/n), and a
    # first failure early enough for its step to be far from normal
    # escaped it: here the true sf at the first event, 0.99974, is above
    # the band's upper end, 0.99920. On the arcsine scale the band
    # reaches 1 there; by default it does not claim the first events at
    # all (a < 0.1).
    true, model = _censored_weibull(21, 50)
    first = model.x[0]
    full = (first, model.x[-1])
    old = model.band(first, method="nair", bound_type="exp", x_range=full)
    assert true.sf(first) > old[1]
    lower, upper = model.band(first, method="nair", x_range=full)
    assert lower <= true.sf(first) <= upper
    assert np.isnan(model.band(first, method="nair")).all()


@pytest.mark.parametrize("method", ["hall-wellner", "nair"])
def test_arcsine_band_is_klein_and_moeschberger(method):
    # Klein and Moeschberger (2003), Section 4.4: sin^2 of arcsin(sqrt(S))
    # -+ half the band's half width (c sigma for equal precision,
    # c (1 + N sigma^2) / sqrt(N) for Hall-Wellner, sigma^2 the Greenwood
    # sum) times sqrt(S / (1 - S)), cut at 0 and pi / 2.
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    c = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    model = sp.KaplanMeier.fit(x, c=c)
    R, s2, N = model.R, model.greenwood, 10.0
    valid = np.isfinite(s2) & (s2 > 0) & (R > 0) & (R < 1)
    a = N * s2 / (1 + N * s2)
    crit = NonParametric._band_critical_value(
        a[valid].min(), a[valid].max(), 0.05, method == "nair"
    )
    if method == "nair":
        half = crit * np.sqrt(s2)
    else:
        half = crit * (1 + N * s2) / np.sqrt(N)
    with np.errstate(all="ignore"):
        d = 0.5 * half * np.sqrt(R / (1 - R))
        angle = np.arcsin(np.sqrt(R))
    want = np.c_[
        np.sin(np.maximum(angle - d, 0)) ** 2,
        np.sin(np.minimum(angle + d, np.pi / 2)) ** 2,
    ]
    want[~valid] = np.nan
    band = model.band(method=method)
    np.testing.assert_allclose(band, want, rtol=1e-12, equal_nan=True)
    assert np.all(band[valid] >= 0) and np.all(band[valid] <= 1)


def test_band_over_a_single_time_is_the_pointwise_arcsine_interval():
    # One failure, then censoring: the supremum is one normal variable, so
    # both bands reduce to the pointwise interval on their scale.
    model = sp.KaplanMeier.fit([1.0, 2, 3, 4], c=[0, 1, 1, 1])
    S, s2 = 0.75, 1 / 12
    z = 1.959963984540054
    d = 0.5 * z * np.sqrt(s2) * np.sqrt(S / (1 - S))
    angle = np.arcsin(np.sqrt(S))
    want = [np.sin(angle - d) ** 2, min(np.sin(angle + d) ** 2, 1.0)]
    for method in ("hall-wellner", "nair"):
        np.testing.assert_allclose(
            model.band([1, 2.5], method=method), [want, want], atol=1e-6
        )


def test_band_refuses_an_unknown_bound_type_naming_the_choices():
    model = sp.KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="'arcsine', 'exp' or 'normal'"):
        model.band(bound_type="log")


# ---------------------------------------------------------------------------
# ``smoothed_hf`` works on Turnbull models (they carry ``H``).
# ---------------------------------------------------------------------------


class TestSmoothedHazard:
    @pytest.mark.parametrize(
        "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
    )
    def test_smoothed_hf_on_turnbull(self, estimator):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator=estimator
        )
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(model.H, -np.log(model.R))
        h = model.smoothed_hf([4.0, 7.0, 10.0], bandwidth=3)
        assert np.isfinite(h).all() and (h >= 0).all()

    def test_matches_kaplan_meier_on_right_censored_data(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        c = [0, 1, 0, 0, 1, 0, 0, 1]
        tb = fit_turnbull_quietly(x=x, c=c, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c)
        t = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(
            tb.smoothed_hf(t, bandwidth=2),
            km.smoothed_hf(t, bandwidth=2),
            atol=1e-9,
        )

    def test_restored_legacy_turnbull_dict(self):
        # Turnbull models serialised before they carried H stored it as
        # None; the restored model still supports smoothed_hf.
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator="Kaplan-Meier"
        )
        payload = model.to_dict()
        payload["H"] = None
        restored = NonParametric.from_dict(payload)
        np.testing.assert_allclose(
            restored.smoothed_hf([4.0, 7.0], bandwidth=3),
            model.smoothed_hf([4.0, 7.0], bandwidth=3),
        )


# ---------------------------------------------------------------------------
# The band critical values; ``smoothed_hf``'s default bandwidth
# on one value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_hall_wellner_critical_value_is_kolmogorov_over_whole_range(alpha):
    cv = NonParametric._band_critical_value(0.0, 1.0, alpha, False)
    assert cv == pytest.approx(kstwobign.ppf(1 - alpha), abs=1e-6)


def test_band_critical_values_at_a_single_point():
    z = norm.ppf(0.975)
    hw = NonParametric._band_critical_value(0.3, 0.3, 0.05, False)
    assert hw == pytest.approx(z * np.sqrt(0.3 * 0.7), rel=1e-4)
    ep = NonParametric._band_critical_value(0.3, 0.3, 0.05, True)
    assert ep == pytest.approx(z, rel=1e-4)


def test_band_critical_values_are_symmetric_in_a():
    # B(a) and B(1 - a) have the same law.
    for standardized in (False, True):
        lo = NonParametric._band_critical_value(0.05, 0.4, 0.05, standardized)
        hi = NonParametric._band_critical_value(0.6, 0.95, 0.05, standardized)
        assert lo == pytest.approx(hi, rel=1e-8)


def test_equal_precision_critical_value():
    # Klein & Moeschberger table C.4 range; Miller-Siegmund's
    # approximation gives a tail of about 0.05 here too.
    cv = NonParametric._band_critical_value(0.1, 0.9, 0.05, True)
    assert cv == pytest.approx(3.052, abs=2e-3)


def test_band_on_a_narrow_range_does_not_crash():
    one_event = sp.KaplanMeier.fit([1, 2, 3], c=[0, 1, 1]).band()
    assert np.isfinite(one_event).all()
    x = np.r_[[1, 2, 3], np.full(10000, 10)]
    c = np.r_[[0, 0, 0], np.ones(10000, int)]
    band = sp.KaplanMeier.fit(x, c=c).band()
    assert np.isfinite(band[:3]).all()


def test_band_sim_arguments_are_removed():
    with pytest.raises(TypeError, match="n_sims"):
        small_kaplan_meier().band(n_sims=100)


def test_smoothed_hf_default_bandwidth_on_one_value():
    model = sp.KaplanMeier.fit([2, 2, 2])
    with pytest.raises(ValueError, match="single distinct value"):
        model.smoothed_hf([2])
