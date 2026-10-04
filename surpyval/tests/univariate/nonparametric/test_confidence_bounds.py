import numpy as np
import pytest
from scipy.stats import norm

import surpyval
from surpyval.tests._helpers import (
    TURNBULL_MIXED_CENSORING,
    fit_turnbull_quietly,
    small_kaplan_meier,
)
from surpyval.univariate.nonparametric import (
    greenwood_variance,
    nelson_aalen_variance,
)
from surpyval.univariate.nonparametric.nonparametric import NonParametric


def test_kaplan_meier_greenwood_variance():
    # Greenwood's formula: cumsum(d / (r * (r - d))). With no censoring
    # the variance is undefined (NaN) at the last point where d == r.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    r = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    d = np.ones(5)
    with np.errstate(all="ignore"):
        expected = np.cumsum(d / (r * (r - d)))
    assert np.allclose(model.greenwood[:-1], expected[:-1], atol=1e-12)
    assert np.isnan(model.greenwood[-1])


def test_nelson_aalen_aalen_variance():
    # Aalen's (Poisson) variance: cumsum(d / r**2). Klein (1991).
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.NelsonAalen.fit(x)
    r = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    d = np.ones(5)
    expected = np.cumsum(d / r**2)
    assert np.allclose(model.greenwood, expected, atol=1e-12)


def test_fleming_harrington_tie_corrected_variance():
    # Tie-split variance: sum(1 / (r - j)**2 for j in 0 .. d - 1),
    # mirroring the FH hazard increment sum(1 / (r - j)).
    x = [1, 2, 3, 4]
    n = [3, 2, 4, 1]
    model = surpyval.FlemingHarrington.fit(x=x, n=n)
    expected = np.cumsum(
        [
            1.0 / 10**2 + 1.0 / 9**2 + 1.0 / 8**2,
            1.0 / 7**2 + 1.0 / 6**2,
            1.0 / 5**2 + 1.0 / 4**2 + 1.0 / 3**2 + 1.0 / 2**2,
            1.0,
        ]
    )
    assert np.allclose(model.greenwood, expected, atol=1e-12)


def test_fleming_harrington_variance_equals_nelson_aalen_without_ties():
    x = np.array([2.0, 4.0, 6.0, 8.0, 9.0, 13.0, 17.0, 22.0, 30.0, 45.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    fh = surpyval.FlemingHarrington.fit(x, c=c)
    na = surpyval.NelsonAalen.fit(x, c=c)
    assert np.allclose(fh.greenwood, na.greenwood, atol=1e-12)


def test_cb_exp_bounds_match_manual_formula():
    # The default ('exp') bounds are the log(-log) transformed interval:
    # exp(-exp(log(-log(R)) -/+ z * sqrt(var) / -log(R)))
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    z = norm.ppf(0.975)
    R = model.R
    var = model.greenwood
    with np.errstate(all="ignore"):
        theta = np.log(-np.log(R))
        se = np.sqrt(var / np.log(R) ** 2)
        lower = np.exp(-np.exp(theta + z * se))
        upper = np.exp(-np.exp(theta - z * se))
    cb = model.cb(model.x)
    finite = np.isfinite(var)
    assert np.allclose(cb[finite, 0], lower[finite], atol=1e-12)
    assert np.allclose(cb[finite, 1], upper[finite], atol=1e-12)


def test_cb_normal_bounds_match_manual_formula():
    # The 'normal' bounds are R -/+ z * sqrt(var * R**2)
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0])
    model = surpyval.NelsonAalen.fit(x, c=c)
    z = norm.ppf(0.975)
    R = model.R
    se = np.sqrt(model.greenwood * R**2)
    cb = model.cb(model.x, bound_type="normal")
    assert np.allclose(cb[:, 0], R - z * se, atol=1e-12)
    assert np.allclose(cb[:, 1], R + z * se, atol=1e-12)


def test_cb_hf_bounds_match_log_transformed_hazard_interval():
    # On the cumulative hazard the 'exp' bounds are equivalent to the
    # log-transformed interval H * exp(-/+ z * sqrt(var) / H), the
    # standard interval for the Nelson-Aalen estimator (and the one
    # used by lifelines).
    x = np.array([2.0, 4.0, 6.0, 8.0, 9.0, 13.0, 17.0, 22.0, 30.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    model = surpyval.NelsonAalen.fit(x, c=c)
    z = norm.ppf(0.975)
    H = -np.log(model.R)
    se = np.sqrt(model.greenwood)
    expected_lower = H * np.exp(-z * se / H)
    expected_upper = H * np.exp(z * se / H)
    cb = model.cb(model.x, on="Hf")
    assert np.allclose(cb[:, 0], expected_lower, atol=1e-12)
    assert np.allclose(cb[:, 1], expected_upper, atol=1e-12)


def test_cb_two_sided_ordering_and_consistency():
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    x_test = np.array([5.0, 10.0, 20.0])

    cb_sf = model.cb(x_test, on="sf")
    cb_ff = model.cb(x_test, on="ff")
    cb_hf = model.cb(x_test, on="Hf")

    # Columns are [lower, upper] and bracket the point estimate
    assert (cb_sf[:, 0] <= model.sf(x_test)).all()
    assert (model.sf(x_test) <= cb_sf[:, 1]).all()
    assert (cb_hf[:, 0] <= model.Hf(x_test)).all()
    assert (model.Hf(x_test) <= cb_hf[:, 1]).all()

    # ff bounds are the complement of the sf bounds
    assert np.allclose(cb_ff, 1 - cb_sf[:, ::-1], atol=1e-12)

    # One sided bounds match the relevant side of a one sided interval
    lower = model.cb(x_test, bound="lower")
    upper = model.cb(x_test, bound="upper")
    assert (lower <= model.sf(x_test)).all()
    assert (upper >= model.sf(x_test)).all()


def test_cb_last_point_defined_for_na_and_fh():
    # The NA and FH variances remain finite when d == r so, unlike
    # Kaplan-Meier, the bounds at the last point need no fill values.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for est in [surpyval.NelsonAalen, surpyval.FlemingHarrington]:
        model = est.fit(x)
        assert np.isfinite(model.greenwood).all()
        cb = model.cb(model.x)
        assert np.isfinite(cb).all()


def test_cb_invalid_bound_type_raises():
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        model.cb([2.0], bound_type="regular")


def test_cb_dist_t_removed():
    # The unjustified Student-t heuristic was removed; only the normal ('z')
    # statistic is supported, and 't' now raises an informative error
    # pointing to the principled alternatives.
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="bootstrap_cb"):
        model.cb([2.0, 3.0], dist="t")
    with pytest.raises(ValueError, match="'dist' must be 'z'"):
        model.R_cb([2.0, 3.0], dist="t")
    # 'z' (the default) is unaffected.
    assert model.cb([2.0, 3.0], dist="z").shape == (2, 2)
    assert np.allclose(model.cb([2.0, 3.0]), model.cb([2.0, 3.0], dist="z"))


def test_cb_turnbull_uses_selected_estimator_variance():
    left = np.array([1, 8, 8, 7, 7, 17, 37, 46, 46, 45.0])
    right = np.array([7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf])
    for est in ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]:
        model = surpyval.Turnbull.fit(
            xl=left, xr=right, turnbull_estimator=est
        )
        cb = model.cb([10.0, 20.0])
        assert np.isfinite(cb).all()
        assert (cb[:, 0] <= cb[:, 1]).all()


def test_cb_coverage_of_default_bounds():
    # Simulation check that the default (exponential Greenwood, z) two
    # sided 95% bounds cover the true survival function at close to the
    # nominal rate. Loose tolerance to keep the runtime low.
    rng = np.random.default_rng(123)
    n, reps = 40, 200
    t_eval = np.array([0.2877, 0.6931])
    S_true = np.exp(-t_eval)
    covered = np.zeros((reps, t_eval.size), dtype=bool)
    for i in range(reps):
        lifetimes = rng.exponential(1.0, n)
        censoring = rng.exponential(2.0, n)
        obs = np.minimum(lifetimes, censoring)
        c = (lifetimes > censoring).astype(int)
        model = surpyval.KaplanMeier.fit(obs, c=c)
        cb = model.cb(t_eval)
        covered[i] = (cb[:, 0] <= S_true) & (S_true <= cb[:, 1])
    coverage = covered.mean(axis=0)
    assert (coverage > 0.88).all()
    assert (coverage <= 1.0).all()


def test_random_samples_with_estimated_probabilities():
    # random() must respect the estimated probability masses, not
    # sample uniformly from the unique observed values.
    x = np.array([1.0] * 8 + [2.0, 3.0])
    model = surpyval.KaplanMeier.fit(x)
    np.random.seed(42)
    samples = model.random(20000)
    freqs = np.array([(samples == v).mean() for v in model.x])
    assert np.allclose(freqs, [0.8, 0.1, 0.1], atol=0.02)


def test_random_with_right_censoring():
    # With right censoring the survival estimate does not reach zero: the
    # mass it leaves beyond its last time is drawn as inf, so the draws
    # follow the estimate's own sf.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    c = np.array([0, 0, 0, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    samples = model.random(40_000, random_state=0)
    finite = samples[np.isfinite(samples)]
    assert np.isin(finite, model.x).all()
    assert np.all(np.isposinf(samples[~np.isfinite(samples)]))
    left = float(np.ravel(model.sf(np.array([4.0])))[0])
    assert np.isinf(samples).mean() == pytest.approx(left, abs=0.01)
    for t in (1.0, 2.0, 3.0):
        share = (samples <= t).mean()
        want = float(np.ravel(model.ff(np.array([t])))[0])
        assert share == pytest.approx(want, abs=0.01)


def test_scalar_input_to_hf_and_df():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.NelsonAalen.fit(x)
    # Must not raise; with a single point there is no neighbouring step
    # so the rate is undefined.
    # A scalar query gives a scalar (principle 7).
    assert np.shape(model.hf(2)) == ()
    assert np.shape(model.df(2)) == ()
    # Array input remains well defined
    assert np.isfinite(model.hf([1.5, 2.5, 3.5])).all()


def test_fit_from_ecdf_cb_raises_informative_error():
    model = NonParametric.fit_from_ecdf(
        np.array([1.0, 2.0, 3.0]), np.array([0.9, 0.5, 0.1])
    )
    with pytest.raises(ValueError, match="variance"):
        model.cb([2.0])


# ---------------------------------------------------------------------------
# Greenwood's variance treats round-off in the proportion
# failing as exact, so the last value is undefined rather than
# ~1e14.
# ---------------------------------------------------------------------------


class TestGreenwoodRoundOff:
    def test_last_value_with_round_off_is_undefined(self):
        # The EM's last r and d, equal but for round-off (and fractional).
        r = np.array([3.0, 1.2142857142857106])
        d = np.array([1.0, 1.2142857142857142])
        var = greenwood_variance(r, d)
        assert var[0] == pytest.approx(1 / 6)
        assert np.isnan(var[1])
        # Exactly as for exactly equal counts.
        exact = greenwood_variance(np.array([3.0, 2.0]), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(np.isnan(var), np.isnan(exact))

    def test_zero_count_with_round_off_is_no_event(self):
        var = greenwood_variance(
            np.array([17.0, 17.0]), np.array([2e-16, 1.0])
        )
        assert var[0] == 0.0
        assert var[1] == pytest.approx(1 / (17 * 16))
        var = nelson_aalen_variance(np.array([17.0]), np.array([2e-16]))
        assert var[0] == 0.0

    def test_turnbull_last_value_bounds(self):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator="Kaplan-Meier"
        )
        assert np.isnan(model.greenwood[-1])
        assert np.nanmax(np.abs(model.greenwood)) < 1e3
        # Undefined at the last value: lower 0, upper the last finite one.
        lower, upper = model.cb(12)
        assert lower == 0.0
        assert upper == pytest.approx(model.cb(11)[1])
        # The estimate there is 0, not round-off below it.
        assert model.sf(12) == 0.0

    def test_integer_counts_unchanged(self):
        r = np.array([10.0, 8.0, 5.0, 2.0])
        d = np.array([2.0, 1.0, 3.0, 2.0])
        with np.errstate(all="ignore"):
            raw = np.cumsum(
                np.where(
                    np.isfinite(d / (r * (r - d))), d / (r * (r - d)), np.nan
                )
            )
        np.testing.assert_array_equal(greenwood_variance(r, d), raw)


# ---------------------------------------------------------------------------
# ``cb`` and its relatives refuse an unknown ``on`` or
# ``bound``.
# ---------------------------------------------------------------------------


def test_cb_rejects_unknown_on():
    with pytest.raises(ValueError, match="'on'"):
        small_kaplan_meier().cb(2, on="hf")


def test_cb_and_friends_reject_unknown_bound():
    model = small_kaplan_meier()
    with pytest.raises(ValueError, match="'bound'"):
        model.cb(2, bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.R_cb(2, bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.plot(bound="both")
    with pytest.raises(ValueError, match="'bound'"):
        model.bootstrap_cb(2, bound="both", n_boot=5)


def test_666_cb_takes_alpha_ci_third_as_every_model_does():
    model = surpyval.KaplanMeier.fit(
        [5, 8, 12, 15, 20, 22, 30], [1, 0, 0, 1, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        model.cb([10.0], "sf", 0.1), model.cb([10.0], alpha_ci=0.1)
    )
    np.testing.assert_array_equal(
        model.cb([10.0], "sf", 0.1, "lower"),
        model.cb([10.0], alpha_ci=0.1, bound="lower"),
    )
    # the old order still reads as before, with a warning
    with pytest.warns(DeprecationWarning, match="old order"):
        old = model.cb([10.0], "sf", "lower", "step", 0.1)
    np.testing.assert_array_equal(
        old, model.cb([10.0], alpha_ci=0.1, bound="lower")
    )
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError, match="multiple values"):
            model.cb([10.0], "sf", "lower", bound="upper")
