import numpy as np
import pytest
from scipy.stats import norm

import surpyval
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
    # With right censoring the survival estimate does not reach zero;
    # sampling renormalises over the observed event masses.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    c = np.array([0, 0, 0, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    samples = model.random(1000)
    assert np.isin(samples, model.x).all()


def test_scalar_input_to_hf_and_df():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.NelsonAalen.fit(x)
    # Must not raise; with a single point there is no neighbouring step
    # so the rate is undefined.
    assert model.hf(2).shape == (1,)
    assert model.df(2).shape == (1,)
    # Array input remains well defined
    assert np.isfinite(model.hf([1.5, 2.5, 3.5])).all()


def test_fit_from_ecdf_cb_raises_informative_error():
    model = NonParametric.fit_from_ecdf(
        np.array([1.0, 2.0, 3.0]), np.array([0.9, 0.5, 0.1])
    )
    with pytest.raises(ValueError, match="variance"):
        model.cb([2.0])
