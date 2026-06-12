import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import surpyval  # noqa: E402


def test_qf_uncensored():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    assert np.allclose(model.qf([0.1, 0.5, 0.9]), [1.0, 3.0, 5.0])
    assert model.median == 3.0


def test_qf_not_reached_is_nan():
    # With heavy right censoring the CDF never reaches 0.5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 1, 1, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    assert np.isnan(model.qf(0.5)).all()
    assert np.isnan(model.median)


def test_qf_invalid_p_raises():
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        model.qf(0.0)
    with pytest.raises(ValueError):
        model.qf(1.5)


def test_quantile_cb_brookmeyer_crowley_inversion():
    # The quantile interval limits must be the first observed times at
    # which the survival bounds cross 1 - p.
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    p = 0.5
    bounds = model.cb(model.x)
    level = 1 - p
    expected_lower = model.x[np.argmax(bounds[:, 0] <= level)]
    if (bounds[:, 1] < level).any():
        expected_upper = model.x[np.argmax(bounds[:, 1] < level)]
    else:
        expected_upper = np.nan
    cb = model.quantile_cb(p)
    assert cb.shape == (1, 2)
    assert cb[0, 0] == expected_lower
    assert np.isnan(cb[0, 1]) == np.isnan(expected_upper)
    # The interval contains the point estimate when the median is reached
    assert cb[0, 0] <= model.median


def test_quantile_cb_brackets_estimate_large_sample():
    rng = np.random.default_rng(5)
    x = rng.exponential(1.0, 200)
    model = surpyval.KaplanMeier.fit(x)
    cb = model.quantile_cb(0.5)
    assert cb[0, 0] <= model.median <= cb[0, 1]
    # With n=200 the median CI should be reasonably tight around ln(2)
    assert cb[0, 0] > 0.4
    assert cb[0, 1] < 1.1


def test_mean_uncensored_equals_sample_mean():
    # With no censoring the KM restricted mean to the largest
    # observation is the sample mean.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    assert np.isclose(model.mean(), 3.0, atol=1e-12)


def test_mean_restricted_to_tau():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    # integral of S over [0, 3): 1*1 + 1*0.8 + 1*0.6
    assert np.isclose(model.mean(tau=3.0), 2.4, atol=1e-12)


def test_mean_cb_matches_manual_variance():
    # Klein & Moeschberger RMST variance: sum(A_i^2 * v_i) with A_i the
    # area under S from x_i to tau and v_i the Greenwood increments.
    # For uncensored x = 1..5: A = [2.0, 1.2, 0.6, 0.2, 0] and
    # v = [1/20, 1/12, 1/6, 1/2, nan] giving var = 0.4 (the final term
    # is zero since A is zero there).
    from scipy.stats import norm

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    z = norm.ppf(0.975)
    se = np.sqrt(0.4)
    cb = model.mean_cb()
    assert np.allclose(cb, [3.0 - z * se, 3.0 + z * se], atol=1e-12)


def test_mean_with_censoring_below_uncensored_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 0, 0, 1])
    model = surpyval.KaplanMeier.fit(x, c=c)
    # S now plateaus at 0.2 after x=4; RMST to 5 is
    # 1 + .8 + .6 + .4 + .2 = 3.0
    assert np.isclose(model.mean(), 3.0, atol=1e-12)
    lower, upper = model.mean_cb()
    assert lower < model.mean() < upper


def test_mean_negative_support_raises():
    model = surpyval.KaplanMeier.fit(np.array([-1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        model.mean()


def test_bootstrap_cb_kaplan_meier():
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    x_test = [10.0, 30.0]
    cb = model.bootstrap_cb(x_test, B=100, random_state=42)
    assert cb.shape == (2, 2)
    assert (cb[:, 0] <= cb[:, 1]).all()
    # Bootstrap interval should contain the point estimate
    sf = model.sf(x_test)
    assert (cb[:, 0] <= sf).all()
    assert (sf <= cb[:, 1]).all()
    # Reproducible with the same seed
    cb2 = model.bootstrap_cb(x_test, B=100, random_state=42)
    assert np.allclose(cb, cb2)


def test_bootstrap_cb_turnbull():
    left = np.array([1, 8, 8, 7, 7, 17, 37, 46, 46, 45.0])
    right = np.array([7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf])
    model = surpyval.Turnbull.fit(xl=left, xr=right)
    cb = model.bootstrap_cb([10.0, 20.0], B=30, random_state=7)
    assert cb.shape == (2, 2)
    assert (cb[:, 0] <= cb[:, 1]).all()
    sf = model.sf([10.0, 20.0])
    assert (cb[:, 0] <= sf + 1e-12).all()
    assert (sf <= cb[:, 1] + 1e-12).all()


def test_bootstrap_cb_one_sided():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x)
    lower = model.bootstrap_cb([2.5], bound="lower", B=50, random_state=1)
    upper = model.bootstrap_cb([2.5], bound="upper", B=50, random_state=1)
    assert lower.shape == (1,)
    assert (lower <= upper).all()
    with pytest.raises(ValueError):
        model.bootstrap_cb([2.5], bound="middle")


def test_bootstrap_cb_requires_data():
    model = surpyval.KaplanMeier.from_xrd([1, 2, 3], [10, 8, 6], [2, 1, 1])
    with pytest.raises(ValueError, match="data"):
        model.bootstrap_cb([2.0])


def test_plot_with_bounds_and_censors():
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    model = surpyval.KaplanMeier.fit(x, c=c)
    fig, ax = plt.subplots()
    out = model.plot(ax=ax, color="C2", label="KM")
    assert out is ax
    # survival curve + censor markers, and the shaded bound band
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert ax.lines[0].get_color() == "C2"
    plt.close(fig)


def test_plot_without_bounds_or_censors():
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0]))
    fig, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False, show_censors=False)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    plt.close(fig)
