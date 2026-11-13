"""``Parametric.likelihood_contour`` and ``plot_likelihood_contour``:
the joint likelihood-ratio region of two parameters. The comparison with
R is in ``tests/reference/test_likelihood_contour.py``."""

import json
import warnings

import matplotlib
import numpy as np
import pytest

import surpyval as sp

matplotlib.use("Agg")

X = [61.9, 74.2, 82.2, 106.3, 76.2, 40.7, 83.2, 42.3, 50.4, 82.4]
C = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]


@pytest.fixture(scope="module")
def weibull():
    return sp.Weibull.fit(X, c=C)


def _deviance(model, theta):
    return 2.0 * (
        model._lr_neg_ll(np.asarray(theta, dtype=float))
        - model._lr_neg_ll(np.asarray(model.params, dtype=float))
    )


@pytest.mark.parametrize(
    "dist",
    [
        sp.Normal,
        sp.LogNormal,
        sp.Gamma,
        sp.Gumbel,
        sp.Logistic,
        sp.LogLogistic,
        sp.Beta,
        sp.Weibull,
    ],
)
def test_points_are_on_the_contour_for_each_family(dist):
    rng = np.random.default_rng(237)
    if dist is sp.Beta:
        x = rng.beta(2, 3, 30)
    else:
        x = 10 * rng.weibull(2, 30)
    model = dist.fit(x)
    contour = model.likelihood_contour(alpha_ci=0.05, n_points=40)
    assert contour.closed
    assert contour.points.shape == (44, 2)
    dev = [_deviance(model, p) for p in contour.points]
    np.testing.assert_allclose(dev, -2 * np.log(0.05), atol=1e-6)
    # The tangent points are the profile-likelihood bounds at the
    # chi-squared(2) critical value: the region's extremes.
    np.testing.assert_allclose(contour.points.min(0), contour.bounds[:, 0])
    np.testing.assert_allclose(contour.points.max(0), contour.bounds[:, 1])


def test_bounds_are_profile_bounds_at_the_two_df_level(weibull):
    # The extreme of one parameter over the joint region at alpha is its
    # profile-likelihood interval at the critical value -2 log(alpha):
    # that of the one-parameter interval at the level whose normal
    # quantile squared is that value.
    from scipy.stats import chi2

    contour = weibull.likelihood_contour(alpha_ci=0.05)
    level_1df = chi2.sf(-2 * np.log(0.05), 1)
    for k, name in enumerate(contour.params):
        np.testing.assert_allclose(
            contour.bounds[k],
            weibull.param_cb(name, alpha_ci=level_1df, method="lr"),
            rtol=1e-8,
        )


def test_regions_nest_and_contain_the_estimate(weibull):
    narrow = weibull.likelihood_contour(alpha_ci=0.2)
    wide = weibull.likelihood_contour(alpha_ci=0.01)
    assert np.all(wide.bounds[:, 0] < narrow.bounds[:, 0])
    assert np.all(wide.bounds[:, 1] > narrow.bounds[:, 1])
    for c in (narrow, wide):
        assert np.all(c.bounds[:, 0] < c.estimate)
        assert np.all(c.estimate < c.bounds[:, 1])
    np.testing.assert_array_equal(narrow.estimate, weibull.params)


def test_inside_the_polygon_is_inside_the_region(weibull):
    # The contour drawn as a polygon separates the pairs with deviance
    # below the critical value from those above it.
    from matplotlib.path import Path

    contour = weibull.likelihood_contour(alpha_ci=0.05, n_points=200)
    poly = Path(contour.points)
    rng = np.random.default_rng(1)
    lo, hi = contour.bounds[:, 0], contour.bounds[:, 1]
    span = hi - lo
    pts = lo - 0.2 * span + 1.4 * span * rng.random((300, 2))
    crit = contour.critical_value
    for p in pts:
        dev = _deviance(weibull, p)
        if abs(dev - crit) < 0.05:
            continue  # within the polygon's chords of the curve
        assert poly.contains_point(p) == (dev < crit)


def test_to_dict_round_trip_is_strict_json(weibull):
    contour = weibull.likelihood_contour(alpha_ci=0.1, n_points=12)
    text = json.dumps(contour.to_dict(), allow_nan=False)
    again = sp.LikelihoodContour.from_dict(json.loads(text))
    np.testing.assert_array_equal(again.points, contour.points)
    np.testing.assert_array_equal(again.bounds, contour.bounds)
    assert again.params == contour.params
    assert again.closed is True
    assert "90% joint region" in repr(again)
    with pytest.raises(ValueError, match="LikelihoodContour"):
        sp.LikelihoodContour.from_dict({"result": "Forecast"})


def test_non_finite_values_round_trip():
    contour = sp.LikelihoodContour(
        distribution="Weibull",
        params=["alpha", "beta"],
        estimate=np.array([1.0, 2.0]),
        alpha_ci=0.05,
        critical_value=5.99,
        log_likelihood=-3.0,
        log_likelihood_cut=-6.0,
        points=np.array([[1.0, np.nan], [2.0, 3.0]]),
        bounds=np.array([[0.5, np.inf], [1.0, 4.0]]),
        closed=False,
    )
    d = contour.to_dict()
    json.dumps(d, allow_nan=False)
    again = sp.LikelihoodContour.from_dict(d)
    assert np.isnan(again.points[0, 1])
    assert again.bounds[0, 1] == np.inf
    assert "open region" in repr(again)


def test_a_ray_that_misses_is_nan_with_a_warning(weibull, monkeypatch):
    model = sp.Weibull.fit(X, c=C)
    real = type(model)._lr_crossing
    calls = {"n": 0}

    def flaky(excess, u_hat, d, crit):
        calls["n"] += 1
        return None if calls["n"] == 3 else real(excess, u_hat, d, crit)

    monkeypatch.setattr(type(model), "_lr_crossing", staticmethod(flaky))
    with pytest.warns(RuntimeWarning, match="1 of the 20 rays"):
        contour = model.likelihood_contour(n_points=20)
    assert not contour.closed
    assert np.isnan(contour.points).any(axis=1).sum() == 1


def test_a_fixed_parameter_leaves_a_contour_of_the_other_two():
    rng = np.random.default_rng(5)
    x = 10 * rng.weibull(2, 40)
    model = sp.ExpoWeibull.fit(x, fixed={"mu": 1})
    contour = model.likelihood_contour(n_points=20)
    assert contour.params == ["alpha", "beta"]
    dev = [_deviance(model, np.r_[p, 1.0]) for p in contour.points]
    np.testing.assert_allclose(dev, contour.critical_value, atol=1e-6)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: sp.Exponential.fit(X, c=C), "exactly two fitted"),
        (
            lambda: sp.Weibull.fit(X, c=C, fixed={"beta": 2}),
            "beta fixed at fit time",
        ),
        (lambda: sp.ExpoWeibull.fit(X, c=C), "has 3"),
        (lambda: sp.Weibull.fit(X, c=C, how="MPP"), "Only MLE"),
        (
            lambda: sp.Weibull.fit(X, c=C, lfp=True),
            "Refit without it",
        ),
    ],
)
def test_refusals_say_what_to_do(make, match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make()
    with pytest.raises(ValueError, match=match):
        model.likelihood_contour()


@pytest.mark.filterwarnings("ignore:The weights leave")
def test_weighted_fit_is_refused():
    model = sp.Weibull.fit(X, c=C, w=np.linspace(0.5, 1.5, len(X)))
    with pytest.raises(ValueError, match="likelihood contour"):
        model.likelihood_contour()


@pytest.mark.parametrize("n_points", [3, 2.5, True, "100"])
def test_n_points_is_checked(weibull, n_points):
    with pytest.raises(ValueError, match="n_points"):
        weibull.likelihood_contour(n_points=n_points)


@pytest.mark.parametrize("alpha_ci", [0, 1, -0.1])
def test_alpha_ci_is_checked(weibull, alpha_ci):
    with pytest.raises(ValueError, match="alpha_ci"):
        weibull.likelihood_contour(alpha_ci=alpha_ci)


def test_plot_one_model_several_levels(weibull):
    import matplotlib.pyplot as plt

    out = sp.plot_likelihood_contour(weibull, alpha_ci=[0.1, 0.05, 0.01])
    assert out.params == ["alpha", "beta"]
    (label,) = out.contours
    assert [c.alpha_ci for c in out.contours[label]] == [0.1, 0.05, 0.01]
    assert len(out.lines[label]) == 3
    # A closed contour is drawn as a closed curve.
    xs = out.lines[label][0].get_xdata()
    assert xs[0] == xs[-1]
    np.testing.assert_array_equal(
        out.estimates[label].get_xydata()[0], weibull.params
    )
    assert out.ax.get_xlabel() == "alpha"
    assert out.ax.get_legend() is not None
    plt.close(out.fig)


def test_plot_compares_two_fits_on_given_axes(weibull):
    import matplotlib.pyplot as plt

    other = sp.Weibull.fit([88, 102, 115, 127, 139, 156, 170, 191])
    fig, ax = plt.subplots()
    out = sp.plot_likelihood_contour(
        [weibull, other],
        ax=ax,
        labels=["A", "B"],
        colors={"A": "black", "B": "red"},
        title="Two designs",
    )
    assert out.ax is ax
    assert list(out.contours) == ["A", "B"]
    assert out.lines["B"][0].get_color() == "red"
    assert ax.get_title() == "Two designs"
    plt.close(fig)


def test_plot_refusals(weibull):
    with pytest.raises(ValueError, match="same two"):
        sp.plot_likelihood_contour([weibull, sp.Normal.fit(X)])
    with pytest.raises(ValueError, match="labels"):
        sp.plot_likelihood_contour([weibull], labels=["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        sp.plot_likelihood_contour([])
    with pytest.raises(ValueError, match="empty"):
        sp.plot_likelihood_contour(weibull, alpha_ci=[])
    with pytest.raises(TypeError, match="no likelihood"):
        sp.plot_likelihood_contour({"km": sp.KaplanMeier.fit(X)})
