"""The pivotal Monte Carlo bounds of a probability-plotting (MPP) fit,
``method="pivotal"``, and the reduced bias adjustment ``rba()`` of a
maximum-likelihood fit. Their reference checks are in
``reference/test_weibullr.py`` and their coverage in
``calibration/test_coverage_pivotal.py``."""

import json
import pickle

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric import _pivotal

X = np.array([30.0, 49, 82, 90, 96, 125, 151, 172, 199, 236])
C_MIXED = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])

LOG_FAMILIES = [sp.Weibull, sp.LogNormal, sp.LogLogistic, sp.Exponential]
LINE_FAMILIES = [sp.Normal, sp.Logistic, sp.Gumbel, sp.GumbelLEV]
HEURISTICS = ["Nelson-Aalen", "Benard", "Blom", "Kaplan-Meier", "Filliben"]


def _data(dist):
    return X if dist in LOG_FAMILIES else np.log(X)


def _line(model):
    """The fit's line refitted by the simulation's regression to the
    data's own ordered values."""
    family = _pivotal._family(model)
    design = _pivotal._design(model, family)
    x = np.asarray(model.data["x"], dtype=float)
    c = np.asarray(model.data["c"])
    order = np.lexsort((c, x))
    t = x[order][design.ranks]
    y = design.y
    if family.origin:
        if design.rr == "y":
            rate = (t @ y) / (t @ t)
        else:
            rate = (y @ y) / (t @ y)
        return -np.log(rate), 1.0
    z = np.log(t) if family.log_time else t
    zc, yc = z - z.mean(), y - y.mean()
    if design.rr == "y":
        b = (zc @ zc) / (zc @ yc)
        return z.mean() - y.mean() * b, b
    b = (zc @ yc) / (yc @ yc)
    return z.mean() - b * y.mean(), b


@pytest.mark.parametrize("heuristic", HEURISTICS)
@pytest.mark.parametrize("rr", ["x", "y"])
@pytest.mark.parametrize("dist", LOG_FAMILIES + LINE_FAMILIES)
def test_the_simulation_fits_by_the_same_regression(dist, rr, heuristic):
    # The regression the simulated samples are fitted by, applied to the
    # data, gives the model's own fit: same positions, same direction.
    for c in (np.zeros(10, int), C_MIXED):
        model = dist.fit(
            _data(dist), c=c, how="MPP", rr=rr, heuristic=heuristic
        )
        family = _pivotal._family(model)
        np.testing.assert_allclose(
            _line(model), _pivotal._ub_hat(model, family), rtol=1e-10
        )


def test_on_d_is_0_is_followed():
    model = sp.Weibull.fit(X, c=C_MIXED, how="MPP", on_d_is_0=True)
    family = _pivotal._family(model)
    np.testing.assert_allclose(
        _line(model), _pivotal._ub_hat(model, family), rtol=1e-10
    )


@pytest.fixture(scope="module")
def weibull():
    return sp.Weibull.fit(X, c=C_MIXED, how="MPP")


KW = {"method": "pivotal", "random_state": 7, "n_sim": 1999}


def test_bounds_contain_the_estimate(weibull):
    for name, value in zip(weibull.parameter_names, weibull.params):
        lo, hi = weibull.param_cb(name, **KW)
        assert lo < value < hi
    p = np.array([0.01, 0.1, 0.5, 0.9])
    q = weibull.quantile_cb(p, **KW)
    assert np.all(q[:, 0] < weibull.qf(p)) and np.all(weibull.qf(p) < q[:, 1])
    t = np.array([20.0, 100.0, 300.0])
    for on in ("sf", "ff", "Hf"):
        b = weibull.cb(t, on=on, **KW)
        f = getattr(weibull, on)(t)
        assert np.all(b[:, 0] < f) and np.all(f < b[:, 1])


def test_one_sided_bounds_are_ends_of_the_two_sided(weibull):
    two = weibull.cb([50.0, 150.0], alpha_ci=0.2, **KW)
    np.testing.assert_allclose(
        weibull.cb([50.0, 150.0], alpha_ci=0.1, bound="lower", **KW),
        two[:, 0],
    )
    np.testing.assert_allclose(
        weibull.cb([50.0, 150.0], alpha_ci=0.1, bound="upper", **KW),
        two[:, 1],
    )
    two = weibull.param_cb("beta", alpha_ci=0.2, **KW)
    assert weibull.param_cb("beta", alpha_ci=0.1, bound="lower", **KW)[
        0
    ] == pytest.approx(two[0])
    two = weibull.quantile_cb(0.1, alpha_ci=0.2, **KW)
    assert weibull.quantile_cb(
        0.1, alpha_ci=0.1, bound="upper", **KW
    ) == pytest.approx(two[1])


def test_the_band_and_the_quantile_bounds_are_one(weibull):
    # The band on ff at a bound on the p quantile is p.
    p = np.array([0.05, 0.1, 0.5])
    q = weibull.quantile_cb(p, **KW)
    ff_at_upper = weibull.cb(q[:, 1], on="ff", **KW)[:, 0]
    ff_at_lower = weibull.cb(q[:, 0], on="ff", **KW)[:, 1]
    np.testing.assert_allclose(ff_at_upper, p, rtol=1e-6)
    np.testing.assert_allclose(ff_at_lower, p, rtol=1e-6)


def test_sf_ff_and_Hf_are_one_band(weibull):
    t = np.array([20.0, 100.0, 300.0])
    sf = weibull.cb(t, on="sf", **KW)
    ff = weibull.cb(t, on="ff", **KW)
    Hf = weibull.cb(t, on="Hf", **KW)
    np.testing.assert_allclose(sf, 1 - ff[:, ::-1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Hf, -np.log(sf[:, ::-1]), rtol=1e-10)


def test_a_higher_level_is_wider(weibull):
    narrow = weibull.param_cb("alpha", alpha_ci=0.2, **KW)
    wide = weibull.param_cb("alpha", alpha_ci=0.05, **KW)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_outside_the_support_and_missing():
    model = sp.Weibull.fit(X, how="MPP")
    b = model.cb([-1.0, 0.0, np.nan], **KW)
    np.testing.assert_array_equal(b[:2], [[1.0, 1.0], [1.0, 1.0]])
    assert np.isnan(b[2]).all()
    assert model.cb(np.inf, on="ff", **KW).tolist() == [1.0, 1.0]


def test_shapes(weibull):
    assert weibull.cb(50.0, **KW).shape == (2,)
    assert weibull.cb([[50.0, 60.0]], **KW).shape == (1, 2, 2)
    assert weibull.cb([50.0, 60.0], bound="lower", **KW).shape == (2,)
    assert weibull.quantile_cb(0.1, **KW).shape == (2,)
    assert weibull.quantile_cb([0.1, 2.0], **KW).shape == (2, 2)
    assert weibull.param_cb("beta", bound="upper", **KW).shape == (1,)


def test_seeds():
    model = sp.Weibull.fit(X, how="MPP")
    a = model.param_cb("beta", method="pivotal", random_state=3)
    assert model._pivotal_draws  # kept for the integer seed
    b = model.param_cb("beta", method="pivotal", random_state=3)
    np.testing.assert_array_equal(a, b)
    np.random.seed(11)
    c = model.param_cb("beta", method="pivotal")
    np.random.seed(11)
    d = model.param_cb("beta", method="pivotal")
    np.testing.assert_array_equal(c, d)
    gen = np.random.default_rng(3)
    e = model.param_cb("beta", method="pivotal", random_state=gen)
    f = model.param_cb("beta", method="pivotal", random_state=gen)
    assert not np.array_equal(e, f)


def test_the_monte_carlo_error_is_small_at_the_default():
    # Two seeds at the default 9999 simulations: the 95% bounds on the
    # shape differ by well under 2%.
    model = sp.Weibull.fit(X, how="MPP")
    a = model.param_cb("beta", method="pivotal", random_state=1)
    b = model.param_cb("beta", method="pivotal", random_state=2)
    np.testing.assert_allclose(a, b, rtol=0.02)


def test_case_and_exponential():
    model = sp.Exponential.fit(X, how="MPP")
    lo, hi = model.param_cb("failure_rate", method="Pivotal", random_state=1)
    assert lo < model.params[0] < hi
    q = model.quantile_cb(0.1, method="pivotal", random_state=1)
    # One parameter: the quantile's bounds are the rate's, mapped.
    np.testing.assert_allclose(q, -np.log(0.9) / np.array([hi, lo]))


def test_saved_and_loaded_with_its_data(weibull):
    rr_x = sp.LogNormal.fit(X, c=C_MIXED, how="MPP", rr="x", heuristic="Blom")
    for model in (weibull, rr_x):
        d = json.loads(json.dumps(model.to_dict(with_data=True)))
        back = sp.from_dict(d)
        np.testing.assert_array_equal(
            back.cb([50.0, 150.0], **KW), model.cb([50.0, 150.0], **KW)
        )
    with pytest.raises(ValueError, match="with_data=True"):
        sp.from_dict(weibull.to_dict()).cb(50.0, method="pivotal")


def test_the_draws_are_not_pickled(weibull):
    weibull.param_cb("beta", **KW)
    assert weibull._pivotal_draws
    assert pickle.loads(pickle.dumps(weibull))._pivotal_draws is None


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: sp.Weibull.fit(X), "method='wald', 'lr' or 'bootstrap'"),
        (lambda: sp.Weibull.fit(X, how="MPS"), "Refit with how='MPP'"),
        (lambda: sp.Weibull.from_params([100, 2]), "built by given"),
        (
            lambda: sp.Rayleigh.fit(X, how="MPP"),
            "the Rayleigh is not among them",
        ),
        (lambda: sp.Weibull.fit(X, how="MPP", offset=True), "offset"),
        (
            lambda: sp.Weibull.fit(X, how="MPP", tl=10.0),
            "without truncation",
        ),
        (
            lambda: sp.Weibull.fit(
                xl=X, xr=X + 5, how="MPP", heuristic="Turnbull"
            ),
            "interval-censored",
        ),
    ],
)
def test_refusals_say_what_to_use(make, match):
    model = make()
    for call in (
        lambda: model.cb(50.0, method="pivotal"),
        lambda: model.quantile_cb(0.1, method="pivotal"),
        lambda: model.param_cb(
            model.dist.parameter_names[0], method="pivotal"
        ),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_other_refusals(weibull):
    with pytest.raises(ValueError, match="not a monotone function"):
        weibull.cb(50.0, on="hf", method="pivotal")
    with pytest.raises(ValueError, match="'on' must be one of"):
        weibull.cb(50.0, on="xx", method="pivotal")
    with pytest.raises(ValueError, match="'n_sim' must be"):
        weibull.cb(50.0, method="pivotal", n_sim=0)
    with pytest.raises(ValueError, match="Use n_sim of at least 199"):
        weibull.cb(50.0, method="pivotal", n_sim=50, alpha_ci=0.01)
    with pytest.raises(ValueError, match="Unknown parameter"):
        weibull.param_cb("gamma", method="pivotal")
    with pytest.raises(ValueError, match="'bound' must be one of"):
        weibull.cb(50.0, method="pivotal", bound="both")


def test_other_bounds_of_an_mpp_fit_point_to_pivotal(weibull):
    with pytest.raises(ValueError, match="method='pivotal'"):
        weibull.cb(50.0)


def test_bootstrap_refits_by_the_fits_own_method():
    # Before, every refit was by maximum likelihood, so an MPP fit's
    # bootstrap bound was the spread of the MLE about the MPP estimate.
    model = sp.Weibull.fit(X, c=C_MIXED, how="MPP", rr="x", heuristic="Blom")
    boot = model.bootstrap(20, random_state=1)
    assert {m.method for m in boot.models} == {"MPP"}
    info = boot.models[0].fitting_info
    assert (info["rr"], info["heuristic"]) == ("x", "Blom")


# -- the reduced bias adjustment ----------------------------------------------
def test_rba_shrinks_the_weibull_shape_and_keeps_the_scale():
    model = sp.Weibull.fit(X, c=C_MIXED)
    median = model.rba()
    mean = model.rba("mean")
    assert median.params[0] == model.params[0]
    assert mean.params[1] < median.params[1] < model.params[1]
    assert median.bias_adjustment == "rba (median, 8 failures)"
    back = sp.from_dict(json.loads(json.dumps(median.to_dict())))
    assert back.bias_adjustment == median.bias_adjustment


def test_rba_stretches_the_normal_sigma():
    for dist, data in ((sp.LogNormal, X), (sp.Normal, np.log(X))):
        model = dist.fit(data)
        adjusted = model.rba()
        assert adjusted.params[0] == model.params[0]
        assert adjusted.params[1] > model.params[1]
    # A complete normal sample: sigma becomes s / C4
    from surpyval.univariate.parametric._rba import c4

    x = np.log(X)
    sigma = sp.Normal.fit(x).rba().params[1]
    assert sigma == pytest.approx(np.std(x, ddof=1) / c4(x.size), rel=1e-6)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: sp.Weibull.fit(X, how="MPP"), "Refit with how='MLE'"),
        (lambda: sp.Gamma.fit(X), "the Gamma has none"),
        (lambda: sp.Weibull.fit(X, offset=True), "plain two-parameter"),
        (lambda: sp.Weibull.fit(X, fixed={"beta": 2.0}), "'beta' was fixed"),
        (lambda: sp.Weibull.fit(X, tl=10.0), "without truncation"),
        (lambda: sp.Weibull.fit(X[:2]), "at least 3 failures"),
        (lambda: sp.Weibull.from_params([100, 2]), "Refit with how='MLE'"),
    ],
)
def test_rba_refusals(make, match):
    with pytest.raises(ValueError, match=match):
        make().rba()


def test_rba_basis_refusals():
    with pytest.raises(ValueError, match="'basis' must be one of"):
        sp.Weibull.fit(X).rba("mode")
    with pytest.raises(ValueError, match="'basis' is for the Weibull"):
        sp.LogNormal.fit(X).rba("mean")
