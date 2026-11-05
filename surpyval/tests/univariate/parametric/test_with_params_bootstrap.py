"""``Parametric.with_params`` with the full covariance vector, and the
public ``Parametric.bootstrap``.

``with_params`` takes the distribution's parameters (as before) or every
row of ``covariance()`` in the order of ``covariance_names`` (the
distribution's parameters, then ``lfp_p`` and ``f0``), with the offset
``gamma`` as a keyword; ``bootstrap`` returns the refits behind the
``method="bootstrap"`` bounds. The seeds and tolerances below were fixed
before the results were seen.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.univariate.parametric import ParametricBootstrap

T = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0])
FUNCTIONS = ("sf", "ff", "df", "hf", "Hf")


def _data(seed=20261105):
    rng = np.random.default_rng(seed)
    x = 10 * rng.weibull(2.0, 60)
    c = (x > 15).astype(int)
    return np.minimum(x, 15), c, rng


def _lfp_data(seed=20261105):
    rng = np.random.default_rng(seed)
    x = np.where(rng.uniform(size=200) < 0.6, 10 * rng.weibull(2.0, 200), 99)
    c = (x > 25).astype(int)
    return np.minimum(x, 25), c


def _fits():
    x, c, rng = _data()
    xl, cl = _lfp_data()
    zeros = np.zeros(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "plain": surv.Weibull.fit(x, c=c),
            "lfp": surv.Weibull.fit(xl, c=cl, lfp=True),
            "zi": surv.Weibull.fit(
                np.r_[x, zeros], c=np.r_[c, zeros], zi=True
            ),
            "lfp+zi": surv.Weibull.fit(
                np.r_[xl, zeros], c=np.r_[cl, zeros], lfp=True, zi=True
            ),
            "offset": surv.Weibull.fit(x + 5, c=c, offset=True),
            "fixed": surv.Weibull.fit(x, c=c, fixed={"beta": 2.0}),
            "weighted": surv.Weibull.fit(
                x, c=c, w=rng.uniform(0.5, 2.0, x.size)
            ),
        }


FITS = _fits()


def _same(a, b, t=T):
    for f in FUNCTIONS:
        np.testing.assert_allclose(
            getattr(a, f)(t), getattr(b, f)(t), rtol=1e-12, err_msg=f
        )


@pytest.mark.parametrize("kind", list(FITS))
def test_round_trip_of_the_estimates(kind):
    model = FITS[kind]
    rebuilt = model.with_params(model._estimates())
    assert rebuilt.extras == model.extras
    assert rebuilt.unit == model.unit
    _same(model, rebuilt, T + (5 if kind == "offset" else 0))


def test_estimates_are_in_covariance_order():
    model = FITS["lfp+zi"]
    assert model.covariance_names == ["alpha", "beta", "lfp_p", "f0"]
    np.testing.assert_array_equal(
        model._estimates(), [*model.params, model.lfp_p, model.f0]
    )
    assert len(model._estimates()) == model.covariance().shape[0]


def test_the_short_vector_keeps_the_fitted_extras():
    model = FITS["lfp+zi"]
    other = model.with_params([12.0, 1.5])
    assert (other.lfp_p, other.f0) == (model.lfp_p, model.f0)


def test_the_full_vector_sets_lfp_p_and_f0():
    model = FITS["lfp+zi"]
    other = model.with_params([12.0, 1.5, 0.7, 0.02])
    assert (other.lfp_p, other.f0) == (0.7, 0.02)
    np.testing.assert_array_equal(other.params, [12.0, 1.5])


def test_gamma_keyword_and_kept_gamma():
    model = FITS["offset"]
    assert model.with_params(model.params).gamma == model.gamma
    assert model.with_params(model.params, gamma=1.5).gamma == 1.5


def test_lfp_p_and_f0_keywords_with_the_short_vector():
    model = FITS["lfp+zi"]
    other = model.with_params([12.0, 1.5], lfp_p=0.8)
    assert (other.lfp_p, other.f0) == (0.8, model.f0)
    assert model.with_params([12.0, 1.5], f0=0.01).f0 == 0.01


@pytest.mark.parametrize(
    "kind, args, kwargs, match",
    [
        ("plain", ([10.0, 2.0],), {"gamma": 1.0}, "has no gamma"),
        ("plain", ([10.0, 2.0],), {"lfp_p": 0.9}, "has no lfp_p"),
        ("lfp", ([10.0, 2.0],), {"f0": 0.1}, "has no f0"),
        ("lfp", ([10.0, 2.0, 0.5],), {"lfp_p": 0.9}, "already holds"),
        ("lfp", ([10.0, 2.0, 0.5, 0.1],), {}, "covariance_names"),
        ("plain", ([10.0],), {}, "2 entries"),
        ("plain", ([[10.0, 2.0]],), {}, "flat vector"),
        ("lfp", ([10.0, 2.0, 1.02],), {}, "bootstrap()"),
        ("zi", ([10.0, 2.0, -0.01],), {}, "logit scale"),
        ("lfp+zi", ([10.0, 2.0, 0.3, 0.4],), {}, "less than lfp_p"),
    ],
)
def test_with_params_refusals(kind, args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        FITS[kind].with_params(*args, **kwargs)


def test_weibull_and_lognormal_against_flexsurv():
    # R 4.6.2, flexsurv: flexsurvreg(..., inits=, fixedpars=TRUE) and
    # summary(type="survival") at t = 5, 15
    wb = FITS["plain"].with_params([10.0, 2.0]).sf([5.0, 15.0])
    np.testing.assert_allclose(
        wb, [0.7788007830714050, 0.1053992245618645], rtol=1e-13
    )
    ln = surv.LogNormal.from_params([1.0, 1.0]).with_params([2.0, 0.5])
    np.testing.assert_allclose(
        ln.sf([5.0, 15.0]),
        [0.78263526772848868, 0.07837305518484261],
        rtol=1e-13,
    )


def _delta_se(model, t):
    e = model._estimates()
    h = 1e-6
    g = np.array(
        [
            (
                model.with_params(e + h * np.eye(e.size)[i]).sf(t)
                - model.with_params(e - h * np.eye(e.size)[i]).sf(t)
            )
            / (2 * h)
            for i in range(e.size)
        ]
    )
    return np.sqrt(np.einsum("it,ij,jt->t", g, model.covariance(), g))


@pytest.mark.parametrize("kind", ["plain", "lfp"])
def test_draws_through_with_params_agree_with_the_delta_method(kind):
    model = FITS[kind]
    t = np.array([5.0, 10.0])
    rng = np.random.default_rng(831)
    draws = rng.multivariate_normal(
        model._estimates(), model.covariance(), 4000
    )
    sf = np.array([model.with_params(d).sf(t) for d in draws])
    ratio = sf.std(axis=0) / _delta_se(model, t)
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)


def test_draws_spread_like_the_wald_band():
    model = FITS["plain"]
    rng = np.random.default_rng(832)
    draws = rng.multivariate_normal(model.params, model.covariance(), 4000)
    sf = np.array([model.with_params(d).sf(5.0) for d in draws])
    band = model.cb(5.0, on="sf", method="wald")
    np.testing.assert_allclose(
        np.quantile(sf, [0.025, 0.975]), band, atol=0.01
    )


@pytest.mark.parametrize("kind", ["plain", "lfp", "offset", "weighted"])
def test_with_params_of_each_refit_is_the_refit(kind):
    model = FITS[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = model.bootstrap(20, random_state=5)
    assert isinstance(boot, ParametricBootstrap)
    assert boot.names == model.covariance_names
    assert boot.params.shape == (len(boot), len(model.covariance_names))
    assert len(boot) == boot.n_boot - boot.failed
    if kind == "offset":
        assert boot.gamma is not None and boot.gamma.shape == (len(boot),)
    else:
        assert boot.gamma is None
    t = T + (5 if kind == "offset" else 0)
    for i, refit in enumerate(boot.models):
        gamma = None if boot.gamma is None else boot.gamma[i]
        _same(refit, model.with_params(boot.params[i], gamma=gamma), t)


def test_bootstrap_gives_the_refits_of_the_bounds():
    x, c, _ = _data()
    model = surv.Weibull.fit(x, c=c)
    bounds = model.cb(5.0, method="bootstrap", n_boot=30, random_state=4)
    boot = model.bootstrap(30, random_state=4)
    kept = model._bootstrap_refits[(30, 4)].models
    assert all(a is b for a, b in zip(boot.models, kept))
    again = model.cb(5.0, method="bootstrap", n_boot=30, random_state=4)
    np.testing.assert_array_equal(bounds, again)
    fresh = surv.Weibull.fit(x, c=c).bootstrap(30, random_state=4)
    np.testing.assert_array_equal(fresh.params, boot.params)


def test_bootstrap_spread_matches_the_standard_errors():
    rng = np.random.default_rng(20261106)
    model = surv.Weibull.fit(10 * rng.weibull(2.0, 100))
    boot = model.bootstrap(300, random_state=7)
    ratio = boot.params.std(axis=0, ddof=1) / model.standard_errors()
    np.testing.assert_allclose(ratio, 1.0, atol=0.15)


def test_bootstrap_to_dict_is_strict_json():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = FITS["offset"].bootstrap(10, random_state=2)
    d = boot.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["result"] == "ParametricBootstrap"
    assert d["names"] == ["alpha", "beta"]
    assert len(d["gamma"]) == len(d["params"]) == len(boot)
    assert "ParametricBootstrap(" in repr(boot)


@pytest.mark.parametrize(
    "model, match",
    [
        (surv.Weibull.from_params([10.0, 2.0]), "with_params"),
        (
            surv.Weibull.fit(
                xl=[1.0, 2.0, 3.0, 4.0], xr=[2.0, 3.0, 4.0, 6.0], c=[2] * 4
            ),
            "interval-censored",
        ),
        (surv.Poisson.fit([1, 2, 2, 3, 4, 5]), "continuous"),
    ],
)
def test_bootstrap_refusals(model, match):
    with pytest.raises(ValueError, match=match):
        model.bootstrap(10, random_state=1)


def test_bootstrap_refuses_a_weighted_lfp_fit_and_says_what_to_do():
    from surpyval.utils.weights import WeightedRefusal

    xl, cl = _lfp_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.Weibull.fit(xl, c=cl, lfp=True, w=np.ones(xl.size))
    with pytest.raises(WeightedRefusal, match="with_params"):
        model.bootstrap(10, random_state=1)
