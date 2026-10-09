"""Parametric bootstrap bounds for univariate models, and the warning on
an offset model's Wald bounds (#645)."""

import pickle
import warnings

import numpy as np
import pytest

import surpyval as surv

W = surv.Weibull


@pytest.fixture(scope="module")
def offset_model():
    # A 3-parameter Weibull with a finite maximum (shape 3, 40 points)
    rng = np.random.default_rng(0)
    x = 50 + W.random(40, 100, 3, random_state=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = W.fit(x, offset=True)
    assert model.maximum == "verified"
    return model


def _boot(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, method="bootstrap", n_boot=40, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.cb([60.0, 120.0]),
        lambda m: m.quantile_cb([0.01, 0.1]),
        lambda m: m.param_cb("alpha"),
    ],
)
def test_645_an_offset_wald_bound_warns_that_gamma_is_held(offset_model, call):
    with pytest.warns(UserWarning, match="hold the offset gamma"):
        call(offset_model)


def test_645_a_plain_wald_bound_and_the_plot_do_not_warn(offset_model):
    plain = W.fit(W.random(30, 10, 3, random_state=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain.cb([5.0])
        plain.quantile_cb(0.1)
        offset_model.get_plot_data()


def test_645_bootstrap_bounds_contain_the_estimate(offset_model):
    m = offset_model
    q = _boot(m.quantile_cb, [0.01, 0.1, 0.5], random_state=1)
    est = m.qf([0.01, 0.1, 0.5])
    assert np.all(q[:, 0] <= est) and np.all(est <= q[:, 1])
    s = _boot(m.cb, [60.0, 120.0, 200.0], random_state=1)
    est = m.sf([60.0, 120.0, 200.0])
    assert np.all(s[:, 0] <= est) and np.all(est <= s[:, 1])
    g = _boot(m.param_cb, "gamma", random_state=1)
    assert g[0] <= m.gamma <= g[1]


def test_645_the_bootstrap_bound_on_b1_is_wider_than_wald(offset_model):
    m = offset_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wald = m.quantile_cb(0.01)
    boot = _boot(m.quantile_cb, 0.01, random_state=1)
    assert boot[0] < wald[0]


def test_645_sf_ff_and_hf_are_one_interval(offset_model):
    m = offset_model
    x = [70.0, 150.0]
    sf = _boot(m.cb, x, on="sf", random_state=2)
    ff = _boot(m.cb, x, on="ff", random_state=2)
    Hf = _boot(m.cb, x, on="Hf", random_state=2)
    np.testing.assert_allclose(ff, 1 - sf[:, ::-1])
    np.testing.assert_allclose(Hf, -np.log(sf[:, ::-1]))
    lower = _boot(m.cb, x, on="sf", bound="lower", random_state=2)
    np.testing.assert_allclose(lower, sf[:, 0], rtol=0.2)


def test_645_refits_are_kept_per_seed_and_not_pickled(offset_model):
    m = offset_model
    a = _boot(m.quantile_cb, 0.1, random_state=7)
    assert (40, 7) in m._bootstrap_refits
    b = _boot(m.quantile_cb, 0.1, random_state=7)
    np.testing.assert_array_equal(a, b)
    restored = pickle.loads(pickle.dumps(m))
    assert restored._bootstrap_refits is None


def test_645_a_fixed_parameter_is_its_value(offset_model):
    x = 50 + W.random(30, 100, 3, random_state=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = W.fit(x, offset=True, fixed={"beta": 3.0})
    np.testing.assert_array_equal(_boot(m.param_cb, "beta"), [3.0, 3.0])


@pytest.mark.parametrize(
    "build, match",
    [
        (
            lambda: surv.Parametric.from_dict(
                W.fit(W.random(30, 10, 2, random_state=6)).to_dict()
            ),
            "need the data",
        ),
        (
            lambda: W.fit(xl=[1, 2, 3, 4], xr=[2, 3, 4, 6], c=[2, 2, 2, 2]),
            "left- or interval-censored",
        ),
        (
            lambda: W.fit(W.random(30, 10, 2, random_state=4), lfp=True),
            "limited-failure or zero-inflated",
        ),
    ],
)
def test_645_what_cannot_be_resampled_is_refused(build, match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build()
    with pytest.raises(ValueError, match=match):
        m.cb([5.0], method="bootstrap", n_boot=5)


def test_645_bootstrap_is_refused_where_not_offered():
    m = W.fit(W.random(30, 10, 3, random_state=5))
    with pytest.raises(ValueError, match="method"):
        m.mean_cb(method="bootstrap")


def test_645_gamma_wald_refusal_points_at_the_bootstrap(offset_model):
    with pytest.raises(ValueError, match="method='bootstrap'"):
        offset_model.param_cb("gamma")
