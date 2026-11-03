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


# ---------------------------------------------------------------------------
# Limited-failure and zero-inflated models (#286)
# ---------------------------------------------------------------------------


def _lfp_data(seed, lfp_p=0.6, n=100):
    rng = np.random.default_rng(seed)
    t = 10 * rng.weibull(2.0, n)
    t = np.where(rng.uniform(size=n) > lfp_p, np.inf, t)
    cens = rng.uniform(0, 30, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _zi_data(seed, f0=0.1, n=100):
    rng = np.random.default_rng(seed)
    t = 10 * rng.weibull(2.0, n)
    t = np.where(rng.uniform(size=n) < f0, 0.0, t)
    cens = rng.uniform(0, 30, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _fit(dist, x, c, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dist.fit(x, c=c, **kw)


@pytest.mark.parametrize(
    "dist, data, kw, extra",
    [
        (W, _lfp_data, {"lfp": True}, "lfp_p"),
        (W, _zi_data, {"zi": True}, "f0"),
        (surv.LogNormal, _lfp_data, {"lfp": True}, "lfp_p"),
        (W, lambda s: _lfp_data(s), {"lfp": True, "zi": True}, "lfp_p"),
    ],
)
def test_286_bootstrap_bounds_of_lfp_and_zi_models(dist, data, kw, extra):
    x, c = data(1)
    m = _fit(dist, x, c, **kw)
    names = list(m.dist.parameter_names) + [extra]
    for name in names:
        lo, hi = _boot(m.param_cb, name, random_state=1)
        value = (
            getattr(m, extra)
            if name == extra
            else m.params[m.dist.param_map[name]]
        )
        assert lo <= value <= hi, (name, lo, value, hi)
    t = np.array([3.0, 8.0, 15.0])
    s = _boot(m.cb, t, random_state=1)
    est = m.sf(t)
    assert np.all(s[:, 0] <= est) and np.all(est <= s[:, 1])
    q = _boot(m.quantile_cb, [0.05, 0.2], random_state=1)
    est = m.qf([0.05, 0.2])
    assert np.all(q[:, 0] <= est) and np.all(est <= q[:, 1])
    # One set of refits, so on sf and ff they are one interval.
    ff = _boot(m.cb, t, on="ff", random_state=1)
    np.testing.assert_allclose(ff, 1 - s[:, ::-1])


def test_286_an_offset_lfp_model_bounds_gamma_and_lfp_p():
    x, c = _lfp_data(2)
    m = _fit(W, x + 5.0, c, lfp=True, offset=True)
    g = _boot(m.param_cb, "gamma", random_state=3)
    p = _boot(m.param_cb, "lfp_p", random_state=3)
    assert g[0] <= m.gamma <= g[1]
    assert p[0] <= m.lfp_p <= p[1]
    # The refits are of the same model: offset, lfp_p and all.
    refit = m._bootstrap_refits[(40, 3)].models[0]
    assert refit.offset and refit.lfp and not refit.zi


def test_286_lfp_p_bootstrap_bounds_track_the_likelihood_ratio_bounds():
    m = _fit(W, *_lfp_data(1), lfp=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = m.param_cb(
            "lfp_p", method="bootstrap", n_boot=200, random_state=286
        )
        lr = m.param_cb("lfp_p", method="lr")
    np.testing.assert_allclose(boot, lr, atol=0.05)


def test_286_a_quantile_beyond_lfp_p_is_infinite():
    m = _fit(W, *_lfp_data(1), lfp=True)
    q = _boot(m.quantile_cb, [m.lfp_p + 0.1], random_state=1)
    assert np.isinf(q[0, 1])


def test_286_refits_on_the_edge_are_kept_there():
    # Every unit can fail: the estimate of lfp_p is on its edge, 1, and
    # so are many refits; they count at 1, so the upper bound is 1.
    rng = np.random.default_rng(5)
    t, cens = 10 * rng.weibull(2.0, 100), rng.uniform(0, 30, 100)
    m = _fit(W, np.minimum(t, cens), (cens < t).astype(int), lfp=True)
    assert m.lfp_p > 1 - 1e-6
    upper = _boot(m.param_cb, "lfp_p", bound="upper", random_state=1)
    assert upper[0] == 1.0
    fits = m._bootstrap_refits[(40, 1)]
    assert len(fits.models) == 40
    assert fits.estimate[-1] == 1.0
    # A zero-inflated fit with no zeros: f0 on its edge, 0.
    mz = _fit(W, np.minimum(t, cens), (cens < t).astype(int), zi=True)
    lower = _boot(mz.param_cb, "f0", bound="lower", random_state=1)
    assert lower[0] == 0.0


def test_286_a_fixed_lfp_p_is_its_value_and_is_held_in_the_refits():
    x, c = _lfp_data(4)
    m = _fit(W, x, c, lfp=True, fixed={"lfp_p": 0.6})
    np.testing.assert_array_equal(_boot(m.param_cb, "lfp_p"), [0.6, 0.6])
    _boot(m.param_cb, "beta", random_state=2)
    refits = m._bootstrap_refits[(40, 2)]
    assert np.all(refits.values[:, -1] == 0.6)


def test_286_unknown_parameters_name_the_extra_ones():
    m = _fit(W, *_lfp_data(1), lfp=True)
    with pytest.raises(ValueError, match="'lfp_p'"):
        m.param_cb("f0", method="bootstrap", n_boot=5)
