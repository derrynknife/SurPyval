"""The hazard ratio ``hr`` against closed forms, far into both tails.

The identities every distribution keeps (``R ** (1 - b) * R ** b = R``,
``isf(sf(t)) == t``, ``hr = 1`` unchanged, saving) are swept over the
registry in ``tests/conformance/test_hazard_ratio.py``. Here the values:
the Weibull and exponential, whose hazard-scaled forms are their own
families, with exact references (``fractions``); the lognormal and gamma
against their survival functions raised to ``hr``; the moments worked out
numerically against closed forms (a gamma of shape 1 is an exponential,
and the minimum of two normals has a known mean and variance); and the
refusals.
"""

import math
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import gammaincc, log_ndtr

import surpyval as sp
from surpyval import serialisation

EPS = np.finfo(float).eps


def _close(got, want, rtol):
    """``got`` within ``rtol`` (relative tolerances, one per entry) of
    ``want``: ``assert_allclose`` takes one ``rtol``."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    zero = want == 0
    assert np.all(got[zero] == 0), (got, want)
    with np.errstate(invalid="ignore"):
        err = np.where(zero, 0.0, np.abs(got - want) / np.abs(want))
    assert np.all(err <= rtol), (got, want, err / rtol)


def _sensitive(hr_times_H):
    """The relative error ``exp(-hr H)`` may carry from rounding its
    argument: a few ulps of ``1 + hr H``."""
    return 8 * EPS * (1 + np.asarray(hr_times_H, dtype=float))


# x, and the Weibull(10, 2) cumulative hazard there, exactly
X = [
    Fraction(1, 10**7),
    Fraction(1, 1000),
    Fraction(1, 2),
    Fraction(5),
    Fraction(10),
    Fraction(40),
    Fraction(120),
    Fraction(260),
]
H_WEIBULL = [(x / 10) ** 2 for x in X]


@pytest.mark.parametrize("hr", [Fraction(1, 4), Fraction(3)])
def test_weibull_functions_are_exact(hr):
    model = sp.Weibull.from_params([10, 2], hr=float(hr))
    x = np.array([float(v) for v in X])
    hH = [hr * H for H in H_WEIBULL]
    want_H = np.array([float(v) for v in hH])
    np.testing.assert_allclose(model.Hf(x), want_H, rtol=2 * EPS, atol=0)
    want_sf = np.array([math.exp(-float(v)) for v in hH])
    _close(model.sf(x), want_sf, _sensitive(want_H))
    # ff in the lower tail: -expm1(-u) = u - u^2/2 + u^3/6 - ...
    small = [v for v in hH if v < Fraction(1, 100)]
    want_ff = [
        float(sum(-((-u) ** k) / math.factorial(k) for k in range(1, 12)))
        for u in small
    ]
    np.testing.assert_allclose(
        model.ff(x[: len(small)]), want_ff, rtol=4 * EPS, atol=0
    )
    # the hazard: hr * 2 x / 100, and the density hr h0 R
    want_h = np.array([float(hr * 2 * Fraction(v) / 100) for v in X])
    np.testing.assert_allclose(model.hf(x), want_h, rtol=2 * EPS, atol=0)
    _close(model.df(x), want_h * want_sf, _sensitive(want_H))
    # isf back in the upper tail (below 1/2: above, sf near 1 holds too
    # few digits of t to give it back, as for qf)
    up = (want_sf > 0) & (want_sf < 0.5)
    np.testing.assert_allclose(model.isf(want_sf[up]), x[up], rtol=1e-13)


def test_weibull_is_its_family_at_a_shorter_scale():
    hr = 3.0
    model = sp.Weibull.from_params([10, 2], hr=hr, gamma=2.0, f0=0.1)
    same = sp.Weibull.from_params(
        [10 * hr ** (-1 / 2), 2],
        gamma=2.0,
        f0=float(1 - Fraction(9, 10) ** 3),
    )
    x = np.array([0.0, 1.0, 2.5, 5.0, 12.0, 30.0])
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        np.testing.assert_allclose(
            getattr(model, fname)(x),
            getattr(same, fname)(x),
            rtol=1e-13,
            err_msg=fname,
        )
    p = np.array([0.0, 0.05, 0.2, 0.5, 0.9, 1 - 1e-12])
    # 1 - p = ((1 - f0) R0(t - gamma)) ** hr: t = gamma + alpha sqrt(-log
    # R0), log R0 = log(1 - p) / hr - log(1 - f0) (the family's own qf
    # loses digits near 1 in its (p - f0') / (1 - f0'))
    log_r0 = np.log1p(-p) / hr - np.log1p(-0.1)
    want = np.where(
        p <= 1 - 0.9**hr, 0.0, 2.0 + 10 * np.sqrt(np.maximum(-log_r0, 0))
    )
    np.testing.assert_allclose(model.qf(p), want, rtol=1e-13)
    np.testing.assert_allclose(model.qf(p[:-1]), same.qf(p[:-1]), rtol=1e-13)
    # the moments are the family's closed forms
    assert model.mean() == same.mean()
    assert model.var() == same.var()
    assert model.moment(3) == same.moment(3)
    np.testing.assert_array_equal(model.mrl(x), same.mrl(x))
    assert model.mean(tau=7.0) == same.mean(tau=7.0)
    plain = sp.Weibull.from_params([10, 2], hr=hr)
    assert (
        plain.entropy()
        == sp.Weibull.from_params([10 * hr ** (-1 / 2), 2]).entropy()
    )


def test_exponential_rate_is_scaled():
    model = sp.Exponential.from_params([0.5], hr=4.0)
    assert model.mean() == 0.5
    assert model.var() == 0.25
    # sf = exp(-2 t): isf(1e-300) = 150 log(10)
    np.testing.assert_allclose(
        model.isf(1e-300), 150 * math.log(10), rtol=2 * EPS
    )
    np.testing.assert_allclose(model.Hf(300.0), 600.0, rtol=EPS)
    np.testing.assert_allclose(model.ff(1e-20), 2e-20, rtol=EPS)
    np.testing.assert_allclose(model.qf(1e-20), 5e-21, rtol=2 * EPS)


def test_lognormal_is_its_survival_raised_to_hr():
    mu, sigma, hr = 2.0, 0.5, 0.3
    model = sp.LogNormal.from_params([mu, sigma], hr=hr)
    x = np.array([1e-3, 0.5, 7.4, 30.0, 300.0, 3000.0])
    z = (np.log(x) - mu) / sigma
    want_H = -hr * log_ndtr(-z)
    np.testing.assert_allclose(model.Hf(x), want_H, rtol=1e-13)
    _close(model.sf(x), np.exp(-want_H), 1e-13 * (1 + want_H))
    np.testing.assert_allclose(
        model.ff(x[:2]), -np.expm1(-want_H[:2]), rtol=1e-13
    )


def test_gamma_is_its_survival_raised_to_hr():
    shape, rate, hr = 2.5, 0.5, 0.5
    model = sp.Gamma.from_params([shape, rate], hr=hr)
    x = np.array([0.1, 3.0, 20.0, 200.0, 1000.0])
    sf0 = gammaincc(shape, rate * x)
    np.testing.assert_allclose(model.sf(x), sf0**hr, rtol=1e-12)
    s = np.asarray(model.sf(x))
    np.testing.assert_allclose(model.isf(s), x, rtol=1e-12)


def test_zero_inflation_mass_is_scaled():
    model = sp.Weibull.from_params([10, 2], f0=0.2, hr=2.0)
    mass = 1 - 0.8**2
    for value in (model.df(0.0), model.hf(0.0), model.ff(0.0)):
        np.testing.assert_allclose(value, mass, rtol=4 * EPS)
    assert model.df(0.0, continuous=True) == 0.0
    assert model.sf(-1.0) == 1.0
    assert model.qf(mass * 0.99) == 0.0


def test_numerical_moments_against_closed_forms():
    # A gamma of shape 1 is an exponential, so not its own family here:
    # rate 0.5 at hr 3 is an exponential of rate 1.5
    model = sp.Gamma.from_params([1.0, 0.5], hr=3.0)
    np.testing.assert_allclose(model.mean(), 1 / 1.5, rtol=1e-10)
    np.testing.assert_allclose(model.var(), 1 / 1.5**2, rtol=1e-9)
    np.testing.assert_allclose(model.moment(3), 6 / 1.5**3, rtol=1e-9)
    np.testing.assert_allclose(model.mrl([0.0, 4.0]), 1 / 1.5, rtol=1e-9)
    np.testing.assert_allclose(
        model.mean(tau=2.0), -math.expm1(-3.0) / 1.5, rtol=1e-10
    )
    # The minimum of two normals: mu - sigma / sqrt(pi), variance
    # sigma^2 (1 - 1 / pi)
    normal = sp.Normal.from_params([10.0, 2.0], hr=2.0)
    np.testing.assert_allclose(
        normal.mean(), 10 - 2 / math.sqrt(math.pi), rtol=1e-10
    )
    np.testing.assert_allclose(normal.var(), 4 * (1 - 1 / math.pi), rtol=1e-8)


def test_limited_failure_population_never_fails():
    model = sp.Gamma.from_params([2.0, 1.0], lfp_p=0.8, hr=2.0)
    assert model.mean() == np.inf and model.var() == np.inf
    np.testing.assert_allclose(model.sf(1e6), 0.2**2, rtol=4 * EPS)
    assert model.qf(1 - 0.2**2 + 1e-9) == np.inf
    with pytest.raises(ValueError, match="mean.*tau"):
        model.mean(defective=True)


def test_refusals_say_what_to_do():
    for bad in (0, -1.0, np.inf, np.nan, True, "2"):
        with pytest.raises(ValueError, match="hr, the hazard ratio"):
            sp.Weibull.from_params([10, 2], hr=bad)
    with pytest.raises(ValueError, match="1 - \\(1 - q\\) \\*\\* hr"):
        sp.Bernoulli.from_params([0.3], hr=2.0)
    with pytest.raises(ValueError, match="integrate"):
        sp.Gamma.from_params([2.0, 1.0], hr=2.0).entropy()
    with pytest.raises(ValueError, match="sum over"):
        sp.Poisson.from_params([3.0], hr=2.0).mean()


def test_with_params_extras_and_repr():
    model = sp.Weibull.from_params([10, 2], gamma=1.0)
    scaled = model.with_params([10, 2], hr=0.5)
    assert scaled.extras == {"gamma": 1.0, "hr": 0.5}
    assert scaled.with_params([12, 2]).hr == 0.5
    assert "Hazard ratio (hr)   : 0.5" in repr(scaled)
    assert "Hazard ratio" not in repr(model)
    with pytest.raises(ValueError, match="hr, the hazard ratio"):
        model.with_params([10, 2], hr=-2)


def test_saving_needs_schema_5(monkeypatch):
    model = sp.Weibull.from_params([10, 2], hr=0.5)
    d = model.to_dict()
    assert d["hr"] == 0.5 and d["schema"] == 5
    plain = sp.Weibull.from_params([10, 2], hr=1)
    assert "hr" not in plain.to_dict()
    assert plain.to_dict()["schema"] == 1
    assert pickle.loads(pickle.dumps(model)).hr == 0.5
    # A reader of schema 4 (SurPyval 0.28) refuses it
    monkeypatch.setattr(serialisation, "SCHEMA_VERSION", 4)
    with pytest.raises(ValueError, match="Upgrade surpyval"):
        sp.from_dict(d)


def test_draws_go_through_isf():
    model = sp.Weibull.from_params([10, 2], hr=4.0)
    same = sp.Weibull.from_params([5.0, 2])
    np.random.seed(7)
    drawn = model.random(200)
    np.random.seed(7)
    np.testing.assert_allclose(drawn, same.random(200), rtol=1e-13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(model.qf(1.5)) and np.isnan(model.isf(-0.1))


# -- fitting with a known hazard ratio, and its bounds --------------------

FIT_RTOL = 1e-6  # two searches of the same likelihood, each to its tolerance


def _censored_sample():
    x = sp.Weibull.random(60, 10, 2.5, random_state=1)
    c = (x > 14).astype(int)
    return np.minimum(x, 14), c


def _data_of_every_kind():
    return sp.SurpyvalData(
        x=[0.5, 2.0, 3.0, 4.0, [5.0, 6.0], [1.0, 7.0], 8.0, 9.0, 12.0],
        c=[0, 0, 1, -1, 2, 2, 0, 1, 0],
        n=[1, 2, 1, 1, 1, 1, 3, 2, 1],
        t=[[0.1, np.inf]] * 4 + [[-np.inf, 20.0]] * 5,
    )


@pytest.mark.parametrize("hr", [0.3, 2.0])
def test_likelihood_of_a_weibull_is_its_familys(hr):
    data = _data_of_every_kind()
    a, b = 10.0, 1.7
    got = sp.Weibull._neg_ll_func(data, a, b, 0.0, 0.0, 1.0, hr=hr)
    want = sp.Weibull._neg_ll_func(data, a * hr ** (-1 / b), b, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("hr", [0.3, 2.0])
def test_likelihood_is_the_models_own(hr):
    # every kind of observation, against the hazard-scaled model's own
    # functions: the density, R, F and the windows of R
    gamma, f0, p = 0.0, 0.1, 0.8
    model = sp.Gamma.from_params([2.0, 0.4], f0=f0, lfp_p=p, hr=hr)
    data = _data_of_every_kind()
    got = sp.Gamma._log_likelihood(data, 2.0, 0.4, gamma, f0, p, hr=hr)
    # the likelihood's terms as SurpyvalData sorts the rows (a censored
    # row with a finite truncation is the interval it implies)
    sf = model.sf
    want = np.sum(data.n_o * np.log(model.df(data.x_o)))
    want += np.sum(data.n_r * np.log(sf(data.x_r)))
    want += np.sum(data.n_l * np.log(model.ff(data.x_l)))
    want += np.sum(data.n_i * np.log(sf(data.x_il) - sf(data.x_ir)))
    # a window open above holds the units that never fail too
    tr = data.tr_unique
    beyond = np.where(np.isfinite(tr), sf(np.where(np.isfinite(tr), tr, 1)), 0)
    want -= np.sum(data.n_t_unique * np.log(sf(data.tl_unique) - beyond))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_a_fit_with_hr_is_the_familys_fit():
    x, c = _censored_sample()
    plain = sp.Weibull.fit(x, c)
    scaled = sp.Weibull.fit(x, c, hr=2.0)
    assert scaled.hr == 2.0 and scaled.maximum == "verified"
    alpha, beta = plain.params
    np.testing.assert_allclose(
        scaled.params, [alpha * 2 ** (1 / beta), beta], rtol=FIT_RTOL
    )
    np.testing.assert_allclose(scaled.neg_ll(), plain.neg_ll(), rtol=1e-10)
    t = np.array([3.0, 8.0, 13.0])
    for on in ("sf", "ff", "Hf", "hf", "df"):
        np.testing.assert_allclose(
            scaled.cb(t, on=on), plain.cb(t, on=on), rtol=FIT_RTOL
        )
    np.testing.assert_allclose(
        scaled.quantile_cb([0.1, 0.5]),
        plain.quantile_cb([0.1, 0.5]),
        rtol=FIT_RTOL,
    )
    # the shape's profile likelihood is the same
    np.testing.assert_allclose(
        scaled.param_cb("beta", method="lr"),
        plain.param_cb("beta", method="lr"),
        rtol=FIT_RTOL,
    )
    # the refits carry hr
    boot = scaled.bootstrap(20, random_state=1)
    assert all(m.hr == 2.0 for m in boot.models)
    # saved with it, and bounded the same once restored
    back = sp.from_dict(scaled.to_dict())
    np.testing.assert_array_equal(back.cb(t), scaled.cb(t))


def test_a_fit_with_hr_one_is_the_fit():
    x, c = _censored_sample()
    one = sp.Weibull.fit(x, c, hr=1)
    plain = sp.Weibull.fit(x, c)
    np.testing.assert_array_equal(one.params, plain.params)
    np.testing.assert_array_equal(one.covariance(), plain.covariance())
    assert one.to_dict() == plain.to_dict()


def test_fits_with_hr_refuse_what_they_cannot_carry():
    x, c = _censored_sample()
    with pytest.raises(ValueError, match="how='MLE'"):
        sp.Weibull.fit(x, c, how="MPP", hr=2.0)
    model = sp.Weibull.fit(x, c, hr=2.0)
    with pytest.raises(ValueError, match="method='wald'"):
        model.cb([5.0], method="lr")
    with pytest.raises(ValueError, match="method='wald'"):
        model.quantile_cb(0.5, method="lr")
    for call in (
        lambda: model.mean_cb(),
        lambda: model.mrl_cb([5.0]),
        lambda: model.rmst_cb(10.0),
        lambda: model.rmst(10.0),
    ):
        with pytest.raises(ValueError, match="bootstrap"):
            call()
