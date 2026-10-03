"""The regression covariance comes from the fit's exact Hessian (#392).

The no-maximum check computes the Hessian of the negative log-likelihood at
the fit with autograd, in the optimiser's search space. The fit converts it
exactly to the natural parameters (``_fit_skeleton.natural_information``)
and keeps it, and ``covariance()``, ``standard_errors()``, ``cb()`` and
``param_cb()`` invert it instead of differencing the likelihood numerically
on every call; the covariance is then kept too. A model without it -- an
accelerated-life or AFT time-varying fit, a runaway, a model whose
parameters have moved -- falls back to the numerical Hessian as before.
"""

import copy
import warnings

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, hessian

import surpyval as sp
import surpyval.univariate.competing_risks.regression.fine_gray as fine_gray
import surpyval.univariate.regression._fit_skeleton as skeleton
import surpyval.univariate.regression._inference as inference
import surpyval.univariate.regression.frailty.frailty_fitter as frailty
from surpyval.tests.conformance.registry import CASE_BY_NAME, reg_data
from surpyval.univariate.regression._fit_skeleton import (
    centred_copy,
    natural_information,
)

FAMILIES = [
    name
    for name in CASE_BY_NAME
    if name.endswith(("PH", "AFT", "PO", "AH")) and name != "CoxPH"
]

#: Where the old numerical standard errors are further than 1e-4 from the
#: exact ones, by their own rounding error: coefficients near 0 (-0.13 and
#: 0.07) get the step's floor, eps^(1/3) * 1e-2, at which rounding in the
#: likelihood (about 84) reaches the fourth digit. Ten times the step
#: agrees with the exact values to 1.5e-5 and 3.3e-7. How far the default
#: step lands depends on the point: at the ExponentialPO fit the gradient
#: ladder reaches (#499; neg_ll 4e-9 lower) it is 1.8e-3, while ten times
#: the step still agrees to 1.3e-5, and the exact values move by 1e-6.
OLD_ROUNDING = {"ExponentialPO": 2e-3, "ExponentialAH": 2e-4}


def _fit(fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit()


def _registry(name):
    case = CASE_BY_NAME[name]
    return _fit(lambda: case.fit(case.data()))


def _numerical_se(model, scale=1.0):
    """The standard errors from the numerical Hessian, as before (its step
    times ``scale``)."""
    model._information = None
    model._covariance_cache = None
    step = model._hessian_step
    model._hessian_step = lambda p: scale * step(p)
    try:
        return model.standard_errors()
    finally:
        del model._hessian_step
        model._covariance_cache = None


def _count_numerical(monkeypatch, module):
    calls = []
    numerical = module.numerical_hessian

    def counted(*args, **kwargs):
        calls.append(1)
        return numerical(*args, **kwargs)

    monkeypatch.setattr(module, "numerical_hessian", counted)
    return calls


@pytest.mark.parametrize("name", FAMILIES)
def test_standard_errors_match_the_numerical_ones(name):
    model = _registry(name)
    assert model._information is not None
    exact = model.standard_errors()
    rtol = OLD_ROUNDING.get(name, 1e-4)
    np.testing.assert_allclose(exact, _numerical_se(model), rtol=rtol)
    if name in OLD_ROUNDING:
        np.testing.assert_allclose(
            exact, _numerical_se(model, 10.0), rtol=1e-4
        )


def _direct_hessian(model):
    """The natural-space Hessian of the free parameters by autograd,
    straight from the likelihood, at the point the model kept one."""
    (p_hat, center, data, _), _ = model._information
    free = [
        i
        for i, name in enumerate(model.parameter_names)
        if name not in model.fixed
    ]
    if center is not None:
        data = centred_copy(data, center)

    def neg_ll(v):
        full = anp.array(
            [v[free.index(i)] if i in free else p for i, p in enumerate(p_hat)]
        )
        return model.model.neg_ll(data, *full)

    return np.asarray(hessian(neg_ll)(p_hat[free]))


def _assert_same_hessian(kept, direct, rtol=1e-10):
    scale = np.max(np.abs(direct))
    assert np.max(np.abs(kept - direct)) <= rtol * scale


@pytest.mark.parametrize("name", FAMILIES)
def test_the_kept_hessian_is_the_natural_space_hessian(name):
    model = _registry(name)
    # The point is the one the covariance is computed at: the centred fit
    # behind a model mapped back to Z = 0.
    (p_hat, _, data, _), kept = model._information
    if model._fit_centring is not None:
        np.testing.assert_array_equal(p_hat, model._fit_centring[0])
    else:
        np.testing.assert_array_equal(p_hat, model.params)
    assert data is model.data
    # Exact to rounding (at most 4.3e-16 of the largest entry), except
    # through a Gamma baseline's shape, whose derivatives autograd takes by
    # central differences inside the incomplete gamma function
    # (utils/autograd_gamma_compat.py), so that the two routes agree only
    # to their accuracy (2e-8).
    rtol = 1e-7 if name.startswith("Gamma") else 1e-10
    _assert_same_hessian(kept, _direct_hessian(model), rtol)


def test_covariance_reads_no_numerical_hessian_and_is_kept(monkeypatch):
    calls = _count_numerical(monkeypatch, inference)
    model = _registry("WeibullPH")
    model.standard_errors()
    model.cb([5.0, 10.0], Z=[1.0, 0.0])
    model.param_cb("beta_0")
    assert calls == []
    # Without the exact Hessian it is differenced, once, and then kept.
    model._information = None
    model._covariance_cache = None
    first = model.covariance()
    second = model.covariance()
    model.cb([5.0, 10.0], Z=[1.0, 0.0])
    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)
    # (a copy: changing it changes nothing kept)
    first[0, 0] = -1.0
    assert model.covariance()[0, 0] > 0


def test_a_model_with_moved_parameters_falls_back(monkeypatch):
    # (LogNormalPH is fitted on the covariates as given, so its covariance
    # is computed at ``params``.)
    calls = _count_numerical(monkeypatch, inference)
    model = _registry("LogNormalPH")
    fitted = model.params
    exact = model.covariance()
    model.params = fitted + 1e-3
    moved = model.covariance()
    assert len(calls) == 1
    assert not np.array_equal(moved, exact)
    # and back again: the exact Hessian is at the fitted parameters
    model.params = fitted
    np.testing.assert_array_equal(model.covariance(), exact)
    assert len(calls) == 1


def test_a_model_with_other_data_falls_back(monkeypatch):
    calls = _count_numerical(monkeypatch, inference)
    model = _registry("WeibullAFT")
    model.covariance()
    model.data = copy.copy(model.data)
    model.covariance()
    assert len(calls) == 1


def test_accelerated_life_keeps_the_exact_information(monkeypatch):
    # Since #555 the accelerated-life fit keeps its exact information. The
    # registry case holds alpha, whose standard error is 0.
    model = _registry("WeibullAL[Power]")
    assert model._information is not None
    calls = _count_numerical(monkeypatch, inference)
    se = dict(zip(model.parameter_names, model.standard_errors()))
    assert not calls and se.pop("alpha") == 0
    assert all(np.isfinite(v) and v > 0 for v in se.values())


def test_aft_time_varying_fit_keeps_the_exact_information(monkeypatch):
    # Since #555 its accumulated-age likelihood is written for autograd.
    rng = np.random.default_rng(3)
    n = 200
    Z = rng.normal(0, 1, (n, 1))
    x = np.abs(rng.weibull(1.5, n) * 10 * np.exp(-0.4 * Z[:, 0])) + 0.5
    model = _fit(
        lambda: sp.WeibullAFT.fit_tvc(
            np.arange(n), np.zeros(n), x, np.zeros(n, dtype=int), Z
        )
    )
    assert model._information is not None
    calls = _count_numerical(monkeypatch, inference)
    se = model.standard_errors()
    assert not calls and np.all(np.isfinite(se)) and np.all(se > 0)


@pytest.mark.parametrize(
    "fitter, options, maps_back",
    [
        # a baseline parameter fixed, on centred covariates mapped back to
        # Z = 0 (the jacobian carries the covariance over)
        ("WeibullPH", {"fixed": {"beta": 1.8}}, True),
        # a coefficient fixed: a zero row and column (mapped, and not)
        ("WeibullAFT", {"fixed": {"beta_0": -0.5}}, True),
        ("LogNormalPH", {"fixed": {"beta_1": -0.7}}, False),
        # a baseline parameter the map moves, fixed: not centred
        ("WeibullPH", {"fixed": {"alpha": 9.0}}, False),
        # the baseline kept at the covariate means
        ("WeibullPH", {"center": True}, False),
        ("LogisticPO", {"center": True}, False),
        ("WeibullAH", {"center": True}, False),
    ],
)
def test_fixed_and_centred_fits(monkeypatch, fitter, options, maps_back):
    model = _fit(lambda: getattr(sp, fitter).fit(**reg_data(), **options))
    assert (model._fit_centring is not None) == maps_back
    assert model._information is not None
    (_, center, _, fixed), kept = model._information
    # (kept at the centre the fit ran at, mapped back or not)
    assert (center is not None) == (maps_back or "center" in options)
    assert fixed == tuple(sorted(options.get("fixed", {})))
    _assert_same_hessian(kept, _direct_hessian(model))
    calls = _count_numerical(monkeypatch, inference)
    exact = model.covariance()
    assert calls == []
    for name in options.get("fixed", {}):
        k = model.parameter_names.index(name)
        assert not np.any(exact[k]) and not np.any(exact[:, k])
    np.testing.assert_allclose(
        np.sqrt(np.diag(exact)), _numerical_se(model), rtol=1e-4, atol=0
    )


def test_a_runaway_keeps_no_hessian():
    # Its covariance is as before (the estimate is infinite anyway).
    d = reg_data()
    d["Z"][:, 0] = d["c"] == 1
    model = _fit(lambda: sp.WeibullPH.fit(**d))
    assert model._information is None


def test_natural_information_converts_exactly():
    # p = exp(t) and p = t, for f(p) = sum(w (p - a)^2) + p0^2 p1^2 away
    # from its minimum, where the gradient term of the chain rule counts.
    a, w = np.array([2.0, -1.0]), np.array([3.0, 5.0])

    def f_p(p):
        return anp.sum(w * (p - a) ** 2) + p[0] ** 2 * p[1] ** 2

    def to_natural(t):
        return anp.array([anp.exp(t[0]), t[1]])

    t = np.array([0.4, -0.2])
    H_t = hessian(lambda u: f_p(to_natural(u)))(t)
    g_t = grad(lambda u: f_p(to_natural(u)))(t)
    H_p = natural_information((H_t, g_t), to_natural, t)
    np.testing.assert_allclose(
        H_p, hessian(f_p)(to_natural(t)), rtol=1e-14, atol=1e-14
    )
    # Not positive definite, or not finite, in either space: none.
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert natural_information((indefinite, g_t), to_natural, t) is None
    bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert natural_information((bad, g_t), to_natural, t) is None
    assert natural_information(None, to_natural, t) is None


def test_serialisation_round_trips_the_covariance():
    model = _registry("WeibullPO")
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    np.testing.assert_array_equal(
        restored.standard_errors(), model.standard_errors()
    )
    again = sp.from_dict(restored.to_dict())
    np.testing.assert_array_equal(again.covariance(), model.covariance())


@pytest.mark.parametrize(
    "name", ["WeibullPH", "LogNormalPH", "WeibullAFT", "LogisticPO"]
)
def test_to_dict_round_trips_unchanged(name):
    # The kept Hessian and covariance are not part of the dict: it is the
    # same before and after a covariance call, and after a reload.
    model = _registry(name)
    first = model.to_dict()
    assert "_information" not in first and "_covariance_cache" not in first
    model.standard_errors()
    assert model.to_dict() == first
    assert sp.from_dict(first).to_dict() == first


def _frailty_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 6)
    u = rng.gamma(2.0, 0.5, 30)[groups]
    Z = rng.binomial(1, 0.5, (180, 1))
    H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    return {"x": 10 * H**0.5, "Z": Z, "groups": groups}


def _frailty_direct_hessian(fitter, d, model):
    x = np.asarray(d["x"], dtype=float)
    Zc = np.asarray(d["Z"], dtype=float)
    inv = np.unique(d["groups"], return_inverse=True)[1]
    nat = np.concatenate([model.dist_params, model.beta, [model.theta]])

    def neg_ll(v):
        return fitter._neg_ll_natural(
            v,
            x,
            np.zeros(x.size, int),
            np.ones(x.size),
            Zc,
            inv,
            Zc.shape[1],
            frailty._AUTOGRAD,
        )

    return np.asarray(hessian(neg_ll)(nat))


@pytest.mark.parametrize(
    "name",
    [
        "WeibullFrailty",
        "ExponentialFrailty",
        "GammaFrailty",
        "LogNormalFrailty",
    ],
)
def test_frailty_uses_the_exact_hessian(monkeypatch, name):
    # A frailty variance inside its range (0.43, 0.055, 0.29 and 0.14):
    # exact.
    fitter = getattr(sp, name)
    kept = []
    convert = frailty.natural_information

    def keeping(*args):
        kept.append(convert(*args))
        return kept[-1]

    monkeypatch.setattr(frailty, "natural_information", keeping)
    calls = _count_numerical(monkeypatch, frailty)
    model = _fit(lambda: fitter.fit(**_frailty_data()))
    assert calls == [] and model.covariance() is not None
    # (a Gamma baseline to the accuracy of its shape derivatives, as above)
    _assert_same_hessian(
        kept[-1],
        _frailty_direct_hessian(fitter, _frailty_data(), model),
        1e-7 if name.startswith("Gamma") else 1e-10,
    )
    monkeypatch.setattr(frailty, "natural_information", lambda *a: None)
    numerical = _fit(lambda: fitter.fit(**_frailty_data()))
    assert len(calls) == 1
    np.testing.assert_allclose(
        np.sqrt(np.diag(model.covariance())),
        np.sqrt(np.diag(numerical.covariance())),
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "name",
    [
        "WeibullFrailty",
        "ExponentialFrailty",
        "GammaFrailty",
        "LogNormalFrailty",
    ],
)
def test_frailty_variance_at_its_limit_falls_back(monkeypatch, name):
    # The variance of the registry's data runs to its limit of 0, where the
    # likelihood no longer depends on it: the exact Hessian is singular and
    # the covariance is what it was, from the numerical one. (The search
    # stops once theta no longer changes the likelihood, 1e-16 to 1e-18
    # here; Nelder-Mead, the first rung before #515, went on to 1e-102 to
    # 1e-23.)
    calls = _count_numerical(monkeypatch, frailty)
    model = _registry(name)
    # Where the search stops, the exact Hessian can still, just, be
    # positive definite and then is used: whether it is depends on the last
    # bits of theta (the Weibull's at 2.8e-17 here; the Gamma's and the
    # Exponential's on some CPUs), so either covariance is accepted.
    assert model.theta < 1e-15 and len(calls) in (0, 1)
    assert np.all(np.isfinite(model.covariance()))


def test_fine_gray_standard_errors_come_from_the_check(monkeypatch):
    # The subdistribution fit's standard errors invert the exact Hessian
    # the no-maximum check took (``judge_search``, which checks the
    # maximum with it too), rather than taking it again.
    case = CASE_BY_NAME["FineGray"]
    calls, seen = [], []
    original, take = fine_gray.hessian, skeleton.search_derivatives

    def counted(f):
        calls.append(1)
        return original(f)

    def taking(neg_ll, beta):
        seen.append((neg_ll, np.array(beta)))
        return take(neg_ll, beta)

    monkeypatch.setattr(fine_gray, "hessian", counted)
    monkeypatch.setattr(skeleton, "search_derivatives", taking)
    model = _fit(lambda: case.fit(case.data()))
    assert calls == [] and len(seen) == 1
    neg_ll, beta = seen[0]
    np.testing.assert_array_equal(beta, model.beta)
    direct = np.linalg.inv(original(neg_ll)(beta))
    np.testing.assert_allclose(model.covariance(), direct, rtol=1e-10, atol=0)
