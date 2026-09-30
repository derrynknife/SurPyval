"""The regression covariance comes from the fit's exact Hessian (#392).

The no-maximum check computes the Hessian of the negative log-likelihood at
the fit with autograd, in the optimiser's search space. The fit converts it
exactly to the natural parameters (``_fit_skeleton.natural_information``)
and keeps it, and ``covariance()``, ``standard_errors()``, ``cb()`` and
``param_cb()`` invert it instead of differencing the likelihood numerically
on every call; the covariance is then kept too. A model without it -- an
accelerated-life fit, a runaway, a restored model -- falls back to the
numerical Hessian as before.
"""

import warnings

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import hessian

import surpyval as sp
import surpyval.univariate.regression.frailty.frailty_fitter as frailty
import surpyval.univariate.regression.parametric_regression_model as prm
from surpyval.tests.conformance.registry import CASE_BY_NAME, reg_data
from surpyval.univariate.regression._fit_skeleton import centred_copy

FAMILIES = [
    name
    for name in CASE_BY_NAME
    if name.endswith(("PH", "AFT", "PO", "AH")) and name != "CoxPH"
]


def _fit(fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit()


def _registry(name):
    case = CASE_BY_NAME[name]
    return _fit(lambda: case.fit(case.data()))


def _numerical_se(model):
    """The standard errors from the numerical Hessian, as before."""
    model._information = None
    model._covariance_cache = None
    return model.standard_errors()


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
    # The numerical Hessian's own error is what differs: at most 4.9e-4
    # (ExponentialPO), which shrinks towards the exact values as its step
    # grows from rounding's reach; 5e-5 or less elsewhere.
    np.testing.assert_allclose(exact, _numerical_se(model), rtol=1e-3)


def _direct_hessian(model):
    """The natural-space Hessian of the free parameters by autograd,
    straight from the likelihood."""
    p_hat, center, _ = model._information
    free = [
        i
        for i, name in enumerate(model.parameter_names())
        if name not in model.fixed
    ]
    data = model.data
    if center is not None and np.any(center):
        data = centred_copy(data, center)

    def neg_ll(v):
        full = anp.array(
            [v[free.index(i)] if i in free else p for i, p in enumerate(p_hat)]
        )
        return model.model.neg_ll(data, *full)

    return np.asarray(hessian(neg_ll)(p_hat[free]))


@pytest.mark.parametrize("name", FAMILIES)
def test_the_kept_hessian_is_the_natural_space_hessian(name):
    model = _registry(name)
    kept = model._information[2]
    direct = _direct_hessian(model)
    # Exact to rounding, except through a Gamma baseline's shape, whose
    # derivatives autograd takes by central differences inside the
    # incomplete gamma function (utils/autograd_gamma_compat.py), so that
    # the two routes agree only to their accuracy.
    rtol = 1e-6 if name.startswith("Gamma") else 1e-10
    scale = np.max(np.abs(direct))
    assert np.max(np.abs(kept - direct)) <= rtol * scale


def test_covariance_reads_no_numerical_hessian_and_is_kept(monkeypatch):
    calls = _count_numerical(monkeypatch, prm)
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
    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)
    # (a copy: changing it changes nothing kept)
    first[0, 0] = -1.0
    assert model.covariance()[0, 0] > 0


def test_a_model_with_moved_parameters_falls_back(monkeypatch):
    # (LogNormalPH is fitted on the covariates as given, so its covariance
    # is computed at ``params``.)
    calls = _count_numerical(monkeypatch, prm)
    model = _registry("LogNormalPH")
    model.params = model.params + 1e-3
    model._covariance_cache = None
    model.covariance()
    assert len(calls) == 1


def test_accelerated_life_uses_the_numerical_hessian(monkeypatch):
    calls = _count_numerical(monkeypatch, prm)
    model = _registry("WeibullAL[Power]")
    assert model._information is None
    model.standard_errors()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "fitter, options",
    [
        # a baseline parameter fixed, on centred covariates mapped back to
        # Z = 0 (the jacobian carries the covariance over)
        ("WeibullPH", {"fixed": {"beta": 1.8}}),
        # a coefficient fixed: a zero row and column
        ("WeibullAFT", {"fixed": {"beta_0": -0.5}}),
        ("LogNormalPH", {"fixed": {"beta_1": -0.7}}),
        # the baseline kept at the covariate means
        ("WeibullPH", {"center": True}),
        ("LogisticPO", {"center": True}),
        ("WeibullAH", {"center": True}),
    ],
)
def test_fixed_and_centred_fits(monkeypatch, fitter, options):
    model = _fit(lambda: getattr(sp, fitter).fit(**reg_data(), **options))
    assert model._information is not None
    calls = _count_numerical(monkeypatch, prm)
    exact = model.covariance()
    assert calls == []
    for name in options.get("fixed", {}):
        k = model.parameter_names().index(name)
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


def test_serialisation_round_trips_the_covariance():
    model = _registry("WeibullPO")
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    np.testing.assert_array_equal(
        restored.standard_errors(), model.standard_errors()
    )


def _frailty_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 6)
    u = rng.gamma(2.0, 0.5, 30)[groups]
    Z = rng.binomial(1, 0.5, (180, 1))
    H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    return {"x": 10 * H**0.5, "Z": Z, "groups": groups}


@pytest.mark.parametrize("name", ["WeibullFrailty", "LogNormalFrailty"])
def test_frailty_uses_the_exact_hessian(monkeypatch, name):
    # A frailty variance inside its range (0.43 and 0.14): exact.
    calls = _count_numerical(monkeypatch, frailty)
    model = _fit(lambda: getattr(sp, name).fit(**_frailty_data()))
    assert calls == [] and model.covariance is not None
    monkeypatch.setattr(frailty, "natural_information", lambda *a: None)
    numerical = _fit(lambda: getattr(sp, name).fit(**_frailty_data()))
    assert len(calls) == 1
    np.testing.assert_allclose(
        np.sqrt(np.diag(model.covariance)),
        np.sqrt(np.diag(numerical.covariance)),
        rtol=1e-4,
    )


def test_frailty_variance_at_its_limit_falls_back(monkeypatch):
    # The variance of these data runs to its limit of 0, where the
    # likelihood no longer depends on it: the exact Hessian is singular and
    # the covariance is what it was, from the numerical one.
    calls = _count_numerical(monkeypatch, frailty)
    model = _registry("WeibullFrailty")
    assert model.theta < 1e-50 and len(calls) == 1
