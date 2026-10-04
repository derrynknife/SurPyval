"""``params`` and ``parameter_names`` (principle 21, #483).

User code reads a fitted model's parameters from ``params``, and code
written against one model (a report builder, a serialiser) has to work for
the others. So every registered model with ``params`` has
``parameter_names``, a list of strings that names ``params`` entry by
entry: the same length, no repeats. Every distribution has one too, naming
the parameters its functions take. Two models name more or less than a
flat ``params``, and say so in their docstrings:

- ``MixtureModel``: ``params`` has one row per component, and
  ``parameter_names`` names its columns (the component distribution's);
- ``ProportionalIntensityModel``: ``params`` is the base rate and
  ``coeffs`` the coefficients; ``parameter_names`` names both, base rate
  first, the order of ``covariance`` and ``standard_errors``.

An accelerated life model names its placeholder life-parameter slot too,
and a fit with fixed parameters names them, so the list always lines up
with ``params``.

``parameter_names`` replaced three spellings: the ``param_names``
attribute, the regression models' ``parameter_names()`` method and the
recurrent models' ``parameter_names`` property. The old spellings,
deprecated in v0.22, are removed in v0.23 (principle 21), and the package
never uses a deprecated name: a fit, its predictions, summary and
serialisation raise no DeprecationWarning.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    CASES,
    fitted,
    predictions,
)
from surpyval.univariate.parametric import ParametricFitter

_WITH_PARAMS = [
    pytest.param(case, id=case.name)
    for case in CASES
    if case.model_class.rpartition(".")[2]
    not in {
        # No ``params``: step estimates, trees, cause-by-cause models,
        # unit-by-unit degradation analyses.
        "NonParametric",
        "SurvivalTree",
        "RandomSurvivalForest",
        "CompetingRisks",
        "ParametricCompetingRisks",
        "CompetingRisksProportionalHazards",
        "FineGrayModel",
        "NonParametricCounting",
        "CauseSpecificMCF",
        "CauseSpecificNHPP",
        "DegradationModel",
        "DestructiveDegradationModel",
        "InducedFailureDistribution",
    }
    and hasattr(fitted(case), "params")
]

_DISTRIBUTIONS = sorted(
    name
    for name in dir(surpyval)
    if isinstance(getattr(surpyval, name), ParametricFitter)
)


def _named_values(model):
    """What ``parameter_names`` names, entry by entry."""
    kind = type(model).__name__
    if kind == "MixtureModel":
        return np.asarray(model.params)[0]
    if kind == "ProportionalIntensityModel":
        return np.r_[model.params, model.coeffs]
    return np.atleast_1d(np.asarray(model.params))


def _check_names(names):
    assert isinstance(names, list), type(names)
    assert all(isinstance(n, str) for n in names), names
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("case", _WITH_PARAMS)
def test_parameter_names_name_params(case):
    model = fitted(case)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        names = model.parameter_names
    _check_names(names)
    assert len(names) == _named_values(model).size, names


def test_every_model_with_params_is_checked():
    # The skip list above names only models without ``params``.
    for case in CASES:
        if case.name not in {p.id for p in _WITH_PARAMS}:
            assert not hasattr(fitted(case), "params"), case.name


@pytest.mark.parametrize("case", _WITH_PARAMS)
def test_param_names_is_gone(case):
    # The pre-0.22 spelling, deprecated in v0.22, is removed in v0.23.
    assert not hasattr(fitted(case), "param_names")


@pytest.mark.parametrize("name", _DISTRIBUTIONS)
def test_distribution_parameter_names(name):
    dist = getattr(surpyval, name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        names = dist.parameter_names
    _check_names(names)
    assert len(names) == dist.k == len(dist.bounds)
    assert not hasattr(dist, "param_names")
    # A fitted model names its params the same way.
    if names and name in CASE_BY_NAME:
        assert fitted(CASE_BY_NAME[name]).parameter_names == names


def test_regression_parameter_names_is_a_plain_list():
    model = fitted(CASE_BY_NAME["WeibullPH"])
    names = model.parameter_names
    # The method spelling, deprecated in v0.22, is removed in v0.23.
    assert type(names) is list
    with pytest.raises(TypeError, match="not callable"):
        model.parameter_names()
    assert names == ["alpha", "beta", "coef_0", "coef_1"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert names == list(names) and list(names) == names
        assert len(names) == 4 and names[-1] == "coef_1"
        assert [n for n in names] == ["alpha", "beta", "coef_0", "coef_1"]
        assert json.loads(json.dumps(names)) == list(names)
        assert np.asarray(names).tolist() == list(names)
        series = pd.Series(model.params, index=names)
        assert series.index.tolist() == list(names)
        frame = pd.DataFrame([model.params], columns=names)
        assert frame[names].shape == (1, 4)
        assert frame.T.loc[names].shape == (4, 1)
        assert names + ["x"] == ["alpha", "beta", "coef_0", "coef_1", "x"]
        assert names.index("coef_0") == 2


def test_documented_orders():
    # frailty: baseline, coefficients, theta; renewal: restoration first,
    # the order of covariance and standard_errors.
    frailty = fitted(CASE_BY_NAME["WeibullFrailty"])
    assert frailty.parameter_names[:2] == ["alpha", "beta"]
    assert frailty.parameter_names[-1] == "theta"
    np.testing.assert_array_equal(
        frailty.params,
        np.r_[frailty.dist_params, frailty.beta, frailty.theta],
    )
    for name, restoration in [("GeneralizedOneRenewal", "q"), ("ARA", "rho")]:
        renewal = fitted(CASE_BY_NAME[name])
        assert renewal.parameter_names[0] == restoration
        np.testing.assert_array_equal(
            renewal.params, np.r_[renewal.restoration, renewal.model.params]
        )
        assert renewal.params.size == renewal.standard_errors().size
    # accelerated life: the placeholder life-parameter slot is named
    al = fitted(CASE_BY_NAME["WeibullAL[Power]"])
    assert al.parameter_names == ["alpha", "beta", "a", "n"]
    assert al.life_parameter == "alpha"
    # proportional intensity: base rate, then the coefficients
    pi = fitted(CASE_BY_NAME["ProportionalIntensityNHPP"])
    assert pi.parameter_names == ["alpha", "b", "coef_0"]
    assert pi.standard_errors().size == len(pi.parameter_names)


@pytest.mark.parametrize(
    "name", ["WeibullFrailty", "ARA", "WeibullPH", "HPP", "Weibull"]
)
def test_parameter_names_survive_serialisation(name):
    model = fitted(CASE_BY_NAME[name])
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.params, model.params, rtol=1e-12)
    assert restored.parameter_names == model.parameter_names


def test_saved_key_is_still_param_names():
    # The on-disk key is unchanged, so files written by 0.21 load and
    # files written now load in 0.21 (principle 20).
    for name in ["WeibullFrailty", "ProportionalIntensityNHPP", "Weibull"]:
        model = fitted(CASE_BY_NAME[name])
        saved = model.to_dict()
        assert "param_names" in saved and "parameter_names" not in saved
        restored = surpyval.from_dict(json.loads(json.dumps(saved)))
        assert restored.parameter_names == model.parameter_names


def test_old_keyword_and_class_attribute_are_gone():
    # Deprecated in v0.22, removed in v0.23: the keyword is an unknown
    # argument, and a ``param_names`` class attribute is not read.
    from surpyval.degradation import PathModel
    from surpyval.multivariate import Copula

    def Hf(x, *params):
        return params[0] * x ** params[1]

    with pytest.raises(
        TypeError, match="unexpected keyword argument 'param_names'"
    ):
        surpyval.CustomDistribution(
            "ParamNamesKeyword",
            Hf,
            param_names=["a", "b"],
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        class OldPath(PathModel):
            name = "old"
            param_names = ["a", "b"]

            def path(self, x, a, b):
                return a + b * x

            def inv_path(self, y, a, b):
                return (y - a) / b

        class OldCopula(Copula):
            param_names = ["a"]

    assert not hasattr(OldPath, "parameter_names")
    assert OldCopula.parameter_names == ["theta"]


def _sweep(case):
    """Fit, predict, summarise and save one case, and read its names."""
    model = case.fit(case.data())
    predictions(case, model)
    repr(model)
    if callable(getattr(model, "summary", None)):
        model.summary()
    getattr(model, "parameter_names", None)
    if case.applies("serialise"):
        saved = json.loads(json.dumps(model.to_dict()))
        surpyval.from_dict(saved)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in CASES])
def test_package_uses_no_deprecated_name(case):
    # Every DeprecationWarning is an error, whoever raises it; the fits'
    # own warnings (convergence, no maximum, ...) are not what is checked.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", DeprecationWarning)
        _sweep(case)
