"""``params`` and ``param_names`` (principle 21, #483).

User code reads a fitted model's parameters from ``params``, and code
written against one model (a report builder, a serialiser) has to work for
the others. So wherever a registered model has a ``param_names`` list, it
names ``params`` entry by entry: the same length, strings, no repeats. The
frailty and renewal models, which had neither (#483), must have both.

Not every model has ``param_names`` yet: the univariate parametric models
name theirs in ``dist.param_names``, the parametric recurrence models and
the regression models in ``parameter_names``. Those are left to the
maintainer's decision on one spelling (see #483).
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import CASES, fitted

_NAMED = {"FrailtyModel", "RenewalModel"}


def _cases(where):
    return [
        pytest.param(case, id=case.name)
        for case in CASES
        if where(case.model_class.rpartition(".")[2])
    ]


@pytest.mark.parametrize("case", _cases(lambda cls: cls in _NAMED))
def test_model_has_params_and_param_names(case):
    model = fitted(case)
    params = model.params
    assert isinstance(params, np.ndarray) and params.ndim == 1
    assert params.dtype.kind == "f" and np.all(np.isfinite(params))
    assert isinstance(model.param_names, list)
    assert len(model.param_names) == params.size


@pytest.mark.parametrize("case", _cases(lambda cls: True))
def test_param_names_name_params(case):
    model = fitted(case)
    names = getattr(model, "param_names", None)
    if names is None or callable(names):
        pytest.skip("no param_names list")
    names = list(names)
    assert all(isinstance(n, str) for n in names)
    assert len(set(names)) == len(names), names
    assert len(names) == np.asarray(model.params).size, names


@pytest.mark.parametrize("case", _cases(lambda cls: cls in _NAMED))
def test_params_survive_serialisation(case):
    import json

    import surpyval

    model = fitted(case)
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.params, model.params, rtol=1e-12)
    assert restored.param_names == model.param_names


def test_documented_orders():
    # frailty: baseline, coefficients, theta; renewal: restoration first,
    # the order of parameter_names, covariance and standard_errors.
    by_name = {case.name: case for case in CASES}
    frailty = fitted(by_name["WeibullFrailty"])
    assert frailty.param_names[:2] == ["alpha", "beta"]
    assert frailty.param_names[-1] == "theta"
    np.testing.assert_array_equal(
        frailty.params,
        np.r_[frailty.dist_params, frailty.beta, frailty.theta],
    )
    for name, restoration in [("GeneralizedOneRenewal", "q"), ("ARA", "rho")]:
        renewal = fitted(by_name[name])
        assert renewal.param_names == renewal.parameter_names
        assert renewal.param_names[0] == restoration
        np.testing.assert_array_equal(
            renewal.params, np.r_[renewal.restoration, renewal.model.params]
        )
        assert renewal.params.size == renewal.standard_errors().size
