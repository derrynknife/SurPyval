"""Every public model is registered (#379).

Walks SurPyval's public namespaces and fails for any public class,
fitter or function defined in the package that is neither covered by a
registered case (as one of its ``fitters`` or its ``model_class``) nor
listed in ``OUT_OF_SCOPE`` with a reason. A new model therefore gets the
whole conformance battery by being registered, or an explicit decision
that it should not.

The rest checks the registry itself: every case fits to its declared
model class, and every name, property and known failure it refers to
exists.
"""

import importlib
import inspect

import pytest

from surpyval.tests.conformance.registry import (
    CASES,
    KNOWN_FAILURES,
    OUT_OF_SCOPE,
    PROPERTIES,
    cases_for,
    fitted,
)

NAMESPACES = (
    "surpyval",
    "surpyval.univariate.parametric",
    "surpyval.univariate.nonparametric",
    "surpyval.univariate.regression",
    "surpyval.univariate.competing_risks",
    "surpyval.recurrent",
    "surpyval.degradation",
    "surpyval.multivariate",
    "surpyval.beta",
    "surpyval.beta.ml",
)


def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _public(namespace):
    """(dotted name, object) for each public class, fitter instance or
    function that SurPyval itself defines in ``namespace``."""
    module = importlib.import_module(namespace)
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    for name in names:
        obj = getattr(module, name)
        if inspect.ismodule(obj):
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            owner = obj.__module__
        elif hasattr(obj, "fit"):  # a singleton fitter instance
            owner = type(obj).__module__
        else:
            continue  # constants, dicts, typing helpers
        if owner.startswith("surpyval"):
            yield f"{namespace}.{name}", obj


def _covered():
    objects = {}
    for case in CASES:
        for dotted in case.fitters + (case.model_class,):
            objects[id(_resolve(dotted))] = case.name
    for dotted in OUT_OF_SCOPE:
        objects[id(_resolve(dotted))] = "out of scope"
    return objects


def test_every_public_model_is_registered_or_out_of_scope():
    covered = _covered()
    missing = sorted(
        {
            dotted
            for namespace in NAMESPACES
            for dotted, obj in _public(namespace)
            if id(obj) not in covered
        }
    )
    assert not missing, (
        "Public names neither registered in "
        "surpyval/tests/conformance/registry.py nor listed in its "
        f"OUT_OF_SCOPE with a reason: {missing}"
    )


def test_every_serialisable_model_is_registered():
    # The package-level reader knows every model class that can be saved,
    # exported or not (FineGrayModel, RenewalModel, ...): each must be some
    # case's model class (test_case_fits_to_its_model_class checks that
    # the case really produces one).
    from surpyval.serialisation import (
        _PARAMETERIZATIONS,
        _TAGGED_MODELS,
    )
    from surpyval.serialisation import _resolve as resolve_tag

    covered = _covered()
    classes = [resolve_tag(m, tag) for tag, m in _TAGGED_MODELS.items()]
    classes += [resolve_tag(m, n) for m, n in _PARAMETERIZATIONS.values()]
    missing = [
        getattr(cls, "__name__", type(cls).__name__)
        for cls in classes
        if id(cls) not in covered and id(type(cls)) not in covered
    ]
    assert not missing, missing


def test_out_of_scope_entries_exist_and_have_reasons():
    for dotted, reason in OUT_OF_SCOPE.items():
        _resolve(dotted)
        assert reason.strip(), dotted


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c, id=c.name, marks=[pytest.mark.slow] if "*" in c.slow else []
        )
        for c in CASES
    ],
)
def test_case_fits_to_its_model_class(case):
    model = fitted(case)
    cls = _resolve(case.model_class)
    if not isinstance(cls, type):  # a singleton: the fitter is the model's
        cls = type(cls)
    assert model is cls or isinstance(model, cls), type(model)
    for dotted in case.fitters:
        _resolve(dotted)


def test_exclusions_and_failures_name_real_properties():
    names = {case.name for case in CASES}
    assert set(KNOWN_FAILURES) <= names, set(KNOWN_FAILURES) - names
    for case in CASES:
        for prop, reason in {**case.exclude, **case.xfail}.items():
            assert prop.split("[")[0] in PROPERTIES, (case.name, prop)
            assert reason.strip(), (case.name, prop)
        for key in case.xfail:
            if key.startswith("fit_paths["):
                assert key[len("fit_paths[") : -1] in case.paths, key
            if key.startswith("missing_fit["):
                inputs = (case.covariates,) + case.labels
                assert key[len("missing_fit[") : -1] in inputs, key


@pytest.mark.parametrize("prop", sorted(PROPERTIES))
def test_every_property_runs_on_some_model(prop):
    assert cases_for(prop), prop


def test_every_known_failure_names_its_issue():
    """Each known failure is tracked by an issue, whose number leads its
    xfail reason (from ``KNOWN_FAILURE_ISSUES`` or the reason itself)."""
    import re

    from surpyval.tests.conformance.registry import KNOWN_INCONSISTENCIES

    untracked = [
        f"{case.name}: {prop}"
        for case in CASES
        for prop, reason in case.xfail.items()
        if not re.match(r"#\d+: ", reason)
    ]
    untracked += [
        key
        for key, reason in KNOWN_INCONSISTENCIES.items()
        if not re.match(r"#\d+: ", reason)
    ]
    assert not untracked, untracked
