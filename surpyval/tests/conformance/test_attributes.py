"""Every builder of a model gives it the same attributes.

A fitted regression model is created empty and filled in by its builder:
``fit``, the alternate fit paths (``fit_from_df`` by column names or by a
formula, ``fit_tvc``) and ``from_dict``. Each attribute it may carry is
declared on its class, with its type and, where it is optional, its
default. These properties check that every builder gives the model the
same attributes (the declared names it can be asked for) and nothing
undeclared, so a builder that drifts from the others -- one that forgets
an attribute, or adds one of its own -- fails here rather than only on
the one path that reads it.

``ParametricRegressionModel.from_params`` does not exist, so it is not a
builder here. A model restored by ``from_dict`` does not carry the data
it was fitted to, nor what came with them (``RESTORED_WITHOUT``).
"""

import inspect

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    cases_for,
    refit,
)

# What a model restored by ``from_dict`` lacks: the fitted data, the
# optimiser's result and (accelerated life) the search objective, none of
# which a dict stores.
RESTORED_WITHOUT = frozenset({"data", "res", "fun"})


def declared(cls):
    """The attribute names ``cls`` and its bases annotate."""
    names = set()
    for klass in cls.__mro__:
        if klass is not object:
            names |= set(inspect.get_annotations(klass))
    return names


def attributes(model):
    """The declared attributes ``model`` has (set or defaulted)."""
    return {name for name in declared(type(model)) if hasattr(model, name)}


def _tvc_path(case):
    # One (0, x] interval per subject is the time-fixed data.
    fitter = getattr(sp, case.name, None)
    if not hasattr(fitter, "fit_tvc"):
        return None

    def fit_tvc(d):
        n = np.asarray(d["n"])
        return fitter.fit_tvc(
            np.arange(n.sum()),
            np.zeros(n.sum()),
            np.repeat(d["x"], n),
            np.repeat(d["c"], n),
            np.repeat(d["Z"], n, axis=0),
        )

    return fit_tvc


def _from_dict(case):
    def restore(d):
        model = refit(case, d)
        return type(model).from_dict(model.to_dict())

    return restore


def _builders(case):
    out = {"fit": lambda d: refit(case, d), **case.paths}
    tvc = _tvc_path(case)
    if tvc is not None:
        out["fit_tvc"] = tvc
    out["from_dict"] = _from_dict(case)
    return out


def _builder_params(skip=()):
    params = []
    for param in cases_for("attributes"):
        case = param.values[0]
        for name in _builders(case):
            if name in skip:
                continue
            params.append(
                pytest.param(
                    case, name, id=f"{case.name}-{name}", marks=param.marks
                )
            )
    return params


@pytest.mark.parametrize("case, builder", _builder_params(skip={"fit"}))
def test_every_builder_gives_the_same_attributes(case, builder):
    data = case.data()
    ref = attributes(refit(case, data))
    got = attributes(_builders(case)[builder](data))
    if builder == "from_dict":
        ref -= RESTORED_WITHOUT
    assert (
        got == ref
    ), f"missing: {sorted(ref - got)}; extra: {sorted(got - ref)}"


@pytest.mark.parametrize("case, builder", _builder_params())
def test_every_attribute_is_declared(case, builder):
    model = _builders(case)[builder](case.data())
    undeclared = set(vars(model)) - declared(type(model))
    assert not undeclared, sorted(undeclared)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#TBD: a model restored by from_dict has no data, res (or, for "
        "accelerated life, fun): the dict does not store them. Whether "
        "to declare them None there is the maintainer's decision."
    ),
)
@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAL[Power]"])
def test_a_restored_model_has_every_attribute(name):
    case = CASE_BY_NAME[name]
    data = case.data()
    ref = attributes(refit(case, data))
    got = attributes(_from_dict(case)(data))
    assert got == ref, f"missing: {sorted(ref - got)}"
