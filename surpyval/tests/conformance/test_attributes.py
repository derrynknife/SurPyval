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

import pytest

from surpyval.tests.conformance.registry import (
    cases_for,
    refit,
    tvc_path,
)

# What a model restored by ``from_dict`` lacks, by class: the fitted data
# and what was computed from them -- the optimiser's result, the search
# objective (accelerated life), Cox's likelihood closures, Lin-Ying's
# estimating-equation terms -- none of which a dict stores.
RESTORED_WITHOUT = {
    "ParametricRegressionModel": frozenset({"data", "res", "fun"}),
    "SemiParametricRegressionModel": frozenset(
        {"_fit_data", "jac", "neg_ll", "res"}
    ),
    "AdditiveHazardsModel": frozenset({"_A", "_b"}),
}


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


def _from_dict(case):
    def restore(d):
        model = refit(case, d)
        return type(model).from_dict(model.to_dict())

    return restore


def _builders(case):
    out = {"fit": lambda d: refit(case, d), **case.paths}
    tvc = tvc_path(case)
    if tvc is not None and "fit_tvc" not in out:
        out["fit_tvc"] = tvc
    if case.applies("serialise"):
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
        name = case.model_class.rpartition(".")[2]
        ref -= RESTORED_WITHOUT.get(name, frozenset())
    assert (
        got == ref
    ), f"missing: {sorted(ref - got)}; extra: {sorted(got - ref)}"


@pytest.mark.parametrize("case, builder", _builder_params())
def test_every_attribute_is_declared(case, builder):
    model = _builders(case)[builder](case.data())
    undeclared = set(vars(model)) - declared(type(model))
    assert not undeclared, sorted(undeclared)
