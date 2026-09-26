"""Package-level readers for serialised SurPyval models.

Every serialisable fitted SurPyval model writes ``to_dict`` /
``to_json``; the matching class-level ``from_dict`` / ``from_json``
readers require knowing up front which class wrote the file. The
package-level readers here dispatch on the serialised dictionary
itself, so one call restores a model of the right class no matter
which model wrote it:

.. code:: python

    import surpyval

    model = surpyval.from_json("model.json")  # any model's file
    model = surpyval.from_dict(model_dict)    # any model's dict

Dispatch uses the two conventions in the serialised dictionaries:
most classes write a ``"model"`` tag equal to their class name, and
the core univariate families are identified by their
``"parameterization"`` (``"parametric"``, ``"non-parametric"`` or
``"parametric-regression"``).
"""

import json
import numbers
import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# ``"model"`` tag -> defining module. The tag every class writes into
# its dict is its own class name, so the registry only needs to find
# the defining module, lazily (importing everything eagerly here would
# be a heavy import and a cycle risk).
_TAGGED_MODELS: dict[str, str] = {
    "SemiParametricRegressionModel": (
        "surpyval.univariate.regression.semi_parametric_regression_model"
    ),
    "FrailtyModel": ("surpyval.univariate.regression.frailty.frailty_model"),
    "AdditiveHazardsModel": (
        "surpyval.univariate.regression.additive_hazards.additive_hazards"
    ),
    "BuckleyJamesModel": (
        "surpyval.univariate.regression.buckley_james.buckley_james"
    ),
    "MixtureModel": "surpyval.univariate.parametric.mixture_model",
    "RoystonParmarModel": ("surpyval.univariate.parametric.royston_parmar"),
    "FineGrayModel": (
        "surpyval.univariate.competing_risks.regression.fine_gray"
    ),
    "CompetingRisksProportionalHazards": (
        "surpyval.univariate.competing_risks.regression"
        ".competing_risks_proportional_hazard"
    ),
    "ParametricCompetingRisks": (
        "surpyval.univariate.competing_risks.parametric"
        ".parametric_competing_risks"
    ),
    "CompetingRisks": (
        "surpyval.univariate.competing_risks.nonparametric.competing_risks"
    ),
    "CauseSpecificMCF": (
        "surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf"
    ),
    "CauseSpecificNHPP": (
        "surpyval.recurrent.competing_risks.parametric.cause_specific_nhpp"
    ),
    "NonParametricCounting": "surpyval.recurrent.nonparametric.mcf",
    "ParametricRecurrenceModel": (
        "surpyval.recurrent.parametric.parametric_recurrence"
    ),
    "ProportionalIntensityModel": (
        "surpyval.recurrent.regression.proportional_intensity"
    ),
    "RenewalModel": "surpyval.recurrent.renewal.renewal_model",
    "DegradationModel": "surpyval.degradation.degradation_analysis",
    "InducedFailureDistribution": (
        "surpyval.degradation.degradation_analysis"
    ),
    "WienerProcessModel": "surpyval.degradation.process_models",
    "GammaProcessModel": "surpyval.degradation.process_models",
    "DestructiveDegradationModel": "surpyval.degradation.destructive",
    "SurvivalTree": "surpyval.beta.ml.forest.tree",
    "RandomSurvivalForest": "surpyval.beta.ml.forest.forest",
    # The degenerate distributions are stateless: the class is the model,
    # so they serialise by name alone.
    "NeverOccurs": ("surpyval.univariate.parametric.distributions.degenerate"),
    "InstantlyOccurs": (
        "surpyval.univariate.parametric.distributions.degenerate"
    ),
}

# ``"parameterization"`` value -> (defining module, class name), for
# the core univariate families, which carry no ``"model"`` class tag.
# (The non-parametric dicts do have a ``"model"`` key, but it holds the
# estimator name -- e.g. ``"Kaplan-Meier"`` -- not a class name.)
_PARAMETERIZATIONS: dict[str, tuple[str, str]] = {
    "parametric": (
        "surpyval.univariate.parametric.parametric",
        "Parametric",
    ),
    "non-parametric": (
        "surpyval.univariate.nonparametric.nonparametric",
        "NonParametric",
    ),
    "parametric-regression": (
        "surpyval.univariate.regression.parametric_regression_model",
        "ParametricRegressionModel",
    ),
    "copula": (
        "surpyval.multivariate.parametric.copula.copula_model",
        "CopulaModel",
    ),
}


# The version of the serialised-dictionary layout, stamped into every
# ``to_dict`` output as ``"schema"``. Bump it only when a dictionary's
# shape changes incompatibly; the readers use it to recognise (and
# refuse, with a clear error) documents written by a newer SurPyval,
# and to migrate older layouts where needed. Documents with no
# ``"schema"`` key predate versioning and read as schema 0.
SCHEMA_VERSION = 1


def require_model_tag(model_dict: dict, tag: str, human: str) -> None:
    """Reject a ``from_dict`` dictionary whose ``model`` tag is not ``tag``.

    Every serialisable model's ``from_dict`` starts with this check;
    each used to write out the three-line guard itself. ``human`` is the
    phrase for the thing being built ("a renewal model", "an MCF
    estimate", ...); the raised message always contains ``tag`` so a
    caller can match on the model name.
    """
    if model_dict.get("model") != tag:
        article = "an" if tag[0] in "AEIOU" else "a"
        raise ValueError(
            "Must create {} from {} {} dict".format(human, article, tag)
        )


def stamp_schema(model_dict: dict) -> dict:
    """Stamp the serialisation schema version into a ``to_dict`` output."""
    model_dict["schema"] = SCHEMA_VERSION
    return model_dict


def check_schema(model_dict: dict) -> int:
    """Return the ``"schema"`` version of a serialised dictionary, refusing
    one this SurPyval cannot read.

    The version is an integer (``to_dict`` always writes one). Anything
    else is refused rather than guessed at: ``"2"`` and ``2.0`` used to
    slip past the version check, which only looked at ``int`` values, and
    were then read as if they were the current layout. A dictionary with
    no ``"schema"`` key predates versioning and reads as schema 0.
    """
    schema = model_dict.get("schema", 0)
    # ``bool`` is an ``int`` subclass, but ``True`` is not a version.
    if isinstance(schema, bool) or not isinstance(schema, numbers.Integral):
        raise ValueError(
            "The serialised model's 'schema' must be an integer version"
            f" number, got {schema!r}."
        )
    schema = int(schema)
    if schema < 0:
        raise ValueError(
            f"The serialised model's 'schema' version {schema} is invalid:"
            " versions are non-negative."
        )
    if schema > SCHEMA_VERSION:
        raise ValueError(
            f"This serialised model uses schema version {schema}, but "
            f"this version of SurPyval reads schema versions up to "
            f"{SCHEMA_VERSION}. Upgrade surpyval to load it."
        )
    return schema


def check_parameters(dist: Any, params: Any) -> None:
    """Refuse parameters a distribution cannot take, naming the culprit.

    A serialised dictionary is plain data, so a hand-edited or corrupted
    one could restore a model with, say, a negative Weibull scale, which
    then answered every query with NaN or nonsense. ``from_params`` has
    always refused such values; this applies the same bounds (the
    distribution's ``bounds``, ``None`` meaning unbounded) to restored
    parameters. A value *at* a bound is let through, since a fit may
    legitimately end there; NaN is always refused. Distributions without
    ``bounds`` are not checked.
    """
    values = np.atleast_1d(np.asarray(params, dtype=float))
    names = list(getattr(dist, "param_names", []) or [])
    if np.isnan(values).any():
        raise ValueError(
            f"The serialised parameters of '{getattr(dist, 'name', dist)}'"
            " contain NaN."
        )
    bounds = getattr(dist, "bounds", None)
    if bounds is None or values.ndim != 1 or len(bounds) != values.size:
        return
    for idx, ((low, high), value) in enumerate(zip(bounds, values)):
        if (low is not None and value < low) or (
            high is not None and value > high
        ):
            name = names[idx] if idx < len(names) else f"#{idx}"
            raise ValueError(
                f"The serialised parameter {name}={float(value)!r} of"
                f" '{getattr(dist, 'name', dist)}' is outside its bounds"
                f" {(low, high)}."
            )


def _check_restored_parametric(model: Any) -> None:
    """Bounds-check a restored univariate parametric model, including the
    limited-failure-population ``p`` and zero-inflation ``f0`` fractions,
    which are probabilities."""
    check_parameters(model.dist, model.params)
    for flag, attr in (("lfp", "p"), ("zi", "f0")):
        if getattr(model, flag, False):
            value = float(getattr(model, attr))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"The serialised '{attr}'={value!r} is a proportion and"
                    " must be in [0, 1]."
                )
    if getattr(model, "offset", False) and not np.isfinite(
        float(getattr(model, "gamma", 0.0))
    ):
        raise ValueError("The serialised offset 'gamma' must be finite.")


def read_model_dict(reader: Any, model_dict: dict) -> Any:
    """``reader.from_dict(model_dict)`` with the checks every reader needs.

    Shared by the package-level :func:`from_dict` and every class's
    ``from_json``, so a model read either way gets the same schema check,
    the same error for a missing entry and the same parameter check.
    """
    check_schema(model_dict)
    # The class readers index the dictionary directly, so a truncated or
    # hand-edited one surfaced as a bare ``KeyError: 'distribution'``
    # from deep inside a reader. Name the missing entry instead.
    try:
        model = reader.from_dict(model_dict)
    except KeyError as err:
        key = err.args[0] if err.args else None
        name = getattr(reader, "__name__", type(reader).__name__)
        raise ValueError(
            "The serialised model dictionary is incomplete or corrupt: it"
            f" has no {key!r} entry, which {name}.from_dict needs."
        ) from err
    # Only the core univariate parametric dictionary carries no "model"
    # tag; tagged models that also say "parametric" (mixtures, ...) keep
    # their parameters in other shapes.
    if (
        model_dict.get("parameterization") == "parametric"
        and "model" not in model_dict
    ):
        _check_restored_parametric(model)
    return model


def _resolve(module_name: str, class_name: str) -> Any:
    return getattr(import_module(module_name), class_name)


def to_native(value: Any) -> Any:
    """
    Convert numpy scalars and arrays (recursively, through lists and
    tuples) to native Python types.

    ``to_dict`` implementations use this so their dictionaries contain
    only native types: BSON encoders (e.g. MongoDB's) reject numpy
    scalars such as ``np.int64`` outright, unlike ``json.dumps``.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


class SerialisableMixin:
    """Shared ``to_json`` / ``from_json`` plumbing for serialisable
    models: every class keeps only its ``to_dict`` / ``from_dict``
    pair (this used to be copy-pasted into ~20 classes)."""

    if TYPE_CHECKING:
        # The contract every user of this mixin fulfils, declared for
        # the type checker but deliberately *not* defined. A stub
        # raising NotImplementedError would read better, but it would
        # also be inherited -- and ``copula_model`` decides whether a
        # margin is serialisable with ``hasattr(m, "to_dict")``, which
        # an inherited stub would answer True for every time.
        def to_dict(self) -> dict: ...

        @classmethod
        def from_dict(cls, model_dict: dict) -> Any: ...

    def to_json(self, fp: str | os.PathLike) -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> Any:
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            model_dict = json.load(f)
        if not isinstance(model_dict, dict):
            raise ValueError(
                "Expected a serialised model dict, got "
                f"{type(model_dict).__name__}"
            )
        return read_model_dict(cls, model_dict)


def from_dict(model_dict: dict) -> Any:
    """
    Restore any serialised SurPyval model from its dictionary.

    Reads the dictionary written by any fitted model's ``to_dict`` and
    dispatches to the right class's ``from_dict``, so the caller does
    not need to know which class wrote it.

    Parameters
    ----------
    model_dict : dict
        A dictionary produced by a SurPyval model's ``to_dict``.

    Returns
    -------
    The restored model, of whichever class serialised the dictionary.

    Raises
    ------
    ValueError
        If the dictionary is not recognisable as a serialised SurPyval
        model, lacks an entry its reader needs (the message names it), has
        a ``"schema"`` that is not a non-negative integer or was written
        by a newer SurPyval (a higher ``"schema"`` version), holds
        parameters outside the distribution's bounds (for a univariate
        parametric model), or names a distribution the reader does not
        know -- which includes a ``CustomDistribution`` that has not been
        constructed again in this session (a dictionary stores only its
        name, since its cumulative hazard is a Python function).

    Notes
    -----
    What a restored model keeps differs by family: in general the
    parameters and whatever predictions need, but not the fitted data,
    so methods that need the data (``plot``, ``bic``, bootstrap and
    likelihood-ratio bounds, residuals) raise on the restored model.
    See "Saving and Loading Models" in the Conventions page.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval import Weibull
    >>> model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0])
    >>> restored = surpyval.from_dict(model.to_dict())
    >>> restored.dist.name
    'Weibull'
    """
    if not isinstance(model_dict, dict):
        raise ValueError(
            "Expected a serialised model dict, got "
            f"{type(model_dict).__name__}"
        )

    # Before dispatch too: a model class added by a newer SurPyval is
    # unknown here, and "upgrade" is the useful answer for it.
    check_schema(model_dict)

    tag = model_dict.get("model")
    parameterization = model_dict.get("parameterization")
    if isinstance(tag, str) and tag in _TAGGED_MODELS:
        reader = _resolve(_TAGGED_MODELS[tag], tag)
    elif parameterization in _PARAMETERIZATIONS:
        reader = _resolve(*_PARAMETERIZATIONS[parameterization])
    else:
        reader = None

    if reader is not None:
        return read_model_dict(reader, model_dict)

    described = ", ".join(
        f"{k}={model_dict[k]!r}"
        for k in ("model", "parameterization")
        if k in model_dict
    )
    raise ValueError(
        "Not a recognisable serialised SurPyval model"
        + (f" ({described})" if described else "")
        + ": expected a 'model' class tag or a known 'parameterization'."
    )


def from_json(fp: str | Path) -> Any:
    """
    Restore any serialised SurPyval model from a JSON file.

    Reads a file written by any fitted model's ``to_json`` and
    dispatches to the right class's reader; see :func:`from_dict`.

    Parameters
    ----------
    fp : str | Path
        Path to a JSON file written by a SurPyval model's ``to_json``.

    Returns
    -------
    The restored model, of whichever class serialised the file.

    Examples
    --------
    >>> import os, tempfile
    >>> import surpyval
    >>> from surpyval import Weibull
    >>> model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0])
    >>> path = os.path.join(tempfile.mkdtemp(), "weibull.json")
    >>> model.to_json(path)
    >>> restored = surpyval.from_json(path)
    >>> restored.params
    array([5.53092634, 4.04187535])
    """
    with open(fp, "r") as f:
        return from_dict(json.load(f))
