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
            return cls.from_dict(json.load(f))


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
        model.

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

    schema = model_dict.get("schema", 0)
    if isinstance(schema, int) and schema > SCHEMA_VERSION:
        raise ValueError(
            f"This serialised model uses schema version {schema}, but "
            f"this version of SurPyval reads schema versions up to "
            f"{SCHEMA_VERSION}. Upgrade surpyval to load it."
        )

    tag = model_dict.get("model")
    if isinstance(tag, str) and tag in _TAGGED_MODELS:
        return _resolve(_TAGGED_MODELS[tag], tag).from_dict(model_dict)

    parameterization = model_dict.get("parameterization")
    if parameterization in _PARAMETERIZATIONS:
        return _resolve(*_PARAMETERIZATIONS[parameterization]).from_dict(
            model_dict
        )

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
    """
    with open(fp, "r") as f:
        return from_dict(json.load(f))
