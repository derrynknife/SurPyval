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
from importlib import import_module
from pathlib import Path
from typing import Any

# ``"model"`` tag -> defining module. The tag every class writes into
# its dict is its own class name, so the registry only needs to find
# the defining module, lazily (importing everything eagerly here would
# be a heavy import and a cycle risk).
_TAGGED_MODELS: dict[str, str] = {
    "SemiParametricRegressionModel": (
        "surpyval.univariate.regression.semi_parametric_regression_model"
    ),
    "AdditiveHazardsModel": (
        "surpyval.univariate.regression.additive_hazards.additive_hazards"
    ),
    "BuckleyJamesModel": (
        "surpyval.univariate.regression.buckley_james.buckley_james"
    ),
    "MixtureModel": "surpyval.univariate.parametric.mixture_model",
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
}


def _resolve(module_name: str, class_name: str) -> Any:
    return getattr(import_module(module_name), class_name)


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
