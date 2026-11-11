"""The other names some distributions and life models go by.

``Gauss`` is another name for ``Normal`` and ``Galton`` another name for
``LogNormal``; in ``surpyval.life_models``, ``Arrhenius`` is another name
for ``Exponential``. A model fitted with one of these names carries it:
its ``repr`` shows it, and it saves and loads as that name. Everywhere the
package asks *which* distribution or life model a model has (registries,
comparisons, candidate lists), it asks by the canonical name, so a model
fitted with an alias behaves exactly as one fitted with the canonical name.

Examples
--------
>>> from surpyval.utils.aliases import canonical_name
>>> [canonical_name(n) for n in ("Gauss", "galton", "Weibull")]
['Normal', 'LogNormal', 'Weibull']
>>> import surpyval as sp
>>> sp.Gauss.name, sp.Gauss.canonical_name
('Gauss', 'Normal')
"""

from typing import Any

__all__ = [
    "DISTRIBUTION_ALIASES",
    "LIFE_MODEL_ALIASES",
    "canonical_name",
    "canonical_life_model_name",
]

#: Each distribution alias and the canonical name of its distribution.
DISTRIBUTION_ALIASES: "dict[str, str]" = {
    "Gauss": "Normal",
    "Galton": "LogNormal",
}

#: Each life-model alias and the canonical name of its life model.
LIFE_MODEL_ALIASES: "dict[str, str]" = {
    "Arrhenius": "Exponential",
}

_LOWER = {k.lower(): v for k, v in DISTRIBUTION_ALIASES.items()}


def canonical_name(dist: Any) -> Any:
    """The canonical name of a distribution, or of a distribution name.

    ``dist`` is a distribution or a name. A distribution gives its
    ``canonical_name`` where it has one (``Gauss`` gives ``"Normal"``),
    else its ``name`` (``None`` when it has none). A name that is an alias
    (``"Gauss"``, ``"Galton"``, matched without regard to case) gives the
    canonical name (``"Normal"``, ``"LogNormal"``), any other name itself.
    """
    if not isinstance(dist, str):
        canon = getattr(dist, "canonical_name", None)
        if isinstance(canon, str):
            return canon
        return getattr(dist, "name", None)
    return _LOWER.get(dist.lower(), dist)


def canonical_life_model_name(life_model: Any) -> Any:
    """The canonical name of a life model, or of a life-model name:
    ``"Arrhenius"`` gives ``"Exponential"``, any other name itself."""
    if not isinstance(life_model, str):
        canon = getattr(life_model, "canonical_name", None)
        if isinstance(canon, str):
            return canon
        return getattr(life_model, "name", None)
    return LIFE_MODEL_ALIASES.get(life_model, life_model)
