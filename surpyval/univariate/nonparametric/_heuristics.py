"""The plotting heuristics' names, as ``how=`` reads them.

Shared by the non-parametric plotting positions and by the parametric
fits' probability plotting (``surpyval.univariate.parametric``), which
import them from here; the non-parametric package keeps its private
names for them (and the deprecated public ones) as aliases.
"""

from typing import Any

PLOTTING_METHODS = [
    "Blom",
    "Median",
    "ECDF",
    "ECDF_Adj",
    "Modal",
    "Midpoint",
    "Mean",
    "Weibull",
    "Benard",
    "Beard",
    "Hazen",
    "Gringorten",
    "None",
    "Tukey",
    "DPW",
    "Fleming-Harrington",
    "Kaplan-Meier",
    "Nelson-Aalen",
    "Filliben",
    "Larsen",
    "Turnbull",
]

#: Heuristic names, written in any case, that mean a listed one: Benard's
#: approximation is often spelled "Bernard".
HEURISTIC_ALIASES = {"bernard": "Benard"}


def canonical_heuristic(heuristic: Any) -> Any:
    """The plotting heuristic as ``PLOTTING_METHODS`` spells it, whatever
    its case, as ``how=`` is read (0.24 review); ``'Bernard'`` is
    ``'Benard'``. Anything that is not a known name is returned as given,
    for the check that refuses it to name it.

    Examples
    --------
    >>> from surpyval.univariate.nonparametric._heuristics import (
    ...     canonical_heuristic,
    ... )
    >>> canonical_heuristic("blom"), canonical_heuristic("BERNARD")
    ('Blom', 'Benard')
    """
    if not isinstance(heuristic, str):
        return heuristic
    key = heuristic.lower()
    if key in HEURISTIC_ALIASES:
        return HEURISTIC_ALIASES[key]
    for name in PLOTTING_METHODS:
        if name.lower() == key:
            return name
    return heuristic
