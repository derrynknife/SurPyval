from typing import Any, Callable

from .filliben import filliben
from .fleming_harrington import (
    FlemingHarrington,
    fleming_harrington,
    fleming_harrington_variance,
)
from .kaplan_meier import KaplanMeier, greenwood_variance, kaplan_meier
from .logrank import LogRankResult, logrank
from .nelson_aalen import NelsonAalen, nelson_aalen, nelson_aalen_variance
from .nonparametric import NonParametric, rmst_diff
from .plotting_positions import plotting_positions
from .rank_adjust import rank_adjust
from .success_run import success_run
from .turnbull import Turnbull, turnbull

FIT_FUNCS: dict[str, Callable[..., Any]] = {
    "Nelson-Aalen": nelson_aalen,
    "Kaplan-Meier": kaplan_meier,
    "Fleming-Harrington": fleming_harrington,
    "Turnbull": turnbull,
}

VAR_FUNCS: dict[str, Callable[..., Any]] = {
    "Nelson-Aalen": nelson_aalen_variance,
    "Kaplan-Meier": greenwood_variance,
    "Fleming-Harrington": fleming_harrington_variance,
}

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
_HEURISTIC_ALIASES = {"bernard": "Benard"}


def canonical_heuristic(heuristic: Any) -> Any:
    """The plotting heuristic as ``PLOTTING_METHODS`` spells it, whatever
    its case, as ``how=`` is read (0.24 review); ``'Bernard'`` is
    ``'Benard'``. Anything that is not a known name is returned as given,
    for the check that refuses it to name it.

    Examples
    --------
    >>> from surpyval.univariate.nonparametric import canonical_heuristic
    >>> canonical_heuristic("blom"), canonical_heuristic("BERNARD")
    ('Blom', 'Benard')
    """
    if not isinstance(heuristic, str):
        return heuristic
    key = heuristic.lower()
    if key in _HEURISTIC_ALIASES:
        return _HEURISTIC_ALIASES[key]
    for name in PLOTTING_METHODS:
        if name.lower() == key:
            return name
    return heuristic
