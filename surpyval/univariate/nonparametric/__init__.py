from collections.abc import Callable
from typing import Any

from .at_risk import at_risk_table
from .filliben import filliben
from .fleming_harrington import (
    FlemingHarrington,
)
from .fleming_harrington import fleming_harrington as _fleming_harrington
from .fleming_harrington import (
    fleming_harrington_variance as _fleming_harrington_variance,
)
from .interval_logrank import IntervalLogRankResult, interval_logrank
from .kaplan_meier import KaplanMeier
from .kaplan_meier import greenwood_variance as _greenwood_variance
from .kaplan_meier import kaplan_meier as _kaplan_meier
from .life_table import life_table
from .logrank import (
    LogRankResult,
    PairwiseLogRankResult,
    logrank,
    pairwise_logrank,
)
from .logrank_maxcombo import MaxComboResult, maxcombo_logrank
from .logrank_permutation import PermutationLogRankResult, permutation_logrank
from .logrank_trend import TrendLogRankResult, trend_logrank
from .maxstat import MaxStatResult, maxstat_logrank
from .nelson_aalen import NelsonAalen
from .nelson_aalen import nelson_aalen as _nelson_aalen
from .nelson_aalen import nelson_aalen_variance as _nelson_aalen_variance
from .nonparametric import NonParametric, rmst_diff
from .person_years import PersonYears, person_years, poisson_rate_bounds
from .plot_estimates import NonParametricPlot, plot_nonparametric
from .plotting_positions import plotting_positions
from .rank_adjust import rank_adjust as _rank_adjust
from .success_run import success_run
from .survival_difference import sf_diff
from .turnbull import Turnbull
from .turnbull import turnbull as _turnbull

_FIT_FUNCS: dict[str, Callable[..., Any]] = {
    "Nelson-Aalen": _nelson_aalen,
    "Kaplan-Meier": _kaplan_meier,
    "Fleming-Harrington": _fleming_harrington,
    "Turnbull": _turnbull,
}

_VAR_FUNCS: dict[str, Callable[..., Any]] = {
    "Nelson-Aalen": _nelson_aalen_variance,
    "Kaplan-Meier": _greenwood_variance,
    "Fleming-Harrington": _fleming_harrington_variance,
}

_PLOTTING_METHODS = [
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


def _canonical_heuristic(heuristic: Any) -> Any:
    """The plotting heuristic as ``_PLOTTING_METHODS`` spells it, whatever
    its case, as ``how=`` is read (0.24 review); ``'Bernard'`` is
    ``'Benard'``. Anything that is not a known name is returned as given,
    for the check that refuses it to name it.

    Examples
    --------
    >>> from surpyval.univariate.nonparametric import _canonical_heuristic
    >>> _canonical_heuristic("blom"), _canonical_heuristic("BERNARD")
    ('Blom', 'Benard')
    """
    if not isinstance(heuristic, str):
        return heuristic
    key = heuristic.lower()
    if key in _HEURISTIC_ALIASES:
        return _HEURISTIC_ALIASES[key]
    for name in _PLOTTING_METHODS:
        if name.lower() == key:
            return name
    return heuristic


# The submodules of the same names as the functions above (``kaplan_meier``
# and the rest): the package's attribute is the deprecated function, not
# the module (which ``import`` reached before, as the function shadowed it).
for _name in (
    "fleming_harrington",
    "kaplan_meier",
    "nelson_aalen",
    "rank_adjust",
    "turnbull",
):
    globals().pop(_name, None)
del _name

#: The names public until v0.29 that are the fitters' internal steps, each
#: with what to use instead (reachable here, with a FutureWarning, until
#: v0.30).
_INTERNAL = {
    "kaplan_meier": ("_kaplan_meier", "KaplanMeier.fit(x, c, n, t)"),
    "nelson_aalen": ("_nelson_aalen", "NelsonAalen.fit(x, c, n, t)"),
    "fleming_harrington": (
        "_fleming_harrington",
        "FlemingHarrington.fit(x, c, n, t)",
    ),
    "turnbull": ("_turnbull", "Turnbull.fit(x, c, n, t)"),
    "greenwood_variance": (
        "_greenwood_variance",
        "KaplanMeier.fit(...).cb(...), whose Greenwood bounds use it",
    ),
    "nelson_aalen_variance": (
        "_nelson_aalen_variance",
        "NelsonAalen.fit(...).cb(...)",
    ),
    "fleming_harrington_variance": (
        "_fleming_harrington_variance",
        "FlemingHarrington.fit(...).cb(...)",
    ),
    "rank_adjust": (
        "_rank_adjust",
        "plotting_positions(x, c, n, how=...), which adjusts the ranks",
    ),
    "canonical_heuristic": (
        "_canonical_heuristic",
        "plotting_positions(..., how=...), which reads the name in any case",
    ),
    "FIT_FUNCS": (
        "_FIT_FUNCS",
        "the fitters KaplanMeier, NelsonAalen, FlemingHarrington and "
        "Turnbull",
    ),
    "VAR_FUNCS": ("_VAR_FUNCS", "the fitters' cb()"),
    "PLOTTING_METHODS": (
        "_PLOTTING_METHODS",
        "the how= names listed by plotting_positions' documentation",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _INTERNAL:
        from surpyval.utils.deprecation import REMOVED_IN_NEXT, renamed_name

        private, new = _INTERNAL[name]
        return renamed_name(
            "surpyval.univariate.nonparametric",
            name,
            new,
            globals()[private],
            REMOVED_IN_NEXT,
            FutureWarning,
        )
    raise AttributeError(
        f"module 'surpyval.univariate.nonparametric' has no attribute "
        f"{name!r}"
    )
