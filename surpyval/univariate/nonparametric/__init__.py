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
