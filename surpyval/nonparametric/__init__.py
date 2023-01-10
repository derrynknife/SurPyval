from .filliben import filliben
from .fleming_harrington import FlemingHarrington, fleming_harrington
from .kaplan_meier import KaplanMeier, kaplan_meier
from .nelson_aalen import NelsonAalen, nelson_aalen
from .nonparametric import NonParametric
from .plotting_positions import plotting_positions
from .rank_adjust import rank_adjust
from .success_run import success_run
from .turnbull import Turnbull, turnbull

FIT_FUNCS = {
    "Nelson-Aalen": nelson_aalen,
    "Kaplan-Meier": kaplan_meier,
    "Fleming-Harrington": fleming_harrington,
    "Turnbull": turnbull,
}

PLOTTING_METHODS = [
    "Blom",
    "Median",
    "ECDF",
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
