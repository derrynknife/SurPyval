
PLOTTING_METHODS = [ "Blom", "Median", "ECDF", "Modal", "Midpoint", 
"Mean", "Weibull", "Benard", "Beard", "Hazen", "Gringorten", 
"None", "Tukey", "DPW", "Fleming-Harrington", "Kaplan-Meier",
"Nelson-Aalen", "Filliben", "Larsen", "Turnbull"]

from .plotting_positions import plotting_positions
from .turnbull import turnbull
from .filliben import filliben
from .kaplan_meier import kaplan_meier
from .nelson_aalen import nelson_aalen
from .fleming_harrington import fleming_harrington
from .success_run import success_run
from .rank_adjust import rank_adjust

from .nonparametric import NonParametric
from .kaplan_meier import KaplanMeier
from .nelson_aalen import NelsonAalen
from .fleming_harrington import FlemingHarrington
from .turnbull import Turnbull

FIT_FUNCS = {
	'Nelson-Aalen'       : nelson_aalen,
	'Kaplan-Meier'       : kaplan_meier,
	'Fleming-Harrington' : fleming_harrington,
	'Turnbull'           : turnbull
}



