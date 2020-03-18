import numpy as np

NUM = np.float64

PLOTTING_METHODS = [ "Blom", "Median", "ECDF", "Modal", "Midpoint", 
"Mean", "Weibull", "Benard", "Beard", "Hazen", "Gringorten", 
"None", "Tukey", "DPW", "Fleming-Harrington", "Kaplan-Meier",
"Nelson-Aalen", "Filliben"]

from .plotting_positions import plotting_positions
from .turnbull import turnbull
from .filliben import filliben
from .kaplan_meier import kaplan_meier
from .nelson_aalen import nelson_aalen
from .fleming_harrington import fleming_harrington
from .success_run import success_run
from .rank_adjust import rank_adjust

from .nonparametric import NonParametric

"""
Conventions for surpyval package
- c = censoring
- x = random variable (time, stress etc.)
- n = counts
- r = risk set
- d = deaths

- ff / F = Failure Function
- sf / R = Survival Function
- h = hazard rate
- H = Cumulative hazard function

- Censoring: -1 = left
              0 = failure / event
              1 = right
This is done to give an intuitive feel for when the 
event happened on the timeline.

- Censoring vectors. Pass repeat x's not censoring counts.
- Count doesn't assume unique x, it assumes repeats of censoring possible
"""




