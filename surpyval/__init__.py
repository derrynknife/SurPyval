"""

Surpyval
========

Survival analysis in python. The, at the time of writing, only survival analysis package that can be used with an arbitrary combination of observed, censored, and truncated data.
"""
import numpy as np

NUM     = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

import surpyval.datasets
import surpyval.utils

from surpyval.utils import (
	xcn_sort, 
	xcn_handler, 
	xcn_to_xrd, 
	xrd_to_xcn,
	xcnt_handler, 
	xcnt_to_xrd,
	fsli_to_xcn,
	fsl_to_xcn, 
	fs_to_xcn, 
	fs_to_xrd, 
	round_sig
)

from surpyval.parametric import (
	Gumbel,
	Uniform,
	Exponential,
	Weibull,
	ExpoWeibull,
	Normal, Gauss,
	LogNormal, Galton,
	Logistic,
	LogLogistic,
	Gamma,
	Beta,
	Distribution,
	MixtureModel
)

from surpyval.nonparametric import (
	KaplanMeier,
	NelsonAalen,
	FlemingHarrington,
	Turnbull
)


