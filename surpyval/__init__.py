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

from surpyval.utils import xcn_sort, xcn_handler, xcn_to_xrd, xrd_to_xcn
from surpyval.utils import xcnt_handler, xcnt_to_xrd
from surpyval.utils import fsl_to_xcn, fs_to_xcn, fs_to_xrd, round_sig

import surpyval.parametric
import surpyval.nonparametric

from surpyval.parametric import Gumbel
from surpyval.parametric import Uniform
from surpyval.parametric import Exponential
from surpyval.parametric import Weibull
from surpyval.parametric import ExpoWeibull
from surpyval.parametric import Normal, Gauss
from surpyval.parametric import LogNormal, Galton
from surpyval.parametric import Logistic
from surpyval.parametric import LogLogistic
from surpyval.parametric import Gamma
from surpyval.parametric import Beta

from surpyval.parametric import LFP
from surpyval.parametric import MixtureModel

from surpyval.nonparametric import KaplanMeier
from surpyval.nonparametric import NelsonAalen
from surpyval.nonparametric import FlemingHarrington
from surpyval.nonparametric import Turnbull



