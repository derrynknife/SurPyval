"""

Surpyval Again
===============

More stuff about the surpyval package

"""

import numpy as np
import surpyval.datasets

from surpyval.utils import xcn_sort, xcn_handler, xcn_to_xrd, xrd_to_xcn
from surpyval.utils import fsl_to_xcn, fs_to_xcn, fs_to_xrd, round_sig

import surpyval.parametric
import surpyval.nonparametric

from surpyval.parametric import Gumbel
from surpyval.parametric import Uniform
from surpyval.parametric import Exponential
from surpyval.parametric import Weibull
from surpyval.parametric import ExpoWeibull
from surpyval.parametric import Weibull3p
from surpyval.parametric import Normal
from surpyval.parametric import LogNormal
from surpyval.parametric import Logistic
from surpyval.parametric import LogLogistic
from surpyval.parametric import Gamma

from surpyval.parametric import LFP
from surpyval.parametric import MixtureModel
from surpyval.parametric import Weibull_Mix_Two

from surpyval.nonparametric import KaplanMeier
from surpyval.nonparametric import NelsonAalen
from surpyval.nonparametric import FlemingHarrington
from surpyval.nonparametric import Turnbull

NUM     = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

