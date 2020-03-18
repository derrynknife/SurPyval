"""
.. module:: surpyval.parametric
    :synopsis: module for parametric survival analysis
 
.. moduleauthor:: Derryn Knife <derryn@reliafy.com>

Conventions for surpyval package
- c = censoring
- x = random variable (time, stress etc.)
- n = counts
- r = risk set
- d = deaths

usual format for data:
xcn = x variables, with c as the censoring schemed and n as the counts
xrd = x variables, with the risk set, r,  at x and the deaths, d, also at x

wranglers for formats:
fs = failure times, f, and right censored times, s
fsl = fs format plus a vector for left censored times

- df = Density Function
- ff / F = Failure Function
- sf / R = Survival Function
- h = hazard rate
- H = Cumulative hazard function

- Censoring: -1 = left
              0 = failure / event
              1 = right
              2 = interval censoring. Must have left and right coord
This is done to give an intuitive feel for when the 
event happened on the timeline.
"""

import autograd.numpy as np

from surpyval import nonparametric as nonp
from .surpyval_dist import SurpyvalDist
from .parametric_dist import Parametric

from .weibull import Weibull
from .weibull3p import Weibull3p
from .gumbel import Gumbel
from .exponential import Exponential
from .normal import Normal
from .lognormal import LogNormal
from .gamma import Gamma
from .uniform import Uniform
from .weibull_mix2 import Weibull_Mix_Two
from .wmm import WMM
from .logistic import Logistic
from .loglogistic import LogLogistic

from .mixture_model import MixtureModel
from .lfp import LFP

