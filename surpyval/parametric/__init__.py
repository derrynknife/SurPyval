"""

Parametric Analysis
===================

.. code:: python

	import surpyval.parametric as para

	model = para.Weibull.fit(x)

"""

import autograd.numpy as np

from surpyval import nonparametric as nonp
from .parametric_fitter import ParametricFitter
from .parametric import Parametric

from .weibull import Weibull
from .weibull3p import Weibull3p
from .gumbel import Gumbel
from .exponential import Exponential
from .expo_weibull import ExpoWeibull
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

