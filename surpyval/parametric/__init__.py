"""

Parametric Analysis
===================

.. code:: python

    import surpyval.parametric as para

    model = para.Weibull.fit(x)

"""
from surpyval import nonparametric as nonp

from .bernoulli import Bernoulli, FixedEventProbability
from .beta import Beta
from .custom_distribution import CustomDistribution
from .exact_event_time import ExactEventTime
from .expo_weibull import ExpoWeibull
from .exponential import Exponential
from .gamma import Gamma
from .gumbel import Gumbel
from .logistic import Logistic
from .loglogistic import LogLogistic
from .lognormal import Galton, LogNormal
from .mixture_model import MixtureModel
from .normal import Gauss, Normal
from .parametric import Parametric
from .parametric_fitter import ParametricFitter
from .rayleigh import Rayleigh
from .uniform import Uniform
from .weibull import Weibull
