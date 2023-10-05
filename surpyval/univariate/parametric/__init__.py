"""

Parametric Analysis
===================

.. code:: python

    import surpyval.parametric as para

    model = para.Weibull.fit(x)

"""
import numpy as np

from .bernoulli import Bernoulli, FixedEventProbability
from .beta import Beta
from .custom_distribution import CustomDistribution
from .exact_event_time import ExactEventTime
from .expo_weibull import ExpoWeibull
from .exponential import Exponential
from .gamma import Gamma
from .gumbel import Gumbel
from .gumbel_lev import GumbelLEV
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

# from surpyval.univariate import nonparametric as nonp


class NeverOccurs:
    @classmethod
    def sf(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def Hf(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def random(cls, size):
        return np.ones(size) * np.inf


class InstantlyOccurs:
    @classmethod
    def sf(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def Hf(cls, x):
        return np.zeros_like(x).astype(float) * np.inf

    @classmethod
    def random(self, size):
        return np.zeros(size)
