"""

Parametric Analysis
===================

.. code:: python

    import surpyval.parametric as para

    model = para.Weibull.fit(x)

"""

import numpy as np

from surpyval.distribution import Distribution

from .distributions import (
    Bernoulli,
    Beta,
    Beta4,
    Binomial,
    CustomDistribution,
    DiscreteWeibull,
    ExactEventTime,
    ExpoWeibull,
    Exponential,
    FixedEventProbability,
    Galton,
    Gamma,
    Gauss,
    Geometric,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    NegativeBinomial,
    Normal,
    Rayleigh,
    Uniform,
    Weibull,
)
from .mixture_model import MixtureModel
from .parametric import Parametric
from .parametric_fitter import ParametricFitter


class NeverOccurs(Distribution):
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


class InstantlyOccurs(Distribution):
    @classmethod
    def sf(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def Hf(cls, x):
        return np.full_like(x, np.inf, dtype=float)

    @classmethod
    def random(cls, size):
        return np.zeros(size)
