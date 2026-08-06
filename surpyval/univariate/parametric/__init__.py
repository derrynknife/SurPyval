"""

Parametric Analysis
===================

.. code:: python

    import surpyval.parametric as para

    model = para.Weibull.fit(x)

"""

from .discrete_fitter import DiscreteParametricFitter
from .distributions import (
    Bernoulli,
    Beta,
    Beta4,
    BetaGeometric,
    Binomial,
    CustomDistribution,
    DiscreteWeibull,
    Discretize,
    DiscretizedFitter,
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
    InstantlyOccurs,
    Logistic,
    LogLogistic,
    LogNormal,
    NegativeBinomial,
    NeverOccurs,
    Normal,
    Poisson,
    Rayleigh,
    Uniform,
    Weibull,
)
from .mixture_model import MixtureModel
from .parametric import Parametric
from .parametric_fitter import OptimisedFitMixin, ParametricFitter
from .royston_parmar import RoystonParmar, RoystonParmarModel
