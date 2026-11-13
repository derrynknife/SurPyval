"""Distributional regression: covariates on every parameter of a
distribution, each through its own link (see :mod:`.fitter`)."""

from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Weibull,
)

from .fitter import Distributional, DistributionalFitter
from .mixture_cure import MixtureCure, MixtureCureFitter, MixtureCureModel
from .model import DistributionalRegressionModel
from .zero_inflated import (
    ZeroInflated,
    ZeroInflatedFitter,
    ZeroInflatedModel,
)

# Prebuilt fitters, one per common distribution
WeibullDistributional = Distributional(Weibull)
ExponentialDistributional = Distributional(Exponential)
GammaDistributional = Distributional(Gamma)
LogNormalDistributional = Distributional(LogNormal)
LogLogisticDistributional = Distributional(LogLogistic)
NormalDistributional = Distributional(Normal)
GumbelDistributional = Distributional(Gumbel)
LogisticDistributional = Distributional(Logistic)

# Mixture-cure regression, one fitter per common latency distribution
WeibullMixtureCure = MixtureCure(Weibull)
LogNormalMixtureCure = MixtureCure(LogNormal)
LogLogisticMixtureCure = MixtureCure(LogLogistic)
GammaMixtureCure = MixtureCure(Gamma)
ExponentialMixtureCure = MixtureCure(Exponential)

# Zero-inflated regression, one fitter per common latency distribution
WeibullZeroInflated = ZeroInflated(Weibull)
LogNormalZeroInflated = ZeroInflated(LogNormal)
LogLogisticZeroInflated = ZeroInflated(LogLogistic)
GammaZeroInflated = ZeroInflated(Gamma)
ExponentialZeroInflated = ZeroInflated(Exponential)

__all__ = [
    "Distributional",
    "DistributionalFitter",
    "DistributionalRegressionModel",
    "ExponentialDistributional",
    "ExponentialMixtureCure",
    "GammaMixtureCure",
    "LogLogisticMixtureCure",
    "LogNormalMixtureCure",
    "MixtureCure",
    "MixtureCureFitter",
    "MixtureCureModel",
    "WeibullMixtureCure",
    "GammaDistributional",
    "GumbelDistributional",
    "LogisticDistributional",
    "LogLogisticDistributional",
    "LogNormalDistributional",
    "NormalDistributional",
    "WeibullDistributional",
    "ExponentialZeroInflated",
    "GammaZeroInflated",
    "LogLogisticZeroInflated",
    "LogNormalZeroInflated",
    "WeibullZeroInflated",
    "ZeroInflated",
    "ZeroInflatedFitter",
    "ZeroInflatedModel",
]
