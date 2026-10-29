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
from .model import DistributionalRegressionModel

# Prebuilt fitters, one per common distribution
WeibullDistributional = Distributional(Weibull)
ExponentialDistributional = Distributional(Exponential)
GammaDistributional = Distributional(Gamma)
LogNormalDistributional = Distributional(LogNormal)
LogLogisticDistributional = Distributional(LogLogistic)
NormalDistributional = Distributional(Normal)
GumbelDistributional = Distributional(Gumbel)
LogisticDistributional = Distributional(Logistic)

__all__ = [
    "Distributional",
    "DistributionalFitter",
    "DistributionalRegressionModel",
    "ExponentialDistributional",
    "GammaDistributional",
    "GumbelDistributional",
    "LogisticDistributional",
    "LogLogisticDistributional",
    "LogNormalDistributional",
    "NormalDistributional",
    "WeibullDistributional",
]
