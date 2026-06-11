from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)

from .aft_fitter import AFT, AFTFitter

# Pre-built convenience instances — one per distribution
ExponentialAFT = AFT(Exponential)
NormalAFT = AFT(Normal)
WeibullAFT = AFT(Weibull)
GumbelAFT = AFT(Gumbel)
LogisticAFT = AFT(Logistic)
LogNormalAFT = AFT(LogNormal)
GammaAFT = AFT(Gamma)

__all__ = [
    "AFT",
    "AFTFitter",
    "ExponentialAFT",
    "GammaAFT",
    "GumbelAFT",
    "LogisticAFT",
    "LogNormalAFT",
    "NormalAFT",
    "WeibullAFT",
]
