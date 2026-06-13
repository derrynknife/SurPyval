from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)

from .proportional_odds_fitter import PO, ProportionalOddsFitter

# Pre-built PO instances — one per distribution
ExponentialPO = PO(Exponential)
NormalPO = PO(Normal)
WeibullPO = PO(Weibull)
GumbelPO = PO(Gumbel)
LogisticPO = PO(Logistic)
LogNormalPO = PO(LogNormal)
GammaPO = PO(Gamma)

__all__ = [
    "ExponentialPO",
    "GammaPO",
    "GumbelPO",
    "LogisticPO",
    "LogNormalPO",
    "NormalPO",
    "PO",
    "ProportionalOddsFitter",
    "WeibullPO",
]
