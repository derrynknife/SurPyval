from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)

from .cox_ph import CoxPH
from .proportional_hazards_fitter import ProportionalHazardsFitter
from .proportional_odds_fitter import PO, ProportionalOddsFitter


def PH(distribution):
    """
    Create a Proportional Hazards fitter for the given distribution.

    Uses exp(beta'Z) as the hazard multiplier — the standard parameterisation
    for parametric PH models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``Exponential``).

    Returns
    -------
    ProportionalHazardsFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.regression import PH
    >>> model = PH(Weibull).fit(x, Z=covariates, c=c)
    """
    return ProportionalHazardsFitter.create(distribution)


# Pre-built PH instances — one per distribution
ExponentialPH = ProportionalHazardsFitter.create_general_log_linear_fitter(
    "ExponentialPH", Exponential
)
NormalPH    = ProportionalHazardsFitter.create_general_log_linear_fitter("NormalPH",    Normal)
WeibullPH   = ProportionalHazardsFitter.create_general_log_linear_fitter("WeibullPH",   Weibull)
GumbelPH    = ProportionalHazardsFitter.create_general_log_linear_fitter("GumbelPH",    Gumbel)
LogisticPH  = ProportionalHazardsFitter.create_general_log_linear_fitter("LogisticPH",  Logistic)
LogNormalPH = ProportionalHazardsFitter.create_general_log_linear_fitter("LogNormalPH", LogNormal)
GammaPH     = ProportionalHazardsFitter.create_general_log_linear_fitter("GammaPH",     Gamma)

# Pre-built PO instances — one per distribution
ExponentialPO = PO(Exponential)
NormalPO      = PO(Normal)
WeibullPO     = PO(Weibull)
GumbelPO      = PO(Gumbel)
LogisticPO    = PO(Logistic)
LogNormalPO   = PO(LogNormal)
GammaPO       = PO(Gamma)

__all__ = [
    "CoxPH",
    "ExponentialPH",
    "ExponentialPO",
    "GammaPH",
    "GammaPO",
    "GumbelPH",
    "GumbelPO",
    "LogisticPH",
    "LogisticPO",
    "LogNormalPH",
    "LogNormalPO",
    "NormalPH",
    "NormalPO",
    "PH",
    "PO",
    "ProportionalHazardsFitter",
    "ProportionalOddsFitter",
    "WeibullPH",
    "WeibullPO",
]
