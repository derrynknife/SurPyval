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


_cglf = ProportionalHazardsFitter.create_general_log_linear_fitter

# Pre-built PH instances — one per distribution
ExponentialPH = _cglf("ExponentialPH", Exponential)
NormalPH = _cglf("NormalPH", Normal)
WeibullPH = _cglf("WeibullPH", Weibull)
GumbelPH = _cglf("GumbelPH", Gumbel)
LogisticPH = _cglf("LogisticPH", Logistic)
LogNormalPH = _cglf("LogNormalPH", LogNormal)
GammaPH = _cglf("GammaPH", Gamma)

__all__ = [
    "CoxPH",
    "ExponentialPH",
    "GammaPH",
    "GumbelPH",
    "LogisticPH",
    "LogNormalPH",
    "NormalPH",
    "PH",
    "ProportionalHazardsFitter",
    "WeibullPH",
]
