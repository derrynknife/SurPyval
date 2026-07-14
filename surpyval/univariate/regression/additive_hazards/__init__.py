from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)

from .additive_hazards import AdditiveHazards, AdditiveHazardsModel
from .additive_hazards_fitter import AdditiveHazardsFitter


def AH(distribution):
    """
    Create a parametric Additive Hazards fitter for the given distribution.

    Fits ``h(x | Z) = h_0(x) + beta'Z`` -- the covariate is a risk difference
    added to a parametric baseline hazard, rather than a multiplier as in the
    proportional hazards models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``Exponential``).

    Returns
    -------
    AdditiveHazardsFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> from surpyval import Weibull, AH
    >>> model = AH(Weibull).fit(x, Z=covariates, c=c)
    """
    return AdditiveHazardsFitter.create(distribution)


_create = AdditiveHazardsFitter.create

# Pre-built parametric additive hazards instances -- one per distribution.
ExponentialAH = _create(Exponential)
NormalAH = _create(Normal)
WeibullAH = _create(Weibull)
GumbelAH = _create(Gumbel)
LogisticAH = _create(Logistic)
LogNormalAH = _create(LogNormal)
GammaAH = _create(Gamma)

__all__ = [
    "AH",
    "AdditiveHazards",
    "AdditiveHazardsFitter",
    "AdditiveHazardsModel",
    "ExponentialAH",
    "GammaAH",
    "GumbelAH",
    "LogisticAH",
    "LogNormalAH",
    "NormalAH",
    "WeibullAH",
]
