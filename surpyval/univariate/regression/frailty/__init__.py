from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    LogNormal,
    Weibull,
)

from .frailty_fitter import FrailtyFitter
from .frailty_model import FrailtyModel


def Frailty(distribution, family="gamma"):
    """
    Create a shared-frailty proportional-hazards fitter for a distribution.

    A shared-frailty model adds a random hazard multiplier shared within a
    group (a lot, site, or repairable unit) on top of a proportional-hazards
    baseline, capturing unobserved between-group heterogeneity and the
    within-group correlation it induces. The frailty family is Gamma (the only
    one currently available), whose closed-form marginal likelihood keeps the
    fit fast.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``Exponential``).
    family : str, optional
        The frailty distribution. Only ``"gamma"`` is currently supported.

    Returns
    -------
    FrailtyFitter
        A fitter with ``.fit(x, Z, c, groups=...)`` and ``.fit_from_df``.

    Examples
    --------
    >>> from surpyval import Frailty, Weibull
    >>> model = Frailty(Weibull).fit(x, Z=Z, c=c, groups=unit_id)
    """
    return FrailtyFitter.create(distribution, family)


# Pre-built gamma-frailty instances -- one per distribution.
ExponentialFrailty = FrailtyFitter.create(Exponential)
WeibullFrailty = FrailtyFitter.create(Weibull)
LogNormalFrailty = FrailtyFitter.create(LogNormal)
GammaFrailty = FrailtyFitter.create(Gamma)

__all__ = [
    "ExponentialFrailty",
    "Frailty",
    "FrailtyFitter",
    "FrailtyModel",
    "GammaFrailty",
    "LogNormalFrailty",
    "WeibullFrailty",
]
