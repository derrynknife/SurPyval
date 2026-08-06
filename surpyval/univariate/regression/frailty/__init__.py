from typing import Any

from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    LogNormal,
    Weibull,
)

from .frailty_fitter import FrailtyFitter
from .frailty_model import FrailtyModel


def Frailty(distribution: Any, family: str = "gamma") -> FrailtyFitter:
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
    >>> import numpy as np
    >>> from surpyval import Frailty, Weibull
    >>> np.random.seed(1)
    >>> unit_id = np.repeat(np.arange(20), 5)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = Frailty(Weibull).fit(x, Z=Z, groups=unit_id)
    >>> model.beta.round(3)
    array([0.829])
    >>> model.n_groups
    20
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
