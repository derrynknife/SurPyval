from typing import Any

from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    LogNormal,
    Weibull,
)

from .cox_frailty import CoxFrailty, CoxFrailtyFitter, CoxFrailtyModel
from .frailty_fitter import FrailtyFitter
from .frailty_model import FrailtyModel


def Frailty(distribution: Any, family: str = "gamma") -> FrailtyFitter:
    """
    Create a shared-frailty proportional-hazards fitter for a distribution.

    A shared-frailty model adds a random hazard multiplier shared within a
    group (a lot, site, or repairable unit) on top of a proportional-hazards
    baseline, capturing unobserved between-group heterogeneity and the
    within-group correlation it induces. The frailty is Gamma by default,
    whose closed-form marginal likelihood keeps the fit fast, or log-normal
    (``family="lognormal"``), integrated out by adaptive Gauss-Hermite
    quadrature.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``Exponential``).
    family : str, optional
        The frailty distribution: ``"gamma"`` (the default), mean 1 and
        variance ``theta``; or ``"lognormal"``, ``u = exp(w)`` with ``w``
        normal of mean 0 and variance ``theta``, the parameterisation of
        R's ``frailtypack`` (``RandDist = "LogN"``) and ``coxme``.
        ``frailty_variance`` (``Var(u) / E(u)^2``) and ``kendall_tau``
        compare the two; so does ``aic()``.

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

    A log-normal frailty, on the kidney catheter data (two infection times
    per patient):

    >>> from surpyval.datasets import load_kidney
    >>> df = load_kidney()
    >>> Z = np.column_stack([df["age"], df["sex"] == 2])
    >>> lognormal = Frailty(Weibull, family="lognormal").fit(
    ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
    ... )
    >>> round(lognormal.theta, 3), round(lognormal.kendall_tau, 3)
    (0.593, 0.196)
    """
    return FrailtyFitter.create(distribution, family)


# Pre-built gamma-frailty instances -- one per distribution.
ExponentialFrailty = FrailtyFitter.create(Exponential)
WeibullFrailty = FrailtyFitter.create(Weibull)
LogNormalFrailty = FrailtyFitter.create(LogNormal)
GammaFrailty = FrailtyFitter.create(Gamma)

__all__ = [
    "CoxFrailty",
    "CoxFrailtyFitter",
    "CoxFrailtyModel",
    "ExponentialFrailty",
    "Frailty",
    "FrailtyFitter",
    "FrailtyModel",
    "GammaFrailty",
    "LogNormalFrailty",
    "WeibullFrailty",
]
