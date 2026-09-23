from typing import Any

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


def PH(distribution: Any) -> ProportionalHazardsFitter:
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
    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval import PH
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = PH(Weibull).fit(x, Z=Z)
    >>> model.params.round(3)
    array([9.629, 1.751, 0.829])
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
