from surpyval.univariate.parametric import Exponential, Weibull

from .cox_ph import CoxPH
from .proportional_hazards_fitter import ProportionalHazardsFitter
from .proportional_odds_fitter import PO, ProportionalOddsFitter

# Convenience factory function mirroring AFT and PO
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


# Pre-built convenience instances kept for backward compatibility
ExponentialPH = ProportionalHazardsFitter.create_general_log_linear_fitter(
    "ExponentialPH", Exponential
)
WeibullPH = ProportionalHazardsFitter.create_general_log_linear_fitter(
    "WeibullPH", Weibull
)

__all__ = [
    "CoxPH",
    "ExponentialPH",
    "PH",
    "PO",
    "ProportionalHazardsFitter",
    "ProportionalOddsFitter",
    "WeibullPH",
]
