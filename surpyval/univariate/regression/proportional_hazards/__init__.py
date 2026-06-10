from surpyval.univariate.parametric import Exponential, Weibull

from .cox_ph import CoxPH
from .proportional_hazards_fitter import ProportionalHazardsFitter

# Parametric Proportional Hazard
ExponentialPH = ProportionalHazardsFitter.create_general_log_linear_fitter(
    "ExponentialPH", Exponential
)

# Parametric Proportional Hazard
WeibullPH = ProportionalHazardsFitter.create_general_log_linear_fitter(
    "WeibullPH", Weibull
)

__all__ = [
    "CoxPH",
    "ExponentialPH",
    "ProportionalHazardsFitter",
    "WeibullPH",
]
