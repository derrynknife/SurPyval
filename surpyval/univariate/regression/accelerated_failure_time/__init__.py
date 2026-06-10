from surpyval.univariate.parametric import Weibull

from ..accelerated_life import InversePower
from .accelerated_failure_time import AcceleratedFailureTimeFitter

# Parametric AFT
WeibullInversePowerAFT = AcceleratedFailureTimeFitter(
    "WeibullInversePowerAFT", Weibull, InversePower
)

__all__ = [
    "AcceleratedFailureTimeFitter",
    "WeibullInversePowerAFT",
]
