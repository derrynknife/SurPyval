from collections import defaultdict
from collections.abc import Callable, Hashable

import autograd.numpy as np

from surpyval.univariate.parametric import (
    Exponential,
    Gamma,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter

from .dual_exponential import DualExponential
from .dual_power import DualPower
from .exponential import ExponentialLifeModel, InverseExponential
from .eyring import Eyring, InverseEyring
from .general_log_linear import GeneralLogLinear
from .lifemodel import LifeModel
from .linear import Linear
from .parameter_substitution import ParameterSubstitutionFitter
from .power import InversePower, Power
from .power_exponential import PowerExponential

DISTS: list[ParametricFitter] = [
    Exponential,
    Normal,
    Weibull,
    Gumbel,
    Logistic,
    LogNormal,
    Gamma,
]
LIFE_PARAMS = ["lambda", "mu", "alpha", "mu", "mu", "mu", "beta"]
LIFE_MODELS: list[LifeModel] = [
    Power,
    InversePower,
    ExponentialLifeModel,
    InverseExponential,
    Eyring,
    InverseEyring,
    Linear,
    DualExponential,
    DualPower,
    PowerExponential,
]

life_parameter_transform: dict[Hashable, Callable | None] = defaultdict(
    lambda: None
)
life_parameter_inverse_transform: dict[Hashable, Callable | None] = (
    defaultdict(lambda: None)
)

life_parameter_transform["LogNormal"] = lambda x: np.log(x)
life_parameter_transform["Exponential"] = lambda x: 1.0 / x

life_parameter_inverse_transform["LogNormal"] = lambda x: np.exp(x)
life_parameter_inverse_transform["Exponential"] = lambda x: 1.0 / x

__all__ = [
    "DualExponential",
    "DualPower",
    "ExponentialLifeModel",
    "Eyring",
    "GeneralLogLinear",
    "InverseExponential",
    "InverseEyring",
    "InversePower",
    "LifeModel",
    "Linear",
    "ParameterSubstitutionFitter",
    "Power",
    "PowerExponential",
]

# Quite remarkable - creates every life model and distribution class!
for dist, parameter in zip(DISTS, LIFE_PARAMS):
    for life_model in LIFE_MODELS:
        name = dist.name + life_model.name + "AL"
        vars()[name] = ParameterSubstitutionFitter(
            "Accelerated Life",
            name,
            dist,
            life_model,
            parameter,
            param_transform=life_parameter_transform[dist.name],
            inverse_param_transform=life_parameter_inverse_transform[
                dist.name
            ],
        )
        __all__.append(name)

# I think the baseline feature should be removed
# I think the logic behind it was flawed from the start.
# for dist, parameter in zip(DISTS, LIFE_PARAMS):
#     name = dist.name + "GeneralLogLinearAL"
#     vars()[name] = ParameterSubstitutionFitter(
#         "Accelerated Life",
#         name,
#         dist,
#         GeneralLogLinear,
#         parameter,
#         baseline=[parameter],
#         param_transform=life_parameter_transform[dist.name],
#         inverse_param_transform=life_parameter_inverse_transform[dist.name],
#     )
