from collections import defaultdict
from typing import Any, Callable, Hashable

import autograd.numpy as np

import surpyval as surv
from surpyval.regression.lifemodels.lifemodel import LifeModel
from surpyval.univariate.parametric import (
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    Weibull,
)
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter

from .accelerated_failure_time import AcceleratedFailureTimeFitter
from .cox_ph import CoxPH
from .forest.forest import RandomSurvivalForest
from .forest.tree import SurvivalTree
from .lifemodels import (
    DualExponential,
    DualPower,
    Exponential,
    Eyring,
    GeneralLogLinear,
    InverseExponential,
    InverseEyring,
    InversePower,
    Linear,
    Power,
    PowerExponential,
)
from .parameter_substitution import ParameterSubstitutionFitter
from .proportional_hazards import ProportionalHazardsFitter

# Useful for proportional odds
# https://data.princeton.edu/pop509/parametricsurvival.pdf

# Semi-Parametric Proportional Hazard
# CoxPH = CoxProportionalHazardsFitter()

DISTS: list[ParametricFitter] = [
    surv.Exponential,
    Normal,
    Weibull,
    Gumbel,
    Logistic,
    LogNormal,
]
LIFE_PARAMS = ["lambda", "mu", "alpha", "mu", "mu", "mu"]
LIFE_MODELS: list[LifeModel] = [
    Power,
    InversePower,
    Exponential,
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
life_parameter_inverse_transform: dict[
    Hashable, Callable | None
] = defaultdict(lambda: None)

life_parameter_transform["LogNormal"] = lambda x: np.log(x)
life_parameter_transform["Exponential"] = lambda x: 1.0 / x

life_parameter_inverse_transform["LogNormal"] = lambda x: np.exp(x)
life_parameter_inverse_transform["Exponential"] = lambda x: 1.0 / x

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

# I think the baseline feature should be removed
# I think the logic behind it was flawed from the start.
for dist, parameter in zip(DISTS, LIFE_PARAMS):
    name = dist.name + life_model.name + "AL"
    vars()[name] = ParameterSubstitutionFitter(
        "Accelerated Life",
        name,
        dist,
        GeneralLogLinear,
        parameter,
        baseline=[parameter],
        param_transform=life_parameter_transform[dist.name],
        inverse_param_transform=life_parameter_inverse_transform[dist.name],
    )

# Parametric Proportional Hazard
ExponentialPH = ProportionalHazardsFitter(
    "ExponentialPH",
    surv.Exponential,
    lambda Z, *params: np.exp(np.dot(Z, np.array(params))),
    lambda Z: (((None, None),) * Z.shape[1]),
    phi_param_map=lambda Z: {"beta_" + str(i): i for i in range(Z.shape[1])},
    baseline=[],
    phi_init=lambda Z: np.zeros(Z.shape[1]),
)

# Parametric Proportional Hazard
WeibullPH = ProportionalHazardsFitter(
    "WeibullPH",
    surv.Weibull,
    lambda Z, *params: np.exp(np.dot(Z, np.array(params))),
    lambda Z: (((None, None),) * Z.shape[1]),
    phi_param_map=lambda Z: {"beta_" + str(i): i for i in range(Z.shape[1])},
    baseline=[],
    phi_init=lambda Z: np.zeros(Z.shape[1]),
)


# Parametric AFT
WeibullInversePowerAFT = AcceleratedFailureTimeFitter(
    "WeibullInversePowerAFT", Weibull, InversePower
)
