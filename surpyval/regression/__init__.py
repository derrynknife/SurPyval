import autograd.numpy as np
from collections import defaultdict

from .cox_ph import CoxProportionalHazardsFitter

from .proportional_hazards import ProportionalHazardsFitter
from .accelerated_failure_time import AcceleratedFailureTimeFitter
from .parameter_substitution import ParameterSubstitutionFitter

import surpyval as surv
from .lifemodels.power import InversePower, Power
from .lifemodels.exponential import InverseExponential, Exponential
from .lifemodels.linear import Linear
from .lifemodels.eyring import Eyring, InverseEyring
from .lifemodels.dual_exponential import DualExponential
from .lifemodels.dual_power import DualPower
from .lifemodels.power_exponential import PowerExponential
from .lifemodels.general_log_linear import GeneralLogLinear

from ..parametric import (
    LogNormal,
    Normal,
    Weibull,
    Gumbel,
    Logistic
)

# Semi-Parametric Proportional Hazard
CoxPH = CoxProportionalHazardsFitter()

DISTS = [surv.Exponential, Normal, Weibull, Gumbel, Logistic, LogNormal]
LIFE_PARAMS = ['lambda', 'mu', 'alpha', 'mu', 'mu', 'mu']
LIFE_MODELS = [
    Power, 
    InversePower,
    Exponential,
    InverseExponential,
    Eyring,
    InverseEyring,
    Linear,
    DualExponential,
    DualPower,
    PowerExponential
]

life_parameter_transform = defaultdict(lambda: None)
life_parameter_inverse_transform = defaultdict(lambda: None)
baseline_parameters = defaultdict(lambda: [])

life_parameter_transform['LogNormal'] = lambda x: np.log(x)
life_parameter_transform['Exponential'] = lambda x: 1./x

life_parameter_inverse_transform['LogNormal'] = lambda x: np.exp(x)
life_parameter_inverse_transform['Exponential'] = lambda x: 1./x

# Quite remarkable - creates every life model and distribution class!
for dist, parameter in zip(DISTS, LIFE_PARAMS):
    for life_model in LIFE_MODELS:
        name = dist.name + life_model.name + "AL"
        vars()[name] = ParameterSubstitutionFitter(
                            'Accelerated Life', 
                            name, dist, life_model, 
                            parameter,
                            param_transform=life_parameter_transform[dist.name],
                            inverse_param_transform=life_parameter_inverse_transform[dist.name]
                        )

for dist, parameter in zip(DISTS, LIFE_PARAMS):
    name = dist.name + life_model.name + "AL"
    vars()[name] = ParameterSubstitutionFitter(
                        'Accelerated Life', 
                        name,
                        dist,
                        GeneralLogLinear,
                        parameter,
                        baseline=[parameter],
                        param_transform=life_parameter_transform[dist.name],
                        inverse_param_transform=life_parameter_inverse_transform[dist.name]
                    )

# Parametric Proportional Hazard
ExponentialPH = ProportionalHazardsFitter(
    'ExponentialPH', 
    surv.Exponential,
    lambda X, *params: np.exp(np.dot(X, np.array(params))),
    lambda X: (((None, None),) * X.shape[1]),
    phi_param_map=lambda X: {'beta_' + str(i) : i for i in range(X.shape[1])},
    baseline=['lambda'],
    phi_init=lambda X: np.zeros(X.shape[1])
)


# Parametric AFT
WeibullInversePowerAFT = AcceleratedFailureTimeFitter(
    'WeibullInversePowerAFT', 
    surv.Weibull,
    InversePower
)
