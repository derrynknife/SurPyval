import autograd.numpy as np
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

from ..parametric import (
    LogNormal,
    Normal,
    Weibull,
    Gumbel,
    Logistic
)

# Semi-Parametric Proportional Hazard
CoxPH = CoxProportionalHazardsFitter()

DISTS = [surv.Exponential, LogNormal, Normal, 
    Weibull, Gumbel, Logistic]
LIFE_PARAMS = ['lambda', 'mu', 'mu', 'alpha', 'mu', 'mu']

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

# Quite remarkable - creates every life model and distribution class!
for dist, parameter in zip(DISTS, LIFE_PARAMS):
    for life_model in LIFE_MODELS:
        name = dist.name + life_model.name + "AL"
        vars()[name] = ParameterSubstitutionFitter(
                            'Accelerated Life', 
                            name, dist, life_model, 
                            parameter
                        )

# Parametric Proportional Hazard
def phi_param_map(X):
    base = 'beta_'
    return {base + str(i) : i for i in range(X.shape[1])}

ExponentialPH = ProportionalHazardsFitter(
    'ExponentialPH', 
    surv.Exponential,
    lambda X, *params: np.exp(np.dot(X, np.array(params))),
    lambda X: (((None, None),) * X.shape[1]),
    phi_param_map=phi_param_map,
    baseline=['lambda'],
    phi_init=lambda X: np.zeros(X.shape[1])
    )

WeibullPowerAFT = AcceleratedFailureTimeFitter(
    'WeibullPowerAFT', 
    surv.Weibull,
    InversePower,
    'alpha',
    lambda x: 1./x
    )

WeibullExponentialAFT = AcceleratedFailureTimeFitter(
    'WeibullExponentialAFT',
    surv.Weibull,
    InverseExponential,
    'alpha',
    lambda x: 1./x
   )

NormalExponentialAFT = AcceleratedFailureTimeFitter('NormalExponentialAFT',
    surv.Normal,
    InverseExponential,
    'mu',
    lambda x: x
    )
