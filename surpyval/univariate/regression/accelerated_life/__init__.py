from .accelerated_life import AcceleratedLife
from .dual_exponential import DualExponential
from .dual_power import DualPower
from .exponential import ExponentialLifeModel, InverseExponential
from .eyring import Eyring, InverseEyring
from .lifemodel import LifeModel
from .linear import Linear
from .parameter_substitution import ParameterSubstitutionFitter
from .power import InversePower, Power
from .power_exponential import PowerExponential

# Registry of the named life models keyed by their ``.name``, used by
# ``ParametricRegressionModel`` serialisation to rebuild an accelerated-life
# fitter from a stored name. Keyed by ``.name`` (not the Python identifier)
# because a life model's name can differ from its symbol --
# ``ExponentialLifeModel`` has ``name == "Exponential"``, which also collides
# with the ``Exponential`` distribution in the top-level namespace, so an
# explicit map is required.
# ``GeneralLogLinear`` is intentionally excluded: its parameterisation depends
# on the covariate dimension (its ``phi_param_map``/``phi_bounds`` are
# callables), so it cannot be rebuilt from a name alone.
LIFE_MODELS = {
    model.name: model
    for model in (
        Power,
        InversePower,
        Eyring,
        InverseEyring,
        Linear,
        ExponentialLifeModel,
        InverseExponential,
        DualExponential,
        DualPower,
        PowerExponential,
    )
}

__all__ = [
    "AcceleratedLife",
    "DualExponential",
    "DualPower",
    "ExponentialLifeModel",
    "Eyring",
    "InverseExponential",
    "InverseEyring",
    "InversePower",
    "LIFE_MODELS",
    "LifeModel",
    "Linear",
    "ParameterSubstitutionFitter",
    "Power",
    "PowerExponential",
]
