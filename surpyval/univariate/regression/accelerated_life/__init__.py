from .accelerated_life import AcceleratedLife
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

# Registry of the named life models keyed by their ``.name``, used by
# ``ParametricRegressionModel`` serialisation to rebuild an accelerated-life
# fitter from a stored name. Keyed by ``.name`` (not the Python identifier)
# because a life model's name can differ from its symbol --
# ``ExponentialLifeModel`` has ``name == "Exponential"``, which also collides
# with the ``Exponential`` distribution in the top-level namespace, so an
# explicit map is required.
# ``GeneralLogLinear``'s parameters depend on the number of stress columns:
# it is stored unresolved, and a dict rebuilds the model for its columns
# with ``resolve`` (its ``"n_stresses"``).
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
        GeneralLogLinear,
    )
}

__all__ = [
    "AcceleratedLife",
    "DualExponential",
    "DualPower",
    "ExponentialLifeModel",
    "Eyring",
    "GeneralLogLinear",
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
