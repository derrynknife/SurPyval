from .accelerated_life import AcceleratedLife
from .dual_exponential import DualExponential
from .dual_power import DualPower
from .exponential import (
    ArrheniusLifeModel,
    ExponentialLifeModel,
    InverseExponential,
)
from .eyring import Eyring, InverseEyring
from .general_linear import GeneralLinear
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
# ``GeneralLogLinear``'s and ``GeneralLinear``'s parameters depend on the
# number of stress columns: each is stored unresolved, and a dict rebuilds
# the model for its columns with ``resolve`` (its ``"n_stresses"``).
# ``Arrhenius`` is another name for the exponential life model: a model
# built with it saves that name and loads back with it.
LIFE_MODELS = {
    model.name: model
    for model in (
        Power,
        InversePower,
        Eyring,
        InverseEyring,
        Linear,
        ExponentialLifeModel,
        ArrheniusLifeModel,
        InverseExponential,
        DualExponential,
        DualPower,
        PowerExponential,
        GeneralLogLinear,
        GeneralLinear,
    )
}

__all__ = [
    "AcceleratedLife",
    "ArrheniusLifeModel",
    "DualExponential",
    "DualPower",
    "ExponentialLifeModel",
    "Eyring",
    "GeneralLinear",
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
