import autograd.numpy as np

from .parameter_substitution import ParameterSubstitutionFitter

# Map each supported distribution to its life parameter name and any
# transforms needed to convert between the life-model output space and
# the distribution's native parameter space.
_LIFE_PARAM_MAP = {
    "Exponential": ("lambda", lambda x: 1.0 / x, lambda x: 1.0 / x),
    "Normal": ("mu", None, None),
    "Weibull": ("alpha", None, None),
    "Gumbel": ("mu", None, None),
    "Logistic": ("mu", None, None),
    "LogNormal": ("mu", np.log, np.exp),
    "Gamma": ("beta", None, None),
}


def AcceleratedLife(distribution, life_model):
    """
    Create an Accelerated Life fitter for the given distribution and
    life model.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``LogNormal``).
    life_model : LifeModel
        A stress-relationship model (e.g. ``Power``, ``Eyring``).

    Returns
    -------
    ParameterSubstitutionFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.regression import AcceleratedLife, Power
    >>> model = AcceleratedLife(Weibull, Power).fit(x, c=c, Z=stress)
    """
    if distribution.name not in _LIFE_PARAM_MAP:
        supported = list(_LIFE_PARAM_MAP.keys())
        raise ValueError(
            f"Distribution '{distribution.name}' is not supported by "
            f"AcceleratedLife. Supported distributions: {supported}"
        )

    life_param, transform, inv_transform = _LIFE_PARAM_MAP[distribution.name]
    name = f"{distribution.name}{life_model.name}AL"

    return ParameterSubstitutionFitter(
        kind="Accelerated Life",
        name=name,
        distribution=distribution,
        life_model=life_model,
        life_parameter=life_param,
        param_transform=transform,
        inverse_param_transform=inv_transform,
    )
