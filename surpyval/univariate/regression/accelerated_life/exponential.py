from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class InverseExponential_(LifeModel):
    r"""
    The inverse exponential life model: the Arrhenius law written for the
    rate rather than the life,

    .. math::
        L(V) = \frac{1}{b\, e^{a / V}}.

    Parameters (as the fitted model reports them):

    - ``a``: the temperature sensitivity, with ``V`` in kelvin: the rate
      ``1 / L`` grows as :math:`e^{a / V}`, so with ``a < 0`` the rate rises
      (life falls) with temperature.
    - ``b`` (> 0): the rate as ``V`` grows without bound.

    It is :class:`Exponential` with ``b`` replaced by ``1 / b`` and ``a``
    by ``-a``: the same fits, a different parameterisation.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.InverseExponential.phi(np.array([300.0]), -3000.0, 1.0)
    array([22026.46579481])
    """

    kelvin_stress_columns = (0,)
    phi_takes_rows = True

    def __init__(self) -> None:
        super().__init__(
            "InverseExponential",
            {"a": 0, "b": 1},
            ((None, None), (0, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        a = params[0]
        b = params[1]
        return 1.0 / (b * np.exp(a / Z))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        a, b = np.polyfit(1.0 / Z, np.log(1.0 / life), 1)
        return [a, np.exp(b)]


InverseExponential = InverseExponential_()


class ExponentialLifeModel_(LifeModel):
    r"""
    The exponential (Arrhenius) life model, for thermally activated
    failure (chemical reaction, diffusion, electromigration):

    .. math::
        L(V) = b\, e^{a / V},

    with ``V`` the absolute temperature, in kelvin.

    Parameters (as the fitted model reports them):

    - ``a``: the activation energy over Boltzmann's constant,
      :math:`E_a / k_B`, in kelvin; the activation energy in electronvolts
      is ``a * 8.617e-5``. The acceleration factor from temperature
      :math:`V_1` to :math:`V_2` is :math:`e^{a (1/V_1 - 1/V_2)}`.
    - ``b`` (> 0): the life as the temperature grows without bound (the
      pre-exponential factor).

    In ``surpyval.life_models`` it is ``Exponential``; at the top level
    that name is the distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> T = np.array([358.0, 398.0])  # 85 and 125 degrees C
    >>> life_models.Exponential.phi(T, 8000.0, 1e-5).round(1)
    array([50687.9,  5364.6])
    """

    kelvin_stress_columns = (0,)
    phi_takes_rows = True

    def __init__(self) -> None:
        super().__init__(
            "Exponential",
            {"a": 0, "b": 1},
            ((None, None), (0, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        a = params[0]
        b = params[1]
        return b * np.exp(a / Z)

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        a, b = np.polyfit(1.0 / Z, np.log(life), 1)
        return [a, np.exp(b)]


ExponentialLifeModel = ExponentialLifeModel_()
