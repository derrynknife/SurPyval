from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class Eyring_(LifeModel):
    r"""
    The Eyring life model, from reaction-rate (transition-state) theory:
    the Arrhenius law with a :math:`1 / V` pre-factor,

    .. math::
        L(V) = \frac{1}{V} e^{-(b - a / V)},

    with ``V`` the absolute temperature, in kelvin (or, for a non-thermal
    stress, the stress itself).

    Parameters (as the fitted model reports them):

    - ``a``: the activation energy over Boltzmann's constant, in kelvin, as
      for :class:`Exponential`.
    - ``b``: a constant: :math:`e^{-b}` is the pre-factor that
      :class:`Exponential` calls ``b``, less the :math:`1 / V`.

    :class:`InverseEyring` is its reciprocal, with the constant named
    ``c``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.Eyring.phi(np.array([358.0, 398.0]), 8000.0, 5.0).round(3)
    array([95400.179,  9082.007])
    """

    positive_stress_columns = (0,)
    kelvin_stress_columns = (0,)
    warns_below_kelvin = False
    phi_takes_rows = True

    def __init__(self) -> None:
        super().__init__(
            "Eyring",
            {"a": 0, "b": 1},
            ((None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        a = params[0]
        c = params[1]
        return (1.0 / Z) * np.exp(-(c - a / Z))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        a, c = np.polyfit(1.0 / Z, np.log(life) + np.log(Z), 1)
        return [a, -c]


Eyring = Eyring_()


class InverseEyring_(LifeModel):
    r"""
    The inverse Eyring life model: the reciprocal of :class:`Eyring`, so
    that the rate ``1 / L`` follows the Eyring law,

    .. math::
        L(V) = V e^{c - a / V},

    with ``V`` the absolute temperature, in kelvin.

    Parameters (as the fitted model reports them):

    - ``a``: the temperature sensitivity, in kelvin: life falls with
      temperature where ``a < -V``.
    - ``c``: a constant, the log of the pre-factor; :class:`Eyring` calls
      its constant ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.InverseEyring.phi(np.array([358.0]), -8000.0, 5.0)
    array([2.693147e+14])
    """

    positive_stress_columns = (0,)
    kelvin_stress_columns = (0,)
    phi_takes_rows = True

    def __init__(self) -> None:
        super().__init__(
            "InverseEyring",
            {"a": 0, "c": 1},
            ((None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        a = params[0]
        c = params[1]
        return 1.0 / ((1.0 / Z) * np.exp(-(c - a / Z)))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        a, c = np.polyfit(1.0 / Z, np.log(1.0 / life) + np.log(Z), 1)
        return [a, -c]


InverseEyring = InverseEyring_()
