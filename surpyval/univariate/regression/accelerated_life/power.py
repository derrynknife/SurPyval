from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class InversePower_(LifeModel):
    r"""
    The inverse power law life model: life falls as a power of the stress,
    written as a rate,

    .. math::
        L(V) = \frac{1}{a V^{n}}.

    The usual model for voltage endurance and mechanical fatigue, where
    the stress ``V`` (voltage, load, cycles' amplitude) must be positive.

    Parameters (as the fitted model reports them):

    - ``a`` (> 0): the rate, ``1 / L``, at unit stress (``V = 1`` in the
      units of ``V``).
    - ``n``: the power. Doubling the stress multiplies the life by
      :math:`2^{-n}`; with ``n > 0`` life falls as stress rises.

    It is :class:`Power` with ``a`` replaced by ``1 / a`` and ``n`` by
    ``-n``: the same fits, a different parameterisation.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.InversePower.phi(np.array([1.0, 2.0]), 0.01, 1.5)
    array([100.        ,  35.35533906])
    """

    positive_stress_columns = (0,)
    phi_takes_rows = True
    log_scale_parameters = ("a",)

    def __init__(self) -> None:
        super().__init__(
            "InversePower",
            {"a": 0, "n": 1},
            ((0, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        # One exponent, as the other log-linear life models (#634)
        return self._phi_from_log_life(Z, params)

    def log_life(self, Z: ndarray, *params: float) -> ndarray:
        return -(params[0] + params[1] * np.log(Z))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        n, a = np.polyfit(np.log(Z), np.log(1.0 / life), 1)
        return [np.exp(a), n]


InversePower = InversePower_()


class Power_(LifeModel):
    r"""
    The power law life model: life is a power of the stress,

    .. math::
        L(V) = a V^{n}.

    For non-thermal stresses (voltage, load, pressure, cycling rate), which
    must be positive.

    Parameters (as the fitted model reports them):

    - ``a`` (> 0): the life at unit stress (``V = 1`` in the units of
      ``V``).
    - ``n``: the power. Doubling the stress multiplies the life by
      :math:`2^{n}`; with ``n < 0`` life falls as stress rises.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.Power.phi(np.array([1.0, 2.0]), 100.0, -1.5)
    array([100.        ,  35.35533906])
    """

    positive_stress_columns = (0,)
    phi_takes_rows = True
    log_scale_parameters = ("a",)

    def __init__(self) -> None:
        super().__init__(
            "Power",
            {"a": 0, "n": 1},
            ((0, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        # One exponent, as the other log-linear life models (#634)
        return self._phi_from_log_life(Z, params)

    def log_life(self, Z: ndarray, *params: float) -> ndarray:
        return params[0] + params[1] * np.log(Z)

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        n, a = np.polyfit(np.log(Z), np.log(life), 1)
        return [np.exp(a), n]


Power = Power_()
