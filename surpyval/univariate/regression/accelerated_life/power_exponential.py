from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class PowerExponential_(LifeModel):
    r"""
    The power-exponential life model, for a thermal and a non-thermal
    stress (temperature with voltage, say): Arrhenius in the first, a
    power law in the second,

    .. math::
        L(U, V) = c\, e^{a / U} V^{n},

    with ``U`` the absolute temperature, in kelvin, and ``V`` the other
    stress (positive), the two columns of ``Z``.

    Parameters (as the fitted model reports them):

    - ``c`` (> 0): a scale constant, the life at unit ``V`` as the
      temperature grows without bound.
    - ``a``: the activation energy over Boltzmann's constant, in kelvin,
      as for :class:`Exponential`.
    - ``n``: the power of ``V``: doubling ``V`` multiplies the life by
      :math:`2^{n}`.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> Z = np.array([[358.0, 2.0]])
    >>> life_models.PowerExponential.phi(Z, 1e-5, 8000.0, -1.5).round(1)
    array([17920.9])
    """

    n_stresses = 2
    positive_stress_columns = (1,)
    kelvin_stress_columns = (0,)
    phi_takes_rows = True

    def __init__(self) -> None:
        super().__init__(
            "PowerExponential",
            {"c": 0, "a": 1, "n": 2},
            ((0, None), (None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        Z = np.atleast_2d(Z)
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]
        c = params[0]
        a = params[1]
        n = params[2]
        return c * np.exp(a / Z1) * Z2**n

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        A = np.atleast_2d(Z)
        A = np.hstack([np.ones(Z.shape[0]).reshape(-1, 1), Z])
        A[:, 1] = 1.0 / A[:, 1]
        A[:, 2] = np.log(A[:, 2])
        y = np.log(life)
        c, a, n = np.linalg.lstsq(A, y, rcond=None)[0]
        return [np.exp(c), a, n]

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        # log L = log c + a / s1 + n log s2: identified even with equal
        # stress columns, unless both are constant.
        Z = np.atleast_2d(Z)
        terms = np.stack([1.0 / Z[:, 0], np.log(Z[:, 1])], axis=1)
        return terms, ("a", "n"), True


PowerExponential = PowerExponential_()
