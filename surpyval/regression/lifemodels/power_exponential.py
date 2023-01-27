from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class PowerExponential_(LifeModel):
    def __init__(self):
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


PowerExponential = PowerExponential_()
