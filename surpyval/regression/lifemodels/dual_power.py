from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class DualPower_(LifeModel):
    def __init__(self):
        super().__init__(
            "DualPower",
            {"c": 0, "m": 1, "n": 2},
            ((0, None), (None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        Z = np.atleast_2d(Z)
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]
        c = params[0]
        m = params[1]
        n = params[2]
        return c * Z1**m * Z2**n

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        A = np.atleast_2d(Z)
        A = np.hstack([np.ones(Z.shape[0]).reshape(-1, 1), np.log(Z)])
        y = np.log(life)
        c, m, n = np.linalg.lstsq(A, y, rcond=None)[0]
        return [np.exp(c), m, n]


DualPower = DualPower_()
