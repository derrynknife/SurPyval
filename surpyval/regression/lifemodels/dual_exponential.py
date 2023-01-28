from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class DualExponential_(LifeModel):
    def __init__(self):
        super().__init__(
            "DualExponential",
            {"a": 0, "b": 1, "c": 2},
            ((None, None), (None, None), (0, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        Z = np.atleast_2d(Z)
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]
        a = params[0]
        b = params[1]
        c = params[2]
        return c * np.exp(a / Z1 + b / Z2)

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        A = np.atleast_2d(Z)
        A = 1.0 / np.hstack([np.ones(Z.shape[0]).reshape(-1, 1), Z])
        y = np.log(life)
        c, a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return [a, b, np.exp(c)]


DualExponential = DualExponential_()
