from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class Linear_(LifeModel):
    def __init__(self):
        super().__init__(
            "Linear",
            {"a": 0, "b": 1},
            ((None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        a = params[0]
        b = params[1]
        return a + b * Z

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        b, a = np.polyfit(Z, life, 1)
        return [a, b]


Linear = Linear_()
