from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class InversePower_(LifeModel):
    def __init__(self):
        super().__init__(
            "InversePower",
            {"a": 0, "n": 1},
            ((0, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        return 1.0 / (params[0] * Z ** params[1])

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        n, a = np.polyfit(np.log(Z), np.log(1.0 / life), 1)
        return [np.exp(a), n]


InversePower = InversePower_()


class Power_(LifeModel):
    def __init__(self):
        super().__init__(
            "Power",
            {"a": 0, "n": 1},
            ((0, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        return params[0] * Z ** params[1]

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        Z = Z.flatten()
        n, a = np.polyfit(np.log(Z), np.log(life), 1)
        return [np.exp(a), n]


Power = Power_()
