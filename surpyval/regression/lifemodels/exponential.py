from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class InverseExponential_(LifeModel):
    def __init__(self):
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


class Exponential_(LifeModel):
    def __init__(self):
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


Exponential = Exponential_()
