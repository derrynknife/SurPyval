from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class Eyring_(LifeModel):
    def __init__(self):
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
    def __init__(self):
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
