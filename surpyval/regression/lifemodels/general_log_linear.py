from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class GeneralLogLinear_(LifeModel):
    def __init__(self):
        super().__init__(
            "GeneralLogLinear",
            lambda Z: (((None, None),) * Z.shape[1]),
            lambda Z: {"beta_" + str(i): i for i in range(Z.shape[1])},
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        return np.exp(np.dot(Z, np.array(params)))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        # return np.zeros(Z.shape[1])
        return 1.0 / Z.mean(axis=0)


GeneralLogLinear = GeneralLogLinear_()
