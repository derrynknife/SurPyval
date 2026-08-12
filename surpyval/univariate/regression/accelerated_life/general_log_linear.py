from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class GeneralLogLinear_(LifeModel):
    def __init__(self) -> None:
        super().__init__(
            "GeneralLogLinear",
            # Swapped until 0.19.1: the bounds lambda sat in the
            # phi_param_map slot and vice versa. LifeModel takes
            # (name, phi_param_map, phi_bounds).
            #
            # Both are callables of Z rather than the dict and tuple the
            # base declares, because this model's parameterisation
            # depends on the covariate dimension. That is why it is left
            # out of LIFE_MODELS and cannot be rebuilt from a name -- and
            # why this module is not in the type ratchet.
            lambda Z: {  # type: ignore[arg-type]
                "beta_" + str(i): i for i in range(Z.shape[1])
            },
            lambda Z: (((None, None),) * Z.shape[1]),  # type: ignore[arg-type]
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        return np.exp(np.dot(Z, np.array(params)))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        return (1.0 / Z.mean(axis=0)).tolist()


GeneralLogLinear = GeneralLogLinear_()
