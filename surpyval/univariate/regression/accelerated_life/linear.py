from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class Linear_(LifeModel):
    def __init__(self) -> None:
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
        lives = np.asarray(life, dtype=float)
        b, a = np.polyfit(Z, lives, 1)
        # A least-squares line through positive lives can still cross zero
        # at an observed stress (steeply falling lives), and a Weibull scale,
        # an exponential rate or a log-normal log-life cannot start there:
        # the fit then refused to begin. Flatten the line about the mean
        # life just enough that it stays at least half the mean life at
        # every observed stress -- still a sloped, and so informative, start.
        if np.all(lives > 0) and np.min(a + b * Z) <= 0:
            z_bar, l_bar = Z.mean(), lives.mean()
            drop = -b * (Z - z_bar)
            worst = np.max(drop)
            if worst > 0:
                b = b * min(1.0, 0.5 * l_bar / worst)
            a = l_bar - b * z_bar
        return [a, b]


Linear = Linear_()
