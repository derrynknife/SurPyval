from autograd import numpy as np


class Linear_:
    def __init__(self):
        self.name = "Linear"
        self.phi_param_map = {"a": 0, "b": 1}
        self.phi_bounds = (
            (None, None),
            (None, None),
        )

    def phi(self, Z, *params):
        a = params[0]
        b = params[1]
        return a + b * Z

    def phi_init(self, life, Z):
        Z = Z.flatten()
        b, a = np.polyfit(Z, life, 1)
        return [a, b]


Linear = Linear_()
