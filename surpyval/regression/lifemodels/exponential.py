from autograd import numpy as np


class InverseExponential_:
    def __init__(self):
        self.name = "InverseExponential"
        self.phi_param_map = {"a": 0, "b": 1}
        self.phi_bounds = (
            (None, None),
            (0, None),
        )

    def phi(self, Z, *params):
        a = params[0]
        b = params[1]
        return 1.0 / (b * np.exp(a / Z))

    def phi_init(self, life, Z):
        Z = Z.flatten()
        a, b = np.polyfit(1.0 / Z, np.log(1.0 / life), 1)
        return [a, np.exp(b)]


InverseExponential = InverseExponential_()


class Exponential_:
    def __init__(self):
        self.name = "Exponential"
        self.phi_param_map = {"a": 0, "b": 1}
        self.phi_bounds = (
            (None, None),
            (0, None),
        )

    def phi(self, Z, *params):
        a = params[0]
        b = params[1]
        return b * np.exp(a / Z)

    def phi_init(self, life, Z):
        Z = Z.flatten()
        a, b = np.polyfit(1.0 / Z, np.log(life), 1)
        return [a, np.exp(b)]


Exponential = Exponential_()
