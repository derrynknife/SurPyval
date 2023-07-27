import numpy as np


class HPP:
    def __init__(self):
        self.hpp_param_names = ["lambda"]
        self.hpp_bounds = ((0, None),)
        self.hpp_support = (0.0, np.inf)

    def iif(self, x, *params):
        rate = params[0]
        return np.ones_like(x) * rate

    def cif(self, x, *params):
        rate = params[0]
        return rate * x

    def rocof(self, x, *params):
        rate = params[0]
        return np.ones_like(x) * rate

    def inv_cif(self, cif, *params):
        rate = params[0]
        return cif / rate

    def fit(self):
        pass
