import numpy as np
import numpy_indexed as npi
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.utils.recurrent_utils import handle_xicn


class HPP:
    def __init__(self):
        self.hpp_param_names = ["lambda"]
        self.hpp_bounds = ((0, None),)
        self.hpp_support = (0.0, np.inf)
        self.name = "Homogeneous Poisson Process"

    def iif(self, x, rate):
        return np.ones_like(x) * rate

    def cif(self, x, rate):
        return rate * x

    def rocof(self, x, rate):
        return np.ones_like(x) * rate

    def inv_cif(self, cif, rate):
        return cif / rate

    @classmethod
    def create_negll_func(cls, x, i, c, n):
        def negll_func(rate):
            ll = 0

            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                x_prev = 0

                for j in range(0, len(x_item)):
                    if x_item.ndim == 1:
                        x_j = x_item[j]
                    else:
                        x_j = x_item[j][0]
                        x_jr = x_item[j][1]
                    if c_item[j] == 0:
                        ll += np.log(rate) - (rate * x_j) + (rate * x_prev)
                        x_prev = x_j
                    elif c_item[j] == 1:
                        ll += rate * (x_prev - x_j)
                        x_prev = x_j
                    elif c_item[j] == 2:
                        delta_x = x_jr - x_j
                        ll += (
                            n_item[j] * np.log(rate * delta_x)
                            - gammaln(n_item[j] + 1)
                            - (rate * delta_x)
                        )
                        x_prev = x_jr
                    elif c_item[j] == -1:
                        ll += (
                            n_item[j] * np.log(rate * x_j)
                            - gammaln(n_item[j] + 1)
                            - (rate * x_j)
                        )
                        x_prev = x_j

            return -ll

        return negll_func

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, init=None):
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)

        out = ParametricRecurrenceModel()
        out.dist = cls()
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)
        out.name = "Homogeneous Poisson Process"

        init = (data.n[data.c == 0]).sum() / npi.group_by(data.i).max(data.x)[
            1
        ].sum()
        neg_ll = cls.create_negll_func(data.x, data.i, data.c, data.n)
        res = minimize(
            neg_ll, [init], method="Nelder-Mead", bounds=((0, None),)
        )
        out.res = res
        out.params = res.x

        return out
