# from autograd import elementwise_grad
import numpy as np
import numpy_indexed as npi
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.parametric import Duane
from surpyval.utils.recurrent_utils import handle_xicn


class ProportionalIntensityModel:
    def __repr__(self):
        return "Parametric Proportional Intensity Model with {} CIF".format(
            self.name
        )

    def cif(self, x, Z):
        return self.dist.cif(x, *self.params) * np.exp(Z @ self.coeffs)

    def iif(self, x, Z):
        return self.dist.iif(x, *self.params) * np.exp(Z @ self.coefs)

    def inv_cif(self, x, Z):
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x / np.exp(self.coeffs @ Z), *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    # TODO: random, to T, and to N


class ProportionalIntensityHPP:
    @classmethod
    def iif(self, x, rate):
        return np.ones_like(x) * rate

    @classmethod
    def cif(self, x, rate):
        return rate * x

    @classmethod
    def inv_cif(self, cif, rate):
        return cif / rate

    @classmethod
    def create_negll_func(cls, x, i, c, n, Z):
        def negll_func(params):
            rate = np.exp(params[0])
            beta_coeffs = params[1:]
            ll = 0

            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                Z_item = np.atleast_2d(Z[mask_item])
                x_prev = 0

                for j in range(0, len(x_item)):
                    phi_exponent = Z_item[j] @ beta_coeffs
                    phi = np.exp(phi_exponent)
                    if x_item.ndim == 1:
                        x_j = x_item[j]
                    else:
                        x_j = x_item[j][0]
                        x_jr = x_item[j][1]
                    if c_item[j] == 0:
                        ll += (
                            np.log(rate)
                            + phi_exponent
                            + (phi * rate) * (x_prev - x_j)
                            # If this becomes time varying, will need to
                            # use phi_prev
                        )
                        x_prev = x_j
                    elif c_item[j] == 1:
                        ll += rate * phi * (x_prev - x_j)
                        x_prev = x_j
                    elif c_item[j] == 2:
                        delta_x = x_jr - x_j
                        ll += (
                            n_item[j] * (np.log(rate * delta_x) + phi_exponent)
                            - (phi * rate * delta_x)
                            - gammaln(n_item[j] + 1)
                        )
                        x_prev = x_jr
                    elif c_item[j] == -1:
                        ll += (
                            n_item[j] * (np.log(rate * x_j) + phi_exponent)
                            - (phi * rate * x_j)
                            - gammaln(n_item[j] + 1)
                        )
                        x_prev = x_j

            return -ll

        return negll_func

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, init=None):
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)

        out = ProportionalIntensityModel()
        out.dist = cls
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)
        out.name = "Homogeneous Poisson Process"

        init = (data.n[data.c == 0]).sum() / npi.group_by(data.i).max(data.x)[
            1
        ].sum()

        num_covariates = Z.shape[1]
        init = np.append(np.log(init), np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(data.x, data.i, data.c, data.n, data.Z)

        res = minimize(
            neg_ll,
            [init],
        )
        out.res = res
        out.params = np.atleast_1d(np.exp(res.x[0]))
        out.coeffs = np.atleast_1d(res.x[1:])
        out.name = "Homogeneous Poisson Process"

        return out


class ProportionalIntensityNHPP:
    @classmethod
    def iif(self, x, rate):
        return np.ones_like(x) * rate

    @classmethod
    def cif(self, x, rate):
        return rate * x

    @classmethod
    def inv_cif(self, cif, rate):
        return cif / rate

    @classmethod
    def create_negll_func(cls, x, i, c, n, Z, dist):
        def negll_func(params):
            dist_params = params[: len(dist.param_names)]
            beta_coeffs = params[len(dist.param_names) :]
            ll = 0

            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                Z_item = np.atleast_2d(Z[mask_item])
                x_prev = 0

                for j in range(0, len(x_item)):
                    phi_exponent = Z_item[j] @ beta_coeffs
                    phi = np.exp(phi_exponent)
                    if x_item.ndim == 1:
                        x_j = x_item[j]
                    else:
                        x_j = x_item[j][0]
                        x_jr = x_item[j][1]

                    if c_item[j] == 0:
                        ll += (
                            np.log(dist.iif(x_j, *dist_params))
                            + phi_exponent
                            + phi
                            * (
                                dist.cif(x_prev, *dist_params)
                                - dist.cif(x_j, *dist_params)
                            )
                        )
                        x_prev = x_j
                    elif c_item[j] == 1:
                        ll += phi * (
                            dist.cif(x_prev, *dist_params)
                            - dist.cif(x_j, *dist_params)
                        )
                        x_prev = x_j
                    elif c_item[j] == 2:
                        delta_cif = phi * (
                            dist.cif(x_jr, *dist_params)
                            - dist.cif(x_j, *dist_params)
                        )
                        ll += (
                            n_item[j] * np.log(delta_cif)
                            + phi_exponent
                            - (delta_cif)
                            - gammaln(n_item[j] + 1)
                        )
                        x_prev = x_jr
                    elif c_item[j] == -1:
                        delta_cif = phi * dist.cif(x_j, *params)
                        ll += (
                            n_item[j] * np.log(delta_cif)
                            - (delta_cif)
                            - gammaln(n_item[j] + 1)
                        )
                        x_prev = x_j

            return -ll

        return negll_func

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, dist=Duane, init=None):
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)

        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)
        out.name = "Homogeneous Poisson Process"

        init = np.ones(len(dist.param_names))

        num_covariates = Z.shape[1]
        init = np.append(init, np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(
            data.x, data.i, data.c, data.n, data.Z, dist
        )

        res = minimize(
            neg_ll,
            [init],
        )
        out.res = res
        out.params = res.x[: len(dist.param_names)]
        out.coeffs = res.x[len(dist.param_names) :]
        out.name = "Non-Homogeneous Poisson Process"

        return out
