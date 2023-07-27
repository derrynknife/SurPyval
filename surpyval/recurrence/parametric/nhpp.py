from inspect import signature

from autograd import elementwise_grad, jacobian
from autograd import numpy as np
from scipy.optimize import minimize

from surpyval.recurrence.parametric import ParametricRecurrenceModel
from surpyval.utils.recurrent_utils import handle_xicn


class NHPP:
    def __init__(
        self,
        name,
        param_names,
        param_bounds,
        support,
        cif,
        iif=None,
        parameter_initialiser=None,
    ):
        self.name = name
        self.param_names = param_names
        self.bounds = param_bounds
        self.support = support

        if str(signature(cif)) != "(x, *params)":
            raise ValueError("cif must have signature of '(x, *params)'")

        self.cif = cif

        if iif is None:
            self.iif = elementwise_grad(cif)
        else:
            if str(signature(iif)) != "(x, *params)":
                raise ValueError("iif must have signature of '(x, *params)'")
            self.iif = iif

        if parameter_initialiser is None:
            self.parameter_initialiser = lambda _: np.ones(
                len(self.param_names)
            )
        else:
            self.parameter_initialiser = parameter_initialiser

    def create_ll_func(self, x, i, c, n):
        """
        Need to allow for multiple items, i.
        """
        ll_dict = {}

        for ii in set(i):
            T = x[i == ii].max()
            mask = (i == ii) & (c == 0)
            xi = x[mask]
            ni = n[mask]
            ll_dict[ii] = {"T": T, "x": xi, "n": ni}

        def ll_func(params):
            rv = 0
            for i in ll_dict.keys():
                rv += (
                    self.cif(ll_dict[i]["T"], *params)
                    - (
                        ll_dict[i]["n"]
                        * np.log(self.iif(ll_dict[i]["x"], *params))
                    ).sum()
                )
            return rv

        return ll_func

    def fit(self, x, i=None, c=None, n=None, how="MSE", init=None):
        """
        Format for counting processes...
        Thinking x, i, n.
        x = obvious
        i is the identity.

        n is counts ... maybe

        """

        if init is None:
            param_init = self.parameter_initialiser(x)
        else:
            param_init = np.array(init)

        model = ParametricRecurrenceModel()

        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        x_unqiue, r, d = data.to_xrd()

        mcf_hat = np.cumsum(d / r)

        def fun(params):
            return np.sum((self.cif(x_unqiue, *params) - mcf_hat) ** 2)

        res = minimize(fun, param_init)
        model.mcf_hat = mcf_hat

        if how == "MSE":
            params = res.x

        elif how == "MLE":
            ll_func = self.create_ll_func(x, i, c, n)
            jac = jacobian(ll_func)
            res = minimize(ll_func, res.x, jac=jac, method="TNC")
            params = res.x

            model._neg_ll = ll_func(params)
        model.res = res
        model.params = params
        model.x = x_unqiue
        model.dist = self
        model.how = how
        return model

    def from_params(self, params):
        model = ParametricRecurrenceModel()
        model.params = params
        model.dist = self
        model.how = "from_params"
        return model


# Duane Model
# Parameterisation done in accordance with:
# http://reliawiki.org/index.php/Duane_Model

duane_param_names = ["alpha", "b"]
duane_bounds = ((0, None), (0, None))
duane_support = (0.0, np.inf)


def duane_cif(x, *params):
    return params[1] * x ** params[0]


def duane_iif(x, *params):
    return params[0] * params[1] * x ** (params[0] - 1.0)


def duane_rocof(x, *params):
    return (1.0 / params[1]) * x ** (-params[0])


def duane_inv_cif(N, *params):
    return (N / params[1]) ** (1.0 / params[0])


Duane = NHPP(
    "Duane",
    duane_param_names,
    duane_bounds,
    duane_support,
    duane_cif,
    duane_iif,
)

Duane.rocof = duane_rocof  # type: ignore
Duane.inv_cif = duane_inv_cif  # type: ignore

# Cox-Lewis
cox_lewis_param_names = ["alpha", "beta"]
cox_lewis_bounds = ((0, None), (None, None))
cox_lewis_support = (0.0, np.inf)


def cox_lewis_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (np.exp(alpha + beta * x) - np.exp(alpha)) / beta


def cox_lewis_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (np.exp(alpha + beta * x) - np.exp(alpha)) / beta


def cox_lewis_inv_cif(cif, *params):
    alpha = params[0]
    beta = params[1]
    return (np.log((cif * beta) + np.exp(alpha)) - alpha) / beta


def cox_lewis_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return np.exp(alpha + beta * x)


CoxLewis = NHPP(
    "CoxLewis",
    cox_lewis_param_names,
    cox_lewis_bounds,
    cox_lewis_support,
    cox_lewis_cif,
    cox_lewis_iif,
)

CoxLewis.rocof = cox_lewis_rocof  # type: ignore
CoxLewis.inv_cif = cox_lewis_inv_cif  # type: ignore

# Crow

crow_param_names = ["alpha", "beta"]
crow_bounds = ((0, None), (0, None))
crow_support = (0.0, np.inf)


def crow_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (x**beta) / alpha


def crow_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha) * (x ** (beta - 1))


def crow_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha) * x ** (beta - 1.0)


def crow_inv_cif(mcf, *params):
    alpha = params[0]
    beta = params[1]
    return (alpha * mcf) ** (1.0 / beta)


Crow = NHPP(
    "Crow", crow_param_names, crow_bounds, crow_support, crow_cif, crow_iif
)

Crow.rocof = crow_rocof  # type: ignore
Crow.inv_cif = crow_inv_cif  # type: ignore

# Crow

crow_amsaa_param_names = ["alpha", "beta"]
crow_amsaa_bounds = ((0, None), (0, None))
crow_amsaa_support = (0.0, np.inf)


def crow_amsaa_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (x / alpha) ** beta


def crow_amsaa_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha**beta) * (x ** (beta - 1))


def crow_amsaa_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha) * x ** (beta - 1.0)


def crow_amsaa_inv_cif(mcf, *params):
    alpha = params[0]
    beta = params[1]
    return (alpha * mcf) ** (1.0 / beta)


CrowAMSAA = NHPP(
    "CrowAMSAA",
    crow_amsaa_param_names,
    crow_amsaa_bounds,
    crow_amsaa_support,
    crow_amsaa_cif,
    crow_amsaa_iif,
)

CrowAMSAA.rocof = crow_amsaa_rocof  # type: ignore
CrowAMSAA.inv_cif = crow_amsaa_inv_cif  # type: ignore
