from inspect import signature

from autograd import elementwise_grad
from autograd import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
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

    def create_negll_func(self, data):
        x, c, n = data.x, data.c, data.n
        # Covariates
        x_prev = data.find_x_previous()

        has_interval_censoring = True if 2 in c else False
        has_observed = True if 0 in c else False
        has_left_censoing = True if -1 in c else False
        has_right_censoring = True if 1 in c else False

        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None

        x_prev_l = x_prev if x_prev.ndim == 1 else x[:, 0]
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else None

        x_o = x_l[c == 0] if has_observed else np.array([])
        if has_interval_censoring:
            x_o_prev = x_prev_r[c == 0] if has_observed else np.array([])
        else:
            x_o_prev = x_prev_l[c == 0] if has_observed else np.array([])

        x_right = x_l[c == 1] if has_right_censoring else np.array([])
        if has_interval_censoring:
            x_right_prev = (
                x_prev_r[c == 1] if has_right_censoring else np.array([])
            )
        else:
            x_right_prev = (
                x_prev_l[c == 1] if has_right_censoring else np.array([])
            )

        x_left = x_l[c == -1] if has_left_censoing else np.array([])
        n_left = n[c == -1] if has_left_censoing else np.array([])

        x_i_l = x_l[c == 2] if has_interval_censoring else np.array([])
        x_i_r = x_r[c == 2] if has_interval_censoring else np.array([])
        n_i = n[c == 2] if has_interval_censoring else np.array([])

        # Using the empty arrays avoids the need for if statements in the
        # likelihood function. It also means that the likelihood function
        # will not encounter any invalid values since taking the log of 0
        # will not occur.

        def negll_func(params):
            # ll of directly observed
            ll = (
                np.log(self.iif(x_o, *params))
                + self.cif(x_o_prev, *params)
                - self.cif(x_o, *params)
            ).sum()

            # ll of right censored
            ll += (
                self.cif(x_right_prev, *params) - self.cif(x_right, *params)
            ).sum()

            # ll of left censored
            left_delta_cif = self.cif(x_left, *params)
            ll += (
                n_left * np.log(left_delta_cif)
                - (left_delta_cif)
                - gammaln(n_left + 1)
            ).sum()

            # ll of interval censored
            interval_delta_cif = self.cif(x_i_r, *params) - self.cif(
                x_i_l, *params
            )
            ll += (
                n_i * np.log(interval_delta_cif)
                - (interval_delta_cif)
                - gammaln(n_i + 1)
            ).sum()

            return -ll

        return negll_func

    def fit_from_recurrent_data(self, data, how="MLE", init=None):
        if init is None:
            param_init = self.parameter_initialiser(data.x)
        else:
            param_init = np.array(init)

        x_unqiue, r, d = data.to_xrd()
        mcf_hat = np.cumsum(d / r)

        def fun(params):
            return np.sum((self.cif(x_unqiue, *params) - mcf_hat) ** 2)

        res = minimize(fun, param_init, bounds=self.bounds)
        param_init = res.x

        if how == "MSE":
            params = res.x

        elif how == "MLE":
            ll_func = self.create_negll_func(data)
            res = minimize(
                ll_func,
                param_init,
                method="Nelder-Mead",
                bounds=self.bounds,
            )
            params = res.x

        model = ParametricRecurrenceModel()
        model.mcf_hat = mcf_hat
        model.res = res
        model.params = params
        model.data = data
        model.dist = self
        model.how = how
        return model

    def fit(self, x, i=None, c=None, n=None, how="MLE", init=None):
        """
        Format for counting processes...
        Thinking x, i, n.
        x = obvious
        i is the identity.

        n is counts ... maybe
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return self.fit_from_recurrent_data(data, how, init)

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
    return np.exp(alpha + beta * x)


def cox_lewis_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(alpha + beta * x)


def cox_lewis_inv_cif(cif, *params):
    alpha = params[0]
    beta = params[1]
    return (np.log(cif) - alpha) / beta


def cox_lewis_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return beta * np.exp(alpha + beta * x)


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
    return (beta / alpha**beta) * (x ** (beta - 1))


def crow_amsaa_inv_cif(mcf, *params):
    alpha = params[0]
    beta = params[1]
    return alpha * (mcf ** (1.0 / beta))


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
