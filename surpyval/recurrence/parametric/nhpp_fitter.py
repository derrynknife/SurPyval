from autograd import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.utils.recurrent_utils import handle_xicn


class NHPPFitter:
    def create_negll_func(self, data):
        x, c, n = data.x, data.c, data.n
        x_prev = data.find_x_prevget_previous_xious()

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
                self.log_iif(x_o, *params)
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
        """
        Fit the NHPP model from recurrent data using either Maximum Likelihood
        Estimation (MLE) or Mean Square Error (MSE) methods.

        Parameters
        ----------

        data: Recurrent
            Recurrent data object containing properties x, c, and n.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error. Default
            is 'MLE'.
        init: array_like, optional
            Initial parameters for optimization.

        Returns
        -------

        ParametricRecurrenceModel
            An instance of the ParametricRecurrenceModel class containing the
            fitted model, estimated parameters, and other relevant attributes.
        """
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
        Fit the NHPP model from the provided data. This function prepares the
        data to ensure that it is in the correct format for the fitting.

        Parameters
        ----------

        x: array_like
            The input data.
        i: array_like, optional
            Identity of each observation.
        c: array_like, optional
            Censoring indicator.
        n: array_like, optional
            Counts for each observation.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error.
            Default is 'MLE'.
        init: array_like, optional
            Initial parameters for optimization.

        Returns
        -------

        ParametricRecurrenceModel
            An object of fitted model returned by the fit_from_recurrent_data
            method.
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return self.fit_from_recurrent_data(data, how, init)

    def from_params(self, params):
        """
        Create a model instance directly from parameters without fitting.

        Parameters
        ----------

        params: array_like
            Parameters to be used directly to create the model.

        Returns
        -------

        ParametricRecurrenceModel
            An instance of the ParametricRecurrenceModel class initialized with
            the provided parameters.
        """
        model = ParametricRecurrenceModel()
        model.params = params
        model.dist = self
        model.how = "from_params"
        return model
