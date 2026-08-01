from autograd import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrent.parametric.counting_process import IntensityModel
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.utils.recurrent_utils import handle_xicn


class NHPPFitter(IntensityModel):
    def create_negll_func(self, data):
        s = data.split_for_nhpp_likelihood()
        x_o, x_o_prev = s["x_o"], s["x_o_prev"]
        x_right, x_right_prev = s["x_right"], s["x_right_prev"]
        x_left, n_left = s["x_left"], s["n_left"]
        x_i_l, x_i_r, n_i = s["x_i_l"], s["x_i_r"], s["n_i"]
        x_close_last, x_close_tr = s["x_close_last"], s["x_close_tr"]

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

            # extend the integral from each item's last in-window time to its
            # right-truncation time tr (zero when tr is infinite or already
            # coincides with a right-censoring row)
            ll += (
                self.cif(x_close_last, *params) - self.cif(x_close_tr, *params)
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

        ll_func = None
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
        # The MLE objective is already in natural parameter space, so it serves
        # directly as the likelihood used for AIC/BIC/standard errors. The MSE
        # fit has no likelihood, so leave the inference attributes unset (the
        # inference methods then raise).
        if ll_func is not None:
            model._neg_ll = ll_func
            model._mle = np.asarray(params, dtype=float)
            model._n_obs = len(data.x)
        return model

    def fit(
        self,
        x,
        i=None,
        c=None,
        n=None,
        t=None,
        tl=None,
        tr=None,
        how="MLE",
        init=None,
        windows=None,
    ):
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
        t: array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl: array_like or scalar, optional
            Left truncation (delayed entry) time per item; the observation of
            each item begins here. Scalar broadcasts to all items.
        tr: array_like or scalar, optional
            Right truncation time per item.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error.
            Default is 'MLE'.
        init: array_like, optional
            Initial parameters for optimization.
        windows: dict, optional
            Gapped (multi-window) observation: a mapping ``{item: [(start,
            end), ...]}`` giving each item's disjoint observation windows,
            with unobserved gaps between them. When given, every row in ``x``
            must be an observed event (``c=0``); the windows supply the
            end-of-window censoring rows. Because event counts over disjoint
            windows are independent for an NHPP, each window is fitted as its
            own observation period. Mutually exclusive with ``t``/``tl``/
            ``tr``.

        Returns
        -------

        ParametricRecurrenceModel
            An object of fitted model returned by the fit_from_recurrent_data
            method.
        """
        data = handle_xicn(
            x,
            i,
            c,
            n,
            t=t,
            tl=tl,
            tr=tr,
            as_recurrent_data=True,
            windows=windows,
        )
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
