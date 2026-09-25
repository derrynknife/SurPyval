from typing import Callable

from autograd import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrent.inference import observed_event_count
from surpyval.recurrent.parametric.counting_process import IntensityModel
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.utils.recurrent_event_data import RecurrentEventData
from surpyval.utils.recurrent_utils import handle_xicn


class NHPPFitter(IntensityModel):
    #: Natural-space parameter bounds, set by each concrete intensity model.
    bounds: tuple

    def create_negll_func(self, data: RecurrentEventData) -> Callable:
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

        def negll_func(params: np.ndarray) -> float:
            # ll of directly observed
            ll = np.sum(
                self.log_iif(x_o, *params)
                + self.cif(x_o_prev, *params)
                - self.cif(x_o, *params)
            )

            # ll of right censored
            ll += np.sum(
                self.cif(x_right_prev, *params) - self.cif(x_right, *params)
            )

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
            ll += np.sum(
                self.cif(x_close_last, *params) - self.cif(x_close_tr, *params)
            )

            return -ll

        return negll_func

    def fit_from_recurrent_data(
        self,
        data: RecurrentEventData,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> ParametricRecurrenceModel:
        """
        Fit the NHPP model from recurrent data using either Maximum Likelihood
        Estimation (MLE) or Mean Square Error (MSE) methods.

        Parameters
        ----------

        data: RecurrentEventData
            The recurrent event data, as built by ``surpyval.handle_xicn``.
            :meth:`fit` builds one from its arrays and calls this.
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

        def fun(params: np.ndarray) -> float:
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
            model._n_obs = observed_event_count(data)
        return model

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        t: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
        tr: "ArrayLike | None" = None,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
        windows: "dict | None" = None,
    ) -> ParametricRecurrenceModel:
        """
        Fit the NHPP model from the provided data. This function prepares the
        data to ensure that it is in the correct format for the fitting.

        Parameters
        ----------

        x: array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
        i: array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c: array_like, optional
            Censoring indicators: 0 an observed event, 1 the right-censored
            end of an item's observation (the time it was last seen), -1
            left-censored and 2 interval-censored counts (with ``n``).
            Defaults to all observed.
        n: array_like, optional
            Number of events in each row (for left- and interval-censored
            counts). Defaults to 1.
        t: array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl: array_like or scalar, optional
            Left truncation (delayed entry) time per item; the observation of
            each item begins here. Scalar broadcasts to all items.
        tr: array_like or scalar, optional
            Right truncation time per item.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error (least
            squares between the model's cumulative intensity and the
            non-parametric MCF of the data). Default is 'MLE'; the MLE
            search starts from the MSE fit.
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
            The fitted model.

        Examples
        --------
        Two systems, each observed to t = 60 (the ``c=1`` rows), whose
        failures become less frequent over time (``beta < 1``):

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> model = CrowAMSAA.fit(x, i=i, c=c)
        >>> model.params
        array([7.82428586, 0.73833691])
        >>> model.cif(60)
        np.float64(4.49998940938965)
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

    def from_params(self, params: ArrayLike) -> ParametricRecurrenceModel:
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

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> model = CrowAMSAA.from_params([10, 1.5])
        >>> model.cif([10, 20])
        array([1.        , 2.82842712])
        """
        model = ParametricRecurrenceModel()
        model.params = np.asarray(params)
        model.dist = self
        model.how = "from_params"
        return model
