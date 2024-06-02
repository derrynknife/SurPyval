import numpy as np
from autograd import hessian, jacobian
from autograd import numpy as anp
from scipy.optimize import root
from scipy.special import gammaln

from surpyval.recurrence.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.utils.recurrent_utils import handle_xicn


class HPP:
    """
    Represents the Homogeneous Poisson Process (HPP) model.
    This class includes methods to evaluate various statistical functions of
    the model and perform parameter estimation based on input data.

    Examples
    --------

    >>> from surpyval import HPP, Exponential
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = HPP.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Homogeneous Poisson Process
    Fitted by           : MLE
    Parameters          :
        lambda: 0.000498450145693719
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([0.00049845, 0.0009969 , 0.00149535, 0.0019938 , 0.00249225,
           0.0029907 ])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([0.00049845, 0.00049845, 0.00049845, 0.00049845, 0.00049845,
           0.00049845])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 2006.21869336,  4012.43738672,  6018.65608009,  8024.87477345,
           10031.09346681, 12037.31216017])
    """

    def __init__(self):
        self.param_names = ["lambda"]
        self.bounds = ((0, None),)
        self.support = (0.0, np.inf)
        self.name = "Homogeneous Poisson Process"

    def iif(self, x, rate):
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        HPP model.

        Parameters
        ----------
        x : array_like
            The values at which IIF is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The IIF values at specified x.
        """
        return np.ones_like(x) * rate

    def log_iif(self, x, rate):
        """
        Natural logarithm of the instantaneous intensity function (IIF) of
        the HPP model.

        Parameters
        ----------
        x : array_like
            The values at which log(IIF) is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The log(IIF) values at specified x.
        """
        return np.log(rate) * np.ones_like(x)

    def cif(self, x, rate):
        """
        Cumulative intensity function (CIF) of the HPP model.

        Parameters
        ----------
        x : array_like
            The values at which CIF is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The CIF values at specified x.
        """
        return rate * np.array(x)

    def inv_cif(self, cif, rate):
        """
        Inverse of the cumulative intensity function (CIF) of the HPP model.

        Parameters
        ----------
        cif : array_like
            The CIF values to be inverted.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The inverted CIF values.
        """
        return np.array(cif) / rate

    @classmethod
    def create_negll_func(cls, data):
        x, c, n = data.x, data.c, data.n
        x_prev = data.get_previous_x()

        has_observed = True if 0 in c else False
        has_right_censoring = True if 1 in c else False
        has_left_censoring = True if -1 in c else False
        has_interval_censoring = True if x.ndim == 2 else False

        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else x_prev

        # This code splits each observation type, if it exists, into its own
        # array. This is done to avoid having to simplify the log-likelihood
        # function to account for the different types of observations.

        # Further by calculating the sum of the needed arrays, we can avoid
        # having to do array sums in the log-likelihood function. This will be
        # faster, especially for large datasets.

        # Although this code is a bit more complex it results in a longer time
        # to create the log-likelihood function, but a faster time to evaluate
        # the log-likelihood function.

        # In conclusion, this is a ridiculous optimisation that is probably
        # not worth the effort that went into it.
        if has_observed:
            observed_mask = c == 0
            x_o = x_l[observed_mask]
            x_prev_o = x_prev_r[observed_mask]
            len_observed = len(x_o)
            observed_time = (x_prev_o - x_o).sum()
        else:
            len_observed = 0.0
            observed_time = 0.0

        if has_left_censoring:
            left_mask = c == -1
            x_left = x_l[left_mask]
            n_left = n[left_mask]
            log_xl = np.log(x_left)
            n_log_x_left = n_left * log_xl
            n_log_x_left_sum = n_log_x_left.sum()
            x_left_sum = x_left.sum()
            n_left_sum = n_left.sum()
            n_l_factorial = gammaln(n_left + 1)
            n_l_factorial_sum = n_l_factorial.sum()
        else:
            n_log_x_left_sum = 0.0
            x_left_sum = 0.0
            n_left_sum = 0.0
            n_l_factorial_sum = 0.0

        if has_right_censoring:
            right_mask = c == 1
            x_right = x_l[right_mask]
            x_right_prev = x_prev_r[right_mask]
            right_censored_time = (x_right_prev - x_right).sum()
        else:
            right_censored_time = 0.0

        if has_interval_censoring:
            interval_mask = c == 2
            x_i_l = x_l[interval_mask]
            x_i_r = x_r[interval_mask]
            delta_xi = x_i_r - x_i_l

            x_interval_sum = delta_xi.sum()

            n_interval = n[c == 2]
            n_interval_sum = n_interval.sum()

            n_log_x_interval_sum = (n_interval * np.log(delta_xi)).sum()

            n_i_factorial = gammaln(n_interval + 1)
            n_i_factorial_sum = n_i_factorial.sum()
        else:
            x_interval_sum = 0.0
            n_interval_sum = 0.0
            n_log_x_interval_sum = 0.0
            n_i_factorial_sum = 0.0

        def negll_func(log_rate):
            rate = anp.exp(log_rate)
            ll = len_observed * log_rate + rate * observed_time
            ll += rate * right_censored_time
            ll += (
                log_rate * n_left_sum
                + n_log_x_left_sum
                - rate * x_left_sum
                - n_l_factorial_sum
            )
            ll += (
                log_rate * n_interval_sum
                + n_log_x_interval_sum
                - rate * x_interval_sum
                - n_i_factorial_sum
            )

            return -ll[0]

        return negll_func

    @classmethod
    def fit_from_recurrent_data(cls, data, init=None):
        """
        Fits the HPP model to recurrent data and returns the fitted model.

        Parameters
        ----------
        data : object
            An object containing the recurrent data.
        init : array_like, optional
            Initial parameter values for the optimization.

        Returns
        -------
        ParametricRecurrenceModel
            An object containing the fitted model and related information.
        """
        out = ParametricRecurrenceModel()
        out.dist = cls()
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)
        out.name = "Homogeneous Poisson Process"

        neg_ll = cls.create_negll_func(data)
        jac = jacobian(neg_ll)
        hess = hessian(neg_ll)

        if init is None:
            init = [0.0]
        else:
            init = np.atleast_1d(np.log(init))

        res = root(jac, init, jac=hess)
        out.res = res
        out.params = np.exp(res.x)

        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, init=None):
        """
        Fits the HPP model to the provided data and returns the fitted model.

        Parameters
        ----------
        x : array_like
            The data values.
        i : array_like, optional
            identity of the observation for each x.
        c : array_like, optional
            Censoring indicators at each x.
        n : array_like, optional
            count of the data at each x.
        init : array_like, optional
            Initial parameter estimates for the optimization.

        Returns
        -------
        object
            An object containing the fitted model and related information.
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, init=init)
