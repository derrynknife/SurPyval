import numpy as np
import numpy_indexed as npi
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.utils.recurrent_utils import handle_xicn

from .proportional_intensity import ProportionalIntensityModel


class ProportionalIntensityHPP:
    """
    A class representing the Proportional Intensity Homogeneous Poisson Process
    (HPP).

    The class contains methods to perform various calculations related to the
    HPP, such as instantaneous intensity function, cumulative intensity
    function and its inverse, as well as creating the negative log-likelihood
    function and fitting the model.

    Examples
    --------

    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval import ProportionalIntensityHPP
    >>>
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = data['arrest'].values
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityHPP.fit(x, Z, c)
    >>> model
    Proportional Intensity Recurrence Model
    =======================================
    Type                : Proportional Intensity
    Kind                : HPP
    Parameterization    : Parametric
    Hazard Rate Model   : Constant
    Base Rate Parameters:
        lambda  :  5.687176190869141
    Covariate Coefficients:
        beta_0  :  0.5666297135140156
        beta_1  :  0.031175577872654184
        beta_2  :  -0.6796807726448314
        beta_3  :  0.6602827681955281
        beta_4  :  -2.577091221089804
        beta_5  :  -0.4523927241350517
        beta_6  :  -0.016388459191916338
    """

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
    def create_negll_func(cls, data):
        x, c, n = data.x, data.c, data.n
        Z = data.Z
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
            x_o = x_l[c == 0]
            x_prev_o = x_prev_r[c == 0]
            len_observed = len(x_o)
            # Don't change the order of the subtraction
            # Doing the analytic simplification of the log-likelihood
            # shows that this is the correct order when using "+" for the
            # specific term.
            x_o = x_prev_o - x_o
            Z_o = Z[c == 0]
        else:
            x_o = 0.0
            len_observed = 0.0
            Z_o = np.zeros((1, Z.shape[1]))

        if has_right_censoring:
            x_right = x_l[c == 1]
            x_right_prev = x_prev_r[c == 1]
            x_right = x_right_prev - x_right
            Z_right = Z[c == 1]
        else:
            Z_right = np.zeros((1, Z.shape[1]))
            x_right = 0.0

        if has_left_censoring:
            x_left = x_l[c == -1]
            n_left = n[c == -1]
            log_xl = np.log(x_left)
            n_log_x_left = n_left * log_xl
            n_log_x_left_sum = n_log_x_left.sum()
            n_left_sum = n_left.sum()
            n_l_factorial = gammaln(n_left + 1)
            n_l_factorial_sum = n_l_factorial.sum()
        else:
            n_log_x_left_sum = 0.0
            x_left = 0.0
            n_left_sum = 0.0
            n_left = 0.0
            n_l_factorial_sum = 0.0
            Z_left = np.zeros((1, Z.shape[1]))

        if has_interval_censoring:
            x_i_l = x_l[c == 2]
            x_i_r = x_r[c == 2]
            delta_xi = x_i_r - x_i_l

            n_interval = n[c == 2]
            n_interval_sum = n_interval.sum()

            n_log_x_interval_sum = (n_interval * np.log(delta_xi)).sum()
            n_i_factorial_sum = gammaln(n_interval + 1).sum()
        else:
            n_interval = 0.0
            n_interval_sum = 0.0
            n_log_x_interval_sum = 0.0
            n_i_factorial_sum = 0.0
            Z_i = np.zeros((1, Z.shape[1]))
            delta_xi = 0.0

        def negll_func(params):
            log_rate = params[0]
            rate = np.exp(log_rate)
            beta_coeffs = params[1:]

            phi_exponent_observed = np.dot(Z_o, beta_coeffs)
            ll = (
                phi_exponent_observed.sum()
                + len_observed * log_rate
                + rate * (x_o * np.exp(phi_exponent_observed)).sum()
            )

            phi_right = np.exp(np.dot(Z_right, beta_coeffs))
            ll += rate * (x_right * phi_right).sum()

            phi_exponent_left = np.dot(Z_left, beta_coeffs)
            ll += (
                (n_left * phi_exponent_left).sum()
                + log_rate * n_left_sum
                + n_log_x_left_sum
                - rate * (np.exp(phi_exponent_left) * x_left).sum()
                - n_l_factorial_sum
            )

            phi_exponent_interval = np.dot(Z_i, beta_coeffs)
            ll += (
                (n_interval * phi_exponent_interval).sum()
                + log_rate * n_interval_sum
                + n_log_x_interval_sum
                - rate * (np.exp(phi_exponent_interval) * delta_xi).sum()
                - n_i_factorial_sum
            )

            return -ll

        return negll_func

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, init=None):
        """
        Fit the model using the provided data and initial parameters (if given)

        Parameters
        ----------

        x : array_like
            Input data.
        Z : array_like
            Covariate matrix.
        i : array_like, optional
            identity of the item.
        c : array_like, optional
            Censoring indicators.
        n : array_like, optional
            Number of events.
        init : array_like, optional
            Initial parameter estimates.

        Returns
        -------

        ProportionalIntensityModel
            An object containing the results of the fitting process, including
            parameter estimates.
        """
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)

        out = ProportionalIntensityModel()
        out.dist = cls
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)

        init = (data.n[data.c == 0]).sum() / npi.group_by(data.i).max(data.x)[
            1
        ].sum()

        num_covariates = Z.shape[1]
        init = np.append(np.log(init), np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(data)

        res = minimize(neg_ll, [init])
        out.res = res
        out.params = np.atleast_1d(np.exp(res.x[0]))
        out.coeffs = np.atleast_1d(res.x[1:])
        out.name = "Homogeneous Poisson Process"
        out.kind = "HPP"
        out.parameterization = "Parametric"
        # Super hacky way, but it works.
        dist = lambda x: None  # noqa: E731
        dist.name = "Constant"
        out.dist = dist

        return out
