import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.parametric import Duane
from surpyval.utils.recurrent_utils import handle_xicn

from .proportional_intensity import ProportionalIntensityModel


class ProportionalIntensityNHPP:
    """
    A class representing the Proportional Intensity Non-Homogeneous Poisson
    Process (NHPP).

    The class contains methods to perform various calculations related to the
    NHPP, such as instantaneous intensity function, cumulative intensity
    function and its inverse, as well as creating the negative log-likelihood
    function and fitting the model.

    Examples
    --------

    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval import ProportionalIntensityNHPP
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = data['arrest'].values
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityNHPP.fit(x, Z, c)
    >>> model
    Proportional Intensity Recurrence Model
    =======================================
    Type                : Proportional Intensity
    Kind                : NHPP
    Parameterization    : Parametric
    Hazard Rate Model   : Crow-AMSAA
    Base Rate Parameters:
        alpha  :  31.79321229839296
        beta  :  4.980117330166502
    Covariate Coefficients:
        beta_0  :  -0.09016653537772656
        beta_1  :  0.12020391565561556
        beta_2  :  0.08804834095964903
        beta_3  :  -0.010992803410986032
        beta_4  :  -0.5059158682824993
        beta_5  :  -0.21305472747115095
        beta_6  :  0.16760620140256433
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
    def create_negll_func(self, data, dist):
        x, c, n = data.x, data.c, data.n
        Z = data.Z
        # Covariates
        x_prev = data.get_previous_x()

        has_interval_censoring = True if 2 in c else False
        has_observed = True if 0 in c else False
        has_left_censoing = True if -1 in c else False
        has_right_censoring = True if 1 in c else False

        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None

        x_prev_l = x_prev if x_prev.ndim == 1 else x[:, 0]
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else None

        # Untangle the observed data
        x_o = x_l[c == 0] if has_observed else np.array([])
        if has_interval_censoring:
            x_o_prev = x_prev_r[c == 0] if has_observed else np.array([])
        else:
            x_o_prev = x_prev_l[c == 0] if has_observed else np.array([])
        Z_o = Z[c == 0] if has_observed else np.zeros((1, Z.shape[1]))

        # Untangle the right censored data
        x_right = x_l[c == 1] if has_right_censoring else np.array([])
        if has_interval_censoring:
            x_right_prev = (
                x_prev_r[c == 1] if has_right_censoring else np.array([])
            )
        else:
            x_right_prev = (
                x_prev_l[c == 1] if has_right_censoring else np.array([])
            )
        Z_right = (
            Z[c == 1] if has_right_censoring else np.zeros((1, Z.shape[1]))
        )

        # Untangle the left censored data
        x_left = x_l[c == -1] if has_left_censoing else np.array([])
        n_left = n[c == -1] if has_left_censoing else np.array([])
        Z_left = Z[c == -1] if has_left_censoing else np.zeros((1, Z.shape[1]))

        # Untangle the interval censored data
        x_i_l = x_l[c == 2] if has_interval_censoring else np.array([])
        x_i_r = x_r[c == 2] if has_interval_censoring else np.array([])
        n_i = n[c == 2] if has_interval_censoring else np.array([])
        Z_i = (
            Z[c == 2] if has_interval_censoring else np.zeros((1, Z.shape[1]))
        )

        # Using the empty arrays avoids the need for if statements in the
        # likelihood function. It also means that the likelihood function
        # will not encounter any invalid values since taking the log of 0
        # will not occur.

        def negll_func(params):
            dist_params = params[: len(dist.param_names)]
            beta_coeffs = params[len(dist.param_names) :]
            # ll of directly observed
            phi_exponents_observed = np.dot(Z_o, beta_coeffs)
            delta_cif_o = dist.cif(x_o_prev, *dist_params) - dist.cif(
                x_o, *dist_params
            )
            # TODO: Implement log_iif functions
            ll = (
                dist.log_iif(x_o, *dist_params)
                + phi_exponents_observed
                + (np.exp(phi_exponents_observed) * delta_cif_o)
            ).sum()

            # ll of right censored
            phi_right = np.exp(np.dot(Z_right, beta_coeffs))
            delta_cif_right = dist.cif(x_right_prev, *dist_params) - dist.cif(
                x_right, *dist_params
            )
            ll += (phi_right * delta_cif_right).sum()

            # ll of left censored
            delta_cif_left = dist.cif(x_left, *dist_params)
            phi_exponents_left = np.dot(Z_left, beta_coeffs)
            phi_left = np.exp(phi_exponents_left)
            ll += (
                n_left * phi_exponents_left
                + n_left * np.log(delta_cif_left)
                - phi_left * delta_cif_left
                - gammaln(n_left + 1)
            ).sum()

            # ll of interval censored
            delta_cif_interval = dist.cif(x_i_r, *dist_params) - dist.cif(
                x_i_l, *dist_params
            )
            phi_exponents_interval = np.dot(Z_i, beta_coeffs)
            phi_interval = np.exp(phi_exponents_interval)

            ll += (
                n_i * phi_exponents_interval
                + n_i * np.log(delta_cif_interval)
                - phi_interval * delta_cif_interval
                - gammaln(n_i + 1)
            ).sum()

            return -ll

        return negll_func

    @classmethod
    def fit_from_recurrent_data(cls, data, dist, init=None):
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        init = np.ones(len(dist.param_names))

        num_covariates = data.Z.shape[1]
        init = np.append(init, np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(data, dist)

        res = minimize(
            neg_ll,
            [init],
            method="Nelder-Mead",
        )
        out.res = res
        out.params = res.x[: len(dist.param_names)]
        out.coeffs = res.x[len(dist.param_names) :]
        out.name = "Non-Homogeneous Poisson Process"
        out.kind = "NHPP"
        out.parameterization = "Parametric"
        out.param_names = dist.param_names

        return out

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, dist=Duane, init=None):
        """
        Fit the model using the provided data and initial parameters.

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
        dist : surpyval.recurrence.regression.NHPPFitter, optional
            The parametric model to use for the hazard rate.
        init : array_like, optional
            Initial parameter estimates.

        Returns
        -------

        ProportionalIntensityModel
            An object containing the results of the fitting process, including
            parameter estimates.
        """
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist, init)
