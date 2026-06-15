import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrent.parametric import Duane
from surpyval.recurrent.parametric.counting_process import CountingProcess
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import handle_xicn

from .proportional_intensity import ProportionalIntensityModel


@singleton_fitter
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

    >>> import numpy as np
    >>> from surpyval.recurrent import ProportionalIntensityNHPP
    >>>
    >>> # Four repairable systems observed until t=20; failures get more
    >>> # frequent over time and the Z=1 group fails faster than the Z=0 group.
    >>> x = [9, 14, 18, 20,
    ...      7, 12, 16, 19, 20,
    ...      5, 9, 13, 16, 18, 20,
    ...      6, 10, 13, 15, 17, 19, 20]
    >>> i = [1, 1, 1, 1,
    ...      2, 2, 2, 2, 2,
    ...      3, 3, 3, 3, 3, 3,
    ...      4, 4, 4, 4, 4, 4, 4]
    >>> # c = 0 is an observed failure, c = 1 the right-censored window close
    >>> c = [0, 0, 0, 1,
    ...      0, 0, 0, 0, 1,
    ...      0, 0, 0, 0, 0, 1,
    ...      0, 0, 0, 0, 0, 0, 1]
    >>> Z = np.array([0, 0, 0, 0,
    ...               0, 0, 0, 0, 0,
    ...               1, 1, 1, 1, 1, 1,
    ...               1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    >>> model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c)
    >>> model
    Proportional Intensity Recurrence Model
    =======================================
    Type                : Proportional Intensity
    Kind                : NHPP
    Parameterization    : Parametric
    Hazard Rate Model   : Duane
    Base Rate Parameters:
        alpha  :  2.0294701249769567
        b  :  0.008010947012813689
    <BLANKLINE>
    Covariate Coefficients:
       beta_0  :  0.45194475814452534
    <BLANKLINE>
    """

    def iif(self, x, rate):
        return np.ones_like(x) * rate

    def cif(self, x, rate):
        return rate * x

    def inv_cif(self, cif, rate):
        return cif / rate

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

    def fit_from_recurrent_data(self, data, dist, init=None):
        if not isinstance(dist, CountingProcess):
            raise TypeError(
                "`dist` must be a CountingProcess instance "
                "(e.g. Duane, CrowAMSAA, CoxLewis); got {!r}".format(dist)
            )
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        init = np.ones(len(dist.param_names))

        num_covariates = data.Z.shape[1]
        init = np.append(init, np.zeros(num_covariates))

        neg_ll = self.create_negll_func(data, dist)

        res = minimize(
            neg_ll,
            init,
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

    def fit(self, x, Z, i=None, c=None, n=None, dist=Duane, init=None):
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
        dist : surpyval.recurrent.regression.NHPPFitter, optional
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
        return self.fit_from_recurrent_data(data, dist, init)
