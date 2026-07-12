import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import handle_xicn

from .proportional_intensity import ProportionalIntensityModel


@singleton_fitter
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

    >>> import numpy as np
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.recurrent import ProportionalIntensityHPP
    >>>
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> # 'arrest' == 1 is an observed event (c=0); 0 is right-censored (c=1)
    >>> c = np.where(data['arrest'].values == 1, 0, 1)
    >>> i = np.arange(len(data))
    >>> Z = data[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].values
    >>> model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    >>> model
    Proportional Intensity Recurrence Model
    =======================================
    Type                : Proportional Intensity
    Kind                : HPP
    Parameterization    : Parametric
    Hazard Rate Model   : Constant
    Base Rate Parameters:
        lambda  :  0.012395105741757225
    <BLANKLINE>
    Covariate Coefficients:
       beta_0  :  0.06397367067847898
       beta_1  :  0.011491178797116433
       beta_2  :  -0.02147901865302258
       beta_3  :  -0.014676664859873595
       beta_4  :  0.04738275470382102
       beta_5  :  0.019109337775930837
       beta_6  :  -0.01905662860143533
    <BLANKLINE>
    """

    # Display name of the (constant) baseline hazard rate model, used by
    # ``ProportionalIntensityModel``'s repr via ``dist.name``.
    name = "Constant"

    def iif(self, x, rate):
        return np.ones_like(np.asarray(x, dtype=float)) * rate

    def cif(self, x, rate):
        return rate * np.asarray(x, dtype=float)

    def inv_cif(self, cif, rate):
        return np.asarray(cif, dtype=float) / rate

    def create_negll_func(self, data):
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
            Z_left = Z[c == -1]
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
            Z_i = Z[c == 2]

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

        # Right window-close: for items with a finite right-truncation time
        # ``tr`` the integral closes at ``tr``. For the constant-rate HPP the
        # extension contributes rate * phi * (x_last - tr). Empty for
        # untruncated data.
        x_close_last, x_close_tr, close_idx = data.get_right_truncation_close()
        x_close = x_close_last - x_close_tr
        Z_close = Z[close_idx]

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

            phi_close = np.exp(np.dot(Z_close, beta_coeffs))
            ll += rate * (x_close * phi_close).sum()

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

    def fit(
        self, x, Z, i=None, c=None, n=None, t=None, tl=None, tr=None, init=None
    ):
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
        t : array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl : array_like or scalar, optional
            Left truncation (delayed entry) time per item; the observation of
            each item begins here. Scalar broadcasts to all items.
        tr : array_like or scalar, optional
            Right truncation time per item; the observation window closes here,
            so the intensity is integrated out to ``tr`` even without an
            explicit right-censoring (``c=1``) row.
        init : array_like, optional
            Initial parameter estimates.

        Returns
        -------

        ProportionalIntensityModel
            An object containing the results of the fitting process, including
            parameter estimates.
        """
        data = handle_xicn(
            x, i, c, n, t=t, tl=tl, tr=tr, Z=Z, as_recurrent_data=True
        )

        out = ProportionalIntensityModel()
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)

        # Use the right endpoint for interval-censored (2D) observations when
        # estimating each item's latest event time for the initial rate guess.
        _x_max = data.x if data.x.ndim == 1 else data.x[:, 1]
        _, _inv = np.unique(data.i, return_inverse=True)
        _max_x = np.full(_inv.max() + 1, -np.inf)
        np.maximum.at(_max_x, _inv, _x_max)
        init = (data.n[data.c == 0]).sum() / _max_x.sum()

        num_covariates = Z.shape[1]
        init = np.append(np.log(init), np.zeros(num_covariates))

        neg_ll = self.create_negll_func(data)

        res = minimize(neg_ll, init)
        out.res = res
        out.params = np.atleast_1d(np.exp(res.x[0]))
        out.coeffs = np.atleast_1d(res.x[1:])
        out.name = "Homogeneous Poisson Process"
        out.kind = "HPP"
        out.parameterization = "Parametric"
        # ``neg_ll`` is parameterised by ``log_rate``; expose it in natural
        # (rate) space so ``_neg_ll(_mle)`` works with ``_mle`` the fitted rate
        # and covariate coefficients.
        out._neg_ll = lambda p: neg_ll(np.concatenate([[np.log(p[0])], p[1:]]))
        out._mle = np.concatenate([out.params, out.coeffs])
        out._n_obs = len(data.x)
        # The baseline hazard is this fitter's own constant-rate model, so the
        # fitted model's ``cif``/``iif``/``inv_cif`` (and everything built on
        # them: simulation, ``cif_cb``, ``plot``) delegate back to it.
        out.dist = self

        return out
