from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import handle_xicn

from .proportional_intensity import ProportionalIntensityModel


@singleton_fitter
class ProportionalIntensityHPP:
    """
    Proportional-intensity regression on a homogeneous Poisson process:
    each item's events occur at the constant rate
    :math:`\\lambda e^{\\beta' Z}`, so its expected number of events by
    time ``x`` is :math:`\\lambda e^{\\beta' Z} x`.

    ``ProportionalIntensityHPP`` is an instance of this class. Its
    ``fit`` returns a
    :class:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel`,
    which carries the prediction methods (``cif``, ``iif``, ``inv_cif``,
    ``mcf``, simulation, inference). The ``iif``, ``cif`` and ``inv_cif``
    methods of this class are the *baseline* functions of time and rate,
    without covariates, that the fitted model calls.

    Examples
    --------

    One event (or censoring) per subject, so the fit is an exponential
    regression. In the bundled copy of the Rossi data ``arrest`` is 1 for
    a subject still free at the end of follow-up, so it is already the
    censoring flag ``c``:

    >>> import numpy as np
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.recurrent import ProportionalIntensityHPP
    >>>
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = data['arrest'].values
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
        lambda  :  0.017410386679243283
    <BLANKLINE>
    Covariate Coefficients:
       beta_0  :  -0.36626406174463233
       beta_1  :  -0.05559822615498945
       beta_2  :  0.30493957739153305
       beta_3  :  -0.14674549077957214
       beta_4  :  -0.4269861228181052
       beta_5  :  -0.08264790408652863
       beta_6  :  0.08565920858626697
    <BLANKLINE>
    >>> model.cif(52, Z[:1])
    array([0.32584698])
    """

    # Display name of the (constant) baseline hazard rate model, used by
    # ``ProportionalIntensityModel``'s repr via ``dist.name``.
    name = "Constant"

    def iif(self, x: ArrayLike, rate: ArrayLike) -> ArrayLike:
        return np.ones_like(np.asarray(x, dtype=float)) * rate

    def cif(self, x: ArrayLike, rate: ArrayLike) -> ArrayLike:
        return rate * np.asarray(x, dtype=float)

    def inv_cif(self, cif: ArrayLike, rate: ArrayLike) -> ArrayLike:
        return np.asarray(cif, dtype=float) / rate

    def create_negll_func(self, data: Any) -> Callable:
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
            len_observed = 0
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
            # interval data implies 2-D x, so the right column exists
            assert x_r is not None
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

        def negll_func(params: np.ndarray) -> float:
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
        self,
        x: ArrayLike,
        Z: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        t: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
        tr: "ArrayLike | None" = None,
        init: "ArrayLike | None" = None,
    ) -> Any:
        """
        Fit the model using the provided data and initial parameters (if given)

        Parameters
        ----------

        x : array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``).
        Z : array_like
            Covariate matrix, one row per row of ``x``. Each row's
            covariates apply over the interval from the item's previous row
            to this one.
        i : array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c : array_like, optional
            Censoring indicators: 0 an observed event, 1 the right-censored
            end of an item's observation, -1 left-censored and 2
            interval-censored counts (with ``n``). Defaults to all observed.
        n : array_like, optional
            Number of events in each row (for left- and interval-censored
            counts). Defaults to 1.
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
            Initial parameter estimates: the baseline rate followed by the
            covariate coefficients.

        Returns
        -------

        ProportionalIntensityModel
            An object containing the results of the fitting process, including
            parameter estimates.
        """
        data = handle_xicn(x, i, c, n, t=t, tl=tl, tr=tr, Z=Z)
        return self.fit_from_recurrent_data(data, init=init)

    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any = None,
        init: "ArrayLike | None" = None,
    ) -> Any:
        """
        Fit from a prepared
        :class:`~surpyval.utils.recurrent_event_data.RecurrentEventData`
        (with covariates attached), as built by ``surpyval.handle_xicn``.
        :meth:`fit` builds one from its arrays and calls this.

        Parameters
        ----------

        data : RecurrentEventData
            The recurrent event data, including ``Z``.
        dist : optional
            Accepted for a uniform interface with the NHPP fitter but
            ignored: the homogeneous baseline is this fitter's own
            constant-rate model.
        init : array_like, optional
            Initial parameter estimates, as for :meth:`fit`.

        Returns
        -------

        ProportionalIntensityModel
            The fitted model.
        """
        out = ProportionalIntensityModel()
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)

        num_covariates = data.Z.shape[1]
        if init is None:
            # Use the right endpoint for interval-censored (2D) observations
            # when estimating each item's latest event time for the initial
            # rate guess.
            _x_max = data.x if data.x.ndim == 1 else data.x[:, 1]
            _, _inv = np.unique(data.i, return_inverse=True)
            _max_x = np.full(_inv.max() + 1, -np.inf)
            np.maximum.at(_max_x, _inv, _x_max)
            rate = (data.n[data.c == 0]).sum() / _max_x.sum()
            init = np.append(np.log(rate), np.zeros(num_covariates))
        else:
            # User-supplied starting values were previously overwritten
            # unconditionally (#288). The first value is the baseline
            # rate on its natural scale; optimisation runs on log(rate).
            init = np.atleast_1d(np.asarray(init, dtype=float))
            if init.size != 1 + num_covariates:
                raise ValueError(
                    f"init must have {1 + num_covariates} values (baseline "
                    f"rate + {num_covariates} coefficients); got {init.size}."
                )
            init = np.append(np.log(init[0]), init[1:])

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
        # Keep a reference to this fitter so the Cramer-von Mises bootstrap can
        # refit the full regression model per replicate (``_fitter_dist`` is
        # None: the homogeneous baseline is this fitter itself).
        out._fitter = self
        out._fitter_dist = None

        return out
