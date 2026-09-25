from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
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

    def create_negll_func(self, data: Any, dist: Any) -> Callable:
        Z = data.Z
        s = data.split_for_nhpp_likelihood()
        x_o, x_o_prev = s["x_o"], s["x_o_prev"]
        x_right, x_right_prev = s["x_right"], s["x_right_prev"]
        x_left, n_left = s["x_left"], s["n_left"]
        x_i_l, x_i_r, n_i = s["x_i_l"], s["x_i_r"], s["n_i"]
        x_close_last, x_close_tr = s["x_close_last"], s["x_close_tr"]

        # Covariate rows gathered with the same masks; the zeros((1, p))
        # placeholders keep the dot products defined when a censoring
        # type is absent (the matching x arrays are empty, so the terms
        # vanish in the sums).
        p_cov = Z.shape[1]
        Z_o = Z[s["mask_o"]] if s["mask_o"].any() else np.zeros((1, p_cov))
        Z_right = (
            Z[s["mask_right"]]
            if s["mask_right"].any()
            else np.zeros((1, p_cov))
        )
        Z_left = (
            Z[s["mask_left"]] if s["mask_left"].any() else np.zeros((1, p_cov))
        )
        Z_i = Z[s["mask_i"]] if s["mask_i"].any() else np.zeros((1, p_cov))
        Z_close = Z[s["close_idx"]]

        # Using the empty arrays avoids the need for if statements in the
        # likelihood function. It also means that the likelihood function
        # will not encounter any invalid values since taking the log of 0
        # will not occur.

        def negll_func(params: np.ndarray) -> float:
            dist_params = params[: len(dist.param_names)]
            beta_coeffs = params[len(dist.param_names) :]
            # ll of directly observed
            phi_exponents_observed = np.dot(Z_o, beta_coeffs)
            delta_cif_o = dist.cif(x_o_prev, *dist_params) - dist.cif(
                x_o, *dist_params
            )
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

            # right window-close: extend each item's integral to its tr
            phi_close = np.exp(np.dot(Z_close, beta_coeffs))
            delta_cif_close = dist.cif(x_close_last, *dist_params) - dist.cif(
                x_close_tr, *dist_params
            )
            ll += (phi_close * delta_cif_close).sum()

            return -ll

        return negll_func

    @staticmethod
    def _baseline_start(data: Any, dist: Any) -> np.ndarray:
        """Default baseline start: the covariate-free fit of ``dist``.

        Starting every baseline parameter at one put Duane's ``b`` -- often
        1e-3 or smaller -- orders of magnitude from its optimum, and
        Nelder-Mead stopped short of it while reporting success (AICs
        5-40 worse than the same model fitted as Crow-AMSAA). The fit that
        ignores the covariates is the natural start, with coefficients 0;
        if it fails, the old unit start is used.
        """
        fallback = np.ones(len(dist.param_names))
        try:
            with np.errstate(all="ignore"):
                base = dist.fit_from_recurrent_data(data)
            start = np.asarray(base.params, dtype=float)
        except Exception:
            return fallback
        if start.shape != fallback.shape or not np.all(np.isfinite(start)):
            return fallback
        for value, (low, high) in zip(start, dist.bounds):
            if (low is not None and value <= low) or (
                high is not None and value >= high
            ):
                return fallback
        return start

    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any,
        init: "ArrayLike | None" = None,
    ) -> Any:
        if not isinstance(dist, CountingProcess):
            raise TypeError(
                "`dist` must be a CountingProcess instance "
                "(e.g. Duane, CrowAMSAA, CoxLewis); got {!r}".format(dist)
            )
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        num_covariates = data.Z.shape[1]
        expected = len(dist.param_names) + num_covariates
        if init is None:
            init = np.append(
                self._baseline_start(data, dist), np.zeros(num_covariates)
            )
        else:
            # User-supplied starting values were previously overwritten
            # unconditionally (#288).
            init = np.atleast_1d(np.asarray(init, dtype=float))
            if init.size != expected:
                raise ValueError(
                    f"init must have {expected} values "
                    f"({len(dist.param_names)} baseline parameters + "
                    f"{num_covariates} coefficients); got {init.size}."
                )

        neg_ll = self.create_negll_func(data, dist)

        # Search on an unconstrained scale: a baseline parameter bounded
        # below (Duane's b, Crow-AMSAA's alpha and beta) is optimised as the
        # log of its distance from the bound. Nelder-Mead on the natural
        # scale, with parameters differing by orders of magnitude, stopped
        # well short of the optimum on as few as nine parameters. A
        # gradient search does the work and Nelder-Mead polishes it.
        bounds = list(dist.bounds) + [(None, None)] * num_covariates
        to_natural, to_search = _unconstraining_maps(bounds)

        def objective(u: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                value = neg_ll(to_natural(u))
            return float(value) if np.isfinite(value) else 1e300

        u0 = to_search(init)
        res = minimize(objective, u0, method="BFGS")
        res = minimize(
            objective,
            res.x,
            method="Nelder-Mead",
            options={
                "maxfev": 2000 * expected,
                "xatol": 1e-8,
                "fatol": 1e-10,
            },
        )
        res.x = to_natural(res.x)
        out.res = res
        out.params = res.x[: len(dist.param_names)]
        out.coeffs = res.x[len(dist.param_names) :]
        out.name = "Non-Homogeneous Poisson Process"
        out.kind = "NHPP"
        out.parameterization = "Parametric"
        out.param_names = dist.param_names
        # Keep a reference to this fitter and the baseline so the Cramer-von
        # Mises bootstrap can refit the full regression model per replicate.
        out._fitter = self
        out._fitter_dist = dist
        # The likelihood is in natural parameter space, so the full fitted
        # vector ``[*dist_params, *coeffs]`` is the MLE the shared inference
        # machinery needs for AIC/BIC/standard errors.
        out._neg_ll = neg_ll
        out._mle = np.asarray(res.x, dtype=float)
        out._n_obs = len(data.x)

        return out

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
        dist: Any = Duane,
        init: "ArrayLike | None" = None,
    ) -> Any:
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
        t : array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl : array_like or scalar, optional
            Left truncation (delayed entry) time per item; the observation of
            each item begins here. Scalar broadcasts to all items.
        tr : array_like or scalar, optional
            Right truncation time per item; the observation window closes here,
            so the baseline intensity is integrated out to ``tr`` even without
            an explicit right-censoring (``c=1``) row.
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
        data = handle_xicn(x, i, c, n, t=t, tl=tl, tr=tr, Z=Z)
        return self.fit_from_recurrent_data(data, dist, init)


def _unconstraining_maps(bounds: list) -> tuple[Callable, Callable]:
    """Maps between natural parameters and an unconstrained search space.

    ``(low, None)`` becomes ``low + exp(u)``, ``(None, high)`` becomes
    ``high - exp(u)``, a finite ``(low, high)`` a logistic between them,
    and ``(None, None)`` is left alone. Starting values on or outside a
    bound are nudged inside it.
    """
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]

    def to_natural(u: np.ndarray) -> np.ndarray:
        out = np.array(u, dtype=float)
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                out[k] = low + (high - low) / (1.0 + np.exp(-u[k]))
            elif low is not None:
                out[k] = low + np.exp(u[k])
            elif high is not None:
                out[k] = high - np.exp(u[k])
        return out

    def to_search(v: np.ndarray) -> np.ndarray:
        out = np.array(v, dtype=float)
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                frac = np.clip((v[k] - low) / (high - low), 1e-12, 1 - 1e-12)
                out[k] = np.log(frac / (1.0 - frac))
            elif low is not None:
                out[k] = np.log(max(v[k] - low, 1e-12 * max(1.0, abs(low))))
            elif high is not None:
                out[k] = np.log(max(high - v[k], 1e-12 * max(1.0, abs(high))))
        return out

    return to_natural, to_search
