from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent._convergence import better_result, warn_unconverged
from surpyval.recurrent.inference import bic_sample_size
from surpyval.recurrent.parametric import Duane
from surpyval.recurrent.parametric.counting_process import CountingProcess
from surpyval.recurrent.parametric.nhpp_fitter import nhpp_log_likelihood
from surpyval.utils.dataframe import RecurrentRegressionDataFrameMixin
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import handle_xicn, validate_nhpp_data

from .proportional_intensity import (
    ProportionalIntensityModel,
    alias_covariates,
)


@singleton_fitter
class ProportionalIntensityNHPP(RecurrentRegressionDataFrameMixin):
    """
    Proportional-intensity regression on a non-homogeneous Poisson
    process: each item's intensity is a parametric baseline intensity
    scaled by its covariates,

    .. math::
        \\lambda(t \\mid Z) = \\lambda_0(t)\\, e^{\\beta' Z},

    with the baseline any NHPP model -- ``Duane`` (the default),
    ``CrowAMSAA`` or ``CoxLewis`` -- chosen with ``dist``.

    ``ProportionalIntensityNHPP`` is an instance of this class. Its
    ``fit`` returns a
    :class:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel`,
    which carries the prediction methods (``cif``, ``iif``, ``inv_cif``,
    ``mcf``), simulation and inference.

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

        # Covariate rows gathered with the same masks; the zeros((1, p))
        # placeholders keep the dot products defined when a censoring
        # type is absent (the matching x arrays are empty, so the terms
        # vanish in the sums).
        p_cov = Z.shape[1]
        Z_pieces = {
            key: Z[mask] if mask.any() else np.zeros((1, p_cov))
            for key, mask in [
                ("o", s["mask_o"]),
                ("right", s["mask_right"]),
                ("left", s["mask_left"]),
                ("i", s["mask_i"]),
            ]
        }
        Z_pieces["close"] = Z[s["close_idx"]]
        k_dist = len(dist.parameter_names)

        def negll_func(params: np.ndarray) -> float:
            dist_params = params[:k_dist]
            beta_coeffs = params[k_dist:]
            eta = {
                key: np.dot(Z_piece, beta_coeffs)
                for key, Z_piece in Z_pieces.items()
            }
            return -nhpp_log_likelihood(
                lambda x: dist.cif(x, *dist_params),
                lambda x: dist.log_iif(x, *dist_params),
                s,
                eta,
            )

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
        fallback = np.ones(len(dist.parameter_names))
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
        """
        Fit from a prepared
        :class:`~surpyval.utils.recurrent_event_data.RecurrentEventData`
        (with covariates attached), as built by
        ``surpyval.handle_xicn``. :meth:`fit` builds one from its arrays and
        calls this.

        Parameters
        ----------

        data : RecurrentEventData
            The recurrent event data, including ``Z``.
        dist : CountingProcess
            The baseline intensity model, as for :meth:`fit`.
        init : array_like, optional
            Initial parameter estimates, as for :meth:`fit`.

        Returns
        -------

        ProportionalIntensityModel
            The fitted model.
        """
        if not isinstance(dist, CountingProcess):
            raise TypeError(
                "`dist` must be a CountingProcess instance "
                "(e.g. Duane, CrowAMSAA, CoxLewis); got {!r}".format(dist)
            )
        validate_nhpp_data(data, dist)
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        num_covariates = data.Z.shape[1]
        expected = len(dist.parameter_names) + num_covariates

        def default_init() -> np.ndarray:
            return np.append(
                self._baseline_start(data, dist), np.zeros(num_covariates)
            )

        user_init = init is not None
        if init is None:
            init = default_init()
        else:
            # User-supplied starting values were previously overwritten
            # unconditionally (#288).
            init = np.atleast_1d(np.asarray(init, dtype=float))
            if init.size != expected:
                raise ValueError(
                    f"init must have {expected} values "
                    f"({len(dist.parameter_names)} baseline parameters + "
                    f"{num_covariates} coefficients); got {init.size}."
                )

        neg_ll = self.create_negll_func(data, dist)

        # A coefficient the data cannot determine is held at 0 and reported
        # as nan (#502). A constant column is one only where the baseline
        # has a scale, which is then the intercept.
        n_dist = len(dist.parameter_names)
        aliased = alias_covariates(
            data.Z, intercept=getattr(dist, "has_scale", False)
        )
        free = np.ones(expected, dtype=bool)
        free[n_dist + aliased] = False

        # Search on an unconstrained scale: a baseline parameter bounded
        # below (Duane's b, Crow-AMSAA's alpha and beta) is optimised as the
        # log of its distance from the bound. Nelder-Mead on the natural
        # scale, with parameters differing by orders of magnitude, stopped
        # well short of the optimum on as few as nine parameters. A
        # gradient search does the work and Nelder-Mead polishes it.
        bounds = list(dist.bounds) + [(None, None)] * num_covariates
        bounds = [b for b, keep in zip(bounds, free) if keep]
        to_natural, to_search = unconstraining_maps(bounds)

        def full(values: np.ndarray) -> np.ndarray:
            params = np.zeros(expected)
            params[free] = values
            return params

        def objective(u: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                value = neg_ll(full(to_natural(u)))
            return float(value) if np.isfinite(value) else 1e300

        def search(start: np.ndarray) -> Any:
            res = minimize(objective, to_search(start[free]), method="BFGS")
            return minimize(
                objective,
                res.x,
                method="Nelder-Mead",
                options={
                    "maxfev": 2000 * int(free.sum()),
                    "xatol": 1e-8,
                    "fatol": 1e-10,
                },
            )

        res = search(init)
        # A start the user gave is followed by the default one, and the
        # better answer kept: from Duane's alpha x1e6 the intensity
        # overflows, the search cannot move, and the start was returned
        # in silence (#429).
        if user_init:
            res = better_result(res, search(default_init()))
        if not (res.success and res.fun < 1e300):
            warn_unconverged("The proportional intensity fit")
        res.x = to_natural(res.x)
        out.res = res
        fitted = np.where(free, full(res.x), np.nan)
        out.params = fitted[:n_dist]
        out.coeffs = fitted[n_dist:]
        out.name = "Non-Homogeneous Poisson Process"
        out.kind = "NHPP"
        out.parameterization = "Parametric"
        out._rate_names = list(dist.parameter_names)
        # Keep a reference to this fitter and the baseline so the Cramer-von
        # Mises bootstrap can refit the full regression model per replicate.
        out._fitter = self
        out._fitter_dist = dist
        # The likelihood is in natural parameter space, so the full fitted
        # vector ``[*dist_params, *coeffs]`` is the MLE the shared inference
        # machinery needs for AIC/BIC/standard errors.
        out._neg_ll = neg_ll
        out._mle = fitted
        out._n_obs = bic_sample_size(data)

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
            The event times, pooled over items (each row belongs to the item
            named in ``i``).
        Z : array_like or dict
            Covariates: a matrix with one row per row of ``x`` (a 1-D array
            is a single covariate), or a ``{item: covariates}`` dict. They
            describe the item (they are static), so they must be the same
            on every row of an item; values that change within an item
            raise a ``ValueError``.
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
            Left truncation (delayed entry) time of each item; the
            observation of each item begins here. A scalar applies to every
            item; an array has one value per row (the same on every row of
            an item).
        tr : array_like or scalar, optional
            Right truncation time of each item, given like ``tl``; the
            observation window closes here,
            so the baseline intensity is integrated out to ``tr`` even without
            an explicit right-censoring (``c=1``) row.
        dist : CountingProcess, optional
            The baseline intensity model: ``Duane`` (the default),
            ``CrowAMSAA`` or ``CoxLewis`` from ``surpyval.recurrent``. With
            ``HPP`` the model is the same as ``ProportionalIntensityHPP``.
        init : array_like, optional
            Initial parameter estimates: the baseline parameters followed by
            the covariate coefficients.

        Returns
        -------

        ProportionalIntensityModel
            An object containing the results of the fitting process, including
            parameter estimates.
        """
        data = handle_xicn(x, i, c, n, t=t, tl=tl, tr=tr, Z=Z)
        return self.fit_from_recurrent_data(data, dist, init)
