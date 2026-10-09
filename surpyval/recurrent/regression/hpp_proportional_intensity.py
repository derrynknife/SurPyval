from __future__ import annotations

import functools
from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrent._convergence import better_result
from surpyval.recurrent.inference import bic_sample_size
from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.univariate.regression._aliasing import (
    dataframe_covariates,
    fit_columns,
)
from surpyval.utils.covariates import coefficient_floor
from surpyval.utils.dataframe import RecurrentRegressionDataFrameMixin
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_utils import handle_xicn, validate_nhpp_data

from .proportional_intensity import (
    ProportionalIntensityModel,
    alias_covariates,
)


def _in_rate_space(neg_ll: Callable, p: np.ndarray) -> Any:
    """``neg_ll``, which takes ``[log(rate), *coefficients]``, at
    ``p = [rate, *coefficients]``."""
    return neg_ll(np.concatenate([[np.log(p[0])], p[1:]]))


@singleton_fitter
class ProportionalIntensityHPP(FitterRepr, RecurrentRegressionDataFrameMixin):
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
    regression. In the Rossi data ``arrest`` is 1 for a subject arrested
    during follow-up, so the censoring flag is ``c = 1 - arrest``:

    >>> import numpy as np
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.recurrent import ProportionalIntensityHPP
    >>>
    >>> data = load_rossi_static()
    >>> x = data['week'].values
    >>> c = 1 - data['arrest'].values
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
       coef_0  :  -0.36626406174463233
       coef_1  :  -0.05559822615498945
       coef_2  :  0.30493957739153305
       coef_3  :  -0.14674549077957214
       coef_4  :  -0.4269861228181052
       coef_5  :  -0.08264790408652863
       coef_6  :  0.08565920858626697
    <BLANKLINE>
    >>> model.cif(52, Z[:1])
    np.float64(0.32584697690680187)
    """

    #: The ``repr`` (#614)
    fitter_kind = "proportional intensity fitter"

    def _repr_name(self) -> str:
        return "ProportionalIntensityHPP"

    def _repr_details(self) -> "list[str]":
        return ["HPP baseline"]

    # Display name of the (constant) baseline hazard rate model, used by
    # ``ProportionalIntensityModel``'s repr via ``dist.name``.
    name = "Constant"

    def iif(self, x: ArrayLike, rate: ArrayLike) -> ArrayLike:
        # NaN at a missing time (it was the rate there, #382).
        return (
            np.where(np.isnan(np.asarray(x, dtype=float)), np.nan, 1.0) * rate
        )

    def cif(self, x: ArrayLike, rate: ArrayLike) -> ArrayLike:
        return rate * np.asarray(x, dtype=float)

    def inv_cif(self, cif: ArrayLike, rate: ArrayLike) -> ArrayLike:
        return np.asarray(cif, dtype=float) / rate

    @staticmethod
    def _default_start(data: Any) -> np.ndarray:
        """The default start: the rate of observed events per unit of
        follow-up, ``log``-transformed as the search runs, and every
        coefficient 0."""
        # Use the right endpoint for interval-censored (2D) observations
        # when estimating each item's latest event time for the initial
        # rate guess.
        _x_max = data.x if data.x.ndim == 1 else data.x[:, 1]
        _, _inv = np.unique(data.i, return_inverse=True)
        _max_x = np.full(_inv.max() + 1, -np.inf)
        onp.maximum.at(_max_x, _inv, _x_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = (data.n[data.c == 0]).sum() / _max_x.sum()
        if not (np.isfinite(rate) and rate > 0):
            # e.g. only counts (c=-1 / c=2) and no exact events
            rate = 1.0
        return np.append(np.log(rate), np.zeros(data.Z.shape[1]))

    def create_negll_func(self, data: Any) -> Callable:
        Z = data.Z
        # The pieces of the NHPP likelihood (#350): per censoring type, the
        # times and previous times, the row masks to gather the covariates
        # with, and the right-truncation window close.
        s = data.split_for_nhpp_likelihood()
        p_cov = Z.shape[1]

        def rows(mask: np.ndarray) -> np.ndarray:
            # A zeros((1, p)) placeholder keeps the dot products defined
            # when a censoring type is absent (its times are empty, so its
            # terms vanish from the sums).
            return Z[mask] if mask.any() else np.zeros((1, p_cov))

        # The HPP's cumulative intensity is rate * x: the sums the data fix
        # are taken once here, the covariate terms in the likelihood.
        len_observed = len(s["x_o"])
        # Don't change the order of the subtraction: the analytic
        # simplification of the log-likelihood shows that this is the
        # correct order when using "+" for the term.
        x_o = s["x_o_prev"] - s["x_o"]
        Z_o = rows(s["mask_o"])

        x_right = s["x_right_prev"] - s["x_right"]
        Z_right = rows(s["mask_right"])

        # The count covers the item's window from its entry (``tl``, or the
        # origin 0) -- the row is the item's first, so its previous time is
        # that entry.
        x_left = s["x_left"] - s["x_left_prev"]
        n_left = s["n_left"]
        Z_left = rows(s["mask_left"])
        n_log_x_left_sum = (n_left * np.log(x_left)).sum()
        n_left_sum = n_left.sum()
        n_l_factorial_sum = gammaln(n_left + 1).sum()

        delta_xi = s["x_i_r"] - s["x_i_l"]
        n_interval = s["n_i"]
        Z_i = rows(s["mask_i"])
        n_interval_sum = n_interval.sum()
        n_log_x_interval_sum = (n_interval * np.log(delta_xi)).sum()
        n_i_factorial_sum = gammaln(n_interval + 1).sum()

        # Right window-close: for items with a finite right-truncation time
        # ``tr`` the integral closes at ``tr``. For the constant-rate HPP the
        # extension contributes rate * phi * (x_last - tr). Empty for
        # untruncated data.
        x_close = s["x_close_last"] - s["x_close_tr"]
        Z_close = Z[s["close_idx"]]

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

    @dataframe_covariates
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
            Number of events in each row: a count on a left- or
            interval-censored row (``c=-1`` or ``c=2``). An exact event
            (``c=0``) and an end-of-observation row (``c=1``) stand for one,
            so ``n > 1`` there is refused: repeat the row for simultaneous
            events. Defaults to 1.
        t : array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl : array_like or scalar, optional
            Left truncation (delayed entry) time of each item; the
            observation of each item begins here. A scalar applies to every
            item; an array has one value per row (the same on every row of
            an item) or one value per item, in the sorted order of the item
            ids (``np.unique(i)``; read per row when there are as many rows
            as items).
        tr : array_like or scalar, optional
            Right truncation time of each item, given like ``tl``; the
            observation window closes here,
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
        # The covariates' columns name the coefficients (#614)
        out.feature_names = fit_columns()

        out._rate_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (-np.inf, np.inf)

        # With no events the likelihood only rewards a lower rate: the fit
        # ran to a rate of 0 with log(0) warnings.
        validate_nhpp_data(data, self)
        num_covariates = data.Z.shape[1]
        user_init = init is not None
        if init is None:
            start = self._default_start(data)
        else:
            # User-supplied starting values were previously overwritten
            # unconditionally (#288). The first value is the baseline
            # rate on its natural scale; optimisation runs on log(rate).
            given = onp.atleast_1d(onp.asarray(init, dtype=float))
            if given.size != 1 + num_covariates:
                raise ValueError(
                    f"init must have {1 + num_covariates} values (baseline "
                    f"rate + {num_covariates} coefficients); got {given.size}."
                )
            if not (onp.isfinite(given[0]) and given[0] > 0):
                raise ValueError(
                    "the baseline rate in init must be positive and finite; "
                    f"got {float(given[0])}"
                )
            start = onp.append(onp.log(given[0]), given[1:])

        neg_ll = self.create_negll_func(data)

        # A coefficient the data cannot determine is held at 0 and reported
        # as nan (#502); the rate is the intercept.
        aliased = alias_covariates(data.Z, intercept=True)
        free = np.ones(1 + num_covariates, dtype=bool)
        free[1 + aliased] = False

        def full(values: np.ndarray) -> np.ndarray:
            # (Built by concatenation, so autograd can differentiate it.)
            parts, k = [], 0
            for is_free in free:
                parts.append(values[k : k + 1] if is_free else np.zeros(1))
                k += int(is_free)
            return np.concatenate(parts)

        def neg_ll_free(values: np.ndarray) -> float:
            return neg_ll(full(values))

        def objective(values: np.ndarray) -> float:
            value = neg_ll_free(values)
            return float(value) if np.isfinite(value) else 1e300

        def search(start: np.ndarray) -> Any:
            # From a poor start exp(beta'Z) overflows: the search sees a
            # large finite value there, and no raw warning escapes (from
            # the likelihood or from BFGS's update with those values).
            with np.errstate(all="ignore"):
                return minimize(objective, start[free])

        res = search(start)
        # A start the user gave is followed by the default one, and the
        # better answer kept, as for the NHPP fit (#429, #554): from a
        # coefficient of 5 on the Rossi data exp(beta'Z) overflows, the
        # search cannot move, and the start was returned in silence.
        if user_init:
            res = better_result(res, search(self._default_start(data)))
        # The answer is kept only as a verified maximum (zero gradient,
        # negative-definite Hessian of the log-likelihood), polished where
        # it is not one, each coefficient in its own covariate's units
        # (#577). BFGS's own verdict is no test: it reports a "precision
        # loss" at the maximum of the Rossi fit, and success where it
        # never moved; and its absolute tolerance stopped a covariate in
        # millionths 0.06 short of the maximum.
        n_obs = bic_sample_size(data)
        floor = coefficient_floor(
            int(free.sum()),
            [
                (int(free[:j].sum()), j - 1)
                for j in range(1, 1 + num_covariates)
                if free[j]
            ],
            data.Z,
        )
        verified = False
        if res.fun < 1e300:
            res, verified = verify_or_polish(
                neg_ll_free,
                res,
                max(float(n_obs), 1.0),
                floor=floor,
            )
        out.maximum = "verified" if verified else "unverified"
        if not verified:
            warn_unverified("The proportional intensity fit")
        out.res = res
        fitted = np.full(1 + num_covariates, np.nan)
        fitted[free] = res.x
        out.params = np.atleast_1d(np.exp(fitted[0]))
        out.coeffs = np.atleast_1d(fitted[1:])
        out.name = "Homogeneous Poisson Process"
        out.kind = "HPP"
        out.parameterization = "Parametric"
        # ``neg_ll`` is parameterised by ``log_rate``; expose it in natural
        # (rate) space so ``_neg_ll(_mle)`` works with ``_mle`` the fitted rate
        # and covariate coefficients.
        # (Kept as what it is built from, so the model pickles, #573.)
        out._neg_ll = functools.partial(
            _in_rate_space,
            Rebuilt(self.create_negll_func, (data,), built=neg_ll),
        )
        out._mle = np.concatenate([out.params, out.coeffs])
        out._n_obs = n_obs
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
