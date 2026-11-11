from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

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
from surpyval.utils.event_history_data import (
    accepts_event_history_data,
    refuses_unmodelled_data,
)
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_utils import handle_xicn, validate_nhpp_data
from surpyval.utils.units import takes_unit

from .proportional_intensity import (
    ProportionalIntensityModel,
    alias_covariates,
)
from .tvc import (
    ProportionalIntensityTVCFitMixin,
    check_tvc_rows,
    count_events,
    handle_recurrent_tvc,
    hpp_tvc_neg_ll,
    mark_tvc,
)


def _in_rate_space(neg_ll: Callable, p: np.ndarray) -> Any:
    """``neg_ll``, which takes ``[log(rate), *coefficients]``, at
    ``p = [rate, *coefficients]``."""
    return neg_ll(np.concatenate([[np.log(p[0])], p[1:]]))


@singleton_fitter
class ProportionalIntensityHPP(
    FitterRepr,
    RecurrentRegressionDataFrameMixin,
    ProportionalIntensityTVCFitMixin,
):
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

    def _repr_details(self) -> list[str]:
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

        # Right truncation (#924): the windows over which each truncated
        # item is known to have failed (it is in the data because of it).
        selection_time = s["sel_to"] - s["sel_from"]
        Z_sel = Z[s["sel_idx"]]
        gap_time = s["gap_to"] - s["gap_from"]
        Z_gap = Z[s["gap_idx"]]

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

            phi_sel = np.exp(np.dot(Z_sel, beta_coeffs))
            ll -= np.sum(np.log(-np.expm1(-rate * phi_sel * selection_time)))
            phi_gap = np.exp(np.dot(Z_gap, beta_coeffs))
            ll += np.sum(np.log(-np.expm1(-rate * phi_gap * gap_time)))

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
    @takes_unit
    @accepts_event_history_data
    def fit(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        init: ArrayLike | None = None,
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

    @dataframe_covariates
    @takes_unit
    def fit_tvc(
        self,
        i: ArrayLike,
        xl: ArrayLike,
        xr: ArrayLike,
        c: ArrayLike,
        Z: ArrayLike,
        n: ArrayLike | None = None,
        init: ArrayLike | None = None,
    ) -> Any:
        """
        Fit the model to covariates that change over time, in start-stop
        format.

        Each row is one interval ``(xl, xr]`` of item ``i``'s history
        over which its covariate row ``Z`` is constant. While those
        covariates hold the item fails at the rate
        :math:`\\lambda e^{\\beta' Z}`: the value a covariate has now
        multiplies the rate now, with no memory of its earlier values.
        The arguments are those of the life regressions' ``fit_tvc``; an
        item here may fail many times, so any of its rows may end in a
        failure. With one covariate value per item the fit is that of
        :meth:`fit`.

        An item's rows must not overlap. They may leave gaps (the item
        is not at risk in a gap), and a first ``xl`` above 0 is delayed
        entry. Cutting a row in two with the same covariates changes
        nothing. Left-censored counts (``c = -1``) and right truncation
        (``tr``) are not taken: give a count as a ``c = 2`` row.

        Parameters
        ----------
        i : array_like
            The item of each interval row.
        xl, xr : array_like
            The open-closed interval ``(xl, xr]`` of each row.
        c : array_like
            What ends the row at ``xr``: ``0`` a failure, ``1`` no
            failure (a covariate change or the end of the item's
            record), ``2`` a count ``n`` of failures somewhere within
            the row (failures counted between inspections).
        Z : array_like
            The covariate row in force on each interval (a 1-D array is
            a single covariate). A missing value raises a
            ``ValueError``: the row is part of an item's history.
        n : array_like, optional
            The number of failures on a ``c = 2`` row; 1 (the default)
            on the others.
        init : array_like, optional
            Initial parameter estimates: the baseline rate followed by
            the covariate coefficients.

        Returns
        -------
        ProportionalIntensityModel
            The fitted model, with ``is_tvc`` set. ``cif_tvc``,
            ``iif_tvc`` and ``cb_tvc`` predict along a covariate
            schedule; the methods that read one covariate row per item
            (``residuals``, ``cramer_von_mises``, ``trend_test``,
            ``diagnose``, ``plot``) refuse, saying so.

        Examples
        --------
        Two pumps, each watched for 20 months. The first runs at low
        load until month 10 and at high load after it; the second the
        other way round. Two failures fall in the 20 pump-months at low
        load and four in the 20 at high load:

        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> xl = [0, 6, 10, 13, 17, 0, 3, 7, 10, 16]
        >>> xr = [6, 10, 13, 17, 20, 3, 7, 10, 16, 20]
        >>> c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
        >>> high_load = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        >>> model = ProportionalIntensityHPP.fit_tvc(i, xl, xr, c, high_load)
        >>> model.params.round(4), model.coeffs.round(4)
        (array([0.1]), array([0.6931]))

        The rate at low load is 2 / 20 a month, and high load doubles it
        (``exp(0.6931) = 2``). The expected number of failures of a pump
        that switches to high load at month 12:

        >>> model.cif_tvc([12, 24], [[0], [1]], xl=[0, 12]).round(4)
        array([1.2, 3.6])
        """
        rows = handle_recurrent_tvc(i, xl, xr, c, Z, n)
        check_tvc_rows(rows, self)
        out = ProportionalIntensityModel()
        out.feature_names = fit_columns()
        out._rate_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (-np.inf, np.inf)
        n_obs = count_events(rows)

        def default_start() -> np.ndarray:
            # The failures per unit of time at risk, and no effects.
            rate = n_obs / float((rows["xr"] - rows["xl"]).sum())
            return np.append(np.log(rate), np.zeros(rows["Z"].shape[1]))

        self._maximise(
            out, hpp_tvc_neg_ll, (rows,), rows["Z"], n_obs, default_start, init
        )
        return mark_tvc(out, rows)

    @dataframe_covariates
    @takes_unit
    @accepts_event_history_data
    def fit_penalised(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        penalizer: float | ArrayLike = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tol: float = 1e-10,
        w: ArrayLike | float | None = None,
    ) -> Any:
        """
        Fit with a ridge, lasso or elastic-net penalty on the covariate
        coefficients (the base rate is not penalised).

        The fit minimises the negative log-likelihood of :meth:`fit`
        divided by ``N``, the number of items, plus

        .. math::
            \\sum_j \\lambda_j \\left( \\frac{1 - a}{2} b_j^2
            + a |b_j| \\right)

        on the coefficients of the covariates divided by their standard
        deviations over the items (``standardise=True``, divisor ``N``):
        glmnet's ``family = "poisson"`` fit of each item's number of
        events with the log of its follow-up as offset, which is the same
        likelihood for exactly observed events. Every kind of censoring
        and truncation of :meth:`fit` is taken. The model has no
        covariance, standard errors, bounds or goodness-of-fit bootstrap
        (they raise an error that says why); its ``aic`` and ``bic``
        count the effective degrees of freedom.

        Parameters
        ----------
        x, Z, i, c, n, t, tl, tr : array_like
            As for :meth:`fit`.
        penalizer : float or array_like, optional
            One value, or one per column of ``Z`` (0 leaves a coefficient
            unpenalised). Default 0.1.
        l1_ratio : float, optional
            From 0 (ridge, the default) to 1 (lasso).
        standardise : bool, optional
            Penalise the coefficients of the standardised covariates
            (default ``True``); the coefficients are reported on the
            covariates' own scale either way.
        tol : float, optional
            The optimality tolerance on the gradient of the objective.
        w : array_like, optional
            Not taken yet: weights are refused.

        Returns
        -------
        ProportionalIntensityModel
            The fitted model, with ``penalty`` set.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> Z = df[["fin", "age", "prio"]].values
        >>> model = ProportionalIntensityHPP.fit_penalised(
        ...     df["week"].values, Z, i=np.arange(len(df)),
        ...     c=1 - df["arrest"].values, penalizer=0.02, l1_ratio=1,
        ... )
        >>> model.coeffs.round(4)
        array([-0.1885, -0.0462,  0.0734])
        """
        from surpyval.univariate.regression._penalised import refuse_weights

        refuse_weights(w)
        data = handle_xicn(x, i, c, n, t=t, tl=tl, tr=tr, Z=Z)
        return self._fit_penalised_from_recurrent_data(
            data, penalizer, l1_ratio, standardise, tol
        )

    @refuses_unmodelled_data
    def _fit_penalised_from_recurrent_data(
        self,
        data: Any,
        penalizer: Any,
        l1_ratio: Any,
        standardise: Any,
        tol: float,
    ) -> Any:
        """:meth:`fit_penalised` from a prepared ``EventHistoryData``."""
        import copy

        from autograd import grad as ag_grad
        from autograd import hessian as ag_hessian

        from surpyval.univariate.regression._penalised import (
            PenalisedObjective,
            penalised_fit,
        )

        validate_nhpp_data(data, self)
        Z_rows = onp.asarray(data.Z, dtype=float)
        # The covariates describe the items: standardised over the items,
        # and N is the number of items.
        _, first = onp.unique(data.i, return_index=True)
        Z_items = Z_rows[first]
        p = Z_rows.shape[1]

        def build(scale: onp.ndarray, N: float) -> PenalisedObjective:
            scaled = copy.copy(data)
            scaled.Z = Z_rows / scale
            neg_ll = self.create_negll_func(scaled)

            def objective(theta: Any) -> Any:
                return neg_ll(theta) / N

            gradient = ag_grad(objective)
            curvature = ag_hessian(objective)
            return PenalisedObjective(
                lambda th: float(objective(np.asarray(th))),
                lambda th: onp.asarray(gradient(np.asarray(th)), dtype=float),
                lambda th: onp.asarray(curvature(np.asarray(th)), dtype=float),
                onp.asarray(self._default_start(data), dtype=float),
            )

        fit = penalised_fit(
            build,
            Z_items,
            onp.ones(Z_items.shape[0]),
            onp.arange(1, 1 + p),
            penalizer,
            l1_ratio,
            standardise,
            tol,
        )
        res = fit.result
        out = ProportionalIntensityModel()
        out.data = data
        out.feature_names = fit_columns()
        out._rate_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (-np.inf, np.inf)
        out.maximum = "verified" if res.verified else "unverified"
        if not res.verified:
            warn_unverified("The penalised proportional intensity fit")
        out.res = res
        out.penalty = fit.penalty
        out.params = onp.atleast_1d(onp.exp(res.x[0]))
        out.coeffs = onp.asarray(fit.coefficients, dtype=float)
        out.name = "Homogeneous Poisson Process"
        out.kind = "HPP"
        out.parameterization = "Parametric"
        neg_ll = self.create_negll_func(data)
        out._neg_ll = functools.partial(
            _in_rate_space,
            Rebuilt(self.create_negll_func, (data,), built=neg_ll),
        )
        out._mle = onp.concatenate([out.params, out.coeffs])
        out._n_obs = bic_sample_size(data)
        out.dist = self
        out._fitter = self
        out._fitter_dist = None
        return out

    @refuses_unmodelled_data
    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any = None,
        init: ArrayLike | None = None,
    ) -> Any:
        """
        Fit from a prepared
        :class:`~surpyval.utils.event_history_data.EventHistoryData`
        (with covariates attached), as built by ``surpyval.handle_xicn``.
        :meth:`fit` builds one from its arrays and calls this.

        Parameters
        ----------

        data : EventHistoryData
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
        return self._maximise(
            out,
            self.create_negll_func,
            (data,),
            data.Z,
            bic_sample_size(data),
            functools.partial(self._default_start, data),
            init,
        )

    def _maximise(
        self,
        out: Any,
        build: Callable,
        args: tuple,
        Z: np.ndarray,
        n_obs: float,
        default_start: Callable,
        init: ArrayLike | None,
    ) -> Any:
        """Maximise the likelihood ``build(*args)`` (a function of
        ``[log(rate), *coefficients]``) of a fit to covariate rows ``Z``
        holding ``n_obs`` events, from ``init`` or ``default_start()``,
        and finish the model ``out``: the search of :meth:`fit` and of
        :meth:`fit_tvc`."""
        num_covariates = Z.shape[1]
        user_init = init is not None
        if init is None:
            start = default_start()
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

        neg_ll = build(*args)

        # A coefficient the data cannot determine is held at 0 and reported
        # as nan (#502); the rate is the intercept.
        aliased = alias_covariates(Z, intercept=True)
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
            res = better_result(res, search(default_start()))
        # The answer is kept only as a verified maximum (zero gradient,
        # negative-definite Hessian of the log-likelihood), polished where
        # it is not one, each coefficient in its own covariate's units
        # (#577). BFGS's own verdict is no test: it reports a "precision
        # loss" at the maximum of the Rossi fit, and success where it
        # never moved; and its absolute tolerance stopped a covariate in
        # millionths 0.06 short of the maximum.
        floor = coefficient_floor(
            int(free.sum()),
            [
                (int(free[:j].sum()), j - 1)
                for j in range(1, 1 + num_covariates)
                if free[j]
            ],
            Z,
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
            Rebuilt(build, args, built=neg_ll),
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
