from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent._convergence import better_result
from surpyval.recurrent.inference import bic_sample_size
from surpyval.recurrent.parametric import Duane
from surpyval.recurrent.parametric.counting_process import CountingProcess
from surpyval.recurrent.parametric.nhpp_fitter import nhpp_log_likelihood
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
from surpyval.utils.removed_names import removed_arguments
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
    mark_tvc,
    nhpp_tvc_neg_ll,
    rows_as_event_times,
)


@singleton_fitter
class ProportionalIntensityNHPP(
    FitterRepr,
    RecurrentRegressionDataFrameMixin,
    ProportionalIntensityTVCFitMixin,
):
    """
    Proportional-intensity regression on a non-homogeneous Poisson
    process: each item's intensity is a parametric baseline intensity
    scaled by its covariates,

    .. math::
        \\lambda(t \\mid Z) = \\lambda_0(t)\\, e^{\\beta' Z},

    with the baseline any NHPP model -- ``Duane`` (the default),
    ``CrowAMSAA`` or ``CoxLewis`` -- chosen with ``baseline``, as for
    ``ARI`` (``dist``, its name before v0.24, was removed in v0.25).

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
       coef_0  :  0.45194475814452534
    <BLANKLINE>
    """

    #: The ``repr`` (#614)
    fitter_kind = "proportional intensity fitter"

    def _repr_details(self) -> list[str]:
        return ["NHPP baseline"]

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
        Z_pieces["sel"] = Z[s["sel_idx"]]
        Z_pieces["gap"] = Z[s["gap_idx"]]
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
                # Without the covariates, which the baseline's own fit
                # refuses (#959).
                base = dist.fit_from_recurrent_data(data.without("covariates"))
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

    @removed_arguments("0.25", dist="'baseline'")
    @refuses_unmodelled_data
    def fit_from_recurrent_data(
        self,
        data: Any,
        baseline: Any = Duane,
        init: ArrayLike | None = None,
    ) -> Any:
        """
        Fit from a prepared
        :class:`~surpyval.utils.event_history_data.EventHistoryData`
        (with covariates attached), as built by
        ``surpyval.handle_xicn``. :meth:`fit` builds one from its arrays and
        calls this.

        Parameters
        ----------

        data : EventHistoryData
            The recurrent event data, including ``Z``.
        baseline : CountingProcess, optional
            The baseline intensity model, as for :meth:`fit`.
        init : array_like, optional
            Initial parameter estimates, as for :meth:`fit`.

        Returns
        -------

        ProportionalIntensityModel
            The fitted model.
        """
        dist = baseline
        if not isinstance(dist, CountingProcess):
            raise TypeError(
                "`baseline` must be a CountingProcess instance "
                "(e.g. Duane, CrowAMSAA, CoxLewis); got {!r}".format(dist)
            )
        validate_nhpp_data(data, dist)
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data
        # The covariates' columns name the coefficients (#614)
        out.feature_names = fit_columns()
        return self._maximise(
            out,
            dist,
            self.create_negll_func,
            (data, dist),
            data.Z,
            bic_sample_size(data),
            lambda: self._baseline_start(data, dist),
            init,
        )

    def _maximise(
        self,
        out: Any,
        dist: Any,
        build: Callable,
        args: tuple,
        Z: np.ndarray,
        n_obs: float,
        baseline_start: Callable,
        init: ArrayLike | None,
    ) -> Any:
        """Maximise the likelihood ``build(*args)`` (a function of
        ``[*baseline parameters, *coefficients]``) of a fit to covariate
        rows ``Z`` holding ``n_obs`` events, from ``init`` or the baseline
        at ``baseline_start()`` with no effects, and finish the model
        ``out``: the search of :meth:`fit` and of :meth:`fit_tvc`."""
        num_covariates = Z.shape[1]
        expected = len(dist.parameter_names) + num_covariates

        def default_init() -> np.ndarray:
            return np.append(baseline_start(), np.zeros(num_covariates))

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

        neg_ll = build(*args)

        # A coefficient the data cannot determine is held at 0 and reported
        # as nan (#502). A constant column is one only where the baseline
        # has a scale, which is then the intercept.
        n_dist = len(dist.parameter_names)
        aliased = alias_covariates(
            Z, intercept=getattr(dist, "has_scale", False)
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
        # Accepted only as a verified maximum: polished where it is not one
        # (by central differences: the likelihood is in plain numpy), and
        # said otherwise (principle 13).
        verified = False
        if res.fun < 1e300:
            # Each coefficient judged in its own covariate's units (#577)
            floor = coefficient_floor(
                int(free.sum()),
                [
                    (int(free[:j].sum()), j - n_dist)
                    for j in range(n_dist, expected)
                    if free[j]
                ],
                Z,
            )
            res, verified = verify_or_polish(
                objective,
                res,
                n_obs,
                numerical=True,
                floor=floor,
            )
        out.maximum = "verified" if verified else "unverified"
        if not verified:
            warn_unverified("The proportional intensity fit")
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
        # Kept as what it is built from, so the model pickles (#573)
        out._neg_ll = Rebuilt(build, args, built=neg_ll)
        out._mle = fitted
        out._n_obs = n_obs

        return out

    @removed_arguments("0.25", dist="'baseline'")
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
        baseline: Any = Duane,
        init: ArrayLike | None = None,
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
            so the baseline intensity is integrated out to ``tr`` even without
            an explicit right-censoring (``c=1``) row.
        baseline : CountingProcess, optional
            The baseline intensity model: ``Duane`` (the default),
            ``CrowAMSAA`` or ``CoxLewis`` from ``surpyval.recurrent``. With
            ``HPP`` the model is the same as ``ProportionalIntensityHPP``.
            Named as ``ARI``'s baseline is (#653); its name before v0.24,
            ``dist``, was removed in v0.25.
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
        return self.fit_from_recurrent_data(data, baseline, init)

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
        baseline: Any = Duane,
        init: ArrayLike | None = None,
    ) -> Any:
        """
        Fit the model to covariates that change over time, in start-stop
        format.

        Each row is one interval ``(xl, xr]`` of item ``i``'s history
        over which its covariate row ``Z`` is constant. The item's
        intensity at time ``t`` is the baseline's at ``t`` times
        :math:`e^{\\beta' Z}` for the covariates in force at ``t``: the
        value a covariate has now multiplies the intensity now, with no
        memory of its earlier values. So the expected number of failures
        over a row is the baseline's expected number over ``(xl, xr]``
        times that factor. The arguments are those of the life
        regressions' ``fit_tvc``; an item here may fail many times, so
        any of its rows may end in a failure. With one covariate value
        per item the fit is that of :meth:`fit`.

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
            The open-closed interval ``(xl, xr]`` of each row, in the
            item's own time (its age), which the baseline's trend is
            measured in.
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
        baseline : CountingProcess, optional
            The baseline intensity model, as for :meth:`fit`: ``Duane``
            (the default), ``CrowAMSAA`` or ``CoxLewis``.
        init : array_like, optional
            Initial parameter estimates: the baseline parameters
            followed by the covariate coefficients.

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
        Three machines watched to 30 months, each overhauled part of the
        way through (``overhauled`` steps from 0 to 1). Each row is the
        time from one event to the next:

        >>> from surpyval.recurrent import CrowAMSAA
        >>> from surpyval.recurrent import ProportionalIntensityNHPP
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        >>> xl = [0, 8, 13, 15, 18, 26, 0, 6, 11, 16, 19, 20, 27, 0, 9, 10,
        ...       17, 22]
        >>> xr = [8, 13, 15, 18, 26, 30, 6, 11, 16, 19, 20, 27, 30, 9, 10,
        ...       17, 22, 30]
        >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
        >>> overhauled = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
        ...               1]
        >>> model = ProportionalIntensityNHPP.fit_tvc(
        ...     i, xl, xr, c, overhauled, baseline=CrowAMSAA
        ... )
        >>> model.is_tvc
        True
        >>> bool(model.coeffs[0] < 0)  # fewer failures after an overhaul
        True

        The expected number of failures by months 12 and 24 of a machine
        overhauled at month 12, and of one never overhauled:

        >>> plan = [[0], [1]]
        >>> with_overhaul = model.cif_tvc([12, 24], plan, xl=[0, 12])
        >>> without = model.cif([12, 24], [0])
        >>> bool(with_overhaul[0] == without[0])
        True
        >>> bool(with_overhaul[1] < without[1])
        True
        """
        dist = baseline
        if not isinstance(dist, CountingProcess):
            raise TypeError(
                "`baseline` must be a CountingProcess instance "
                "(e.g. Duane, CrowAMSAA, CoxLewis); got {!r}".format(dist)
            )
        rows = handle_recurrent_tvc(i, xl, xr, c, Z, n)
        check_tvc_rows(rows, dist)
        out = ProportionalIntensityModel()
        out.dist = dist
        out.feature_names = fit_columns()

        def baseline_start() -> np.ndarray:
            # The covariate-free fit of the baseline to the failure
            # times, as for fit (gaps in an item's record ignored).
            x, item, flags, tl = rows_as_event_times(rows)
            try:
                with np.errstate(all="ignore"):
                    data = handle_xicn(x, item, flags, tl=tl)
            except ValueError:
                return np.ones(len(dist.parameter_names))
            return self._baseline_start(data, dist)

        self._maximise(
            out,
            dist,
            nhpp_tvc_neg_ll,
            (rows, dist),
            rows["Z"],
            count_events(rows),
            baseline_start,
            init,
        )
        return mark_tvc(out, rows)
