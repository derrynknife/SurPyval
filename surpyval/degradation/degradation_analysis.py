"""Degradation analysis.

Classic (pseudo-failure-time) degradation analysis: a degradation
measurement is tracked over time on each unit, a
:class:`~surpyval.degradation.PathModel` is fitted to each unit's
measurements, each fitted path is extrapolated to the failure threshold
to get that unit's pseudo failure time, and a lifetime distribution is
fitted to the pseudo failure times. Units whose fitted path never
reaches the threshold are treated as right censored at their last
observed time; units already past the threshold at their first
measurement (the fitted path crossed at or before time zero) as left
censored at their first measurement time.

With ``acceleration="clock"`` the stress -- which may change during a
unit's test, as in a step-stress test -- speeds up the clock of every
unit's path: the path is the ordinary path model evaluated on the
reference-stress time the unit has aged, the pseudo failure times are
reference-stress lifetimes, and life under any stress profile follows from
the reference-stress life distribution. See :mod:`.step_stress`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.univariate.parametric import Weibull
from surpyval.univariate.regression import AFT
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)
from surpyval.utils.deprecation import renamed_arguments
from surpyval.utils.linalg import psd_project, safe_inv
from surpyval.utils.validation import check_option, option_error
from surpyval.utils.warnings import caller_stacklevel

from ._clock import stress_row
from ._measurements import validate_xy

# DegradationModel, RULPrediction and InducedFailureDistribution are
# re-exported: code imports them from this module.
from .degradation_model import (  # noqa: F401
    DegradationModel,
    _failure_side,
    _is_regression_fitter,
    _path_at,
)
from .path_models import PATH_MODELS, PathModel, get_path_model
from .population import reml_estimate, reml_estimate_nonlinear
from .rul import InducedFailureDistribution, RULPrediction  # noqa: F401
from .step_stress import (
    clock_units,
    mixed_model_estimate,
    profile_least_squares,
)
from .stress import (
    LinkedPathModel,
    fixed_effect_names,
    stress_design,
    validate_links,
)

#: What a between-unit covariance on the boundary of the positive
#: semi-definite cone means, and what to do: the end of the warning both
#: population methods give when their estimate is singular.
_BOUNDARY_CONSEQUENCE = (
    "a direction of between-unit variation is estimated as zero (a "
    "variance of zero, or a correlation of +-1 between path parameters). "
    "The standard errors and intervals that depend on the population "
    "parameters (path_param_cov and the bounds drawn from it) are "
    "unreliable there. More units, or a path model with fewer parameters, "
    "give the data a better chance to determine it"
)


@dataclass
class _UnitFits:
    """Each unit's path fit, and the sums the population estimates use."""

    path_params: npt.NDArray
    pseudo: npt.NDArray
    last_time: npt.NDArray
    estimation_cov_sum: npt.NDArray
    link_params: npt.NDArray
    rss_total: Any = 0.0
    dof_total: int = 0
    y_by_unit: list = field(default_factory=list)
    x_by_unit: list = field(default_factory=list)
    design_by_unit: list = field(default_factory=list)
    link_design_by_unit: list = field(default_factory=list)
    link_estimation_covs: list = field(default_factory=list)


class _Population(NamedTuple):
    """The moment estimate of the path-parameter population."""

    mean: npt.NDArray
    cov: npt.NDArray
    sample_cov: npt.NDArray
    measurement_var: Any
    was_clipped: bool


class DegradationAnalysis_:
    """
    Pseudo-failure-time degradation analysis.

    Fits a degradation path model to each unit's measurements,
    extrapolates each fitted path to the failure ``threshold`` to
    obtain per-unit pseudo failure times, and fits a lifetime
    distribution to those times. Units whose fitted path never reaches
    the threshold at a positive finite time are right censored at their
    last observed time (with a warning); units already past the threshold
    at their first measurement -- the path crossed at or before time zero
    -- have failed by then and are left censored at their first
    measurement time (with a warning). The side of the threshold that
    counts as failed is read from the units whose paths cross it.

    Examples
    --------

    >>> import numpy as np
    >>> from surpyval.degradation import DegradationAnalysis
    >>> # 4 units measured every 100 hours; degradation grows linearly
    >>> # at a different rate per unit; failure is defined at level 150.
    >>> x = np.tile(np.arange(100, 1100, 100), 4)
    >>> slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    >>> i = np.repeat([1, 2, 3, 4], 10)
    >>> y = 10 + slopes * x
    >>> model = DegradationAnalysis.fit(x, y, i, threshold=150)
    >>> print(model)
    Degradation Analysis SurPyval Model
    ===================================
    Path Model          : Linear
    Threshold           : 150.0
    Number of Units     : 4
    Censored Units      : 0
    Life Distribution   : Weibull
    Parameters          :
         alpha: 441.47809611105606
          beta: 6.987078889297555
    >>> model.pseudo_failure_times
    array([451.61290323, 500.        , 318.18181818, 378.37837838])
    """

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        i: npt.ArrayLike,
        threshold: float,
        path: "str | PathModel" = "linear",
        distribution: Any = Weibull,
        how: str = "MLE",
        population_method: str = "moments",
        Z: npt.ArrayLike | None = None,
        links: "dict[str, str] | None" = None,
        acceleration: "str | None" = None,
        stress_ref: npt.ArrayLike | None = None,
    ) -> DegradationModel:
        """
        Fit a degradation analysis model.

        Parameters
        ----------
        x : array like
            Times at which the degradation measurements were taken.
        y : array like
            The degradation measurements.
        i : array like
            The unit each measurement belongs to. Must have the same
            length as ``x`` and ``y``.
        threshold : float
            The degradation level at which a unit is defined to have
            failed.
        path : str or PathModel, optional
            The degradation path model fitted to each unit: one of
            ``"linear"`` (default), ``"quadratic"``, ``"exponential"``,
            ``"offset-exponential"``, ``"power"``, ``"logarithmic"``,
            ``"lloyd-lipow"``, ``"gompertz"``, ``"michaelis-menten"``,
            a :class:`~surpyval.degradation.PathModel` instance, or
            ``"best"`` to fit every registered model to all units and
            select the one with the smallest AICc (the per-candidate
            scores are exposed as ``path_selection`` on the returned
            model; candidates that cannot be fitted to every unit are
            excluded).
        distribution : ParametricFitter, optional
            The lifetime distribution fitted to the pseudo failure
            times. Defaults to ``Weibull``.
        how : str, optional
            The method used to fit the lifetime distribution (passed
            to ``distribution.fit``). Defaults to ``"MLE"``.
        population_method : str, optional
            How the population path-parameter distribution
            (``path_param_mean``, ``path_param_cov``,
            ``measurement_var``) is estimated. ``"moments"`` (default)
            uses the two-stage noise-corrected sample moments;
            ``"reml"`` maximises the restricted marginal likelihood of
            the mixed model, which never needs clipping and is
            preferable with few units; both warn when the estimated
            covariance is singular (on the boundary: a variance of zero
            or a correlation of +-1). Linear-in-parameter paths (linear,
            quadratic, logarithmic, lloyd-lipow) are fitted as an exact
            linear mixed model; nonlinear paths (exponential, power,
            gompertz, ...) are fitted by the Lindstrom-Bates FOCE
            linearisation. Either way REML requires measurement noise
            (some unit with more measurements than path parameters).
        Z : array like, optional
            Stress covariates for accelerated degradation testing (ADT).
            When given, the life model is fitted as a *regression* on the
            pseudo failure times -- ``log(pseudo failure time) = f(Z) +
            noise`` -- so that life can be predicted at any stress. ``Z`` is
            aligned to ``x``/``y``/``i`` (one row per measurement) and must be
            constant within each unit (a unit is tested at a single stress);
            it is reduced to one covariate row per unit. If ``distribution``
            is already a regression fitter (e.g. ``AFT(Weibull)``,
            ``WeibullPH``, ``CoxPH``) it is used directly; a plain
            distribution (e.g. ``Weibull``) is wrapped in an accelerated
            failure time model, ``AFT(distribution)``. The returned model's
            prediction methods (``sf``, ``ff``, ``qf``, ``random`` ...) then
            take the stress vector ``Z`` at which to evaluate life.
        links : dict, optional
            Model the degradation *mechanism* against stress as well
            (requires ``Z``): the path parameters named here depend on
            the unit's stress, the rest do not. Each value is the link
            the parameter is modelled on -- ``"identity"`` (the
            parameter is linear in ``Z``) or ``"log"`` (its log is
            linear in ``Z``, so a rate with ``Z = 1/T`` follows an
            Arrhenius relationship and the parameter stays positive).
            Per unit, ``eta_i = D(z_i) gamma + u_i`` on the link scale
            with a between-unit random effect ``u_i ~ MVN(0, Sigma)``;
            ``gamma`` and ``Sigma`` are estimated by the same two-stage
            or REML route as the plain population, and stored as
            ``path_param_fixed`` (labelled by ``path_param_fixed_names``)
            and ``path_param_link_cov``. The life model is still the
            covariate regression on the pseudo failure times, so every
            prediction method works as without ``links``. For example
            ``links={"b": "log"}`` with the linear path lets the
            degradation rate ``b`` accelerate log-linearly with stress
            while the intercept ``a`` (the initial state) is common.
        acceleration : {None, "clock"}, optional
            ``"clock"`` models stress as speeding up the clock of every
            unit's path, which allows ``Z`` to change *during* a unit's
            test (a step-stress test) as well as between units. A unit
            at stress ``z`` ages ``AF(z) = exp(gamma' (z - stress_ref))``
            times faster than at the reference stress, and its path is
            the path model evaluated on the reference-stress time it has
            aged, ``tau(t) = integral of AF(z(s)) ds``. ``Z`` is then one
            row per measurement giving the stress applied over the
            interval that *ends* at that measurement (the first interval
            starts at time zero, so times must be non-negative). The path
            parameters, their population and the pseudo failure times
            are all on the reference-stress clock; ``distribution`` is
            fitted to those reference-stress lifetimes, and the
            prediction methods take the stress as ``Z`` (one stress row
            or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`)
            to give life under any stress history,
            ``F(t) = F0(tau(t))``. The stress
            coefficients are stored as ``gamma``. With
            ``population_method="moments"`` they are estimated by
            profile least squares, which needs units whose stress changes
            during the test (a unit held at one stress can absorb any
            acceleration into its own path parameters); with ``"reml"``
            by the mixed model, which also uses the differences between
            units at different stresses. Cannot be combined with
            ``links`` or ``path="best"``.
        stress_ref : array like, optional
            The reference stress for ``acceleration="clock"`` (usually the
            use condition), one row. Defaults to the mean stress over the
            measurement intervals.

        Returns
        -------
        DegradationModel
            The fitted degradation model, with the per-unit paths,
            pseudo failure times, and the fitted life model.
        """
        x_arr, y_arr, i_arr = self._handle_xyi(x, y, i)
        threshold = self._check_fit_arguments(
            threshold, population_method, i_arr, acceleration, stress_ref
        )
        units = np.unique(i_arr)
        is_best = isinstance(path, str) and path.lower() == "best"
        if acceleration == "clock":
            self._check_clock_arguments(Z, links, is_best, distribution, x_arr)

        path_selection = None
        if is_best:
            path_model, path_selection = self._select_path_model(
                x_arr, y_arr, i_arr, units
            )
        else:
            path_model = get_path_model(path)

        # Stage-3 accelerated degradation: stress speeds up every unit's
        # clock, and the path is fitted on the reference-stress time.
        x_path = x_arr
        Z_rows = gamma = z_ref = clock_population = None
        if acceleration == "clock":
            x_path, Z_rows, gamma, z_ref, clock_population = self._fit_clock(
                x_arr,
                y_arr,
                i_arr,
                units,
                Z,
                stress_ref,
                path_model,
                population_method,
            )

        # Stage-2 accelerated degradation: the path parameters depend on
        # stress, modelled on a link scale by a wrapped path model.
        Z_units = (
            None
            if Z is None or acceleration == "clock"
            else self._handle_Z(Z, i_arr, units)
        )
        linked: "LinkedPathModel | None" = None
        if links is not None:
            if Z_units is None:
                raise ValueError(
                    "links models the path parameters against stress, so "
                    "the stress covariates Z must be given too"
                )
            links = validate_links(path_model, links)
            linked = LinkedPathModel(path_model, links)

        # A bootstrap refit's units are the model's own (see
        # ``utils.refits.DEGRADATION_REFIT``, #522); not where the clock,
        # fitted to all the units, sets their times.
        from surpyval.utils.refits import DEGRADATION_REFIT

        refit = DEGRADATION_REFIT.get()
        kept = (
            refit["units"]
            if refit is not None and acceleration is None
            else None
        )
        fits = self._fit_unit_paths(
            x_path, y_arr, i_arr, units, path_model, threshold, linked, kept
        )

        population = self._moment_population(fits, len(units))
        if population.was_clipped and population_method == "moments":
            warnings.warn(
                "The noise-corrected between-unit covariance of the path "
                "parameters was not positive semi-definite (the estimation "
                "noise is comparable to the between-unit scatter); its "
                "negative eigenvalues were clipped to zero, so "
                + _BOUNDARY_CONSEQUENCE
                + "; population_method='reml' estimates it by restricted "
                "maximum likelihood instead, though it may also land on the "
                "boundary",
                stacklevel=caller_stacklevel(),
            )
        path_param_mean = population.mean
        path_param_cov = population.cov
        measurement_var = population.measurement_var

        reml_diagnostics: dict = {}
        if population_method == "reml":
            path_param_mean, path_param_cov, measurement_var = (
                self._reml_population(
                    fits,
                    population,
                    path_model,
                    y_arr,
                    clock_population,
                    reml_diagnostics,
                )
            )

        path_param_fixed = None
        path_param_fixed_names = None
        path_param_link_cov = None
        link_diagnostics: dict = {}
        if linked is not None:
            assert Z_units is not None and links is not None
            path_param_fixed, path_param_link_cov, link_var = (
                self._fit_stress_population(
                    linked,
                    links,
                    Z_units,
                    fits.y_by_unit,
                    fits.x_by_unit,
                    fits.link_params,
                    fits.link_design_by_unit,
                    fits.link_estimation_covs,
                    measurement_var,
                    population_method,
                    diagnostics=link_diagnostics,
                )
            )
            path_param_fixed_names = fixed_effect_names(
                linked.parameter_names,
                path_model.parameter_names,
                links,
                Z_units.shape[1],
            )
            if population_method == "reml":
                # the stress-conditional model is the population model
                # of a linked fit; its noise estimate supersedes the
                # pooled one
                measurement_var = link_var
        self._warn_reml_boundary(reml_diagnostics, link_diagnostics)

        pseudo_failure_times, c = self._censor_units(
            fits, path_model, threshold, y_arr, x_path, i_arr, units
        )
        life_model = self._fit_life_model(
            distribution, how, pseudo_failure_times, c, Z_units, refit
        )

        model = DegradationModel(
            x=x_arr,
            y=y_arr,
            i=i_arr,
            units=units,
            threshold=threshold,
            path_model=path_model,
            path_params=fits.path_params,
            pseudo_failure_times=pseudo_failure_times,
            c=c,
            life_model=life_model,
            measurement_var=measurement_var,
            path_param_mean=path_param_mean,
            path_param_cov=path_param_cov,
            path_param_sample_cov=population.sample_cov,
            population_method=population_method,
            path_selection=path_selection,
            Z=Z_rows if acceleration == "clock" else Z_units,
            links=links,
            path_param_fixed=path_param_fixed,
            path_param_fixed_names=path_param_fixed_names,
            path_param_link_cov=path_param_link_cov,
            acceleration=acceleration,
            gamma=gamma,
            stress_ref=z_ref,
        )
        # Recorded so the bootstrap confidence bounds can rerun the pipeline
        # (with the selected path model held fixed) on resampled units.
        model._distribution = distribution
        model._how = how
        return model

    @staticmethod
    def _check_fit_arguments(
        threshold: float,
        population_method: str,
        i_arr: npt.NDArray,
        acceleration: "str | None",
        stress_ref: Any,
    ) -> float:
        """Check ``fit``'s scalar options; the threshold as a float."""
        # a 0-d array (e.g. ``np.array(15.0)`` or a reduction's result) is a
        # number too; anything with a shape is not
        if isinstance(threshold, np.ndarray) and threshold.ndim == 0:
            threshold = threshold.item()
        if not isinstance(threshold, Number) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
        threshold = float(threshold)

        check_option(
            "population_method", population_method, ("moments", "reml")
        )

        units = np.unique(i_arr)
        if len(units) < 2:
            raise ValueError(
                "Degradation analysis requires at least 2 units; "
                "got {}".format(len(units))
            )

        if acceleration not in (None, "clock"):
            raise option_error("acceleration", acceleration, (None, "clock"))
        if acceleration is None and stress_ref is not None:
            raise ValueError(
                "stress_ref is the reference stress of acceleration='clock' "
                "and is only used with it"
            )
        return threshold

    @staticmethod
    def _fit_unit_paths(
        x_path: npt.NDArray,
        y_arr: npt.NDArray,
        i_arr: npt.NDArray,
        units: npt.NDArray,
        path_model: PathModel,
        threshold: float,
        linked: "LinkedPathModel | None",
        kept: "dict | None",
    ) -> _UnitFits:
        """Fit the path model to each unit, and each path's crossing.

        ``kept`` holds a bootstrap refit's earlier fits, keyed by a unit's
        measurements, and is filled with the ones made here.
        """
        n_params = len(path_model.parameter_names)
        fits = _UnitFits(
            path_params=np.empty((len(units), n_params)),
            pseudo=np.empty(len(units)),
            last_time=np.empty(len(units)),
            estimation_cov_sum=np.zeros((n_params, n_params)),
            link_params=np.empty((len(units), n_params)),
        )
        for idx, unit in enumerate(units):
            mask = i_arr == unit
            x_unit, y_unit = x_path[mask], y_arr[mask]
            if len(np.unique(x_unit)) < n_params:
                raise ValueError(
                    "Unit {} needs measurements at {} or more distinct "
                    "times to fit the {} path model".format(
                        unit, n_params, path_model.name
                    )
                )
            key = (x_unit.tobytes(), y_unit.tobytes())
            if kept is not None and key in kept:
                params, crossing, rss, inv_jtj, jacobian = kept[key]
            else:
                params = path_model.fit(x_unit, y_unit)
                crossing = path_model.inv_path(threshold, *params)
                residuals = y_unit - path_model.path(x_unit, *params)
                rss = residuals @ residuals
                jacobian = path_model.jacobian(x_unit, *params)
                inv_jtj = safe_inv(jacobian.T @ jacobian)
                if kept is not None:
                    kept[key] = (params, crossing, rss, inv_jtj, jacobian)
            fits.path_params[idx] = params
            fits.pseudo[idx] = crossing
            fits.last_time[idx] = x_unit.max()

            fits.rss_total += rss
            fits.dof_total += len(x_unit) - n_params
            fits.estimation_cov_sum += inv_jtj
            fits.y_by_unit.append(y_unit)
            fits.x_by_unit.append(x_unit)
            fits.design_by_unit.append(jacobian)

            if linked is not None:
                # the same fit on the link scale, with the Jacobian
                # (and hence the estimation covariance) mapped there
                eta = linked.to_link(params)
                fits.link_params[idx] = eta
                link_jacobian = linked.jacobian(x_unit, *eta)
                fits.link_design_by_unit.append(link_jacobian)
                fits.link_estimation_covs.append(
                    safe_inv(link_jacobian.T @ link_jacobian)
                )
        return fits

    @staticmethod
    def _moment_population(fits: _UnitFits, n_units: int) -> _Population:
        """The two-stage (Lu-Meeker) moment estimate of the population.

        The scatter of the per-unit estimates is Sigma + V_i, so
        subtracting the average estimation covariance leaves the
        between-unit covariance.
        """
        measurement_var = (
            fits.rss_total / fits.dof_total if fits.dof_total > 0 else 0.0
        )
        path_param_mean = fits.path_params.mean(axis=0)
        path_param_sample_cov = np.atleast_2d(
            np.cov(fits.path_params, rowvar=False, ddof=1)
        )
        mean_estimation_cov = (
            measurement_var * fits.estimation_cov_sum / n_units
        )
        path_param_cov, was_clipped = psd_project(
            path_param_sample_cov - mean_estimation_cov
        )
        return _Population(
            path_param_mean,
            path_param_cov,
            path_param_sample_cov,
            measurement_var,
            was_clipped,
        )

    @staticmethod
    def _reml_population(
        fits: _UnitFits,
        population: _Population,
        path_model: PathModel,
        y_arr: npt.NDArray,
        clock_population: "tuple | None",
        diagnostics: dict,
    ) -> tuple[npt.NDArray, npt.NDArray, float]:
        """``(mean, cov, measurement_var)`` of the population by REML.

        The moment estimates are the starting values; a
        linear-in-parameters path is an exact linear mixed model, a
        nonlinear one is fitted by FOCE linearisation. A clock fit has
        already estimated its population with its clock.
        """
        measurement_var = population.measurement_var
        noise_floor = np.finfo(float).eps * float(np.mean(y_arr**2))
        if not measurement_var > noise_floor:
            raise ValueError(
                "population_method='reml' requires measurement noise, "
                "but the pooled measurement variance is 0 (every unit's "
                "path fitted its measurements exactly, or no unit has "
                "more measurements than path parameters)"
            )
        if clock_population is not None:
            reml_mean, reml_cov, reml_var, converged = clock_population
        elif path_model.linear_in_parameters:
            reml_mean, reml_cov, reml_var, converged = reml_estimate(
                fits.y_by_unit,
                fits.design_by_unit,
                population.cov,
                measurement_var,
                diagnostics=diagnostics,
            )
        else:
            reml_mean, reml_cov, reml_var, converged = reml_estimate_nonlinear(
                fits.y_by_unit,
                fits.x_by_unit,
                path_model,
                population.mean,
                population.cov,
                measurement_var,
                fits.path_params,
                diagnostics=diagnostics,
            )
        if not converged:
            warnings.warn(
                "The REML optimisation did not report convergence; the "
                "population path-parameter estimates may be inaccurate",
                stacklevel=3,
            )
        return reml_mean, reml_cov, reml_var

    @staticmethod
    def _warn_reml_boundary(
        reml_diagnostics: dict, link_diagnostics: dict
    ) -> None:
        """Warn if a REML between-unit covariance is singular."""
        on_boundary = [
            name
            for name, diagnostics in (
                ("path_param_cov", reml_diagnostics),
                ("path_param_link_cov", link_diagnostics),
            )
            if diagnostics.get("on_boundary", False)
        ]
        if on_boundary:
            warnings.warn(
                "The REML estimate of the between-unit covariance of the "
                "path parameters ({}) is singular, on the boundary of the "
                "positive semi-definite cone, so ".format(
                    " and ".join(on_boundary)
                )
                + _BOUNDARY_CONSEQUENCE,
                stacklevel=caller_stacklevel(),
            )

    def _censor_units(
        self,
        fits: _UnitFits,
        path_model: PathModel,
        threshold: float,
        y_arr: npt.NDArray,
        x_path: npt.NDArray,
        i_arr: npt.NDArray,
        units: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(pseudo_failure_times, c)``: each unit's time and censoring.

        A unit whose path crosses the threshold at a positive time fails
        there; one already past it at its first measurement is left
        censored there, and one that never reaches it right censored at
        its last time (each with a warning).
        """
        pseudo = fits.pseudo
        events = np.isfinite(pseudo) & (pseudo > 0)
        if not events.any():
            raise ValueError(
                "No unit's fitted degradation path reaches the threshold "
                "{} at a positive time, so there is no failure time to fit "
                "the life distribution to (a unit already past the "
                "threshold at its first measurement only bounds its "
                "failure time); check the threshold and the path "
                "model".format(threshold)
            )
        started = self._already_failed(
            path_model,
            fits.path_params,
            pseudo,
            events,
            threshold,
            y_arr,
            [x_path[i_arr == unit] for unit in units],
            units,
        )
        if started.any():
            warnings.warn(
                "The fitted degradation path(s) of unit(s) {} are already "
                "past the threshold {} at their first measurement (they "
                "crossed it at or before time zero); these units are "
                "treated as failed by then: left censored at their first "
                "measurement time".format(units[started].tolist(), threshold),
                stacklevel=3,
            )
        never = ~(events | started)
        if never.any():
            warnings.warn(
                "The fitted degradation path(s) of unit(s) {} never reach "
                "the threshold {}; these units are treated as right "
                "censored at their last observed time".format(
                    units[never].tolist(), threshold
                ),
                stacklevel=3,
            )

        first_time = np.array(
            [
                np.min(x_unit[x_unit > 0], initial=np.inf)
                for x_unit in (x_path[i_arr == unit] for unit in units)
            ]
        )
        pseudo_failure_times = np.where(
            events, pseudo, np.where(started, first_time, fits.last_time)
        )
        c = np.where(events, 0, np.where(started, -1, 1))
        return pseudo_failure_times, c

    @staticmethod
    def _fit_life_model(
        distribution: Any,
        how: str,
        pseudo_failure_times: npt.NDArray,
        c: npt.NDArray,
        Z_units: "npt.NDArray | None",
        refit: "dict | None",
    ) -> Any:
        """The life model fitted to the pseudo failure times.

        A regression on the units' stresses when there are any (a plain
        distribution wrapped in ``AFT``); a bootstrap refit starts from
        the original fit's parameters.
        """
        from surpyval.utils.refits import warm_starts

        life_init = None if refit is None else refit.get("life_init")
        if Z_units is None and life_init is not None and how == "MLE":
            with warm_starts():
                return distribution.fit(
                    x=pseudo_failure_times, c=c, how=how, init=life_init
                )
        if Z_units is None:
            return distribution.fit(x=pseudo_failure_times, c=c, how=how)
        reg = (
            distribution
            if _is_regression_fitter(distribution)
            else AFT(distribution)
        )
        return reg.fit(x=pseudo_failure_times, Z=Z_units, c=c)

    @staticmethod
    def _already_failed(
        path_model: PathModel,
        path_params: npt.NDArray,
        pseudo: npt.NDArray,
        events: npt.NDArray,
        threshold: float,
        y_arr: npt.NDArray,
        x_by_unit: "list[npt.NDArray]",
        units: npt.NDArray,
    ) -> npt.NDArray:
        """
        Which units without a positive crossing are already past the
        threshold at their first positive measurement time.

        Such a unit's fitted path crossed at or before time zero: it has
        failed, and its failure time is known only to be before its first
        measurement (left censored there). Counting it as never reaching
        the threshold -- right censored at its last time -- would make the
        worst unit a survivor. A unit on the good side whose
        path moves away from the threshold still never reaches it.
        """
        started = np.zeros(len(units), dtype=bool)
        if events.all():
            return started
        side = _failure_side(
            path_model,
            path_params[events],
            pseudo[events],
            threshold,
            y_arr,
        )
        for idx in np.flatnonzero(~events):
            x_unit = x_by_unit[idx]
            positive = x_unit[x_unit > 0]
            t_ref = positive.min() if positive.size else x_unit.min()
            level = _path_at(path_model, float(t_ref), path_params[idx])[0]
            if not side * (level - threshold) >= 0:
                continue
            if not positive.size:
                raise ValueError(
                    "unit {} is already past the threshold {} at its "
                    "first measurement, but has no measurement at a "
                    "positive time by which its failure is known to have "
                    "happened".format(units[idx], threshold)
                )
            started[idx] = True
        return started

    @staticmethod
    def _check_clock_arguments(
        Z: Any,
        links: Any,
        is_best: bool,
        distribution: Any,
        x_arr: npt.NDArray,
    ) -> None:
        """The combinations ``acceleration="clock"`` does not support."""
        if Z is None:
            raise ValueError(
                "acceleration='clock' speeds up each unit's clock by its "
                "stress, so the stress covariates Z must be given"
            )
        if links is not None:
            raise ValueError(
                "links and acceleration='clock' cannot be combined: the clock "
                "model accelerates every path parameter's effect together, "
                "while links let stress change the path's shape. A shape "
                "change under a stress that varies during the test is not "
                "supported"
            )
        if is_best:
            raise ValueError(
                "path='best' is not supported with acceleration='clock'; "
                "choose the path model"
            )
        if _is_regression_fitter(distribution):
            raise ValueError(
                "With acceleration='clock' the life model is the "
                "reference-stress life distribution, fitted to the pseudo "
                "failure times on the reference-stress clock; stress enters "
                "through the clock, so pass a plain distribution (e.g. "
                "Weibull) rather than a regression fitter"
            )
        if (x_arr < 0).any():
            raise ValueError(
                "With acceleration='clock' the measurement times must be "
                "non-negative: every unit's clock starts at time zero"
            )

    @staticmethod
    def _fit_clock(
        x_arr: npt.NDArray,
        y_arr: npt.NDArray,
        i_arr: npt.NDArray,
        units: npt.NDArray,
        Z: Any,
        stress_ref: Any,
        path_model: PathModel,
        population_method: str,
    ) -> tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, "tuple | None"
    ]:
        """
        Estimate the accelerated clock of ``acceleration="clock"``.

        Returns ``(tau, Z_rows, gamma, stress_ref, population)``: every
        measurement's reference-stress time (aligned to ``x``), the
        validated stress rows, the stress coefficients on the scale of
        ``Z``, the reference stress, and -- for
        ``population_method="reml"`` -- the REML population
        ``(mu, Sigma, sigma2, converged)`` at that clock (``None`` for
        moments).
        """
        Z_rows = np.asarray(Z, dtype=float)
        if Z_rows.ndim == 1:
            Z_rows = Z_rows.reshape(-1, 1)
        if Z_rows.ndim != 2 or len(Z_rows) != len(x_arr):
            raise ValueError(
                "Z must have one row per measurement (same length as x, y "
                "and i); got shape {} for {} measurements".format(
                    np.shape(Z), len(x_arr)
                )
            )
        if Z_rows.shape[1] == 0 or not np.isfinite(Z_rows).all():
            raise ValueError(
                "Z must have at least one column and only finite values"
            )
        q = Z_rows.shape[1]

        # the stress over each measurement interval of positive length --
        # what the data can say about the acceleration
        dt = np.empty_like(x_arr)
        for unit in units:
            mask = np.flatnonzero(i_arr == unit)
            order = mask[np.argsort(x_arr[mask], kind="stable")]
            dt[order] = np.diff(np.concatenate([[0.0], x_arr[order]]))
        exposed = dt > 0
        z_int = Z_rows[exposed]
        design = np.column_stack([np.ones(len(z_int)), z_int])
        if len(z_int) == 0 or np.linalg.matrix_rank(design) < q + 1:
            raise ValueError(
                "the stress coefficients cannot be estimated: Z needs at "
                "least two distinct stress levels across the measurement "
                "intervals, and no covariate may be constant or a "
                "combination of the others"
            )
        within = np.vstack(
            [
                z_int[i_arr[exposed] == unit]
                - z_int[i_arr[exposed] == unit].mean(axis=0)
                for unit in units
                if (i_arr[exposed] == unit).any()
            ]
        )
        # on the scale of the stress spread, so round-off in the deviations
        # of a unit held at one stress does not count as a step
        scale = z_int.std(axis=0)
        stepped = (
            np.linalg.matrix_rank(
                within / scale, tol=1e-9 * np.sqrt(len(within))
            )
            == q
        )
        if population_method == "moments" and not stepped:
            raise ValueError(
                "With population_method='moments' the stress coefficients "
                "are estimated from units whose stress changes during the "
                "test -- a unit held at one stress absorbs any acceleration "
                "into its own path parameters -- and the stress does not "
                "change enough within units to identify them. Use "
                "population_method='reml', which also uses the differences "
                "between units tested at different stresses."
            )
        z_ref = (
            z_int.mean(axis=0)
            if stress_ref is None
            else stress_row(stress_ref, q)
        )
        data = clock_units(x_arr, y_arr, i_arr, units, Z_rows, z_ref, scale)

        if stepped:
            g = profile_least_squares(data, path_model, q)
        else:
            g = np.zeros(q)
        population = None
        if population_method == "reml":
            theta = np.array([path_model.fit(u.tau(g), u.y) for u in data])
            resid = np.concatenate(
                [
                    u.y - path_model.path(u.tau(g), *t)
                    for u, t in zip(data, theta)
                ]
            )
            dof = max(resid.size - theta.size, 1)
            sigma2_init = float(resid @ resid) / dof
            # Checked here, before the mixed-model clock estimate: without
            # noise (or with next to none) the variance components are not
            # identified, and the estimate wanders off (gamma 1.87 for a
            # true 2.0) with overflow warnings -- the REML noise check after
            # the fit would come too late.
            if not sigma2_init > 1e-8 * float(np.var(y_arr)):
                raise ValueError(
                    "population_method='reml' requires measurement noise, "
                    "but on the estimated clock the paths fit the "
                    "measurements (almost) exactly: noise variance {:.3g} "
                    "against a variance of the measurements of {:.3g}. Use "
                    "population_method='moments', which estimates the clock "
                    "by profile least squares (it needs units whose stress "
                    "changes during the test)".format(
                        sigma2_init, float(np.var(y_arr))
                    )
                )
            cov, _ = psd_project(np.atleast_2d(np.cov(theta, rowvar=False)))
            g, converged, population = mixed_model_estimate(
                data,
                path_model,
                g,
                theta,
                theta.mean(axis=0),
                cov,
                sigma2_init,
            )
            if not converged:
                warnings.warn(
                    "The mixed-model estimate of the stress coefficients did "
                    "not report convergence; gamma may be inaccurate",
                    stacklevel=3,
                )

        tau = np.empty_like(x_arr)
        for unit, unit_data in zip(units, data):
            mask = np.flatnonzero(i_arr == unit)
            order = mask[np.argsort(x_arr[mask], kind="stable")]
            tau[order] = unit_data.tau(g)
        return tau, Z_rows, g / scale, z_ref, population

    @staticmethod
    def _fit_stress_population(
        linked: LinkedPathModel,
        links: dict[str, str],
        Z_units: npt.NDArray,
        y_by_unit: list,
        x_by_unit: list,
        link_params: npt.NDArray,
        link_design_by_unit: list,
        link_estimation_covs: list,
        measurement_var: float,
        population_method: str,
        diagnostics: "dict | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray, float]:
        """
        Estimate the stress-conditional population of link-scale path
        parameters, ``eta_i = D(z_i) gamma + u_i``.

        The two-stage estimate regresses the per-unit link-scale fits on
        their stress designs by least squares for ``gamma``, and takes
        the covariance of the residuals less the average link-scale
        estimation covariance (the Lu-Meeker correction) for ``Sigma``.
        With ``population_method="reml"`` that is the starting point of
        the mixed-model REML fit, exact for a linear-in-parameters
        linked path and by FOCE linearisation otherwise.

        Returns ``(gamma, Sigma, sigma2)``.
        """
        n_units, n_params = link_params.shape
        designs = [
            stress_design(z, links, linked.base.parameter_names)
            for z in Z_units
        ]
        stacked = np.vstack(designs)
        gamma, *_ = np.linalg.lstsq(stacked, link_params.ravel(), rcond=None)
        residuals = link_params - np.array([d @ gamma for d in designs])
        ddof = 1 if n_units > 1 else 0
        residual_cov = np.atleast_2d(
            np.cov(residuals, rowvar=False, ddof=ddof)
        )
        mean_estimation_cov = measurement_var * np.mean(
            link_estimation_covs, axis=0
        )
        link_cov, _ = psd_project(residual_cov - mean_estimation_cov)
        sigma2 = measurement_var

        if population_method == "reml":
            if linked.linear_in_parameters:
                a_by_unit = [
                    jac @ d for jac, d in zip(link_design_by_unit, designs)
                ]
                gamma, link_cov, sigma2, converged = reml_estimate(
                    y_by_unit,
                    link_design_by_unit,
                    link_cov,
                    measurement_var,
                    a_mat_list=a_by_unit,
                    diagnostics=diagnostics,
                )
            else:
                gamma, link_cov, sigma2, converged = reml_estimate_nonlinear(
                    y_by_unit,
                    x_by_unit,
                    linked,
                    gamma,
                    link_cov,
                    measurement_var,
                    link_params,
                    d_mat_list=designs,
                    diagnostics=diagnostics,
                )
            if not converged:
                warnings.warn(
                    "The REML optimisation of the stress-conditional path "
                    "population did not report convergence; "
                    "path_param_fixed and path_param_link_cov may be "
                    "inaccurate",
                    stacklevel=3,
                )
        return gamma, link_cov, sigma2

    @staticmethod
    def _select_path_model(
        x_arr: npt.NDArray,
        y_arr: npt.NDArray,
        i_arr: npt.NDArray,
        units: npt.NDArray,
    ) -> "tuple[PathModel, dict[str, float]]":
        """
        Select the registered path model with the smallest AICc over
        all units' measurements.

        Every unit is fitted with every candidate; the residual sums
        of squares are pooled under a common Gaussian error variance,
        so a candidate's AICc is
        ``N ln(RSS/N) + 2k + 2k(k+1)/(N - k - 1)`` with
        ``k = n_units * n_params + 1``. Candidates that cannot be
        fitted to every unit (domain violations, too few distinct
        times, non-convergence, or too few total measurements for the
        AICc correction) are excluded and scored ``nan``.
        """
        n_total = len(x_arr)
        rss_floor = n_total * np.finfo(float).eps * float(np.mean(y_arr**2))
        scores: "dict[str, float]" = {}
        for candidate in PATH_MODELS.values():
            n_params = len(candidate.parameter_names)
            k = n_params * len(units) + 1
            if n_total - k - 1 < 1:
                scores[candidate.name] = np.nan
                continue
            rss = 0.0
            try:
                for unit in units:
                    mask = i_arr == unit
                    x_unit, y_unit = x_arr[mask], y_arr[mask]
                    if len(np.unique(x_unit)) < n_params:
                        raise ValueError("too few distinct times")
                    params = candidate.fit(x_unit, y_unit)
                    residuals = y_unit - candidate.path(x_unit, *params)
                    if not np.isfinite(residuals).all():
                        raise ValueError("non-finite fit")
                    rss += float(residuals @ residuals)
            except Exception:
                scores[candidate.name] = np.nan
                continue
            rss = max(rss, rss_floor)
            scores[candidate.name] = (
                n_total * np.log(rss / n_total)
                + 2.0 * k
                + 2.0 * k * (k + 1.0) / (n_total - k - 1.0)
            )

        finite = {
            name: score for name, score in scores.items() if np.isfinite(score)
        }
        if not finite:
            raise ValueError(
                "path='best' could not fit any registered path model to "
                "every unit's measurements"
            )
        best_name = min(finite, key=lambda name: finite[name])
        best_model = next(
            model for model in PATH_MODELS.values() if model.name == best_name
        )
        return best_model, scores

    @renamed_arguments(x="x_col", y="y_col", i="i_col")
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        i_col: str = "i",
        Z_cols: "str | list[str] | None" = None,
        **fit_kwargs: Any,
    ) -> DegradationModel:
        """
        Fit a degradation analysis model from a DataFrame.

        The column arguments end in ``_col`` (``_cols`` for a list), as in
        every ``fit_from_df`` (principle 21); their v0.21 names ``x``,
        ``y`` and ``i`` still work, with a ``DeprecationWarning``, until
        v0.23.

        Parameters
        ----------
        df : DataFrame
            DataFrame with the degradation data.
        x_col : str, optional
            Column of the measurement times. Defaults to ``"x"``.
        y_col : str, optional
            Column of the degradation measurements. Defaults to
            ``"y"``.
        i_col : str, optional
            Column of the unit identifiers. Defaults to ``"i"``.
        Z_cols : str or list of str, optional
            Column(s) of the stress covariates for accelerated degradation
            testing. When given, the selected columns are passed as ``Z`` to
            :meth:`fit`, fitting a covariate (ADT) life model. Their names
            are recorded on the model as ``Z_cols`` (and kept by
            ``to_dict``), so every method that takes ``Z`` also takes a
            DataFrame and selects these columns by name.
        **fit_kwargs
            Remaining arguments passed to :meth:`fit`: ``threshold``
            (required), and optionally ``path``, ``distribution``,
            ``how``, ``population_method``, ``links``, ``acceleration``
            and ``stress_ref``.

        Returns
        -------
        DegradationModel
            The fitted degradation model.
        """
        cols = None
        if Z_cols is not None:
            cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
            fit_kwargs["Z"] = df[cols].to_numpy()
        model = self.fit(
            df[x_col].to_numpy(),
            df[y_col].to_numpy(),
            df[i_col].to_numpy(),
            **fit_kwargs,
        )
        # The names are kept so the model reads a DataFrame Z by them;
        # without them it would refuse one and tell the user to fit with
        # fit_from_df -- which they had done.
        model.Z_cols = cols
        if cols is not None and isinstance(
            model.life_model, ParametricRegressionModel
        ):
            # so ``model.life_model`` reads a DataFrame by name too
            model.life_model.feature_names = cols
        return model

    @staticmethod
    def _handle_Z(
        Z: npt.ArrayLike, i_arr: npt.NDArray, units: npt.NDArray
    ) -> npt.NDArray:
        """
        Reduce a per-measurement covariate array to one row per unit.

        ``Z`` is aligned to the measurement arrays (one row per measurement,
        like ``x``/``y``/``i``); a unit is tested at a single stress, so ``Z``
        must be constant within each unit. Returns a ``(n_units, n_cov)`` array
        aligned to ``units``.
        """
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.ndim != 2:
            raise ValueError("Z must be one or two dimensional")
        if len(Z_arr) != len(i_arr):
            raise ValueError(
                "Z must have one row per measurement (same length as x, y, "
                "and i); got {} rows for {} measurements".format(
                    len(Z_arr), len(i_arr)
                )
            )
        if not np.isfinite(Z_arr).all():
            raise ValueError("Z must contain only finite values")

        Z_units = np.empty((len(units), Z_arr.shape[1]))
        for idx, unit in enumerate(units):
            rows = Z_arr[i_arr == unit]
            if not np.allclose(rows, rows[0]):
                raise ValueError(
                    "Z must be constant within each unit (unit {} has "
                    "varying covariates); a unit is tested at a single "
                    "stress. For a step-stress test, where a unit's stress "
                    "changes during the test, fit with "
                    "acceleration='clock'".format(unit)
                )
            Z_units[idx] = rows[0]
        # With one stress level (or a constant covariate, or one that is a
        # combination of the others) the stress effect is confounded with
        # the intercept: the regression life fit would return an arbitrary
        # coefficient, and ``links`` would split the log rate into an
        # invented stress effect. The clock and process fitters refuse this too.
        design = np.column_stack([np.ones(len(Z_units)), Z_units])
        if np.linalg.matrix_rank(design) < Z_units.shape[1] + 1:
            raise ValueError(
                "the stress effect cannot be estimated: Z needs at least two "
                "distinct stress levels across the units, and no covariate "
                "may be constant or a combination of the others. Without "
                "stress variation, fit without Z."
            )
        return Z_units

    @staticmethod
    def _handle_xyi(
        x: npt.ArrayLike, y: npt.ArrayLike, i: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return validate_xy(x, y, i)


DegradationAnalysis = DegradationAnalysis_()
