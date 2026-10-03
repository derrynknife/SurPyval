"""The fitted degradation model.

:class:`DegradationModel` is what ``DegradationAnalysis.fit`` returns (see
:mod:`surpyval.degradation.degradation_analysis`): the per-unit fitted
paths, their pseudo failure times, the life distribution fitted to them
and, where fitted, the population of path parameters. It predicts the
degradation path, a unit's failure time and remaining life, and the life
distribution's functions and bounds.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)
from surpyval.univariate.regression.tvc_schedule import StepSchedule
from surpyval.utils.linalg import psd_precision, psd_root
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    alpha_ci_error,
    check_option,
    option_error,
)

from ._bounds import (
    analytic_cb,
    bootstrap_cb,
    life_parameter_covariance,
)
from ._clock import (
    HistoryClock,
    StressClock,
    covariates_by_name,
    stress_row,
)
from ._measurements import validate_xy
from .path_models import PathModel, get_path_model, path_model_key
from .rul import InducedFailureDistribution, RULPrediction
from .stress import LinkedPathModel, stress_design


def _optional_list(arr: "npt.NDArray | None") -> "list | None":
    """``to_dict`` helper: an optional array as a nested list."""
    return None if arr is None else np.asarray(arr, dtype=float).tolist()


def _optional_array(value: "list | None") -> "npt.NDArray | None":
    """``from_dict`` helper: the inverse of :func:`_optional_list`."""
    return None if value is None else np.array(value, dtype=float)


def _life_fitter(life_model: Any) -> Any:
    """The fitter that produced ``life_model``, for refitting it.

    A regression model keeps its regression fitter as ``model``; a plain
    parametric model's fitter is its ``dist``. ``None`` if neither is
    there.
    """
    fitter = getattr(life_model, "model", None)
    if fitter is not None and _is_regression_fitter(fitter):
        return fitter
    return getattr(life_model, "dist", None)


def _is_regression_fitter(fitter: Any) -> bool:
    """True if ``fitter.fit`` takes a covariate matrix ``Z`` (i.e. it is one of
    the regression fitters -- AFT, PH, PO, additive hazards, accelerated
    life)."""
    try:
        return "Z" in inspect.signature(fitter.fit).parameters
    except (TypeError, ValueError):
        return False


def _path_at(
    path_model: PathModel, t: float, params: npt.NDArray
) -> npt.NDArray:
    """
    The path at the time ``t`` for each row of ``params`` (one parameter
    vector per row). Built-in paths broadcast over parameter arrays; a
    custom path that does not is evaluated row by row.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    with np.errstate(all="ignore"):
        try:
            out = np.asarray(path_model.path(t, *params.T), dtype=float)
            if out.shape == (len(params),):
                return out
        except Exception:
            pass
        return np.array(
            [float(np.ravel(path_model.path(t, *row))[0]) for row in params]
        )


def _failure_side(
    path_model: PathModel,
    params: npt.NDArray,
    crossings: npt.NDArray,
    threshold: float,
    y: npt.NDArray,
) -> float:
    """
    The side of the threshold a failed unit is on: ``+1`` when degradation
    rises through the threshold (failure is ``y >= threshold``), ``-1``
    when it falls through it.

    Read from the direction in which the paths of the units that do cross
    (``params`` rows with positive ``crossings``) pass through the
    threshold -- the majority, by a forward difference just after each
    crossing. Without a majority, the threshold's position relative to the
    data decides (a threshold above the typical measurement is reached by
    rising).
    """
    signs = []
    for row, t in zip(np.atleast_2d(params), np.atleast_1d(crossings)):
        with np.errstate(all="ignore"):
            before = float(np.ravel(path_model.path(t, *row))[0])
            after = float(
                np.ravel(path_model.path(t * (1.0 + 1e-6) + 1e-12, *row))[0]
            )
        signs.append(np.sign(after - before))
    total = float(np.nansum(signs))
    if total != 0:
        return float(np.sign(total))
    return 1.0 if threshold >= float(np.median(y)) else -1.0


def _quantiles(samples: npt.NDArray, q: "list[float]") -> npt.NDArray:
    """
    ``np.quantile`` of samples that may hold ``inf`` (paths that never
    reach the threshold): a quantile reaching into the ``inf`` mass is
    ``inf``. Plain ``np.quantile`` interpolates ``inf - inf`` there and
    returns ``nan`` with a RuntimeWarning.
    """
    samples = np.asarray(samples, dtype=float)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return np.full(len(q), np.inf)
    big = np.finfo(float).max
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.quantile(np.where(np.isposinf(samples), big, samples), q)
    # an interpolation that touches the stand-in exceeds every finite draw
    return np.where(out > finite.max(), np.inf, out)


class DegradationModel(SerialisableMixin):
    """
    A fitted degradation analysis model.

    This is the model object returned by
    :meth:`DegradationAnalysis.fit`. It holds the per-unit fitted
    degradation paths, the pseudo failure times extrapolated from them,
    and the lifetime distribution fitted to those pseudo failure times.
    The usual lifetime functions (``sf``, ``ff``, ``df``, ``hf``,
    ``Hf``, ``qf``, ``mean``, ``random``) are forwarded to the fitted
    life model, and the failure time of a new, partially observed unit
    can be estimated from its trajectory with
    :meth:`predict_failure_time` / :meth:`predict_remaining_life`.

    Parameters
    ----------
    x, y, i : ndarray
        The degradation data: measurement times, measurements, and the
        unit each measurement belongs to.
    units : ndarray
        The distinct unit identifiers.
    threshold : float
        The degradation level at which a unit is considered failed.
    path_model : PathModel
        The degradation path model fitted to each unit.
    path_params : ndarray
        Per-unit fitted path parameters, one row per entry of
        ``units``.
    pseudo_failure_times : ndarray
        Per-unit pseudo failure time: the extrapolated threshold
        crossing time; the unit's last observed time for a unit whose
        path never reaches the threshold; its first (positive)
        measurement time for a unit already past the threshold there.
    c : ndarray
        Per-unit censor flags: 0 where the fitted path crosses the
        threshold at a positive time, 1 (right censored) where it never
        reaches it, -1 (left censored: failed before its first
        measurement) where the path is already past the threshold at
        the first measurement, having crossed at or before time zero.
    life_model : Parametric
        The lifetime distribution fitted to the pseudo failure times.
    measurement_var : float
        Pooled estimate of the measurement-error variance around the
        per-unit paths (the per-unit residual sums of squares over the
        total residual degrees of freedom). Zero when every unit has
        exactly as many measurements as path parameters.
    path_param_mean : ndarray
        Mean of the per-unit fitted path parameters: the estimated
        population mean path.
    path_param_cov : ndarray
        Noise-corrected estimate of the *between-unit* covariance of
        the true path parameters (Lu-Meeker two-stage): the sample
        covariance of the per-unit estimates minus the average
        least-squares estimation covariance, projected onto the
        positive semi-definite cone.
    path_param_sample_cov : ndarray
        The raw (uncorrected) sample covariance of the per-unit
        fitted path parameters. This overstates the between-unit
        variability because each per-unit estimate also carries
        least-squares estimation noise.
    population_method : str
        How the population estimates (``measurement_var``,
        ``path_param_mean``, ``path_param_cov``) were obtained:
        ``"moments"`` (two-stage correction) or ``"reml"``.
    path_selection : dict or None
        When fitted with ``path="best"``, the AICc score of every
        candidate path model (``nan`` for candidates that could not be
        fitted to every unit); ``None`` otherwise. The fitted
        ``path_model`` is the candidate with the smallest score. The
        keys are the models' display names (``"Offset Exponential"``),
        not the ``path=`` strings.
    Z : ndarray or None
        The stresses of an accelerated model: one row per unit (aligned
        to ``units``) for a model fitted with ``Z`` alone or with
        ``links``; for a step-stress (``acceleration="clock"``) model the
        stress rows as given, one per measurement (aligned to ``x``).
        ``None`` for a model fitted without stress.
    links : dict or None
        When the path parameters were modelled against stress
        (``links`` given to :meth:`DegradationAnalysis.fit`), the
        stress-dependent parameters and their links; ``None``
        otherwise. With ``links`` the population of path parameters is
        stress-conditional, on the link scale: ``eta_i = D(z_i) gamma
        + u_i`` with ``u_i ~ MVN(0, Sigma)`` and ``theta_i = h(eta_i)``.
    path_param_fixed : ndarray or None
        The fixed effects ``gamma`` of the stress-conditional population
        model, labelled by ``path_param_fixed_names``: for every path
        parameter its link-scale intercept, followed (for the
        stress-dependent ones) by its coefficient on each covariate.
    path_param_fixed_names : list of str or None
        Labels for ``path_param_fixed``: the link-scale parameter name
        (``"log(b)"`` for a log link) and ``"<name>:Z<j>"`` for the
        coefficient on covariate ``j``.
    path_param_link_cov : ndarray or None
        The between-unit covariance ``Sigma`` of the link-scale path
        parameters *given* the stress -- the scatter left after the
        stress effect is removed, unlike the pooled ``path_param_cov``
        which mixes the stress levels.
    acceleration : str or None
        ``"clock"`` when stress was modelled as speeding up the clock of
        every unit's path (``acceleration="clock"`` in
        :meth:`DegradationAnalysis.fit`); ``None`` otherwise. The path
        parameters, their population, the pseudo failure times and the
        life model are then all on the reference-stress clock, and ``Z``
        holds the stress rows aligned to ``x``.
    gamma : ndarray or None
        The stress coefficients of the clock: a unit at stress ``z`` ages
        ``exp(gamma' (z - stress_ref))`` times faster than at the
        reference stress.
    stress_ref : ndarray or None
        The reference stress of the clock.
    Z_cols : list of str or None
        The covariate columns of a model fitted with
        :meth:`DegradationAnalysis.fit_from_df`: every method that takes
        ``Z`` then also takes a DataFrame and selects these columns by
        name. ``None`` for a model fitted from arrays, which refuses a
        DataFrame.

    Examples
    --------
    Eight units, each degrading linearly at its own rate and measured
    ten times, fail when the measurement reaches 450:

    >>> import numpy as np
    >>> from surpyval.degradation import DegradationAnalysis
    >>> rng = np.random.default_rng(1)
    >>> x = np.tile(np.arange(100.0, 1100.0, 100.0), 8)
    >>> i = np.repeat(np.arange(8), 10)
    >>> a = np.repeat(rng.normal(10.0, 3.0, 8), 10)
    >>> b = np.repeat(rng.normal(0.3, 0.05, 8), 10)
    >>> y = a + b * x + rng.normal(0, 3.0, x.size)
    >>> model = DegradationAnalysis.fit(x, y, i, threshold=450)
    >>> model.pseudo_failure_times.round(1)
    array([1393.1, 1377.5, 1455.4, 1358.9, 1673.9, 1495.6, 1592.4, 1334. ])

    A Weibull is fitted to those, and the lifetime functions use it:

    >>> model.life_model.params.round(3)
    array([1515.035,   12.785])
    >>> model.sf([1200, 1500]).round(4)
    array([0.9505, 0.4147])
    """

    x: npt.NDArray
    y: npt.NDArray
    i: npt.NDArray
    units: npt.NDArray
    threshold: float
    path_model: PathModel
    path_params: npt.NDArray
    pseudo_failure_times: npt.NDArray
    c: npt.NDArray
    #: A plain ``Parametric`` life model, or the regression model
    #: for an accelerated (covariate) fit.
    life_model: Any
    measurement_var: float
    path_param_mean: npt.NDArray
    path_param_cov: npt.NDArray
    path_param_sample_cov: npt.NDArray
    population_method: str
    path_selection: "dict[str, float] | None"
    #: Per-unit covariates when fitted as an accelerated-degradation model
    #: (``Z`` given to :meth:`DegradationAnalysis.fit`); ``None`` otherwise.
    Z: "npt.NDArray | None"
    links: "dict[str, str] | None"
    path_param_fixed: "npt.NDArray | None"
    path_param_fixed_names: "list[str] | None"
    path_param_link_cov: "npt.NDArray | None"
    acceleration: "str | None"
    gamma: "npt.NDArray | None"
    stress_ref: "npt.NDArray | None"
    Z_cols: "list[str] | None"
    # Recorded after construction so the bootstrap bounds can rerun the fit.
    _distribution: Any
    _how: str

    def __init__(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        i: npt.NDArray,
        units: npt.NDArray,
        threshold: float,
        path_model: Any,
        path_params: npt.NDArray,
        pseudo_failure_times: npt.NDArray,
        c: npt.NDArray,
        life_model: Any,
        measurement_var: float,
        path_param_mean: npt.NDArray,
        path_param_cov: npt.NDArray,
        path_param_sample_cov: npt.NDArray,
        population_method: str,
        path_selection: "dict | None" = None,
        Z: "npt.NDArray | None" = None,
        links: "dict[str, str] | None" = None,
        path_param_fixed: "npt.NDArray | None" = None,
        path_param_fixed_names: "list[str] | None" = None,
        path_param_link_cov: "npt.NDArray | None" = None,
        acceleration: "str | None" = None,
        gamma: "npt.ArrayLike | None" = None,
        stress_ref: "npt.ArrayLike | None" = None,
        Z_cols: "list[str] | None" = None,
    ) -> None:
        self.x = x
        self.y = y
        self.i = i
        self.units = units
        self.threshold = threshold
        self.path_model = path_model
        self.path_params = path_params
        self.pseudo_failure_times = pseudo_failure_times
        self.c = c
        self.life_model = life_model
        self.measurement_var = measurement_var
        self.path_param_mean = path_param_mean
        self.path_param_cov = path_param_cov
        self.path_param_sample_cov = path_param_sample_cov
        self.population_method = population_method
        self.path_selection = path_selection
        self.Z = Z
        self.links = links
        self.path_param_fixed = path_param_fixed
        self.path_param_fixed_names = path_param_fixed_names
        self.path_param_link_cov = path_param_link_cov
        self.acceleration = acceleration
        self.gamma = _optional_array(
            None if gamma is None else np.atleast_1d(gamma).tolist()
        )
        self.stress_ref = _optional_array(
            None if stress_ref is None else np.atleast_1d(stress_ref).tolist()
        )
        self.Z_cols = None if Z_cols is None else list(Z_cols)
        self._unit_index = {unit: idx for idx, unit in enumerate(units)}

    # -- serialisation -----------------------------------------------------

    @staticmethod
    def _life_model_to_dict(life_model: Any) -> dict:
        out = life_model.to_dict()
        out["_life_class"] = (
            "ParametricRegressionModel"
            if isinstance(life_model, ParametricRegressionModel)
            else "Parametric"
        )
        return out

    @staticmethod
    def _life_model_from_dict(life_dict: dict) -> Any:
        life_dict = dict(life_dict)
        life_class = life_dict.pop("_life_class")
        if life_class == "ParametricRegressionModel":
            return ParametricRegressionModel.from_dict(life_dict)
        return Parametric.from_dict(life_dict)

    def to_dict(self) -> dict:
        """
        Serialise this fitted degradation model to a plain, JSON-serialisable
        dict.

        Everything needed to rebuild the model is stored: the raw measurement
        data, the path model (by name) and its per-unit fitted parameters, the
        pseudo failure times and censor flags, the fitted life model (its own
        ``to_dict``), and the population summaries. The restored model
        reproduces the life predictions and per-unit paths, and (because the
        raw data is kept) its bootstrap confidence bounds too.

        See Also
        --------
        from_dict, to_json, from_json
        """
        return stamp_schema(
            {
                "model": "DegradationModel",
                "x": np.asarray(self.x, dtype=float).tolist(),
                "y": np.asarray(self.y, dtype=float).tolist(),
                "i": np.asarray(self.i).tolist(),
                "units": np.asarray(self.units).tolist(),
                "threshold": float(self.threshold),
                # The registry key (what ``path=`` accepts), not the
                # display name: the two differ for some models
                # ("offset-exponential" vs "Offset Exponential").
                "path_model": path_model_key(self.path_model),
                "path_params": np.asarray(
                    self.path_params, dtype=float
                ).tolist(),
                "pseudo_failure_times": np.asarray(
                    self.pseudo_failure_times, dtype=float
                ).tolist(),
                "c": np.asarray(self.c).tolist(),
                "life_model": self._life_model_to_dict(self.life_model),
                "measurement_var": float(self.measurement_var),
                "path_param_mean": np.asarray(
                    self.path_param_mean, dtype=float
                ).tolist(),
                "path_param_cov": np.asarray(
                    self.path_param_cov, dtype=float
                ).tolist(),
                "path_param_sample_cov": np.asarray(
                    self.path_param_sample_cov, dtype=float
                ).tolist(),
                "population_method": self.population_method,
                "path_selection": self.path_selection,
                "Z": None if self.Z is None else np.asarray(self.Z).tolist(),
                "how": self._how,
                "links": None if self.links is None else dict(self.links),
                "path_param_fixed": _optional_list(self.path_param_fixed),
                "path_param_fixed_names": (
                    None
                    if self.path_param_fixed_names is None
                    else list(self.path_param_fixed_names)
                ),
                "path_param_link_cov": _optional_list(
                    self.path_param_link_cov
                ),
                **self._clock_dict(),
                # only for a model fitted with named covariates, so other
                # dicts are unchanged (an older reader ignores the key)
                **({} if self.Z_cols is None else {"Z_cols": self.Z_cols}),
            }
        )

    def _clock_dict(self) -> dict:
        """The clock's entries for :meth:`to_dict` (none for other models,
        whose dicts are unchanged)."""
        if self.acceleration is None:
            return {}
        return {
            "acceleration": self.acceleration,
            "gamma": _optional_list(self.gamma),
            "stress_ref": _optional_list(self.stress_ref),
        }

    @classmethod
    def from_dict(cls, model_dict: dict) -> "DegradationModel":
        """
        Rebuild a degradation model from a :meth:`to_dict` dictionary.

        The path model is resolved by name (its ``PATH_MODELS`` key, or
        the display name that older dictionaries stored) and the life
        model by its own ``from_dict``; both are restricted to the known
        types.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "DegradationModel", "a degradation model"
        )
        Z = model_dict.get("Z")
        out = cls(
            x=np.array(model_dict["x"], dtype=float),
            y=np.array(model_dict["y"], dtype=float),
            i=np.array(model_dict["i"]),
            units=np.array(model_dict["units"]),
            threshold=float(model_dict["threshold"]),
            path_model=get_path_model(model_dict["path_model"]),
            path_params=np.array(model_dict["path_params"], dtype=float),
            pseudo_failure_times=np.array(
                model_dict["pseudo_failure_times"], dtype=float
            ),
            c=np.array(model_dict["c"]),
            life_model=cls._life_model_from_dict(model_dict["life_model"]),
            measurement_var=float(model_dict["measurement_var"]),
            path_param_mean=np.array(
                model_dict["path_param_mean"], dtype=float
            ),
            path_param_cov=np.array(model_dict["path_param_cov"], dtype=float),
            path_param_sample_cov=np.array(
                model_dict["path_param_sample_cov"], dtype=float
            ),
            population_method=model_dict["population_method"],
            path_selection=model_dict.get("path_selection"),
            Z=None if Z is None else np.array(Z, dtype=float),
            links=model_dict.get("links"),
            path_param_fixed=_optional_array(
                model_dict.get("path_param_fixed")
            ),
            path_param_fixed_names=model_dict.get("path_param_fixed_names"),
            path_param_link_cov=_optional_array(
                model_dict.get("path_param_link_cov")
            ),
            acceleration=model_dict.get("acceleration"),
            gamma=model_dict.get("gamma"),
            stress_ref=model_dict.get("stress_ref"),
            Z_cols=model_dict.get("Z_cols"),
        )
        # Recorded so bootstrap bounds can rerun the pipeline. The fitter is
        # not serialised as such, but the restored life model carries it:
        # a regression life model keeps its regression fitter (e.g.
        # ``WeibullPH`` or ``AFT(Weibull)``), a plain one its distribution.
        # Leaving this as None made every bootstrap refit fail after a
        # reload.
        out._distribution = _life_fitter(out.life_model)
        out._how = model_dict.get("how", "MLE")
        return out

    @property
    def is_accelerated(self) -> bool:
        """True when life depends on stress: the life model is a covariate
        (ADT) regression model, or stress accelerates the clock
        (``acceleration="clock"``)."""
        return self._is_clock or isinstance(
            self.life_model, ParametricRegressionModel
        )

    @property
    def _is_clock(self) -> bool:
        return self.acceleration == "clock"

    @property
    def _failure_side(self) -> float:
        """``+1`` if failure is degradation at or above the threshold,
        ``-1`` if at or below it (from the units that cross)."""
        events = self.c == 0
        return _failure_side(
            self.path_model,
            self.path_params[events],
            self.pseudo_failure_times[events],
            self.threshold,
            self.y,
        )

    @property
    def _start_time(self) -> float:
        """The earliest positive measurement time of the training data, on
        the path's clock: a path already past the threshold by then has
        failed "at time zero"."""
        if self._is_clock:
            taus = np.concatenate(
                [
                    self._unit_clock(unit, self.x[self.i == unit])
                    for unit in self.units
                ]
            )
        else:
            taus = np.asarray(self.x, dtype=float)
        positive = taus[taus > 0]
        return float(positive.min()) if positive.size else np.nan

    def _past_threshold(
        self, t_ref: float, params: npt.NDArray
    ) -> npt.NDArray:
        """For each row of path parameters, whether the path is already on
        the failed side of the threshold at the time ``t_ref``."""
        if not np.isfinite(t_ref):
            return np.zeros(len(np.atleast_2d(params)), dtype=bool)
        level = _path_at(self.path_model, t_ref, params)
        with np.errstate(invalid="ignore"):
            return self._failure_side * (level - self.threshold) >= 0

    def acceleration_factor(self, Z: Any) -> float:
        """
        How much faster a unit ages at stress ``Z`` than at the reference
        stress: ``exp(gamma' (z - stress_ref))``, for a model fitted with
        ``acceleration="clock"``.

        A unit held at ``Z`` degrades along its path this many times
        faster, and a life at the reference stress divides by it to give
        the life at ``Z``.

        Parameters
        ----------
        Z : array like or DataFrame
            One stress row (``nan`` for a missing value gives ``nan``); a
            one-row DataFrame for a model fitted with ``fit_from_df``.
        """
        if not self._is_clock or self.gamma is None:
            raise ValueError(
                "acceleration_factor is defined for a model fitted with "
                "acceleration='clock'"
            )
        assert self.stress_ref is not None
        z = stress_row(self._covariates(Z), self.gamma.size, allow_nan=True)
        return float(np.exp(self.gamma @ (z - self.stress_ref)))

    def _covariates(self, Z: Any) -> Any:
        """``Z`` as an array: a DataFrame is read by the covariate names
        recorded by ``fit_from_df`` (and refused without them)."""
        return covariates_by_name(
            Z, self.Z_cols, "DegradationAnalysis.fit_from_df"
        )

    def _clock(self, Z: Any) -> StressClock:
        """The clock for stress ``Z`` (a row or a StepSchedule)."""
        if Z is None:
            raise ValueError(
                "This step-stress (acceleration='clock') model's life "
                "depends on stress; pass Z -- one stress row for a constant "
                "stress, or a StepSchedule for a stress profile."
            )
        assert self.gamma is not None
        return StressClock(
            self.acceleration_factor, self.gamma.size, self._covariates(Z)
        )

    def _unit_clock(self, unit: Any, t: npt.NDArray) -> npt.NDArray:
        """
        A training unit's reference-stress time at calendar times ``t``,
        from its recorded stress history; beyond its last measurement the
        last stress is held.
        """
        assert self.Z is not None
        mask = np.flatnonzero(self.i == unit)
        order = mask[np.argsort(self.x[mask], kind="stable")]
        x_unit = self.x[order]
        rates = np.array([self.acceleration_factor(z) for z in self.Z[order]])
        tau = np.cumsum(np.diff(np.concatenate([[0.0], x_unit])) * rates)
        knots_t = np.concatenate([[0.0], x_unit])
        knots_tau = np.concatenate([[0.0], tau])
        t = np.asarray(t, dtype=float)
        out = np.interp(t, knots_t, knots_tau)
        beyond = t > x_unit[-1]
        return np.where(beyond, tau[-1] + rates[-1] * (t - x_unit[-1]), out)

    def _unit_calendar_time(self, unit: Any, tau: float) -> float:
        """The calendar time at which a training unit's clock reads ``tau``
        (the inverse of :meth:`_unit_clock`)."""
        assert self.Z is not None
        mask = np.flatnonzero(self.i == unit)
        order = mask[np.argsort(self.x[mask], kind="stable")]
        x_unit = self.x[order]
        knots_t = np.concatenate([[0.0], x_unit])
        knots_tau = self._unit_clock(unit, knots_t)
        if tau <= knots_tau[-1]:
            return float(np.interp(tau, knots_tau, knots_t))
        rate = self.acceleration_factor(self.Z[order][-1])
        return float(x_unit[-1] + (tau - knots_tau[-1]) / rate)

    def _history_clock(
        self, x: npt.NDArray, Z: Any, Z_future: Any
    ) -> "HistoryClock | None":
        """
        A new unit's clock for the trajectory methods: from its measured
        stress history ``Z`` and, after its last measurement, ``Z_future``.
        ``None`` for a model without a clock, which refuses both.
        """
        if not self._is_clock:
            if Z_future is not None:
                raise ValueError(
                    "Z_future is the stress from now on for a step-stress "
                    "(acceleration='clock') model; this model has no clock"
                )
            return None
        if Z is None:
            raise ValueError(
                "This step-stress (acceleration='clock') model needs the new "
                "unit's stress history: pass Z with one row per measurement "
                "(the stress over the interval ending at it, as at fit), or "
                "one row for a constant stress"
            )
        assert self.gamma is not None
        q = self.gamma.size
        Z_arr = np.asarray(self._covariates(Z), dtype=float)
        if Z_arr.size == q:
            Z_arr = np.tile(Z_arr.reshape(1, q), (len(x), 1))
        elif Z_arr.ndim == 1 and q == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.shape != (len(x), q):
            raise ValueError(
                "Z must have one row of {} covariate(s) per measurement, or "
                "be a single row; got shape {} for {} measurements".format(
                    q, np.shape(Z), len(x)
                )
            )
        if not np.isfinite(Z_arr).all():
            raise ValueError("Z must contain only finite values")
        Z_future = self._covariates(Z_future)
        if Z_future is not None and not isinstance(Z_future, StepSchedule):
            # the stress this one unit will run at: a missing value is
            # refused, not carried through as nan
            stress_row(Z_future, q)
        if (x < 0).any():
            raise ValueError(
                "With a step-stress model the measurement times must be "
                "non-negative: the unit's clock starts at time zero"
            )
        return HistoryClock(x, Z_arr, self.acceleration_factor, q, Z_future)

    @property
    def _reg(self) -> ParametricRegressionModel:
        """The life model viewed as a regression model (accelerated only)."""
        return cast(ParametricRegressionModel, self.life_model)

    def _predict_Z(self, Z: Any) -> Any:
        """Validate the covariate argument for the prediction methods: an
        accelerated model needs a stress vector ``Z``; a plain model rejects
        one."""
        if self.is_accelerated:
            if Z is None:
                raise ValueError(
                    "This is an accelerated-degradation (covariate) model; "
                    "pass the covariate vector Z (the stress conditions) to "
                    "predict life."
                )
            return self._covariates(Z)
        if Z is not None:
            raise ValueError(
                "This degradation model has no covariates; do not pass Z."
            )
        return None

    def path(self, x: npt.ArrayLike, unit: Any) -> npt.NDArray:
        """
        Evaluate the fitted degradation path of ``unit`` at ``x``.

        For a step-stress (``acceleration="clock"``) model ``x`` is
        calendar time: the path is evaluated on the unit's
        reference-stress clock, from its recorded stress history (the last
        stress held beyond its last measurement).
        """
        if unit not in self._unit_index:
            raise ValueError(
                "unit {!r} is not one of the model's units; units holds "
                "the identifiers it was fitted with".format(unit)
            )
        idx = self._unit_index[unit]
        if self._is_clock:
            x = self._unit_clock(unit, np.asarray(x, dtype=float))
        return self.path_model.path(x, *self.path_params[idx])

    # -- the stress-conditional path population (``links``) ----------------

    def _stress_row(self, Z: Any, allow_nan: bool = False) -> npt.NDArray:
        """Validate one stress row for the stress-conditional population;
        ``allow_nan`` lets a missing value through (to give ``nan``)."""
        if (
            self.links is None
            or self.path_param_fixed is None
            or self.Z is None
        ):
            raise ValueError(
                "This model's path parameters were not modelled against "
                "stress, so there is no stress-conditional path population; "
                "fit with links (e.g. links={'b': 'log'}) alongside Z to get "
                "one."
            )
        if Z is None:
            raise ValueError(
                "This model's path parameters depend on stress; pass the "
                "stress vector Z at which to predict."
            )
        z = np.asarray(self._covariates(Z), dtype=float)
        if z.ndim == 2 and z.shape[0] == 1:
            z = z[0]
        z = np.atleast_1d(z)
        n_cov = self.Z.shape[1]
        if z.shape != (n_cov,):
            raise ValueError(
                "Z must be a single stress row with {} covariate(s), like one "
                "row of the Z the model was fitted with; got shape {}".format(
                    n_cov, z.shape
                )
            )
        if not np.isfinite(z[~np.isnan(z)] if allow_nan else z).all():
            raise ValueError("Z must contain only finite values")
        return z

    def _stress_prior(
        self, Z: Any, allow_nan: bool = False
    ) -> tuple[LinkedPathModel, npt.NDArray, npt.NDArray]:
        """The link-scale path population at stress ``Z``: the linked path
        model, the mean ``D(z) gamma`` and the covariance ``Sigma``."""
        z = self._stress_row(Z, allow_nan)
        assert self.links is not None and self.path_param_fixed is not None
        design = stress_design(z, self.links, self.path_model.parameter_names)
        mean = design @ np.asarray(self.path_param_fixed, dtype=float)
        cov = np.asarray(self.path_param_link_cov, dtype=float)
        return LinkedPathModel(self.path_model, self.links), mean, cov

    def path_param_link_mean(self, Z: Any) -> npt.NDArray:
        r"""
        Mean of the path parameters at stress ``Z``, on the link scale.

        For a model fitted with ``links`` this is
        :math:`D(z)\,\gamma` -- the population mean of the link-scale
        path parameters ``eta`` for a unit tested at stress ``Z``. The
        parameters are in path order, named like the intercepts in
        ``path_param_fixed_names`` (``"log(b)"`` for a log-linked
        ``b``). The between-unit covariance around it is
        ``path_param_link_cov``, the same at every stress.

        Parameters
        ----------
        Z : array like or DataFrame
            One stress row, with as many covariates as the model was
            fitted with (a one-row DataFrame for a model fitted with
            ``fit_from_df``). A missing (``nan``) covariate makes the
            parameters that depend on it ``nan``.

        Returns
        -------
        ndarray
            The link-scale mean path parameters at ``Z``.
        """
        return self._stress_prior(Z, allow_nan=True)[1]

    def path_param_median(self, Z: Any) -> npt.NDArray:
        r"""
        Median path parameters at stress ``Z``, on their natural scale.

        Each link is monotone and each link-scale parameter is normal,
        so mapping the link-scale mean through the links gives every
        parameter's population median exactly: :math:`h(D(z)\,\gamma)`.
        For a log-linked rate that is the geometric-mean rate at ``Z``
        (the rate's population mean is larger, by the log-normal
        factor). Evaluate the typical path at a stress with
        ``model.path_model.path(t, *model.path_param_median(Z))``.

        Parameters
        ----------
        Z : array like or DataFrame
            One stress row, with as many covariates as the model was
            fitted with (a one-row DataFrame for a model fitted with
            ``fit_from_df``). A missing (``nan``) covariate makes the
            parameters that depend on it ``nan``.

        Returns
        -------
        ndarray
            The median path parameters at ``Z``, in path order.
        """
        linked, mean, _ = self._stress_prior(Z, allow_nan=True)
        return linked.to_natural(mean)

    def predict_failure_time(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        Z: Any = None,
        Z_future: Any = None,
    ) -> float:
        """
        Estimate the failure time of a new unit from its (partial)
        degradation trajectory.

        Fits this model's path model to the new unit's measurements
        and extrapolates the fitted path to this model's failure
        threshold, exactly as was done for each unit during fitting.

        Parameters
        ----------
        x : array like
            Times at which the new unit's measurements were taken.
        y : array like
            The new unit's degradation measurements.
        Z : array like, optional
            For a step-stress (``acceleration="clock"``) model, required:
            the new unit's stress history, one row per measurement (the
            stress over the interval ending at it), or one row for a
            constant stress. The path is fitted on the unit's clock.
        Z_future : array like or StepSchedule, optional
            For a step-stress model, the stress from the last measurement on:
            one row, or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            whose time zero is the last measurement. Defaults to holding the
            last stress.

        Returns
        -------
        float
            The time at which the new unit's fitted path reaches the
            threshold. This can be smaller than the last observed time
            if the trajectory has already crossed the threshold. A
            trajectory already past the threshold at its first
            measurement returns the non-positive time at which its fitted
            path crossed (``0`` if the path is past the threshold
            throughout, and for a step-stress model, whose clock starts
            at zero). Returns ``nan`` (with a warning) if the fitted path
            never reaches the threshold.
        """
        x_arr, y_arr = self._handle_new_trajectory(x, y)
        if Z is not None and not self._is_clock:
            raise ValueError(
                "predict_failure_time takes Z only for a step-stress "
                "(acceleration='clock') model"
            )
        clock = self._history_clock(x_arr, Z, Z_future)
        path_x = x_arr if clock is None else clock.tau
        params = self.path_model.fit(path_x, y_arr)
        t = float(self.path_model.inv_path(self.threshold, *params))
        if not (np.isfinite(t) and t > 0):
            positive = path_x[path_x > 0]
            if (
                positive.size
                and self._past_threshold(float(positive.min()), params)[0]
            ):
                # Already past the threshold at its first measurement: the
                # fitted path crossed at or before time zero. A plain model
                # reports that (non-positive) crossing time; the clock of a
                # step-stress model starts at zero, so it reports 0.
                if clock is not None or not np.isfinite(t):
                    return 0.0
                return t
            warnings.warn(
                "The fitted degradation path of the new trajectory never "
                "reaches the threshold {}; returning nan".format(
                    self.threshold
                ),
                stacklevel=2,
            )
            return float("nan")
        if clock is not None:
            return float(clock.calendar(t))
        return t

    def predict_remaining_life(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        Z: Any = None,
        Z_future: Any = None,
    ) -> float:
        """
        Estimate the remaining life of a new unit from its (partial)
        degradation trajectory.

        This is :meth:`predict_failure_time` minus the new unit's last
        observed time. A negative value means the fitted path crossed
        the threshold before the last observation (the unit is
        predicted to have already failed); ``nan`` (with a warning)
        means the fitted path never reaches the threshold. ``Z`` and
        ``Z_future`` are as for :meth:`predict_failure_time`.
        """
        x_arr, y_arr = self._handle_new_trajectory(x, y)
        return self.predict_failure_time(
            x_arr, y_arr, Z=Z, Z_future=Z_future
        ) - float(x_arr.max())

    def predict_rul(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        Z: Any = None,
        Z_future: Any = None,
        alpha_ci: float = 0.05,
        n_samples: int = 10_000,
        random_state: "int | None" = None,
    ) -> RULPrediction:
        """
        Bayesian remaining-useful-life prediction for a new unit.

        The population distribution of path parameters estimated at
        fit time (``path_param_mean``, ``path_param_cov``) is used as
        a prior, the new unit's measurements as the likelihood (with
        the pooled ``measurement_var`` as the noise variance), and the
        Gaussian posterior of the unit's path parameters is pushed
        through the threshold crossing by Monte Carlo. The posterior
        is exact (conjugate) for path models that are linear in their
        parameters, and an iterated-linearisation (Laplace)
        approximation otherwise.

        Compared to :meth:`predict_failure_time`, this shrinks short
        or noisy trajectories toward the population instead of
        trusting the raw extrapolation, works from a single
        measurement, and returns credible intervals. With many
        measurements the posterior concentrates on the least-squares
        fit and the two agree.

        Parameters
        ----------
        x : array like
            Times at which the new unit's measurements were taken.
            One or more measurements are required.
        y : array like
            The new unit's degradation measurements.
        Z : array like, optional
            The stress the new unit runs at. Required for a model whose
            path parameters were modelled against stress (fitted with
            ``links``): the prior is then the *stress-conditional*
            population, ``eta ~ N(D(z) gamma, Sigma)`` on the link
            scale, rather than the pooled population that mixes the
            stress levels. The posterior is taken on the link scale (so a
            log-linked rate stays positive) and pushed through the
            threshold crossing in the same way. Refused for a model
            without ``links``. For a step-stress
            (``acceleration="clock"``) model, required: the new unit's
            stress history, one row per measurement (the stress over the
            interval ending at it, as at fit), or one row for a constant
            stress. The posterior is then taken on the unit's clock
            against the reference-stress population, and each sampled
            reference-stress failure time is mapped back to calendar time
            along the unit's history and ``Z_future``.
        Z_future : array like or StepSchedule, optional
            For a step-stress model, the stress from the last measurement on:
            one row, or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            whose time zero is the last measurement. Defaults to holding the
            last stress. Refused for other models.
        alpha_ci : float, optional
            Significance level for the equal-tailed credible
            intervals, between 0 and 1. Defaults to 0.05 (95%
            intervals).
        n_samples : int, optional
            Number of Monte Carlo posterior samples (at least one).
            Defaults to 10,000.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for reproducible sampling. ``None`` (the default)
            seeds from numpy's global RNG, so ``np.random.seed`` controls it.

        Returns
        -------
        RULPrediction
            Posterior medians, credible intervals, failure
            probabilities, and the parameter posterior.

        Examples
        --------
        Eight units with their own start and rate, then a new unit seen
        three times:

        >>> import numpy as np
        >>> from surpyval.degradation import DegradationAnalysis
        >>> rng = np.random.default_rng(1)
        >>> x = np.tile(np.arange(100.0, 1100.0, 100.0), 8)
        >>> i = np.repeat(np.arange(8), 10)
        >>> a = np.repeat(rng.normal(10.0, 3.0, 8), 10)
        >>> b = np.repeat(rng.normal(0.3, 0.05, 8), 10)
        >>> y = a + b * x + rng.normal(0, 3.0, x.size)
        >>> model = DegradationAnalysis.fit(x, y, i, threshold=450)
        >>> pred = model.predict_rul(
        ...     [100.0, 200.0, 300.0], [42.0, 71.0, 99.0], random_state=0
        ... )
        >>> round(pred.failure_time), round(pred.rul)
        (1473, 1173)
        >>> [round(v) for v in pred.failure_time_interval]
        [1395, 1562]
        """
        # a numerically-zero variance (exact path fits) makes the
        # posterior degenerate; compare against the scale of y
        noise_floor = np.finfo(float).eps * float(np.mean(self.y**2))
        if not self.measurement_var > noise_floor:
            raise ValueError(
                "predict_rul requires a positive measurement variance, but "
                "the fitted model's measurement_var is 0 (every training "
                "unit's path fitted its measurements exactly, or there were "
                "no residual degrees of freedom); use predict_failure_time "
                "instead"
            )
        if not 0.0 < float(alpha_ci) < 1.0:
            raise alpha_ci_error(alpha_ci)
        if int(n_samples) < 1:
            raise ValueError(
                "n_samples must be a positive integer, got {!r}".format(
                    n_samples
                )
            )
        x_arr, y_arr, _ = validate_xy(x, y)
        clock = self._history_clock(x_arr, Z, Z_future)
        path_x = x_arr if clock is None else clock.tau
        self.path_model.check_data(path_x, y_arr)

        linked: "LinkedPathModel | None" = None
        if clock is not None or (Z is None and self.links is None):
            posterior_mean, posterior_cov = self._path_posterior(
                path_x,
                y_arr,
                self.path_model,
                self.path_param_mean,
                self.path_param_cov,
            )
        else:
            # the stress-conditional population is the prior; the update
            # runs on the link scale
            linked, prior_mean, prior_cov = self._stress_prior(Z)
            posterior_mean, posterior_cov = self._path_posterior(
                x_arr, y_arr, linked, prior_mean, prior_cov
            )

        rng = as_generator(random_state)
        theta_samples = rng.multivariate_normal(
            posterior_mean, posterior_cov, size=n_samples
        )
        if linked is not None:
            theta_samples = linked.to_natural(theta_samples)
        try:
            failure_times = np.asarray(
                self.path_model.inv_path(self.threshold, *theta_samples.T),
                dtype=float,
            )
            if failure_times.shape != (n_samples,):
                raise ValueError("inv_path did not broadcast")
        except Exception:
            # custom path models need not broadcast over parameter
            # arrays; fall back to a per-sample loop
            failure_times = np.array(
                [
                    float(self.path_model.inv_path(self.threshold, *theta))
                    for theta in theta_samples
                ]
            )
        reaches = np.isfinite(failure_times) & (failure_times > 0)
        # A draw whose path is already past the threshold at the unit's
        # first measurement crossed it at or before time zero: it has
        # failed (an atom at time zero), not "never fails".
        positive = path_x[path_x > 0]
        started = np.zeros(n_samples, dtype=bool)
        if positive.size and not reaches.all():
            started[~reaches] = self._past_threshold(
                float(positive.min()), theta_samples[~reaches]
            )
        failure_times = np.where(
            reaches, failure_times, np.where(started, 0.0, np.inf)
        )
        never = ~(reaches | started)
        if clock is not None:
            # reference-stress failure times to calendar time along the
            # unit's history and future stress
            failure_times = clock.calendar(failure_times)

        age = float(x_arr.max())
        quantiles = [0.5, alpha_ci / 2.0, 1.0 - alpha_ci / 2.0]
        ft_med, ft_lower, ft_upper = _quantiles(failure_times, quantiles)
        rul_samples = failure_times - age
        rul_med, rul_lower, rul_upper = _quantiles(rul_samples, quantiles)

        return RULPrediction(
            failure_time=float(ft_med),
            failure_time_interval=(float(ft_lower), float(ft_upper)),
            rul=float(rul_med),
            rul_interval=(float(rul_lower), float(rul_upper)),
            prob_failed=float((failure_times <= age).mean()),
            prob_never_fails=float(never.mean()),
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            alpha_ci=alpha_ci,
            samples=failure_times,
        )

    def _path_posterior(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        path_model: PathModel,
        prior_mean: npt.NDArray,
        prior_cov: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Gaussian posterior of a new unit's path parameters given a
        population prior ``N(prior_mean, prior_cov)`` and the unit's
        measurements.

        ``path_model`` is the model the prior is expressed in: the plain
        path model for the pooled population, or its link-scale
        :class:`LinkedPathModel` for a stress-conditional one. Exact for
        linear-in-parameter path models (one Gauss-Newton step is the
        conjugate update); iterated linearisation to the MAP otherwise.
        """
        prior_mean = np.asarray(prior_mean, dtype=float)
        # floor the prior covariance's eigenvalues so a clipped
        # (rank-deficient) covariance still gives a proper, very tight
        # prior in the deficient directions
        prior_precision = psd_precision(prior_cov, 1e-8, 0.0)
        noise_var = self.measurement_var

        theta = prior_mean.copy()
        precision = prior_precision
        max_iter = 1 if path_model.linear_in_parameters else 100
        for _ in range(max_iter):
            jacobian = path_model.jacobian(x, *theta)
            fitted = path_model.path(x, *theta)
            precision = prior_precision + jacobian.T @ jacobian / noise_var
            rhs = (
                prior_precision @ prior_mean
                + jacobian.T @ (y - fitted + jacobian @ theta) / noise_var
            )
            theta_new = np.linalg.solve(precision, rhs)
            if not np.isfinite(theta_new).all():
                raise ValueError(
                    "The linearised posterior update diverged for this "
                    "trajectory; the {} path model could not be updated "
                    "against the population prior".format(path_model.name)
                )
            if np.allclose(theta_new, theta, rtol=1e-10, atol=1e-12):
                theta = theta_new
                break
            theta = theta_new

        posterior_cov = np.linalg.inv(precision)
        posterior_cov = (posterior_cov + posterior_cov.T) / 2.0
        return theta, posterior_cov

    def _handle_new_trajectory(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray]:
        x_arr, y_arr, _ = validate_xy(x, y)
        n_params = len(self.path_model.parameter_names)
        if len(x_arr) < n_params or len(np.unique(x_arr)) < 2:
            raise ValueError(
                "The trajectory needs at least {} measurements at 2 or "
                "more distinct times to fit the {} path model".format(
                    n_params, self.path_model.name
                )
            )
        return x_arr, y_arr

    def _life_fn(self, name: str, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        # One dispatcher for the five distribution functions: the
        # accelerated model evaluates its regression at stress ``Z``, the
        # plain model evaluates its fitted life distribution. The named
        # methods below each carried this body verbatim.
        if self._is_clock:
            return self._clock_life_fn(name, x, Z)
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return getattr(self._reg, name)(x, Z)
        return getattr(self.life_model, name)(x)

    def _clock_life_fn(
        self, name: str, x: npt.ArrayLike, Z: Any
    ) -> npt.NDArray:
        """
        A life function under the stress ``Z`` for a step-stress model: the
        reference-stress life at the clock time ``tau(x)``. The density and
        hazard also carry the clock's rate at ``x`` (the ``AF`` of the
        stress in force).
        """
        clock = self._clock(Z)
        t = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.asarray(getattr(self.life_model, name)(clock.tau(t)))
        if name in ("df", "hf"):
            out = out * clock.rate_at(t)
        return out

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """
        Survival function of the fitted life model.

        For an accelerated-degradation model (fitted with covariates) the
        stress vector ``Z`` at which to evaluate life is required. For a
        step-stress model (``acceleration="clock"``) ``Z`` is one stress row or
        a :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
        stress profile, and life is the reference-stress life at the clock
        time, ``S(t) = S0(tau(t))``; the same holds for every life method
        below.
        """
        return self._life_fn("sf", x, Z)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """CDF of the fitted life model (pass ``Z`` for accelerated models)."""
        return self._life_fn("ff", x, Z)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Density of the fitted life model (``Z`` for accelerated models)."""
        return self._life_fn("df", x, Z)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Hazard rate of the fitted life model (``Z`` for accelerated)."""
        return self._life_fn("hf", x, Z)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Cumulative hazard of the life model (``Z`` for accelerated)."""
        return self._life_fn("Hf", x, Z)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """
        Quantile function of the fitted life model.

        Plain life models expose their own ``qf``; accelerated regression
        models do not, so the quantile at stress ``Z`` is obtained by
        numerically inverting the survival function, pairing each ``p``
        with a row of ``Z`` as :meth:`sf` pairs each ``x`` (a single row,
        or a single ``p``, is broadcast). For a step-stress model it is the
        calendar time at which the clock of ``Z`` reaches the
        reference-stress quantile, :math:`\\tau^{-1}(F_0^{-1}(p))`. A
        missing (``nan``) probability or covariate gives ``nan``.
        """
        if self._is_clock:
            clock = self._clock(Z)
            ref = np.atleast_1d(np.asarray(self.life_model.qf(p), dtype=float))
            return clock.inverse(ref)
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg_qf(p, Z)
        return self.life_model.qf(p)

    def mean(self, Z: Any = None) -> float:
        """
        Mean of the fitted life model.

        For an accelerated model the mean life at stress ``Z`` is obtained by
        integrating the survival function (the regression model has no closed
        ``mean``). For a step-stress model it is the reference-stress mean
        divided by the acceleration factor at a constant stress, and the
        integral of the survival function under a ``StepSchedule``.
        """
        if self._is_clock:
            return self._clock_mean(Z)
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg_mean(Z)
        return self.life_model.mean()

    def _clock_mean(self, Z: Any) -> float:
        """Mean life under ``Z`` for a step-stress model: the reference mean
        over ``AF`` at a constant stress, else the integral of ``sf`` (split
        at the profile's step times)."""
        clock = self._clock(Z)
        if clock.rate is not None:
            return float(self.life_model.mean()) / clock.rate
        upper = float(
            np.ravel(
                clock.inverse(np.atleast_1d(self.life_model.qf(1 - 1e-12)))
            )[0]
        )
        assert clock.schedule is not None
        edges = clock.schedule.edges
        points = edges[np.isfinite(edges) & (edges > 0) & (edges < upper)]
        val, _ = quad(
            lambda t: float(np.ravel(self.sf(t, Z))[0]),
            0.0,
            upper,
            points=points if points.size else None,
            limit=200,
        )
        return float(val)

    def random(
        self,
        size: int,
        Z: Any = None,
        random_state: "int | None" = None,
    ) -> npt.NDArray:
        """
        Random pseudo failure times from the fitted life model.

        For an accelerated model, ``size`` samples are drawn at stress ``Z`` by
        inverse-transform sampling of the fitted survival function (the
        regression models do not all expose ``random`` directly). For a
        step-stress model the reference-stress quantiles are carried to
        calendar time along the clock of ``Z``.

        Parameters
        ----------
        size : int
            Number of draws.
        Z : array like or StepSchedule, optional
            The stress, as for :meth:`sf`; required for an accelerated or
            step-stress model, refused otherwise.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for reproducible draws. ``None`` (the default)
            seeds from numpy's global RNG, so ``np.random.seed`` controls it.
        """
        if self._is_clock:
            clock = self._clock(Z)
            rng = as_generator(random_state)
            u = rng.uniform(size=size)
            return clock.inverse(
                np.asarray(self.life_model.qf(u), dtype=float)
            )
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            rng = as_generator(random_state)
            u = rng.uniform(size=size)
            return self._reg_qf(u, Z)
        # Inverse-transform sampling like the branches above: the life
        # model's own ``random`` takes no seed, so ``random_state`` was
        # silently ignored for a plain model.
        rng = as_generator(random_state)
        u = rng.uniform(size=size)
        return np.asarray(self.life_model.qf(u), dtype=float)

    def induced_life(
        self,
        n_samples: int = 10_000,
        *,
        Z: Any = None,
        random_state: "int | None" = None,
    ) -> InducedFailureDistribution:
        """
        The population failure-time distribution induced by the path model
        (the Lu-Meeker approach), as a Monte-Carlo diagnostic complement to the
        pseudo-failure-time ``life_model``.

        Path parameters are drawn from the fitted population distribution
        ``theta ~ N(path_param_mean, path_param_cov)`` and each draw is pushed
        through the path model's ``inv_path(threshold)`` to a failure time.
        This derives the population life directly from the path model, rather
        than via each unit's noisy extrapolated failure time. Overlaying the
        returned distribution's ``ff`` on this model's own ``ff`` is a check
        that the two agree.

        Parameters
        ----------
        n_samples : int, optional
            Number of Monte-Carlo path-parameter draws. Default 10000.
        Z : array like, optional
            The stress to induce the life at. Required for a model whose
            path parameters were modelled against stress (fitted with
            ``links``): the draws are then ``eta ~ N(D(z) gamma, Sigma)``
            on the link scale, mapped through the links to path
            parameters. Refused for a model without ``links``, unless it
            is a step-stress (``acceleration="clock"``) model: then ``Z``
            is required, as one stress row or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            and each draw's
            reference-stress failure time is read along that stress's
            clock. The returned distribution records a constant stress
            row as its ``stress``; under a profile it records none.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a reproducible result. ``None`` (the default)
            seeds from numpy's global RNG, so ``np.random.seed`` controls it.

        Returns
        -------
        InducedFailureDistribution
            The Monte-Carlo induced failure-time distribution.

        Examples
        --------
        The induced median next to the pseudo-failure fit's:

        >>> import numpy as np
        >>> from surpyval.degradation import DegradationAnalysis
        >>> rng = np.random.default_rng(1)
        >>> x = np.tile(np.arange(100.0, 1100.0, 100.0), 8)
        >>> i = np.repeat(np.arange(8), 10)
        >>> a = np.repeat(rng.normal(10.0, 3.0, 8), 10)
        >>> b = np.repeat(rng.normal(0.3, 0.05, 8), 10)
        >>> y = a + b * x + rng.normal(0, 3.0, x.size)
        >>> model = DegradationAnalysis.fit(x, y, i, threshold=450)
        >>> induced = model.induced_life(random_state=0)
        >>> round(induced.median()), round(float(model.qf(0.5)))
        (1452, 1472)
        >>> induced.prob_never_fails
        0.0
        """
        if self._is_clock:
            return self._clock_induced_life(n_samples, random_state, Z)
        linked: "LinkedPathModel | None" = None
        stress: "list[float] | None" = None
        if Z is not None or self.links is not None:
            linked, mean, cov = self._stress_prior(Z)
            stress = self._stress_row(Z).tolist()
        elif self.is_accelerated:
            raise ValueError(
                "induced_life needs a single population of path parameters, "
                "but this accelerated (covariate) model's population pools "
                "every stress level. Fit with links (e.g. "
                "links={'b': 'log'}) alongside Z to model the path "
                "parameters against stress, then pass the stress Z here."
            )
        else:
            mean = np.asarray(self.path_param_mean, dtype=float)
            cov = np.asarray(self.path_param_cov, dtype=float)
        rng = as_generator(random_state)
        # Robust MVN sampling: symmetrise and clip the (possibly PSD-clipped)
        # covariance's eigenvalues to be non-negative before taking its root.
        root = psd_root(cov)
        z = rng.standard_normal((n_samples, mean.size))
        theta = mean + z @ root.T
        if linked is not None:
            theta = linked.to_natural(theta)

        t = self._induced_times(theta)
        return InducedFailureDistribution(
            t, self.threshold, self.path_model.name, stress=stress
        )

    def _induced_times(self, theta: npt.NDArray) -> npt.NDArray:
        """
        Failure times (on the path's clock) of path-parameter draws, one
        per row of ``theta``: the threshold crossing where it is at a
        positive time; ``0`` for a draw already past the threshold at the
        earliest measurement time (it crossed at or before time zero -- an
        atom of failures at time zero); ``inf`` for a draw that never
        reaches the threshold.
        """
        columns: list[Any] = [theta[:, k] for k in range(theta.shape[1])]
        with np.errstate(all="ignore"):
            t: npt.NDArray = np.asarray(
                self.path_model.inv_path(self.threshold, *columns),
                dtype=float,
            )
        reaches = np.isfinite(t) & (t > 0)
        started = np.zeros(len(t), dtype=bool)
        if not reaches.all():
            started[~reaches] = self._past_threshold(
                self._start_time, theta[~reaches]
            )
        return np.where(reaches, t, np.where(started, 0.0, np.inf))

    def _clock_induced_life(
        self, n_samples: int, random_state: Any, Z: Any
    ) -> InducedFailureDistribution:
        """The induced life of a step-stress model under the stress ``Z``:
        reference-stress failure times from the population of path
        parameters, read along the stress's clock."""
        clock = self._clock(Z)
        assert self.gamma is not None
        # the stress row this population is induced at; a missing value is
        # refused, as for the other methods that describe one unit
        stress = (
            None
            if clock.schedule is not None
            else stress_row(self._covariates(Z), self.gamma.size).tolist()
        )
        mean = np.asarray(self.path_param_mean, dtype=float)
        rng = as_generator(random_state)
        root = psd_root(np.asarray(self.path_param_cov, dtype=float))
        theta = mean + rng.standard_normal((n_samples, mean.size)) @ root.T
        tau = self._induced_times(theta)
        return InducedFailureDistribution(
            clock.inverse(tau),
            self.threshold,
            self.path_model.name,
            stress=stress,
        )

    def _reg_qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        Quantile function of an accelerated life model by bisection.

        Inverts the (monotone decreasing) survival function ``sf(t | Z) = 1 -
        p`` for each requested probability. Brackets are grown geometrically
        from the fitted pseudo-failure-time scale until they straddle the
        target, then bisected. Every probability is searched at once, each
        step evaluating ``sf`` on the array of those still searching (#585:
        a scalar search per probability called ``sf`` some 35 times per
        draw, and ``random(5000)`` took half a minute).
        """
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any((p_arr < 0) | (p_arr > 1)):
            raise ValueError("qf probabilities must lie in [0, 1]")
        # one covariate row per probability, as ``sf`` pairs them with
        # ``x``; only the first row was used, whatever the others held
        Z_rows = np.asarray(Z, dtype=float)
        if Z_rows.ndim < 2:
            Z_rows = Z_rows.reshape(1, -1)
        try:
            p_arr, rows = np.broadcast_arrays(p_arr, np.arange(len(Z_rows)))
        except ValueError:
            raise ValueError(
                "qf pairs each probability with a row of Z: pass as many "
                "probabilities as rows, or one of either; got {} and "
                "{}".format(len(p_arr), len(Z_rows))
            ) from None
        scale = float(np.median(self.pseudo_failure_times))
        if not (np.isfinite(scale) and scale > 0):
            scale = 1.0
        # a missing probability or covariate gives nan (the bracket search
        # never met its target and returned inf); 0 and 1 are the ends
        missing = np.isnan(p_arr) | np.isnan(Z_rows).any(axis=1)[rows]
        out = np.where(p_arr <= 0.0, 0.0, np.inf)
        out[missing] = np.nan
        inner = np.flatnonzero(~missing & (p_arr > 0.0) & (p_arr < 1.0))
        if inner.size:
            Z_in = Z_rows[0] if len(Z_rows) == 1 else Z_rows[rows[inner]]
            out[inner] = self._invert_reg_sf(1.0 - p_arr[inner], Z_in, scale)
        return out

    def _invert_reg_sf(
        self, want: npt.NDArray, Z: npt.NDArray, scale: float
    ) -> npt.NDArray:
        """
        The times at which ``sf(t | Z)`` falls to ``want``, all at once.

        ``Z`` is one covariate row for every target, or a row per target.
        Each upper bracket starts at ``scale`` and doubles (the lower end
        following it) until ``sf`` there is at most the target, at most 200
        times (``inf`` if it never is); each bracket is then bisected until
        it is narrower than ``1e-10 * max(hi, 1)`` or 200 times.
        """
        one_row = Z.ndim == 1

        def sf_at(t: npt.NDArray, idx: npt.NDArray) -> npt.NDArray:
            z = Z if one_row else Z[idx]
            return np.asarray(self._reg.sf(t, z), dtype=float).ravel()

        lo = np.zeros(want.shape)
        hi = np.full(want.shape, scale)
        found = np.zeros(want.shape, dtype=bool)
        # grow the upper brackets until sf(hi) drops below the target
        active = np.arange(want.size)
        for _ in range(200):
            if not active.size:
                break
            reached = sf_at(hi[active], active) <= want[active]
            found[active[reached]] = True
            active = active[~reached]
            lo[active] = hi[active]
            hi[active] *= 2.0
        active = np.flatnonzero(found)
        for _ in range(200):
            if not active.size:
                break
            lo_a, hi_a = lo[active], hi[active]
            mid = 0.5 * (lo_a + hi_a)
            above = sf_at(mid, active) > want[active]
            lo_a = np.where(above, mid, lo_a)
            hi_a = np.where(above, hi_a, mid)
            lo[active], hi[active] = lo_a, hi_a
            active = active[hi_a - lo_a > 1e-10 * np.maximum(hi_a, 1.0)]
        return np.where(found, 0.5 * (lo + hi), np.inf)

    def _reg_mean(self, Z: Any) -> float:
        """
        Mean life of an accelerated model at stress ``Z``.

        ``E[T] = \\int_0^\\infty S(t | Z) dt`` by numerical integration over a
        grid that extends to a high survival quantile. ``nan`` for a
        missing covariate.
        """
        if np.isnan(np.asarray(Z, dtype=float)).any():
            return np.nan
        upper = float(np.ravel(self._reg_qf(0.999, Z))[0])
        if not np.isfinite(upper):
            upper = float(np.max(self.pseudo_failure_times)) * 100.0
        grid = np.linspace(0.0, upper, 4000)
        sf = np.asarray(self._reg.sf(grid, Z), dtype=float).ravel()
        return float(np.trapezoid(sf, grid))

    def life_parameter_covariance(
        self, method: str = "analytic"
    ) -> npt.NDArray:
        """
        Covariance of the fitted life-model parameters, corrected for the
        first-stage (path-fit and extrapolation) uncertainty that the plain
        life-model MLE ignores.

        See :meth:`cb` for the two-stage rationale; ``method='analytic'`` is
        the delta-method / generated-regressor correction
        ``H^{-1} + sum_i v_i (dphi/dt_i)(dphi/dt_i)'``.
        """
        if self._is_clock:
            raise NotImplementedError(
                "The two-stage life-parameter covariance is not derived for "
                "a step-stress (acceleration='clock') model, whose pseudo "
                "failure times also depend on the estimated clock; use "
                "cb(..., method='bootstrap'), which re-estimates the clock "
                "on every resample."
            )
        if self.is_accelerated:
            raise NotImplementedError(
                "The two-stage life-parameter covariance is not implemented "
                "for accelerated-degradation (covariate) models; use "
                "life_model.covariance() for the (first-stage-only) "
                "regression parameter covariance."
            )
        return life_parameter_covariance(self, method=method)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "analytic",
        n_boot: int = 200,
        random_state: "int | None" = None,
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the reliability of the fitted life model that
        account for the degradation analysis being a *two-stage* estimator.

        The pseudo failure times are extrapolated per-unit path fits, not
        observed failures, so ``life_model.cb`` -- which treats them as
        exact -- gives intervals that are too narrow. These bounds fold the
        first-stage (measurement + extrapolation) uncertainty back in.

        For an **accelerated-degradation (covariate) model** the bound is
        evaluated at a stress vector ``Z`` and only ``method='bootstrap'``
        is available: the generated-regressor delta-method correction is not
        derived for the regression life fit, so the first-stage uncertainty is
        folded in by resampling units (each carrying its stress) and rerunning
        the whole accelerated pipeline.

        Parameters
        ----------
        x : array like
            Times at which to evaluate the bound(s).
        Z : array like, optional
            Stress vector at which to evaluate the bound; required for an
            accelerated model, rejected for a plain one. For a step-stress
            (``acceleration="clock"``) model it is one stress row or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            and only ``method='bootstrap'`` is available: units are resampled
            with their stress histories and the clock is re-estimated on each
            resample (with the model's ``population_method``, so a ``"reml"``
            model's bootstrap takes correspondingly longer).
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are accepted as
            aliases of ``'sf'`` and ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability of the bound(s): a two-sided band has
            ``alpha_ci / 2`` in each tail. Default 0.05 (a 95% band).
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        method : {'analytic', 'bootstrap'}, optional
            ``'analytic'`` (default) is a fast delta-method correction;
            ``'bootstrap'`` resamples units and reruns the whole pipeline (a
            slower, assumption-light cross-check). Accelerated (covariate)
            models support ``'bootstrap'`` only.
        n_boot : int, optional
            Bootstrap resamples (``method='bootstrap'`` only). Default 200.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for the bootstrap resampling. ``None`` (the
            default) seeds from numpy's global RNG, so ``np.random.seed``
            controls it.

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``.
        """
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        check_option("bound", bound, BOUNDS)
        if self._is_clock:
            self._clock(Z)  # validates the stress
            Z = self._covariates(Z)
            if method == "analytic":
                raise NotImplementedError(
                    "The two-stage analytic correction is not derived for a "
                    "step-stress (acceleration='clock') model, whose pseudo "
                    "failure times also depend on the estimated clock; use "
                    "method='bootstrap', which re-estimates the clock on "
                    "every resample."
                )
            if method == "bootstrap":
                return bootstrap_cb(
                    self, x, on, alpha_ci, bound, n_boot, random_state, Z=Z
                )
            raise option_error("method", method, ("analytic", "bootstrap"))
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            if method == "analytic":
                raise NotImplementedError(
                    "The two-stage analytic (generated-regressor) correction "
                    "is not derived for accelerated-degradation (covariate) "
                    "life fits; use method='bootstrap' (which folds the "
                    "first-stage uncertainty in by resampling), or "
                    "life_model.cb(x, Z, on=...) for the first-stage-only "
                    "regression bounds."
                )
            if method == "bootstrap":
                return bootstrap_cb(
                    self, x, on, alpha_ci, bound, n_boot, random_state, Z=Z
                )
            raise option_error("method", method, ("analytic", "bootstrap"))
        if method == "analytic":
            return analytic_cb(self, x, on, alpha_ci, bound)
        elif method == "bootstrap":
            return bootstrap_cb(
                self, x, on, alpha_ci, bound, n_boot, random_state
            )
        raise option_error("method", method, ("analytic", "bootstrap"))

    def plot(self, ax: Any = None) -> Any:
        """
        Plot the degradation data, the fitted per-unit paths (extended
        to each unit's pseudo failure time), and the failure threshold.

        Parameters
        ----------
        ax : matplotlib axes, optional
            An axes object to draw the plot on. Creates a new one if
            not provided.

        Returns
        -------
        matplotlib axes
            An axes object with the plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        for idx, unit in enumerate(self.units):
            mask = self.i == unit
            x_unit = self.x[mask]
            y_unit = self.y[mask]
            start, end = x_unit.min(), x_unit.max()
            if self.c[idx] == 0:
                pseudo = self.pseudo_failure_times[idx]
                if self._is_clock:
                    # the calendar time the unit's clock reaches its
                    # reference-stress failure time, holding its last stress
                    pseudo = self._unit_calendar_time(unit, pseudo)
                start, end = min(start, pseudo), max(end, pseudo)
            x_plot = np.linspace(start, end, 200)
            (line,) = ax.plot(
                x_plot,
                self.path(x_plot, unit),
                linewidth=1,
                alpha=0.8,
            )
            ax.scatter(x_unit, y_unit, s=12, color=line.get_color())

        ax.axhline(
            self.threshold, color="k", linestyle="--", label="Threshold"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Degradation")
        ax.legend()
        return ax

    def _left_censored_repr(self) -> str:
        """The units already failed at their first measurement, for
        ``__repr__`` (nothing when there are none)."""
        n_left = int((self.c == -1).sum())
        if n_left == 0:
            return ""
        return f"\nFailed Before Start : {n_left}"

    def _stress_repr(self) -> str:
        """The stress-conditional path population, for ``__repr__``."""
        if self.links is None or self.path_param_fixed is None:
            return ""
        assert self.path_param_fixed_names is not None
        link_string = ", ".join(
            "{}: {}".format(name, link) for name, link in self.links.items()
        )
        effects = "\n".join(
            f"{name:>14}: {value}"
            for name, value in zip(
                self.path_param_fixed_names, self.path_param_fixed
            )
        )
        return (
            f"\nPath Stress Links   : {link_string}"
            "\nPath Fixed Effects  :\n" + effects
        )

    def __repr__(self) -> str:
        if self._is_clock:
            assert self.gamma is not None and self.stress_ref is not None
            param_string = "\n".join(
                f"{name:>10}: {p}"
                for p, name in zip(
                    self.life_model.params,
                    self.life_model.dist.parameter_names,
                )
            )
            return (
                "Degradation Analysis SurPyval Model"
                "\n==================================="
                f"\nPath Model          : {self.path_model.name}"
                f"\nThreshold           : {self.threshold}"
                f"\nNumber of Units     : {len(self.units)}"
                f"\nCensored Units      : {int((self.c == 1).sum())}"
                f"{self._left_censored_repr()}"
                "\nAcceleration        : clock (step-stress)"
                "\nStress coefficients : "
                + np.array2string(self.gamma, precision=6)
                + "\nReference stress    : "
                + np.array2string(self.stress_ref, precision=6)
                + f"\nLife Distribution   : {self.life_model.dist.name} "
                "(reference stress)"
                "\nParameters          :\n" + param_string
            )
        if self.is_accelerated:
            names = self.life_model.parameter_names
            dist_name = self.life_model.distribution.name
            reg_name = self.life_model.reg_model.name
            param_string = "\n".join(
                f"{name:>10}: {p}"
                for name, p in zip(names, self.life_model.params)
            )
            return (
                "Degradation Analysis SurPyval Model"
                "\n==================================="
                f"\nPath Model          : {self.path_model.name}"
                f"\nThreshold           : {self.threshold}"
                f"\nNumber of Units     : {len(self.units)}"
                f"\nCensored Units      : {int((self.c == 1).sum())}"
                f"{self._left_censored_repr()}"
                f"\nLife Distribution   : {dist_name} ({reg_name} covariates)"
                "\nParameters          :\n"
                + param_string
                + self._stress_repr()
            )
        param_string = "\n".join(
            [
                f"{name:>10}: {p}"
                for p, name in zip(
                    self.life_model.params,
                    self.life_model.dist.parameter_names,
                )
            ]
        )
        return (
            "Degradation Analysis SurPyval Model"
            "\n==================================="
            f"\nPath Model          : {self.path_model.name}"
            f"\nThreshold           : {self.threshold}"
            f"\nNumber of Units     : {len(self.units)}"
            f"\nCensored Units      : {int((self.c == 1).sum())}"
            f"{self._left_censored_repr()}"
            f"\nLife Distribution   : {self.life_model.dist.name}"
            "\nParameters          :\n" + param_string
        )
