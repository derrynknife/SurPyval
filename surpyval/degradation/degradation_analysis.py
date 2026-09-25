"""Degradation analysis.

Classic (pseudo-failure-time) degradation analysis: a degradation
measurement is tracked over time on each unit, a
:class:`~surpyval.degradation.PathModel` is fitted to each unit's
measurements, each fitted path is extrapolated to the failure threshold
to get that unit's pseudo failure time, and a lifetime distribution is
fitted to the pseudo failure times. Units whose fitted path never
reaches the threshold are treated as right censored at their last
observed time.

With ``acceleration="clock"`` the stress -- which may change during a
unit's test, as in a step-stress test -- speeds up the clock of every
unit's path: the path is the ordinary path model evaluated on the
reference-stress time the unit has aged, the pseudo failure times are
reference-stress lifetimes, and life under any stress profile follows from
the reference-stress life distribution. See :mod:`.step_stress`.
"""

import inspect
import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import quad

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.univariate.regression import AFT
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)
from surpyval.utils.linalg import (
    psd_precision,
    psd_project,
    psd_root,
    safe_inv,
)

from ._bounds import (
    analytic_cb,
    bootstrap_cb,
    life_parameter_covariance,
)
from ._clock import HistoryClock, StressClock, stress_row
from .path_models import PATH_MODELS, PathModel, get_path_model
from .population import reml_estimate, reml_estimate_nonlinear
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


@dataclass
class RULPrediction:
    """
    Posterior failure-time / remaining-useful-life prediction for a
    new unit, returned by :meth:`DegradationModel.predict_rul`.

    All summaries come from Monte Carlo samples of the new unit's
    path parameters drawn from their Gaussian posterior and pushed
    through the path model's threshold crossing. Samples whose path
    never reaches the threshold contribute ``inf`` failure times, so
    the median and interval endpoints can be ``inf`` when much of the
    posterior mass never fails.

    Parameters
    ----------
    failure_time : float
        Posterior median of the unit's failure time (measured from
        the unit's time zero, like the fitted life model).
    failure_time_interval : tuple of float
        Equal-tailed ``1 - alpha_ci`` credible interval for the
        failure time.
    rul : float
        Posterior median remaining useful life: failure time minus
        the unit's last observed time. Negative means the unit has
        most likely already crossed the threshold.
    rul_interval : tuple of float
        Equal-tailed ``1 - alpha_ci`` credible interval for the
        remaining useful life.
    prob_failed : float
        Posterior probability that the unit's path has already
        crossed the threshold (failure time at or before its last
        observed time).
    prob_never_fails : float
        Posterior probability that the unit's path never reaches the
        threshold.
    posterior_mean, posterior_cov : ndarray
        The Gaussian posterior of the unit's path parameters. For a model
        whose path parameters were modelled against stress (``links``)
        these are on the *link* scale, in the order of the model's
        ``path_param_fixed_names`` intercepts (``"log(b)"`` for a
        log-linked ``b``); otherwise on the natural scale.
    alpha_ci : float
        The interval significance level used.
    samples : ndarray
        The Monte Carlo failure-time samples (``inf`` where the
        sampled path never reaches the threshold).
    """

    failure_time: float
    failure_time_interval: "tuple[float, float]"
    rul: float
    rul_interval: "tuple[float, float]"
    prob_failed: float
    prob_never_fails: float
    posterior_mean: npt.NDArray
    posterior_cov: npt.NDArray
    alpha_ci: float
    samples: npt.NDArray = field(repr=False)


class InducedFailureDistribution(SerialisableMixin):
    """
    The population failure-time distribution *induced by the degradation path
    model* -- the Lu-Meeker approach.

    Where the fitted ``life_model`` fits a lifetime distribution to each unit's
    (noisy) extrapolated pseudo failure time, this instead derives the
    population life directly from the fitted path-parameter distribution: path
    parameters are drawn ``theta ~ N(path_param_mean, path_param_cov)`` and
    each draw is pushed through the path model's ``inv_path(threshold)`` to a
    failure time by Monte Carlo. It is produced by
    :meth:`DegradationModel.induced_life`.

    Draws whose path never crosses the threshold at a positive time are
    recorded as ``inf`` -- a defective ("never fails") mass exposed as
    ``prob_never_fails`` -- so the quantiles and the mean are ``inf`` once they
    reach into that mass.

    Use it as a diagnostic: overlay ``induced.ff(t)`` on the model's own
    ``ff(t)`` (the pseudo-failure fit); close agreement is evidence that the
    path model and its population summary are consistent with the
    pseudo-failure lifetime fit.

    Parameters
    ----------
    samples : numpy array
        The Monte-Carlo failure-time draws (``inf`` where the path never
        reaches the threshold at a positive time).
    threshold : float
        The degradation failure threshold used.
    path_name : str
        Name of the degradation path model.
    stress : list of float, optional
        The stress row the distribution was induced at, for a model whose
        path parameters depend on stress; ``None`` for the plain
        population.
    """

    def __init__(
        self,
        samples: npt.NDArray,
        threshold: float,
        path_name: str,
        stress: "list[float] | None" = None,
    ) -> None:
        self.samples = np.asarray(samples, dtype=float)
        self.threshold = float(threshold)
        self.path_name = path_name
        self.stress = None if stress is None else [float(z) for z in stress]
        self.prob_never_fails = float(np.mean(~np.isfinite(self.samples)))

    def to_dict(self) -> dict:
        """
        Serialise this induced failure-time distribution to a plain dict.

        Stores the Monte-Carlo samples (the ``inf`` never-fails draws are
        written as ``null`` so the result is valid JSON), the threshold and the
        path model's name.
        """
        samples = [
            None if not np.isfinite(s) else float(s) for s in self.samples
        ]
        out = {
            "model": "InducedFailureDistribution",
            "samples": samples,
            "threshold": self.threshold,
            "path_name": self.path_name,
        }
        if self.stress is not None:
            out["stress"] = list(self.stress)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "InducedFailureDistribution":
        """Rebuild an induced failure-time distribution from a dict."""
        require_model_tag(
            model_dict,
            "InducedFailureDistribution",
            "an induced failure-time distribution",
        )
        samples = np.array(
            [np.inf if s is None else s for s in model_dict["samples"]],
            dtype=float,
        )
        return cls(
            samples,
            model_dict["threshold"],
            model_dict["path_name"],
            stress=model_dict.get("stress"),
        )

    def ff(self, x: npt.ArrayLike) -> "float | npt.NDArray":
        """Failure probability ``P(T <= x)`` from the Monte-Carlo draws."""
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = (self.samples[None, :] <= x[:, None]).mean(axis=1)
        return float(out[0]) if scalar else out

    def sf(self, x: npt.ArrayLike) -> "float | npt.NDArray":
        """Survival function ``P(T > x)``."""
        scalar = np.isscalar(x)
        res = 1.0 - np.atleast_1d(self.ff(np.atleast_1d(x)))
        return float(res[0]) if scalar else res

    def qf(self, p: npt.ArrayLike) -> "float | npt.NDArray":
        """Quantile of the induced distribution (``inf`` in the never-fails
        mass)."""
        scalar = np.isscalar(p)
        p = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("qf probabilities must lie in [0, 1]")
        out = np.quantile(self.samples, p, method="lower")
        return float(out[0]) if scalar else out

    def mean(self) -> float:
        """Mean failure time (``inf`` if any draw never fails)."""
        return float(self.samples.mean())

    def median(self) -> float:
        """Median failure time."""
        return float(self.qf(0.5))

    def random(
        self, size: int, random_state: "int | None" = None
    ) -> npt.NDArray:
        """Draw failure times by resampling the Monte-Carlo population."""
        rng = np.random.default_rng(random_state)
        return rng.choice(self.samples, size=size)

    def __repr__(self) -> str:
        at = "" if self.stress is None else ", Z={}".format(self.stress)
        return (
            "InducedFailureDistribution({} path{}, threshold={:.6g}, "
            "median={:.6g}, prob_never_fails={:.4g})".format(
                self.path_name,
                at,
                self.threshold,
                self.median(),
                self.prob_never_fails,
            )
        )


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
        crossing time, or the unit's last observed time for censored
        units.
    c : ndarray
        Per-unit censor flags: 0 where the fitted path crosses the
        threshold, 1 (right censored) where it never does.
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
        ``path_model`` is the candidate with the smallest score.
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
                "path_model": self.path_model.name,
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

        The path model is resolved by name and the life model by its own
        ``from_dict``; both are restricted to the known types.

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
        Z : array like
            One stress row.
        """
        if not self._is_clock or self.gamma is None:
            raise ValueError(
                "acceleration_factor is defined for a model fitted with "
                "acceleration='clock'"
            )
        assert self.stress_ref is not None
        z = stress_row(Z, self.gamma.size)
        return float(np.exp(self.gamma @ (z - self.stress_ref)))

    def _clock(self, Z: Any) -> StressClock:
        """The clock for stress ``Z`` (a row or a StepSchedule)."""
        if Z is None:
            raise ValueError(
                "This step-stress (acceleration='clock') model's life "
                "depends on stress; pass Z -- one stress row for a constant "
                "stress, or a StepSchedule for a stress profile."
            )
        assert self.gamma is not None
        return StressClock(self.acceleration_factor, self.gamma.size, Z)

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
        Z_arr = np.asarray(Z, dtype=float)
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
            return Z
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
        idx = self._unit_index[unit]
        if self._is_clock:
            x = self._unit_clock(unit, np.asarray(x, dtype=float))
        return self.path_model.path(x, *self.path_params[idx])

    # -- the stress-conditional path population (``links``) ----------------

    def _stress_row(self, Z: Any) -> npt.NDArray:
        """Validate one stress row for the stress-conditional population."""
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
        z = np.asarray(Z, dtype=float)
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
        if not np.isfinite(z).all():
            raise ValueError("Z must contain only finite values")
        return z

    def _stress_prior(
        self, Z: Any
    ) -> tuple[LinkedPathModel, npt.NDArray, npt.NDArray]:
        """The link-scale path population at stress ``Z``: the linked path
        model, the mean ``D(z) gamma`` and the covariance ``Sigma``."""
        z = self._stress_row(Z)
        assert self.links is not None and self.path_param_fixed is not None
        design = stress_design(z, self.links, self.path_model.param_names)
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
        Z : array like
            One stress row, with as many covariates as the model was
            fitted with.

        Returns
        -------
        ndarray
            The link-scale mean path parameters at ``Z``.
        """
        return self._stress_prior(Z)[1]

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
        Z : array like
            One stress row, with as many covariates as the model was
            fitted with.

        Returns
        -------
        ndarray
            The median path parameters at ``Z``, in path order.
        """
        linked, mean, _ = self._stress_prior(Z)
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
            For a step-stress model, the stress from the last measurement
            on: one row, or a :class:`~surpyval.StepSchedule` whose time
            zero is the last measurement. Defaults to holding the last
            stress.

        Returns
        -------
        float
            The time at which the new unit's fitted path reaches the
            threshold. This can be smaller than the last observed time
            if the trajectory has already crossed the threshold.
            Returns ``nan`` (with a warning) if the fitted path never
            reaches the threshold.
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
        alpha_ci: float = 0.05,
        n_samples: int = 10_000,
        random_state: "int | None" = None,
        Z: Any = None,
        Z_future: Any = None,
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
        alpha_ci : float, optional
            Significance level for the equal-tailed credible
            intervals. Defaults to 0.05 (95% intervals).
        n_samples : int, optional
            Number of Monte Carlo posterior samples. Defaults to
            10,000.
        random_state : optional
            Seed passed to ``numpy.random.default_rng`` for
            reproducible sampling.
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
            For a step-stress model, the stress from the last measurement
            on: one row, or a :class:`~surpyval.StepSchedule` whose time
            zero is the last measurement. Defaults to holding the last
            stress. Refused for other models.

        Returns
        -------
        RULPrediction
            Posterior medians, credible intervals, failure
            probabilities, and the parameter posterior.
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
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if x_arr.ndim != 1 or y_arr.ndim != 1 or len(x_arr) != len(y_arr):
            raise ValueError(
                "x and y must be one dimensional and the same length"
            )
        if len(x_arr) == 0:
            raise ValueError("At least one measurement is required")
        if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
            raise ValueError("x and y must contain only finite values")
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

        rng = np.random.default_rng(random_state)
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
        failure_times = np.where(reaches, failure_times, np.inf)
        if clock is not None:
            # reference-stress failure times to calendar time along the
            # unit's history and future stress
            failure_times = clock.calendar(failure_times)

        age = float(x_arr.max())
        quantiles = [0.5, alpha_ci / 2.0, 1.0 - alpha_ci / 2.0]
        ft_med, ft_lower, ft_upper = np.quantile(failure_times, quantiles)
        rul_samples = failure_times - age
        rul_med, rul_lower, rul_upper = np.quantile(rul_samples, quantiles)

        return RULPrediction(
            failure_time=float(ft_med),
            failure_time_interval=(float(ft_lower), float(ft_upper)),
            rul=float(rul_med),
            rul_interval=(float(rul_lower), float(rul_upper)),
            prob_failed=float((failure_times <= age).mean()),
            prob_never_fails=float((~reaches).mean()),
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
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one dimensional")
        if len(x) != len(y):
            raise ValueError(
                "x and y must have the same length; got {} and {}".format(
                    len(x), len(y)
                )
            )
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("x and y must contain only finite values")
        n_params = len(self.path_model.param_names)
        if len(x) < n_params or len(np.unique(x)) < 2:
            raise ValueError(
                "The trajectory needs at least {} measurements at 2 or "
                "more distinct times to fit the {} path model".format(
                    n_params, self.path_model.name
                )
            )
        return x, y

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

    def sf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """
        Survival function of the fitted life model.

        For an accelerated-degradation model (fitted with covariates) the
        stress vector ``Z`` at which to evaluate life is required. For a
        step-stress model (``acceleration="clock"``) ``Z`` is one stress row
        or a :class:`~surpyval.StepSchedule` stress profile, and life is the
        reference-stress life at the clock time, ``S(t) = S0(tau(t))``; the
        same holds for every life method below.
        """
        return self._life_fn("sf", x, Z)

    def ff(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """CDF of the fitted life model (pass ``Z`` for accelerated models)."""
        return self._life_fn("ff", x, Z)

    def df(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Density of the fitted life model (``Z`` for accelerated models)."""
        return self._life_fn("df", x, Z)

    def hf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Hazard rate of the fitted life model (``Z`` for accelerated)."""
        return self._life_fn("hf", x, Z)

    def Hf(self, x: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """Cumulative hazard of the life model (``Z`` for accelerated)."""
        return self._life_fn("Hf", x, Z)

    def qf(self, p: npt.ArrayLike, Z: Any = None) -> npt.NDArray:
        """
        Quantile function of the fitted life model.

        Plain life models expose their own ``qf``; accelerated regression
        models do not, so the quantile at stress ``Z`` is obtained by
        numerically inverting the survival function.
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
        ``mean``).
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
        regression models do not all expose ``random`` directly).
        """
        if self._is_clock:
            clock = self._clock(Z)
            rng = np.random.default_rng(random_state)
            u = rng.uniform(size=size)
            return clock.inverse(
                np.asarray(self.life_model.qf(u), dtype=float)
            )
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            rng = np.random.default_rng(random_state)
            u = rng.uniform(size=size)
            return self._reg_qf(u, Z)
        # The life model here is a plain fit (no LFP / zero-inflation),
        # so ``random`` returns a bare array, never the xcnt tuple.
        return np.asarray(self.life_model.random(size))

    def induced_life(
        self,
        n_samples: int = 10_000,
        random_state: "int | None" = None,
        Z: Any = None,
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
        random_state : int or numpy.random.Generator, optional
            Seed for a reproducible result.
        Z : array like, optional
            The stress to induce the life at. Required for a model whose
            path parameters were modelled against stress (fitted with
            ``links``): the draws are then ``eta ~ N(D(z) gamma, Sigma)``
            on the link scale, mapped through the links to path
            parameters. Refused for a model without ``links``, unless it
            is a step-stress (``acceleration="clock"``) model: then ``Z``
            is required, as one stress row or a
            :class:`~surpyval.StepSchedule`, and each draw's
            reference-stress failure time is read along that stress's
            clock. The returned distribution records a constant stress
            row as its ``stress``; under a profile it records none.

        Returns
        -------
        InducedFailureDistribution
            The Monte-Carlo induced failure-time distribution.
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
        rng = np.random.default_rng(random_state)
        # Robust MVN sampling: symmetrise and clip the (possibly PSD-clipped)
        # covariance's eigenvalues to be non-negative before taking its root.
        root = psd_root(cov)
        z = rng.standard_normal((n_samples, mean.size))
        theta = mean + z @ root.T
        if linked is not None:
            theta = linked.to_natural(theta)

        columns = [theta[:, k] for k in range(theta.shape[1])]
        with np.errstate(all="ignore"):
            t = np.asarray(
                self.path_model.inv_path(self.threshold, *columns),
                dtype=float,
            )
        # A draw only defines a failure time if its path crosses the threshold
        # at a positive time; otherwise the unit never fails (inf).
        t = np.where(np.isfinite(t) & (t > 0), t, np.inf)
        return InducedFailureDistribution(
            t, self.threshold, self.path_model.name, stress=stress
        )

    def _clock_induced_life(
        self, n_samples: int, random_state: Any, Z: Any
    ) -> InducedFailureDistribution:
        """The induced life of a step-stress model under the stress ``Z``:
        reference-stress failure times from the population of path
        parameters, read along the stress's clock."""
        clock = self._clock(Z)
        mean = np.asarray(self.path_param_mean, dtype=float)
        rng = np.random.default_rng(random_state)
        root = psd_root(np.asarray(self.path_param_cov, dtype=float))
        theta = mean + rng.standard_normal((n_samples, mean.size)) @ root.T
        with np.errstate(all="ignore"):
            tau = np.asarray(
                self.path_model.inv_path(
                    self.threshold, *(theta[:, k] for k in range(mean.size))
                ),
                dtype=float,
            )
        tau = np.where(np.isfinite(tau) & (tau > 0), tau, np.inf)
        assert self.gamma is not None
        stress = (
            None
            if clock.schedule is not None
            else stress_row(Z, self.gamma.size).tolist()
        )
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
        target, then bisected.
        """
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any((p_arr < 0) | (p_arr > 1)):
            raise ValueError("qf probabilities must lie in [0, 1]")
        scale = float(np.median(self.pseudo_failure_times))
        if not (np.isfinite(scale) and scale > 0):
            scale = 1.0

        def target_sf(t: float) -> float:
            return float(self._reg.sf(np.array([t]), Z).ravel()[0])

        out = np.empty_like(p_arr)
        for k, pk in enumerate(p_arr):
            if pk <= 0.0:
                out[k] = 0.0
                continue
            if pk >= 1.0:
                out[k] = np.inf
                continue
            want = 1.0 - pk  # survival at the quantile
            lo, hi = 0.0, scale
            # grow the upper bracket until sf(hi) drops below the target
            for _ in range(200):
                if target_sf(hi) <= want:
                    break
                lo = hi
                hi *= 2.0
            else:
                out[k] = np.inf
                continue
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if target_sf(mid) > want:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-10 * max(hi, 1.0):
                    break
            out[k] = 0.5 * (lo + hi)
        return out

    def _reg_mean(self, Z: Any) -> float:
        """
        Mean life of an accelerated model at stress ``Z``.

        ``E[T] = \\int_0^\\infty S(t | Z) dt`` by numerical integration over a
        grid that extends to a high survival quantile.
        """
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

    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "analytic",
        n_boot: int = 200,
        seed: "int | None" = None,
        Z: Any = None,
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
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound. Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        method : {'analytic', 'bootstrap'}, optional
            ``'analytic'`` (default) is a fast delta-method correction;
            ``'bootstrap'`` resamples units and reruns the whole pipeline (a
            slower, assumption-light cross-check). Accelerated (covariate)
            models support ``'bootstrap'`` only.
        n_boot : int, optional
            Bootstrap resamples (``method='bootstrap'`` only). Default 200.
        seed : optional
            Seed for the bootstrap resampling.
        Z : array like, optional
            Stress vector at which to evaluate the bound; required for an
            accelerated model, rejected for a plain one. For a step-stress
            (``acceleration="clock"``) model it is one stress row or a
            :class:`~surpyval.StepSchedule`, and only
            ``method='bootstrap'`` is available: units are resampled with
            their stress histories and the clock is re-estimated on each
            resample (with the model's ``population_method``, so a
            ``"reml"`` model's bootstrap takes correspondingly longer).

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``.
        """
        valid = ("sf", "R", "ff", "F", "Hf")
        if on not in valid:
            raise ValueError("`on` must be one of {}".format(valid))
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")
        if self._is_clock:
            self._clock(Z)  # validates the stress
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
                    self, x, on, alpha_ci, bound, n_boot, seed, Z=Z
                )
            raise ValueError("`method` must be 'analytic' or 'bootstrap'")
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
                    self, x, on, alpha_ci, bound, n_boot, seed, Z=Z
                )
            raise ValueError("`method` must be 'analytic' or 'bootstrap'")
        if method == "analytic":
            return analytic_cb(self, x, on, alpha_ci, bound)
        elif method == "bootstrap":
            return bootstrap_cb(self, x, on, alpha_ci, bound, n_boot, seed)
        raise ValueError("`method` must be 'analytic' or 'bootstrap'")

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
                    self.life_model.params, self.life_model.dist.param_names
                )
            )
            return (
                "Degradation Analysis SurPyval Model"
                "\n==================================="
                f"\nPath Model          : {self.path_model.name}"
                f"\nThreshold           : {self.threshold}"
                f"\nNumber of Units     : {len(self.units)}"
                f"\nCensored Units      : {int((self.c == 1).sum())}"
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
            names = self.life_model.parameter_names()
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
                f"\nLife Distribution   : {dist_name} ({reg_name} covariates)"
                "\nParameters          :\n"
                + param_string
                + self._stress_repr()
            )
        param_string = "\n".join(
            [
                f"{name:>10}: {p}"
                for p, name in zip(
                    self.life_model.params, self.life_model.dist.param_names
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
            f"\nLife Distribution   : {self.life_model.dist.name}"
            "\nParameters          :\n" + param_string
        )


class DegradationAnalysis_:
    """
    Pseudo-failure-time degradation analysis.

    Fits a degradation path model to each unit's measurements,
    extrapolates each fitted path to the failure ``threshold`` to
    obtain per-unit pseudo failure times, and fits a lifetime
    distribution to those times. Units whose fitted path never reaches
    the threshold at a positive finite time are right censored at their
    last observed time (with a warning).

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
            the mixed model, which cannot go rank-deficient and is
            preferable with few units. Linear-in-parameter paths (linear,
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
            prediction methods take the stress as ``Z`` -- one stress row
            or a :class:`~surpyval.StepSchedule` -- to give life under any
            stress history, ``F(t) = F0(tau(t))``. The stress
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

        if not isinstance(threshold, Number) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
        threshold = float(threshold)

        if population_method not in ("moments", "reml"):
            raise ValueError(
                "population_method must be 'moments' or 'reml', got "
                "'{}'".format(population_method)
            )

        units = np.unique(i_arr)
        if len(units) < 2:
            raise ValueError(
                "Degradation analysis requires at least 2 units; "
                "got {}".format(len(units))
            )

        if acceleration not in (None, "clock"):
            raise ValueError(
                "acceleration must be None or 'clock', got {!r}".format(
                    acceleration
                )
            )
        if acceleration is None and stress_ref is not None:
            raise ValueError(
                "stress_ref is the reference stress of acceleration='clock' "
                "and is only used with it"
            )
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

        n_params = len(path_model.param_names)
        path_params = np.empty((len(units), n_params))
        pseudo = np.empty(len(units))
        last_time = np.empty(len(units))
        rss_total = 0.0
        dof_total = 0
        estimation_cov_sum = np.zeros((n_params, n_params))
        y_by_unit = []
        x_by_unit = []
        design_by_unit = []
        link_params = np.empty((len(units), n_params))
        link_design_by_unit = []
        link_estimation_covs = []

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
            params = path_model.fit(x_unit, y_unit)
            path_params[idx] = params
            pseudo[idx] = path_model.inv_path(threshold, *params)
            last_time[idx] = x_unit.max()

            residuals = y_unit - path_model.path(x_unit, *params)
            rss_total += residuals @ residuals
            dof_total += len(x_unit) - n_params
            jacobian = path_model.jacobian(x_unit, *params)
            jtj = jacobian.T @ jacobian
            estimation_cov_sum += safe_inv(jtj)
            y_by_unit.append(y_unit)
            x_by_unit.append(x_unit)
            design_by_unit.append(jacobian)

            if linked is not None:
                # the same fit on the link scale, with the Jacobian
                # (and hence the estimation covariance) mapped there
                eta = linked.to_link(params)
                link_params[idx] = eta
                link_jacobian = linked.jacobian(x_unit, *eta)
                link_design_by_unit.append(link_jacobian)
                link_estimation_covs.append(
                    safe_inv(link_jacobian.T @ link_jacobian)
                )

        # Two-stage (Lu-Meeker) noise correction: the scatter of the
        # per-unit estimates is Sigma + V_i, so subtracting the average
        # estimation covariance leaves the between-unit covariance.
        measurement_var = rss_total / dof_total if dof_total > 0 else 0.0
        path_param_mean = path_params.mean(axis=0)
        path_param_sample_cov = np.atleast_2d(
            np.cov(path_params, rowvar=False, ddof=1)
        )
        mean_estimation_cov = measurement_var * estimation_cov_sum / len(units)
        path_param_cov, was_clipped = psd_project(
            path_param_sample_cov - mean_estimation_cov
        )
        if was_clipped and population_method == "moments":
            warnings.warn(
                "The noise-corrected between-unit covariance of the path "
                "parameters was not positive semi-definite (the estimation "
                "noise is comparable to the between-unit scatter); negative "
                "eigenvalues were clipped to zero. With this few units or "
                "measurements per unit, path_param_cov is unreliable; "
                "consider population_method='reml'",
                stacklevel=2,
            )

        if population_method == "reml":
            noise_floor = np.finfo(float).eps * float(np.mean(y_arr**2))
            if not measurement_var > noise_floor:
                raise ValueError(
                    "population_method='reml' requires measurement noise, "
                    "but the pooled measurement variance is 0 (every unit's "
                    "path fitted its measurements exactly, or no unit has "
                    "more measurements than path parameters)"
                )
            # the moment estimates are the starting values; a
            # linear-in-parameters path is an exact linear mixed model,
            # a nonlinear one is fitted by FOCE linearisation. A clock fit
            # has already estimated its population with its clock.
            if clock_population is not None:
                reml_mean, reml_cov, reml_var, converged = clock_population
            elif path_model.linear_in_parameters:
                reml_mean, reml_cov, reml_var, converged = reml_estimate(
                    y_by_unit,
                    design_by_unit,
                    path_param_cov,
                    measurement_var,
                )
            else:
                reml_mean, reml_cov, reml_var, converged = (
                    reml_estimate_nonlinear(
                        y_by_unit,
                        x_by_unit,
                        path_model,
                        path_param_mean,
                        path_param_cov,
                        measurement_var,
                        path_params,
                    )
                )
            if not converged:
                warnings.warn(
                    "The REML optimisation did not report convergence; the "
                    "population path-parameter estimates may be inaccurate",
                    stacklevel=2,
                )
            path_param_mean = reml_mean
            path_param_cov = reml_cov
            measurement_var = reml_var

        path_param_fixed = None
        path_param_fixed_names = None
        path_param_link_cov = None
        if linked is not None:
            assert Z_units is not None and links is not None
            path_param_fixed, path_param_link_cov, link_var = (
                self._fit_stress_population(
                    linked,
                    links,
                    Z_units,
                    y_by_unit,
                    x_by_unit,
                    link_params,
                    link_design_by_unit,
                    link_estimation_covs,
                    measurement_var,
                    population_method,
                )
            )
            path_param_fixed_names = fixed_effect_names(
                linked.param_names,
                path_model.param_names,
                links,
                Z_units.shape[1],
            )
            if population_method == "reml":
                # the stress-conditional model is the population model
                # of a linked fit; its noise estimate supersedes the
                # pooled one
                measurement_var = link_var

        events = np.isfinite(pseudo) & (pseudo > 0)
        if not events.any():
            raise ValueError(
                "No unit's fitted degradation path reaches the threshold "
                "{}; check the threshold and the path model".format(threshold)
            )
        if not events.all():
            warnings.warn(
                "The fitted degradation path(s) of unit(s) {} never reach "
                "the threshold {}; these units are treated as right "
                "censored at their last observed time".format(
                    list(units[~events]), threshold
                ),
                stacklevel=2,
            )

        pseudo_failure_times = np.where(events, pseudo, last_time)
        c = np.where(events, 0, 1)

        if Z_units is None:
            life_model = distribution.fit(x=pseudo_failure_times, c=c, how=how)
        else:
            reg = (
                distribution
                if _is_regression_fitter(distribution)
                else AFT(distribution)
            )
            life_model = reg.fit(x=pseudo_failure_times, Z=Z_units, c=c)

        model = DegradationModel(
            x=x_arr,
            y=y_arr,
            i=i_arr,
            units=units,
            threshold=threshold,
            path_model=path_model,
            path_params=path_params,
            pseudo_failure_times=pseudo_failure_times,
            c=c,
            life_model=life_model,
            measurement_var=measurement_var,
            path_param_mean=path_param_mean,
            path_param_cov=path_param_cov,
            path_param_sample_cov=path_param_sample_cov,
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
            cov, _ = psd_project(np.atleast_2d(np.cov(theta, rowvar=False)))
            g, converged, population = mixed_model_estimate(
                data,
                path_model,
                g,
                theta,
                theta.mean(axis=0),
                cov,
                float(resid @ resid) / dof,
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
            stress_design(z, links, linked.base.param_names) for z in Z_units
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
            n_params = len(candidate.param_names)
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

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        i: str = "i",
        Z_cols: "str | list[str] | None" = None,
        **fit_kwargs: Any,
    ) -> DegradationModel:
        """
        Fit a degradation analysis model from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame with the degradation data.
        x : str, optional
            Column of the measurement times. Defaults to ``"x"``.
        y : str, optional
            Column of the degradation measurements. Defaults to
            ``"y"``.
        i : str, optional
            Column of the unit identifiers. Defaults to ``"i"``.
        Z_cols : str or list of str, optional
            Column(s) of the stress covariates for accelerated degradation
            testing. When given, the selected columns are passed as ``Z`` to
            :meth:`fit`, fitting a covariate (ADT) life model.
        **fit_kwargs
            Remaining arguments (``threshold``, ``path``,
            ``distribution``, ``how``) passed to :meth:`fit`.

        Returns
        -------
        DegradationModel
            The fitted degradation model.
        """
        if Z_cols is not None:
            cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
            fit_kwargs["Z"] = df[cols].to_numpy()
        return self.fit(
            df[x].to_numpy(), df[y].to_numpy(), df[i].to_numpy(), **fit_kwargs
        )

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
        return Z_units

    @staticmethod
    def _handle_xyi(
        x: npt.ArrayLike, y: npt.ArrayLike, i: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        i = np.atleast_1d(np.asarray(i))
        if x.ndim != 1 or y.ndim != 1 or i.ndim != 1:
            raise ValueError("x, y, and i must be one dimensional")
        if not (len(x) == len(y) == len(i)):
            raise ValueError(
                "x, y, and i must have the same length; got {}, {}, "
                "and {}".format(len(x), len(y), len(i))
            )
        if len(x) == 0:
            raise ValueError("x, y, and i must not be empty")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")
        return x, y, i


DegradationAnalysis = DegradationAnalysis_()
