"""Degradation analysis.

Classic (pseudo-failure-time) degradation analysis: a degradation
measurement is tracked over time on each unit, a
:class:`~surpyval.degradation.PathModel` is fitted to each unit's
measurements, each fitted path is extrapolated to the failure threshold
to get that unit's pseudo failure time, and a lifetime distribution is
fitted to the pseudo failure times. Units whose fitted path never
reaches the threshold are treated as right censored at their last
observed time.
"""

import inspect
import json
import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric
from surpyval.univariate.regression import AFT
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)

from ._bounds import (
    analytic_cb,
    bootstrap_cb,
    bootstrap_cb_accelerated,
    life_parameter_covariance,
)
from .path_models import PATH_MODELS, PathModel, get_path_model
from .population import reml_estimate, reml_estimate_nonlinear


def _is_regression_fitter(fitter) -> bool:
    """True if ``fitter.fit`` takes a covariate matrix ``Z`` (i.e. it is one of
    the regression fitters -- AFT, PH, PO, additive hazards, accelerated
    life)."""
    try:
        return "Z" in inspect.signature(fitter.fit).parameters
    except (TypeError, ValueError):
        return False


def _clip_psd(matrix: npt.NDArray) -> tuple[npt.NDArray, bool]:
    """
    Project a symmetric matrix onto the positive semi-definite cone
    by clipping negative eigenvalues to zero.

    Returns the projected matrix and whether any eigenvalue was
    *materially* negative (beyond floating-point noise).
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    tol = 1e-10 * max(np.abs(eigvals).max(), np.finfo(float).tiny)
    clipped = bool((eigvals < -tol).any())
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T, clipped


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
        The Gaussian posterior of the unit's path parameters.
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


class InducedFailureDistribution:
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
    """

    def __init__(self, samples, threshold, path_name):
        self.samples = np.asarray(samples, dtype=float)
        self.threshold = float(threshold)
        self.path_name = path_name
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
        return {
            "model": "InducedFailureDistribution",
            "samples": samples,
            "threshold": self.threshold,
            "path_name": self.path_name,
        }

    def to_json(self, fp) -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "InducedFailureDistribution":
        """Rebuild an induced failure-time distribution from a dict."""
        if model_dict.get("model") != "InducedFailureDistribution":
            raise ValueError(
                "Must create an induced failure-time distribution from an "
                "InducedFailureDistribution dict"
            )
        samples = np.array(
            [np.inf if s is None else s for s in model_dict["samples"]],
            dtype=float,
        )
        return cls(samples, model_dict["threshold"], model_dict["path_name"])

    @classmethod
    def from_json(cls, fp) -> "InducedFailureDistribution":
        """Load from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

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

    def random(self, size: int, random_state=None) -> npt.NDArray:
        """Draw failure times by resampling the Monte-Carlo population."""
        rng = np.random.default_rng(random_state)
        return rng.choice(self.samples, size=size)

    def __repr__(self):
        return (
            "InducedFailureDistribution({} path, threshold={:.6g}, "
            "median={:.6g}, prob_never_fails={:.4g})".format(
                self.path_name,
                self.threshold,
                self.median(),
                self.prob_never_fails,
            )
        )


class DegradationModel:
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
    life_model: Parametric
    measurement_var: float
    path_param_mean: npt.NDArray
    path_param_cov: npt.NDArray
    path_param_sample_cov: npt.NDArray
    population_method: str
    path_selection: "dict[str, float] | None"
    #: Per-unit covariates when fitted as an accelerated-degradation model
    #: (``Z`` given to :meth:`DegradationAnalysis.fit`); ``None`` otherwise.
    Z: "npt.NDArray | None"
    # Recorded after construction so the bootstrap bounds can rerun the fit.
    _distribution: Any
    _how: str

    def __init__(
        self,
        x,
        y,
        i,
        units,
        threshold,
        path_model,
        path_params,
        pseudo_failure_times,
        c,
        life_model,
        measurement_var,
        path_param_mean,
        path_param_cov,
        path_param_sample_cov,
        population_method,
        path_selection=None,
        Z=None,
    ):
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
        self._unit_index = {unit: idx for idx, unit in enumerate(units)}

    # -- serialisation -----------------------------------------------------

    @staticmethod
    def _life_model_to_dict(life_model) -> dict:
        out = life_model.to_dict()
        out["_life_class"] = (
            "ParametricRegressionModel"
            if isinstance(life_model, ParametricRegressionModel)
            else "Parametric"
        )
        return out

    @staticmethod
    def _life_model_from_dict(life_dict: dict):
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
        return {
            "model": "DegradationModel",
            "x": np.asarray(self.x, dtype=float).tolist(),
            "y": np.asarray(self.y, dtype=float).tolist(),
            "i": np.asarray(self.i).tolist(),
            "units": np.asarray(self.units).tolist(),
            "threshold": float(self.threshold),
            "path_model": self.path_model.name,
            "path_params": np.asarray(self.path_params, dtype=float).tolist(),
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
        }

    def to_json(self, fp) -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

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
        if model_dict.get("model") != "DegradationModel":
            raise ValueError(
                "Must create a degradation model from a DegradationModel dict"
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
        )
        # Recorded so bootstrap bounds can rerun the pipeline; the original
        # distribution object is not serialised, so bounds default to the
        # analytic method after a reload.
        out._distribution = None
        out._how = model_dict.get("how", "MLE")
        return out

    @classmethod
    def from_json(cls, fp) -> "DegradationModel":
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    @property
    def is_accelerated(self) -> bool:
        """True when the life model is a covariate (ADT) regression model."""
        return isinstance(self.life_model, ParametricRegressionModel)

    @property
    def _reg(self) -> ParametricRegressionModel:
        """The life model viewed as a regression model (accelerated only)."""
        return cast(ParametricRegressionModel, self.life_model)

    def _predict_Z(self, Z):
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

    def path(self, x: npt.ArrayLike, unit) -> npt.NDArray:
        """Evaluate the fitted degradation path of ``unit`` at ``x``."""
        idx = self._unit_index[unit]
        return self.path_model.path(x, *self.path_params[idx])

    def predict_failure_time(
        self, x: npt.ArrayLike, y: npt.ArrayLike
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
        params = self.path_model.fit(x_arr, y_arr)
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
        return t

    def predict_remaining_life(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> float:
        """
        Estimate the remaining life of a new unit from its (partial)
        degradation trajectory.

        This is :meth:`predict_failure_time` minus the new unit's last
        observed time. A negative value means the fitted path crossed
        the threshold before the last observation (the unit is
        predicted to have already failed); ``nan`` (with a warning)
        means the fitted path never reaches the threshold.
        """
        x_arr, y_arr = self._handle_new_trajectory(x, y)
        return self.predict_failure_time(x_arr, y_arr) - float(x_arr.max())

    def predict_rul(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        alpha_ci: float = 0.05,
        n_samples: int = 10_000,
        random_state=None,
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
        self.path_model.check_data(x_arr, y_arr)

        posterior_mean, posterior_cov = self._path_posterior(x_arr, y_arr)

        rng = np.random.default_rng(random_state)
        theta_samples = rng.multivariate_normal(
            posterior_mean, posterior_cov, size=n_samples
        )
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
        self, x: npt.NDArray, y: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Gaussian posterior of a new unit's path parameters given the
        population prior and the unit's measurements.

        Exact for linear-in-parameter path models (one Gauss-Newton
        step is the conjugate update); iterated linearisation to the
        MAP otherwise.
        """
        prior_mean = self.path_param_mean
        # floor the prior covariance's eigenvalues so a clipped
        # (rank-deficient) covariance still gives a proper, very tight
        # prior in the deficient directions
        eigvals, eigvecs = np.linalg.eigh(self.path_param_cov)
        floor = max(eigvals.max() * 1e-8, np.finfo(float).tiny)
        prior_precision = (
            eigvecs @ np.diag(1.0 / np.clip(eigvals, floor, None)) @ eigvecs.T
        )
        noise_var = self.measurement_var

        theta = prior_mean.copy()
        precision = prior_precision
        max_iter = 1 if self.path_model.linear_in_parameters else 100
        for _ in range(max_iter):
            jacobian = self.path_model.jacobian(x, *theta)
            fitted = self.path_model.path(x, *theta)
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
                    "against the population prior".format(self.path_model.name)
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

    def sf(self, x: npt.ArrayLike, Z=None) -> npt.NDArray:
        """
        Survival function of the fitted life model.

        For an accelerated-degradation model (fitted with covariates) the
        stress vector ``Z`` at which to evaluate life is required.
        """
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg.sf(x, Z)
        return self.life_model.sf(x)

    def ff(self, x: npt.ArrayLike, Z=None) -> npt.NDArray:
        """CDF of the fitted life model (pass ``Z`` for accelerated models)."""
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg.ff(x, Z)
        return self.life_model.ff(x)

    def df(self, x: npt.ArrayLike, Z=None) -> npt.NDArray:
        """Density of the fitted life model (``Z`` for accelerated models)."""
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg.df(x, Z)
        return self.life_model.df(x)

    def hf(self, x: npt.ArrayLike, Z=None) -> npt.NDArray:
        """Hazard rate of the fitted life model (``Z`` for accelerated)."""
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg.hf(x, Z)
        return self.life_model.hf(x)

    def Hf(self, x: npt.ArrayLike, Z=None) -> npt.NDArray:
        """Cumulative hazard of the life model (``Z`` for accelerated)."""
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg.Hf(x, Z)
        return self.life_model.Hf(x)

    def qf(self, p: npt.ArrayLike, Z=None) -> npt.NDArray:
        """
        Quantile function of the fitted life model.

        Plain life models expose their own ``qf``; accelerated regression
        models do not, so the quantile at stress ``Z`` is obtained by
        numerically inverting the survival function.
        """
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg_qf(p, Z)
        return self.life_model.qf(p)

    def mean(self, Z=None) -> float:
        """
        Mean of the fitted life model.

        For an accelerated model the mean life at stress ``Z`` is obtained by
        integrating the survival function (the regression model has no closed
        ``mean``).
        """
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            return self._reg_mean(Z)
        return self.life_model.mean()

    def random(self, size: int, Z=None, random_state=None) -> npt.NDArray:
        """
        Random pseudo failure times from the fitted life model.

        For an accelerated model, ``size`` samples are drawn at stress ``Z`` by
        inverse-transform sampling of the fitted survival function (the
        regression models do not all expose ``random`` directly).
        """
        Z = self._predict_Z(Z)
        if self.is_accelerated:
            rng = np.random.default_rng(random_state)
            u = rng.uniform(size=size)
            return self._reg_qf(u, Z)
        return self.life_model.random(size)

    def induced_life(
        self, n_samples: int = 10_000, random_state=None
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

        Returns
        -------
        InducedFailureDistribution
            The Monte-Carlo induced failure-time distribution.
        """
        if self.is_accelerated:
            raise ValueError(
                "induced_life uses the (non-accelerated) population "
                "path-parameter distribution; an accelerated (covariate) "
                "model's path parameters are not yet stress-conditional, so "
                "there is no single population to induce a life from."
            )
        rng = np.random.default_rng(random_state)
        mean = np.asarray(self.path_param_mean, dtype=float)
        cov = np.asarray(self.path_param_cov, dtype=float)
        # Robust MVN sampling: symmetrise and clip the (possibly PSD-clipped)
        # covariance's eigenvalues to be non-negative before taking its root.
        cov = (cov + cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        root = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
        z = rng.standard_normal((n_samples, mean.size))
        theta = mean + z @ root.T

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
            t, self.threshold, self.path_model.name
        )

    def _reg_qf(self, p: npt.ArrayLike, Z) -> npt.NDArray:
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

        def target_sf(t):
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

    def _reg_mean(self, Z) -> float:
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
        seed=None,
        Z=None,
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
            accelerated model, rejected for a plain one.

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
                return bootstrap_cb_accelerated(
                    self, x, Z, on, alpha_ci, bound, n_boot, seed
                )
            raise ValueError("`method` must be 'analytic' or 'bootstrap'")
        if method == "analytic":
            return analytic_cb(self, x, on, alpha_ci, bound)
        elif method == "bootstrap":
            return bootstrap_cb(self, x, on, alpha_ci, bound, n_boot, seed)
        raise ValueError("`method` must be 'analytic' or 'bootstrap'")

    def plot(self, ax=None):
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
                start, end = min(start, pseudo), max(end, pseudo)
            x_plot = np.linspace(start, end, 200)
            (line,) = ax.plot(
                x_plot,
                self.path_model.path(x_plot, *self.path_params[idx]),
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

    def __repr__(self):
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
                "\nParameters          :\n" + param_string
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
         alpha: 441.4780882117898
          beta: 6.987078993008337
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
        distribution=Weibull,
        how: str = "MLE",
        population_method: str = "moments",
        Z: npt.ArrayLike | None = None,
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

        path_selection = None
        if isinstance(path, str) and path.lower() == "best":
            path_model, path_selection = self._select_path_model(
                x_arr, y_arr, i_arr, units
            )
        else:
            path_model = get_path_model(path)

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

        for idx, unit in enumerate(units):
            mask = i_arr == unit
            x_unit, y_unit = x_arr[mask], y_arr[mask]
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
            try:
                estimation_cov_sum += np.linalg.inv(jtj)
            except np.linalg.LinAlgError:
                estimation_cov_sum += np.linalg.pinv(jtj)
            y_by_unit.append(y_unit)
            x_by_unit.append(x_unit)
            design_by_unit.append(jacobian)

        # Two-stage (Lu-Meeker) noise correction: the scatter of the
        # per-unit estimates is Sigma + V_i, so subtracting the average
        # estimation covariance leaves the between-unit covariance.
        measurement_var = rss_total / dof_total if dof_total > 0 else 0.0
        path_param_mean = path_params.mean(axis=0)
        path_param_sample_cov = np.atleast_2d(
            np.cov(path_params, rowvar=False, ddof=1)
        )
        mean_estimation_cov = measurement_var * estimation_cov_sum / len(units)
        path_param_cov, was_clipped = _clip_psd(
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
            # a nonlinear one is fitted by FOCE linearisation
            if path_model.linear_in_parameters:
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

        Z_units = None
        if Z is None:
            life_model = distribution.fit(x=pseudo_failure_times, c=c, how=how)
        else:
            Z_units = self._handle_Z(Z, i_arr, units)
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
            Z=Z_units,
        )
        # Recorded so the bootstrap confidence bounds can rerun the pipeline
        # (with the selected path model held fixed) on resampled units.
        model._distribution = distribution
        model._how = how
        return model

    @staticmethod
    def _select_path_model(
        x_arr, y_arr, i_arr, units
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
        **fit_kwargs,
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
                    "stress".format(unit)
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
