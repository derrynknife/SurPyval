"""What a degradation model predicts about life.

``RULPrediction`` is the remaining-useful-life prediction of one unit
(:meth:`DegradationModel.predict_rul
<surpyval.degradation.degradation_analysis.DegradationModel.predict_rul>`);
``InducedFailureDistribution`` is the failure-time distribution a
degradation model induces (``DegradationModel.induced_life``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import warn_outside_unit_interval


@dataclass
class RULPrediction:
    """
    Posterior failure-time / remaining-useful-life prediction for a
    new unit, returned by :meth:`DegradationModel.predict_rul`.

    All summaries come from Monte Carlo samples of the new unit's
    path parameters drawn from their Gaussian posterior and pushed
    through the path model's threshold crossing. Samples whose path
    never reaches the threshold contribute ``inf`` failure times, so
    the median and interval endpoints are ``inf`` when that much of the
    posterior mass never fails. Samples whose path is already past the
    threshold at the unit's first measurement (it crossed at or before
    time zero) have failed: they contribute a failure time of ``0``, so a
    trajectory that starts past the threshold has ``prob_failed = 1``,
    ``failure_time = 0`` and a remaining life of minus its age.

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
        threshold (a path already past it has failed, not "never
        fails").
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
        sampled path never reaches the threshold, ``0`` where it is
        already past the threshold at the first measurement).

    Examples
    --------
    A new unit, measured three times, of a population of eight fitted
    units:

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
    >>> round(pred.rul), [round(v) for v in pred.rul_interval]
    (1173, [1095, 1262])
    >>> pred.prob_failed
    0.0
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

    Draws whose path never reaches the threshold are recorded as ``inf`` --
    a defective ("never fails") mass exposed as ``prob_never_fails`` -- so
    the quantiles and the mean are ``inf`` once they reach into that mass.
    Draws whose path is already past the threshold at the earliest
    measurement time (it crossed at or before time zero) have failed from
    the start and are recorded as ``0``, an atom of failures at time zero.

    Use it as a diagnostic: overlay ``induced.ff(t)`` on the model's own
    ``ff(t)`` (the pseudo-failure fit); close agreement is evidence that the
    path model and its population summary are consistent with the
    pseudo-failure lifetime fit.

    Parameters
    ----------
    samples : numpy array
        The Monte-Carlo failure-time draws (``inf`` where the path never
        reaches the threshold, ``0`` where it is past it from the start).
    threshold : float
        The degradation failure threshold used.
    path_name : str
        Name of the degradation path model.
    stress : list of float, optional
        The stress row the distribution was induced at, for a model whose
        path parameters depend on stress; ``None`` for the plain
        population.

    Examples
    --------
    The induced life of a fitted population of eight units, next to the
    Weibull fitted to their pseudo failure times:

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
    >>> induced
    InducedFailureDistribution(Linear path, threshold=450, median=1451.95,
    prob_never_fails=0)
    >>> induced.sf([1200, 1500]).round(4)
    array([0.9949, 0.3489])
    >>> model.sf([1200, 1500]).round(4)
    array([0.9505, 0.4147])
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

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """Failure probability ``P(T <= x)`` from the Monte-Carlo draws
        (``nan`` at a missing time)."""
        x = np.asarray(x, dtype=float)
        out = (self.samples[None, :] <= x[:, None]).mean(axis=1)
        # no draw is <= nan, so a missing time read as ff = 0 (sf = 1)
        return np.where(np.isnan(x), np.nan, out)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Survival function ``P(T > x)``."""
        return 1.0 - self.ff(x)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """Quantile of the induced distribution (``inf`` in the never-fails
        mass, ``nan`` for a missing probability, and ``nan`` with a warning
        for one outside [0, 1], as every model's ``qf`` gives; #611)."""
        p = np.asarray(p, dtype=float)
        missing = np.isnan(p) | warn_outside_unit_interval(p)
        out = np.full(p.shape, np.nan)
        out[~missing] = np.quantile(self.samples, p[~missing], method="lower")
        return out

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
        rng = as_generator(random_state)
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
