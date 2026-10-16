"""Dynamic prediction: a unit's survival given its readings so far
(#891)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.linalg import delta_method_se, sf_link_bound
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    warn_outside_unit_interval,
)

from ._predict import Posterior

if TYPE_CHECKING:  # pragma: no cover
    from .joint_model import JointModel


class DynamicPrediction:
    """
    One unit's failure-age distribution given its marker readings up to
    ``age`` and that it is still running at ``age``: the dynamic
    prediction of a joint model, which :meth:`JointModel.predict
    <surpyval.joint.JointModel.predict>` returns.

    Its random effects are averaged over their posterior given the
    readings and the survival to ``age`` (adaptive Gauss-Hermite
    quadrature at the posterior's mode), so ``sf(x)`` is ``P(T > x | T >
    age, readings)``: 1 up to ``age``, then falling. Each new reading,
    given to ``predict`` again, updates it. The functions take ages
    (times since new, as the model's), not times from now; the remaining
    life is :meth:`remaining_life`.

    Attributes
    ----------
    age : float
        The age the prediction is made at (the unit is running then).
    marker_x, marker_y : numpy.ndarray
        The readings it is conditioned on.

    Examples
    --------
    >>> from surpyval.joint import JointModel
    >>> model = JointModel.from_params(
    ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
    ...      "path_intercept": 3.0, "path_slope": 0.004,
    ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
    ...      "sigma": 0.5},
    ...     marker_centre=3.0,
    ... )
    >>> worn = model.predict([0, 100, 200, 300], [3.4, 4.1, 4.6, 5.2])
    >>> fresh = model.predict([0, 100, 200, 300], [2.6, 2.7, 2.9, 3.0])
    >>> bool(worn.sf(600) < fresh.sf(600))
    True
    >>> worn.sf(300)
    np.float64(1.0)
    """

    def __init__(
        self,
        model: "JointModel",
        marker_x: npt.NDArray,
        marker_y: npt.NDArray,
        z: npt.NDArray,
        w: npt.NDArray,
        age: float,
    ) -> None:
        self.model = model
        self.marker_x = np.asarray(marker_x, dtype=float)
        self.marker_y = np.asarray(marker_y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.age = float(age)
        self._posterior = self._build(model._eval_params())

    def _build(self, values: npt.NDArray) -> Posterior:
        model = self.model
        return Posterior.build(
            model._evaluator(values),
            self.marker_x,
            self.marker_y,
            self.z,
            self.w,
            self.age,
            model.marker_centre,
        )

    # -- the distribution of the failure age ---------------------------------
    def _log_sf_at(self, x: npt.NDArray, post: Posterior, values: Any) -> Any:
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape)
        later = x > self.age
        if np.any(later):
            ev = self.model._evaluator(values)
            out[later] = post.log_sf(ev, x[later])
        return np.where(np.isnan(x), np.nan, out)

    def _log_sf(self, x: npt.NDArray) -> npt.NDArray:
        return self._log_sf_at(x, self._posterior, self.model._eval_params())

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """``P(T > x | T > age, readings)``: the probability the unit is
        still running at age ``x`` (1 up to ``age``)."""
        return np.exp(self._log_sf(np.asarray(x, dtype=float)))

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """The probability the unit has failed by age ``x``, given it was
        running at ``age``: ``1 - sf(x)``."""
        return -np.expm1(self._log_sf(np.asarray(x, dtype=float)))

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The cumulative hazard from ``age`` to ``x``, ``-log sf(x)``."""
        return 0.0 - self._log_sf(np.asarray(x, dtype=float))

    def _log_df(self, x: npt.NDArray) -> npt.NDArray:
        out = np.full(x.shape, -np.inf)
        later = x > self.age
        if np.any(later):
            ev = self.model._evaluator(self.model._eval_params())
            out[later] = self._posterior.log_df(ev, x[later])
        return np.where(np.isnan(x), np.nan, out)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """The density of the failure age at ``x``, given the readings and
        survival to ``age`` (0 up to ``age``)."""
        return np.exp(self._log_df(np.asarray(x, dtype=float)))

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The unit's hazard at age ``x``, given its readings and survival
        to ``age`` (``df / sf``; 0 up to ``age``)."""
        x = np.asarray(x, dtype=float)
        with np.errstate(invalid="ignore"):
            out = np.exp(self._log_df(x) - self._log_sf(x))
        return np.where(x <= self.age, 0.0, out)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """The age by which the unit has failed with probability ``p``,
        given it was running at ``age`` (``age`` itself for ``p = 0``,
        ``nan`` with a warning outside [0, 1])."""
        p = np.asarray(p, dtype=float)
        p = np.where(warn_outside_unit_interval(p), np.nan, p)
        target = np.log1p(-np.where(np.isfinite(p), np.minimum(p, 1.0), 0.0))
        out = np.full(p.shape, np.nan)
        ok = np.isfinite(p)
        out[ok & (p <= 0)] = self.age
        out[ok & (p >= 1)] = np.inf
        inner = np.flatnonzero(ok & (p > 0) & (p < 1))
        if inner.size:
            out[inner] = self._solve(target[inner])
        return out

    def _solve(self, target: npt.NDArray) -> npt.NDArray:
        """The ages at which ``log sf`` reaches ``target`` (each below 0),
        by bisection in the log of the time since ``age``; ``inf`` where
        it never does (a unit whose path may fall away, so that its hazard
        dies out, has a chance of never failing)."""
        return self.age + _bisect_log_time(
            lambda t: self._log_sf(self.age + t), target, self.model.params[0]
        )

    def remaining_life(self, p: npt.ArrayLike = 0.5) -> Any:
        """The remaining life from ``age`` that the unit outlives with
        probability ``1 - p``: ``qf(p) - age``. The default is the median
        remaining life; ``remaining_life([0.05, 0.5, 0.95])`` gives it
        with a 90% prediction interval (of the unit's life, not a
        confidence interval: see :meth:`cb` for the parameters'
        uncertainty).

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
        ...      "path_intercept": 3.0, "path_slope": 0.004,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> pred = model.predict([0, 100, 200], [3.1, 3.3, 3.9])
        >>> life = pred.remaining_life([0.1, 0.5, 0.9])
        >>> bool(life[0] < life[1] < life[2])
        True
        """
        return self.qf(p) - self.age

    # -- uncertainty --------------------------------------------------------
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Wald confidence bounds on the conditional ``sf``, ``ff`` or ``Hf``
        at ages ``x``, from the model's parameter covariance by the delta
        method (the posterior of the unit's random effects recomputed at
        each parameter value), formed on ``log(-log sf)`` so they stay in
        [0, 1]. Up to ``age`` the bounds are the estimate, 1.

        Parameters
        ----------
        x : array_like
            The ages.
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are ``'sf'`` and
            ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Default ``'two-sided'``, ``[lower, upper]`` on a last axis.

        Returns
        -------
        numpy.ndarray
            The bound(s), the shape of ``x`` (with a last axis of 2 for
            two-sided bounds).
        """
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        on = {"R": "sf", "F": "ff"}.get(on, on)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        model = self.model
        cov = model.covariance()
        x_arr = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x_arr).ravel()
        log_sf = self._log_sf(flat)
        sf_hat, ff_hat = np.exp(log_sf), -np.expm1(log_sf)

        def sf_of(estimates: npt.NDArray) -> npt.NDArray:
            values = model._values_with(estimates)
            post = self._build(values)
            return np.exp(self._log_sf_at(flat, post, values))

        with np.errstate(all="ignore"):
            se = delta_method_se(sf_of, model._estimates(), cov)
        se = np.where(flat <= self.age, 0.0, se)
        out = sf_link_bound(
            sf_hat, se, alpha_ci, bound, "loglog", ff_hat=ff_hat, on=on
        )
        shape = x_arr.shape + ((2,) if bound == "two-sided" else ())
        return out.reshape(shape)

    # -- the marker ----------------------------------------------------------
    def marker(self, x: npt.ArrayLike) -> Any:
        """The posterior mean of the unit's true (noise-free) marker at
        ages ``x``, given its readings and survival to ``age``: its
        smoothed path, and beyond ``age`` its projection.

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
        ...      "path_intercept": 3.0, "path_slope": 0.004,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> pred = model.predict([0, 100, 200], [3.1, 3.3, 3.9])
        >>> pred.marker([0, 200, 400]).round(2)
        array([3.02, 3.82, 4.62])
        """
        x_arr = np.asarray(x, dtype=float)
        mean, _ = self._posterior.path_moments(
            np.atleast_1d(x_arr).ravel(), self.model.marker_centre
        )
        return mean.reshape(x_arr.shape)[()]

    def marker_interval(
        self, x: npt.ArrayLike, alpha_ci: float = 0.05
    ) -> npt.NDArray:
        """An equal-tailed ``1 - alpha_ci`` interval for the unit's true
        marker at ages ``x`` (its posterior, normal approximation), at the
        fitted parameters: ``[lower, upper]`` on a last axis. It is the
        spread of the unit's own path, not of a new reading (which adds
        the noise ``sigma``)."""
        from scipy.stats import norm

        check_alpha_ci(alpha_ci)
        x_arr = np.asarray(x, dtype=float)
        mean, sd = self._posterior.path_moments(
            np.atleast_1d(x_arr).ravel(), self.model.marker_centre
        )
        z = norm.ppf(1 - alpha_ci / 2)
        out = np.stack([mean - z * sd, mean + z * sd], axis=-1)
        return out.reshape(x_arr.shape + (2,))

    # -- draws ---------------------------------------------------------------
    def random(
        self,
        size: int,
        random_state: "int | np.random.Generator | None" = None,
    ) -> npt.NDArray:
        """
        Failure ages drawn from the prediction (each after ``age``).

        Parameters
        ----------
        size : int
            The number of draws.
        random_state : int or numpy.random.Generator, optional
            ``None`` (the default) draws from numpy's global generator,
            so ``np.random.seed`` reproduces it; an int or a generator
            gives a stream of its own.
        """
        rng = as_generator(random_state)
        u = rng.uniform(size=int(size))
        return np.asarray(self.qf(u), dtype=float)

    def __repr__(self) -> str:
        median = float(self.remaining_life(0.5))
        return (
            "Dynamic prediction of a joint model\n"
            "===================================\n"
            "Age now             : {:.6g}\n"
            "Readings            : {}\n"
            "Median life left    : {:.6g}".format(
                self.age, self.marker_x.size, median
            )
        )


def _bisect_log_time(
    log_sf: Any, target: npt.NDArray, scale: float
) -> npt.NDArray:
    """The times ``t`` at which ``log_sf(t)`` (decreasing) falls to
    ``target``, by bisection in ``log t`` from around ``scale``; ``inf``
    where it stays above ``target`` beyond ``exp(60)`` times ``scale``."""
    centre = float(np.log(max(scale, 1e-300)))
    lo = np.full(target.shape, centre - 40.0)
    hi = np.full(target.shape, centre + 2.0)
    for _ in range(30):
        short = log_sf(np.exp(hi)) > target
        if not short.any():
            break
        hi = np.where(short, hi + 2.0, hi)
    never = log_sf(np.exp(hi)) > target
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = log_sf(np.exp(mid)) > target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(never, np.inf, np.exp(0.5 * (lo + hi)))
