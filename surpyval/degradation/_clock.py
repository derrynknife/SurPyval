"""The accelerated clock shared by the step-stress degradation models.

Stress acts on a degradation model by speeding up its clock -- the
cumulative-exposure principle. With the acceleration factor
``AF(z) = exp(gamma' (z - z_ref))`` a unit under the stress path ``z(s)``
has aged ``tau(t) = integral_0^t AF(z(s)) ds`` of reference-stress time by
calendar time ``t``. Both the process models and the general-path models
use it: the reference-stress model runs on ``tau`` rather than ``t``.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.univariate.regression.tvc_schedule import (
    StepSchedule,
    segments_from_origin,
)


def stress_row(Z: Any, q: int) -> npt.NDArray:
    """Validate one constant stress row with ``q`` covariates."""
    z = np.asarray(Z, dtype=float)
    if z.ndim == 2 and z.shape[0] == 1:
        z = z[0]
    z = np.atleast_1d(z)
    if z.shape != (q,):
        raise ValueError(
            "Z must be a single stress row with {} covariate(s), or a "
            "StepSchedule for a stress profile; got shape {}".format(
                q, z.shape
            )
        )
    if not np.isfinite(z).all():
        raise ValueError("Z must contain only finite values")
    return z


class StressClock:
    """
    The clock under a stress path, ``tau(t) = int_0^t AF(z(s)) ds``.

    ``Z`` is one constant stress row (``tau`` is then ``AF * t``) or a
    :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
    describing a piecewise-constant profile, in which case ``tau`` is piecewise
    linear with slope ``AF`` of the stress in force. ``acceleration_factor``
    maps one stress row to its ``AF`` and ``q`` is the number of covariates the
    model was fitted with.
    """

    def __init__(
        self,
        acceleration_factor: Callable[[Any], float],
        q: int,
        Z: Any,
    ) -> None:
        self.acceleration_factor = acceleration_factor
        self.schedule: "StepSchedule | None" = None
        self.rate: "float | None" = None
        if isinstance(Z, StepSchedule):
            if Z.p != q:
                raise ValueError(
                    "the StepSchedule has {} covariate(s) but the model was "
                    "fitted with {}".format(Z.p, q)
                )
            self.schedule = Z
        else:
            self.rate = acceleration_factor(Z)

    def _knots(self, horizon: float) -> tuple[npt.NDArray, npt.NDArray]:
        assert self.schedule is not None
        finite_edges = self.schedule.edges[np.isfinite(self.schedule.edges)]
        horizon = max(horizon, float(finite_edges.max()) + 1.0, 1.0)
        starts, ends, Zs = segments_from_origin(self.schedule, horizon)
        af = np.array([self.acceleration_factor(z) for z in Zs])
        knots_t = np.concatenate([[starts[0]], ends])
        knots_tau = np.concatenate([[0.0], np.cumsum(af * (ends - starts))])
        return knots_t, knots_tau

    def tau(self, t: npt.NDArray) -> npt.NDArray:
        if self.rate is not None:
            return self.rate * t
        finite = t[np.isfinite(t)]
        knots_t, knots_tau = self._knots(float(finite.max(initial=0.0)))
        out = np.interp(t, knots_t, knots_tau)
        return np.where(np.isposinf(t), np.inf, out)

    def rate_at(self, t: npt.NDArray) -> npt.NDArray:
        if self.rate is not None:
            return np.full_like(t, self.rate, dtype=float)
        finite = t[np.isfinite(t)]
        knots_t, knots_tau = self._knots(float(finite.max(initial=0.0)))
        slopes = np.diff(knots_tau) / np.diff(knots_t)
        idx = np.searchsorted(knots_t, t, side="right") - 1
        return slopes[np.clip(idx, 0, len(slopes) - 1)]

    def inverse(self, tau: npt.NDArray) -> npt.NDArray:
        """The calendar time at which the clock reads ``tau``."""
        tau = np.asarray(tau, dtype=float)
        if self.rate is not None:
            return tau / self.rate
        finite = tau[np.isfinite(tau)]
        target = float(finite.max(initial=0.0))
        horizon = 1.0
        knots_t, knots_tau = self._knots(horizon)
        for _ in range(200):
            if knots_tau[-1] >= target:
                break
            horizon = 2.0 * float(knots_t[-1])
            knots_t, knots_tau = self._knots(horizon)
        out = np.interp(tau, knots_tau, knots_t)
        return np.where(np.isposinf(tau), np.inf, out)


class HistoryClock:
    """
    A unit's clock from its measured stress history, continued by a future
    stress.

    ``Z_rows`` holds the stress over each measurement interval (one row per
    measurement, the interval ending at it; the first starts at time zero). Up
    to the last measurement the clock is that history; after it, it runs under
    ``Z_future`` -- a stress row or a
    :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` whose
    time zero is the last measurement -- or, by default, the last stress held.
    """

    def __init__(
        self,
        x: npt.NDArray,
        Z_rows: npt.NDArray,
        acceleration_factor: Callable[[Any], float],
        q: int,
        Z_future: Any = None,
    ) -> None:
        order = np.argsort(x, kind="stable")
        x_sorted = x[order]
        rates = np.array([acceleration_factor(z) for z in Z_rows[order]])
        tau_sorted = np.cumsum(
            np.diff(np.concatenate([[0.0], x_sorted])) * rates
        )
        #: reference-stress time at each measurement, aligned to ``x``
        self.tau = np.empty_like(tau_sorted)
        self.tau[order] = tau_sorted
        self.knots_t = np.concatenate([[0.0], x_sorted])
        self.knots_tau = np.concatenate([[0.0], tau_sorted])
        self.age = float(x_sorted[-1])
        self.tau_age = float(tau_sorted[-1])
        future = Z_rows[order][-1] if Z_future is None else Z_future
        self.future = StressClock(acceleration_factor, q, future)

    def calendar(self, tau: npt.ArrayLike) -> npt.NDArray:
        """The calendar time at which the clock reads ``tau`` (``inf``
        stays ``inf``)."""
        tau = np.asarray(tau, dtype=float)
        past = np.interp(tau, self.knots_tau, self.knots_t)
        ahead = np.maximum(
            np.where(np.isfinite(tau), tau, 0.0) - self.tau_age, 0.0
        )
        future = self.age + self.future.inverse(ahead)
        out = np.where(tau <= self.tau_age, past, future)
        return np.where(np.isposinf(tau), np.inf, out)
