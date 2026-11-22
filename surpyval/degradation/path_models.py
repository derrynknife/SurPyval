"""Degradation path models.

A path model describes the deterministic trend of a degradation
measurement over time. Fitting one to a single unit's repeated
measurements gives that unit's expected degradation path; extrapolating
the path to the failure threshold gives the unit's *pseudo failure
time*. The models here are the ones in common reliability engineering
use for this purpose.

Each model implements:

- ``path(x, *params)``: the degradation level at time(s) ``x``,
- ``inv_path(y, *params)``: the time at which the path reaches level
  ``y`` (non-finite when the path never does),
- ``fit(x, y)``: least-squares estimates of the path parameters from
  one unit's measurements,
- ``jacobian(x, *params)``: the analytic derivatives of the path with
  respect to its parameters.

Models that are linear in their parameters (``LinearPath``,
``QuadraticPath``, ``LogarithmicPath``, ``LloydLipowPath``) are fitted
with ordinary least squares in closed form; the others
(``ExponentialPath``, ``OffsetExponentialPath``, ``PowerPath``,
``GompertzPath``, ``MichaelisMentenPath``) use nonlinear least squares
started from a linearised fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from surpyval.utils.units import takes_unit
from surpyval.utils.validation import option_error


class _StopSearch(Exception):
    """Raised by a path a least-squares search evaluates, to end that
    search (see ``OffsetExponentialPath_.fit``)."""


def _ols(z: npt.NDArray, y: npt.NDArray) -> tuple[float, float]:
    """Closed-form least squares fit of ``y = intercept + slope * z``."""
    A = np.column_stack([np.ones_like(z), z])
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    return intercept, slope


def _columns(x: npt.NDArray, *columns: Any) -> npt.NDArray:
    """``columns`` (each a number or an array broadcasting with ``x``)
    side by side, one row per entry of ``x``."""
    return np.column_stack(
        [np.broadcast_to(np.asarray(c, dtype=float), x.shape) for c in columns]
    )


class PathModel(ABC):
    """
    Base class for degradation path models.

    A path model is a deterministic function of time with a small
    number of parameters that is fitted, per unit, to that unit's
    degradation measurements. Subclass this (implementing ``path``,
    ``inv_path`` and the ``name``/``parameter_names`` attributes, and
    either a ``_initial_guess(x, y)`` starting point for the default
    least-squares ``fit`` or ``fit`` itself) to use a custom degradation
    path with ``DegradationAnalysis``.
    """

    name: str
    parameter_names: list[str]
    #: True when ``path`` is linear in its parameters, i.e.
    #: ``path(x, *theta) == jacobian(x) @ theta`` with a Jacobian that
    #: does not depend on ``theta``. Enables exact conjugate posterior
    #: updates and REML population estimation.
    linear_in_parameters: bool = False

    @abstractmethod
    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        """Evaluate the degradation path at time(s) ``x``."""

    @abstractmethod
    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        """
        Time at which the path reaches level ``y``.

        Returns a non-finite value (``nan`` or ``inf``) or a
        non-positive value when the path never reaches ``y`` at a
        positive finite time.
        """

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        """
        Partial derivatives of ``path`` with respect to the
        parameters, evaluated at time(s) ``x``: a
        ``(len(x), n_params)`` matrix.

        Used to estimate the least-squares estimation covariance of
        per-unit fitted parameters. The base implementation uses
        central finite differences; the built-in models override it
        with the analytic derivatives.
        """
        x = np.asarray(x, dtype=float)
        params_arr = np.asarray(params, dtype=float)
        columns = []
        for j in range(len(params_arr)):
            h = 1e-6 * max(abs(params_arr[j]), 1.0)
            upper, lower = params_arr.copy(), params_arr.copy()
            upper[j] += h
            lower[j] -= h
            columns.append(
                (self.path(x, *upper) - self.path(x, *lower)) / (2.0 * h)
            )
        return np.column_stack(columns)

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        """
        The path's rate of change with time, ``d path / d x``, at
        time(s) ``x`` (the parameters numbers, or arrays broadcasting
        with ``x``).

        The base implementation uses central finite differences in
        ``x`` (a step of ``1e-6 max(|x|, 1)``, so the path must be
        defined a little either side of ``x``); the built-in models
        override it with the analytic derivative.
        """
        x = np.asarray(x, dtype=float)
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        return (
            np.asarray(self.path(x + h, *params), dtype=float)
            - np.asarray(self.path(x - h, *params), dtype=float)
        ) / (2.0 * h)

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        """
        Partial derivatives of :meth:`slope` with respect to the
        parameters, at time(s) ``x``: a ``(len(x), n_params)`` matrix.

        The base implementation uses central finite differences of
        :meth:`slope` in each parameter; the built-in models override it
        with the analytic derivatives.
        """
        x = np.asarray(x, dtype=float)
        columns = []
        for j in range(len(params)):
            value = np.asarray(params[j], dtype=float)
            h = 1e-6 * np.maximum(np.abs(value), 1.0)
            upper: list[Any] = list(params)
            lower: list[Any] = list(params)
            upper[j] = value + h
            lower[j] = value - h
            columns.append(
                (self.slope(x, *upper) - self.slope(x, *lower)) / (2.0 * h)
            )
        return np.column_stack([np.broadcast_to(c, x.shape) for c in columns])

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        """Raise ``ValueError`` if the data is outside the model domain."""

    def _require_positive_times(self, x: npt.NDArray, why: str) -> None:
        """Refuse times at or below 0, saying ``why`` the model cannot use
        them and to drop them: the baseline reading at ``t = 0`` is the
        usual case (#663)."""
        if np.any(np.asarray(x, dtype=float) <= 0):
            raise ValueError(
                f"The {self.name.lower()} path model requires strictly "
                "positive times (there are measurements at t <= 0): "
                f"{why}. "
                "Drop the t = 0 rows, or use a path defined there (such as "
                "'linear', 'quadratic' or 'exponential')."
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        """One starting parameter vector, or a list of candidates."""
        raise NotImplementedError

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        """
        Fit the path parameters to one unit's measurements by
        (nonlinear) least squares.

        Parameters
        ----------
        x : array_like
            The unit's measurement times.
        y : array_like
            Its degradation measurements.

        Returns
        -------
        numpy array
            The fitted parameters, in the order of ``parameter_names``.

        Examples
        --------
        >>> from surpyval.degradation import LinearPath
        >>> LinearPath.fit([1, 2, 3, 4], [10.5, 12.1, 13.4, 15.2]).round(4)
        array([8.95, 1.54])
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        guesses = np.atleast_2d(
            np.asarray(self._initial_guess(x, y), dtype=float)
        )
        best_params, best_rss = self._least_squares(x, y, guesses)
        if best_params is None:
            raise ValueError(
                f"Could not fit the {self.name} path model to the data"
            )
        return best_params

    def _least_squares(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        guesses: npt.NDArray,
        path: Any = None,
        stopped: list | None = None,
    ) -> tuple[npt.NDArray | None, float]:
        """``(params, rss)``: the best least-squares fit of ``path`` (the
        model's own by default) from each of ``guesses``, ``(None, inf)``
        where none converges. A search ``path`` stops (by raising
        :class:`_StopSearch`) has its guess appended to ``stopped``."""
        path = self.path if path is None else path
        best_params, best_rss = None, np.inf
        for p0 in guesses:
            try:
                params, _ = curve_fit(path, x, y, p0=p0, maxfev=10_000)
            except RuntimeError:
                continue
            except _StopSearch:
                if stopped is not None:
                    stopped.append(p0)
                continue
            residuals = y - self.path(x, *params)
            rss = float(residuals @ residuals)
            if np.isfinite(rss) and rss < best_rss:
                best_params, best_rss = params, rss
        return best_params, best_rss

    def __repr__(self) -> str:
        return f"{self.name} Degradation Path Model"


class LinearPath_(PathModel):
    """Linear degradation path: ``y = a + b * x``."""

    name = "Linear"
    parameter_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        return a + b * np.asarray(x, dtype=float)

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (y - a) / b

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), x])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        return b + 0.0 * x

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        return _columns(x, 0.0, 1.0)

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        return np.array(_ols(x, y))


class ExponentialPath_(PathModel):
    """Exponential degradation path: ``y = a * exp(b * x)``."""

    name = "Exponential"
    parameter_names = ["a", "b"]

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        return a * np.exp(b * np.asarray(x, dtype=float))

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(y / a) / b

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        e = np.exp(b * x)
        return np.column_stack([e, a * x * e])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        return a * b * np.exp(b * x)

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        e = np.exp(b * x)
        return _columns(x, b * e, a * e * (1.0 + b * x))

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if (y <= 0).any():
            raise ValueError(
                "The exponential path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        intercept, slope = _ols(x, np.log(y))
        return np.exp(intercept), slope


class PowerPath_(PathModel):
    """Power degradation path: ``y = a * x**b``."""

    name = "Power"
    parameter_names = ["a", "b"]

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * np.power(np.asarray(x, dtype=float), b)

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.power(y / a, 1.0 / b)

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        xb = np.power(x, b)
        return np.column_stack([xb, a * xb * np.log(x)])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * b * np.power(x, b - 1.0)

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            xb = np.power(x, b - 1.0)
            return _columns(x, b * xb, a * xb * (1.0 + b * np.log(x)))

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(
            x,
            "y = a * t**b is 0 at t = 0, so a reading there says nothing "
            "about a and b, and the fit works on log t",
        )
        if (y <= 0).any():
            raise ValueError(
                "The power path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        intercept, slope = _ols(np.log(x), np.log(y))
        return np.exp(intercept), slope


class LogarithmicPath_(PathModel):
    """Logarithmic degradation path: ``y = a + b * ln(x)``."""

    name = "Logarithmic"
    parameter_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a + b * np.log(np.asarray(x, dtype=float))

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.exp((y - a) / b)

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), np.log(x)])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / x

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _columns(x, 0.0, 1.0 / x)

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(
            x, "y = a + b * ln(t) is undefined at t = 0"
        )

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        return np.array(_ols(np.log(x), y))


class LloydLipowPath_(PathModel):
    """Lloyd-Lipow degradation path: ``y = a - b / x``."""

    name = "Lloyd-Lipow"
    parameter_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a - b / np.asarray(x, dtype=float)

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / (a - y)

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), -1.0 / x])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / x**2

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _columns(x, 0.0, 1.0 / x**2)

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(x, "y = a - b / t is undefined at t = 0")

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        intercept, slope = _ols(1.0 / x, y)
        return np.array([intercept, -slope])


class QuadraticPath_(PathModel):
    """Quadratic degradation path: ``y = a + b * x + c * x**2``."""

    name = "Quadratic"
    parameter_names = ["a", "b", "c"]
    linear_in_parameters = True

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        return a + b * x + c * x**2

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        """First positive time at which the parabola reaches ``y``."""
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            linear_root = (y - a) / b
            disc = b**2 - 4.0 * c * (a - y)
            sqrt_disc = np.sqrt(np.where(disc >= 0, disc, np.nan))
            # The textbook (-b +/- sqrt(disc)) / 2c cancels catastrophically
            # for the root near the linear one when the curvature is tiny
            # (a quadratic fitted to straight-line data has c ~ 1e-17 and
            # gave a crossing at 42.75 instead of 16). The conjugate form
            # adds terms of one sign only, and its second root tends to
            # the linear root (y - a) / b as c -> 0.
            sign_b = np.where(np.asarray(b) >= 0, 1.0, -1.0)
            q = -0.5 * (b + sign_b * sqrt_disc)
            root_minus = q / c
            root_plus = (a - y) / q
            root_minus = np.where(
                np.isfinite(root_minus) & (root_minus > 0),
                root_minus,
                np.inf,
            )
            root_plus = np.where(
                np.isfinite(root_plus) & (root_plus > 0), root_plus, np.inf
            )
            first = np.minimum(root_minus, root_plus)
            first = np.where(np.isfinite(first), first, np.nan)
            return np.where(c == 0, linear_root, first)

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), x, x**2])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        return b + 2.0 * c * x

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        return _columns(x, 0.0, 1.0, 2.0 * x)

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        design = np.column_stack([np.ones_like(x), x, x**2])
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        return params


class GompertzPath_(PathModel):
    """Gompertz degradation path: ``y = a * exp(-b * exp(-c * x))``.

    An S-shaped path approaching the asymptote ``a``.
    """

    name = "Gompertz"
    parameter_names = ["a", "b", "c"]

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            return a * np.exp(-b * np.exp(-c * x))

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log(-np.log(y / a) / b) / c

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        inner = np.exp(-c * x)
        outer = np.exp(-b * inner)
        return np.column_stack(
            [outer, -a * inner * outer, a * b * x * inner * outer]
        )

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            inner = np.exp(-c * x)
            return a * b * c * inner * np.exp(-b * inner)

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            inner = np.exp(-c * x)
            both = inner * np.exp(-b * inner)
        return _columns(
            x,
            b * c * both,
            a * c * both * (1.0 - b * inner),
            a * b * both * (1.0 - c * x + b * c * x * inner),
        )

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if (y <= 0).any():
            raise ValueError(
                "The Gompertz path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        # slightly above the largest measurement as the asymptote, then
        # -ln(y/a) decays exponentially: linearise on its logarithm
        a0 = 1.05 * y.max()
        z = -np.log(y / a0)
        intercept, slope = _ols(x, np.log(z))
        return [a0, np.exp(intercept), -slope]


class OffsetExponentialPath_(PathModel):
    """Offset exponential degradation path: ``y = a + b * exp(c * x)``.

    Covers exponential growth or decay toward/away from the asymptote
    ``a``; with ``a = 0`` it reduces to the exponential path.
    """

    name = "Offset Exponential"
    parameter_names = ["a", "b", "c"]

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            return a + b * np.exp(c * x)

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((y - a) / b) / c

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        e = np.exp(c * x)
        return np.column_stack([np.ones_like(x), e, b * x * e])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            return b * c * np.exp(c * x)

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            e = np.exp(c * x)
        return _columns(x, 0.0, c * e, b * e * (1.0 + c * x))

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        # for an offset outside the observed range, y - a0 has one
        # sign, so its magnitude linearises on a log scale; try an
        # offset on each side and keep the better fit
        span = y.max() - y.min()
        if span == 0:
            span = max(abs(y.max()), 1.0)
        guesses = []
        for a0 in (y.min() - 0.1 * span, y.max() + 0.1 * span):
            shifted = y - a0
            sign = 1.0 if shifted[0] > 0 else -1.0
            intercept, slope = _ols(x, np.log(np.abs(shifted)))
            guesses.append([a0, sign * np.exp(intercept), slope])
        return guesses

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        """
        Fit the path parameters to one unit's measurements by nonlinear
        least squares, from an offset below the data and one above.

        The two starts bend the path opposite ways (``b > 0`` convex,
        ``b < 0`` concave), and on nearly straight measurements the one
        bending against the data has no least-squares minimum: it runs to
        the straight line the family approaches as ``c -> 0`` with ``b c``
        fixed, ``b`` growing without end, and spent 2,500 evaluations
        getting there against the other's 200 to 700 (95% of a 6.4 s
        ``path="best"`` fit of 200 straight units, #621). Near that limit,
        ``|c|`` times the span of the times below ``_LINE_LIMIT``, the
        path is a line plus ``b c^2 x^2 / 2`` to within a third of a
        percent, and its best fit is the line wherever ``b`` bends
        against the curvature of the measurements' least-squares
        quadratic; a search there whose residual sum of squares is no
        less than the line's is stopped. It cannot have won: the other
        start's answer is kept only where its sum of squares is no more
        than the line's, and the stopped searches are run in full where
        it is not, so the answer is the one the full searches give.

        Parameters
        ----------
        x : array_like
            The unit's measurement times.
        y : array_like
            Its degradation measurements.

        Returns
        -------
        numpy array
            The fitted ``a``, ``b`` and ``c``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.degradation import OffsetExponentialPath
        >>> x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = 2.0 + 0.5 * np.exp(0.4 * x)
        >>> OffsetExponentialPath.fit(x, y).round(3)
        array([2. , 0.5, 0.4])
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        guesses = np.atleast_2d(
            np.asarray(self._initial_guess(x, y), dtype=float)
        )
        toward_line = _TowardLine(self, x, y)
        stopped: list = []
        params, rss = self._least_squares(x, y, guesses, toward_line, stopped)
        if stopped and not rss <= toward_line.rss:
            more, more_rss = self._least_squares(x, y, np.array(stopped))
            if more is not None and more_rss < rss:
                params, rss = more, more_rss
        if params is None:
            raise ValueError(
                f"Could not fit the {self.name} path model to the data"
            )
        return params


#: How close to its straight-line limit (``|c|`` times the span of the
#: times) an offset exponential search must be to be judged against the
#: line (see ``OffsetExponentialPath_.fit``).
_LINE_LIMIT = 0.01


class _TowardLine:
    """The offset exponential path, for a least-squares search of one
    unit's measurements ``(x, y)``, that ends the search (raises
    :class:`_StopSearch`) at a point heading for the straight line the
    family approaches as ``c -> 0`` (see ``OffsetExponentialPath_.fit``).
    ``rss`` is the line's residual sum of squares."""

    def __init__(
        self, model: PathModel, x: npt.NDArray, y: npt.NDArray
    ) -> None:
        self.model = model
        self.y = y
        self.span = float(x.max() - x.min())
        intercept, slope = _ols(x, y)
        line = y - intercept - slope * x
        self.rss = float(line @ line)
        # The measurements' curvature: the leading coefficient of their
        # least-squares quadratic (none with fewer than three times)
        self.curvature = 0.0
        if np.unique(x).size >= 3:
            z = x - x.mean()
            A = np.column_stack([np.ones_like(z), z, z**2])
            self.curvature = float(np.linalg.lstsq(A, y, rcond=None)[0][2])

    def __call__(self, x: npt.NDArray, *params: float) -> npt.NDArray:
        level = self.model.path(x, *params)
        _, b, c = params
        if abs(c) * self.span < _LINE_LIMIT and b * self.curvature < 0:
            residuals = self.y - level
            if float(residuals @ residuals) >= self.rss:
                raise _StopSearch
        return level


class MichaelisMentenPath_(PathModel):
    """Michaelis-Menten degradation path: ``y = a * x / (b + x)``.

    A saturating path rising from zero toward the asymptote ``a``,
    reaching half of it at ``x = b``.
    """

    name = "Michaelis-Menten"
    parameter_names = ["a", "b"]

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * x / (b + x)

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b * y / (a - y)

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        denominator = b + x
        return np.column_stack([x / denominator, -a * x / denominator**2])

    def slope(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * b / (b + x) ** 2

    def slope_jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _columns(x, b / (b + x) ** 2, a * (x - b) / (b + x) ** 3)

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(
            x,
            "y = a * t / (b + t) is 0 at t = 0, so a reading there says "
            "nothing about a and b, and the fit works on 1 / t",
        )
        if (y <= 0).any():
            raise ValueError(
                "The Michaelis-Menten path model requires strictly "
                "positive degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> npt.ArrayLike | list:
        # Lineweaver-Burk linearisation: 1/y = 1/a + (b/a) / x
        intercept, slope = _ols(1.0 / x, 1.0 / y)
        if intercept > 0:
            return [1.0 / intercept, slope / intercept]
        return [1.2 * y.max(), float(np.median(x))]


#: How many points of each piece (between neighbouring knots)
#: ``SplinePath.inv_path`` looks at for the first crossing of a level.
_SPLINE_GRID = 32


class SplinePath(PathModel):
    """
    A smooth curve of joined pieces: a natural cubic spline.

    The path is flat and then climbs, or falls and then rises, wherever
    the measurements say so, without a formula being named. It is a
    cubic between neighbouring knots, joined so that its value, slope and
    curvature are continuous, and a straight line before the first
    boundary knot and after the last. With the interior knots
    :math:`\\kappa_1 < \\dots < \\kappa_K` between the boundary knots
    ``lo`` and ``hi``,

    .. math::

        m(t) = \\text{level} + \\text{rate}\\, t + \\sum_{k=1}^{K} s_k N_k(t),

    where each :math:`N_k` is 0 up to its knot (:math:`N_1`'s is ``lo``,
    :math:`N_{k}`'s is :math:`\\kappa_{k-1}`) and a straight line after
    ``hi``. So ``level`` is the line's value at age 0 and ``rate`` its
    slope before ``lo``, and the ``s_k`` (in the measurement's unit) bend
    the curve after each knot. With no interior knots the path is the
    straight line of ``LinearPath``.

    The parameters are ``level``, ``rate``, ``s1``, ..., ``sK``. The path
    is linear in them, so one unit's fit is linear least squares.

    Parameters
    ----------
    knots : array_like, optional
        The interior knots, ascending and strictly between the boundary
        knots. Default: none (a straight line).
    boundary : (float, float), optional
        The boundary knots ``(lo, hi)``: the path is a straight line
        outside them. Left out, they are the first and last measurement
        times of the data a fitter is given
        (``DegradationAnalysis.fit``, ``JointPH(...).fit``), and the
        fitted model's ``path_model`` records them.

    Notes
    -----
    :meth:`natural` places the knots from the data instead. Outside the
    boundary knots the path is extrapolated as a straight line; a
    prediction far beyond them rests on that line.

    The basis is the truncated-power form of the natural cubic spline:
    with every knot :math:`\\xi_1 = \\text{lo} < \\dots < \\xi_{K+2} =
    \\text{hi}` on the scale :math:`u = (t - \\text{lo}) / (\\text{hi} -
    \\text{lo})`, :math:`d_k(u) = [(u - \\xi_k)_+^3 - (u - 1)_+^3] / (1 -
    \\xi_k)` and :math:`N_k = d_k - d_{K+1}`.

    Examples
    --------
    A reading that is flat for 4 years and then climbs: a knee at 4.

    >>> import numpy as np
    >>> from surpyval.degradation import SplinePath
    >>> path = SplinePath(knots=[4.0], boundary=(0.0, 10.0))
    >>> path.parameter_names
    ['level', 'rate', 's1']
    >>> t = np.arange(0.0, 11.0)
    >>> y = np.where(t < 4, 2.0, 2.0 + 0.5 * (t - 4))
    >>> theta = path.fit(t, y)
    >>> path.path([2.0, 8.0], *theta).round(2)
    array([1.97, 3.92])
    >>> path.slope([2.0, 8.0], *theta).round(2)
    array([0.04, 0.57])

    With no interior knots it is the straight line:

    >>> SplinePath().path([1.0, 2.0], 3.0, 0.5)
    array([3.5, 4. ])
    """

    name = "Spline"
    linear_in_parameters = True
    #: The parameters that differ between items in a joint model
    #: (``JointPH``) unless ``varying=`` names others: the bends are
    #: common to the fleet.
    default_varying = ("level", "rate")

    def __init__(self, knots: Any = (), boundary: Any = None) -> None:
        self.df: int | None = None
        self.knots: tuple[float, ...] | None = _spline_knots(knots)
        self.boundary: tuple[float, float] | None = None
        if boundary is not None:
            self.boundary = _spline_boundary(boundary, self.knots)
        self.parameter_names = ["level", "rate"] + [
            f"s{k + 1}" for k in range(len(self.knots))
        ]

    @classmethod
    def natural(cls, df: int = 3) -> SplinePath:
        """
        A spline whose knots are placed from the data it is fitted to.

        The boundary knots are the first and last measurement times, and
        the ``df - 1`` interior knots are at equally spaced quantiles of
        all the units' measurement times (as R's ``splines::ns(t, df)``
        places them). A fitter (``DegradationAnalysis.fit``,
        ``JointPH(...).fit``) places them, and the fitted model's
        ``path_model`` records them; :meth:`placed` does it directly.

        Parameters
        ----------
        df : int, optional
            How many parameters shape the curve beside its ``level``:
            ``rate`` and ``df - 1`` bends, ``s1`` to ``s{df - 1}``.
            Default 3 (two interior knots); 1 is a straight line.

        Returns
        -------
        SplinePath
            A spline with ``df + 1`` parameters whose knots are not
            placed yet.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.degradation import SplinePath
        >>> path = SplinePath.natural(df=3)
        >>> path.parameter_names
        ['level', 'rate', 's1', 's2']
        >>> placed = path.placed(np.arange(0.0, 13.0))
        >>> placed.knots, placed.boundary
        ((4.0, 8.0), (0.0, 12.0))
        """
        if isinstance(df, bool) or not isinstance(df, (int, np.integer)):
            raise ValueError(
                "df is how many parameters shape the spline beside its "
                f"level, a whole number of 1 or more; got {df!r}."
            )
        if df < 1:
            raise ValueError(
                "df is how many parameters shape the spline beside its "
                f"level, 1 (a straight line) or more; got {df}."
            )
        out = cls()
        out.df = int(df)
        out.knots = None
        out.parameter_names = ["level", "rate"] + [
            f"s{k + 1}" for k in range(int(df) - 1)
        ]
        return out

    @property
    def is_placed(self) -> bool:
        """Whether the knots and the boundary are set (else a fitter
        places them from its data, see :meth:`placed`)."""
        return self.knots is not None and self.boundary is not None

    def placed(self, x: npt.ArrayLike) -> SplinePath:
        """
        This spline with its knots placed from the measurement times
        ``x`` (every unit's, together): itself where they are set
        already.

        A boundary left out is the first and last of ``x``; the interior
        knots of :meth:`natural` are the quantiles ``1/df``, ...,
        ``(df - 1)/df`` of ``x``.

        Examples
        --------
        >>> from surpyval.degradation import SplinePath
        >>> SplinePath(knots=[3.0]).placed([0.0, 1.0, 2.0, 5.0]).boundary
        (0.0, 5.0)
        """
        if self.is_placed:
            return self
        times = np.asarray(x, dtype=float).ravel()
        seen = times[np.isfinite(times)]
        if seen.size == 0 or seen.min() == seen.max():
            raise ValueError(
                "The spline path's knots are placed from the measurement "
                "times, which must span more than one time; got "
                f"{np.unique(seen).size} distinct time(s). Give knots and "
                "boundary to SplinePath(knots=[...], boundary=(lo, hi))."
            )
        boundary = self.boundary
        if boundary is None:
            boundary = (float(seen.min()), float(seen.max()))
        knots = self.knots
        if knots is None:
            df = int(self.df or 1)
            inside = seen[(seen >= boundary[0]) & (seen <= boundary[1])]
            knots = tuple(
                float(v) for v in np.quantile(inside, np.arange(1, df) / df)
            )
            if len(set(knots)) < len(knots) or (
                knots and not boundary[0] < knots[0] <= knots[-1] < boundary[1]
            ):
                raise ValueError(
                    f"SplinePath.natural(df={df}) places {df - 1} interior "
                    "knot(s) at quantiles of the measurement times, and "
                    f"here they are {list(knots)} with the boundary "
                    f"{boundary}: not distinct and inside it, as the "
                    f"times hold {np.unique(inside).size} distinct "
                    "value(s) with many repeats. Use a smaller df, or "
                    "give the knots: SplinePath(knots=[...], "
                    "boundary=(lo, hi))."
                )
        return type(self)(knots=knots, boundary=boundary)

    def check_readings(self, counts: npt.ArrayLike) -> None:
        """Refuse a spline with more parameters than the units' readings
        support: ``counts`` is each unit's number of readings, and their
        median must reach the number of parameters."""
        counts = np.asarray(counts, dtype=float)
        counts = counts[counts > 0]
        n = len(self.parameter_names)
        median = float(np.median(counts)) if counts.size else 0.0
        if median < n:
            most = int(np.floor(median)) - 2
            fix = (
                f"use at most {most} interior knot(s) "
                f"(SplinePath.natural(df={most + 1}))"
                if most >= 1
                else "use LinearPath"
            )
            raise ValueError(
                f"The spline path has {n - 2} interior knot(s), so {n} "
                f"parameters, and the {counts.size} unit(s) with readings "
                f"have a median of {median:g} each: too few to tell the "
                f"curve's bends from noise. With these readings, {fix}."
            )

    @property
    def breaks(self) -> tuple[float, ...]:
        """The ages at which the path's pieces join (every knot, where
        its third derivative jumps): none for a straight line. An
        integral of a function of the path is taken piece by piece."""
        if not self.parameter_names[2:] or not self.is_placed:
            return ()
        lo, hi = self.boundary or (np.nan, np.nan)
        return (lo, *(self.knots or ()), hi)

    def _require_placed(self) -> tuple:
        if self.knots is None or self.boundary is None:
            what = (
                f"SplinePath.natural(df={self.df})"
                if self.knots is None
                else "A SplinePath with no boundary"
            )
            raise ValueError(
                f"{what} has no knots until a fit places them from the "
                "measurement times: fit it with DegradationAnalysis.fit("
                "..., path=...) or JointPH(Weibull, ...).fit(...) and use "
                "the fitted model's path_model, or place them with "
                ".placed(x)."
            )
        lo, hi = self.boundary
        span = hi - lo
        inner = (np.asarray(self.knots, dtype=float) - lo) / span
        return lo, span, inner

    def _powers(self, x: npt.NDArray, power: int) -> list:
        """Each basis function's piecewise polynomial of ``power`` in the
        scaled age (3: the function itself; 2: a third of its derivative;
        4: four times its antiderivative), per unit of the scaled age."""
        lo, span, inner = self._require_placed()
        u = (x - lo) / span
        last = np.maximum(u - 1.0, 0.0) ** power
        starts = np.concatenate([[0.0], inner])

        def d(start: float) -> npt.NDArray:
            return (np.maximum(u - start, 0.0) ** power - last) / (1.0 - start)

        final = d(starts[-1])
        return [d(start) - final for start in starts[:-1]]

    def basis(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The spline's basis at time(s) ``x``: one column for each
        parameter (1, ``x`` and each :math:`N_k(x)`), so the path is
        ``basis(x) @ params``.

        Examples
        --------
        >>> from surpyval.degradation import SplinePath
        >>> path = SplinePath(knots=[4.0], boundary=(0.0, 10.0))
        >>> path.basis([0.0, 5.0, 10.0]).round(4)
        array([[ 1.    ,  0.    ,  0.    ],
               [ 1.    ,  5.    ,  0.1233],
               [ 1.    , 10.    ,  0.64  ]])
        """
        x = np.asarray(x, dtype=float)
        if not self.parameter_names[2:]:
            return np.column_stack([np.ones_like(x), x])
        return _columns(x, 1.0, x, *self._powers(x, 3))

    def path(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        level, rate, *bends = params
        x = np.asarray(x, dtype=float)
        out = level + rate * x
        if bends:
            for s, column in zip(bends, self._powers(x, 3)):
                out = out + s * column
        return out

    def jacobian(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        return self.basis(x)

    def slope(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        level, rate, *bends = params
        x = np.asarray(x, dtype=float)
        out = rate + 0.0 * x
        if bends:
            span = self._require_placed()[1]
            for s, column in zip(bends, self._powers(x, 2)):
                out = out + s * (3.0 / span) * column
        return out

    def slope_jacobian(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        if not self.parameter_names[2:]:
            return _columns(x, 0.0, 1.0)
        span = self._require_placed()[1]
        return _columns(
            x, 0.0, 1.0, *[(3.0 / span) * c for c in self._powers(x, 2)]
        )

    def _areas(self, x: npt.NDArray) -> list:
        """The area under each :math:`N_k` from age 0 to ``x``."""
        span = self._require_placed()[1]
        at_zero = self._powers(np.zeros_like(x), 4)
        return [
            (span / 4.0) * (upper - lower)
            for upper, lower in zip(self._powers(x, 4), at_zero)
        ]

    def area(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        """
        The area under the path from age 0 to ``x``, in closed form (the
        path's pieces are cubics, so the area's are quartics).

        Examples
        --------
        >>> from surpyval.degradation import SplinePath
        >>> path = SplinePath(knots=[4.0], boundary=(0.0, 10.0))
        >>> path.area([4.0, 10.0], 2.0, 0.0, 5.0).round(4)
        array([ 8.32, 29.8 ])
        """
        level, rate, *bends = params
        x = np.asarray(x, dtype=float)
        out = level * x + rate * x**2 / 2.0
        if bends:
            for s, column in zip(bends, self._areas(x)):
                out = out + s * column
        return out

    def area_jacobian(self, x: npt.ArrayLike, *params: Any) -> npt.NDArray:
        """Partial derivatives of :meth:`area` with respect to the
        parameters, at time(s) ``x``: a ``(len(x), n_params)`` matrix."""
        x = np.asarray(x, dtype=float)
        bends = self._areas(x) if self.parameter_names[2:] else []
        return _columns(x, x, x**2 / 2.0, *bends)

    def inv_path(self, y: npt.ArrayLike, *params: Any) -> npt.NDArray:
        """
        The first time after 0 at which the path reaches level ``y``
        (``nan`` when it never does); with no interior knots, the
        straight line's crossing, as ``LinearPath`` gives it.

        The path is looked at on a grid of 32 points between neighbouring
        knots (from age 0, or the lower boundary knot if that is
        earlier), and the first crossing found is refined by bisection;
        after the upper boundary knot the path is a straight line, whose
        crossing is exact. A path that touches the level and turns back
        within one step of the grid is not seen.

        Examples
        --------
        >>> from surpyval.degradation import SplinePath
        >>> path = SplinePath(knots=[4.0], boundary=(0.0, 10.0))
        >>> float(path.inv_path(3.0, 2.0, 0.0, 5.0).round(4))
        5.9696
        >>> float(path.inv_path(1.0, 2.0, 0.0, 5.0))
        nan
        """
        level, rate, *bends = params
        y = np.asarray(y, dtype=float)
        if not bends:
            with np.errstate(divide="ignore", invalid="ignore"):
                return (y - level) / rate
        lo, span, inner = self._require_placed()
        hi = lo + span
        edges = np.concatenate([[lo], lo + span * inner, [hi]])
        edges = np.unique(np.concatenate([[0.0], edges[edges > 0.0]]))
        steps = np.linspace(0.0, 1.0, _SPLINE_GRID + 1)[1:]
        grid = np.concatenate(
            [[edges[0]]]
            + [a + (b - a) * steps for a, b in zip(edges[:-1], edges[1:])]
        )
        theta = [np.asarray(v, dtype=float) for v in (level, rate, *bends)]
        shape = np.broadcast_shapes(y.shape, *[v.shape for v in theta])
        target = np.broadcast_to(y, shape).ravel()
        flat = [np.broadcast_to(v, shape).ravel() for v in theta]
        spread = [v[:, None] for v in flat]
        with np.errstate(all="ignore"):
            gap = self.path(grid[None, :], *spread) - target[:, None]
            side = np.sign(gap)
            # the first step of the grid over which the path reaches y
            reached = (side[:, 1:] != side[:, :-1]) & (side[:, :-1] != 0)
            reached |= side[:, 1:] == 0
            found = reached.any(axis=1)
            first = np.argmax(reached, axis=1)
            left, right = grid[first], grid[first + 1]
            start = np.take_along_axis(side, first[:, None], axis=1)[:, 0]
            for _ in range(60):
                middle = (left + right) / 2.0
                same = np.sign(self.path(middle, *flat) - target) == start
                left = np.where(same, middle, left)
                right = np.where(same, right, middle)
            inside = (left + right) / 2.0
            # past the last knot the path is a straight line
            end = self.path(np.full(target.shape, hi), *flat)
            tilt = self.slope(np.full(target.shape, hi), *flat)
            beyond = hi + (target - end) / tilt
            beyond = np.where(
                np.isfinite(beyond) & (beyond > hi), beyond, np.nan
            )
        out = np.where(found, inside, beyond)
        out = np.where(np.isfinite(target), out, np.nan)
        for v in flat:
            out = np.where(np.isfinite(v), out, np.nan)
        return out.reshape(shape)

    @takes_unit
    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        """
        Fit the path parameters to one unit's measurements by linear
        least squares on the spline's basis (its knots as they are: a
        spline whose knots are not placed is refused, saying how to place
        them).

        Parameters
        ----------
        x : array_like
            The unit's measurement times.
        y : array_like
            Its degradation measurements.

        Returns
        -------
        numpy array
            The fitted ``level``, ``rate`` and ``s1``, ..., ``sK``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.degradation import SplinePath
        >>> path = SplinePath(knots=[4.0], boundary=(0.0, 10.0))
        >>> t = np.arange(0.0, 11.0)
        >>> path.fit(t, path.path(t, 2.0, 0.1, 5.0)).round(6)
        array([2. , 0.1, 5. ])
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        if not self.parameter_names[2:]:
            return np.array(_ols(x, y))
        params, *_ = np.linalg.lstsq(self.basis(x), y, rcond=None)
        return params

    @property
    def formula(self) -> str:
        """The path's formula in words, for a model's printout."""
        bends = self.parameter_names[2:]
        if not bends:
            return "level + rate t"
        terms = " + ".join(f"{s} N{k + 1}(t)" for k, s in enumerate(bends))
        if not self.is_placed:
            return f"level + rate t + {terms} (natural cubic spline)"
        knots = ", ".join(f"{k:g}" for k in self.knots or ())
        lo, hi = self.boundary or (np.nan, np.nan)
        return (
            f"level + rate t + {terms} (natural cubic spline, knots at "
            f"{knots} between {lo:g} and {hi:g})"
        )

    def to_dict(self) -> dict:
        """
        The spline as a dictionary of plain values, ``{"name": "Spline",
        "knots": [...], "boundary": [lo, hi]}``, from which
        ``get_path_model`` (and so a saved model's ``from_dict``) builds
        it again. A spline whose knots are not placed is refused.

        Examples
        --------
        >>> from surpyval.degradation import SplinePath, get_path_model
        >>> saved = SplinePath(knots=[4.0], boundary=(0.0, 10.0)).to_dict()
        >>> saved
        {'name': 'Spline', 'knots': [4.0], 'boundary': [0.0, 10.0]}
        >>> get_path_model(saved).knots
        (4.0,)
        """
        self._require_placed()
        return {
            "name": self.name,
            "knots": [float(k) for k in self.knots or ()],
            "boundary": [float(b) for b in self.boundary or ()],
        }

    def __repr__(self) -> str:
        if self.knots is None:
            return f"Spline Degradation Path Model (natural, df={self.df})"
        where = ", ".join(f"{k:g}" for k in self.knots) or "none"
        if self.boundary is None:
            return f"Spline Degradation Path Model (knots: {where})"
        lo, hi = self.boundary
        return (
            f"Spline Degradation Path Model (knots: {where}; boundary "
            f"{lo:g} to {hi:g})"
        )


def _spline_knots(knots: Any) -> tuple[float, ...]:
    """``knots`` as an ascending tuple of finite floats, or refused."""
    try:
        values = np.asarray(knots, dtype=float).ravel()
    except (TypeError, ValueError):
        values = np.array([np.nan])
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(
            "knots are the spline's interior knots: finite ages, "
            f"ascending, each once; got {knots!r}."
        )
    return tuple(float(v) for v in values)


def _spline_boundary(
    boundary: Any, knots: tuple[float, ...]
) -> tuple[float, float]:
    """``boundary`` as ``(lo, hi)`` around ``knots``, or refused."""
    try:
        lo, hi = (float(v) for v in boundary)
    except (TypeError, ValueError):
        lo = hi = np.nan
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(
            "boundary is the spline's boundary knots (lo, hi), two finite "
            f"ages with lo < hi; got {boundary!r}."
        )
    if knots and not (lo < knots[0] and knots[-1] < hi):
        raise ValueError(
            f"The spline's interior knots {list(knots)} must lie strictly "
            f"between its boundary knots {(lo, hi)}: move the boundary "
            "out, or drop the knots outside it."
        )
    return lo, hi


LinearPath: PathModel = LinearPath_()
QuadraticPath: PathModel = QuadraticPath_()
ExponentialPath: PathModel = ExponentialPath_()
OffsetExponentialPath: PathModel = OffsetExponentialPath_()
PowerPath: PathModel = PowerPath_()
LogarithmicPath: PathModel = LogarithmicPath_()
LloydLipowPath: PathModel = LloydLipowPath_()
GompertzPath: PathModel = GompertzPath_()
MichaelisMentenPath: PathModel = MichaelisMentenPath_()

PATH_MODELS: dict[str, PathModel] = {
    "linear": LinearPath,
    "quadratic": QuadraticPath,
    "exponential": ExponentialPath,
    "offset-exponential": OffsetExponentialPath,
    "power": PowerPath,
    "logarithmic": LogarithmicPath,
    "lloyd-lipow": LloydLipowPath,
    "gompertz": GompertzPath,
    "michaelis-menten": MichaelisMentenPath,
}

# Display name (``PathModel.name``, lower-cased) -> registry key. The
# display name is not always the key ("Offset Exponential" vs
# "offset-exponential"), and serialised models written before the key
# was stored carry the display name, so both must resolve.
_KEY_BY_DISPLAY_NAME: dict[str, str] = {
    model.name.lower(): key for key, model in PATH_MODELS.items()
}


def path_model_key(path_model: PathModel) -> Any:
    """
    The name under which ``path_model`` can be resolved again.

    For a built-in path model this is its ``PATH_MODELS`` key (so
    ``get_path_model(path_model_key(m))`` returns the built-in model);
    for a :class:`SplinePath`, which is made with its knots, its
    ``to_dict()`` (``get_path_model`` builds an equal one from it); a
    custom :class:`PathModel` is not registered, so its ``name`` is
    returned as the best available label.
    """
    if isinstance(path_model, SplinePath):
        return path_model.to_dict()
    for key, model in PATH_MODELS.items():
        if type(model) is type(path_model):
            return key
    return path_model.name


def _spline_from_dict(saved: dict) -> SplinePath:
    """The :class:`SplinePath` of its ``to_dict()``, or a refusal saying
    what the dictionary lacks."""
    if saved.get("name") != SplinePath.name:
        raise ValueError(
            "A path model given as a dictionary is a spline's, "
            "{'name': 'Spline', 'knots': [...], 'boundary': [lo, hi]} "
            f"(SplinePath.to_dict()); got the name {saved.get('name')!r}."
        )
    missing = [key for key in ("knots", "boundary") if key not in saved]
    if missing:
        raise ValueError(
            f"The spline path's dictionary has no {missing}: it holds "
            "'name', 'knots' and 'boundary' (SplinePath.to_dict())."
        )
    return SplinePath(knots=saved["knots"], boundary=saved["boundary"])


def get_path_model(path: str | dict | PathModel) -> PathModel:
    """
    Resolve ``path`` to a :class:`PathModel` instance.

    Accepts a :class:`PathModel` instance (returned unchanged) or one
    of the registered names in ``PATH_MODELS`` (case-insensitive):
    ``"linear"``, ``"quadratic"``, ``"exponential"``,
    ``"offset-exponential"``, ``"power"``, ``"logarithmic"``,
    ``"lloyd-lipow"``, ``"gompertz"``, ``"michaelis-menten"``. A built-in
    model's display ``name`` (e.g. ``"Offset Exponential"``) is accepted
    too, and so is a :class:`SplinePath`'s ``to_dict()``, which a saved
    model holds. (``"best"`` — automatic selection — is handled by
    ``DegradationAnalysis.fit``, not here.)

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.degradation import get_path_model
    >>> power = get_path_model("power")
    >>> power.name, power.parameter_names
    ('Power', ['a', 'b'])
    >>> power.path(np.array([1.0, 4.0]), 2.0, 0.5)
    array([2., 4.])
    >>> get_path_model("Offset Exponential").name
    'Offset Exponential'
    """
    if isinstance(path, PathModel):
        return path
    if isinstance(path, dict):
        return _spline_from_dict(path)
    if isinstance(path, str):
        key = path.lower()
        key = _KEY_BY_DISPLAY_NAME.get(key, key)
        if key in PATH_MODELS:
            return PATH_MODELS[key]
        raise option_error(
            "path", path, sorted(PATH_MODELS), "A PathModel is accepted too."
        )
    raise ValueError(
        "path must be a string, a PathModel instance or a SplinePath's "
        "to_dict(), got {}".format(type(path))
    )
