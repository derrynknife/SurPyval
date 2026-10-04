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

from surpyval.utils.removed_names import RemovedNames
from surpyval.utils.validation import option_error


class _StopSearch(Exception):
    """Raised by a path a least-squares search evaluates, to end that
    search (see ``OffsetExponentialPath_.fit``)."""


def _ols(z: npt.NDArray, y: npt.NDArray) -> tuple[float, float]:
    """Closed-form least squares fit of ``y = intercept + slope * z``."""
    A = np.column_stack([np.ones_like(z), z])
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    return intercept, slope


class PathModel(ABC, RemovedNames):
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
    ) -> "npt.ArrayLike | list":
        """One starting parameter vector, or a list of candidates."""
        raise NotImplementedError

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
                "Could not fit the {} path model to the data".format(self.name)
            )
        return best_params

    def _least_squares(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        guesses: npt.NDArray,
        path: Any = None,
        stopped: "list | None" = None,
    ) -> "tuple[npt.NDArray | None, float]":
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
        return "{} Degradation Path Model".format(self.name)


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

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if (y <= 0).any():
            raise ValueError(
                "The exponential path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> "npt.ArrayLike | list":
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
    ) -> "npt.ArrayLike | list":
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

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(
            x, "y = a + b * ln(t) is undefined at t = 0"
        )

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

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self._require_positive_times(x, "y = a - b / t is undefined at t = 0")

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

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if (y <= 0).any():
            raise ValueError(
                "The Gompertz path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> "npt.ArrayLike | list":
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

    def _initial_guess(
        self, x: npt.NDArray, y: npt.NDArray
    ) -> "npt.ArrayLike | list":
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
                "Could not fit the {} path model to the data".format(self.name)
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
    ) -> "npt.ArrayLike | list":
        # Lineweaver-Burk linearisation: 1/y = 1/a + (b/a) / x
        intercept, slope = _ols(1.0 / x, 1.0 / y)
        if intercept > 0:
            return [1.0 / intercept, slope / intercept]
        return [1.2 * y.max(), float(np.median(x))]


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


def path_model_key(path_model: PathModel) -> str:
    """
    The name under which ``path_model`` can be resolved again.

    For a built-in path model this is its ``PATH_MODELS`` key (so
    ``get_path_model(path_model_key(m))`` returns the built-in model);
    a custom :class:`PathModel` is not registered, so its ``name`` is
    returned as the best available label.
    """
    for key, model in PATH_MODELS.items():
        if type(model) is type(path_model):
            return key
    return path_model.name


def get_path_model(path: "str | PathModel") -> PathModel:
    """
    Resolve ``path`` to a :class:`PathModel` instance.

    Accepts a :class:`PathModel` instance (returned unchanged) or one
    of the registered names in ``PATH_MODELS`` (case-insensitive):
    ``"linear"``, ``"quadratic"``, ``"exponential"``,
    ``"offset-exponential"``, ``"power"``, ``"logarithmic"``,
    ``"lloyd-lipow"``, ``"gompertz"``, ``"michaelis-menten"``. A built-in
    model's display ``name`` (e.g. ``"Offset Exponential"``) is accepted
    too. (``"best"`` — automatic selection — is handled by
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
    if isinstance(path, str):
        key = path.lower()
        key = _KEY_BY_DISPLAY_NAME.get(key, key)
        if key in PATH_MODELS:
            return PATH_MODELS[key]
        raise option_error(
            "path", path, sorted(PATH_MODELS), "A PathModel is accepted too."
        )
    raise ValueError(
        "path must be a string or a PathModel instance, got {}".format(
            type(path)
        )
    )
