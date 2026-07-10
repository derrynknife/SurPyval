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
  one unit's measurements.

Models that are linear in their parameters (``LinearPath``,
``LogarithmicPath``, ``LloydLipowPath``) are fitted with ordinary least
squares in closed form; the others (``ExponentialPath``, ``PowerPath``)
use nonlinear least squares started from the log-linearised fit.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


def _ols(z, y):
    """Closed-form least squares fit of ``y = intercept + slope * z``."""
    A = np.column_stack([np.ones_like(z), z])
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    return intercept, slope


class PathModel(ABC):
    """
    Base class for degradation path models.

    A path model is a deterministic function of time with a small
    number of parameters that is fitted, per unit, to that unit's
    degradation measurements. Subclass this (implementing ``path``,
    ``inv_path``, ``fit`` and the ``name``/``param_names`` attributes)
    to use a custom degradation path with
    :class:`~surpyval.degradation.DegradationAnalysis`.
    """

    name: str
    param_names: list[str]
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

    def _initial_guess(self, x, y):
        """One starting parameter vector, or a list of candidates."""
        raise NotImplementedError

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        """
        Fit the path parameters to one unit's measurements by
        (nonlinear) least squares.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        guesses = np.atleast_2d(
            np.asarray(self._initial_guess(x, y), dtype=float)
        )
        best_params, best_rss = None, np.inf
        for p0 in guesses:
            try:
                params, _ = curve_fit(self.path, x, y, p0=p0, maxfev=10_000)
            except RuntimeError:
                continue
            residuals = y - self.path(x, *params)
            rss = float(residuals @ residuals)
            if np.isfinite(rss) and rss < best_rss:
                best_params, best_rss = params, rss
        if best_params is None:
            raise ValueError(
                "Could not fit the {} path model to the data".format(self.name)
            )
        return best_params

    def __repr__(self):
        return "{} Degradation Path Model".format(self.name)


class LinearPath_(PathModel):
    """Linear degradation path: ``y = a + b * x``."""

    name = "Linear"
    param_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x, *params):
        a, b = params
        return a + b * np.asarray(x, dtype=float)

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (y - a) / b

    def jacobian(self, x, *params):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), x])

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        return np.array(_ols(x, y))


class ExponentialPath_(PathModel):
    """Exponential degradation path: ``y = a * exp(b * x)``."""

    name = "Exponential"
    param_names = ["a", "b"]

    def path(self, x, *params):
        a, b = params
        return a * np.exp(b * np.asarray(x, dtype=float))

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(y / a) / b

    def jacobian(self, x, *params):
        a, b = params
        x = np.asarray(x, dtype=float)
        e = np.exp(b * x)
        return np.column_stack([e, a * x * e])

    def check_data(self, x, y):
        if (y <= 0).any():
            raise ValueError(
                "The exponential path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(self, x, y):
        intercept, slope = _ols(x, np.log(y))
        return np.exp(intercept), slope


class PowerPath_(PathModel):
    """Power degradation path: ``y = a * x**b``."""

    name = "Power"
    param_names = ["a", "b"]

    def path(self, x, *params):
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * np.power(np.asarray(x, dtype=float), b)

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.power(y / a, 1.0 / b)

    def jacobian(self, x, *params):
        a, b = params
        x = np.asarray(x, dtype=float)
        xb = np.power(x, b)
        return np.column_stack([xb, a * xb * np.log(x)])

    def check_data(self, x, y):
        if (x <= 0).any():
            raise ValueError(
                "The power path model requires strictly positive times"
            )
        if (y <= 0).any():
            raise ValueError(
                "The power path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(self, x, y):
        intercept, slope = _ols(np.log(x), np.log(y))
        return np.exp(intercept), slope


class LogarithmicPath_(PathModel):
    """Logarithmic degradation path: ``y = a + b * ln(x)``."""

    name = "Logarithmic"
    param_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x, *params):
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a + b * np.log(np.asarray(x, dtype=float))

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.exp((y - a) / b)

    def jacobian(self, x, *params):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), np.log(x)])

    def check_data(self, x, y):
        if (x <= 0).any():
            raise ValueError(
                "The logarithmic path model requires strictly positive times"
            )

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        return np.array(_ols(np.log(x), y))


class LloydLipowPath_(PathModel):
    """Lloyd-Lipow degradation path: ``y = a - b / x``."""

    name = "Lloyd-Lipow"
    param_names = ["a", "b"]
    linear_in_parameters = True

    def path(self, x, *params):
        a, b = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return a - b / np.asarray(x, dtype=float)

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b / (a - y)

    def jacobian(self, x, *params):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), -1.0 / x])

    def check_data(self, x, y):
        if (x <= 0).any():
            raise ValueError(
                "The Lloyd-Lipow path model requires strictly positive times"
            )

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.check_data(x, y)
        intercept, slope = _ols(1.0 / x, y)
        return np.array([intercept, -slope])


class QuadraticPath_(PathModel):
    """Quadratic degradation path: ``y = a + b * x + c * x**2``."""

    name = "Quadratic"
    param_names = ["a", "b", "c"]
    linear_in_parameters = True

    def path(self, x, *params):
        a, b, c = params
        x = np.asarray(x, dtype=float)
        return a + b * x + c * x**2

    def inv_path(self, y, *params):
        """First positive time at which the parabola reaches ``y``."""
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            linear_root = (y - a) / b
            disc = b**2 - 4.0 * c * (a - y)
            sqrt_disc = np.sqrt(np.where(disc >= 0, disc, np.nan))
            root_minus = (-b - sqrt_disc) / (2.0 * c)
            root_plus = (-b + sqrt_disc) / (2.0 * c)
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

    def jacobian(self, x, *params):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), x, x**2])

    def fit(self, x, y):
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
    param_names = ["a", "b", "c"]

    def path(self, x, *params):
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            return a * np.exp(-b * np.exp(-c * x))

    def inv_path(self, y, *params):
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log(-np.log(y / a) / b) / c

    def jacobian(self, x, *params):
        a, b, c = params
        x = np.asarray(x, dtype=float)
        inner = np.exp(-c * x)
        outer = np.exp(-b * inner)
        return np.column_stack(
            [outer, -a * inner * outer, a * b * x * inner * outer]
        )

    def check_data(self, x, y):
        if (y <= 0).any():
            raise ValueError(
                "The Gompertz path model requires strictly positive "
                "degradation measurements"
            )

    def _initial_guess(self, x, y):
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
    param_names = ["a", "b", "c"]

    def path(self, x, *params):
        a, b, c = params
        x = np.asarray(x, dtype=float)
        with np.errstate(over="ignore"):
            return a + b * np.exp(c * x)

    def inv_path(self, y, *params):
        a, b, c = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((y - a) / b) / c

    def jacobian(self, x, *params):
        a, b, c = params
        x = np.asarray(x, dtype=float)
        e = np.exp(c * x)
        return np.column_stack([np.ones_like(x), e, b * x * e])

    def _initial_guess(self, x, y):
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


class MichaelisMentenPath_(PathModel):
    """Michaelis-Menten degradation path: ``y = a * x / (b + x)``.

    A saturating path rising from zero toward the asymptote ``a``,
    reaching half of it at ``x = b``.
    """

    name = "Michaelis-Menten"
    param_names = ["a", "b"]

    def path(self, x, *params):
        a, b = params
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * x / (b + x)

    def inv_path(self, y, *params):
        a, b = params
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return b * y / (a - y)

    def jacobian(self, x, *params):
        a, b = params
        x = np.asarray(x, dtype=float)
        denominator = b + x
        return np.column_stack([x / denominator, -a * x / denominator**2])

    def check_data(self, x, y):
        if (x <= 0).any():
            raise ValueError(
                "The Michaelis-Menten path model requires strictly "
                "positive times"
            )
        if (y <= 0).any():
            raise ValueError(
                "The Michaelis-Menten path model requires strictly "
                "positive degradation measurements"
            )

    def _initial_guess(self, x, y):
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


def get_path_model(path: "str | PathModel") -> PathModel:
    """
    Resolve ``path`` to a :class:`PathModel` instance.

    Accepts a :class:`PathModel` instance (returned unchanged) or one
    of the registered names in ``PATH_MODELS`` (case-insensitive):
    ``"linear"``, ``"quadratic"``, ``"exponential"``,
    ``"offset-exponential"``, ``"power"``, ``"logarithmic"``,
    ``"lloyd-lipow"``, ``"gompertz"``, ``"michaelis-menten"``.
    (``"best"`` — automatic selection — is handled by
    ``DegradationAnalysis.fit``, not here.)
    """
    if isinstance(path, PathModel):
        return path
    if isinstance(path, str):
        key = path.lower()
        if key in PATH_MODELS:
            return PATH_MODELS[key]
        raise ValueError(
            "Unknown path model '{}'; must be one of {} or a PathModel "
            "instance".format(path, sorted(PATH_MODELS))
        )
    raise ValueError(
        "path must be a string or a PathModel instance, got {}".format(
            type(path)
        )
    )
