"""A fitted model's time on another clock: ``rescale_time(factor)``.

The model of ``R(factor t)``: the same life with its time divided by
``factor``. A unit that operates a fraction ``factor`` of the calendar
ages only then, so its life on the calendar is its operating life at
``factor t``; a life fitted in hours is, in days, its life at ``24 t``.

Each distribution says how its parameters move with its time, one role per
parameter (``ParametricFitter._time_roles``):

- ``"time"``: a time (a scale, a location, an end of the support), divided
  by ``factor``;
- ``"rate"``: a rate (one over a time), multiplied by it;
- ``"log_time"``: the log of a time, less ``log(factor)``;
- ``"shape"``: a number without a unit, kept.

An offset ``gamma`` is a time; a limited-failure proportion and a
zero-inflation fraction are probabilities, kept. As every role is an
affine map of its own parameter, the covariance moves by the delta method
with a diagonal Jacobian, exactly.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import numpy.typing as npt

#: The roles a parameter can have under a change of the time's scale.
TIME_ROLES = ("time", "rate", "log_time", "shape")


def checked_factor(factor: Any) -> float:
    """``factor`` as a float: a finite number above 0, else a
    ``ValueError`` that says what it must be."""
    if (
        isinstance(factor, (bool, np.bool_))
        or not isinstance(factor, numbers.Real)
        or not np.isfinite(float(factor))
        or float(factor) <= 0
    ):
        raise ValueError(
            "'factor' must be a finite number above 0: the time on the "
            "model's clock for each unit of time on the new one (0.25 for "
            "a unit that operates a quarter of the calendar, 24 to go "
            f"from hours to days); got {factor!r}"
        )
    return float(factor)


def time_roles(dist: Any) -> tuple[str, ...]:
    """The role of each of ``dist``'s parameters, or the ``ValueError``
    that says why its time cannot be rescaled and what to do instead."""
    from .parametric import is_custom_distribution

    name = getattr(dist, "name", type(dist).__name__)
    if is_custom_distribution(dist):
        raise ValueError(
            f"rescale_time cannot move the parameters of the custom "
            f"distribution {name!r}: it does not say which of them are "
            "times. Define the distribution on the new clock (its "
            "functions of t replaced by those of factor * t) and build "
            "the model from it, or refit it to the times in the new unit."
        )
    if getattr(dist, "discrete", False):
        raise ValueError(
            f"rescale_time cannot rescale the {name} distribution: it is "
            "a distribution of counts, on the whole numbers, and a count "
            "has no unit of time to change. A count is the same on every "
            "clock; to model the time to the event, fit a distribution of "
            "time (Weibull, Gamma, LogNormal, ...)."
        )
    roles = dist._time_roles
    if roles is None:
        why = getattr(dist, "_time_roles_refusal", None) or (
            "it does not say which of its parameters are times; refit it "
            "to the times in the new unit"
        )
        raise ValueError(
            f"rescale_time cannot rescale the {name} distribution: {why}."
        )
    roles = tuple(roles)
    if len(roles) != len(dist.parameter_names) or not set(roles) <= set(
        TIME_ROLES
    ):
        raise AssertionError(
            f"{name}._time_roles must give one of {TIME_ROLES} for each "
            f"of its parameters {list(dist.parameter_names)}; got {roles}"
        )
    return roles


def moved(
    params: npt.ArrayLike, roles: tuple[str, ...], factor: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """The parameters on the new clock, and the derivative of each with
    respect to its old value (the diagonal of the Jacobian)."""
    values = np.asarray(params, dtype=float)
    slope = np.array(
        [{"time": 1 / factor, "rate": factor}.get(r, 1.0) for r in roles]
    )
    shift = np.array(
        [-np.log(factor) if r == "log_time" else 0.0 for r in roles]
    )
    return values * slope + shift, slope


def carried(matrix: Any, slope: npt.NDArray) -> npt.NDArray:
    """``J matrix J'`` for the diagonal Jacobian ``J`` whose leading
    entries are ``slope`` and the rest 1: a covariance moved to the new
    parameters by the delta method (exact for these affine maps)."""
    matrix = np.asarray(matrix, dtype=float)
    diag = np.ones(matrix.shape[0])
    diag[: len(slope)] = slope
    return matrix * np.outer(diag, diag)


def density_count(
    x: Any, c: Any, n: Any, w: Any = None, zero_mass: bool = False
) -> float:
    """The (weighted) number of units whose likelihood is a density: the
    exact observations, less those at 0 that a zero-inflated model gives
    its mass ``f0``. Each such density is ``factor`` times larger on the
    new clock, so the log-likelihood rises by this times
    ``log(factor)``; every other contribution is a probability, which
    does not change."""
    c = np.asarray(c)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    rows = c == 0
    if zero_mass:
        rows = rows & (x != 0)
    counts = np.asarray(n, dtype=float)
    if w is not None:
        counts = counts * np.asarray(w, dtype=float)
    return float(np.sum(counts[rows]))


#: Why a rescaled model has no log-likelihood: it was rescaled without
#: the data that moving its likelihood to the new clock needs.
NO_LIKELIHOOD = (
    "this model was rescaled with rescale_time from a model without its "
    "data (built from parameters, or restored from a dict saved without "
    "them), and moving a log-likelihood to a new time unit needs the "
    "exact observations it counts. Rescale the fitted model itself, or "
    "one saved with to_dict(with_data=True), or refit to the times in "
    "the new unit"
)
