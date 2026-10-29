"""The link functions of a distributional regression.

Each parameter of the distribution has a linear predictor ``eta`` and a
link ``g`` with ``g(theta) = eta``. A model keeps each parameter's value
at the reference row (``Z = 0``, or the covariate means) on its natural
scale and moves it there by the row's linear predictor,
``theta(z) = g^-1(g(theta_0) + b'z)``, which each link computes in the
form that stays accurate (``theta_0 * exp(b'z)`` for the log link).
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np

#: The links by name, each with the parameter ranges it maps onto the
#: whole line (its inverse stays inside them).
LINKS = ("identity", "log")


def default_link(bounds: tuple) -> str:
    """The default link of a parameter with ``bounds`` ``(lower,
    upper)``: identity for one on the whole line, log for a positive
    one. Any other range has no link here, which the caller refuses."""
    lower, upper = bounds
    if lower is None and upper is None:
        return "identity"
    if lower == 0 and upper is None:
        return "log"
    raise ValueError(
        "A parameter with bounds {} has no link in a distributional "
        "regression.".format(bounds)
    )


def allowed_links(bounds: tuple) -> tuple[str, ...]:
    """The links whose inverse stays inside ``bounds``: identity and log
    for a parameter on the whole line (log keeps it positive), only log
    for a positive one (an identity link could take it below 0, where
    the likelihood is not defined)."""
    if default_link(bounds) == "identity":
        return ("identity", "log")
    return ("log",)


def link(name: str, value: Any) -> Any:
    """``g(value)``."""
    if name == "log":
        with np.errstate(divide="ignore"):
            return np.log(value)
    return value


def inverse(name: str, eta: Any) -> Any:
    """``g^-1(eta)``."""
    if name == "log":
        with np.errstate(over="ignore"):
            return np.exp(eta)
    return eta


def move(name: str, value: Any, shift: Any) -> Any:
    """``g^-1(g(value) + shift)``: the parameter ``value`` at a row whose
    linear predictor is ``shift`` more than at the reference row."""
    if name == "log":
        # A coefficient running off makes the shift large; exp overflows
        # to inf, the limit, without numpy's raw warning (principle 22).
        with np.errstate(over="ignore", invalid="ignore"):
            return value * np.exp(shift)
    return value + shift
