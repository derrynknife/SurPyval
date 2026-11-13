"""The link functions of a distributional regression.

Each parameter of the distribution has a linear predictor ``eta`` and a
link ``g`` with ``g(theta) = eta``. A model keeps each parameter's value
at the reference row (``Z = 0``, or the covariate means) on its natural
scale and moves it there by the row's linear predictor,
``theta(z) = g^-1(g(theta_0) + s b'z)``, which each link computes in the
form that stays accurate (``theta_0 * exp(s b'z)`` for the log link).

The sign ``s`` of each parameter (:func:`sign`) is SurPyval's sign
convention, the one its AFT and PH models have: a positive coefficient
makes lives shorter. A parameter whose increase makes every quantile
later (a scale, a location, a Gamma's or an ExpoWeibull's shape that
orders lives) has ``s = -1``; one whose increase makes every quantile
earlier (a rate) has ``s = +1``; a parameter that makes some quantiles
earlier and others later (a Weibull's shape, a spread ``sigma``) has no
direction of its own, and keeps the link's sign, ``s = +1``.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as np
from autograd.scipy.special import erfc as _erfc
from autograd.scipy.special import erfcinv as _erfcinv
from autograd.scipy.special import expit as _expit

_SQRT2 = 2.0**0.5


def _ndtr(x: Any) -> Any:
    """The standard normal distribution function."""
    return 0.5 * _erfc(-x / _SQRT2)


def _ndtri(p: Any) -> Any:
    """The standard normal quantile function."""
    return -_SQRT2 * _erfcinv(2.0 * p)


#: The links by name, each with the parameter ranges it maps onto the
#: whole line (its inverse stays inside them).
LINKS = ("identity", "log", "logit", "probit", "cloglog")

#: The links of a probability, a parameter in (0, 1).
PROBABILITY_LINKS = ("logit", "probit", "cloglog")

#: The parameters whose increase makes every quantile of the life later
#: (``"longer"``) or earlier (``"shorter"``), by distribution. A
#: parameter not listed moves some quantiles each way (the quantiles of
#: two values cross), so it has no direction. Checked against each
#: distribution's quantile function in the tests.
EFFECT: dict[str, dict[str, str]] = {
    "Exponential": {"failure_rate": "shorter"},
    "Rayleigh": {"sigma": "longer"},
    "Weibull": {"alpha": "longer"},
    "Gamma": {"alpha": "longer", "beta": "shorter"},
    "LogNormal": {"mu": "longer"},
    "LogLogistic": {"alpha": "longer"},
    "Normal": {"mu": "longer"},
    "Gumbel": {"mu": "longer"},
    "GumbelLEV": {"mu": "longer"},
    "Logistic": {"mu": "longer"},
    "Beta": {"alpha": "longer", "beta": "shorter"},
    "ExpoWeibull": {"alpha": "longer", "mu": "longer"},
}


def sign(distribution: str, parameter: str) -> float:
    """The sign ``s`` with which the coefficients of ``parameter`` enter
    its linear predictor, ``g(theta(z)) = g(theta_0) + s b'z``: -1 where
    raising the parameter makes lives longer, so that a positive
    coefficient makes them shorter (SurPyval's AFT sign), else +1.

    >>> sign("Weibull", "alpha"), sign("Weibull", "beta")
    (-1.0, 1.0)
    >>> sign("Exponential", "failure_rate")
    1.0
    """
    effect = EFFECT.get(distribution, {}).get(parameter)
    return -1.0 if effect == "longer" else 1.0


def default_link(bounds: tuple) -> str:
    """The default link of a parameter with ``bounds`` ``(lower,
    upper)``: identity for one on the whole line, log for a positive
    one. Any other range has no link here, which the caller refuses."""
    lower, upper = bounds
    if lower is None and upper is None:
        return "identity"
    if lower == 0 and upper is None:
        return "log"
    if lower == 0 and upper == 1:
        return "logit"
    raise ValueError(
        "A parameter with bounds {} has no link in a distributional "
        "regression.".format(bounds)
    )


def allowed_links(bounds: tuple) -> tuple[str, ...]:
    """The links whose inverse stays inside ``bounds``: identity and log
    for a parameter on the whole line (log keeps it positive), only log
    for a positive one (an identity link could take it below 0, where
    the likelihood is not defined)."""
    default = default_link(bounds)
    if default == "identity":
        return ("identity", "log")
    if default == "logit":
        return PROBABILITY_LINKS
    return ("log",)


def link(name: str, value: Any) -> Any:
    """``g(value)``."""
    if name == "log":
        with np.errstate(divide="ignore"):
            return np.log(value)
    if name == "logit":
        with np.errstate(divide="ignore"):
            return np.log(value) - np.log1p(-value)
    if name == "probit":
        return _ndtri(value)
    if name == "cloglog":
        with np.errstate(divide="ignore"):
            return np.log(-np.log1p(-value))
    return value


def inverse(name: str, eta: Any) -> Any:
    """``g^-1(eta)``."""
    if name == "log":
        with np.errstate(over="ignore"):
            return np.exp(eta)
    if name == "logit":
        return _expit(eta)
    if name == "probit":
        return _ndtr(eta)
    if name == "cloglog":
        with np.errstate(over="ignore"):
            return -np.expm1(-np.exp(eta))
    return eta


def move(name: str, value: Any, shift: Any) -> Any:
    """``g^-1(g(value) + shift)``: the parameter ``value`` at a row whose
    linear predictor is ``shift`` more than at the reference row."""
    if name == "log":
        # A coefficient running off makes the shift large; exp overflows
        # to inf, the limit, without numpy's raw warning (principle 22).
        with np.errstate(over="ignore", invalid="ignore"):
            return value * np.exp(shift)
    if name in PROBABILITY_LINKS:
        return inverse(name, link(name, value) + shift)
    return value + shift
