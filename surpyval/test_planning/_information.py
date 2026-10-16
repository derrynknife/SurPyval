"""The expected Fisher information of a life test, for the log-location-
scale distributions the planning functions take (#863, #864).

A Weibull (``alpha``, ``beta``) or lognormal (``mu``, ``sigma``) lifetime
``T`` has ``log T = mu + sigma Z``, with ``Z`` the standard smallest
extreme value or normal variable: for the Weibull, ``mu = log alpha`` and
``sigma = 1 / beta``. A unit tested until time ``t_c``, censored there if
it has not failed, carries the information ``F(zeta) / sigma ** 2`` about
``(mu, sigma)``, with ``zeta = (log t_c - mu) / sigma`` and ``F`` the
2x2 matrix of Escobar and Meeker (1994), computed here by quadrature.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import special, stats

#: The families planned for, by the name of their SurPyval fitter: the
#: standard variable of ``log T``.
FAMILIES = {"Weibull": "sev", "LogNormal": "normal"}

#: Beyond this standardised time a unit is as good as uncensored: the
#: chance it outlives it is below 1e-300 (sev) or 1e-150 (normal).
_COMPLETE = {"sev": 6.5, "normal": 26.0}


def family_of(distribution: Any) -> str:
    """The standard family of a SurPyval fitter, or a ValueError naming
    those planned for."""
    name = getattr(distribution, "name", None)
    if name not in FAMILIES:
        raise ValueError(
            "distribution must be surpyval.Weibull or surpyval.LogNormal, "
            f"got {distribution!r}."
        )
    return FAMILIES[name]


def location_scale(distribution: Any, params: Any) -> tuple[str, float, float]:
    """The family, ``mu`` and ``sigma`` of ``log T`` for planning values
    ``params`` of a Weibull (``alpha``, ``beta``) or lognormal (``mu``,
    ``sigma``)."""
    family = family_of(distribution)
    values = np.asarray(params, dtype=float).ravel()
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"params must be the {distribution.name}'s two parameters "
            f"({', '.join(distribution.parameter_names)}), got {params!r}."
        )
    if family == "sev":
        alpha, beta = values
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"The Weibull's alpha and beta must be positive, got "
                f"{params!r}."
            )
        return family, float(np.log(alpha)), float(1.0 / beta)
    mu, sigma = values
    if sigma <= 0:
        raise ValueError(
            f"The LogNormal's sigma must be positive, got {params!r}."
        )
    return family, float(mu), float(sigma)


def cdf(family: str, z: npt.ArrayLike) -> npt.NDArray:
    """The standard variable's cdf."""
    z = np.asarray(z, dtype=float)
    if family == "sev":
        return -np.expm1(-np.exp(z))
    return stats.norm.cdf(z)


def ppf(family: str, p: npt.ArrayLike) -> npt.NDArray:
    """The standard variable's quantile."""
    p = np.asarray(p, dtype=float)
    if family == "sev":
        return np.log(-np.log1p(-p))
    return stats.norm.ppf(p)


#: Gauss-Legendre nodes and weights on [-1, 1] for the smallest extreme
#: value's integrals, whose integrand is smooth and negligible more than
#: 40 below its upper limit.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)
_SEV_SPAN = 45.0


def unit_information(family: str, zeta: float) -> npt.NDArray:
    """``F(zeta)``: ``sigma ** 2`` times the information about
    ``(mu, sigma)`` of one unit censored at the standardised time
    ``zeta`` (``inf`` for a unit run to failure).

    It is the expected outer product of the scores: of a failure,
    ``(h(z), z h(z) - 1)`` with ``h = -phi' / phi`` (``e^z - 1`` for the
    smallest extreme value, ``z`` for the normal), and of a unit
    censored at ``zeta``, ``lambda(zeta) (1, zeta)`` with ``lambda`` the
    standard hazard."""
    zeta = min(float(zeta), _COMPLETE[family])
    if family == "normal":
        cdf_, pdf_ = stats.norm.cdf(zeta), stats.norm.pdf(zeta)
        # The moments of z below zeta: int z^k phi(z) dz
        m0, m1 = cdf_, -pdf_
        m2 = cdf_ - zeta * pdf_
        m3 = -(zeta**2 + 2.0) * pdf_
        m4 = 3.0 * cdf_ - (zeta**3 + 3.0 * zeta) * pdf_
        failed = np.array(
            [[m2, m3 - m1], [m3 - m1, m4 - 2.0 * m2 + m0]], dtype=float
        )
        censored = float(
            np.exp(2.0 * stats.norm.logpdf(zeta) - stats.norm.logsf(zeta))
        )
    else:
        half = _SEV_SPAN / 2.0
        z = zeta - half + half * _NODES
        weight = half * _WEIGHTS * np.exp(z - np.exp(z))
        h = np.expm1(z)
        scores = np.stack([h, z * h - 1.0])
        failed = (scores * weight) @ scores.T
        censored = float(np.exp(2.0 * zeta - np.exp(zeta)))
    edge = np.array([1.0, zeta])
    return failed + censored * np.outer(edge, edge)


def log_quantity_gradient(
    family: str, sigma: float, p: float | None
) -> tuple[npt.NDArray, float]:
    """The gradient, in ``(mu, sigma)``, of the log of the quantity
    estimated, and that log less ``mu``: the ``p`` quantile, or the mean
    if ``p`` is None."""
    if p is not None:
        z = float(ppf(family, p))
        return np.array([1.0, z]), sigma * z
    if family == "sev":
        return (
            np.array([1.0, float(special.digamma(1.0 + sigma))]),
            float(special.gammaln(1.0 + sigma)),
        )
    return np.array([1.0, sigma]), sigma**2 / 2.0
