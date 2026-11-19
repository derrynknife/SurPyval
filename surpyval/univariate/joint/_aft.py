"""The accelerated (AFT) survival part of the joint likelihood: the
marker speeds up an item's ageing rather than multiplying its hazard.

Item ``i``'s clock runs at the rate

    psi_i(s) = exp(gamma'z_i + eta_i(s)),

``eta_i`` the marker's part (:func:`~._likelihood.predictor`: the
value's effect times the path less its centre, the slope's effect times
its slope), so that by age ``t`` it has used up the *effective age*

    A_i(t) = int_0^t psi_i(s) ds,

and its survival and hazard are those of the baseline at that age,

    S_i(t) = S0(A_i(t)),     h_i(t) = h0(A_i(t)) psi_i(t).

An hour at a high reading uses more than an hour of life, and that life
is not given back when the reading falls (the cumulative-exposure model
of step-stress testing, with the marker as the stress). With a marker
that never changes and a Weibull baseline this is the proportional
hazards model with every effect multiplied by ``beta``; otherwise the
two differ.

The effective age is by the cumulative hazard's rule of
:mod:`._likelihood`: ``s = lo + (hi - lo) w^4`` and 15-point
Gauss-Legendre in ``w`` (the path at fixed fractions of the age, and a
path whose slope is singular at age 0 kept accurate). An interval's
cumulative hazard ``H0(A(hi)) - H0(A(lo))`` is taken from the effective
age at ``lo`` and the age used up over ``(lo, hi]``, integrated
directly, so a narrow interval late in life keeps its precision.

The baseline enters through three functions of the log effective age
(``_log_H0``, ``_log_h0`` and ``_log_H0_between``), the Weibull's
here; another baseline (a log-normal or log-logistic one) is those three
functions.

Everything else -- the paths, the prior, the readings, the adaptive
Gauss-Hermite nodes and the fit -- is the joint model's own
(:mod:`._likelihood`); this module gives the lifetime's terms of the
integrand and of the search for each item's mode.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.scipy.special import logsumexp

from ._likelihood import (
    GL_LOGW,
    GL_NODES,
    GL_U,
    JointArrays,
    PathFunction,
    _between,
    log1mexp,
    predictor,
)

#: The log weights of the effective age's rule: Gauss-Legendre's with the
#: ``4 w^3`` of ``s = lo + (hi - lo) w^4``.
_AGE_LOGW = GL_LOGW + 3 * np.log(GL_NODES)


# -- the baseline (Weibull), as functions of the log effective age ------------


def _log_H0(log_a: Any, theta: dict) -> Any:
    """``log H0(a)``: the Weibull's ``beta (log a - log alpha)``."""
    return theta["beta"] * (log_a - anp.log(theta["alpha"]))


def _log_h0(log_a: Any, theta: dict) -> Any:
    """``log h0(a)``: the Weibull's ``log(beta / alpha) + (beta - 1)
    (log a - log alpha)``."""
    alpha, beta = theta["alpha"], theta["beta"]
    return (
        anp.log(beta) - anp.log(alpha) + (beta - 1) * (log_a - anp.log(alpha))
    )


def _log_expm1(x: Any) -> Any:
    """``log(exp(x) - 1)`` for ``x > 0``, accurate for a small and a
    large ``x`` (finite derivatives on both branches)."""
    small = anp.maximum(anp.minimum(x, 1.0), 1e-300)
    big = anp.maximum(x, 1.0)
    return anp.where(
        x > 1.0,
        big + anp.log1p(-anp.exp(-big)),
        anp.log(anp.expm1(small)),
    )


def _log_H0_between(
    log_lo: Any, log_width: Any, theta: dict, start: npt.NDArray
) -> Any:
    """``log (H0(a + da) - H0(a))``, ``a = exp(log_lo)`` the effective
    age at an interval's start and ``da = exp(log_width)`` the age used
    up over it; where ``start`` is ``False`` the interval starts at age
    0 (``a = 0``, ``log_lo`` ignored). For the Weibull, ``H0(a) (expm1(
    beta log1p(da / a)))``, which keeps its digits when ``da`` is small
    beside ``a``."""
    beta = theta["beta"]
    from_zero = _log_H0(log_width, theta)
    lo = anp.where(start[:, None], log_lo, 0.0)
    grown = beta * anp.logaddexp(0.0, log_width - lo)
    later = _log_H0(lo, theta) + _log_expm1(grown)
    return anp.where(start[:, None], later, from_zero)


# -- the effective age --------------------------------------------------------


def _log_rate(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    ages: npt.NDArray,
    centre: float,
) -> Any:
    """``log psi`` at ``ages`` ``(N, A)`` for parameters ``(N, K, P)``:
    ``(N, K, A)``, the covariates' part and the marker's."""
    lin = anp.dot(arrays.Z, theta["gamma"])
    return lin[:, None, None] + predictor(theta, path, params, ages, centre)


def log_age_between(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    lo: npt.NDArray,
    hi: npt.NDArray,
    centre: float,
) -> Any:
    """``log int_lo^hi psi(s) ds`` at the nodes ``(N, K)``, the age used
    up over ``(lo_i, hi_i]``, for ``0 <= lo < hi`` (finite)."""
    width = hi - lo
    s = lo[:, None] + width[:, None] * GL_U[None, :]
    log_psi = _log_rate(theta, path, params, arrays, s, centre)
    return np.log(width)[:, None] + logsumexp(_AGE_LOGW + log_psi, axis=-1)


def log_age(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    t: npt.NDArray,
    centre: float,
) -> Any:
    """``log A(t_i | theta)``, the effective age, at the nodes ``(N, K)``:
    finite everywhere (the age 1's where ``t_i`` is 0, for the caller to
    mask), so a derivative through a mask is never ``0 * inf``."""
    safe = np.where(t > 0, t, 1.0)
    return log_age_between(
        theta, path, params, arrays, np.zeros(safe.size), safe, centre
    )


# -- the lifetime's terms of the integrand ------------------------------------


def log_cumhaz(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    t: npt.NDArray,
    centre: float,
) -> Any:
    """``log H(t_i | theta) = log H0(A(t_i))`` at the nodes ``(N, K)``
    (``-inf`` where ``t_i`` is 0)."""
    out = _log_H0(log_age(theta, path, params, arrays, t, centre), theta)
    return anp.where((t > 0)[:, None], out, -anp.inf)


def log_event(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    centre: float,
) -> Any:
    """``log h(T_i | theta) = log h0(A(T_i)) + log psi(T_i)`` at the
    nodes (at age 1 where ``T_i`` is 0, a left censored item's, which has
    no event there)."""
    T = np.where(arrays.T > 0, arrays.T, 1.0)
    log_a = log_age(theta, path, params, arrays, T, centre)
    log_psi = _log_rate(theta, path, params, arrays, T[:, None], centre)
    return _log_h0(log_a, theta) + log_psi[..., 0]


def log_fails_between(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    lo: npt.NDArray,
    hi: npt.NDArray,
    centre: float,
) -> Any:
    """``log (1 - S(hi | theta) / S(lo | theta))`` at the nodes ``(N,
    K)``, the probability of failing in ``(lo, hi]`` having reached
    ``lo``; 0 for a row whose ``hi`` is ``inf``. The interval's
    cumulative hazard is the baseline's between the effective age at
    ``lo`` and that plus the age used up over the interval."""
    rows, lo_s, hi_s = _between(lo, hi)
    if not rows.any():
        return anp.zeros(params.shape[:2])
    log_width = log_age_between(
        theta, path, params, arrays, lo_s, hi_s, centre
    )
    start = lo_s > 0
    log_lo = log_age(theta, path, params, arrays, lo_s, centre)
    log_dH = _log_H0_between(log_lo, log_width, theta, start)
    return anp.where(rows[:, None], log1mexp(log_dH), 0.0)


# -- the lifetime's terms of the search for each item's mode ------------------


def mode_terms(
    theta: dict,
    arrays: JointArrays,
    d: npt.NDArray,
    t: npt.NDArray,
    upper: npt.NDArray | None,
    followed: Any,
    log_effect: Any,
    weighted: Any,
) -> tuple:
    """The lifetime's part of ``-log`` the integrand at one point of the
    varying parameters per row, its gradient and its (Gauss-Newton)
    curvature: ``(f, g, H)`` of shapes ``(N,)``, ``(N, q)``, ``(N, q,
    q)``. ``followed``, ``log_effect`` and ``weighted`` are the
    links' values and Jacobians at given ages, the marker's part of the
    log rate, and the gradient and Gauss-Newton curvature of a weighted
    sum of the rate in the varying parameters (as
    :func:`~._likelihood._mode_objective` defines them).

    With the effective age ``A``, its gradient ``a = sum lam g`` and
    ``S = sum lam g g'`` over the rule's nodes (``lam`` the node's share
    of ``A``, ``g`` the gradient of the log rate there), the Weibull's
    ``H = (A / alpha)^beta`` has gradient ``beta H a / A`` and curvature
    ``(beta H / A) (S - a a' / A) + beta^2 H a a' / A^2``, both parts
    positive semi-definite (Cauchy-Schwarz). An event adds ``-(beta - 1)
    log A - log psi(T)``, whose curvature ``(1 - beta) (S - a a' / A) /
    A`` is joined to the first part, its weight held at 0 or above. An
    interval's term is as in the proportional hazards search: ``-log(1 -
    exp(-Delta))`` with a Gauss-Newton curvature along ``Delta``'s
    gradient."""
    alpha = float(theta["alpha"])
    beta = float(theta["beta"])
    lin = arrays.Z @ np.asarray(theta["gamma"], dtype=float)

    def age(lo: npt.NDArray, width: npt.NDArray) -> tuple:
        """The age used up over ``(lo, lo + width]``, its gradient and
        ``S``."""
        s = lo[:, None] + width[:, None] * GL_U[None, :]
        terms = followed(s)
        log_lam = (
            (lin + np.log(width))[:, None] + _AGE_LOGW + log_effect(terms)
        )
        lam = np.exp(log_lam)
        grad, curve = weighted(lam, terms)
        return lam.sum(axis=1), grad, curve

    positive = t > 0
    safe = np.where(positive, t, 1.0)
    A, a, S = age(np.zeros(safe.size), safe)
    A = np.where(positive, A, 1.0)
    a = np.where(positive[:, None], a, 0.0)
    S = np.where(positive[:, None, None], S, 0.0)
    H_t = np.where(positive, (A / alpha) ** beta, 0.0)
    outer = np.einsum("ni,nj->nij", a, a)
    f = H_t.copy()
    g = (beta * H_t / A)[:, None] * a
    weight = np.maximum(beta * H_t - d * (beta - 1.0), 0.0) / A
    H = (
        weight[:, None, None] * (S - outer / A[:, None, None])
        + (beta**2 * H_t / A**2)[:, None, None] * outer
    )
    if np.any(d > 0):
        f = f - d * (beta - 1.0) * np.log(A)
        g = g - (d * (beta - 1.0) / A)[:, None] * a
        at_T = followed(safe[:, None])
        f = f - d * log_effect(at_T)[:, 0]
        for effect, _, JT in at_T:
            g = g - (d * effect)[:, None] * JT[:, 0]
    if upper is not None:
        rows, lo, hi = _between(t, upper)
        if rows.any():
            width = hi - lo
            dA, da, _ = age(lo, width)
            start = rows & (lo > 0)
            A_lo = np.where(start, A, 1.0)
            a_lo = np.where(start[:, None], a, 0.0)
            ratio = dA / A_lo
            grown = np.log1p(ratio)
            delta = np.where(
                start,
                (A_lo / alpha) ** beta * np.expm1(beta * grown),
                (dA / alpha) ** beta,
            )
            delta = np.where(rows, delta, 1.0)
            # d Delta = h0(A_hi) dA_hi - h0(A_lo) dA_lo, with dA_hi =
            # dA_lo + d(dA): the difference of the two hazards taken
            # from expm1, so a narrow interval keeps its digits
            h_hi = (beta / alpha) * (
                np.where(start, A_lo + dA, dA) / alpha
            ) ** (beta - 1.0)
            h_gap = np.where(
                start,
                (beta / alpha)
                * (A_lo / alpha) ** (beta - 1.0)
                * np.expm1((beta - 1.0) * grown),
                0.0,
            )
            g_delta = h_hi[:, None] * da + h_gap[:, None] * a_lo
            g_delta = np.where(rows[:, None], g_delta, 0.0)
            small = -np.expm1(-delta)
            f = f - np.where(rows, np.log(small), 0.0)
            g = g - np.where(rows, 1.0 / np.expm1(delta), 0.0)[:, None] * (
                g_delta
            )
            curve = np.where(rows, np.exp(-delta) / small**2, 0.0)
            H = H + curve[:, None, None] * np.einsum(
                "ni,nj->nij", g_delta, g_delta
            )
    return f, g, H
