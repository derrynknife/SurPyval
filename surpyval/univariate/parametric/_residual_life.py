"""The mean residual life of a parametric distribution (#825):

.. math::
    \\mathrm{MRL}(t) = E[T - t \\mid T > t]
    = \\frac{1}{S(t)} \\int_t^\\infty S(u)\\, du ,

computed for each distribution by ``ParametricFitter.mrl``, from the
helpers here: the integral of the conditional survival for a continuous
distribution (its closed form where a distribution has one, as the
Weibull and the Exponential do) and the sum over the integers for a
discrete one.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.special import gammaincc, gammaln

#: Terms of a discrete distribution's tail sum taken directly before the
#: sum is taken as the mean less its head instead.
_MAX_TERMS = 1 << 22


#: The increments ``k`` of the cumulative hazard past ``x`` (conditional
#: survival ``e^-k``) at whose times the integral is broken. The small
#: ones matter far before the bulk: from a Normal's x = -4000 the panel
#: to where 1e-3 has failed hid the start of the fall from ``quad``, 6e-4
#: out with an error estimate of 4e-11.
_STEPS = np.array(
    [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    + [16.0, 36.0]
)


def continuous_mrl(dist: Any, x: np.ndarray, *params: Any) -> np.ndarray:
    """The mean residual life at the points ``x``, each strictly inside
    the support of the continuous ``dist``: the integral over ``u > x`` of
    the conditional survival ``exp(H(x) - H(u))``, which stays finite
    where ``S(x)`` underflows. It is broken where ``H`` has risen by each
    of ``_STEPS`` past ``H(x)`` (from ``qf``, where ``S(x)`` is not too
    small for it), which puts the panels where the survival falls:
    integrated in steps of ``1 / h(x)`` alone, a Weibull of shape 5 at
    ``x = 0.01``, where the hazard is 1e-8, came out as 0. The rest, past
    the last break (or from ``x`` where there is none, far in the tail),
    is integrated in steps of ``1 / h`` there. A point whose integral
    ``quad`` cannot settle gets a ``RuntimeWarning``."""
    # Imported here: ``import surpyval`` does not load scipy.integrate
    # (#470).
    from scipy.integrate import IntegrationWarning, quad

    hi = float(dist._support_edges(*params)[1])
    out = np.full(x.shape, np.nan)
    unsettled = []

    def H(u: Any) -> float:
        with np.errstate(all="ignore"):
            return float(dist.Hf(u, *params))

    for k, xk in enumerate(x):
        H_x = H(xk)
        if not np.isfinite(H_x):
            continue

        def conditional(u: float) -> Any:
            return np.exp(H_x - H(u))

        F = -np.expm1(-(H_x + _STEPS))
        with np.errstate(all="ignore"):
            breaks = np.asarray(dist.qf(F[F < 1.0], *params), dtype=float)
        breaks = np.unique(breaks[np.isfinite(breaks) & (breaks > xk)])
        breaks = breaks[breaks < hi]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            head = 0.0
            start = float(xk)
            if breaks.size:
                start = float(breaks[-1])
                head = quad(
                    conditional,
                    float(xk),
                    start,
                    points=breaks[:-1] if breaks.size > 1 else None,
                    epsabs=0.0,
                    epsrel=1e-10,
                    limit=200,
                )[0]
            with np.errstate(all="ignore"):
                h = float(dist.hf(start, *params))
            step = 1.0 / h if 0.0 < h < np.inf else 1.0
            end = (hi - start) / step if np.isfinite(hi) else np.inf
            tail = quad(
                lambda w: conditional(start + step * w),
                0.0,
                end,
                epsabs=0.0,
                epsrel=1e-10,
                limit=200,
            )[0]
        if any(issubclass(w.category, IntegrationWarning) for w in caught):
            unsettled.append(float(xk))
        out[k] = head + step * tail
    if unsettled:
        warnings.warn(
            "The mean residual life's integral did not settle to its "
            "tolerance at {} of {} point(s) (the first at {:g}): the "
            "value there may be inaccurate.".format(
                len(unsettled), len(x), unsettled[0]
            ),
            RuntimeWarning,
            stacklevel=4,
        )
    return out


def discrete_mrl(
    dist: Any, x: np.ndarray, mean: float, *params: Any
) -> np.ndarray:
    """The mean residual life at the points ``x`` of the discrete
    ``dist``, on the non-negative integers with ``sf(k) = P(T > k)``:

    .. math::
        E[T - m \\mid T > m] = \\sum_{k \\ge m} \\frac{S(k)}{S(m)}

    at an integer ``m``, and ``MRL(floor(t)) - (t - floor(t))`` between
    them (surviving past ``t`` is surviving past ``floor(t)``; a
    distribution's ``sf`` between the integers is not always the step
    this needs). The sum runs until its terms are below rounding; one
    that has not by ``_MAX_TERMS`` terms (a heavy tail) is the mean less
    the sum's head, ``sum_{k < m} S(k)``. ``nan`` where ``S(m)`` is 0."""
    out = np.full(x.shape, np.nan)
    floors = np.floor(x)
    for m in np.unique(floors):
        at = floors == m
        with np.errstate(all="ignore"):
            log_s_m = float(np.ravel(dist.log_sf(np.array([m]), *params))[0])
        if log_s_m == 0.0:
            # Certain to survive past m: the whole mean is still to come.
            out[at] = mean - x[at]
            continue
        if not np.isfinite(log_s_m):
            continue
        out[at] = _tail_sum(dist, m, log_s_m, mean, params) - (x[at] - m)
    return out


def _tail_sum(
    dist: Any, m: float, log_s_m: float, mean: float, params: tuple
) -> float:
    """``sum_{k >= m} S(k) / S(m)`` for the integer ``m`` (see
    :func:`discrete_mrl`)."""
    total = 0.0
    start, size = m, 64
    while start - m < _MAX_TERMS:
        k = np.arange(start, start + size, dtype=float)
        with np.errstate(all="ignore"):
            terms = np.exp(np.asarray(dist.log_sf(k, *params)) - log_s_m)
        total += float(np.sum(terms))
        last = float(terms[-1])
        if last == 0.0 or last <= 1e-17 * total:
            return total
        start += size
        size = min(2 * size, 1 << 16)
    # A heavy tail: E[T] = sum_{k >= 0} S(k) for T on the non-negative
    # integers, so the tail is the mean less the head.
    if m > _MAX_TERMS:
        return np.nan
    k = np.arange(0.0, m)
    with np.errstate(all="ignore"):
        head = float(np.sum(np.asarray(dist.sf(k, *params))))
    return (mean - head) / np.exp(log_s_m)


def upper_gamma_scaled(a: float, z: np.ndarray) -> np.ndarray:
    """``Gamma(a, z) e^z``, the upper incomplete gamma function scaled by
    ``e^z``, for ``a > 0`` and ``z >= 0``: from its continued fraction
    (modified Lentz, as in Numerical Recipes' ``gcf``) where ``z > a +
    1``, which never forms ``e^-z`` and so stays exact where ``Gamma(a,
    z)`` underflows (``log Q`` there is ``-z`` to rounding, and ``e^(log
    Q + z)`` lost every digit by ``z = 1e20``); from ``gammaincc``
    below, where ``e^z`` is moderate."""
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape)
    series = z <= a + 1.0
    if series.any():
        zs = z[series]
        out[series] = np.exp(gammaln(a) + zs) * gammaincc(a, zs)
    zc = z[~series]
    if zc.size:
        tiny = 1e-300
        b = zc + 1.0 - a
        c = np.full(zc.shape, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        for i in range(1, 1000):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            delta = d * c
            h = h * delta
            if np.all(np.abs(delta - 1.0) < 1e-16):
                break
        out[~series] = zc**a * h
    return out
