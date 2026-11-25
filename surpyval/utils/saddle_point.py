r"""
Masses, densities and incomplete gamma functions at large shapes, without
the cancellation of large terms.

The log of a Poisson mass, :math:`k \ln \mu - \mu - \ln \Gamma(k + 1)`, is
a sum of terms of size :math:`k \ln k` that cancel to the size of the log
mass itself: at :math:`\mu = 10^6` the terms are of size 1.4e7 and their
rounding (3e-9) is all that is left of a mass of 4e-4. The same is true
of the binomial and negative binomial masses, of the gamma density at a
large shape, and of the factor :math:`x^a e^{-x} / \Gamma(a)` in front of
the incomplete gamma and beta functions.

Catherine Loader's saddle-point form (*Fast and accurate computation of
binomial probabilities*, 2000; the form R's ``dpois`` and ``dbinom``
use) writes each as a sum of terms that are small where the mass is
large:

.. math::
    \frac{\mu^k e^{-\mu}}{\Gamma(k + 1)} = \frac{e^{-\delta(k) -
    D(k, \mu)}}{\sqrt{2 \pi k}}, \qquad
    D(k, \mu) = k \ln \frac{k}{\mu} + \mu - k,

where :math:`\delta(k) = \ln \Gamma(k + 1) - (k + \tfrac12) \ln k + k -
\tfrac12 \ln 2\pi` is the error of Stirling's formula (of size
:math:`1 / 12k`) and the deviance :math:`D \geq 0` is taken, where
:math:`k` is near :math:`\mu`, from its series in :math:`v = (k - \mu) /
(k + \mu)`, :math:`D = (k - \mu) v + 2k \sum_{j \geq 1} v^{2j + 1} / (2j
+ 1)`, which has no cancellation.

For the regularised incomplete gamma functions at a shape :math:`a \geq
100` the prefactor is formed so, and the sum after it by Temme's uniform
asymptotic expansion near :math:`x = a` (:mod:`._temme`, written in the
log time) and by the power series (below) or Legendre's continued
fraction (above) further out, where each converges in tens of terms.

Everything here is plain numpy, for values: the callers keep their
differentiable forms for a fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln as _sc_gammaln

_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))
_LOG_2PI = float(np.log(2.0 * np.pi))
_LN2 = float(np.log(2.0))

# The Stirling series of the remainder, as coefficients of 1/n, 1/n^3, ...,
# 1/n^15: enough for 1e-17 of it at n >= 10.
_STIRLING = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
)
_SERIES_FROM = 10.0

#: Where ``bd0`` takes its series in ``v``: below this ``|v|`` (k / mu
#: between 1/3 and 3), 28 terms reach 1e-17 of it. Above, the direct
#: form loses at most a factor 2.5 of its value to cancellation.
_BD0_SERIES = 0.5
_BD0_TERMS = 40


def stirlerr(n: npt.ArrayLike) -> npt.NDArray:
    r""":math:`\delta(n) = \ln \Gamma(n + 1) - (n + \tfrac12) \ln n + n -
    \tfrac12 \ln 2\pi`, the error of Stirling's formula, for :math:`n > 0`:
    from its asymptotic series from :math:`n = 10` (to 1e-17), and directly
    below (to an absolute 5e-15, where the terms are of size 25 at most).
    """
    n = np.asarray(n, dtype=float)
    big = n >= _SERIES_FROM
    nb = np.where(big, n, _SERIES_FROM)
    inv2 = 1.0 / (nb * nb)
    total: Any = _STIRLING[-1]
    for c in _STIRLING[-2::-1]:
        total = c + inv2 * total
    series = total / nb
    ns = np.where(big, 1.0, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = _sc_gammaln(ns + 1.0) - (ns + 0.5) * np.log(ns) + ns
    return np.where(big, series, direct - _HALF_LOG_2PI)


def bd0(x: npt.ArrayLike, m: npt.ArrayLike) -> npt.NDArray:
    r"""The deviance :math:`D(x, m) = x \ln(x / m) + m - x \geq 0`, for
    :math:`x \geq 0` and :math:`m > 0`, to a few ulps of itself: from its
    series in :math:`v = (x - m) / (x + m)` where :math:`|v| < 1/2`."""
    x, m = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(m, dtype=float)
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        diff = x - m
        v = diff / (x + m)
        near = np.abs(v) < _BD0_SERIES
        vn = np.where(near, v, 0.0)
        xn = np.where(near, x, 0.0)
        s = diff * vn
        ej = 2.0 * xn * vn
        v2 = vn * vn
        for j in range(1, _BD0_TERMS):
            ej = ej * v2
            term = ej / (2 * j + 1)
            s = s + term
            if np.all(np.abs(term) <= 1e-17 * np.abs(s)):
                break
        ratio = x / m
        log_ratio = np.where(
            (ratio > 0) & np.isfinite(ratio),
            np.log(np.where((ratio > 0) & np.isfinite(ratio), ratio, 1.0)),
            np.log(x) - np.log(m),
        )
        direct = np.where(x == 0, m, x * log_ratio + m - x)
    return np.where(near, s, direct)


def log_poisson_mass(k: npt.ArrayLike, mu: npt.ArrayLike) -> npt.NDArray:
    r""":math:`\ln(\mu^k e^{-\mu} / \Gamma(k + 1))` for real :math:`k
    \geq 0` and :math:`\mu \geq 0`, by the saddle-point form: ``-mu`` at
    ``k = 0``, and :math:`-\infty` at :math:`\mu = 0` for :math:`k > 0`."""
    k, mu = np.broadcast_arrays(
        np.asarray(k, dtype=float), np.asarray(mu, dtype=float)
    )
    pos = (k > 0) & (mu > 0)
    ks = np.where(pos, k, 1.0)
    ms = np.where(pos, mu, 1.0)
    out = -stirlerr(ks) - bd0(ks, ms) - 0.5 * np.log(ks) - _HALF_LOG_2PI
    out = np.where(k == 0, -mu, np.where(pos, out, -np.inf))
    return out


def log_binomial_mass(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    p: npt.ArrayLike,
    q: npt.ArrayLike,
) -> npt.NDArray:
    r""":math:`\ln \frac{\Gamma(x + y + 1)}{\Gamma(x + 1) \Gamma(y + 1)}
    p^x q^y` for real :math:`x, y \geq 0` and :math:`p + q = 1`, each
    given (the smaller exact, the larger rounded from it), by the
    saddle-point form

    .. math::
        \delta(n) - \delta(x) - \delta(y) - D(x, n p) - D(y, n q) -
        \tfrac12 \ln(2 \pi x y / n), \qquad n = x + y.
    """
    x, y, p, q = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, y, p, q))
    )
    n = x + y
    inner = (x > 0) & (y > 0) & (p > 0) & (q > 0)
    xs = np.where(inner, x, 1.0)
    ys = np.where(inner, y, 1.0)
    ns = np.where(inner, n, 2.0)
    ps = np.where(inner, p, 0.5)
    qs = np.where(inner, q, 0.5)
    lc = (
        stirlerr(ns)
        - stirlerr(xs)
        - stirlerr(ys)
        - bd0(xs, ns * ps)
        - bd0(ys, ns * qs)
    )
    lf = _LOG_2PI + np.log(xs) + np.log(ys) - np.log(ns)
    out = lc - 0.5 * lf
    with np.errstate(divide="ignore", invalid="ignore"):
        # the edges: all of n on one side (0 * log 0 is 0). The smaller
        # of p and q is the caller's own, exact, and the larger was
        # rounded from it: its log is log1p of minus the smaller one
        # (log(1 - 1e-6) rounded lost 3e-11 of itself)
        log_p = np.where(p <= q, np.log(p), np.log1p(-q))
        log_q = np.where(q <= p, np.log(q), np.log1p(-p))
        at_x0 = np.where(y == 0, 0.0, y * log_q)
        at_y0 = np.where(x == 0, 0.0, x * log_p)
    return np.where(inner, out, np.where(x == 0, at_x0, at_y0))


#: The smallest shape at which ``gamma_logs_large`` is used: the shape
#: of Temme's expansion is 1 / sqrt(a), below 0.1 there (``_temme``).
LARGE_SHAPE = 100.0
_SERIES_TERMS = 200
_FRACTION_TERMS = 500


def _log1mexp(v: npt.NDArray) -> npt.NDArray:
    """log(1 - exp(v)) for v <= 0, from whichever of expm1 and log1p keeps
    its digits."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            v > -_LN2,
            np.log(-np.expm1(v)),
            np.log1p(-np.exp(np.minimum(v, 0.0))),
        )


def gamma_logs_large(
    a: npt.ArrayLike, x: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""
    :math:`(\ln P(a, x), \ln Q(a, x))`, the logs of the regularised lower
    and upper incomplete gamma functions, for a shape :math:`a \geq 100`
    (``LARGE_SHAPE``) and :math:`x \geq 0`.

    Where :math:`|\eta| \leq 1` (:math:`x / a` between about 0.32 and 2.36;
    :math:`\eta^2 / 2 = x / a - 1 - \ln(x / a)`), the smaller tail is
    Temme's expansion (:func:`._temme.small_tail`); further out it is the
    saddle-point prefactor :math:`x^a e^{-x} / \Gamma(a + 1)` times the
    power series of :math:`P` (below :math:`a`, its terms shrinking by
    :math:`x / (a + n) < 0.32`) or Legendre's continued fraction of
    :math:`Q` (above). The larger tail is :math:`\ln(1 - e^{v})` of the
    smaller one, ``v``.

    scipy's ``gammaincc`` forms the prefactor as :math:`\exp(a \ln x - x -
    \ln \Gamma(a))` once :math:`|x - a| > 0.4 a`, terms of size
    :math:`a \ln a` (7e3 at :math:`a = 10^3`) whose rounding was 1.2e-12
    of the tail where it is 1e-30.
    """
    from surpyval.univariate.parametric.distributions._temme import (
        ETA_MAX,
        eta_of,
        small_tail_gamma,
    )

    a, x = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(x, dtype=float)
    )
    shape = a.shape
    a = a.ravel()
    x = x.ravel()
    log_p = np.full(a.shape, np.nan)
    log_q = np.full(a.shape, np.nan)
    at_zero = x <= 0
    at_inf = np.isposinf(x)
    log_p[at_zero], log_q[at_zero] = -np.inf, 0.0
    log_p[at_inf], log_q[at_inf] = 0.0, -np.inf
    todo = ~at_zero & ~at_inf & np.isfinite(x) & np.isfinite(a)
    if np.any(todo):
        at, xt = a[todo], x[todo]
        Q = 1.0 / np.sqrt(at)
        # x - a is exact within a factor 2 of a, so the log time keeps
        # the digits of a small x / a - 1
        w = np.log1p((xt - at) / at) / Q
        # (inf or nan far out, where the expansion is not used)
        with np.errstate(over="ignore", invalid="ignore"):
            eta = eta_of(w, Q)
        small = np.empty_like(at)
        upper = np.empty(at.shape, dtype=bool)
        temme = np.abs(eta) <= ETA_MAX
        if np.any(temme):
            st, ut = small_tail_gamma(
                w[temme], Q[temme], bd0(at[temme], xt[temme])
            )
            small[temme] = st
            upper[temme] = ut
        below = ~temme & (xt < at)
        if np.any(below):
            ab, xb = at[below], xt[below]
            total = np.ones_like(xb)
            term = np.ones_like(xb)
            for n in range(1, _SERIES_TERMS):
                term = term * xb / (ab + n)
                total = total + term
                if np.all(term <= 1e-17 * total):
                    break
            small[below] = log_poisson_mass(ab, xb) + np.log(total)
            upper[below] = False
        above = ~temme & (xt > at)
        if np.any(above):
            aa, xa = at[above], xt[above]
            tiny = 1e-300
            b = xa + 1.0 - aa
            c = np.full_like(xa, 1.0 / tiny)
            d = 1.0 / b
            h = d.copy()
            for i in range(1, _FRACTION_TERMS):
                an = -i * (i - aa)
                b = b + 2.0
                d = an * d + b
                d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
                c = b + an / c
                c = np.where(np.abs(c) < tiny, tiny, c)
                step = d * c
                h = h * step
                if np.all(np.abs(step - 1.0) < 1e-16):
                    break
            # Q = x^a e^-x / Gamma(a) * h = a * mass(a, x) * h
            small[above] = np.log(aa) + log_poisson_mass(aa, xa) + np.log(h)
            upper[above] = True
        large = _log1mexp(small)
        log_q[todo] = np.where(upper, small, large)
        log_p[todo] = np.where(upper, large, small)
    return log_p.reshape(shape), log_q.reshape(shape)


def log_gamma_density_large(a: Any, y: Any) -> Any:
    r""":math:`\ln(y^{a - 1} e^{-y} / \Gamma(a))`, the standard gamma's
    log density, by the saddle-point form, for a shape :math:`a \geq
    100` (``LARGE_SHAPE``) and :math:`y > 0`, in autograd numpy, so a fit
    differentiates it: :math:`-\delta(a - 1) - D(a - 1, y) - \tfrac12
    \ln(2 \pi (a - 1))`. The direct form's terms are of size :math:`a \ln
    a`, and at a shape of 1e12 their rounding was 1e-3 of the log
    density."""
    import autograd.numpy as anp
    from autograd.tracer import getval

    k = a - 1.0
    inv2 = 1.0 / (k * k)
    total: Any = _STIRLING[-1]
    for c in _STIRLING[-2::-1]:
        total = c + inv2 * total
    stirl = total / k
    diff = k - y
    v = diff / (k + y)
    near = np.abs(getval(v)) < _BD0_SERIES
    vn = anp.where(near, v, 0.0)
    s = diff * vn
    ej = 2.0 * anp.where(near, k, 0.0) * vn
    v2 = vn * vn
    for j in range(1, _BD0_TERMS):
        ej = ej * v2
        term = ej / (2 * j + 1)
        s = s + term
        if np.all(np.abs(getval(term)) <= 1e-17 * np.abs(getval(s))):
            break
    y_far = anp.where(near, k, y)
    direct = k * anp.log(k / y_far) + y_far - k
    deviance = anp.where(near, s, direct)
    return -stirl - deviance - 0.5 * anp.log(k) - _HALF_LOG_2PI
