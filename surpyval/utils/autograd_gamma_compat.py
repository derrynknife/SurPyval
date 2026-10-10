"""
Pure-NumPy/SciPy autograd primitives for incomplete gamma and beta functions.

Replaces the abandoned autograd-gamma package. VJP approach:

  x-derivatives (analytical):
    Use autograd.numpy ops so that ArrayBox inputs (which appear when autograd
    computes Hessians by tracing through the backward pass) are handled
    correctly, giving correct second-order derivatives for the x (rate) param.

  shape-parameter derivatives (numerical central difference, traced):
    Each shape derivative is itself a ``primitive`` whose VJPs return
    numerical second derivatives (pure in a, mixed in the other shape
    parameter and in x). The first-order VJP therefore stays inside the
    autograd trace — both the cotangent ``g`` and the derivative factor are
    boxed — so Hessians through shape-parameter paths are correct. The
    previous implementation stripped the trace with ``getval`` on both
    factors, silently zeroing every second-derivative contribution through
    a shape parameter: censored/truncated Gamma and Beta fits got a
    corrupted (even asymmetric) covariance while their point estimates
    were fine (#270). Third- and higher-order derivatives are cut (the
    inner VJPs are plain numpy), which nothing in surpyval needs.

  The logs of the incomplete beta's tails (``betaincln``, ``betainccln``)
  take their shape derivatives analytically instead, from the continued
  fraction differentiated term by term (``_beta_log_shape_grad``, #621):
  to about 1e-15 against mpmath, where the five-point differences were
  1e-12 to 2e-4 off, in one pass for both shapes where the differences
  took eight evaluations. Their second derivatives are differences of
  those analytic first ones.
"""

from typing import Any, Callable

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import betaln as _ag_betaln
from autograd.scipy.special import gammaln as _ag_gammaln
from autograd.tracer import getval
from scipy.special import betainc as _sc_betainc
from scipy.special import betaincc as _sc_betaincc
from scipy.special import digamma as _sc_digamma
from scipy.special import gammainc as _sc_gammainc
from scipy.special import gammaincc as _sc_gammaincc
from scipy.special import gammaln as _sc_gammaln
from scipy.special import polygamma as _sc_polygamma
from scipy.special import zeta as _sc_zeta

# The value-or-box union the distributions use (see parametric_fitter):
# every boundary here may see a plain numpy value or an ArrayBox.
Boxable = npt.NDArray | float | ArrayBox

# Floor for the logs of the regularised incomplete functions: the smallest
# positive double, so log P / log Q stay exact down to scipy's underflow
# (1e-35 capped a Gamma cumulative hazard at 80.6, far short of the tail).
_LOG_EPS = float(np.finfo(float).tiny)
_EPS_H = np.finfo(float).eps ** (1.0 / 3.0)
# log P is taken as log1p(-Q) where Q is below this (and log Q as
# log1p(-P)): the log of a probability within 1e-3 of 1 has lost the
# digits of its complement (up to all of them, #442), and one further away
# keeps its relative precision to within 1e3 * eps.
_COMPLEMENT = 1e-3
_LN2 = float(np.log(2.0))
# A probability is below _COMPLEMENT only where its complement, as scipy
# computes it (to 1e-14), is above this.
_NEAR_ONE = 0.99


def _where_near_one(
    f: Callable, args: tuple, complement: npt.NDArray
) -> npt.NDArray:
    """``f(*args)`` where ``complement`` (f's complement, computed) is above
    ``_NEAR_ONE``, and 1 elsewhere: where f is below ``_COMPLEMENT`` (the
    only place the callers use it, besides choosing a side) it is f, to
    the bit. Each incomplete function costs as much as its complement, and
    computing both everywhere doubled the cost of a censored fit."""
    complement = np.asarray(complement, dtype=float)
    out = np.ones(complement.shape)
    near = complement > _NEAR_ONE
    if np.any(near):
        out[near] = f(
            *(np.broadcast_to(v, complement.shape)[near] for v in args)
        )
    return out


def _step(v: Boxable) -> Boxable:
    """Exact floating-point step size for central differences."""
    h = np.maximum(np.abs(v) * _EPS_H, 1e-7)
    return (v + h) - v


def _cdiff(f: Callable, v: Boxable) -> Boxable:
    """5th-order central difference of a one-argument callable at v."""
    h = _step(v)
    return (-f(v + 2 * h) + 8 * f(v + h) - 8 * f(v - h) + f(v - 2 * h)) / (
        12 * h
    )


def _cdiff_second(f: Callable, v: Boxable) -> Boxable:
    """5-point second-derivative stencil of a one-argument callable at v."""
    h = _step(v)
    return (
        -f(v + 2 * h)
        + 16 * f(v + h)
        - 30 * f(v)
        + 16 * f(v - h)
        - f(v - 2 * h)
    ) / (12 * h * h)


def _cdiff1(f: Callable, a: Boxable, x: Boxable) -> Boxable:
    """5th-order central diff of f(a, x) w.r.t. a.

    Both a and x must be plain numpy values — call with getval().
    """
    return _cdiff(lambda aa: f(aa, x), a)


def _cdiff2_a(f: Callable, a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    """5th-order central diff of f(a, b, x) w.r.t. a — all args plain numpy."""
    return _cdiff(lambda aa: f(aa, b, x), a)


def _cdiff2_b(f: Callable, a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    """5th-order central diff of f(a, b, x) w.r.t. b — all args plain numpy."""
    return _cdiff(lambda bb: f(a, bb, x), b)


#: The most terms of the series and fraction of the incomplete gamma's
#: exact shape derivatives (``_log_pq_da``, #797), and their tolerance.
_PQ_TERMS = 5000
_PQ_TOL = 4e-16


def _log_pq_shape_derivatives(a: npt.NDArray, x: npt.NDArray) -> tuple:
    """``(d log P / da, d log Q / da, converged)`` of the regularised
    incomplete gamma at shapes ``a > 0`` and times ``0 < x < inf``,
    elementwise, each to rounding (#797).

    Below ``x = a + 1`` from the series of P, ``P = x^a e^-x / Gamma(a +
    1) S`` with ``S = sum_n t_n``, ``t_n = t_{n-1} x / (a + n)``, whose
    terms' derivatives are carried along with them, ``t_n' = (t_{n-1}' x
    - t_n) / (a + n)``: ``d log P = log x - psi(a + 1) + S' / S``, and
    ``d log Q = -(P / Q) d log P`` (P is there below about 0.6). At or
    above it from Legendre's continued fraction of Q by the modified
    Lentz method (Numerical Recipes' ``gser`` and ``gcf``), each step's
    derivative in ``a`` carried along it: ``d log Q = log x - psi(a) + h'
    / h``, and ``d log P = -(Q / P) d log Q``. No difference is taken;
    the five-point differences these replace were up to 4e-9 off mpmath's
    (at ``a = 0.5``, ``x = 1``), these 1e-14. ``converged`` is False
    where neither settled within ``_PQ_TERMS`` terms."""
    a, x = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(x, dtype=float)
    )
    d_log_p = np.full(a.shape, np.nan)
    d_log_q = np.full(a.shape, np.nan)
    converged = np.zeros(a.shape, dtype=bool)
    lx = np.log(x)
    series = x < a + 1.0
    with np.errstate(all="ignore"):
        if np.any(series):
            aa, xx = a[series], x[series]
            t = np.ones(aa.shape)
            dt = np.zeros(aa.shape)
            total, d_total = t.copy(), dt.copy()
            done = np.zeros(aa.shape, dtype=bool)
            for n in range(1, _PQ_TERMS):
                inv = 1.0 / (aa + n)
                t_new = t * xx * inv
                dt = (dt * xx - t_new) * inv
                t = t_new
                total += t
                d_total += dt
                # (checked every fourth term: the checks cost as much as
                # the terms, and a converged sum takes no harm from more)
                if n % 4 == 0:
                    done = (np.abs(t) <= _PQ_TOL * total) & (
                        np.abs(dt) <= _PQ_TOL * np.abs(d_total)
                    )
                    if np.all(done):
                        break
            dlp = lx[series] - _sc_digamma(aa + 1.0) + d_total / total
            log_p = (
                aa * lx[series] - xx - _sc_gammaln(aa + 1.0) + np.log(total)
            )
            p_val = np.exp(log_p)
            d_log_p[series] = dlp
            d_log_q[series] = -p_val / (1.0 - p_val) * dlp
            converged[series] = done
        frac = ~series
        if np.any(frac):
            aa, xx = a[frac], x[frac]
            tiny = 1e-300
            b = xx + 1.0 - aa
            c = np.full(aa.shape, 1.0 / tiny)
            dc = np.zeros(aa.shape)
            d = 1.0 / b
            dd = d * d  # d(1 / b)/da, with db/da = -1
            h, dh = d.copy(), dd.copy()
            done = np.zeros(aa.shape, dtype=bool)
            # (From x >= a + 1 the denominators stay away from 0: no
            # guard against one is needed, and a point that is not finite
            # is not converged, so is differenced instead.)
            for i in range(1, _PQ_TERMS):
                an = -i * (i - aa)
                b += 2.0
                inv_c = 1.0 / c
                d_big_d = i * d + an * dd - 1.0
                dc = i * inv_c - an * dc * inv_c * inv_c - 1.0
                c = b + an * inv_c
                d = 1.0 / (an * d + b)
                dd = -d_big_d * d * d
                delta = d * c
                dh_new = dh * delta + h * (dd * c + d * dc)
                h = h * delta
                if i % 4 == 0:
                    done = (np.abs(delta - 1.0) <= _PQ_TOL) & (
                        np.abs(dh_new - dh) <= _PQ_TOL * np.abs(dh_new)
                    )
                    dh = dh_new
                    if np.all(done):
                        break
                else:
                    dh = dh_new
            dlq = lx[frac] - _sc_digamma(aa) + dh / h
            log_q = aa * lx[frac] - xx - _sc_gammaln(aa) + np.log(h)
            q_val = np.exp(log_q)
            d_log_q[frac] = dlq
            d_log_p[frac] = -q_val / (1.0 - q_val) * dlq
            converged[frac] = done
    return d_log_p, d_log_q, converged


def _log_pq_da(which: int) -> Callable:
    """The exact shape derivative of log P (``which`` 0) or log Q (1),
    ``f(a, x)`` in plain numpy (:func:`_log_pq_shape_derivatives`);
    ``nan`` where that did not converge, or at an ``x`` of 0 or inf
    (``_make_da_primitive`` differences those)."""

    def f_a(a: Boxable, x: Boxable) -> npt.NDArray:
        a_arr, x_arr = np.broadcast_arrays(
            np.asarray(a, dtype=float), np.asarray(x, dtype=float)
        )
        inside = (x_arr > 0) & np.isfinite(x_arr) & (a_arr > 0)
        out = np.full(a_arr.shape, np.nan)
        if np.any(inside):
            found = _log_pq_shape_derivatives(a_arr[inside], x_arr[inside])
            value = found[which]
            out[inside] = np.where(
                found[2] & np.isfinite(value), value, np.nan
            )
        return out

    return f_a


def _make_da_primitive(
    f: Callable, dfdx: Callable, f_a: "Callable | None" = None
) -> Callable:
    """Traced first derivative w.r.t. the shape parameter of f(a, x).

    Returns a primitive ``f_da(a, x) = df/da`` whose own VJPs are the
    numerical second derivatives d2f/da2 and d2f/dadx, so a Hessian pass
    through ``f_da`` picks up the correct curvature (further orders cut).

    The mixed derivative d2f/dadx is the difference in ``a`` of ``dfdx``,
    f's analytic derivative in x (plain numpy). It was the difference in
    x of the difference in a, whose step in x has a floor of 1e-7: at an
    x below 2e-7 its stencil reached a negative x, where the incomplete
    gamma is NaN, and the NaN reached every entry of a Hessian through x
    (a GammaAFT whose coefficients ran off to -23, its censored rows at
    x ~ 1e-10, #634). The step in ``a`` is relative to it, and ``a`` is
    positive.

    With ``f_a``, f's exact derivative in the shape where it is not
    ``nan`` (``_log_pq_da``, #797), ``df/da`` is that: the five-point
    differences of ``f`` were up to 4e-9 off mpmath's for the logs of the
    incomplete gamma below x = 30. ``d2f/da2`` is then the difference of
    that first derivative.
    """

    def first(a: Boxable, x: Boxable) -> Boxable:
        if f_a is None:
            return _cdiff1(f, a, x)
        out = np.array(f_a(a, x), dtype=float)
        missing = np.isnan(out)
        if np.any(missing):
            # (only where the exact form did not settle)
            a_b, x_b = np.broadcast_arrays(
                np.asarray(a, dtype=float), np.asarray(x, dtype=float)
            )
            out[missing] = _cdiff1(f, a_b[missing], x_b[missing])
        return out if out.ndim else float(out)

    @primitive
    def f_da(a: Boxable, x: Boxable) -> Boxable:
        return first(a, x)

    def vjp_a(ans: Boxable, a: Boxable, x: Boxable) -> Callable:
        av, xv = getval(a), getval(x)
        if f_a is None:
            d2 = _cdiff_second(lambda aa: f(aa, xv), av)
        else:
            d2 = _cdiff(lambda aa: first(aa, xv), av)
        return unbroadcast_f(a, lambda g: getval(g) * d2)

    def vjp_x(ans: Boxable, a: Boxable, x: Boxable) -> Callable:
        av, xv = getval(a), getval(x)
        with np.errstate(all="ignore"):
            mixed = _cdiff(lambda aa: dfdx(aa, xv), av)
        return unbroadcast_f(x, lambda g: getval(g) * mixed)

    defvjp(f_da, vjp_a, vjp_x)
    return f_da


def _log_gamma_density(a: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    """``log(x^(a-1) e^-x / Gamma(a))``, the log of dP(a, x)/dx, in plain
    numpy (for the mixed derivatives of ``_make_da_primitive``)."""
    with np.errstate(all="ignore"):
        return -x + np.log(x) * (a - 1) - _sc_gammaln(a)


def _make_dab_primitives(f: Callable) -> tuple[Callable, Callable]:
    """Traced first derivatives w.r.t. both shape parameters of f(a, b, x).

    Returns primitives ``(f_da, f_db)`` whose VJPs are the numerical pure,
    cross and mixed-with-x second derivatives.
    """

    @primitive
    def f_da(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
        return _cdiff2_a(f, a, b, x)

    @primitive
    def f_db(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
        return _cdiff2_b(f, a, b, x)

    def _vals(
        a: Boxable, b: Boxable, x: Boxable
    ) -> tuple[Boxable, Boxable, Boxable]:
        return getval(a), getval(b), getval(x)

    def da_vjp_a(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff_second(lambda aa: f(aa, bv, xv), av)
        return unbroadcast_f(a, lambda g: getval(g) * d2)

    def da_vjp_b(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff(lambda bb: _cdiff2_a(f, av, bb, xv), bv)
        return unbroadcast_f(b, lambda g: getval(g) * d2)

    def da_vjp_x(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        mixed = _cdiff(lambda xx: _cdiff2_a(f, av, bv, xx), xv)
        return unbroadcast_f(x, lambda g: getval(g) * mixed)

    def db_vjp_a(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff(lambda aa: _cdiff2_b(f, aa, bv, xv), av)
        return unbroadcast_f(a, lambda g: getval(g) * d2)

    def db_vjp_b(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff_second(lambda bb: f(av, bb, xv), bv)
        return unbroadcast_f(b, lambda g: getval(g) * d2)

    def db_vjp_x(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        mixed = _cdiff(lambda xx: _cdiff2_b(f, av, bv, xx), xv)
        return unbroadcast_f(x, lambda g: getval(g) * mixed)

    defvjp(f_da, da_vjp_a, da_vjp_b, da_vjp_x)
    defvjp(f_db, db_vjp_a, db_vjp_b, db_vjp_x)
    return f_da, f_db


# ---------------------------------------------------------------------------
# gammainc  —  P(a, x), regularised lower incomplete gamma
# ---------------------------------------------------------------------------


@primitive
def gammainc(a: Boxable, x: Boxable) -> Boxable:
    return _sc_gammainc(a, x)


_gammainc_da = _make_da_primitive(
    _sc_gammainc, lambda a, x: np.exp(_log_gamma_density(a, x))
)

defvjp(
    gammainc,
    # d/da: numerical but traced — the product keeps both g and the
    # derivative factor boxed so Hessians through a are correct (#270)
    lambda ans, a, x: unbroadcast_f(a, lambda g: g * _gammainc_da(a, x)),
    # d/dx: analytical; anp.log handles ArrayBox x for correct Hessian
    lambda ans, a, x: unbroadcast_f(
        x, lambda g: g * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a))
    ),
)

# ---------------------------------------------------------------------------
# gammaincln  —  log P(a, x)
# ---------------------------------------------------------------------------


def _gammaincln_raw(a: Boxable, x: Boxable) -> Boxable:
    a_arr, x_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(x, dtype=float)
    )
    p = _sc_gammainc(a_arr, x_arr)
    q = _where_near_one(_sc_gammaincc, (a_arr, x_arr), p)
    with np.errstate(divide="ignore"):
        # log1p(-Q) where P is near 1, and rounds to it (#442)
        out = np.where(
            q < _COMPLEMENT, np.log1p(-np.minimum(q, _COMPLEMENT)), np.log(p)
        )
    # P(a, x) underflows (or goes subnormal) for small x and for x far
    # below a large a, where a clipped log would stop at -708 (#443).
    # There the series
    #   P = x^a e^-x / Gamma(a + 1) * sum_n x^n / ((a + 1) ... (a + n))
    # converges, its terms shrinking by x / (a + n) < 1; it is summed in
    # logs, and to -inf at x = 0.
    tail = (p < 1e-280) & (x_arr < a_arr) & (x_arr > 0)
    if np.any(tail):
        at, xt = a_arr[tail], x_arr[tail]
        total = np.ones_like(xt)
        term = np.ones_like(xt)
        for k in range(1, 2000):
            term = term * xt / (at + k)
            total = total + term
            if np.all(term <= 1e-17 * total):
                break
        out[tail] = (
            at * np.log(xt) - xt - _sc_gammaln(at + 1.0) + np.log(total)
        )
    out = np.where(x_arr == 0, -np.inf, out)
    return out if out.ndim else float(out)


@primitive
def gammaincln(a: Boxable, x: Boxable) -> Boxable:
    return _gammaincln_raw(a, x)


_gammaincln_da = _make_da_primitive(
    _gammaincln_raw,
    lambda a, x: np.exp(_log_gamma_density(a, x) - _gammaincln_raw(a, x)),
    _log_pq_da(0),
)

defvjp(
    gammaincln,
    lambda ans, a, x: unbroadcast_f(a, lambda g: g * _gammaincln_da(a, x)),
    # d/dx of log P = (dP/dx)/P; ans = log P avoids recomputing P
    lambda ans, a, x: unbroadcast_f(
        x,
        lambda g: g
        * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
    ),
)

# ---------------------------------------------------------------------------
# gammainccln  —  log Q(a, x) = log(1 − P(a, x))
# ---------------------------------------------------------------------------


def _gammainccln_raw(a: Boxable, x: Boxable) -> Boxable:
    a_arr, x_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(x, dtype=float)
    )
    q = _sc_gammaincc(a_arr, x_arr)
    out = np.array(np.log(np.clip(q, _LOG_EPS, np.inf)), dtype=float)
    # log1p(-P) where Q is near 1, and rounds to it (#442)
    p = _where_near_one(_sc_gammainc, (a_arr, x_arr), q)
    out = np.where(p < _COMPLEMENT, np.log1p(-np.minimum(p, _COMPLEMENT)), out)
    # Q(a, x) underflows past ~1e-308 (x of ~700 for a small a), where
    # a clipped log would cap the Gamma cumulative hazard. There x >> a
    # and the asymptotic series
    #   log Q = (a-1) log x - x - log Gamma(a)
    #           + log(1 + (a-1)/x + (a-1)(a-2)/x^2 + ...)
    # is accurate; it is summed until the terms stop shrinking.
    tail = (q < 1e-280) & (x_arr > a_arr) & np.isfinite(x_arr)
    if np.any(tail):
        at, xt = a_arr[tail], x_arr[tail]
        total = np.ones_like(xt)
        term = np.ones_like(xt)
        for k in range(1, 30):
            nxt = term * (at - k) / xt
            if np.all(np.abs(nxt) >= np.abs(term)):
                break
            term = np.where(np.abs(nxt) < np.abs(term), nxt, 0.0)
            total = total + term
        out[tail] = (
            (at - 1.0) * np.log(xt) - xt - _sc_gammaln(at) + np.log(total)
        )
    # Q(a, inf) = 0 (the series is inf - inf there, #561)
    out = np.where(np.isposinf(x_arr), -np.inf, out)
    return out if out.ndim else float(out)


@primitive
def gammainccln(a: Boxable, x: Boxable) -> Boxable:
    return _gammainccln_raw(a, x)


_gammainccln_da = _make_da_primitive(
    _gammainccln_raw,
    lambda a, x: -np.exp(_log_gamma_density(a, x) - _gammainccln_raw(a, x)),
    _log_pq_da(1),
)

defvjp(
    gammainccln,
    lambda ans, a, x: unbroadcast_f(a, lambda g: g * _gammainccln_da(a, x)),
    # d/dx of log Q = -(dP/dx)/Q; negated, ans = log Q
    lambda ans, a, x: unbroadcast_f(
        x,
        lambda g: g
        * -anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
    ),
)

# ---------------------------------------------------------------------------
# betainc  —  B(a, b; x), regularised incomplete beta
# ---------------------------------------------------------------------------


@primitive
def betainc(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _sc_betainc(a, b, x)


_betainc_da, _betainc_db = _make_dab_primitives(_sc_betainc)

defvjp(
    betainc,
    lambda ans, a, b, x: unbroadcast_f(a, lambda g: g * _betainc_da(a, b, x)),
    lambda ans, a, b, x: unbroadcast_f(b, lambda g: g * _betainc_db(a, b, x)),
    # d/dx: x^(a-1)*(1-x)^(b-1)/B(a,b); anp handles ArrayBox x
    lambda ans, a, b, x: unbroadcast_f(
        x,
        lambda g: g
        * anp.exp(
            (a - 1) * anp.log(x) + (b - 1) * anp.log(1 - x) - _ag_betaln(a, b)
        ),
    ),
)

# ---------------------------------------------------------------------------
# betaincln  —  log B(a, b; x)
# ---------------------------------------------------------------------------


# The Stirling series of ln Gamma(y) - [(y - 1/2) ln y - y + ln(2 pi)/2],
# as coefficients of 1/y, 1/y^3, ..., 1/y^15: enough for 1e-17 at y >= 10.
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
_ASYMPTOTIC_FROM = 10.0


def _stirling_remainder(y: Boxable) -> Boxable:
    r"""The remainder :math:`\mu(y)` of Stirling's formula for
    :math:`\ln \Gamma(y)`, for :math:`y \geq 10`."""
    inv2 = 1.0 / (y * y)
    total: Boxable = _STIRLING[-1]
    for c in _STIRLING[-2::-1]:
        total = c + inv2 * total
    return total / y


def log_gamma_ratio(y: Boxable, a: Boxable) -> Boxable:
    r""":math:`\ln \Gamma(y + a) - \ln \Gamma(y)`, accurate at any ``y``.

    Two ``gammaln`` of size :math:`y \ln y` lose their difference at a
    large ``y``: at :math:`y = 10^{12}` each is 2.6e13 and carries an
    error of 6e-3, which was the relative error of a BetaGeometric
    survival there (#449). The difference is taken from Stirling's
    formula instead,

    .. math::
        (y - \tfrac12) \operatorname{log1p}(a / y) + a \ln(y + a) - a
        + \mu(y + a) - \mu(y),

    whose terms carry no cancellation (``log1p`` keeps a small ``a / y``),
    after moving a ``y`` below 10 up with :math:`\Gamma(z + 1) = z
    \Gamma(z)`, one :math:`\operatorname{log1p}(a / z)` per step. Each
    ``where`` branch is evaluated at a safe argument, so neither puts a
    nan into the other's gradient.
    """
    z = y
    shifted = 0.0
    for _ in range(int(_ASYMPTOTIC_FROM)):
        below = z < _ASYMPTOTIC_FROM
        z_safe = anp.where(below, z, 1.0)
        shifted = shifted + anp.where(below, anp.log1p(a / z_safe), 0.0)
        z = anp.where(below, z + 1.0, z)
    # z >= 10 here for every y > 0. (A ``where``, not ``maximum``: at a
    # tie, z = 10 exactly from y = 1, 2, ..., 10, autograd's ``maximum``
    # gives each side half the gradient, and d/dy came out halved. A
    # NegativeBinomial fit from r = 4 could not move, #665.)
    z = anp.where(z < _ASYMPTOTIC_FROM, _ASYMPTOTIC_FROM, z)
    return (
        (z - 0.5) * anp.log1p(a / z)
        + a * anp.log(z + a)
        - a
        + _stirling_remainder(z + a)
        - _stirling_remainder(z)
        - shifted
    )


def betaln_accurate(a: Boxable, b: Boxable) -> Boxable:
    r""":math:`\ln B(a, b)` without the cancellation of
    ``gammaln(b) - gammaln(a + b)`` at a large ``b``: 2e-6 at
    ``betaln(1e3, 7e8)`` (#458). With ``s`` the smaller argument and
    ``l`` the larger, :math:`\ln \Gamma(s) - [\ln \Gamma(l + s) - \ln
    \Gamma(l)]`."""
    small = anp.minimum(a, b)
    large = anp.maximum(a, b)
    return _ag_gammaln(small) - log_gamma_ratio(large, small)


def beta_cf(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    x: npt.ArrayLike,
    terms: int | None = None,
) -> npt.NDArray:
    """The continued fraction of the incomplete beta (Lentz), such that
    I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * beta_cf(a, b, x); it
    converges fast for x below (a + 1) / (a + b + 2). ``nan`` where it
    has not converged in ``terms`` terms (``_CF_TERMS`` by default)."""
    a, b, x = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(x, dtype=float),
    )
    tiny = 1e-300
    c = np.ones_like(x)
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, _CF_TERMS if terms is None else terms):
        for aa in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
            c = 1.0 + aa / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            h = np.where(done, h, h * d * c)
        done = done | (np.abs(d * c - 1.0) < 1e-16) | ~np.isfinite(h)
        if np.all(done):
            break
    # Not converged: nan, never the unconverged value (#473, #520).
    return np.where(done, h, np.nan)


#: The most terms of ``beta_cf``. Inside its region, x below (a + 1) /
#: (a + b + 2), it converges in tens of terms on the tails it is used for
#: (71 at most for shapes from 10 to 1e12, 2 to 30 standard deviations
#: out). Outside, the terms it needs grow as 1 / (1 - x): 273 at x = 0.999,
#: 62,042 at 1 - 1e-8, and at an x that rounds to 1 it ran to this limit,
#: 4.6 s a call, and its value was 40% off (#520).
_CF_TERMS = 100000

#: The most terms of a tail's continued fraction in ``_beta_logs`` before
#: the other side's power series is used instead (``_beta_series_log``):
#: past them its x is within 1e-3 of 1, where the series converges in a
#: few terms.
_CF_TAIL_TERMS = 1000


#: ln Gamma(1 + a) - (-euler a) as a power series in a: the coefficients of
#: a^2 ... a^12, (-1)^k zeta(k) / k, for |a| below ``_LNGAMMA1P_SERIES``.
_LNGAMMA1P_SERIES = 1e-2
_LNGAMMA1P_COEFFS = tuple(
    float((-1) ** k * _sc_zeta(k) / k) for k in range(2, 13)
)
_EULER = 0.57721566490153286061
_ZETA2, _ZETA3, _ZETA4 = (float(_sc_zeta(k)) for k in (2, 3, 4))


def _lngamma1p(a: npt.NDArray) -> npt.NDArray:
    """ln Gamma(1 + a), relative to its size for a tiny ``a`` as well:
    ``gammaln(1 + a)`` is 0 once 1 + a rounds to 1."""
    small = np.abs(a) < _LNGAMMA1P_SERIES
    a_s = np.where(small, a, 0.0)
    series = np.zeros_like(a_s)
    for c in _LNGAMMA1P_COEFFS[::-1]:
        series = c + a_s * series
    series = a_s * (-_EULER + a_s * series)
    return np.where(small, series, _sc_gammaln(1.0 + np.where(small, 1.0, a)))


#: The most terms of ``_beta_series_log``; it is used where b x < a + 1,
#: and converges in a few.
_SERIES_TERMS = 10000


def _beta_series_log(
    a: npt.NDArray, b: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    r"""log I_x(a, b) from its power series in x (DLMF 8.17.7),

    .. math::
        I_x(a, b) = x^a \frac{\Gamma(a + b)}{\Gamma(a + 1) \Gamma(b)}
        \Big(1 + a \sum_{n \geq 1} \frac{(1 - b)_n x^n}{n! (a + n)}\Big),

    with each factor's log taken to its own relative precision. It is for
    the side of a tail whose continued fraction does not converge
    (``_beta_logs``): there x is below (a + 1) / (a + b + 2) and I_x(a, b)
    is near 1 because ``a`` is small, b x is below a + 1, and the series
    converges in a few terms. The other tail is ``-expm1`` of this, exact
    to the last digits, where the continued fraction of this side gives a
    log near 0 as a difference of logs of size 1. ``nan`` where the series
    has not converged."""
    total = np.zeros_like(x)
    term = np.ones_like(x)
    done = np.zeros(x.shape, dtype=bool)
    for n in range(1, _SERIES_TERMS):
        term = term * (n - b) * x / n
        step = term / (a + n)
        total = np.where(done, total, total + step)
        done = done | (np.abs(step) <= 1e-17 * np.abs(total)) | (term == 0)
        if np.all(done):
            break
    with np.errstate(divide="ignore"):
        log_x = np.log(x)
    out = a * log_x + _log_beta_front(a, b) + np.log1p(a * total)
    return np.where(done, out, np.nan)


#: Below this multiple of min(1, b), ``_log_beta_front`` takes its Taylor
#: series in ``a``: the terms after a^4 are below 1e-16 of it.
_FRONT_SERIES = 1e-4


def _log_beta_front(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    r""":math:`\ln \Gamma(a + b) - \ln \Gamma(b) - \ln \Gamma(1 + a)`,
    to its own relative precision at a small ``a``, where it is of size
    ``a`` and the gammas' rounding (1e-18 in the Stirling remainders of
    ``log_gamma_ratio``) is not: :math:`a (\psi(b) + \gamma) + \sum_{k
    \geq 2} a^k [\psi^{(k - 1)}(b) / k! - (-1)^k \zeta(k) / k]`, to
    :math:`a^4`, for an ``a`` below 1e-4 of min(1, b)."""
    tiny = a < _FRONT_SERIES * np.minimum(1.0, b)
    a_t = np.where(tiny, a, 0.0)
    b_t = np.where(tiny, b, 1.0)
    series = a_t * (
        (_sc_digamma(b_t) + _EULER)
        + a_t
        * (
            (_sc_polygamma(1, b_t) - _ZETA2) / 2.0
            + a_t
            * (
                (_sc_polygamma(2, b_t) + 2.0 * _ZETA3) / 6.0
                + a_t * (_sc_polygamma(3, b_t) - 6.0 * _ZETA4) / 24.0
            )
        )
    )
    a_g = np.where(tiny, 1.0, a)
    b_g = np.where(tiny, 1.0, b)
    general = log_gamma_ratio(b_g, a_g) - _lngamma1p(a_g)
    return np.where(tiny, series, general)


def _beta_cf_log(
    a: npt.NDArray,
    b: npt.NDArray,
    x: npt.NDArray,
    xc: npt.NDArray,
    terms: int | None = None,
) -> npt.NDArray:
    """log I_x(a, b) from ``beta_cf``, for x below (a + 1) / (a + b + 2),
    with ``xc = 1 - x``: the log of the front factor x^a (1 - x)^b /
    (a B(a, b)) stays finite where I_x itself underflows. Each log is
    taken from whichever of x and xc is below 1/2 (one of them is the
    caller's own, exact, argument): log(1 - p) of a rounded 1 - p, times
    k = 2.7e9, lost 5e-8 of a Negative Binomial's sf (#458). ``nan``
    where ``beta_cf`` has not converged in ``terms`` terms."""
    with np.errstate(divide="ignore"):
        log_x = np.where(x < 0.5, np.log(x), np.log1p(-xc))
        log_xc = np.where(xc < 0.5, np.log(xc), np.log1p(-x))
    front = a * log_x + b * log_xc - betaln_accurate(a, b) - np.log(a)
    return front + np.log(beta_cf(a, b, x, terms))


def _log1mexp_neg(v: npt.NDArray) -> npt.NDArray:
    """log(1 - exp(v)) for v <= 0, from whichever of expm1 and log1p keeps
    its digits."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > -_LN2, np.log(-np.expm1(v)), np.log1p(-np.exp(v)))


def _beta_logs(a: Boxable, b: Boxable, x: Boxable, upper: bool) -> Boxable:
    """log I_x(a, b) (``upper`` False) or log(1 - I_x(a, b)), accurate
    wherever the value is representable, and exact at the edges.

    Where both tails are above 1e-3, scipy's ``betainc`` and ``betaincc``
    are used as they are. Below that the small tail is taken from its own
    continued fraction, in logs, and the large one as log1p of minus it:
    ``log(betainc)`` was capped at -708 where it underflows (#443), the
    log of a probability near 1 lost its complement (#442), and scipy's
    small tail itself can be off (``betaincc(1e-3, 1e3, 0.1)`` is 1.74e-51
    for 1.34e-51, #458). Where that continued fraction does not converge
    in ``_CF_TAIL_TERMS`` terms (its x within about 1e-3 of 1, as it is
    where the other side's first shape is tiny), the other side comes
    from its power series (``_beta_series_log``) and the small tail from
    it: the fraction ran to 100,000 terms there, 4.6 s a call, and its
    value was 40% off (#520)."""
    a_arr, b_arr, x_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(x, dtype=float),
    )
    shape = x_arr.shape
    inside = (x_arr > 0.0) & (x_arr < 1.0)
    xs = np.where(inside, x_arr, 0.5)
    # The side asked for everywhere, the other where it can be the small
    # tail (``_where_near_one``): the two sides' only other use is to
    # choose the small one, and there a 1 chooses as the value would.
    args = (a_arr, b_arr, xs)
    if upper:
        q = _sc_betaincc(*args)
        p = _where_near_one(_sc_betainc, args, q)
    else:
        p = _sc_betainc(*args)
        q = _where_near_one(_sc_betaincc, args, p)
    with np.errstate(divide="ignore"):
        log_p = np.array(np.log(p), dtype=float, ndmin=1)
        log_q = np.array(np.log(q), dtype=float, ndmin=1)
    a_arr, b_arr, xs, p, q, inside, x_arr = (
        np.atleast_1d(v) for v in (a_arr, b_arr, xs, p, q, inside, x_arr)
    )
    tail = inside & (np.minimum(p, q) < _COMPLEMENT)
    if np.any(tail):
        at, bt, xt = a_arr[tail], b_arr[tail], xs[tail]
        low = p[tail] <= q[tail]
        # the small tail from its own side, the large one from it
        # (shapes and x of the small tail's side)
        a_s = np.where(low, at, bt)
        b_s = np.where(low, bt, at)
        x_s = np.where(low, xt, 1.0 - xt)
        xc_s = np.where(low, 1.0 - xt, xt)
        with np.errstate(invalid="ignore"):
            small = _beta_cf_log(a_s, b_s, x_s, xc_s, _CF_TAIL_TERMS)
        far = np.isnan(small)
        if np.any(far):
            # The small tail's continued fraction has not converged: its x
            # is near 1, far above its region, which happens where the
            # other side is near 1 because its first shape is small (a
            # NegativeBinomial r of 1e-172). That side from its power
            # series, and the small tail from it.
            large_far = _beta_series_log(b_s[far], a_s[far], xc_s[far])
            small[far] = _log1mexp_neg(large_far)
        large = _log1mexp_neg(small)
        log_p[tail] = np.where(low, small, large)
        log_q[tail] = np.where(low, large, small)
    out = log_q if upper else log_p
    at_zero, at_one = (0.0, -np.inf) if upper else (-np.inf, 0.0)
    out = np.where(x_arr <= 0.0, at_zero, np.where(x_arr >= 1.0, at_one, out))
    return out.reshape(shape) if shape else float(out[0])


#: The most terms of ``_beta_cf_shape_grad``: on its side of the
#: incomplete beta it converges in tens of terms for moderate shapes, and
#: in about sqrt(max(a, b)) for large ones.
_CF_GRAD_TERMS = 3000


def _beta_cf_shape_grad(
    a: npt.NDArray, b: npt.NDArray, x: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The derivatives in ``a`` and ``b`` of ``log beta_cf(a, b, x)``, by
    forward differentiation of its Lentz recursion (Boik and
    Robison-Cox 1998): each convergent's factors carry their derivatives
    with them. For ``x`` below (a + 1) / (a + b + 2), where the fraction
    converges; ``nan`` where it has not, value and derivatives, in
    ``_CF_GRAD_TERMS`` terms."""
    tiny = 1e-300
    one = np.ones_like(x)
    zero = np.zeros_like(x)
    # c and d with their derivatives in a and b
    c, c_a, c_b = one, zero, zero
    D = 1.0 - (a + b) * x / (a + 1.0)
    D_a = -x * (1.0 - b) / (a + 1.0) ** 2
    D_b = -x / (a + 1.0)
    D = np.where(np.abs(D) < tiny, tiny, D)
    d = 1.0 / D
    d_a, d_b = -D_a * d * d, -D_b * d * d
    g_a, g_b = D_a * -d, D_b * -d  # of log h
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, _CF_GRAD_TERMS):
        k1 = (a + 2 * m - 1) * (a + 2 * m)
        aa1 = m * (b - m) * x / k1
        aa1_a = -aa1 * (1.0 / (a + 2 * m - 1) + 1.0 / (a + 2 * m))
        aa1_b = m * x / k1
        k2 = (a + 2 * m) * (a + 2 * m + 1)
        aa2 = -(a + m) * (a + b + m) * x / k2
        aa2_a = aa2 * (
            1.0 / (a + m)
            + 1.0 / (a + b + m)
            - 1.0 / (a + 2 * m)
            - 1.0 / (a + 2 * m + 1)
        )
        aa2_b = aa2 / (a + b + m)
        step = one
        step_a, step_b = zero, zero
        for aa, aa_a, aa_b in ((aa1, aa1_a, aa1_b), (aa2, aa2_a, aa2_b)):
            D = 1.0 + aa * d
            D_a = aa_a * d + aa * d_a
            D_b = aa_b * d + aa * d_b
            D = np.where(np.abs(D) < tiny, tiny, D)
            d = 1.0 / D
            d_a, d_b = -D_a * d * d, -D_b * d * d
            C = 1.0 + aa / c
            C_a = aa_a / c - aa * c_a / (c * c)
            C_b = aa_b / c - aa * c_b / (c * c)
            C = np.where(np.abs(C) < tiny, tiny, C)
            c, c_a, c_b = C, C_a, C_b
            # log h gains log d + log c
            inc_a = -D_a / D + C_a / C
            inc_b = -D_b / D + C_b / C
            g_a = np.where(done, g_a, g_a + inc_a)
            g_b = np.where(done, g_b, g_b + inc_b)
            step = step * d * c
            step_a, step_b = step_a + inc_a, step_b + inc_b
        done = done | (
            (np.abs(step - 1.0) < 1e-16)
            & (np.abs(step_a) <= 1e-16 * (1.0 + np.abs(g_a)))
            & (np.abs(step_b) <= 1e-16 * (1.0 + np.abs(g_b)))
        )
        if np.all(done):
            break
    return np.where(done, g_a, np.nan), np.where(done, g_b, np.nan)


#: The last ``_beta_log_shape_grad`` computed, as ``(key, value)``: a
#: gradient asks for the derivative in ``a`` and then in ``b`` at the same
#: point, and both come from one pass.
_LAST_SHAPE_GRAD: list = [None, None]


def _beta_log_shape_grad(
    a: Boxable, b: Boxable, x: Boxable, upper: bool
) -> tuple[Any, Any]:
    """:func:`_beta_log_shape_grad_uncached`, remembering the last
    point."""
    arrays = [np.asarray(v, dtype=float) for v in (a, b, x)]
    key = (upper,) + tuple((v.shape, v.tobytes()) for v in arrays)
    if _LAST_SHAPE_GRAD[0] != key:
        _LAST_SHAPE_GRAD[1] = _beta_log_shape_grad_uncached(
            arrays[0], arrays[1], arrays[2], upper
        )
        _LAST_SHAPE_GRAD[0] = key
    g_a, g_b = _LAST_SHAPE_GRAD[1]
    return np.copy(g_a), np.copy(g_b)


def _beta_log_shape_grad_uncached(
    a: Boxable, b: Boxable, x: Boxable, upper: bool
) -> tuple[Any, Any]:
    """The derivatives in ``a`` and ``b`` of :func:`_beta_logs` (``log
    I_x(a, b)``, or ``log(1 - I_x(a, b))`` where ``upper``), analytic
    (#621): the side of the incomplete beta whose continued fraction
    converges, ``I_x(a, b)`` for x below (a + 1) / (a + b + 2) and ``1 -
    I_{1-x}(b, a)`` above, is differentiated through its front factor
    (digamma functions) and its fraction (``_beta_cf_shape_grad``); the
    other side's log follows from it by ``d log(1 - s) = -s / (1 - s) d
    log s``. Where that fraction does not converge (a shape below about
    1e-150, where the values come from ``_beta_series_log``), the
    derivatives are the five-point differences of the values they
    replace."""
    a_arr, b_arr, x_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(x, dtype=float),
    )
    shape = x_arr.shape
    a_arr, b_arr, x_arr = (np.atleast_1d(v) for v in (a_arr, b_arr, x_arr))
    inside = (x_arr > 0.0) & (x_arr < 1.0)
    xs = np.where(inside, x_arr, 0.5)
    # the converging side: its shapes and x (and 1 - x, exact)
    low = xs < (a_arr + 1.0) / (a_arr + b_arr + 2.0)
    a_s = np.where(low, a_arr, b_arr)
    b_s = np.where(low, b_arr, a_arr)
    x_s = np.where(low, xs, 1.0 - xs)
    xc_s = np.where(low, 1.0 - xs, xs)
    with np.errstate(all="ignore"):
        cf_a, cf_b = _beta_cf_shape_grad(a_s, b_s, x_s)
        log_x = np.where(x_s < 0.5, np.log(x_s), np.log1p(-xc_s))
        log_xc = np.where(xc_s < 0.5, np.log(xc_s), np.log1p(-x_s))
        psi_ab = _sc_digamma(a_s + b_s)
        # d/da_s and d/db_s of log I on the converging side
        s_a = log_x - _sc_digamma(a_s) + psi_ab - 1.0 / a_s + cf_a
        s_b = log_xc - _sc_digamma(b_s) + psi_ab + cf_b
        # in the caller's shapes
        g_a = np.where(low, s_a, s_b)
        g_b = np.where(low, s_b, s_a)
        # the converging side's log value; the asked-for side's from it
        log_s = np.asarray(_beta_logs(a_s, b_s, x_s, upper=False))
        log_other = _log1mexp_neg(log_s)
        factor = -np.exp(log_s - log_other)
    # asked for the converging side: its derivative; else the other's
    same = low != upper
    g_a = np.where(same, g_a, factor * g_a)
    g_b = np.where(same, g_b, factor * g_b)
    bad = inside & ~(np.isfinite(g_a) & np.isfinite(g_b))
    if np.any(bad):
        raw = _betainccln_raw if upper else _betaincln_raw
        ab, bb, xb = a_arr[bad], b_arr[bad], x_arr[bad]
        g_a[bad] = _cdiff2_a(raw, ab, bb, xb)
        g_b[bad] = _cdiff2_b(raw, ab, bb, xb)
    # at the edges the log is constant (0 or -inf) in the shapes
    g_a = np.where(inside, g_a, 0.0)
    g_b = np.where(inside, g_b, 0.0)
    if not shape:
        return float(g_a[0]), float(g_b[0])
    return g_a.reshape(shape), g_b.reshape(shape)


def _make_analytic_dab_primitives(
    f: Callable, grad: Callable
) -> tuple[Callable, Callable]:
    """As :func:`_make_dab_primitives`, with the first derivatives from
    ``grad(a, b, x) -> (df/da, df/db)`` rather than differences of ``f``;
    their own VJPs (the Hessian's second derivatives) are the five-point
    differences of ``grad``."""

    @primitive
    def f_da(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
        return grad(a, b, x)[0]

    @primitive
    def f_db(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
        return grad(a, b, x)[1]

    def _vals(
        a: Boxable, b: Boxable, x: Boxable
    ) -> tuple[Boxable, Boxable, Boxable]:
        return getval(a), getval(b), getval(x)

    def vjp(which: int, arg: int) -> Callable:
        def make(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
            av, bv, xv = _vals(a, b, x)
            vals = [av, bv, xv]

            def moved(v: Boxable) -> Boxable:
                args = list(vals)
                args[arg] = v
                return grad(*args)[which]

            second = _cdiff(moved, vals[arg])
            return unbroadcast_f((a, b, x)[arg], lambda g: getval(g) * second)

        return make

    defvjp(f_da, vjp(0, 0), vjp(0, 1), vjp(0, 2))
    defvjp(f_db, vjp(1, 0), vjp(1, 1), vjp(1, 2))
    return f_da, f_db


def _betaincln_raw(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _beta_logs(a, b, x, upper=False)


@primitive
def betaincln(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _betaincln_raw(a, b, x)


_betaincln_da, _betaincln_db = _make_analytic_dab_primitives(
    _betaincln_raw,
    lambda a, b, x: _beta_log_shape_grad(a, b, x, upper=False),
)

defvjp(
    betaincln,
    lambda ans, a, b, x: unbroadcast_f(
        a, lambda g: g * _betaincln_da(a, b, x)
    ),
    lambda ans, a, b, x: unbroadcast_f(
        b, lambda g: g * _betaincln_db(a, b, x)
    ),
    # d/dx of log B = (dB/dx)/B; ans = log B
    lambda ans, a, b, x: unbroadcast_f(
        x,
        lambda g: g
        * anp.exp(
            (a - 1) * anp.log(x)
            + (b - 1) * anp.log(1 - x)
            - _ag_betaln(a, b)
            - ans
        ),
    ),
)

# ---------------------------------------------------------------------------
# betainccln  —  log(1 - B(a, b; x)), the log of the upper tail
# ---------------------------------------------------------------------------


def _betainccln_raw(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _beta_logs(a, b, x, upper=True)


@primitive
def betainccln(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _betainccln_raw(a, b, x)


_betainccln_da, _betainccln_db = _make_analytic_dab_primitives(
    _betainccln_raw,
    lambda a, b, x: _beta_log_shape_grad(a, b, x, upper=True),
)

defvjp(
    betainccln,
    lambda ans, a, b, x: unbroadcast_f(
        a, lambda g: g * _betainccln_da(a, b, x)
    ),
    lambda ans, a, b, x: unbroadcast_f(
        b, lambda g: g * _betainccln_db(a, b, x)
    ),
    # d/dx of log(1 - B) = -(dB/dx)/(1 - B); ans = log(1 - B)
    lambda ans, a, b, x: unbroadcast_f(
        x,
        lambda g: -g
        * anp.exp(
            (a - 1) * anp.log(x)
            + (b - 1) * anp.log(1 - x)
            - _ag_betaln(a, b)
            - ans
        ),
    ),
)
