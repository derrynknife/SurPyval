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
"""

from typing import Callable

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
from scipy.special import gammainc as _sc_gammainc
from scipy.special import gammaincc as _sc_gammaincc
from scipy.special import gammaln as _sc_gammaln

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


def _make_da_primitive(f: Callable) -> Callable:
    """Traced first derivative w.r.t. the shape parameter of f(a, x).

    Returns a primitive ``f_da(a, x) = df/da`` whose own VJPs are the
    numerical second derivatives d2f/da2 and d2f/dadx, so a Hessian pass
    through ``f_da`` picks up the correct curvature (further orders cut).
    """

    @primitive
    def f_da(a: Boxable, x: Boxable) -> Boxable:
        return _cdiff1(f, a, x)

    def vjp_a(ans: Boxable, a: Boxable, x: Boxable) -> Callable:
        av, xv = getval(a), getval(x)
        d2 = _cdiff_second(lambda aa: f(aa, xv), av)
        return unbroadcast_f(a, lambda g: getval(g) * d2)

    def vjp_x(ans: Boxable, a: Boxable, x: Boxable) -> Callable:
        av, xv = getval(a), getval(x)
        mixed = _cdiff(lambda xx: _cdiff1(f, av, xx), xv)
        return unbroadcast_f(x, lambda g: getval(g) * mixed)

    defvjp(f_da, vjp_a, vjp_x)
    return f_da


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


_gammainc_da = _make_da_primitive(_sc_gammainc)

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
    q = _sc_gammaincc(a_arr, x_arr)
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


_gammaincln_da = _make_da_primitive(_gammaincln_raw)

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
    p = _sc_gammainc(a_arr, x_arr)
    out = np.where(p < _COMPLEMENT, np.log1p(-np.minimum(p, _COMPLEMENT)), out)
    # Q(a, x) underflows past ~1e-308 (x of ~700 for a small a), where
    # a clipped log would cap the Gamma cumulative hazard. There x >> a
    # and the asymptotic series
    #   log Q = (a-1) log x - x - log Gamma(a)
    #           + log(1 + (a-1)/x + (a-1)(a-2)/x^2 + ...)
    # is accurate; it is summed until the terms stop shrinking.
    tail = (q < 1e-280) & (x_arr > a_arr)
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
    return out if out.ndim else float(out)


@primitive
def gammainccln(a: Boxable, x: Boxable) -> Boxable:
    return _gammainccln_raw(a, x)


_gammainccln_da = _make_da_primitive(_gammainccln_raw)

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
    # z >= 10 here for every y > 0.
    z = anp.maximum(z, _ASYMPTOTIC_FROM)
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
    a: npt.ArrayLike, b: npt.ArrayLike, x: npt.ArrayLike
) -> npt.NDArray:
    """The continued fraction of the incomplete beta (Lentz), such that
    I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * beta_cf(a, b, x); it
    converges fast for x below (a + 1) / (a + b + 2)."""
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
    for m in range(1, 100000):
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
    return h


def _beta_cf_log(
    a: npt.NDArray, b: npt.NDArray, x: npt.NDArray, xc: npt.NDArray
) -> npt.NDArray:
    """log I_x(a, b) from ``beta_cf``, for x below (a + 1) / (a + b + 2),
    with ``xc = 1 - x``: the log of the front factor x^a (1 - x)^b /
    (a B(a, b)) stays finite where I_x itself underflows. Each log is
    taken from whichever of x and xc is below 1/2 (one of them is the
    caller's own, exact, argument): log(1 - p) of a rounded 1 - p, times
    k = 2.7e9, lost 5e-8 of a Negative Binomial's sf (#458)."""
    with np.errstate(divide="ignore"):
        log_x = np.where(x < 0.5, np.log(x), np.log1p(-xc))
        log_xc = np.where(xc < 0.5, np.log(xc), np.log1p(-x))
    front = a * log_x + b * log_xc - betaln_accurate(a, b) - np.log(a)
    return front + np.log(beta_cf(a, b, x))


def _beta_logs(a: Boxable, b: Boxable, x: Boxable, upper: bool) -> Boxable:
    """log I_x(a, b) (``upper`` False) or log(1 - I_x(a, b)), accurate
    wherever the value is representable, and exact at the edges.

    Where both tails are above 1e-3, scipy's ``betainc`` and ``betaincc``
    are used as they are. Below that the small tail is taken from its own
    continued fraction, in logs, and the large one as log1p of minus it:
    ``log(betainc)`` was capped at -708 where it underflows (#443), the
    log of a probability near 1 lost its complement (#442), and scipy's
    small tail itself can be off (``betaincc(1e-3, 1e3, 0.1)`` is 1.74e-51
    for 1.34e-51, #458)."""
    a_arr, b_arr, x_arr = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(x, dtype=float),
    )
    shape = x_arr.shape
    inside = (x_arr > 0.0) & (x_arr < 1.0)
    xs = np.where(inside, x_arr, 0.5)
    p = _sc_betainc(a_arr, b_arr, xs)
    q = _sc_betaincc(a_arr, b_arr, xs)
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
        small = np.empty_like(xt)
        xl, xh = xt[low], xt[~low]
        small[low] = _beta_cf_log(at[low], bt[low], xl, 1.0 - xl)
        small[~low] = _beta_cf_log(bt[~low], at[~low], 1.0 - xh, xh)
        large = np.where(
            small > -_LN2,
            np.log(-np.expm1(small)),
            np.log1p(-np.exp(small)),
        )
        log_p[tail] = np.where(low, small, large)
        log_q[tail] = np.where(low, large, small)
    out = log_q if upper else log_p
    at_zero, at_one = (0.0, -np.inf) if upper else (-np.inf, 0.0)
    out = np.where(x_arr <= 0.0, at_zero, np.where(x_arr >= 1.0, at_one, out))
    return out.reshape(shape) if shape else float(out[0])


def _betaincln_raw(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _beta_logs(a, b, x, upper=False)


@primitive
def betaincln(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _betaincln_raw(a, b, x)


_betaincln_da, _betaincln_db = _make_dab_primitives(_betaincln_raw)

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


_betainccln_da, _betainccln_db = _make_dab_primitives(_betainccln_raw)

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
