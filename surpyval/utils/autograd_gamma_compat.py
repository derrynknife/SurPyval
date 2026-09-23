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
from autograd.scipy.special import betaln as _ag_betaln
from autograd.scipy.special import gammaln as _ag_gammaln
from autograd.tracer import getval
from scipy.special import betainc as _sc_betainc
from scipy.special import gammainc as _sc_gammainc
from scipy.special import gammaincc as _sc_gammaincc

# The value-or-box union the distributions use (see parametric_fitter):
# every boundary here may see a plain numpy value or an ArrayBox.
Boxable = npt.NDArray | float | ArrayBox

_LOG_EPS = 1e-35
_EPS_H = np.finfo(float).eps ** (1.0 / 3.0)


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
        return lambda g: (getval(g) * d2).sum()

    def vjp_x(ans: Boxable, a: Boxable, x: Boxable) -> Callable:
        av, xv = getval(a), getval(x)
        mixed = _cdiff(lambda xx: _cdiff1(f, av, xx), xv)
        return lambda g: getval(g) * mixed

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
        return lambda g: (getval(g) * d2).sum()

    def da_vjp_b(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff(lambda bb: _cdiff2_a(f, av, bb, xv), bv)
        return lambda g: (getval(g) * d2).sum()

    def da_vjp_x(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        mixed = _cdiff(lambda xx: _cdiff2_a(f, av, bv, xx), xv)
        return lambda g: getval(g) * mixed

    def db_vjp_a(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff(lambda aa: _cdiff2_b(f, aa, bv, xv), av)
        return lambda g: (getval(g) * d2).sum()

    def db_vjp_b(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        d2 = _cdiff_second(lambda bb: f(av, bb, xv), bv)
        return lambda g: (getval(g) * d2).sum()

    def db_vjp_x(ans: Boxable, a: Boxable, b: Boxable, x: Boxable) -> Callable:
        av, bv, xv = _vals(a, b, x)
        mixed = _cdiff(lambda xx: _cdiff2_b(f, av, bv, xx), xv)
        return lambda g: getval(g) * mixed

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
    lambda ans, a, x: lambda g: anp.sum(g * _gammainc_da(a, x)),
    # d/dx: analytical; anp.log handles ArrayBox x for correct Hessian
    lambda ans, a, x: lambda g: g
    * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a)),
)

# ---------------------------------------------------------------------------
# gammaincln  —  log P(a, x)
# ---------------------------------------------------------------------------


def _gammaincln_raw(a: Boxable, x: Boxable) -> Boxable:
    return np.log(np.clip(_sc_gammainc(a, x), _LOG_EPS, np.inf))


@primitive
def gammaincln(a: Boxable, x: Boxable) -> Boxable:
    return _gammaincln_raw(a, x)


_gammaincln_da = _make_da_primitive(_gammaincln_raw)

defvjp(
    gammaincln,
    lambda ans, a, x: lambda g: anp.sum(g * _gammaincln_da(a, x)),
    # d/dx of log P = (dP/dx)/P; ans = log P avoids recomputing P
    lambda ans, a, x: lambda g: g
    * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
)

# ---------------------------------------------------------------------------
# gammainccln  —  log Q(a, x) = log(1 − P(a, x))
# ---------------------------------------------------------------------------


def _gammainccln_raw(a: Boxable, x: Boxable) -> Boxable:
    return np.log(np.clip(_sc_gammaincc(a, x), _LOG_EPS, np.inf))


@primitive
def gammainccln(a: Boxable, x: Boxable) -> Boxable:
    return _gammainccln_raw(a, x)


_gammainccln_da = _make_da_primitive(_gammainccln_raw)

defvjp(
    gammainccln,
    lambda ans, a, x: lambda g: anp.sum(g * _gammainccln_da(a, x)),
    # d/dx of log Q = -(dP/dx)/Q; negated, ans = log Q
    lambda ans, a, x: lambda g: g
    * -anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
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
    lambda ans, a, b, x: lambda g: anp.sum(g * _betainc_da(a, b, x)),
    lambda ans, a, b, x: lambda g: anp.sum(g * _betainc_db(a, b, x)),
    # d/dx: x^(a-1)*(1-x)^(b-1)/B(a,b); anp handles ArrayBox x
    lambda ans, a, b, x: lambda g: g
    * anp.exp(
        (a - 1) * anp.log(x) + (b - 1) * anp.log(1 - x) - _ag_betaln(a, b)
    ),
)

# ---------------------------------------------------------------------------
# betaincln  —  log B(a, b; x)
# ---------------------------------------------------------------------------


def _betaincln_raw(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return np.log(np.clip(_sc_betainc(a, b, x), _LOG_EPS, np.inf))


@primitive
def betaincln(a: Boxable, b: Boxable, x: Boxable) -> Boxable:
    return _betaincln_raw(a, b, x)


_betaincln_da, _betaincln_db = _make_dab_primitives(_betaincln_raw)

defvjp(
    betaincln,
    lambda ans, a, b, x: lambda g: anp.sum(g * _betaincln_da(a, b, x)),
    lambda ans, a, b, x: lambda g: anp.sum(g * _betaincln_db(a, b, x)),
    # d/dx of log B = (dB/dx)/B; ans = log B
    lambda ans, a, b, x: lambda g: g
    * anp.exp(
        (a - 1) * anp.log(x)
        + (b - 1) * anp.log(1 - x)
        - _ag_betaln(a, b)
        - ans
    ),
)
