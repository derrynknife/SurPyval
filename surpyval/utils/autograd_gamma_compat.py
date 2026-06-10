"""
Pure-NumPy/SciPy autograd primitives for incomplete gamma and beta functions.

Replaces the abandoned autograd-gamma package. VJP approach:

  x-derivatives (analytical):
    Use autograd.numpy ops so that ArrayBox inputs (which appear when autograd
    computes Hessians by tracing through the backward pass) are handled
    correctly, giving correct second-order derivatives for the x (rate) param.

  shape-parameter derivatives (numerical central difference):
    Use getval() to strip ArrayBox before all scipy calls. The returned value
    is plain numpy (via getval(g) + ndarray.sum()), so the Hessian
    contribution from shape-parameter paths is zero — an acceptable
    approximation that matches the original autograd-gamma behaviour.
    This also avoids autograd 1.8's np.sum VJP bug with the `out` kwarg.
"""

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive
from autograd.scipy.special import betaln as _ag_betaln
from autograd.scipy.special import gammaln as _ag_gammaln
from autograd.tracer import getval
from scipy.special import betainc as _sc_betainc
from scipy.special import gammainc as _sc_gammainc
from scipy.special import gammaincc as _sc_gammaincc

_LOG_EPS = 1e-35
_EPS_H = np.finfo(float).eps ** (1.0 / 3.0)


def _step(v):
    """Exact floating-point step size for central differences."""
    h = np.maximum(np.abs(v) * _EPS_H, 1e-7)
    return (v + h) - v


def _cdiff1(f, a, x):
    """5th-order central diff of f(a, x) w.r.t. a.

    Both a and x must be plain numpy values — call with getval().
    """
    h = _step(a)
    return (
        -f(a + 2 * h, x) + 8 * f(a + h, x) - 8 * f(a - h, x) + f(a - 2 * h, x)
    ) / (12 * h)


def _cdiff2_a(f, a, b, x):
    """5th-order central diff of f(a, b, x) w.r.t. a — all args plain numpy."""
    h = _step(a)
    return (
        -f(a + 2 * h, b, x)
        + 8 * f(a + h, b, x)
        - 8 * f(a - h, b, x)
        + f(a - 2 * h, b, x)
    ) / (12 * h)


def _cdiff2_b(f, a, b, x):
    """5th-order central diff of f(a, b, x) w.r.t. b — all args plain numpy."""
    h = _step(b)
    return (
        -f(a, b + 2 * h, x)
        + 8 * f(a, b + h, x)
        - 8 * f(a, b - h, x)
        + f(a, b - 2 * h, x)
    ) / (12 * h)


# ---------------------------------------------------------------------------
# gammainc  —  P(a, x), regularised lower incomplete gamma
# ---------------------------------------------------------------------------


@primitive
def gammainc(a, x):
    return _sc_gammainc(a, x)


defvjp(
    gammainc,
    # d/da: numerical; getval(g) + .sum() avoids autograd 1.8 np.sum bug
    lambda ans, a, x: lambda g: (
        getval(g) * _cdiff1(_sc_gammainc, getval(a), getval(x))
    ).sum(),
    # d/dx: analytical; anp.log handles ArrayBox x for correct Hessian
    lambda ans, a, x: lambda g: g
    * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a)),
)

# ---------------------------------------------------------------------------
# gammaincln  —  log P(a, x)
# ---------------------------------------------------------------------------


def _gammaincln_raw(a, x):
    return np.log(np.clip(_sc_gammainc(a, x), _LOG_EPS, np.inf))


@primitive
def gammaincln(a, x):
    return _gammaincln_raw(a, x)


defvjp(
    gammaincln,
    lambda ans, a, x: lambda g: (
        getval(g) * _cdiff1(_gammaincln_raw, getval(a), getval(x))
    ).sum(),
    # d/dx of log P = (dP/dx)/P; ans = log P avoids recomputing P
    lambda ans, a, x: lambda g: g
    * anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
)

# ---------------------------------------------------------------------------
# gammainccln  —  log Q(a, x) = log(1 − P(a, x))
# ---------------------------------------------------------------------------


def _gammainccln_raw(a, x):
    return np.log(np.clip(_sc_gammaincc(a, x), _LOG_EPS, np.inf))


@primitive
def gammainccln(a, x):
    return _gammainccln_raw(a, x)


defvjp(
    gammainccln,
    lambda ans, a, x: lambda g: (
        getval(g) * _cdiff1(_gammainccln_raw, getval(a), getval(x))
    ).sum(),
    # d/dx of log Q = -(dP/dx)/Q; negated, ans = log Q
    lambda ans, a, x: lambda g: g
    * -anp.exp(-x + anp.log(x) * (a - 1) - _ag_gammaln(a) - ans),
)

# ---------------------------------------------------------------------------
# betainc  —  B(a, b; x), regularised incomplete beta
# ---------------------------------------------------------------------------


@primitive
def betainc(a, b, x):
    return _sc_betainc(a, b, x)


defvjp(
    betainc,
    lambda ans, a, b, x: lambda g: (
        getval(g) * _cdiff2_a(_sc_betainc, getval(a), getval(b), getval(x))
    ).sum(),
    lambda ans, a, b, x: lambda g: (
        getval(g) * _cdiff2_b(_sc_betainc, getval(a), getval(b), getval(x))
    ).sum(),
    # d/dx: x^(a-1)*(1-x)^(b-1)/B(a,b); anp handles ArrayBox x
    lambda ans, a, b, x: lambda g: g
    * anp.exp(
        (a - 1) * anp.log(x) + (b - 1) * anp.log(1 - x) - _ag_betaln(a, b)
    ),
)

# ---------------------------------------------------------------------------
# betaincln  —  log B(a, b; x)
# ---------------------------------------------------------------------------


def _betaincln_raw(a, b, x):
    return np.log(np.clip(_sc_betainc(a, b, x), _LOG_EPS, np.inf))


@primitive
def betaincln(a, b, x):
    return _betaincln_raw(a, b, x)


defvjp(
    betaincln,
    lambda ans, a, b, x: lambda g: (
        getval(g) * _cdiff2_a(_betaincln_raw, getval(a), getval(b), getval(x))
    ).sum(),
    lambda ans, a, b, x: lambda g: (
        getval(g) * _cdiff2_b(_betaincln_raw, getval(a), getval(b), getval(x))
    ).sum(),
    # d/dx of log B = (dB/dx)/B; ans = log B
    lambda ans, a, b, x: lambda g: g
    * anp.exp(
        (a - 1) * anp.log(x)
        + (b - 1) * anp.log(1 - x)
        - _ag_betaln(a, b)
        - ans
    ),
)
