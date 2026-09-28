"""High-precision references for the distribution functions in the tails
(#398), for ``surpyval/tests/reference/test_tails.py``.

For every closed-form univariate distribution in SurPyval this computes,
with mpmath at 50 significant digits, the true ``sf``, ``ff``, ``df``,
``hf``, ``Hf`` and the logs the likelihood uses (``log_sf``, ``log_ff``,
``log_df``) on a grid of extreme parameters and times, and the true ``qf``
on a grid of probabilities, and writes them to
``surpyval/tests/reference/data/tails_mpmath.json``. The tests compare
SurPyval with the stored values, so they need no mpmath.

The grid, per parameter set:

* the support edges;
* the times where the CDF is 1e-300, 1e-100, 1e-30 ("deep left"), 1e-8,
  1e-3 ("left"), 0.5 ("centre"), where the survival function is 1e-3,
  1e-8 ("right"), 1e-30, 1e-100, 1e-300 ("deep right"), and where either
  is 1e-400 ("beyond left" / "beyond right": the value underflows a
  double, its log does not);
* for the discrete distributions, integer times found the same way and
  ``k`` of 1e6 and 1e12 ("large k");
* ``qf`` at 0, 1e-300, 1e-100 ("p tiny"), 1e-30, 1e-8 ("p small"), 0.5,
  1 - 1e-8, the double just below 1 ("p near 1") and 1.

Each time is rounded to a double and the reference is the value *at that
double*, so the only error left in SurPyval's value is its own. Times
that are not representable (a Weibull with shape 1e-3 reaches a CDF of
1e-8 only at 1e-8000) are left out.

Next to each value is its sensitivity to the rounding of the inputs,
``sum_i |d f / d log(theta_i)|`` over the time (or probability) and the
parameters. A double-precision implementation cannot do better than
about ``eps`` times that, whatever its formula, because the arguments'
own rounding (``x / alpha``, ``x - mu``) is amplified by it; the test
allows 64 times that as its absolute floor. For ``qf`` the probability
enters as ``u`` below 0.5 and as ``1 - u`` above it (both exact).

Run from the repository root, with mpmath installed (it is not a SurPyval
dependency; use a separate virtual environment)::

    python -m venv --system-site-packages /tmp/tails_venv
    /tmp/tails_venv/bin/pip install mpmath
    /tmp/tails_venv/bin/python scripts/reference/tails_mpmath.py

Re-running reproduces the file byte for byte (a fixed seed chooses the
points that are re-checked at 80 digits).
"""

import json
import math
import platform
import random
import sys
from pathlib import Path

import mpmath
from mpmath import mp, mpf

DATA = (
    Path(__file__).resolve().parents[2]
    / "surpyval"
    / "tests"
    / "reference"
    / "data"
)

DPS = 50
CHECK_DPS = 80
CHECK_FRACTION = 0.1
FUNCTIONS = ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df")

# (regime, side, target): the time where ff (side "left") or sf (side
# "right") equals the target.
TARGETS = [
    ("beyond left", "left", "1e-400"),
    ("deep left", "left", "1e-300"),
    ("deep left", "left", "1e-100"),
    ("deep left", "left", "1e-30"),
    ("left", "left", "1e-8"),
    ("left", "left", "1e-3"),
    ("centre", "left", "0.5"),
    ("right", "right", "1e-3"),
    ("right", "right", "1e-8"),
    ("deep right", "right", "1e-30"),
    ("deep right", "right", "1e-100"),
    ("deep right", "right", "1e-300"),
    ("beyond right", "right", "1e-400"),
]

QF_POINTS = [
    ("p edge", 0.0),
    ("p tiny", 1e-300),
    ("p tiny", 1e-100),
    ("p small", 1e-30),
    ("p small", 1e-8),
    ("p centre", 0.5),
    ("p near 1", 1 - 1e-8),
    ("p near 1", 0.9999999999999999),
    ("p edge", 1.0),
]

LARGE_K = (10**6, 10**12)
MAX_INT = 2**53

SHAPES = (1e-3, 0.5, 1.0, 3.0, 1e3)
SCALES = (1e-6, 1.0, 1e6)
LOCATIONS = (-1e6, 0.0, 1e6)

SKIPPED = {
    "Galton": "alias of LogNormal (the same class)",
    "Gauss": "alias of Normal (the same class)",
    "Bernoulli": "two mass points; its values are p and 1 - p",
    "FixedEventProbability": "two mass points; its values are p, 1 - p",
    "ExactEventTime": "a point mass; no tail",
    "InstantlyOccurs": "degenerate; constant functions",
    "NeverOccurs": "degenerate; constant functions",
    "CustomDistribution": "user-supplied hazard; accuracy is the user's",
    "Discretize": "built from a continuous distribution's sf, which is "
    "checked here",
    "DiscretizedFitter": "the class Discretize returns (see Discretize)",
    "MixtureModel": "a fitted combination of the distributions here",
    "RoystonParmar": "a fitted spline model, not a closed-form "
    "distribution",
}
# Specified below, but not generated yet: the reference computation itself
# fails or is too slow on this grid (#448). Listed as skipped so the check
# names them; build() leaves them out.
NOT_GENERATED = {
    "Beta": "not generated yet (#448): mpmath's incomplete beta returns a "
    "complex value at the extreme shapes",
    "NegativeBinomial": "not generated yet (#448): 11 minutes for the "
    "grid, then an integer-to-string limit at the largest counts",
    "DiscreteWeibull": "not generated yet (#448): at shape 1000 the values "
    "are about 1e-(3e18) and fail the 80-digit re-check",
    "Binomial": "not generated yet (#448): over 30 minutes at the largest "
    "numbers of trials",
}
SKIPPED.update(NOT_GENERATED)

INF = mpmath.inf


def _log(v):
    if v == 0:
        return -INF
    return mp.log(v)


def _log1m(v):
    """log(1 - v)."""
    if v == 1:
        return -INF
    return mp.log1p(-v)


# A density within 1e-25 of 1 (an Exponential with rate 1 near 0) has its
# log recomputed with this many more digits.
NEAR_ONE_DPS = 350


def _near_one(v):
    return _finite(v) and abs(v - 1) < mpf(10) ** -25


def _finite(v):
    return mpmath.isfinite(v)


def _values_from(sf, ff, df, sf_before=None):
    """Every function from accurate sf, ff and df (for a discrete
    distribution, ``sf_before`` is sf(k - 1) and df the mass)."""
    log_sf = _log(sf) if sf < 0.5 else _log1m(ff)
    log_ff = _log(ff) if ff < 0.5 else _log1m(sf)
    at_risk = sf if sf_before is None else sf_before
    if at_risk == 0:
        hf = INF
    else:
        hf = df / at_risk
    return {
        "sf": sf,
        "ff": ff,
        "df": df,
        "hf": hf,
        "Hf": -log_sf,
        "log_sf": log_sf,
        "log_ff": log_ff,
        "log_df": _log(df),
    }


# ---------------------------------------------------------------------------
# Continuous distributions: ``core(x, *p)`` returns (sf, ff, df) strictly
# inside the support, each computed directly so the small one of sf and ff
# is accurate; ``edge_df(side, *p)`` is the density's limit at an edge.
# ---------------------------------------------------------------------------


class Continuous:
    discrete = False

    def __init__(
        self,
        name,
        grid,
        core,
        support,
        edge_df,
        qf=None,
        start=None,
        extra_dps=0,
    ):
        self.name = name
        self.grid = grid
        self.core = core
        self.support = support
        self.edge_df = edge_df
        self.qf = qf
        self.start = start
        self.extra_dps = extra_dps

    def values(self, x, p):
        lo, hi = self.support(*p)
        with mp.workdps(mp.dps + self.extra_dps):
            if x == lo:
                return _values_from(mpf(1), mpf(0), self.edge_df("lo", *p))
            if x == hi:
                out = _values_from(mpf(0), mpf(1), self.edge_df("hi", *p))
                out["hf"] = INF
                return out
            sf, ff, df = self.core(x, *p)
            out = _values_from(sf, ff, df)
            if _near_one(df):
                with mp.workdps(mp.dps + NEAR_ONE_DPS):
                    out["log_df"] = mp.log(self.core(x, *p)[2])
        return {k: +v for k, v in out.items()}

    def small_side(self, x, p, side):
        """The CDF (side "left") or survival function at ``x``."""
        with mp.workdps(mp.dps + self.extra_dps):
            sf, ff, _ = self.core(x, *p)
        return ff if side == "left" else sf

    def _to_x(self, y, p):
        lo, hi = self.support(*p)
        if lo == -INF:
            return y
        if hi == INF:
            return lo + mp.exp(y)
        return lo + (hi - lo) / (1 + mp.exp(-y))

    def invert(self, p, side, target):
        """The time where ff (side "left") or sf (side "right") equals
        ``target``, by bracketing and bisection on a transformed scale."""
        y0, step = self.start(*p)
        log_t = mp.log(target)

        edge_lo, edge_hi = self.support(*p)

        def h(y):
            x = self._to_x(y, p)
            # at a finite edge in working precision: ff = 0 below, sf = 0
            # above
            if x == edge_lo:
                return -INF
            if x == edge_hi:
                return INF
            v = _log(self.small_side(x, p, side)) - log_t
            return v if side == "left" else -v

        y0, step = mpf(y0), mpf(step)
        h0 = h(y0)
        direction = 1 if h0 < 0 else -1
        a = y0
        b = y0 + direction * step
        for _ in range(4000):
            hb = h(b)
            if (hb >= 0) if direction > 0 else (hb < 0):
                break
            a = b
            step *= 2
            b = y0 + direction * step
        else:
            raise RuntimeError("no bracket for {} {}".format(self.name, p))
        lo, hi = (a, b) if a < b else (b, a)
        for _ in range(400):
            mid = (lo + hi) / 2
            if h(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= mpf(10) ** (-(mp.dps - 5)) * max(1, abs(mid)):
                break
        return self._to_x((lo + hi) / 2, p)

    def quantile(self, p, u, uc):
        """The exact quantile at ``u`` (``uc = 1 - u``, both exact)."""
        if self.qf is not None:
            return self.qf(u, uc, *p)
        if u <= 0.5:
            return self.invert(p, "left", u)
        return self.invert(p, "right", uc)


def _pos_start(scale):
    return lambda *p: (mp.log(scale(*p)), 1)


def _positive(*p):
    return (mpf(0), INF)


def _real(*p):
    return (-INF, INF)


def _power_edge(power, value_at_one):
    """Limit at 0 of a density that behaves like c x^(power - 1)."""

    def edge(side, *p):
        k = power(*p)
        if k < 1:
            return INF
        if k == 1:
            return value_at_one(*p)
        return mpf(0)

    return edge


def _L(u, uc):
    """-log(1 - u), accurately on both sides."""
    return -mp.log1p(-u) if u <= 0.5 else -mp.log(uc)


def _logu(u, uc):
    """log(u), accurately on both sides."""
    return mp.log(u) if u <= 0.5 else mp.log1p(-uc)


def _weibull(x, alpha, beta):
    H = (x / alpha) ** beta
    df = beta / alpha * (x / alpha) ** (beta - 1) * mp.exp(-H)
    return mp.exp(-H), -mp.expm1(-H), df


def _exponential(x, rate):
    H = rate * x
    return mp.exp(-H), -mp.expm1(-H), rate * mp.exp(-H)


def _rayleigh(x, sigma):
    H = x**2 / (2 * sigma**2)
    return mp.exp(-H), -mp.expm1(-H), x / sigma**2 * mp.exp(-H)


def _gamma_pq(a, x):
    """The regularised incomplete gamma functions (P, Q), the smaller of
    the two summed directly: the series for P below x = a + 1, the
    continued fraction for Q above it. (mpmath's ``gammainc`` does not
    converge for a Poisson's k + 1 ~ 5e5 and mean 1e6.)"""
    if x == 0:
        return mpf(0), mpf(1)
    tol = mpf(10) ** -(mp.dps + 5)
    if x < a + 1:
        term = total = 1 / a
        n = 0
        while abs(term) > tol * abs(total):
            n += 1
            term *= x / (a + n)
            total += term
        p = mp.exp(a * mp.log(x) - x - mp.loggamma(a)) * total
        return p, 1 - p
    tiny = mpf(10) ** -(mp.dps * 4)
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    frac = d
    i = 0
    while True:
        i += 1
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if d == 0 else d
        c = b + an / c
        c = tiny if c == 0 else c
        d = 1 / d
        delta = d * c
        frac *= delta
        if abs(delta - 1) < tol:
            break
    q = mp.exp(a * mp.log(x) - x - mp.loggamma(a)) * frac
    return 1 - q, q


def _gamma(x, alpha, beta):
    z = beta * x
    ff, sf = _gamma_pq(alpha, z)
    log_df = (
        alpha * mp.log(beta)
        + (alpha - 1) * mp.log(x)
        - z
        - mp.loggamma(alpha)
    )
    return sf, ff, mp.exp(log_df)


def _lognormal(x, mu, sigma):
    z = (mp.log(x) - mu) / sigma
    return mp.ncdf(-z), mp.ncdf(z), mp.npdf(z) / (sigma * x)


def _normal(x, mu, sigma):
    z = (x - mu) / sigma
    return mp.ncdf(-z), mp.ncdf(z), mp.npdf(z) / sigma


def _logistic(x, mu, sigma):
    z = (x - mu) / sigma
    e = mp.exp(-abs(z))
    return (
        1 / (1 + mp.exp(z)),
        1 / (1 + mp.exp(-z)),
        e / (sigma * (1 + e) ** 2),
    )


def _loglogistic(x, alpha, beta):
    t = (x / alpha) ** beta
    return 1 / (1 + t), t / (1 + t), beta / x * t / (1 + t) ** 2


def _gumbel(x, mu, sigma):
    ez = mp.exp((x - mu) / sigma)
    return mp.exp(-ez), -mp.expm1(-ez), ez * mp.exp(-ez) / sigma


def _gumbel_lev(x, mu, sigma):
    e = mp.exp(-(x - mu) / sigma)
    return -mp.expm1(-e), mp.exp(-e), e * mp.exp(-e) / sigma


def _expo_weibull(x, alpha, beta, mu):
    w = (x / alpha) ** beta
    g = -mp.expm1(-w)
    log_g = mp.log(g) if g < 0.5 else mp.log1p(-mp.exp(-w))
    ff = mp.exp(mu * log_g)
    sf = -mp.expm1(mu * log_g)
    df = mu * mp.exp((mu - 1) * log_g) * beta / x * w * mp.exp(-w)
    return sf, ff, df


def _betainc(a, b, x):
    """The regularised incomplete beta function from 0 to ``x``. (mpmath's
    ``betainc(a, b, x, 1)`` is formed as a difference, so its upper tail
    is lost below ~1e-50; the upper tail here is always the lower tail of
    the mirrored function.)"""
    return mp.betainc(a, b, 0, x, regularized=True)


def _betainc_pair(a, b, x, xc):
    """(I_x(a, b), 1 - I_x(a, b)) with ``xc = 1 - x``, the smaller one
    summed directly. The upper tail is tried first; when it is below 1/4
    the lower is its complement (mpmath is slow, 10 s and more, on the
    lower tail of a Negative Binomial's I_p(r, k) at k = 1e12)."""
    upper = _betainc(b, a, xc)
    if upper < 0.25:
        return 1 - upper, upper
    return _betainc(a, b, x), upper


def _expo_weibull_L(u, uc, mu):
    """-log(1 - u^(1/mu)), accurately whichever side of 1/2 u^(1/mu) is."""
    log_v = _logu(u, uc) / mu
    v = mp.exp(log_v)
    return -mp.log1p(-v) if v < 0.5 else -mp.log(-mp.expm1(log_v))


def _beta_core(z, zc, alpha, beta):
    ff = _betainc(alpha, beta, z)
    sf = _betainc(beta, alpha, zc)
    df = z ** (alpha - 1) * zc ** (beta - 1) / mp.beta(alpha, beta)
    return sf, ff, df


def _beta(x, alpha, beta):
    return _beta_core(x, 1 - x, alpha, beta)


def _beta4(x, alpha, beta, a, b):
    sf, ff, df = _beta_core((x - a) / (b - a), (b - x) / (b - a), alpha, beta)
    return sf, ff, df / (b - a)


def _beta_edge(side, alpha, beta, a=0, b=1):
    k, other = (alpha, beta) if side == "lo" else (beta, alpha)
    if k < 1:
        return INF
    if k == 1:
        # the density at the edge is 1 / B(1, other) = other
        return mpf(other) / (b - a)
    return mpf(0)


def _uniform(x, a, b):
    return (b - x) / (b - a), (x - a) / (b - a), 1 / (b - a)


def _hypo_coefficients(rates):
    return [
        mp.fprod(r / (r - rj) for r in rates if r != rj) for rj in rates
    ]


def _hypoexponential(x, *rates):
    rates = [mpf(r) for r in rates]
    coef = _hypo_coefficients(rates)
    sf = mp.fsum(c * mp.exp(-r * x) for c, r in zip(coef, rates))
    ff = -mp.fsum(c * mp.expm1(-r * x) for c, r in zip(coef, rates))
    df = mp.fsum(c * r * mp.exp(-r * x) for c, r in zip(coef, rates))
    return sf, ff, df


def _hypo_edge(side, *rates):
    return mpf(rates[0]) if len(rates) == 1 else mpf(0)


def _product(*axes):
    out = [()]
    for axis in axes:
        out = [o + (v,) for o in out for v in axis]
    return out


LOG_SCALES = tuple(math.log(s) for s in SCALES)

CONTINUOUS = [
    Continuous(
        "Weibull",
        _product(SCALES, SHAPES),
        _weibull,
        _positive,
        _power_edge(lambda a, b: b, lambda a, b: 1 / mpf(a)),
        qf=lambda u, uc, a, b: a * _L(u, uc) ** (1 / mpf(b)),
        start=_pos_start(lambda a, b: a),
    ),
    Continuous(
        "Exponential",
        _product(SCALES),
        _exponential,
        _positive,
        lambda side, r: mpf(r),
        qf=lambda u, uc, r: _L(u, uc) / r,
        start=_pos_start(lambda r: 1 / mpf(r)),
    ),
    Continuous(
        "Rayleigh",
        _product(SCALES),
        _rayleigh,
        _positive,
        lambda side, s: mpf(0),
        qf=lambda u, uc, s: s * mp.sqrt(2 * _L(u, uc)),
        start=_pos_start(lambda s: s),
    ),
    Continuous(
        "Gamma",
        _product(SHAPES, SCALES),
        _gamma,
        _positive,
        _power_edge(lambda a, b: a, lambda a, b: mpf(b)),
        start=_pos_start(lambda a, b: mpf(a) / b),
    ),
    Continuous(
        "LogNormal",
        _product(LOG_SCALES, SHAPES[:2] + SHAPES[3:]),
        _lognormal,
        _positive,
        lambda side, mu, s: mpf(0),
        start=lambda mu, s: (mu, s),
    ),
    Continuous(
        "Normal",
        _product(LOCATIONS, SCALES),
        _normal,
        _real,
        None,
        start=lambda mu, s: (mu, s),
    ),
    Continuous(
        "Logistic",
        _product(LOCATIONS, SCALES),
        _logistic,
        _real,
        None,
        qf=lambda u, uc, mu, s: mu + s * (_logu(u, uc) - _logu(uc, u)),
        start=lambda mu, s: (mu, s),
    ),
    Continuous(
        "LogLogistic",
        _product(SCALES, SHAPES),
        _loglogistic,
        _positive,
        _power_edge(lambda a, b: b, lambda a, b: 1 / mpf(a)),
        qf=lambda u, uc, a, b: a * (mpf(u) / uc) ** (1 / mpf(b)),
        start=_pos_start(lambda a, b: a),
    ),
    Continuous(
        "Gumbel",
        _product(LOCATIONS, SCALES),
        _gumbel,
        _real,
        None,
        qf=lambda u, uc, mu, s: mu + s * mp.log(_L(u, uc)),
        start=lambda mu, s: (mu, s),
    ),
    Continuous(
        "GumbelLEV",
        _product(LOCATIONS, SCALES),
        _gumbel_lev,
        _real,
        None,
        qf=lambda u, uc, mu, s: mu - s * mp.log(-_logu(u, uc)),
        start=lambda mu, s: (mu, s),
    ),
    Continuous(
        "ExpoWeibull",
        # the product beta * mu decides the density at 0; the grid keeps
        # it away from 1 except at (1, 1), where it is exactly 1
        _product((1e-6, 1e6), (1e-3, 1.0, 1e3), (0.01, 1.0, 500.0)),
        _expo_weibull,
        _positive,
        _power_edge(lambda a, b, m: mpf(b) * m, lambda a, b, m: 1 / mpf(a)),
        qf=lambda u, uc, a, b, m: a * _expo_weibull_L(u, uc, m) ** (1 / mpf(b)),
        start=_pos_start(lambda a, b, m: a),
    ),
    Continuous(
        "Beta",
        _product(SHAPES, SHAPES),
        _beta,
        lambda a, b: (mpf(0), mpf(1)),
        _beta_edge,
        start=lambda a, b: (0, 1),
    ),
    Continuous(
        "Beta4",
        [
            (al, be, lo, hi)
            for al, be in ((0.5, 3.0), (3.0, 0.5), (1e3, 1e-3), (2.0, 2.0))
            for lo, hi in ((-1e6, 1e6), (1e6, 1e6 + 1), (-1.0, 1.0))
        ],
        _beta4,
        lambda al, be, a, b: (mpf(a), mpf(b)),
        _beta_edge,
        start=lambda al, be, a, b: (0, 1),
    ),
    Continuous(
        "Uniform",
        [(0.0, 1.0), (-1e6, 1e6), (1e6, 1e6 + 1), (1e-6, 2e-6)],
        _uniform,
        lambda a, b: (mpf(a), mpf(b)),
        lambda side, a, b: 1 / (mpf(b) - a),
        qf=lambda u, uc, a, b: a + u * (mpf(b) - a),
    ),
    Continuous(
        "Hypoexponential",
        [(1.0, 2.0), (0.5, 1.5, 3.0), (1e-3, 1.0, 1e3), (1e-6, 1e6)],
        _hypoexponential,
        _positive,
        _hypo_edge,
        start=_pos_start(lambda *r: mp.fsum(1 / mpf(v) for v in r)),
        # the partial-fraction sum cancels near 0 (ff ~ x^m); 450 extra
        # digits cover ff down to 1e-400 for m = 3
        extra_dps=450,
    ),
]


# ---------------------------------------------------------------------------
# Discrete distributions on the integers: ``core(k, *p)`` returns (sf, ff,
# pmf) for k from the first mass point ``first`` on (sf(k) = P(T > k)).
# ---------------------------------------------------------------------------


class Discrete:
    discrete = True

    def __init__(self, name, grid, core, first, last=None, extra_dps=0):
        self.name = name
        self.grid = grid
        self.core = core
        self.first = first
        self.last = last
        self.extra_dps = extra_dps

    def _sf_ff_pmf(self, k, p):
        if k < self.first:
            return mpf(1), mpf(0), mpf(0)
        last = self.last(*p) if self.last else None
        if last is not None and k >= last:
            _, _, pmf = self.core(last, *p) if k == last else (0, 0, mpf(0))
            return mpf(0), mpf(1), pmf
        return self.core(k, *p)

    def values(self, k, p):
        with mp.workdps(mp.dps + self.extra_dps):
            sf, ff, pmf = self._sf_ff_pmf(k, p)
            sf_before = self._sf_ff_pmf(k - 1, p)[0]
            out = _values_from(sf, ff, pmf, sf_before)
            if _near_one(pmf):
                with mp.workdps(mp.dps + NEAR_ONE_DPS):
                    out["log_df"] = mp.log(self._sf_ff_pmf(k, p)[2])
        return {key: +v for key, v in out.items()}

    def sf(self, k, p):
        with mp.workdps(mp.dps + self.extra_dps):
            return +self._sf_ff_pmf(k, p)[0]

    def ff(self, k, p):
        with mp.workdps(mp.dps + self.extra_dps):
            return +self._sf_ff_pmf(k, p)[1]

    def smallest_k(self, p, test):
        """The smallest k >= first with ``test(k)``, or None past 2^53."""
        hi = max(self.first, 1)
        if test(self.first):
            return self.first
        while not test(hi):
            hi *= 2
            if hi > MAX_INT:
                return None
        lo = hi // 2 if hi // 2 >= self.first else self.first
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if test(mid):
                hi = mid
            else:
                lo = mid
        return hi


def _poisson(k, mu):
    # P(T > k) = P(k + 1, mu) and P(T <= k) = Q(k + 1, mu)
    sf, ff = _gamma_pq(k + 1, mu)
    pmf = mp.exp(k * mp.log(mu) - mu - mp.loggamma(k + 1))
    return sf, ff, pmf


def _geometric(k, p):
    log_q = mp.log1p(-mpf(p))
    return (
        mp.exp(k * log_q),
        -mp.expm1(k * log_q),
        p * mp.exp((k - 1) * log_q),
    )


def _negative_binomial(k, r, p):
    ff, sf = _betainc_pair(r, k, p, 1 - p)
    log_pmf = (
        mp.loggamma(k - 1 + r)
        - mp.loggamma(r)
        - mp.loggamma(k)
        + r * mp.log(p)
        + (k - 1) * mp.log1p(-mpf(p))
    )
    return sf, ff, mp.exp(log_pmf)


def _discrete_weibull(k, q, beta):
    log_q = mp.log(q)
    now = mpf(k) ** beta
    before = mpf(k - 1) ** beta if k > 1 else mpf(0)
    return (
        mp.exp(now * log_q),
        -mp.expm1(now * log_q),
        mp.exp(before * log_q) * -mp.expm1((now - before) * log_q),
    )


def _log_beta(a, b):
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


def _beta_geometric(k, a, b):
    log_sf = _log_beta(a, b + k) - _log_beta(a, b)
    log_pmf = _log_beta(a + 1, b + k - 1) - _log_beta(a, b)
    return mp.exp(log_sf), -mp.expm1(log_sf), mp.exp(log_pmf)


def _binomial(k, n, p):
    n = int(n)
    ff, sf = _betainc_pair(n - k, k + 1, 1 - p, p)
    log_pmf = (
        mp.loggamma(n + 1)
        - mp.loggamma(k + 1)
        - mp.loggamma(n - k + 1)
        + k * mp.log(p)
        + (n - k) * mp.log1p(-mpf(p))
    )
    return sf, ff, mp.exp(log_pmf)


def _binomial_core(k, n, p):
    if k == int(n):
        # all mass is at or below n; only the mass at n is needed here
        return mpf(0), mpf(1), mpf(p) ** int(n)
    return _binomial(k, n, p)


DISCRETE = [
    Discrete("Poisson", _product((1e-6, 1e-3, 1.0, 1e3, 1e6)), _poisson, 0),
    Discrete(
        "Geometric",
        _product((1e-9, 1e-3, 0.5, 1 - 1e-9)),
        _geometric,
        1,
    ),
    Discrete(
        "NegativeBinomial",
        # r = 1e3 with p = 1e-6 is left out: its mass sits at k ~ 1e9 with
        # a spread of 3e7, where mpmath's incomplete beta takes seconds a
        # call; r = 1e3 with p = 1e-3 stands in for it
        _product((1e-3, 1.0), (1e-6, 0.5, 1 - 1e-6))
        + _product((1e3,), (1e-3, 0.5, 1 - 1e-6)),
        _negative_binomial,
        1,
    ),
    Discrete(
        "DiscreteWeibull",
        _product((1e-6, 0.5, 1 - 1e-6), (0.1, 1.0, 3.0, 1e3)),
        _discrete_weibull,
        1,
    ),
    Discrete(
        "BetaGeometric",
        _product((1e-3, 1.0, 1e3), (1e-3, 1.0, 1e3)),
        _beta_geometric,
        1,
        # log sf is a difference of log-gammas of size ~k log k; the extra
        # digits keep its value near 0 (sf near 1) exact
        extra_dps=40,
    ),
    Discrete(
        "Binomial",
        _product((10.0, 1000.0, 1e6), (1e-6, 0.5, 1 - 1e-6)),
        _binomial_core,
        0,
        last=lambda n, p: int(n),
    ),
]

# Parameters that are integers by definition and are not perturbed, and
# the probabilities (see ``_perturbed``).
INTEGER_PARAMS = {"Binomial": (0,)}
PROBABILITY_PARAMS = {
    "Geometric": (0,),
    "NegativeBinomial": (1,),
    "DiscreteWeibull": (0,),
    "Binomial": (1,),
}


# ---------------------------------------------------------------------------
# Sensitivities and encoding
# ---------------------------------------------------------------------------


def _h():
    return mpf(10) ** -20


def _perturbed(name, params, i, factor):
    """``params`` with parameter ``i`` scaled by ``factor``; a probability
    above one half has its complement scaled instead (the complement of a
    double in [0.5, 1] is exact, so it is the complement's rounding that
    matters)."""
    out = list(params)
    v = mpf(params[i])
    if i in PROBABILITY_PARAMS.get(name, ()) and v > 0.5:
        out[i] = 1 - (1 - v) * factor
    else:
        out[i] = v * factor
    return tuple(out)


def _sensitivities(spec, base, x, params, perturb_x):
    """sum_i |d f / d log(theta_i)| for every function, by central
    differences in the relative step 1e-20."""
    h = _h()
    total = {k: mpf(0) for k in FUNCTIONS}
    skip = INTEGER_PARAMS.get(spec.name, ())
    inputs = ([None] if perturb_x and x != 0 else []) + [
        i for i in range(len(params)) if i not in skip and params[i] != 0
    ]
    for i in inputs:
        if i is None:
            up = spec.values(x * (1 + h), params)
            down = spec.values(x * (1 - h), params)
        else:
            up = spec.values(x, _perturbed(spec.name, params, i, 1 + h))
            down = spec.values(x, _perturbed(spec.name, params, i, 1 - h))
        for k in FUNCTIONS:
            if not _finite(base[k]):
                continue
            if _finite(up[k]) and _finite(down[k]):
                total[k] += abs(up[k] - down[k]) / (2 * h)
            else:
                total[k] = INF
    return total


def _enc(v, digits=17):
    if v == INF:
        return "inf"
    if v == -INF:
        return "-inf"
    if v != 0 and abs(mp.mag(v)) > 10**6:
        # beyond 1e+-300000 (a discrete Weibull's q^(k^1000)): too long to
        # print, and a double holds only 0 or inf
        return "{}1e{}999999".format(
            "-" if v < 0 else "", "+" if mp.mag(v) > 0 else "-"
        )
    return mp.nstr(v, digits, min_fixed=-5, max_fixed=16)


def _enc_s(v):
    return _enc(v, digits=2)


def _agree(a, b):
    if not (_finite(a) and _finite(b)):
        return a == b
    if a == b:
        return True
    return abs(a - b) <= mpf(10) ** -30 * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# Building the grid
# ---------------------------------------------------------------------------


def _continuous_points(spec, p):
    lo, hi = spec.support(*p)
    points = []
    if _finite(lo):
        points.append(("lower edge", float(lo)))
    for regime, side, target in TARGETS:
        if spec.qf is not None:
            t = mpf(target)
            u, uc = (t, 1 - t) if side == "left" else (1 - t, t)
            x = spec.qf(u, uc, *p)
        else:
            if spec.start is None:
                continue
            x = spec.invert(p, side, mpf(target))
        xf = float(x)
        if not math.isfinite(xf) or not (float(lo) < xf < float(hi)):
            continue
        if lo == 0 and xf == 0.0:
            continue
        points.append((regime, xf))
    if _finite(hi):
        points.append(("upper edge", float(hi)))
    return points


def _discrete_points(spec, p):
    points = [("lower edge", spec.first - 1), ("lower edge", spec.first)]
    last = spec.last(*p) if spec.last else None
    for regime, side, target in TARGETS:
        t = mpf(target)
        if side == "left":
            k = spec.smallest_k(p, lambda k: spec.ff(k, p) > t)
            if k is None:
                continue
            k -= 1
            if k < spec.first:
                continue
        else:
            k = spec.smallest_k(p, lambda k: spec.sf(k, p) <= t)
            if k is None:
                continue
        if last is not None and k >= last:
            continue
        points.append((regime, k))
    for k in LARGE_K:
        if last is None or k < last:
            points.append(("large k", k))
    if last is not None:
        points.append(("upper edge", last))
    return points


def _dedupe(points):
    seen, out = set(), []
    for regime, x in points:
        if x in seen:
            continue
        seen.add(x)
        out.append((regime, x))
    return out


def _qf_entries(spec, p):
    out = []
    for regime, u in QF_POINTS:
        if spec.discrete:
            entry = _discrete_qf(spec, p, u)
        else:
            entry = _continuous_qf(spec, p, u)
        if entry is not None:
            out.append((regime, u) + entry)
    return out


def _continuous_qf(spec, p, u):
    lo, hi = spec.support(*p)
    if u == 0.0:
        return lo, mpf(0)
    if u == 1.0:
        return hi, mpf(0)
    um = mpf(u)
    uc = 1 - um
    x = spec.quantile(p, um, uc)
    if not _finite(x) or x in (lo, hi):
        return x, mpf(0)
    # sensitivity through the implicit function F(x) = u
    side = "left" if u <= 0.5 else "right"
    with mp.workdps(mp.dps + spec.extra_dps):
        sf, ff, df = spec.core(x, *p)
    if df == 0 or not _finite(df):
        return x, INF
    s = (um if side == "left" else uc) / df
    h = _h()
    for i in range(len(p)):
        if p[i] == 0:
            continue
        du = spec.small_side(x, _perturbed(spec.name, p, i, 1 + h), side)
        dd = spec.small_side(x, _perturbed(spec.name, p, i, 1 - h), side)
        s += abs(du - dd) / (2 * h) / df
    return x, s


def _discrete_qf(spec, p, u):
    if u == 0.0:
        # inf{k : F(k) >= 0} is not a support point; conventions differ
        return None
    last = spec.last(*p) if spec.last else None
    uc = 1 - mpf(u)
    if u == 1.0:
        return (mpf(last) if last is not None else INF), mpf(0)
    k = spec.smallest_k(p, lambda k: spec.sf(k, p) <= uc)
    if k is None:
        return None
    # leave out a probability within rounding of a step of the CDF, where
    # the answer depends on the last bit of F
    margin = mpf(10) ** -9 * min(mpf(u), uc)
    if abs(spec.sf(k, p) - uc) < margin:
        return None
    if k > spec.first and abs(spec.sf(k - 1, p) - uc) < margin:
        return None
    return mpf(k), mpf(0)


def _case(spec, params, rng, checks):
    p = tuple(mpf(v) for v in params)
    if spec.discrete:
        points = _dedupe(_discrete_points(spec, p))
    else:
        points = _dedupe(_continuous_points(spec, p))
    xs, regimes, vals, sens = [], [], [], []
    for regime, x in points:
        xm = mpf(x)
        base = spec.values(xm, p)
        if regime in ("lower edge", "upper edge"):
            # the limits there are exact (0, 1, inf or a parameter)
            s = {k: mpf(0) for k in FUNCTIONS}
        else:
            s = _sensitivities(
                spec, base, xm, p, perturb_x=not spec.discrete
            )
        if rng.random() < CHECK_FRACTION:
            with mp.workdps(CHECK_DPS):
                again = spec.values(xm, p)
            for k in FUNCTIONS:
                if not _agree(base[k], again[k]):
                    raise RuntimeError(
                        "precision check: {} {} x={} {}: {} vs {}".format(
                            spec.name, params, x, k, base[k], again[k]
                        )
                    )
            checks[0] += 1
        xs.append(float(x))
        regimes.append(regime)
        vals.append([_enc(base[k]) for k in FUNCTIONS])
        sens.append([_enc_s(s[k]) for k in FUNCTIONS])
    qf = _qf_entries(spec, p)
    return {
        "dist": spec.name,
        "params": [float(v) for v in params],
        "x": xs,
        "regime": regimes,
        "values": vals,
        "sens": sens,
        "qf": {
            "u": [e[1] for e in qf],
            "regime": [e[0] for e in qf],
            "value": [_enc(e[2]) for e in qf],
            "sens": [_enc_s(e[3]) for e in qf],
        },
    }


def build(only=None):
    mp.dps = DPS
    rng = random.Random(398)
    checks = [0]
    cases = []
    for spec in CONTINUOUS + DISCRETE:
        if only and spec.name not in only:
            continue
        if spec.name in NOT_GENERATED:
            continue
        for p in spec.grid:
            cases.append(_case(spec, tuple(p), rng, checks))
        print(spec.name, len(spec.grid), "parameter sets", flush=True)
    return {
        "generator": "scripts/reference/tails_mpmath.py",
        "software": "mpmath",
        "version": mpmath.__version__,
        "python": platform.python_version(),
        "settings": {
            "dps": DPS,
            "rechecked_at_dps": CHECK_DPS,
            "rechecked_points": checks[0],
            "functions": list(FUNCTIONS),
            "targets": [list(t) for t in TARGETS],
            "large_k": list(LARGE_K),
            "qf_points": [list(q) for q in QF_POINTS],
            "sensitivity": "sum over the time (or probability) and the "
            "parameters of |d f / d log(theta)|, central differences with "
            "relative step 1e-20; 0 where the value is exact (an edge)",
        },
        "skipped": SKIPPED,
        "cases": cases,
    }


def _dump(out):
    """One case per line, so a diff shows which cases moved."""
    head = {k: v for k, v in out.items() if k != "cases"}
    text = json.dumps(head, indent=1, allow_nan=False)[:-2]
    lines = [
        json.dumps(case, separators=(",", ":"), allow_nan=False)
        for case in out["cases"]
    ]
    return text + ',\n "cases": [\n' + ",\n".join(lines) + "\n ]\n}\n"


def main():
    only = set(sys.argv[1:])
    path = DATA / "tails_mpmath.json"
    if only:
        # a trial run of some distributions, written beside the real file
        path = path.with_suffix(".partial.json")
    path.write_text(_dump(build(only)))
    print("wrote", path)


if __name__ == "__main__":
    main()
