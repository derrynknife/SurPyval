"""The built-in ``BirnbaumSaunders`` against independent references.

The issue's reference is R's ``VGAM`` (``dbisa``, ``pbisa``, ``qbisa``),
which is not installed where this was written; the R script that writes
it, ``scripts/reference/reference_r_birnbaum_saunders.R``, is committed,
and ``test_against_r_vgam`` skips until its output exists. Meanwhile:

- the functions against ``scipy.stats.fatiguelife``, the same
  distribution (``fatiguelife(alpha, scale=beta)``), to ``rtol=1e-10``,
  and against mpmath (60 digits) in the far tails, where scipy's
  probabilities underflow or its lower quantile cancels;
- the mean, variance and raw moments against their closed forms and
  quadrature;
- the maximum likelihood fits with right, left and interval censoring
  and truncation against a likelihood written here from the formulas
  (the normal distribution function from ``scipy.special.log_ndtr``
  only), maximised by Nelder-Mead from several starts, standard errors
  from its finite-difference Hessian; and a simulation recovery.

Tolerances, seeds and sample sizes were fixed before the results were
seen.
"""

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, special, stats
from scipy.optimize import minimize

import surpyval as surv
from surpyval import BirnbaumSaunders as BS

PARAMS = [(0.05, 1.0), (0.5, 10.0), (1.0, 3.0), (2.5, 200.0), (8.0, 0.01)]
R_REFERENCE = (
    Path(__file__).parent / "data" / "r_birnbaum_saunders.json"
)  # written by scripts/reference/reference_r_birnbaum_saunders.R


def _grid(beta):
    return beta * np.geomspace(1e-3, 1e3, 61)


@pytest.mark.parametrize("alpha, beta", PARAMS)
def test_functions_match_scipy(alpha, beta):
    ref = stats.fatiguelife(alpha, scale=beta)
    t = _grid(beta)
    assert_allclose(BS.df(t, alpha, beta), ref.pdf(t), rtol=1e-10)
    assert_allclose(BS.ff(t, alpha, beta), ref.cdf(t), rtol=1e-10)
    assert_allclose(BS.sf(t, alpha, beta), ref.sf(t), rtol=1e-10)
    # the logs where scipy's probabilities have not underflowed, each
    # from its smaller tail (the log of a probability near 1 from 1 - the
    # other tail)
    cdf, sf = ref.cdf(t), ref.sf(t)
    with np.errstate(divide="ignore"):
        log_cdf = np.where(cdf < 0.5, np.log(cdf), np.log1p(-sf))
        log_sf = np.where(sf < 0.5, np.log(sf), np.log1p(-cdf))
    for mine, theirs in (
        (BS.log_df(t, alpha, beta), ref.logpdf(t)),
        (BS.log_ff(t, alpha, beta), log_cdf),
        (BS.log_sf(t, alpha, beta), log_sf),
    ):
        ok = np.isfinite(theirs)
        assert_allclose(mine[ok], theirs[ok], rtol=1e-10)
    p = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6])
    # scipy's lower quantile cancels (alpha z + sqrt(alpha^2 z^2 + 4)):
    # upper half against scipy, the whole range by the round trip
    upper = p >= 0.5
    assert_allclose(
        BS.qf(p[upper], alpha, beta), ref.ppf(p[upper]), rtol=1e-10
    )
    assert_allclose(BS.isf(1 - p[upper], alpha, beta), ref.isf(1 - p[upper]))
    assert_allclose(BS.ff(BS.qf(p, alpha, beta), alpha, beta), p, rtol=1e-12)


def _mp_xi(x, alpha, beta):
    x, alpha, beta = (mp.mpf(v) for v in (x, alpha, beta))
    return (mp.sqrt(x / beta) - mp.sqrt(beta / x)) / alpha


def _mp_log_sf(x, alpha, beta):
    return float(mp.log(mp.ncdf(-_mp_xi(x, alpha, beta))))


def _mp_log_ff(x, alpha, beta):
    return float(mp.log(mp.ncdf(_mp_xi(x, alpha, beta))))


def _mp_quantile(s, alpha, beta, upper):
    """The time with S = s (``upper``) or F = s, by mpmath's root finder
    on the 60-digit log tail."""
    target = mp.log(mp.mpf(s))
    x0 = float(BS.isf(s, alpha, beta) if upper else BS.qf(s, alpha, beta))

    def g(y):
        x = mp.e**y
        xi = _mp_xi(x, alpha, beta)
        return mp.log(mp.ncdf(-xi if upper else xi)) - target

    return float(mp.e ** mp.findroot(g, mp.log(x0)))


@pytest.mark.parametrize("alpha, beta", [(0.5, 10.0), (2.5, 200.0)])
def test_far_tails_match_mpmath(alpha, beta):
    mp.mp.dps = 60
    up = beta * np.array([1e2, 1e4, 1e6])
    down = beta * np.array([1e-2, 1e-4, 1e-6])
    assert_allclose(
        BS.log_sf(up, alpha, beta),
        [_mp_log_sf(x, alpha, beta) for x in up],
        rtol=1e-12,
    )
    assert_allclose(
        BS.log_ff(down, alpha, beta),
        [_mp_log_ff(x, alpha, beta) for x in down],
        rtol=1e-12,
    )
    # quantiles deep in both tails
    for s in (1e-300, 1e-100, 1e-20):
        assert BS.isf(s, alpha, beta) == pytest.approx(
            _mp_quantile(s, alpha, beta, True), rel=1e-10
        )
        assert BS.qf(s, alpha, beta) == pytest.approx(
            _mp_quantile(s, alpha, beta, False), rel=1e-10
        )
    # and the hazard tends to 1 / (2 alpha^2 beta)
    assert BS.hf(1e12 * beta, alpha, beta) == pytest.approx(
        1 / (2 * alpha**2 * beta), rel=1e-4
    )
    assert BS.hf(np.inf, alpha, beta) == 1 / (2 * alpha**2 * beta)


def _mp_hf(x, alpha, beta):
    """The hazard ``J(x) phi(xi) / Q(xi)`` in mpmath, the inverse Mills
    ratio from its asymptotic series where ``xi`` is above 1e6 (its next
    term, 706 / xi**9, is below 1e-50 relative there) and from the
    functions themselves below."""
    x, a, b = mp.mpf(x), mp.mpf(alpha), mp.mpf(beta)
    xi = _mp_xi(x, a, b)
    J = (x + b) / (2 * a * mp.sqrt(b) * x ** mp.mpf(1.5))
    if xi > 1e6:
        mills = xi + 1 / xi - 2 / xi**3 + 10 / xi**5 - 74 / xi**7
    else:
        mills = mp.npdf(xi) / mp.ncdf(-xi)
    return J * mills


@pytest.mark.parametrize(
    "x, alpha, beta",
    [
        # far out at extreme parameters, where the logs of the density
        # and survival function are both -inf (the hazard was NaN)
        (1e300, 4.6e-9, 0.5),
        (1.7e308, 4.6e-9, 0.5),
        (1e300, 0.3, 1e-7),
        (1e10, 1e-12, 3.0),
        # above the median, where the difference of their logs lost
        # xi**2 of the precision (6e-10 at x = 1e6)
        (1e3, 0.3, 4.0),
        (1e6, 0.3, 4.0),
        # either side of xi = 1e8, where the series takes over
        (3.52836e15, 0.3, 4.0),
        (3.67236e15, 0.3, 4.0),
        # near and below the median
        (1.0, 0.5, 10.0),
        (5.0, 0.5, 10.0),
        (12.0, 0.5, 10.0),
    ],
)
def test_the_hazard_is_right_however_far_out(x, alpha, beta):
    # rule fixed before the run: 1e-13 relative
    mp.mp.dps = 60
    assert float(BS.hf(np.array([x]), alpha, beta)[0]) == pytest.approx(
        float(_mp_hf(x, alpha, beta)), rel=1e-13
    )


def test_edges_of_the_support():
    assert BS.sf(0.0, 0.5, 10) == 1.0 and BS.ff(0.0, 0.5, 10) == 0.0
    assert BS.df(0.0, 0.5, 10) == 0.0
    assert BS.sf(np.inf, 0.5, 10) == 0.0 and BS.df(np.inf, 0.5, 10) == 0.0
    assert BS.qf(0.0, 0.5, 10) == 0.0 and BS.qf(1.0, 0.5, 10) == np.inf
    # below the support, as every lifetime distribution
    assert BS.sf(-1.0, 0.5, 10) == 1.0 and BS.df(-1.0, 0.5, 10) == 0.0
    # beta is the median
    assert BS.qf(0.5, 0.7, 13.0) == 13.0


@pytest.mark.parametrize("alpha, beta", [(0.5, 10.0), (1.0, 3.0), (2.5, 2.0)])
def test_moments(alpha, beta):
    mean = beta * (1 + alpha**2 / 2)
    var = (alpha * beta) ** 2 * (1 + 5 * alpha**2 / 4)
    assert BS.mean(alpha, beta) == pytest.approx(mean, rel=1e-15)
    assert BS.moment(1, alpha, beta) == pytest.approx(mean, rel=1e-15)
    assert BS.moment(2, alpha, beta) - mean**2 == pytest.approx(var, rel=1e-13)
    ref = stats.fatiguelife(alpha, scale=beta)
    assert ref.var() == pytest.approx(var, rel=1e-13)
    for m in (2, 3, 4):
        numeric = integrate.quad(
            lambda t: t**m * BS.df(t, alpha, beta),
            0,
            np.inf,
            limit=500,
            epsrel=1e-12,
        )[0]
        assert BS.moment(m, alpha, beta) == pytest.approx(numeric, rel=1e-8)


def test_small_shape_tends_to_the_lognormal():
    # alpha -> 0: ln x normal with mean ln beta and sd alpha
    alpha, beta = 1e-4, 50.0
    t = beta * np.exp(alpha * np.array([-2.0, -0.5, 0.0, 1.0, 2.5]))
    assert_allclose(
        BS.ff(t, alpha, beta),
        surv.LogNormal.ff(t, np.log(beta), alpha),
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# The fits, against a likelihood written here
# ---------------------------------------------------------------------------
def _log_f(x, a, b):
    xi = (np.sqrt(x / b) - np.sqrt(b / x)) / a
    return (
        -0.5 * xi**2
        - 0.5 * np.log(2 * np.pi)
        + np.log((b / x) ** 0.5 + (b / x) ** 1.5)
        - np.log(2 * a * b)
    )


def _log_cdf(x, a, b):
    with np.errstate(divide="ignore"):
        xi = (np.sqrt(x / b) - np.sqrt(b / x)) / a
    return special.log_ndtr(xi)


def _log_sf(x, a, b):
    with np.errstate(divide="ignore"):
        xi = (np.sqrt(x / b) - np.sqrt(b / x)) / a
    return special.log_ndtr(-xi)


def _log_window(lo, hi, a, b):
    """log(F(hi) - F(lo)), lo < hi, from the larger term."""
    return _log_cdf(hi, a, b) + np.log1p(
        -np.exp(_log_cdf(lo, a, b) - _log_cdf(hi, a, b))
    )


def _ll(p, data):
    a, b = p
    if a <= 0 or b <= 0:
        return -np.inf
    ll = np.sum(_log_f(data["x_o"], a, b))
    ll += np.sum(_log_sf(data["x_r"], a, b))
    ll += np.sum(_log_cdf(data["x_l"], a, b))
    ll += np.sum(_log_window(data["x_il"], data["x_ir"], a, b))
    tl = data["tl"]
    ll -= np.sum(_log_sf(tl, a, b))
    return ll


def _fit_independently(data, starts):
    best = None
    for s in starts:
        res = minimize(
            lambda u: -_ll(np.exp(u), data),
            np.log(s),
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-13, "maxiter": 20000},
        )
        if best is None or res.fun < best.fun:
            best = res
    p = np.exp(best.x)
    # standard errors from the central-difference Hessian
    h = 1e-4 * p
    H = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            H[i, j] = (
                _ll(p + ei + ej, data)
                - _ll(p + ei - ej, data)
                - _ll(p - ei + ej, data)
                + _ll(p - ei - ej, data)
            ) / (4 * h[i] * h[j])
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    return p, -best.fun, se


TRUE = (0.7, 100.0)


def _lifetimes(n=150, seed=20261127):
    return stats.fatiguelife(TRUE[0], scale=TRUE[1]).rvs(
        size=n, random_state=np.random.default_rng(seed)
    )


def _empty():
    return {
        k: np.array([]) for k in ("x_o", "x_r", "x_l", "x_il", "x_ir", "tl")
    }


def _right_censored():
    t = _lifetimes()
    cens = np.random.default_rng(1).uniform(50, 250, t.size)
    x, c = np.minimum(t, cens), (t > cens).astype(int)
    data = _empty()
    data["x_o"], data["x_r"] = x[c == 0], x[c == 1]
    return dict(x=x, c=c), data


def _left_censored():
    # units found failed at an inspection at 60 are left censored there
    t = _lifetimes()
    c = np.where(t < 60.0, -1, 0)
    x = np.where(c == -1, 60.0, t)
    data = _empty()
    data["x_o"], data["x_l"] = x[c == 0], x[c == -1]
    return dict(x=x, c=c), data


def _interval_censored():
    # inspections every 40, and the units still working at 240 right
    # censored there
    t = _lifetimes()
    edges = np.arange(0.0, 281.0, 40.0)
    k = np.searchsorted(edges, t)
    lo, hi = edges[k - 1], edges[np.minimum(k, edges.size - 1)]
    right = t > 240.0
    lo = np.where(right, 240.0, lo)
    hi = np.where(right, np.inf, hi)
    data = _empty()
    first = lo == 0
    data["x_l"] = hi[first & ~right]
    data["x_r"] = lo[right]
    keep = ~first & ~right
    data["x_il"], data["x_ir"] = lo[keep], hi[keep]
    return dict(xl=lo, xr=hi), data


def _truncated():
    # left truncated at a quarter of each life (entry into the study)
    t = _lifetimes()
    tl = np.floor(t / 4.0)
    data = _empty()
    data["x_o"], data["tl"] = t, tl
    return dict(x=t, tl=tl), data


@pytest.mark.parametrize(
    "make", [_right_censored, _left_censored, _interval_censored, _truncated]
)
def test_fit_matches_an_independent_likelihood(make):
    kwargs, data = make()
    model = BS.fit(**kwargs)
    p, ll, se = _fit_independently(
        data, [model.params, [0.5, 80.0], [1.5, 150.0]]
    )
    assert -model.neg_ll() == pytest.approx(ll, abs=1e-8)
    assert_allclose(model.params, p, rtol=1e-6)
    assert_allclose(model.standard_errors(), se, rtol=1e-3)


def test_fit_on_lung():
    from ._data import fixture

    d = fixture("lung")
    x, c = np.asarray(d["time"], float), np.asarray(d["c"]).astype(int)
    data = _empty()
    data["x_o"], data["x_r"] = x[c == 0], x[c == 1]
    model = BS.fit(x, c=c)
    p, ll, _ = _fit_independently(data, [model.params, [1.0, 300.0]])
    assert -model.neg_ll() == pytest.approx(ll, abs=1e-8)
    assert_allclose(model.params, p, rtol=1e-6)


def test_simulation_recovery():
    """2000 lifetimes, censored at 200: every estimate within three standard
    errors of the truth (seed fixed before the result was seen)."""
    truth = np.array(TRUE)
    t = BS.random(2000, *truth, random_state=np.random.default_rng(31))
    c = (t > 200).astype(int)
    model = BS.fit(np.minimum(t, 200.0), c=c)
    assert np.all(np.abs(model.params - truth) < 3 * model.standard_errors())


def test_in_fit_best():
    from surpyval.fit_best import distributions

    assert BS in distributions


@pytest.mark.skipif(
    not R_REFERENCE.exists(),
    reason="the VGAM reference file has not been generated yet",
)
def test_against_r_vgam():  # pragma: no cover - needs the file
    refs = json.loads(R_REFERENCE.read_text())
    funcs = refs["functions"]
    t = np.array(funcs["t"], dtype=float)
    p = np.array(funcs["p"], dtype=float)
    for k, (alpha, beta) in enumerate(funcs["params"]):
        assert_allclose(BS.df(t, alpha, beta), funcs["pdf"][k], rtol=1e-10)
        assert_allclose(BS.ff(t, alpha, beta), funcs["cdf"][k], rtol=1e-10)
        assert_allclose(BS.sf(t, alpha, beta), funcs["sf"][k], rtol=1e-10)
        assert_allclose(
            BS.qf(p, alpha, beta), funcs["quantile"][k], rtol=1e-10
        )
    for name, ref in refs["fits"].items():
        x = np.array(ref["x"], dtype=float)
        c = np.array(ref["c"], dtype=int)
        model = BS.fit(x, c=c)
        assert -model.neg_ll() == pytest.approx(ref["loglik"], abs=1e-6)
        assert np.all(
            np.abs(model.params - np.array(ref["est"]))
            < 0.01 * np.array(ref["se"])
        ), name
