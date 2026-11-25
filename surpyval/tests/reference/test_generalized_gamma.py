"""The built-in ``GeneralizedGamma`` against R's flexsurv and closed forms.

``scripts/reference/reference_r_distributions.R`` writes
``data/r_distributions.json``: flexsurv 2.3.2's ``dgengamma``,
``pgengamma`` and ``qgengamma`` at fixed parameters (both signs of Q, Q
near 0, Q = 0), and its ``flexsurvreg`` fits to survival's ``lung`` data,
right censored and left truncated. The AFT fit with ``sex`` is the one
already in ``r_custom_recipes.json``.

Tolerances. The functions are the same function at the same parameters:
``rtol=1e-10``. The fits come from different optimisers (flexsurv's BFGS
at ``reltol = 1e-14`` against SurPyval's), so the maximised
log-likelihoods agree to ``1e-6``, each estimate to within a hundredth of
R's standard error, and the standard errors to ``rtol=1e-3``, as for the
recipes (``test_custom_recipes.py``).

Where flexsurv itself loses digits the check is against mpmath instead:
``dgengamma`` near Q = 0, where ``q log q - lgamma(q)`` cancels (4e-7 of
the density at Q = 1e-4), and ``qgengamma`` for Q < 0 at probabilities
below 1e-6, where ``qgamma`` stops early (``pgengamma`` of its answer is
8e-8 off the probability asked for).
"""

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, stats

import surpyval as surv
from surpyval import GeneralizedGamma as GG
from surpyval.univariate.parametric.distributions import _temme

from ._data import fixture, reference

LOGLIK = dict(abs=1e-6)
EST_IN_SE = 0.01
SE = dict(rtol=1e-3)


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return np.asarray(d["time"], float), np.asarray(d["c"]), d["sex"]


# ----------------------------------------------------------------------
# The functions against flexsurv


def test_functions_match_flexsurv():
    ref = reference("r_distributions", "functions_gengamma")
    t = np.array(ref["settings"]["t"], dtype=float)
    p = np.array(ref["settings"]["p"], dtype=float)
    v = ref["values"]
    for k, params in enumerate(ref["settings"]["params"]):
        assert_allclose(GG.df(t, *params), v["pdf"][k], rtol=1e-10)
        assert_allclose(GG.log_df(t, *params), v["log_pdf"][k], rtol=1e-10)
        assert_allclose(GG.ff(t, *params), v["cdf"][k], rtol=1e-10)
        assert_allclose(GG.log_sf(t, *params), v["log_sf"][k], rtol=1e-10)
        assert_allclose(GG.qf(p, *params), v["quantile"][k], rtol=1e-10)


def _mp_logpdf(x, mu, sigma, Q):
    mp.mp.dps = 50
    x, mu, sigma, Q = (mp.mpf(v) for v in (x, mu, sigma, Q))
    w = (mp.log(x) - mu) / sigma
    q = 1 / Q**2
    return float(
        mp.log(abs(Q))
        + q * mp.log(q)
        - mp.loggamma(q)
        + q * (Q * w - mp.exp(Q * w))
        - mp.log(sigma * x)
    )


@pytest.mark.parametrize("Q", [1e-4, -3e-3, 0.02, 1e-7])
def test_density_near_the_lognormal_matches_mpmath(Q):
    x = np.array([0.3, 1.0, 2.0, 5.0, 30.0])
    ref = [_mp_logpdf(xi, 0.5, 0.3, Q) for xi in x]
    assert_allclose(GG.log_df(x, 0.5, 0.3, Q), ref, rtol=1e-13)


@pytest.mark.parametrize("Q", [-0.5, -2.0, 0.5])
def test_quantile_inverts_the_distribution_function(Q):
    p = np.array([1e-12, 1e-6, 0.3, 0.5, 0.97])
    x = GG.qf(p, 1.5, 0.6, Q)
    assert_allclose(GG.ff(x, 1.5, 0.6, Q), p, rtol=1e-13)
    # (at Q = -2, S = 1e-300 is past the largest double: e^829)
    s = np.array([1e-100, 1e-40, 1e-3])
    assert_allclose(GG.sf(GG.isf(s, 1.5, 0.6, Q), 1.5, 0.6, Q), s, rtol=1e-12)


# ----------------------------------------------------------------------
# Its special cases


def test_special_cases():
    t = np.array([0.05, 0.5, 2.0, 5.0, 12.0, 80.0])
    mu, sigma = 1.0, 0.7
    # Q = 0 is the lognormal exactly
    assert_allclose(
        GG.sf(t, mu, sigma, 0.0), surv.LogNormal.sf(t, mu, sigma), rtol=1e-14
    )
    assert_allclose(
        GG.df(t, mu, sigma, 0.0), surv.LogNormal.df(t, mu, sigma), rtol=1e-13
    )
    # Q = 1 the Weibull with scale e^mu and shape 1 / sigma
    assert_allclose(
        GG.sf(t, mu, sigma, 1.0),
        surv.Weibull.sf(t, np.exp(mu), 1 / sigma),
        rtol=1e-13,
    )
    # Q = sigma the gamma with shape 1 / sigma^2, rate sigma^2 e^-mu
    a = sigma**-2
    assert_allclose(
        GG.sf(t, mu, sigma, sigma),
        surv.Gamma.sf(t, a, a * np.exp(-mu)),
        rtol=1e-12,
    )
    # Q = sigma = 1 the exponential with rate e^-mu
    assert_allclose(
        GG.sf(t, mu, 1.0, 1.0),
        surv.Exponential.sf(t, np.exp(-mu)),
        rtol=1e-13,
    )


def test_continuous_through_q_zero():
    """No switch at Q = 0: the functions at Q = +-1e-8 are the lognormal's
    plus its first correction in Q, an Edgeworth term (error O(Q^2)), and
    the derivative in Q at 0 agrees with differences across it."""
    w = np.array([-6.0, -2.0, 0.0, 0.5, 3.0, 8.0])
    x = np.exp(w)  # mu = 0, sigma = 1
    for Q in (1e-8, -1e-8):
        edgeworth = stats.norm.sf(w) - Q * stats.norm.pdf(w) * (w**2 + 2) / 6
        assert_allclose(GG.sf(x, 0.0, 1.0, Q), edgeworth, rtol=1e-12)
    from autograd import grad

    def log_sf(Q, xi):
        return GG.log_sf(xi, 0.0, 1.0, Q)

    for xi in x:
        exact = grad(log_sf)(0.0, xi)
        h = 1e-4
        fd = (log_sf(h, xi) - log_sf(-h, xi)) / (2 * h)
        assert exact == pytest.approx(fd, rel=1e-6, abs=1e-10)


def test_the_two_methods_agree_where_they_meet():
    """Temme's expansion (|Q| < 0.1) and the incomplete gamma functions
    agree at |Q| = 0.05 and 0.09, where both are accurate."""
    w = np.linspace(-6, 6, 25)
    for Q in (0.05, -0.05, 0.09, -0.09):
        small, upper = _temme.small_tail(w, Q)
        q = 1 / Q**2
        u = q * np.exp(Q * w)
        lower_gamma = Q > 0
        from surpyval.utils.autograd_gamma_compat import (
            _gammainccln_raw,
            _gammaincln_raw,
        )

        log_s = (
            _gammainccln_raw(q, u) if lower_gamma else _gammaincln_raw(q, u)
        )
        log_f = (
            _gammaincln_raw(q, u) if lower_gamma else _gammainccln_raw(q, u)
        )
        assert_allclose(small, np.where(upper, log_s, log_f), rtol=1e-12)


def test_far_tails_are_finite_and_ordered():
    x = np.array([1e-30, 1e-8, 1e8, 1e30])
    for Q in (-1.5, -0.05, 0.0, 1e-6, 0.05, 1.5):
        log_sf = GG.log_sf(x, 0.0, 1.0, Q)
        log_ff = GG.log_ff(x, 0.0, 1.0, Q)
        assert np.all(np.isfinite(log_sf[:-1])) and np.all(log_sf <= 0)
        assert np.all(np.isfinite(log_ff[1:])) and np.all(log_ff <= 0)
        # each tail's own log keeps falling where the other rounds to 0
        assert np.all(np.diff(log_sf[1:]) < 0), (Q, log_sf)
        assert np.all(np.diff(log_ff[:-1]) > 0), (Q, log_ff)


@pytest.mark.parametrize(
    "params",
    [(1.5, 0.6, 0.5), (1.5, 0.6, -0.5), (0.2, 1.2, 2.0), (1.0, 0.8, 0.0)]
    + [(1.0, 0.8, 1e-6), (1.0, 0.8, -0.05)],
)
def test_moments_match_integration(params):
    for m in (1, 2):
        numeric = integrate.quad(
            lambda t: t**m * GG.df(t, *params), 0, np.inf, limit=500
        )[0]
        assert float(GG.moment(m, *params)) == pytest.approx(
            numeric, rel=1e-10
        )


def test_moment_is_infinite_when_the_tail_is_too_heavy():
    # Q < 0 with m sigma |Q| >= 1
    assert GG.mean(0.5, 0.5, -2.0) == np.inf
    assert np.isfinite(GG.mean(0.5, 0.4, -2.0))
    assert GG.moment(2, 0.5, 0.4, -2.0) == np.inf


# ----------------------------------------------------------------------
# The fits against flexsurvreg, and the nested families


def _check_fit(model, source, ref_id, flip_last=False):
    v = reference(source, ref_id)["values"]
    est = np.array(v["est"])
    if flip_last:
        # flexsurv's coefficient shifts mu; SurPyval's AFT is its negative
        est[-1] = -est[-1]
    assert -model.neg_ll() == pytest.approx(v["loglik"], **LOGLIK)
    assert np.all(np.abs(model.params - est) < EST_IN_SE * np.array(v["se"]))
    assert_allclose(model.standard_errors(), v["se"], **SE)


def test_fit_matches_flexsurv(lung):
    x, c, _ = lung
    _check_fit(GG.fit(x, c=c), "r_distributions", "lung_gengamma")


def test_truncated_fit_matches_flexsurv(lung):
    x, c, _ = lung
    _check_fit(
        GG.fit(x, c=c, tl=np.floor(x / 4)),
        "r_distributions",
        "lung_gengamma_truncated",
    )


def test_aft_matches_flexsurv(lung):
    x, c, sex = lung
    model = surv.AFT(GG).fit(x, sex, c=c)
    _check_fit(model, "r_custom_recipes", "lung_gengamma_sex", flip_last=True)


def test_nested_fits(lung):
    """Q fixed at 1 is the Weibull fit, at 0 the lognormal fit, and Q tied
    to sigma the gamma; twice the log-likelihood gap is the likelihood
    ratio test of each restriction."""
    x, c, _ = lung
    weibull = surv.Weibull.fit(x, c=c)
    at_one = GG.fit(x, c=c, fixed={"Q": 1.0})
    assert at_one.neg_ll() == pytest.approx(weibull.neg_ll(), rel=1e-9)
    assert_allclose(
        at_one.params[:2],
        [np.log(weibull.params[0]), 1 / weibull.params[1]],
        rtol=1e-5,
    )
    lognormal = surv.LogNormal.fit(x, c=c)
    at_zero = GG.fit(x, c=c, fixed={"Q": 0.0})
    assert at_zero.neg_ll() == pytest.approx(lognormal.neg_ll(), rel=1e-9)
    assert_allclose(at_zero.params[:2], lognormal.params, rtol=1e-5)
    full = GG.fit(x, c=c)
    assert full.neg_ll() <= min(weibull.neg_ll(), lognormal.neg_ll()) + 1e-9


def test_simulation_recovery():
    """2000 lifetimes, a quarter censored: every estimate within three
    standard errors of the truth (seed fixed before the result was seen)."""
    truth = np.array([2.0, 0.5, -0.4])
    rng = np.random.default_rng(20261125)
    t = GG.random(2000, *truth, random_state=rng)
    cens = rng.uniform(0, 25, 2000)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    model = GG.fit(x, c=c)
    assert np.all(np.abs(model.params - truth) < 3 * model.standard_errors())


def test_in_fit_best():
    from surpyval.fit_best import distributions

    assert GG in distributions
