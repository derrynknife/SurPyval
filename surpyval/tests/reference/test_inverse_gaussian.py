"""The built-in ``InverseGaussian`` against R's statmod and SuppDists.

``scripts/reference/reference_r_distributions.R`` writes, in
``data/r_distributions.json``, statmod 1.5.2's ``dinvgauss``,
``pinvgauss`` and ``qinvgauss`` and SuppDists 1.1-9.9's ``dinvGauss`` and
``pinvGauss`` at fixed parameters (``lam / mu`` from 0.15 to 500), and
the censored fits to survival's ``lung`` data (right censored, and left
truncated), maximised in R by ``optim`` on a likelihood written with
``pinvgauss(log.p = TRUE)``, standard errors from numDeriv's Hessian.

Tolerances. The functions to ``rtol=1e-10``. statmod's ``qinvgauss``
gives NA at ``lam / mu = 500``, ``p = 1e-6``; there the check is that
``ff(qf(p)) = p``. The fits: log-likelihoods to 1e-6, estimates within a
hundredth of R's standard error, standard errors to ``rtol=1e-3``.
"""

import mpmath as mp
import numpy as np
import pytest
from autograd import grad
from numpy.testing import assert_allclose
from scipy import integrate

import surpyval as surv
from surpyval import InverseGaussian as IG

from ._data import fixture, reference

LOGLIK = dict(abs=1e-6)
EST_IN_SE = 0.01
SE = dict(rtol=1e-3)


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return np.asarray(d["time"], float), np.asarray(d["c"])


def test_functions_match_statmod():
    ref = reference("r_distributions", "functions_invgauss")
    t = np.array(ref["settings"]["t"], dtype=float)
    p = np.array(ref["settings"]["p"], dtype=float)
    v = ref["values"]
    for k, params in enumerate(ref["settings"]["params"]):
        assert_allclose(IG.df(t, *params), v["pdf"][k], rtol=1e-10)
        assert_allclose(IG.log_df(t, *params), v["log_pdf"][k], rtol=1e-10)
        assert_allclose(IG.ff(t, *params), v["cdf"][k], rtol=1e-10)
        assert_allclose(IG.log_sf(t, *params), v["log_sf"][k], rtol=1e-10)
        q = np.array([np.nan if z is None else z for z in v["quantile"][k]])
        mine = IG.qf(p, *params)
        ok = ~np.isnan(q)
        assert_allclose(mine[ok], q[ok], rtol=1e-10)
        assert_allclose(IG.ff(mine, *params), p, rtol=1e-12)


def test_functions_match_suppdists():
    ref = reference("r_distributions", "functions_invgauss_suppdists")
    t = np.array(ref["settings"]["t"], dtype=float)
    v = ref["values"]
    for k, params in enumerate(ref["settings"]["params"]):
        assert_allclose(IG.df(t, *params), v["pdf"][k], rtol=1e-10)
        if params[1] / params[0] > 300:
            # SuppDists drops the term e^{2 lam / mu} Phi(-b) where its
            # exponential overflows: its cdf at the mean of IG(1, 500) is
            # 0.5, where it is 0.5 + 0.0089 (statmod and mpmath agree)
            continue
        assert_allclose(IG.ff(t, *params), v["cdf"][k], rtol=1e-10)


def _mp_log_sf(t, mu, lam):
    mp.mp.dps = 60
    t, mu, lam = (mp.mpf(v) for v in (t, mu, lam))
    r = mp.sqrt(lam / t)
    return float(
        mp.log(
            mp.ncdf(-r * (t / mu - 1))
            - mp.exp(2 * lam / mu) * mp.ncdf(-r * (t / mu + 1))
        )
    )


@pytest.mark.parametrize("params", [(10.0, 20.0), (1.0, 1.0), (2.0, 0.3)])
def test_far_upper_tail_matches_mpmath(params):
    """Far above the mean the two terms of S nearly cancel (and the
    difference of Mills' ratios lost every digit by x / mu = 1e16): the
    log survival function keeps its relative accuracy."""
    t = np.array([30.0, 1e3, 1e5, 1e8]) * params[0]
    ref = [_mp_log_sf(x, *params) for x in t]
    assert_allclose(IG.log_sf(t, *params), ref, rtol=1e-13)
    # and the hazard tends to lam / (2 mu^2)
    mu, lam = params
    assert IG.hf(1e18 * mu, *params) == pytest.approx(
        lam / (2 * mu**2), rel=1e-6
    )


def test_quantiles_in_both_tails():
    for params in ((10.0, 20.0), (2.0, 0.3), (1.0, 500.0)):
        s = np.array([1e-300, 1e-100, 1e-9, 0.3])
        assert_allclose(IG.sf(IG.isf(s, *params), *params), s, rtol=1e-12)
        assert_allclose(IG.ff(IG.qf(s, *params), *params), s, rtol=1e-12)


def test_moments_match_integration():
    for mu, lam in ((10.0, 20.0), (2.0, 0.3)):
        assert IG.mean(mu, lam) == mu
        for m in (2, 3):
            numeric = integrate.quad(
                lambda t: t**m * IG.df(t, mu, lam), 0, np.inf, limit=500
            )[0]
            assert IG.moment(m, mu, lam) == pytest.approx(numeric, rel=1e-8)
    # the variance is mu^3 / lam
    assert IG.moment(2, 10.0, 20.0) - 100.0 == pytest.approx(1000 / 20)


def test_closed_form_mle_on_exact_data():
    x = IG.random(200, 10.0, 20.0, random_state=1)
    model = IG.fit(x)
    assert model.params[0] == pytest.approx(x.mean(), rel=1e-14)
    inv = np.mean(1 / x - 1 / x.mean())
    assert model.params[1] == pytest.approx(1 / inv, rel=1e-14)
    # and it is the maximum: the gradient of the log-likelihood is 0 there
    g = grad(lambda p: np.sum(IG.log_df(x, p[0], p[1])))(model.params)
    assert np.all(np.abs(g) < 1e-8 * len(x))


def _check_fit(model, ref_id):
    v = reference("r_distributions", ref_id)["values"]
    est = np.array(v["est"])
    assert -model.neg_ll() == pytest.approx(v["loglik"], **LOGLIK)
    assert np.all(np.abs(model.params - est) < EST_IN_SE * np.array(v["se"]))
    assert_allclose(model.standard_errors(), v["se"], **SE)


def test_fit_matches_r(lung):
    x, c = lung
    _check_fit(IG.fit(x, c=c), "lung_invgauss")


def test_truncated_fit_matches_r(lung):
    x, c = lung
    _check_fit(IG.fit(x, c=c, tl=np.floor(x / 4)), "lung_invgauss_truncated")


def test_large_shape_tends_to_the_normal():
    # lam / mu -> inf: a normal with mean mu and variance mu^3 / lam
    mu, lam = 10.0, 1e8
    t = mu + np.array([-2.0, 0.0, 1.0]) * np.sqrt(mu**3 / lam)
    assert_allclose(
        IG.ff(t, mu, lam),
        surv.Normal.ff(t, mu, np.sqrt(mu**3 / lam)),
        atol=1e-3,
    )


def test_simulation_recovery():
    """2000 lifetimes, censored at 25: every estimate within three standard
    errors of the truth (seed fixed before the result was seen)."""
    truth = np.array([10.0, 20.0])
    rng = np.random.default_rng(20261127)
    t = IG.random(2000, *truth, random_state=rng)
    c = (t > 25).astype(int)
    model = IG.fit(np.minimum(t, 25.0), c=c)
    assert np.all(np.abs(model.params - truth) < 3 * model.standard_errors())


def test_in_fit_best():
    from surpyval.fit_best import distributions

    assert IG in distributions
