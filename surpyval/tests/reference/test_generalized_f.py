"""The built-in ``GeneralizedF`` against R's flexsurv and closed forms.

``scripts/reference/reference_r_distributions.R`` writes, in
``data/r_distributions.json``, flexsurv 2.3.2's ``dgenf``, ``pgenf`` and
``qgenf`` at fixed parameters (both signs of Q, Q = 0, P from 1e-4 to 2),
its ``flexsurvreg`` fit to survival's ``lung`` data (right censored), and
the left-truncated log-likelihood at fixed parameters. The AFT fit with
``sex`` is the one already in ``r_custom_recipes.json``.

Tolerances. The functions to ``rtol=1e-10``. flexsurv's ``qgenf`` goes
through R's ``qf`` and is itself off by up to 6e-8 of the probability (and
by orders of magnitude at Q = 2, P = 0.5), so the quantiles are compared
with it only where it inverts ``pgenf`` to 1e-12, and everywhere must
invert ``ff`` to 1e-13. The fits as for the generalised gamma
(``test_generalized_gamma.py``).
"""

import mpmath as mp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

import surpyval as surv
from surpyval import GeneralizedF as GF

from ._data import fixture, reference

LOGLIK = dict(abs=1e-6)
EST_IN_SE = 0.01
SE = dict(rtol=1e-3)


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return np.asarray(d["time"], float), np.asarray(d["c"]), d["sex"]


def test_functions_match_flexsurv():
    ref = reference("r_distributions", "functions_genf")
    t = np.array(ref["settings"]["t"], dtype=float)
    p = np.array(ref["settings"]["p"], dtype=float)
    v = ref["values"]
    for k, params in enumerate(ref["settings"]["params"]):
        assert_allclose(GF.df(t, *params), v["pdf"][k], rtol=1e-10)
        assert_allclose(GF.log_df(t, *params), v["log_pdf"][k], rtol=1e-10)
        assert_allclose(GF.ff(t, *params), v["cdf"][k], rtol=1e-10)
        assert_allclose(GF.log_sf(t, *params), v["log_sf"][k], rtol=1e-10)
        mine = GF.qf(p, *params)
        assert_allclose(GF.ff(mine, *params), p, rtol=1e-13)
        theirs = np.array(
            [np.nan if q is None else q for q in v["quantile"][k]]
        )
        good = np.isfinite(theirs)
        good[good] = np.isclose(
            GF.ff(theirs[good], *params), p[good], rtol=1e-12, atol=0
        )
        assert_allclose(mine[good], theirs[good], rtol=1e-10)


def _mp_log_sf(x, mu, sigma, Q, P):
    mp.mp.dps = 50
    x, mu, sigma, Q, P = (mp.mpf(v) for v in (x, mu, sigma, Q, P))
    tmp = Q**2 + 2 * P
    d = mp.sqrt(tmp)
    s1, s2 = 2 / (tmp + Q * d), 2 / (tmp - Q * d)
    z = d * (mp.log(x) - mu) / sigma
    c = s2 / (s2 + s1 * mp.exp(z))
    return float(mp.log(mp.betainc(s2, s1, 0, c, regularized=True)))


@pytest.mark.parametrize(
    "params",
    [(1, 0.5, 0.5, 1e-4), (1, 0.5, 0.5, 1e-6), (1, 0.5, -0.5, 1e-4)]
    + [(1, 0.5, 0.0, 1e-4), (1.0, 0.7, -1.5, 2.0)],
)
def test_near_the_generalised_gamma_matches_mpmath(params):
    """Small P puts one beta shape near 2/P: the tails keep their digits
    there (one shape of 2e6 at P = 1e-6)."""
    x = np.array([0.3, 1.0, 2.7, 8.0, 30.0])
    ref = [_mp_log_sf(xi, *params) for xi in x]
    assert_allclose(GF.log_sf(x, *params), ref, rtol=5e-12)


def test_special_cases():
    t = np.array([0.05, 0.5, 2.0, 5.0, 12.0, 80.0])
    # Q = 0, P = 1 is the log-logistic with scale e^mu, shape sqrt(2)/sigma
    assert_allclose(
        GF.sf(t, 1.0, 0.7, 0.0, 1.0),
        surv.LogLogistic.sf(t, np.e, np.sqrt(2) / 0.7),
        rtol=1e-12,
    )
    # P -> 0 is the generalised gamma, to O(P)
    for Q in (0.5, -0.5, 0.0):
        gap = np.max(
            np.abs(
                GF.log_sf(t, 1.0, 0.7, Q, 1e-7)
                - surv.GeneralizedGamma.log_sf(t, 1.0, 0.7, Q)
            )
        )
        assert gap < 1e-4, (Q, gap)


def test_moments():
    params = (1.5, 0.6, 0.5, 1.0)
    for m in (1, 2):
        numeric = integrate.quad(
            lambda t: t**m * GF.df(t, *params), 0, np.inf, limit=500
        )[0]
        assert float(GF.moment(m, *params)) == pytest.approx(
            numeric, rel=1e-10
        )
    # the upper tail falls as a power: moments with k >= s2 are infinite
    assert GF.mean(1.5, 2.0, 0.5, 1.0) == np.inf


def test_quantiles_in_both_tails():
    params = (1.5, 0.6, 0.5, 1.0)
    s = np.array([1e-300, 1e-40, 1e-3])
    assert_allclose(GF.sf(GF.isf(s, *params), *params), s, rtol=1e-12)
    assert_allclose(GF.ff(GF.qf(s, *params), *params), s, rtol=1e-12)


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
    _check_fit(GF.fit(x, c=c), "r_distributions", "lung_genf")


def test_truncated_likelihood_matches_flexsurv(lung):
    """flexsurvreg cannot fit the left-truncated genf on lung (optim stops
    on a non-finite difference), so its likelihood is compared at fixed
    parameters, from dgenf and pgenf, and SurPyval's truncated fit must
    climb at least as high."""
    x, c, _ = lung
    ref = reference("r_distributions", "lung_genf_truncated_loglik")
    params = ref["settings"]["params"]
    data = surv.SurpyvalData(x, c, tl=np.floor(x / 4))
    ll = GF._log_likelihood(data, *params, 0.0, 0.0, 1.0)
    assert float(ll) == pytest.approx(ref["values"]["loglik"], abs=1e-8)
    # its search runs off (Q and P growing past 17 and 39, and on to 63
    # and 521 since the density is exact far in its upper tail) and says
    # so; it still climbs past the likelihood at the untruncated estimates
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = GF.fit(x, c=c, tl=np.floor(x / 4))
    assert -model.neg_ll() >= ref["values"]["loglik"]


def test_aft_matches_flexsurv(lung):
    x, c, sex = lung
    model = surv.AFT(GF).fit(x, sex, c=c)
    _check_fit(model, "r_custom_recipes", "lung_genf_sex", flip_last=True)


def test_nested_generalised_gamma(lung):
    """The generalised F contains the generalised gamma at the edge P = 0,
    so it fits at least as well, and the likelihood ratio between them is
    the test of P (half a chi-square(1) on the edge)."""
    x, c, _ = lung
    gf = GF.fit(x, c=c)
    gg = surv.GeneralizedGamma.fit(x, c=c)
    assert gf.neg_ll() <= gg.neg_ll() + 1e-9
    near = GF.fit(x, c=c, fixed={"P": 1e-8})
    assert near.neg_ll() == pytest.approx(gg.neg_ll(), abs=1e-4)


def test_simulation_recovery():
    """3000 lifetimes, a fifth censored: every estimate within three
    standard errors of the truth (seed fixed before the result was seen)."""
    truth = np.array([1.0, 0.6, 0.3, 3.0])
    rng = np.random.default_rng(20261128)
    t = GF.random(3000, *truth, random_state=rng)
    cens = rng.uniform(0, 40, 3000)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    model = GF.fit(x, c=c)
    assert np.all(np.abs(model.params - truth) < 3 * model.standard_errors())


def test_in_fit_best():
    from surpyval.fit_best import distributions

    assert GF in distributions
