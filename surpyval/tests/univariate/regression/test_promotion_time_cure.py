"""Promotion-time (bounded cumulative hazard) cure regression
(``PromotionTimeCure``): S(t | z) = exp(-theta(z) F0(t)).

Validated from first principles, since flexsurvcure is not installed:

* the fit is the maximum of a likelihood written here independently and
  maximised by a general optimiser;
* as the cure fraction exp(-theta) tends to 0 the model tends to the
  uncured one: with the covariates on theta it is then a Weibull
  proportional hazards model (``WeibullPH``), and a fit to data with no
  plateau lands on WeibullPH's fit, with the warning that the cure
  fraction is not identified;
* with the covariates on theta alone it is a proportional hazards model:
  the cumulative hazards of two rows are in the ratio exp(b'z) at every
  age;
* a seeded simulation recovers the cure fraction inside its bounds.

``scripts/reference/reference_r_promotion_time_cure.R`` is to write a
flexsurvcure (``mixture = FALSE``) reference, which
``test_against_flexsurvcure`` checks once it exists. The script has not
been run, and its mapping of flexsurvcure's ``link = "loglog"`` onto
SurPyval's log-link theta is to be confirmed on its first run.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.univariate.regression.distributional.promotion_time import (
    PromotionTimeDistribution,
)

REF = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "data"
    / "promotion_time_cure_r.json"
)


def _simulate(n, seed, theta0=0.6, b=(0.7, -0.4), end=(30.0, 60.0)):
    """Units carrying a Poisson number of latent causes, mean theta(z),
    each with a Weibull(12, 1.6) promotion time; a unit fails at the
    first and never if it has none."""
    rng = np.random.default_rng(seed)
    z1 = rng.binomial(1, 0.5, n)
    z2 = rng.normal(size=n)
    theta = theta0 * np.exp(b[0] * z1 + b[1] * z2)
    k = rng.poisson(theta)
    life = np.array(
        [(12.0 * rng.weibull(1.6, j)).min() if j else np.inf for j in k]
    )
    stop = rng.uniform(*end, n)
    x = np.minimum(life, stop)
    c = (life > stop).astype(int)
    return x, c, np.column_stack([z1, z2])


@pytest.fixture(scope="module")
def data():
    return _simulate(500, 20261127)


@pytest.fixture(scope="module")
def model(data):
    x, c, Z = data
    return sp.WeibullPromotionTimeCure.fit(x, Z, c=c)


def test_is_the_maximum_of_an_independent_likelihood(data, model):
    """The log-likelihood written out by hand, maximised by BFGS from a
    start of its own: sum over failures of log(theta f0) and over
    everyone of -theta F0."""
    x, c, Z = data
    died = c == 0

    def neg_ll(v):
        log_alpha, log_beta, log_theta, b1, b2 = v
        alpha, beta = np.exp(log_alpha), np.exp(log_beta)
        theta = np.exp(log_theta + Z @ [b1, b2])
        u = x / alpha
        F0 = -np.expm1(-(u**beta))
        log_f0 = np.log(beta / alpha) + (beta - 1) * np.log(u) - u**beta
        return -(
            np.sum(np.log(theta[died]) + log_f0[died]) - np.sum(theta * F0)
        )

    res = minimize(neg_ll, [np.log(20.0), 0.0, 0.0, 0.0, 0.0], method="BFGS")
    res = minimize(
        neg_ll,
        res.x,
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-12,
            "maxiter": 20000,
        },
    )
    alpha, beta, theta, b1, b2 = model.params
    ours = [np.log(alpha), np.log(beta), np.log(theta), b1, b2]
    np.testing.assert_allclose(ours, res.x, atol=1e-5)
    assert abs(model.neg_ll() - res.fun) < 1e-8


def test_recovers_the_cure_fraction_within_its_bounds():
    """40 seeded replicates of 2000 units, truth theta = 0.6 (cure
    fraction exp(-0.6) = 0.549 at z = 0), alpha = 12, beta = 1.6, b =
    (0.7, -0.4): every estimate's mean within 3 Monte Carlo standard
    errors of the truth, and theta's Wald 95% interval covering it in at
    least 34 of 40 (3 binomial standard errors below 95%). The full
    coverage study is calibration/test_coverage_promotion_time_cure.py.

    (A first version checked one replicate, ``_simulate(4000, 7)`` with
    its default settings, whose Wald interval on theta, (0.609, 0.692),
    just missed 0.6: one draw of a 95% interval misses 1 time in 20, so
    it was replaced by this study, not reseeded.)"""
    truth = np.array([12.0, 1.6, 0.6, 0.7, -0.4])
    est, covered = [], 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(100, 140):
            x, c, Z = _simulate(2000, seed)
            m = sp.WeibullPromotionTimeCure.fit(x, Z, c=c)
            est.append(m.params)
            lo, hi = m.param_cb("theta")
            covered += int(lo < 0.6 < hi)
    est = np.array(est)
    mc_se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - truth) < 3 * mc_se)
    assert covered >= 34


def test_is_proportional_hazards_with_covariates_on_theta(model):
    t = np.array([1.0, 5.0, 12.0, 30.0, 100.0])
    b = model.params[-2:]
    for z in ([1.0, 0.0], [0.0, 1.5], [1.0, -0.7]):
        ratio = model.Hf(t, [z] * t.size) / model.Hf(t, [[0.0, 0.0]] * t.size)
        np.testing.assert_allclose(ratio, np.exp(np.dot(b, z)), rtol=1e-12)
        h_ratio = model.hf(t, [z] * t.size) / model.hf(
            t, [[0.0, 0.0]] * t.size
        )
        np.testing.assert_allclose(h_ratio, np.exp(np.dot(b, z)), rtol=1e-12)
    # The cumulative hazard rises to theta, the cure fraction exp(-theta)
    theta = model.theta_at([0.0, 0.0])
    assert model.Hf(1e6, [0.0, 0.0]) == pytest.approx(theta, rel=1e-12)
    assert model.sf(1e6, [0.0, 0.0]) == pytest.approx(np.exp(-theta))


def test_limit_of_no_cure_is_the_uncured_distribution():
    """With theta -> inf and F0's scale growing as theta^(1/beta) the
    promotion-time Weibull tends to a Weibull: the difference in the
    density and survival shrinks like 1/theta."""
    dist = PromotionTimeDistribution(sp.Weibull)
    t = np.array([0.5, 2.0, 5.0, 10.0])
    alpha, beta = 4.0, 1.7
    errors = []
    for theta in (1e2, 1e4, 1e6):
        a = alpha * theta ** (1 / beta)
        errors.append(
            max(
                np.max(
                    np.abs(
                        dist.sf(t, a, beta, theta)
                        - sp.Weibull.sf(t, alpha, beta)
                    )
                ),
                np.max(
                    np.abs(
                        np.exp(dist.log_df(t, a, beta, theta))
                        / sp.Weibull.df(t, alpha, beta)
                        - 1
                    )
                ),
            )
        )
    assert errors[0] < 1e-1 and errors[2] < 1e-5
    assert errors[0] > 50 * errors[1] > 2500 * errors[2]


def test_no_plateau_warns_and_reaches_the_weibull_ph_fit():
    """Follow-up stopped well inside the lives (no cure in the truth, no
    plateau): theta runs off, the cure fraction to 0, with the warning
    that it is not identified; the fit is then WeibullPH's."""
    rng = np.random.default_rng(3)
    n = 300
    z = rng.normal(size=n)
    t = 100 * rng.weibull(1.5, n) * np.exp(-0.3 * z)
    x, c = np.minimum(t, 60.0), (t > 60).astype(int)
    with pytest.warns(UserWarning, match="cure fraction is not identified"):
        model = sp.WeibullPromotionTimeCure.fit(x, z, c=c)
    assert model.maximum == "no finite maximum"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = sp.WeibullPH.fit(x, z, c=c)
    assert abs(model.neg_ll() - ph.neg_ll()) < 1e-6
    np.testing.assert_allclose(model.params[1], ph.params[1], rtol=1e-4)
    np.testing.assert_allclose(model.params[-1], ph.params[-1], rtol=1e-4)
    times = [5.0, 20.0, 40.0, 60.0]
    np.testing.assert_allclose(
        model.sf(times, [0.5]), ph.sf(times, [0.5]), rtol=1e-5
    )
    # Maller and Zhou's check says the same
    assert not model.follow_up()["sufficient"]


def test_susceptible_distribution(model):
    z = [1.0, 0.3]
    theta = model.theta_at(z)
    q = 1 - np.exp(-theta)
    t = np.array([0.5, 3.0, 10.0, 25.0])
    S = model.sf(t, [z] * t.size)
    Su = model.sf(t, [z] * t.size, susceptible=True)
    np.testing.assert_allclose(Su, (S - np.exp(-theta)) / q, rtol=1e-10)
    np.testing.assert_allclose(
        model.ff(t, [z] * t.size, susceptible=True), 1 - Su, rtol=1e-10
    )
    p = np.array([0.1, 0.5, 0.9])
    tq = model.qf(p, [z] * p.size, susceptible=True)
    np.testing.assert_allclose(
        model.ff(tq, [z] * p.size, susceptible=True), p, rtol=1e-8
    )
    # The density integrates to 1 over the susceptible's life
    from scipy.integrate import quad

    total = quad(lambda s: float(model.df(s, z, susceptible=True)), 0, np.inf)
    assert total[0] == pytest.approx(1.0, abs=1e-7)
    assert np.isinf(model.mean(z))
    assert model.mean(z, susceptible=True) == pytest.approx(
        quad(lambda s: float(model.sf(s, z, susceptible=True)), 0, np.inf)[0]
    )
    # The population's quantile is infinite past the chance of failing
    assert np.isinf(model.qf(q + 0.01, z))
    assert model.cure_given_survival(0.0, z) == pytest.approx(np.exp(-theta))


def test_random_draws_never_failing_units_as_inf(model):
    x, _ = model.random(4000, [[0.0, 0.0]], random_state=1)
    cure = model.cure_probability([0.0, 0.0])
    assert np.mean(np.isinf(x)) == pytest.approx(cure, abs=0.025)


def test_saves_and_restores(model):
    restored = sp.from_dict(model.to_dict())
    assert type(restored).__name__ == "PromotionTimeCureModel"
    np.testing.assert_allclose(
        restored.sf([3.0, 30.0], [1.0, 0.0]), model.sf([3.0, 30.0], [1.0, 0.0])
    )


def test_every_listed_f0_fits(data):
    x, c, Z = data
    for name in ("LogNormal", "LogLogistic", "Gamma", "Exponential"):
        m = getattr(sp, name + "PromotionTimeCure").fit(x, Z, c=c)
        assert m.maximum == "verified", name
        # The cure fraction is a property of the data's plateau, near
        # whatever F0
        assert (
            abs(
                m.cure_probability([0, 0])
                - sp.WeibullPromotionTimeCure.fit(x, Z, c=c).cure_probability(
                    [0, 0]
                )
            )
            < 0.05
        )


def test_refusals():
    with pytest.raises(ValueError, match="Gompertz"):
        sp.PromotionTimeCure(sp.Gompertz)
    with pytest.raises(ValueError, match="not available"):
        sp.PromotionTimeCure(sp.Normal)
    with pytest.raises(ValueError, match="takes the distribution"):
        sp.PromotionTimeCure("Weibull")


def test_distribution_at_refuses(model):
    with pytest.raises(ValueError, match="no univariate distribution"):
        model.distribution_at([0.0, 0.0])


def test_likelihood_ratio_test_against_no_covariates(data, model):
    x, c, Z = data
    null = sp.WeibullPromotionTimeCure.fit(x, Z, c=c, covariates={})
    res = model.likelihood_ratio_test()
    assert res["df"] == 2
    assert res["log_likelihood_null"] == pytest.approx(
        -null.neg_ll(), abs=1e-6
    )


@pytest.mark.skipif(
    not REF.exists(),
    reason="not run yet: run scripts/reference/reference_r_promotion_time_"
    "cure.R (needs flexsurvcure) and confirm its link mapping first",
)
def test_against_flexsurvcure():
    ref = json.loads(REF.read_text())
    d = ref["data"]
    x = np.asarray(d["time"], dtype=float)
    c = 1 - np.asarray(d["status"], dtype=int)
    Z = np.column_stack([d["z1"], d["z2"]])
    fitters = {
        "weibull": sp.WeibullPromotionTimeCure,
        "lnorm": sp.LogNormalPromotionTimeCure,
        "llogis": sp.LogLogisticPromotionTimeCure,
        "weibull_scale": sp.WeibullPromotionTimeCure,
    }
    for key, fitter in fitters.items():
        r = ref["references"][key]
        covariates = {"theta": [0, 1]}
        if key == "weibull_scale":
            covariates["alpha"] = [0]
        m = fitter.fit(x, Z, c=c, covariates=covariates)
        assert -m.neg_ll() == pytest.approx(r["loglik"], abs=1e-5), key
        names = list(r["names"])
        est = dict(zip(names, r["estimate"]))
        coef = m.params[-2:]
        np.testing.assert_allclose(coef, [est["z1"], est["z2"]], atol=1e-4)
        # flexsurvcure's theta (link scale) is log of ours at Z = 0
        theta = m.params[m.parameter_names.index("theta")]
        assert np.log(theta) == pytest.approx(est["theta"], abs=1e-4)
        rows = np.column_stack([r["rows"]["z1"], r["rows"]["z2"]])
        times = np.asarray(r["times"], dtype=float)
        sf = np.asarray(r["sf"], dtype=float).reshape(len(rows), -1)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(
                m.sf(times, [row] * times.size), sf[i], atol=1e-5
            )
