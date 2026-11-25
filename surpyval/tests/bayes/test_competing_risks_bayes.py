"""Bayesian parametric competing risks
(``ParametricCompetingRisks.fit_bayes``): a composition of each cause's
``fit_bayes``.

The references are the causes' own fits (the likelihood factorises, so
each cause's posterior must be that fit's exactly), closed forms (two
Exponential causes with Gamma priors: conjugate, with the incidence
``lambda_a / (lambda_a + lambda_b) (1 - exp(-(lambda_a + lambda_b) t))``
and a Lomax predictive survival), and the maximum-likelihood model's own
incidence and quantile at each draw's parameters.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.competing_risks import ParametricCompetingRisks
from surpyval.univariate.competing_risks.parametric import bayes


def _data(n=60, seed=1):
    ta = sp.Weibull.random(n, 100, 2, random_state=seed)
    tb = sp.Weibull.random(n, 150, 1.2, random_state=seed + 1)
    cen = np.full(n, 120.0)
    x = np.minimum.reduce([ta, tb, cen])
    e = np.where(x == cen, None, np.where(ta < tb, "a", "b")).astype(object)
    c = (x == cen).astype(int)
    return x, e, c


@pytest.fixture(scope="module")
def weibull_fit():
    x, e, c = _data()
    return ParametricCompetingRisks.fit_bayes(x, e, c=c), (x, e, c)


@pytest.fixture(scope="module")
def exponential_fit():
    x, e, c = _data(40, seed=5)
    priors = {
        "a": {"failure_rate": sp.Gamma.from_params([2.0, 300.0])},
        "b": {"failure_rate": sp.Gamma.from_params([3.0, 500.0])},
    }
    model = ParametricCompetingRisks.fit_bayes(
        x, e, c=c, dist=sp.Exponential, priors=priors
    )
    return model, (x, e, c)


def test_each_cause_is_its_own_fit(weibull_fit):
    model, (x, e, c) = weibull_fit
    for k in ("a", "b"):
        c_k = np.where(e == k, 0, 1)
        alone = sp.Weibull.fit_bayes(x, c_k)
        got = model.models[k]
        assert got.method == alone.method == "grid"
        np.testing.assert_array_equal(
            got.posterior.points, alone.posterior.points
        )
        np.testing.assert_array_equal(
            got.posterior.weights, alone.posterior.weights
        )
        np.testing.assert_allclose(got.sf([50, 100]), alone.sf([50, 100]))


def test_predictive_incidence_is_exact(weibull_fit, exponential_fit):
    model, _ = weibull_fit
    t = np.array([20.0, 50.0, 100.0, 200.0])
    # The predictive incidence, int f_a(u) S_b(u) du, by adaptive
    # quadrature over time from the causes' predictive functions.
    from scipy.integrate import quad

    fa, sb = model.models["a"], model.models["b"]
    ref = quad(lambda u: float(fa.df(u) * sb.sf(u)), 0, 100.0, limit=200)
    assert model.cif(100.0, "a") == pytest.approx(ref[0], abs=2e-6)
    total = model.cif(t, "a") + model.cif(t, "b")
    np.testing.assert_allclose(total, model.ff(t), atol=2e-6)
    assert model.cif(np.inf, "a") + model.cif(np.inf, "b") == pytest.approx(
        1.0, abs=2e-6
    )
    np.testing.assert_allclose(
        model.sf(t), model.models["a"].sf(t) * model.models["b"].sf(t)
    )
    # The maximum-likelihood machinery on the causes' predictive
    # distributions (Lomax, for Exponential causes with Gamma priors).
    expo, _ = exponential_fit
    exact = ParametricCompetingRisks.from_fitted(expo.models)
    for k in ("a", "b"):
        np.testing.assert_allclose(expo.cif(t, k), exact.cif(t, k), atol=2e-6)
    assert expo.probability_of_cause("a") == pytest.approx(
        exact.probability_of_cause("a"), abs=2e-6
    )


def test_exponential_closed_forms(exponential_fit):
    model, (x, e, c) = exponential_fit
    assert model.method == {"a": "conjugate", "b": "conjugate"}
    t = np.array([10.0, 100.0, 400.0])
    # Predictive survival: the product of two Lomax survivals,
    # (b / (b + t))^a with the Gamma posterior's shape a and rate b.
    expected = np.ones_like(t)
    for k, (a0, rate0) in {"a": (2.0, 300.0), "b": (3.0, 500.0)}.items():
        events = np.sum(e == k)
        a1, b1 = a0 + events, rate0 + x.sum()
        expected *= (b1 / (b1 + t)) ** a1
    np.testing.assert_allclose(model.sf(t), expected, rtol=1e-10)
    # Each joint draw's incidence against its closed form.
    d = model._joint()
    la = d["a"]["failure_rate"][:, None]
    lb = d["b"]["failure_rate"][:, None]
    closed = la / (la + lb) * -np.expm1(-(la + lb) * t[None, :])
    np.testing.assert_allclose(model.cif_draws(t, "a"), closed, atol=1e-12)
    # and the all-cause B10 life, -log(0.9) / (lambda_a + lambda_b)
    b10 = -np.log(0.9) / (la + lb)[:, 0]
    np.testing.assert_allclose(model.quantile_draws(0.1), b10, rtol=1e-10)


def test_draws_match_the_maximum_likelihood_model(weibull_fit):
    model, _ = weibull_fit
    d = model._joint()
    t = np.array([50.0, 100.0, 200.0, np.inf])
    cif = model.cif_draws(t, "a")
    q = model.quantile_draws(0.1)
    for i in (0, 17, 999):
        at = ParametricCompetingRisks.from_fitted(
            {
                k: sp.Weibull.from_params([d[k]["alpha"][i], d[k]["beta"][i]])
                for k in ("a", "b")
            }
        )
        np.testing.assert_allclose(cif[i], at.cif(t, "a"), atol=1e-8)
        assert q[i] == pytest.approx(at.qf(0.1), rel=1e-10)


def test_intervals(weibull_fit):
    model, _ = weibull_fit
    lo, hi = model.cri(100.0, "a")
    assert lo < model.cif(100.0, "a") < hi
    assert model.cri([50.0, 100.0], "a").shape == (2, 2)
    s_lo, s_hi = model.cri(100.0, on="sf")
    assert s_lo < model.sf(100.0) < s_hi
    f_lo, f_hi = model.cri(100.0, on="ff")
    assert f_lo == pytest.approx(1 - s_hi)
    # A cause's net sf is its own fit's interval.
    np.testing.assert_allclose(
        model.cri(100.0, "a", on="sf"), model.models["a"].cri(100.0)
    )
    b = model.quantile_cri([0.1, 0.5])
    assert b.shape == (2, 2) and np.all(b[:, 0] < b[:, 1])
    med = model.quantile_cri(0.1, point=True)
    assert b[0, 0] < med < b[0, 1]
    # The predictive quantile inverts the predictive ff.
    assert model.ff(model.qf(0.1)) == pytest.approx(0.1, abs=1e-9)
    upper = model.cri(100.0, "a", bound="upper", alpha_ci=0.05)
    assert upper == pytest.approx(
        model.cri(100.0, "a", alpha_ci=0.1)[1], rel=1e-12
    )
    # repeatable: the draws have a fixed seed
    again = ParametricCompetingRisks.fit_bayes(*_data()[:1], _data()[1])
    np.testing.assert_allclose(again.cri(100.0, "a"), (lo, hi))


def test_priors_a_cause_or_for_all():
    x, e, c = _data()
    prior = {"beta": sp.prior.around(2.0, between=(1.5, 3.0))}
    every = ParametricCompetingRisks.fit_bayes(x, e, c=c, priors=prior)
    one = ParametricCompetingRisks.fit_bayes(x, e, c=c, priors={"a": prior})
    np.testing.assert_allclose(
        every.models["a"].params, one.models["a"].params
    )
    assert every.models["b"].params[1] != pytest.approx(
        one.models["b"].params[1], rel=1e-6
    )
    with pytest.raises(ValueError, match="mixes causes"):
        ParametricCompetingRisks.fit_bayes(
            x, e, c=c, priors={"a": prior, "beta": prior["beta"]}
        )


def test_distribution_a_cause_and_truncation():
    x, e, c = _data()
    model = ParametricCompetingRisks.fit_bayes(
        x, e, c=c, dist={"a": sp.Weibull, "b": sp.Exponential}, tl=5.0
    )
    assert model.models["b"].dist is sp.Exponential
    keep = x > 5
    alone = sp.Exponential.fit_bayes(
        x[keep], np.where(e[keep] == "b", 0, 1), tl=5.0
    )
    np.testing.assert_allclose(
        model.models["b"].params, alone.params, rtol=1e-9
    )


def test_refusals():
    x = [[2, 4], [5, 5], [6, 8], [9, 9]]
    e = ["a", "b", "a", None]
    with pytest.raises(ValueError, match="window"):
        ParametricCompetingRisks.fit_bayes(x, e, c=[2, 0, 2, 1])
    model = ParametricCompetingRisks.fit_bayes(
        [1, 2, 3, 4, 5, 6],
        ["a", "b", "a", "b", "a", None],
        dist=sp.Exponential,
    )
    with pytest.raises(ValueError):
        model.cif(1.0, "c")
    with pytest.raises(ValueError):
        model.cri(1.0, "a", on="hf")


def test_saved_and_loaded(exponential_fit):
    model, _ = exponential_fit
    back = bayes.BayesianCompetingRisks.from_dict(model.to_dict())
    np.testing.assert_allclose(
        back.cif([10, 100], "a"), model.cif([10, 100], "a")
    )
    np.testing.assert_allclose(back.cri(100.0, "b"), model.cri(100.0, "b"))
    loaded = sp.from_dict(model.to_dict())
    assert type(loaded).__name__ == "BayesianCompetingRisks"
