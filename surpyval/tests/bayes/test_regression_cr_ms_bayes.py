"""Bayesian competing-risks and multi-state regressions
(``ParametricCompetingRisksRegression.fit_bayes``,
``ParametricMultiStateRegression.fit_bayes``): compositions of each
cause's (transition's) regression ``fit_bayes``.

The references: each cause's or transition's own regression fit (the
likelihood factorises, so each posterior must be that fit's, point for
point); one cause or transition alone, which is the plain regression's
Bayesian fit; with flat priors, the maximum-likelihood estimates as the
posterior's peak; and the maximum-likelihood models' own incidence and
quantile at each draw's parameters.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import ParametricCompetingRisksRegression as CRR
from surpyval.multistate import ParametricMultiStateRegression as MSR
from surpyval.univariate.competing_risks.regression import bayes as crb


def _cr(n=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, (n, 1)).astype(float)
    t_a = rng.weibull(2.0, n) * 10 * np.exp(-0.4 * Z[:, 0])
    t_b = rng.exponential(20, n)
    x = np.minimum(t_a, t_b).round(3)
    e = np.where(t_a < t_b, "a", "b").astype(object)
    cen = x > 15
    x = np.where(cen, 15.0, x)
    e = np.where(cen, None, e)
    return x, Z, e, cen.astype(int)


def _quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kw)


@pytest.fixture(scope="module")
def cr_fit():
    x, Z, e, c = _cr()
    return _quiet(CRR.fit_bayes, x, Z, e, c=c), (x, Z, e, c)


def test_each_cause_is_its_own_regression_fit(cr_fit):
    model, (x, Z, e, c) = cr_fit
    assert model.method == {"a": "laplace", "b": "laplace"}
    for k in ("a", "b"):
        alone = _quiet(sp.WeibullPH.fit_bayes, x, Z, c=np.where(e == k, 0, 1))
        got = model.models[k]
        np.testing.assert_array_equal(
            got.posterior.points, alone.posterior.points
        )
        np.testing.assert_array_equal(
            got.posterior.weights, alone.posterior.weights
        )


def test_one_cause_alone_is_the_plain_regression():
    x, Z, e, c = _cr()
    e1 = np.where(c == 1, None, "a").astype(object)
    model = _quiet(CRR.fit_bayes, x, Z, e1, c=c)
    plain = _quiet(sp.WeibullPH.fit_bayes, x, Z, c=c)
    np.testing.assert_array_equal(
        model.models["a"].posterior.points, plain.posterior.points
    )
    t = np.array([2.0, 5.0, 10.0, 14.0])
    z = [1.0]
    np.testing.assert_allclose(
        model.cif(t, z, "a"), plain.ff(t, [z]), rtol=1e-12
    )
    np.testing.assert_allclose(model.sf(t, z), plain.sf(t, [z]), rtol=1e-12)
    # The interval from the joint draws (1,024) against the regression's
    # own (its 1,024 posterior points): the same posterior.
    np.testing.assert_allclose(
        model.cri(10.0, z, "a"), plain.cri(10.0, [z], on="ff"), atol=0.02
    )


def test_flat_priors_peak_at_the_maximum_likelihood_estimate():
    x, Z, e, c = _cr()
    flat = {"coef_0": sp.prior.flat()}
    model = _quiet(CRR.fit_bayes, x, Z, e, c=c, priors=flat)
    for k in ("a", "b"):
        mode = np.asarray(model.models[k].diagnostics["mode"])
        np.testing.assert_allclose(mode, model.mle.models[k].params, rtol=1e-4)


def test_draws_match_the_maximum_likelihood_model(cr_fit):
    model, _ = cr_fit
    unit = model.at([1.0])
    t = np.array([2.0, 5.0, 10.0, np.inf])
    cif = unit.cif_draws(t, "a")
    q = unit.quantile_draws(0.1)
    lives = unit._draw_lives()
    for i in (0, 17, 1000):
        at = CRR()
        at.causes = ["a", "b"]
        at.models = {k: lives[k][i] for k in ("a", "b")}
        np.testing.assert_allclose(cif[i], at.cif(t, [[1.0]], "a"), atol=1e-8)
        H = sum(float(lives[k][i].Hf(q[i], [[1.0]])) for k in "ab")
        assert -np.expm1(-H) == pytest.approx(0.1, rel=1e-8)


def test_the_unit_at_a_row_is_consistent(cr_fit):
    model, _ = cr_fit
    unit = model.at([0.0])
    t = np.array([1.0, 5.0, 10.0, 30.0])
    np.testing.assert_allclose(
        unit.cif(t, "a") + unit.cif(t, "b"), unit.ff(t), atol=2e-6
    )
    assert unit.probability_of_cause("a") + unit.probability_of_cause(
        "b"
    ) == pytest.approx(1.0, abs=2e-6)
    assert unit.ff(unit.qf(0.1)) == pytest.approx(0.1, abs=1e-9)
    lo, hi = unit.cri(10.0, "a")
    assert lo < unit.cif(10.0, "a") < hi
    b = unit.quantile_cri([0.1, 0.5])
    assert b.shape == (2, 2) and np.all(b[:, 0] < b[:, 1])
    # the covariate moves cause a's incidence up, as simulated
    assert model.cif(5.0, [1.0], "a") > model.cif(5.0, [0.0], "a")
    np.testing.assert_allclose(
        model.cri(10.0, [0.0], "a"), unit.cri(10.0, "a")
    )
    with pytest.raises(ValueError, match="one covariate row"):
        model.at([[0.0], [1.0]])
    back = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(
        back.cif(t, [0.0], "a"), model.cif(t, [0.0], "a")
    )
    assert "Cause 'a'" in model.summary()


def test_cause_priors():
    x, Z, e, c = _cr()
    tight = {"coef_0": sp.Normal.from_params([0.0, 0.05])}
    model = _quiet(CRR.fit_bayes, x, Z, e, c=c, priors={"b": tight})
    free = _quiet(CRR.fit_bayes, x, Z, e, c=c)
    lo, hi = model.param_cri("b", "coef_0")
    flo, fhi = free.param_cri("b", "coef_0")
    assert hi - lo < fhi - flo
    np.testing.assert_array_equal(
        model.models["a"].posterior.points, free.models["a"].posterior.points
    )


# -- multi-state -----------------------------------------------------------


def _two_states(n=120, seed=3):
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, (n, 1)).astype(float)
    t = sp.Weibull.random(n, 50, 1.5, random_state=seed) * np.exp(
        -0.5 * Z[:, 0]
    )
    x = np.minimum(t, 80.0)
    e = np.where(t < 80.0, "failed", None).astype(object)
    return {
        "x": x,
        "Z": Z,
        "e": e,
        "state": np.full(n, "working", dtype=object),
        "tl": np.zeros(n),
        "transitions": [("working", "failed")],
    }


def test_two_states_are_the_plain_regression():
    d = _two_states()
    model = _quiet(MSR.fit_bayes, **d)
    c = (d["e"] == None).astype(int)  # noqa: E711
    plain = _quiet(sp.WeibullPH.fit_bayes, d["x"], d["Z"], c=c)
    got = model.models[("working", "failed")]
    np.testing.assert_allclose(
        got.posterior.points, plain.posterior.points, rtol=1e-10
    )
    t = np.array([10.0, 30.0, 60.0])
    occ = model.occupancy(t, [1.0], "failed")
    # the mean over the joint draws of each draw's failure probability
    draws = model._draw_models()
    mean = np.mean([m[0].ff(t, [[1.0]]) for m in draws], axis=0)
    np.testing.assert_allclose(occ, mean, atol=1e-6)
    # and, to the Monte Carlo error of 1,000 draws, the predictive
    np.testing.assert_allclose(occ, plain.ff(t, [[1.0]]), atol=0.01)
    np.testing.assert_allclose(
        model.occupancy(t, [1.0], "working") + occ, 1.0, atol=1e-6
    )


@pytest.fixture(scope="module")
def ms_fit():
    from surpyval.tests.conformance.registry_fixtures import ms_data

    d = ms_data(True)
    return _quiet(MSR.fit_bayes, **{**d, "model": sp.ExponentialPH}), d


def test_each_transition_is_its_own_regression_fit(ms_fit):
    model, d = ms_fit
    states = np.asarray(d["state"])
    for (a, b), fit in model.models.items():
        rows = states == a
        c = np.where(np.asarray(d["e"])[rows] == b, 0, 1)
        tl = np.asarray(d["tl"], dtype=float)[rows]
        tl = np.where(tl <= 0, -np.inf, tl)
        extra = {"tl": tl} if np.any(np.isfinite(tl)) else {}
        alone = _quiet(
            sp.ExponentialPH.fit_bayes,
            np.asarray(d["x"])[rows],
            np.asarray(d["Z"])[rows],
            c=c,
            n=np.asarray(d["n"])[rows],
            **extra,
        )
        np.testing.assert_allclose(
            fit.posterior.points, alone.posterior.points, rtol=1e-10
        )


def test_multistate_flat_priors_and_predictions(ms_fit):
    from surpyval.tests.conformance.registry_fixtures import ms_data

    d = ms_data(True)
    flat = {"z0": sp.prior.flat()}
    model = _quiet(
        MSR.fit_bayes, **{**d, "model": sp.ExponentialPH, "priors": flat}
    )
    for pair, fit in model.models.items():
        np.testing.assert_allclose(
            np.asarray(fit.diagnostics["mode"]),
            model.mle.models[pair].params,
            rtol=1e-4,
        )
    fitted, _ = ms_fit
    t = np.array([2.0, 5.0])
    total = sum(
        fitted.occupancy(t, [0.5], s) for s in ("healthy", "ill", "dead")
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-8)
    lo, hi = fitted.occupancy_cri(5.0, [0.5], "dead")
    assert lo <= fitted.occupancy(5.0, [0.5], "dead") <= hi
    P = fitted.transition_probabilities(5.0, [0.5])
    np.testing.assert_allclose(P.sum(axis=-1), 1.0, atol=1e-8)
    assert fitted.length_of_stay(5.0, [0.5], "healthy") > 0
    back = sp.from_dict(fitted.to_dict())
    np.testing.assert_allclose(
        back.occupancy(t, [0.5], "dead"), fitted.occupancy(t, [0.5], "dead")
    )


def test_draws_constant():
    assert crb.DRAWS == 1024
