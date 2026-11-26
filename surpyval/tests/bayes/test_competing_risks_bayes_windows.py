"""Bayesian competing risks with failures found in a window (``c`` of -1
or 2): every cause's parameters in one joint target.

The references:

- the maximum-likelihood fit's own joint likelihood
  (``_interval_likelihood.neg_ll``, with its quadrature refined at each
  parameter point): the target's log-likelihood equals it;
- a window's probability by adaptive quadrature in time of the
  cause's sub-density, ``int_a^b h_k(u) S(u) du`` (``scipy.integrate``);
- with flat priors, the posterior's peak is the maximum-likelihood
  estimate;
- with exact and right-censored data only, the joint target gives the
  factorised posterior (each cause's own ``fit_bayes``);
- a seeded simulation, whose true values the 95% intervals cover.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval as sp
from surpyval.univariate.competing_risks import ParametricCompetingRisks
from surpyval.univariate.competing_risks.parametric import (
    _interval_likelihood as joint,
)
from surpyval.univariate.competing_risks.parametric import bayes
from surpyval.univariate.competing_risks.parametric import (
    parametric_competing_risks as pcr,
)
from surpyval.univariate.competing_risks.parametric._joint_bayes import (
    JointCompetingRisksTarget,
    window_log_probability,
)

X = [[2, 2], [2, 4], [4, 6], [5, 5], [6, 8], [7, 7], [8, 10], [9, 9]]
X += [[10, 10], [12, 12]]
E = ["a", "b", "a", "b", "a", "a", "b", None, "a", None]
C = [-1, 2, 2, 0, 2, 0, 2, 1, 0, 1]


def _inspected(n, seed, every=20.0, end=120.0):
    """Two causes, Weibull(100, 2) and Exponential(rate 1/150); every
    other unit inspected every ``every`` hours (found failed in a
    window), the rest seen failing; all stopped at ``end``."""
    ta = sp.Weibull.random(n, 100, 2, random_state=seed)
    tb = sp.Exponential.random(n, 1 / 150, random_state=seed + 1)
    t = np.minimum(ta, tb)
    cause = np.where(ta < tb, "a", "b").astype(object)
    x, e, c = [], [], []
    for i in range(n):
        if t[i] > end:
            x.append([end, end])
            e.append(None)
            c.append(1)
        elif i % 2 == 0:
            x.append([t[i], t[i]])
            e.append(cause[i])
            c.append(0)
        else:
            lo = np.floor(t[i] / every) * every
            if lo > 0:
                x.append([lo, lo + every])
                c.append(2)
            else:
                x.append([every, every])
                c.append(-1)
            e.append(cause[i])
    return np.array(x), np.array(e, dtype=object), np.array(c)


def _quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kw)


@pytest.mark.parametrize(
    "dists, tl",
    [
        ((sp.Exponential, sp.Exponential), None),
        ((sp.Weibull, sp.Exponential), None),
        ((sp.Weibull, sp.LogNormal), 1.0),
    ],
)
def test_target_is_the_maximum_likelihood_fits_likelihood(dists, tl):
    x, c, n, e, t = pcr._validate(X, C, None, E, tl)
    mle = _quiet(
        ParametricCompetingRisks.fit,
        X,
        E,
        c=C,
        dist=dict(zip("ab", dists)),
        tl=tl,
    )
    target = JointCompetingRisksTarget(
        ["a", "b"], list(dists), x, c, n, e, t, mle
    )
    theta0 = target.estimates()
    rng = np.random.default_rng(7)
    for s in range(4):
        theta = theta0 * np.exp(rng.normal(0, 0.3, theta0.size))
        data = joint.prepare(["a", "b"], list(dists), x, c, n, e, t)
        data.adapt(theta)
        ref = -float(joint.neg_ll(theta, data))
        got = target.log_lik(theta[None, :])[0]
        assert got == pytest.approx(ref, abs=1e-10)
    # many points at once, as the grid evaluates them
    points = theta0 * np.exp(rng.normal(0, 0.2, (5, theta0.size)))
    rows = np.array([target.log_lik(p[None, :])[0] for p in points])
    np.testing.assert_allclose(target.log_lik(points), rows, rtol=1e-13)


@pytest.mark.parametrize(
    "da, pa, db, pb, a, b",
    [
        (sp.Weibull, [100.0, 2.0], sp.Exponential, [1 / 150], 40.0, 60.0),
        (sp.Weibull, [100.0, 0.7], sp.Weibull, [80.0, 3.0], -np.inf, 20.0),
        (sp.LogNormal, [4.0, 0.5], sp.Weibull, [60.0, 1.5], 50.0, 90.0),
        (sp.Weibull, [100.0, 2.0], sp.Exponential, [1 / 150], 300.0, 310.0),
    ],
)
def test_a_window_is_the_cause_sub_density_integrated(da, pa, db, pb, a, b):
    """For cause a in (a, b]: int h_a(u) S_a(u) S_b(u) du, by adaptive
    quadrature in time; not the net F_a(b) - F_a(a)."""

    def sub_density(u):
        return float(da.df(u, *pa) * db.sf(u, *pb))

    lo = 0.0 if not np.isfinite(a) else a
    ref, err = quad(sub_density, lo, b, epsabs=0, epsrel=1e-13, limit=500)
    params = [
        [np.array([[v]]) for v in pa],
        [np.array([[v]]) for v in pb],
    ]
    got = np.exp(
        window_log_probability(
            0, np.array([a]), np.array([b]), [da, db], [0.0, 0.0], params
        )[0, 0]
    )
    assert got == pytest.approx(ref, rel=1e-9)
    net = float(da.ff(b, *pa)) - (float(da.ff(a, *pa)) if a > 0 else 0.0)
    assert abs(net - got) > 1e-3 * got


@pytest.mark.parametrize(
    "dist, method",
    [
        (sp.Exponential, "grid"),
        ({"a": sp.Weibull, "b": sp.Exponential}, "grid"),
        (sp.Weibull, "laplace"),
    ],
)
def test_flat_priors_peak_at_the_maximum_likelihood_estimate(dist, method):
    x, e, c = _inspected(80, seed=3)
    model = _quiet(ParametricCompetingRisks.fit_bayes, x, e, c=c, dist=dist)
    assert model.joint
    assert model.method == {"a": method, "b": method}
    mode = np.asarray(model.result.diagnostics["mode"])
    np.testing.assert_allclose(mode, model.mle.params, rtol=2e-5)


def test_exact_data_reproduce_the_factorised_posterior():
    n, seed = 60, 1
    ta = sp.Weibull.random(n, 100, 2, random_state=seed)
    tb = sp.Weibull.random(n, 150, 1.2, random_state=seed + 1)
    cen = np.full(n, 120.0)
    x = np.minimum.reduce([ta, tb, cen])
    e = np.where(x == cen, None, np.where(ta < tb, "a", "b")).astype(object)
    c = (x == cen).astype(int)
    t = np.array([20.0, 50.0, 100.0, 200.0])
    for dist in (sp.Exponential, {"a": sp.Weibull, "b": sp.Exponential}):
        alone = ParametricCompetingRisks.fit_bayes(x, e, c=c, dist=dist)
        both = bayes.fit_joint(
            ParametricCompetingRisks,
            pcr._validate(x, c, None, e, None),
            dist,
            None,
        )
        assert both.joint and both.method == {"a": "grid", "b": "grid"}
        np.testing.assert_allclose(both.sf(t), alone.sf(t), atol=1e-7)
        for k in ("a", "b"):
            # (the factorised incidence is good to about 1e-6)
            np.testing.assert_allclose(
                both.cif(t, k), alone.cif(t, k), atol=2e-6
            )
            for name in alone.models[k].posterior.parameter_names:
                assert both.models[k].posterior.mean(name) == pytest.approx(
                    alone.models[k].posterior.mean(name), rel=1e-8
                )
                np.testing.assert_allclose(
                    both.param_cri(k, name),
                    alone.param_cri(k, name),
                    rtol=2e-3,
                )
        # the joint log-likelihood is the causes' sum
        target = both.result.target
        theta = target.estimates()[None, :]
        parts = -sum(float(both.mle.models[k].neg_ll()) for k in "ab")
        assert target.log_lik(theta)[0] == pytest.approx(parts, rel=1e-10)


@pytest.fixture(scope="module")
def recovered():
    x, e, c = _inspected(400, seed=2026)
    return _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist={"a": sp.Weibull, "b": sp.Exponential},
    )


def test_seeded_recovery(recovered):
    model = recovered
    assert model.method == {"a": "grid", "b": "grid"}
    for k, name, truth in (
        ("a", "alpha", 100.0),
        ("a", "beta", 2.0),
        ("b", "failure_rate", 1 / 150),
    ):
        lo, hi = model.param_cri(k, name)
        assert lo < truth < hi
    exact = ParametricCompetingRisks.from_fitted(
        {
            "a": sp.Weibull.from_params([100.0, 2.0]),
            "b": sp.Exponential.from_params([1 / 150]),
        }
    )
    for k in ("a", "b"):
        lo, hi = model.cri(100.0, k)
        assert lo < float(exact.cif(100.0, k)) < hi


def test_the_joint_model_is_consistent(recovered):
    model = recovered
    t = np.array([10.0, 50.0, 100.0, 250.0])
    total = model.cif(t, "a") + model.cif(t, "b")
    np.testing.assert_allclose(total, model.ff(t), atol=1e-8)
    assert model.probability_of_cause("a") + model.probability_of_cause(
        "b"
    ) == pytest.approx(1.0, abs=1e-8)
    assert model.ff(model.qf(0.1)) == pytest.approx(0.1, abs=1e-9)
    # the predictive hazard is the predictive density over the survival
    np.testing.assert_allclose(
        model.hf(t), model.df(t) / model.sf(t), rtol=1e-12
    )
    # the predictive survival is the posterior mean of the product
    th, w = model._support()
    s = np.ones((w.size, t.size))
    s = s * sp.Weibull.sf(
        t, th["a"]["alpha"][:, None], th["a"]["beta"][:, None]
    )
    s = s * sp.Exponential.sf(t, th["b"]["failure_rate"][:, None])
    np.testing.assert_allclose(model.sf(t), w @ s, rtol=1e-12)
    # each joint draw's incidence is the maximum-likelihood model's
    d = model._joint()
    cif = model.cif_draws(t, "a")
    for i in (0, 123):
        at = ParametricCompetingRisks.from_fitted(
            {
                "a": sp.Weibull.from_params(
                    [d["a"]["alpha"][i], d["a"]["beta"][i]]
                ),
                "b": sp.Exponential.from_params([d["b"]["failure_rate"][i]]),
            }
        )
        np.testing.assert_allclose(cif[i], at.cif(t, "a"), atol=1e-8)
    lo, hi = model.quantile_cri(0.1)
    assert lo < model.quantile_cri(0.1, point=True) < hi
    assert "fitted together" in model.summary()


def test_priors_saving_and_refusals(recovered):
    x, e, c = _inspected(80, seed=3)
    prior = {"beta": sp.prior.around(2.0, between=(1.5, 3.0))}
    model = _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist={"a": sp.Weibull, "b": sp.Exponential},
        priors={"a": prior},
    )
    assert "a.beta" in model.result.priors.single
    flat = _quiet(
        ParametricCompetingRisks.fit_bayes,
        x,
        e,
        c=c,
        dist={"a": sp.Weibull, "b": sp.Exponential},
    )
    lo, hi = model.param_cri("a", "beta")
    flo, fhi = flat.param_cri("a", "beta")
    assert hi - lo < fhi - flo
    back = sp.from_dict(model.to_dict())
    assert back.joint
    t = [20.0, 80.0]
    np.testing.assert_allclose(back.cif(t, "a"), model.cif(t, "a"))
    np.testing.assert_allclose(back.cri(80.0, "b"), model.cri(80.0, "b"))
    np.testing.assert_allclose(back.sf(t), model.sf(t))
    with pytest.raises(ValueError, match="no parameter 'shape'"):
        ParametricCompetingRisks.fit_bayes(
            x, e, c=c, priors={"a": {"shape": prior["beta"]}}
        )
    belief = sp.prior.on_quantities(
        sp.Weibull, B10=sp.prior.around(40, between=(20, 80))
    )
    with pytest.raises(ValueError, match="a parameter at a time"):
        ParametricCompetingRisks.fit_bayes(x, e, c=c, priors={"a": belief})
    # a window from an unknown cause is refused, as by fit
    with pytest.raises(ValueError, match="missing event type"):
        ParametricCompetingRisks.fit_bayes(
            [[2, 4], [5, 5], [6, 8]], [None, "b", "a"], c=[2, 0, 2]
        )
    assert np.isfinite(recovered.result.waic().elpd)
