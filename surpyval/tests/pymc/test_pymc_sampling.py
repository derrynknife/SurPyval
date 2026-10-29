"""Sampling with ``censored_likelihood`` recovers known parameters, and the
prior helpers build what they say. Skipped where PyMC is not installed
(it is not a test dependency of SurPyval).

The seeds, sample sizes and tolerances were fixed before the first run.
"""

import numpy as np
import pytest

import surpyval as sp

pm = pytest.importorskip("pymc")

from surpyval.pymc import (  # noqa: E402
    censored_likelihood,
    prior_from_model,
    prior_from_spec,
)


def _censored_truncated_weibull():
    # 300 lifetimes from a Weibull(100, 2), seen only from age 20 (left
    # truncation), censored at 150.
    rng = np.random.default_rng(2026)
    life = 100.0 * rng.weibull(2.0, size=600)
    life = life[life > 20.0][:300]
    c = (life > 150.0).astype(int)
    x = np.minimum(life, 150.0)
    return x, c, np.full(x.shape, 20.0)


def test_sampling_recovers_known_weibull_parameters():
    x, c, tl = _censored_truncated_weibull()
    with pm.Model():
        alpha = pm.LogNormal("alpha", mu=np.log(80.0), sigma=1.0)
        beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)
        censored_likelihood(
            sp.Weibull, x=x, c=c, tl=tl, alpha=alpha, beta=beta, name="d"
        )
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            random_seed=1,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = trace.posterior
    mle = sp.Weibull.fit(x=x, c=c, tl=tl)
    for i, (name, true) in enumerate((("alpha", 100.0), ("beta", 2.0))):
        draws = post[name].values.ravel()
        mean, sd = draws.mean(), draws.std()
        # Within three posterior standard deviations of the truth, and,
        # with priors this weak on 300 rows, close to the MLE.
        assert abs(mean - true) < 3 * sd, (name, mean, sd)
        assert abs(mean - mle.params[i]) < 0.5 * sd, (name, mean, sd)


def test_find_map_with_flat_priors_is_the_mle():
    x, c, tl = _censored_truncated_weibull()
    with pm.Model():
        log_alpha = pm.Flat("log_alpha")
        log_beta = pm.Flat("log_beta")
        censored_likelihood(
            sp.Weibull,
            x=x,
            c=c,
            tl=tl,
            alpha=pm.math.exp(log_alpha),
            beta=pm.math.exp(log_beta),
        )
        found = pm.find_MAP(
            start={"log_alpha": np.log(80.0), "log_beta": 0.0},
            progressbar=False,
        )
    mle = sp.Weibull.fit(x=x, c=c, tl=tl)
    got = np.exp([found["log_alpha"], found["log_beta"]])
    np.testing.assert_allclose(got, mle.params, rtol=1e-4)


def test_censored_likelihood_outside_a_model_is_refused():
    with pytest.raises(TypeError, match="inside `with pm.Model"):
        censored_likelihood(sp.Weibull, x=[1.0, 2.0], alpha=1.0, beta=1.0)


def test_prior_from_model_carries_the_covariance_on_the_log_scale():
    old = sp.Weibull.fit(sp.Weibull.random(40, 100.0, 2.0, random_state=3))
    with pm.Model() as m:
        prior = prior_from_model(old, name="fleet", weight=0.5)
    assert set(prior) == {"alpha", "beta"}
    z = m["fleet_z"]
    mu, cov = z.owner.inputs[-2:]
    jac = 1.0 / np.asarray(old.params)
    np.testing.assert_allclose(mu.eval(), np.log(old.params))
    np.testing.assert_allclose(
        cov.eval(), old.covariance() * np.outer(jac, jac) / 0.5
    )


def test_prior_from_model_refusals():
    old = sp.Weibull.fit([310, 520, 640, 790, 905, 1100, 1240])
    with pm.Model():
        with pytest.raises(ValueError, match=r"in \(0, 1\]"):
            prior_from_model(old, weight=2.0)
        with pytest.raises(ValueError, match="needs the fit's covariance"):
            prior_from_model(sp.Weibull.from_params([100.0, 2.0]))
        with pytest.raises(TypeError, match="fitted SurPyval model"):
            prior_from_model(sp.Weibull)
        with pytest.raises(ValueError, match="an offset"):
            prior_from_model(
                sp.Weibull.fit([310, 520, 640, 790, 905], offset=True)
            )


def test_prior_from_spec_uses_surpyval_parameter_order():
    with pm.Model() as m:
        prior_from_spec(
            {
                "alpha": ("weibull", 100.0, 2.0),
                "beta": {"dist": "Gamma", "params": [4.0, 2.0]},
            },
            dist=sp.Weibull,
        )
    # SurPyval's Weibull (scale 100, shape 2) is PyMC's Weibull(alpha=2,
    # beta=100); SurPyval's Gamma has rate beta, as PyMC's.
    alpha_shape, alpha_scale = m["alpha"].owner.inputs[-2:]
    assert float(alpha_shape.eval()) == 2.0
    assert float(alpha_scale.eval()) == 100.0
    draws = pm.draw(m["beta"], draws=20000, random_seed=5)
    assert draws.mean() == pytest.approx(2.0, rel=0.02)


def test_prior_from_spec_refusals():
    with pm.Model():
        with pytest.raises(ValueError, match="puts mass below it"):
            prior_from_spec({"beta": ("normal", 2.0, 1.0)}, dist=sp.Weibull)
        with pytest.raises(ValueError, match="no parameter 'shape'"):
            prior_from_spec({"shape": ("lognormal", 0, 1)}, dist=sp.Weibull)
        with pytest.raises(ValueError, match="no PyMC prior for 'Rayleigh'"):
            prior_from_spec({"beta": ("Rayleigh", 1.0)})
        with pytest.raises(ValueError, match="takes 2 parameters"):
            prior_from_spec({"beta": ("lognormal", 1.0)})
