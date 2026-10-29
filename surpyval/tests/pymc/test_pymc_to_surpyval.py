"""``to_surpyval``: PyMC's posterior draws back into a SurPyval Bayesian
model. Skipped where PyMC is not installed (it is not a test dependency
of SurPyval). The draws are made with fixed seeds, chosen before the
first run."""

import numpy as np
import pytest

import surpyval as sp

pm = pytest.importorskip("pymc")
az = pytest.importorskip("arviz")

from surpyval.pymc import to_surpyval  # noqa: E402


def _trace():
    # Two chains of 500 draws: a shared shape and two log scales.
    rng = np.random.default_rng(29)
    beta = rng.lognormal(np.log(2.0), 0.05, size=(2, 500))
    log_alpha = np.stack(
        [
            rng.normal(np.log(100.0), 0.05, size=(2, 500)),
            rng.normal(np.log(60.0), 0.1, size=(2, 500)),
        ],
        axis=-1,
    )
    trace = az.from_dict({"posterior": {"beta": beta, "log_alpha": log_alpha}})
    return trace, beta.ravel(), np.exp(log_alpha[..., 1]).ravel()


def test_draws_become_a_bayesian_model():
    trace, beta, alpha = _trace()
    model = to_surpyval(
        trace,
        sp.Weibull,
        alpha="log_alpha[1]",
        transform={"alpha": "exp"},
        beta="beta",
    )
    assert type(model).__name__ == "BayesianParametric"
    np.testing.assert_allclose(
        model.params, [np.median(alpha), np.median(beta)]
    )
    # sf is the posterior predictive: the mean over the draws.
    t = np.array([20.0, 50.0, 90.0])
    expected = np.exp(-((t[:, None] / alpha) ** beta)).mean(axis=1)
    np.testing.assert_allclose(model.sf(t), expected, rtol=1e-10)
    lower, upper = model.quantile_cri(0.10)
    assert lower < model.qf(0.10) < upper


def test_a_parameter_not_mapped_is_the_variable_of_its_name():
    rng = np.random.default_rng(3)
    trace = az.from_dict(
        {
            "posterior": {
                "alpha": rng.lognormal(np.log(100.0), 0.05, (1, 200)),
                "beta": rng.lognormal(np.log(2.0), 0.05, (1, 200)),
            }
        }
    )
    model = to_surpyval(trace, sp.Weibull)
    assert model.params[0] == pytest.approx(100.0, rel=0.02)


def test_a_multitrace_gives_the_same_model():
    with pm.Model():
        pm.LogNormal("alpha", mu=np.log(100.0), sigma=0.1)
        pm.LogNormal("beta", mu=np.log(2.0), sigma=0.1)
        idata = pm.sample(
            draws=100,
            tune=100,
            chains=1,
            cores=1,
            random_seed=4,
            progressbar=False,
            compute_convergence_checks=False,
        )
        with pytest.warns(FutureWarning):
            multitrace = pm.sample(
                draws=100,
                tune=100,
                chains=1,
                cores=1,
                random_seed=4,
                progressbar=False,
                compute_convergence_checks=False,
                return_inferencedata=False,
            )
    a = to_surpyval(idata, sp.Weibull)
    b = to_surpyval(multitrace, sp.Weibull)
    np.testing.assert_allclose(a.params, b.params)


def test_refusals():
    trace, _, _ = _trace()
    with pytest.raises(ValueError, match="no variable 'alpha'"):
        to_surpyval(trace, sp.Weibull)
    with pytest.raises(ValueError, match="needs 1 index"):
        to_surpyval(
            trace, sp.Weibull, alpha="log_alpha", transform={"alpha": "exp"}
        )
    with pytest.raises(ValueError, match="outside log_alpha"):
        to_surpyval(
            trace,
            sp.Weibull,
            alpha="log_alpha[2]",
            transform={"alpha": "exp"},
        )
    log_rate = np.random.default_rng(5).normal(np.log(0.01), 0.1, (1, 50))
    rates = az.from_dict({"posterior": {"log_rate": log_rate}})
    with pytest.raises(ValueError, match="Is a transform missing"):
        to_surpyval(rates, sp.Exponential, failure_rate="log_rate")
    model = to_surpyval(
        rates,
        sp.Exponential,
        failure_rate="log_rate",
        transform={"failure_rate": "exp"},
    )
    assert model.params[0] == pytest.approx(0.01, rel=0.05)
    with pytest.raises(TypeError, match="no parameter 'shape'"):
        to_surpyval(trace, sp.Weibull, shape="beta")
    with pytest.raises(ValueError, match="use one of identity, exp"):
        to_surpyval(
            trace,
            sp.Weibull,
            alpha="log_alpha[1]",
            transform={"alpha": "sqrt"},
        )
    with pytest.raises(TypeError, match="takes the trace"):
        to_surpyval({"alpha": [1.0]}, sp.Weibull)
