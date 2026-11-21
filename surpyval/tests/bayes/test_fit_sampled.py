"""``fit_bayes(method="nuts")``: the sampler against the closed forms
and the grid, its convergence report, and more than three free
parameters.

The rule for every comparison is the one of ``test_sampler.py``, fixed
before the first run with the seeds and sizes: within 4 Monte Carlo
standard errors of the reference, the error from the draws' own
effective sample size.
"""

import math
import pickle
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import surpyval as sp
from surpyval.bayes import convergence as cv
from surpyval.tests.bayes.test_sampler import within_monte_carlo_error


def chains_of(model, name):
    d = model.diagnostics
    col = model.posterior.parameter_names.index(name)
    return model.posterior.points[:, col].reshape(
        d["chains"], d["draws_per_chain"]
    )


def grid_cdf(grid, name):
    """The grid posterior's CDF of a parameter, by inverting its
    quantile function."""

    def cdf(q):
        def gap(p):
            return float(grid.posterior.quantile(name, p)) - q

        if gap(1e-9) > 0:
            return 0.0
        if gap(1 - 1e-9) < 0:
            return 1.0
        return brentq(gap, 1e-9, 1 - 1e-9, xtol=1e-12)

    return cdf


def censored_weibull():
    x = sp.Weibull.random(60, 100.0, 2.0, random_state=1)
    c = (x > 120).astype(int)
    return np.minimum(x, 120.0), c


BETA = {"beta": sp.prior.around(2, between=(1.2, 3.5))}


@pytest.fixture(scope="module")
def weibull_pair():
    x, c = censored_weibull()
    grid = sp.Weibull.fit_bayes(x, c, priors=BETA)
    sampled = sp.Weibull.fit_bayes(
        x, c, priors=BETA, method="nuts", random_state=459013
    )
    return grid, sampled


def test_exponential_with_a_gamma_prior_matches_the_closed_form():
    x = [120.0, 340.0, 410.0, 900.0, 1000.0, 1000.0]
    c = [0, 0, 0, 0, 1, 1]
    priors = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
    exact = sp.Exponential.fit_bayes(x, c, priors=priors)
    assert exact.method == "conjugate"
    model = sp.Exponential.fit_bayes(
        x, c, priors=priors, method="nuts", random_state=459011
    )
    assert model.method == "nuts" and model.ok
    family = exact.posterior._exact("failure_rate")
    within_monte_carlo_error(
        chains_of(model, "failure_rate"),
        family.cdf,
        float(family.mean()),
        float(family.std()),
    )


def test_normal_mean_with_sigma_known_matches_the_closed_form():
    x = [9.1, 10.4, 11.3, 8.7, 10.9, 9.9, 12.0]
    priors = {"mu": sp.Normal.from_params([8.0, 2.0])}
    kw = dict(priors=priors, fixed={"sigma": 1.5})
    exact = sp.Normal.fit_bayes(x, **kw)
    assert exact.method == "conjugate"
    model = sp.Normal.fit_bayes(x, method="nuts", random_state=459012, **kw)
    assert model.ok
    family = exact.posterior._exact("mu")
    within_monte_carlo_error(
        chains_of(model, "mu"),
        family.cdf,
        float(family.mean()),
        float(family.std()),
    )


def test_weibull_matches_the_grid(weibull_pair):
    grid, model = weibull_pair
    assert model.ok and model.diagnostics["divergences"] == 0
    for name in ("alpha", "beta"):
        col = grid.posterior.parameter_names.index(name)
        w, v = grid.posterior.weights, grid.posterior.points[:, col]
        mean = float(w @ v)
        sd = math.sqrt(float(w @ (v - mean) ** 2))
        within_monte_carlo_error(
            chains_of(model, name), grid_cdf(grid, name), mean, sd
        )


def test_lognormal_with_intervals_and_truncation_matches_the_grid():
    rng = np.random.default_rng(3)
    life = np.exp(rng.normal(4.0, 0.5, 50))
    life = life[life > 25.0]
    lo = np.floor(life / 20.0) * 20.0
    x = np.c_[np.maximum(lo, 25.0), lo + 20.0]
    kw = dict(xl=x[:, 0], xr=x[:, 1], tl=25.0)
    # (with no prior on mu the grid refuses these data: left truncation
    # leaves the posterior without finite mass)
    priors = {
        "mu": sp.prior.around(4, between=(3, 5)),
        "sigma": sp.prior.around(0.5, between=(0.2, 1.2)),
    }
    grid = sp.LogNormal.fit_bayes(priors=priors, **kw)
    model = sp.LogNormal.fit_bayes(
        priors=priors, method="nuts", random_state=459014, **kw
    )
    assert model.ok
    for name in ("mu", "sigma"):
        col = grid.posterior.parameter_names.index(name)
        w, v = grid.posterior.weights, grid.posterior.points[:, col]
        mean = float(w @ v)
        sd = math.sqrt(float(w @ (v - mean) ** 2))
        within_monte_carlo_error(
            chains_of(model, name), grid_cdf(grid, name), mean, sd
        )


def test_predictions_match_the_grid(weibull_pair):
    """The predictive functions and the intervals, which are sums over
    the draws: within 2% of the grid's (1,000 draws a chain)."""
    grid, model = weibull_pair
    x = np.array([30.0, 80.0, 150.0])
    assert model.sf(x) == pytest.approx(grid.sf(x), rel=0.02)
    assert model.qf(0.1) == pytest.approx(grid.qf(0.1), rel=0.02)
    assert model.params == pytest.approx(grid.params, rel=0.02)
    assert model.param_cri("beta") == pytest.approx(
        grid.param_cri("beta"), rel=0.03
    )


def test_the_report(weibull_pair):
    _, model = weibull_pair
    d = model.diagnostics
    assert d["method"] == "nuts" and d["chains"] == 4
    assert d["draws_per_chain"] == 1000 and d["warmup"] == 1000
    assert set(d["rhat"]) == {"alpha", "beta"}
    assert all(v < 1.01 for v in d["rhat"].values())
    assert all(v > 400 for v in d["ess_bulk"].values())
    assert all(v > 400 for v in d["ess_tail"].values())
    assert d["problems"] == [] and d["tree_depth_saturated"] == 0
    assert len(d["step_size"]) == 4
    # the numbers are those of the stored draws
    assert d["rhat"]["alpha"] == cv.rhat(chains_of(model, "alpha"))
    text = model.summary()
    assert "No-U-Turn sampler" in text and "4 chains of 1000 draws" in text
    assert np.isnan(model.log_evidence)


def test_saving_and_loading(weibull_pair):
    _, model = weibull_pair
    for again in (
        sp.from_dict(model.to_dict()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert again.method == "nuts"
        assert again.sf(90.0) == model.sf(90.0)
        assert again.diagnostics["rhat"] == model.diagnostics["rhat"]
        assert again.ok


def test_seeded_fits_do_not_depend_on_n_jobs():
    x, c = censored_weibull()
    kw = dict(
        priors=BETA,
        method="nuts",
        chains=2,
        draws=40,
        warmup=40,
        random_state=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = sp.Weibull.fit_bayes(x, c, **kw)
        two = sp.Weibull.fit_bayes(x, c, n_jobs=2, **kw)
        other = sp.Weibull.fit_bayes(x, c, **{**kw, "random_state": 6})
    assert np.array_equal(one.posterior.points, two.posterior.points)
    assert not np.array_equal(one.posterior.points, other.posterior.points)


def test_a_failed_check_warns_with_the_parameter_and_the_number():
    """Two chains of 20 draws cannot give 200 effective draws (the
    estimate is capped at ``draws * log10(draws)``, 64 here)."""
    x, c = censored_weibull()
    with pytest.warns(UserWarning) as caught:
        model = sp.Weibull.fit_bayes(
            x,
            c,
            priors=BETA,
            method="nuts",
            chains=2,
            draws=20,
            warmup=20,
            random_state=1,
        )
    said = [str(w.message) for w in caught if "sampler" in str(w.message)]
    assert len(said) == 1
    assert "effective sample size below 200: alpha " in said[0]
    assert "warmup=, draws=" in said[0] and "model.ok is False" in said[0]
    assert model.ok is False
    assert model.diagnostics["problems"]
    assert "NOT CONVERGED" in model.summary()
    with pytest.raises(ValueError, match="did not pass its checks"):
        model.raise_if_failed()


def test_more_than_three_free_parameters():
    """An offset and a limited-failure proportion with a Weibull: four
    free parameters, which the grid refuses and ``method="auto"`` now
    samples. (Short chains, to keep the test quick: whether they pass
    their checks is reported, not required.)"""
    x, c = censored_weibull()
    priors = {
        **BETA,
        "gamma": sp.prior.around(25, between=(10, 35)),
        "lfp_p": sp.prior.between(0.7, 0.999),
    }
    kw = dict(offset=True, lfp=True, priors=priors)
    with pytest.raises(ValueError, match="method='grid' takes 1 to 3"):
        sp.Weibull.fit_bayes(x + 30, c, method="grid", **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.Weibull.fit_bayes(
            x + 30,
            c,
            chains=2,
            draws=150,
            warmup=200,
            random_state=459015,
            **kw,
        )
    warned = [w for w in caught if "convergence checks" in str(w.message)]
    assert model.method == "nuts"
    assert model.ok == (not warned)
    assert model.posterior.free_names == ["gamma", "alpha", "beta", "lfp_p"]
    assert model.diagnostics["draws"] == 300
    lo, hi = model.param_cri("gamma")
    assert 0 < lo < model.gamma < hi < 30 + x.min()
    assert 0.5 < model.lfp_p < 1
    again = sp.from_dict(model.to_dict())
    assert again.sf(100.0) == model.sf(100.0)


def test_update_refits_with_the_sampler_and_its_settings():
    x, c = censored_weibull()
    kw = dict(priors=BETA, method="nuts", chains=2, draws=40, warmup=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = sp.Weibull.fit_bayes(x[:30], c[:30], random_state=1, **kw)
        both = first.update(x[30:], c[30:], random_state=2)
        once = sp.Weibull.fit_bayes(x, c, random_state=2, **kw)
    assert both.method == "nuts"
    assert both.diagnostics["draws_per_chain"] == 40
    assert np.array_equal(
        np.sort(both.posterior.points, axis=0),
        np.sort(once.posterior.points, axis=0),
    )


def test_refusals_say_what_to_do():
    x, c = censored_weibull()
    with pytest.raises(ValueError, match="chains must be a whole number"):
        sp.Weibull.fit_bayes(x, c, method="nuts", chains=0)
    with pytest.raises(ValueError, match="draws must be a whole number"):
        sp.Weibull.fit_bayes(x, c, method="nuts", draws=2.5)
    with pytest.raises(ValueError, match="neither the grid nor the sampler"):
        sp.Uniform.fit_bayes(x, method="nuts")
    with pytest.raises(ValueError, match="method must be one of .*nuts"):
        sp.Weibull.fit_bayes(x, c, method="hmc")
    with pytest.raises(ValueError, match="With no failures"):
        sp.Weibull.fit_bayes(x, np.ones_like(x), method="nuts")


def test_the_grid_is_still_the_default_up_to_three_parameters():
    x, c = censored_weibull()
    assert sp.Weibull.fit_bayes(x, c, priors=BETA).method == "grid"
