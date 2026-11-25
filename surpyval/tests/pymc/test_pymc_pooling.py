"""Sampling with the bridge: a regression against SurPyval's Laplace
engine, and groups sharing a population (``pooled_likelihood``,
``group_models``). Skipped where PyMC is not installed (it is not a test
dependency of SurPyval).

The data, seeds and rules were fixed before the first run:

- a Weibull proportional hazards model, 80 units, weak priors, sampled by
  PyMC (2 chains of 1,000 draws after 1,000 tuning steps, seed 461001):
  each parameter's 5%, 50% and 95% points within 0.15 posterior standard
  deviations of ``fit_bayes(..., importance=True)``'s;
- eight sites of Weibull lives, the sites' alphas log-normal about 1,000
  with a spread of 0.3 on the log scale, a shape of 2; six sites of 30
  units and two of 4 (seed 461010; sampled with seed 461011): the 95%
  interval of the spread ``tau`` holds 0.3; each small site's posterior
  median alpha lies between its own maximum-likelihood alpha and the
  population's median; and the new group's 90% interval of alpha is
  wider than every observed group's.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp

pm = pytest.importorskip("pymc")

from surpyval.pymc import (  # noqa: E402
    censored_likelihood,
    group_models,
    pooled_likelihood,
)


def _ph_data():
    rng = np.random.default_rng(461000)
    Z = rng.normal(size=(80, 1))
    life = sp.Weibull.random(80, 100.0, 1.8, random_state=461002)
    life = life * np.exp(-0.5 * Z[:, 0] / 1.8)
    x = np.minimum(life, 150.0)
    return x, (life > 150.0).astype(int), Z


def test_a_regression_sampled_by_pymc_agrees_with_the_laplace_engine():
    x, c, Z = _ph_data()
    priors = {
        "alpha": sp.LogNormal.from_params([np.log(100.0), 1.0]),
        "beta": sp.LogNormal.from_params([np.log(1.5), 0.5]),
        "coef_0": sp.Normal.from_params([0.0, 1.0]),
    }
    laplace = sp.WeibullPH.fit_bayes(x, Z, c, priors=priors, importance=True)
    with pm.Model():
        alpha = pm.LogNormal("alpha", mu=np.log(100.0), sigma=1.0)
        beta = pm.LogNormal("beta", mu=np.log(1.5), sigma=0.5)
        coef = pm.Normal("coef_0", mu=0.0, sigma=1.0)
        censored_likelihood(
            sp.WeibullPH, x=x, c=c, Z=Z, alpha=alpha, beta=beta, coef_0=coef
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                1000,
                tune=1000,
                chains=2,
                cores=1,
                random_seed=461001,
                progressbar=False,
                compute_convergence_checks=False,
            )
    q = [0.05, 0.5, 0.95]
    for name in ("alpha", "beta", "coef_0"):
        draws = trace.posterior[name].values.ravel()
        sd = float(draws.std())
        gap = np.abs(
            np.quantile(draws, q) - laplace.posterior.quantile(name, q)
        )
        assert np.all(gap <= 0.15 * sd), (name, gap / sd)


def _sites():
    rng = np.random.default_rng(461010)
    alphas = np.exp(np.log(1000.0) + 0.3 * rng.normal(size=8))
    sizes = [30, 30, 30, 30, 30, 30, 4, 4]
    x, c, group = [], [], []
    for k, (a, m) in enumerate(zip(alphas, sizes)):
        life = sp.Weibull.random(m, a, 2.0, random_state=461020 + k)
        x.append(np.minimum(life, 1500.0))
        c.append((life > 1500.0).astype(int))
        group += [f"site {k}"] * m
    return np.concatenate(x), np.concatenate(c), np.array(group)


@pytest.fixture(scope="module")
def pooled():
    x, c, group = _sites()
    with pm.Model():
        beta = pm.LogNormal("beta", mu=np.log(2.0), sigma=0.5)
        pooling = pooled_likelihood(
            sp.Weibull,
            x=x,
            c=c,
            group=group,
            pool={"alpha": (np.log(1000.0), 1.0)},
            shared={"beta": beta},
            tau_scale=0.5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                1000,
                tune=1000,
                chains=2,
                cores=1,
                random_seed=461011,
                progressbar=False,
                target_accept=0.95,
                compute_convergence_checks=False,
            )
    return x, c, group, pooling, trace


def test_the_spread_between_sites_is_recovered(pooled):
    _, _, _, pooling, trace = pooled
    tau = trace.posterior[pooling.pooled["alpha"]["tau"]].values.ravel()
    lo, hi = np.quantile(tau, [0.025, 0.975])
    assert lo < 0.3 < hi


def test_small_sites_borrow_strength(pooled):
    x, c, group, pooling, trace = pooled
    models = group_models(trace, pooling, random_state=461012)
    mu = trace.posterior[pooling.pooled["alpha"]["mu"]].values.ravel()
    population = float(np.exp(np.median(mu)))
    for site in ("site 6", "site 7"):
        rows = group == site
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = float(
                sp.Weibull.fit(x[rows], c[rows], fixed={"beta": 2.0}).params[0]
            )
        pulled = float(models[site].params[0])
        assert min(alone, population) < pulled < max(alone, population), (
            site,
            alone,
            pulled,
            population,
        )


def test_a_new_site_is_less_certain_than_any_seen(pooled):
    _, _, _, pooling, trace = pooled
    models = group_models(trace, pooling, random_state=461012)
    assert set(models) == {f"site {k}" for k in range(8)} | {"new group"}

    def width(m):
        lo, hi = m.param_cri("alpha", alpha_ci=0.1)
        return float(np.log(hi) - np.log(lo))

    new = width(models["new group"])
    assert all(new > width(models[f"site {k}"]) for k in range(8))
    # each group's model predicts as a SurPyval model does
    assert 0 < float(models["site 0"].sf(1000.0)) < 1
    assert models["site 0"].data is not None


def test_pooling_refusals():
    x, c, group = _sites()
    with pm.Model():
        with pytest.raises(TypeError, match="pool or in shared, once"):
            pooled_likelihood(
                sp.Weibull,
                x=x,
                c=c,
                group=group,
                pool={"alpha": (7.0, 1.0)},
            )
        with pytest.raises(ValueError, match="one label per data row"):
            pooled_likelihood(
                sp.Weibull,
                x=x,
                c=c,
                group=group[:5],
                pool={"alpha": (7.0, 1.0)},
                shared={"beta": 2.0},
            )
        with pytest.raises(ValueError, match="nothing to pool"):
            pooled_likelihood(
                sp.Weibull,
                x=x,
                c=c,
                group=np.full(len(x), "one"),
                pool={"alpha": (7.0, 1.0)},
                shared={"beta": 2.0},
            )
    with pytest.raises(TypeError, match="inside"):
        pooled_likelihood(
            sp.Weibull,
            x=x,
            group=group,
            pool={"alpha": (7.0, 1.0)},
            shared={"beta": 2.0},
        )
    with pm.Model():
        with pytest.warns(UserWarning, match="have no failures"):
            pooled_likelihood(
                sp.Weibull,
                x=np.r_[x, 100.0, 200.0],
                c=np.r_[c, 1, 1],
                group=np.concatenate([group, ["quiet", "quiet"]]),
                pool={"alpha": (7.0, 1.0)},
                shared={"beta": 2.0},
            )
