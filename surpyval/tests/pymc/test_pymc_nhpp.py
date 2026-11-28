"""The power-law NHPP likelihood in PyMC (``nhpp_log_likelihood``,
``nhpp_likelihood``), fleet pooling of growth rates
(``pooled_likelihood(CrowAMSAA, ...)``), the centred form and the
shrinkage table. Skipped where PyMC is not installed (it is not a test
dependency of SurPyval).

The data, seeds and rules were fixed before the first run (26 Nov
2026): eight systems sharing ``alpha = 20``, their growth rates
``beta`` log-normal about 0.8 with a spread of 0.3 on the log scale
(seed 700010); six watched to 1,000 and two to 150 (a handful of events
each); sampled by PyMC (2 chains of 500 draws after 500 tuning steps,
seed 700011). Checked: the 95% interval of the spread ``tau`` holds 0.3;
each of the two short systems' posterior median ``beta`` lies between
its own maximum-likelihood ``beta`` and the population's median; and a
new system's 90% interval of ``beta`` is wider than every observed
system's.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp

pm = pytest.importorskip("pymc")

from surpyval.pymc import (  # noqa: E402
    group_models,
    nhpp_likelihood,
    nhpp_log_likelihood,
    pooled_likelihood,
    shrinkage,
)
from surpyval.recurrent import CrowAMSAA  # noqa: E402
from surpyval.recurrent.bayes import (  # noqa: E402
    BayesianRecurrence,
    RecurrenceTarget,
)

X = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
C = [0] * 11 + [1]


def _own(x, i=None, c=None, n=None, params=(50.0, 0.8)):
    fit = CrowAMSAA.fit(x, i=i, c=c, n=n)
    return -RecurrenceTarget(fit.dist, fit.data).neg_ll(list(params))


def _fleet(seed=700010):
    rng = np.random.default_rng(seed)
    alpha = 20.0
    betas = np.exp(np.log(0.8) + 0.3 * rng.standard_normal(8))
    ends = np.r_[[1000.0] * 6, [150.0] * 2]
    xs, ii, cs = [], [], []
    for s in range(8):
        expected = (ends[s] / alpha) ** betas[s]
        m = rng.poisson(expected)
        u = np.sort(rng.uniform(0, expected, m))
        xs += list(alpha * u ** (1 / betas[s])) + [ends[s]]
        ii += [s] * (m + 1)
        cs += [0] * m + [1]
    return np.array(xs), np.array(ii), np.array(cs), betas


def test_the_likelihood_is_the_fitters():
    for params in [(50.0, 0.8), (10.0, 1.4), (300.0, 0.3)]:
        ll = nhpp_log_likelihood(X, c=C, alpha=params[0], beta=params[1])
        assert np.isclose(ll.eval(), _own(X, c=C, params=params), rtol=1e-12)


def test_counts_in_windows_and_the_gradient():
    # three windows of counts, then the end of observation
    x = np.array([[0, 100], [100, 200], [200, 300], [300, 300]], float)
    c = [2, 2, 2, 1]
    n = [3, 2, 2, 1]
    ll = nhpp_log_likelihood(x, c=c, n=n, alpha=30.0, beta=0.9)
    assert np.isclose(ll.eval(), _own(x, c=c, n=n, params=(30.0, 0.9)))
    pt = pytest.importorskip("pytensor.tensor")
    b = pt.dscalar("b")
    grad = pt.grad(nhpp_log_likelihood(X, c=C, alpha=50.0, beta=b), b)
    assert np.isfinite(grad.eval({b: 0.8}))


def test_one_value_a_system_or_a_group():
    x, i, c, _ = _fleet()
    alphas = np.array([20.0, 25.0, 18.0, 22.0, 20.0, 30.0, 15.0, 20.0])
    betas = np.linspace(0.6, 1.2, 8)
    ll = nhpp_log_likelihood(
        x, i=i, c=c, alpha=alphas, beta=betas, group="system"
    )
    each = sum(
        _own(x[i == s], c=c[i == s], params=(alphas[s], betas[s]))
        for s in range(8)
    )
    assert np.isclose(ll.eval(), each, rtol=1e-12)
    site = {s: ("north" if s < 5 else "south") for s in range(8)}
    ll = nhpp_log_likelihood(
        x, i=i, c=c, alpha=20.0, beta=np.array([0.7, 1.1]), group=site
    )
    north = np.isin(i, range(5))
    both = _own(x[north], i[north], c[north], params=(20.0, 0.7)) + _own(
        x[~north], i[~north], c[~north], params=(20.0, 1.1)
    )
    assert np.isclose(ll.eval(), both, rtol=1e-12)


def test_refusals():
    x, i, c, _ = _fleet()
    with pytest.raises(ValueError, match="one value for every system"):
        nhpp_log_likelihood(x, i=i, c=c, alpha=20.0, beta=np.ones(8))
    with pytest.raises(ValueError, match="one value a group"):
        nhpp_log_likelihood(
            x, i=i, c=c, alpha=20.0, beta=np.ones(3), group="system"
        )
    with pytest.raises(ValueError, match="more than one group"):
        nhpp_log_likelihood(
            x,
            i=i,
            c=c,
            alpha=20.0,
            beta=np.ones(2),
            group=np.arange(len(x)) % 2,
        )
    with pytest.raises(TypeError, match="pm.Model"):
        nhpp_likelihood(X, c=C, alpha=50.0, beta=0.8)
    with pm.Model():
        with pytest.raises(ValueError, match="nothing to pool"):
            pooled_likelihood(
                CrowAMSAA,
                x=X,
                c=C,
                pool={"beta": (0.0, 0.5)},
                shared={"alpha": 50.0},
            )


def test_a_term_in_a_model():
    with pm.Model() as model:
        alpha = pm.LogNormal("alpha", mu=np.log(50.0), sigma=1.0)
        beta = pm.LogNormal("beta", mu=np.log(0.8), sigma=0.3)
        nhpp_likelihood(X, c=C, alpha=alpha, beta=beta)
    point = {"alpha_log__": np.log(50.0), "beta_log__": np.log(0.8)}
    (term,) = model.replace_rvs_by_values(model.potentials)
    potential = model.compile_fn(term, inputs=model.value_vars)
    assert np.isclose(potential(point), _own(X, c=C))


@pytest.fixture(scope="module")
def fleet_fit():
    x, i, c, betas = _fleet()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pm.Model():
            alpha = pm.LogNormal("alpha", mu=np.log(20.0), sigma=1.0)
            pooling = pooled_likelihood(
                CrowAMSAA,
                x=x,
                i=i,
                c=c,
                pool={"beta": (np.log(0.8), 0.5)},
                shared={"alpha": alpha},
                tau_scale=0.5,
            )
            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                random_seed=700011,
                progressbar=False,
                compute_convergence_checks=False,
            )
    return trace, pooling, betas


def test_pooling_recovers_the_fleet(fleet_fit):
    trace, pooling, _ = fleet_fit
    assert pooling.kind == "nhpp" and pooling.groups == list(range(8))
    tau = np.asarray(trace.posterior["groups_beta_tau"]).ravel()
    lo, hi = np.quantile(tau, [0.025, 0.975])
    assert lo < 0.3 < hi
    table = shrinkage(trace, pooling)
    for s in (6, 7):
        row = table[table["group"] == s].iloc[0]
        ends = sorted([row["alone"], row["population"]])
        assert ends[0] < row["pooled"] < ends[1]
    models = group_models(trace, pooling, random_state=1)
    assert isinstance(models[0], BayesianRecurrence)
    widths = [
        np.diff(models[s].param_cri("beta", alpha_ci=0.1))[0] for s in range(8)
    ]
    new = np.diff(models["new group"].param_cri("beta", alpha_ci=0.1))[0]
    assert new > max(widths)
    lo, hi = models[6].mtbf_cri()
    assert 0 < lo < models[6].mtbf(models[6].end) < hi


def test_the_centred_form():
    rng = np.random.default_rng(1)
    x = sp.Weibull.random(20, 1000.0, 2.0, random_state=2)
    sites = np.repeat(["a", "b"], 10)
    with pm.Model() as model:
        pooling = pooled_likelihood(
            sp.Weibull,
            x=x,
            group=sites,
            pool={"alpha": (np.log(1000.0), 1.0)},
            shared={"beta": 2.0},
            centred=True,
        )
    assert pooling.centred and "groups_alpha" in model.named_vars
    assert "groups_alpha_z" not in model.named_vars
    az = pytest.importorskip("arviz")
    trace = az.from_dict(
        {
            "posterior": {
                "groups_alpha_mu": rng.normal(np.log(1000), 0.05, (2, 200)),
                "groups_alpha_tau": np.abs(rng.normal(0.2, 0.02, (2, 200))),
                "groups_alpha": rng.normal(np.log(1000), 0.1, (2, 200, 2)),
            }
        }
    )
    models = group_models(trace, pooling, random_state=1)
    assert sorted(models) == ["a", "b", "new group"]
    table = shrinkage(trace, pooling)
    assert list(table["group"]) == ["a", "b"]
    assert np.all(np.isfinite(table["alone"]))
