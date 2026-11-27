"""Bayesian fits of the renewal and imperfect-repair processes
(``surpyval.recurrent.renewal.bayes``).

Checked against:

- the fitters' own likelihoods (the all-rows likelihood the grid uses,
  at random parameter rows, to 1e-9);
- an independent brute-force integration: a midpoint rule on a
  400 x 160 x 160 box in ``(rho, log alpha, log beta)`` with proper
  priors, for the evidence, the repair parameter's distribution function
  and the mass near each end of its range;
- the maximum-likelihood fit: with priors flat on the natural scale of
  the repair parameter and on the life's log scales, the posterior's
  peak is the maximum-likelihood estimate;
- an identity of the models: Kijima I is ARA with memory 1 and ``rho =
  1 - q``, so the two posteriors on the same data and priors agree;
- the stability of the mass near an end of the range when the grid is
  refined;
- ``RenewalModel.next_failure_sf`` at single posterior points (the
  vectorised states);
- simulation-based calibration is in
  ``tests/calibration/test_sbc_renewal_bayes.py``;
- a frozen PyMC reference (``references/renewal_ara_pymc.json``, NUTS
  on the likelihood written from the model's definition), by the kit's
  rule.
"""

import json
import pickle
import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

import surpyval as sp
from surpyval.recurrent import ARA, ARI, GeneralizedRenewal
from surpyval.recurrent.renewal.bayes import BayesianRenewal, RenewalTarget

PERFECT = "perfect repair (as good as new)"
MINIMAL = "minimal repair (as bad as old)"


def _data(fitter, params, r, T, items, seed, **kw):
    d = fitter.from_params(params, r, **kw).time_terminated_simulation_data(
        T, items=items, random_state=seed
    )
    return d.x, d.i, d.c


#: Three systems to 30, repaired as good as new (ARA memory 1, rho = 1):
#: ten failures, the maximum-likelihood rho on its end, 1.
AT_END = _data(ARA, [10.0, 2.0], 1.0, 30.0, 3, 4, m=1)
#: Five systems to 60 with ARA memory 2, rho = 0.7.
INSIDE = _data(ARA, [10.0, 2.5], 0.7, 60.0, 5, 1, m=2)
PROPER = {
    "rho": sp.prior.uniform(0, 1),
    "alpha": sp.LogNormal.from_params([np.log(10.0), 0.5]),
    "beta": sp.LogNormal.from_params([np.log(2.0), 0.3]),
}


@pytest.fixture(scope="module")
def at_end():
    return ARA.fit_bayes(*AT_END[:2], c=AT_END[2], m=1, priors=PROPER)


# -- the likelihood --------------------


@pytest.mark.parametrize(
    "fitter, params, r, kw",
    [
        (GeneralizedRenewal, [10.0, 2.5], 0.3, {}),
        (GeneralizedRenewal, [10.0, 2.5], 0.5, {"kijima": "ii"}),
        (ARA, [10.0, 2.5], 0.7, {"m": 2}),
        (ARA, [10.0, 2.5], 0.6, {"m": np.inf}),
        (ARI, [5.0, 1.8], 0.4, {"m": 2}),
    ],
)
def test_rows_likelihood_is_the_fitters(fitter, params, r, kw):
    x, i, c = _data(fitter, params, r, 40.0, 3, 2, **kw)
    target = RenewalTarget(fitter.fit(x, i, c=c, **kw))
    assert target._vector
    rng = np.random.default_rng(3)
    theta = target.estimates() * np.exp(0.2 * rng.standard_normal((50, 3)))
    high = target.bounds[0][1]
    theta[:, 0] = np.clip(theta[:, 0], 1e-3, 0.999 if high else None)
    fast = target.log_lik(theta)
    slow = np.array([-target.neg_ll(t) for t in theta])
    ok = np.isfinite(slow)
    assert ok.sum() > 40
    np.testing.assert_allclose(fast[ok], slow[ok], rtol=1e-9, atol=1e-9)


# -- against a brute-force integration --------------------


def test_brute_force_evidence_and_repair_distribution(at_end):
    target = at_end.target
    nr, na = 400, 160
    rho = (np.arange(nr) + 0.5) / nr
    la = np.log(10.0) + 0.5 * np.linspace(-6, 6, na)
    lb = np.log(2.0) + 0.3 * np.linspace(-6, 6, na)
    R, A, B = np.meshgrid(rho, la, lb, indexing="ij")
    theta = np.column_stack([R.ravel(), np.exp(A.ravel()), np.exp(B.ravel())])
    ll = np.concatenate(
        [
            target.log_lik(theta[k : k + 200_000])
            for k in range(0, len(theta), 200_000)
        ]
    )
    lp = (
        ll
        + stats.norm.logpdf(A.ravel(), np.log(10.0), 0.5)
        + stats.norm.logpdf(B.ravel(), np.log(2.0), 0.3)
    )
    log_z = (
        logsumexp(lp)
        - np.log(nr)
        + np.log(la[1] - la[0])
        + np.log(lb[1] - lb[0])
    )
    assert at_end.log_evidence == pytest.approx(log_z, abs=1e-3)
    w = np.exp(lp - logsumexp(lp)).reshape(nr, -1).sum(axis=1)
    # the distribution function of rho at the cells' edges
    edges = np.array([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    brute = np.array([w[rho < e].sum() for e in edges])
    grid = np.array([at_end._cdf(e) for e in edges])
    np.testing.assert_allclose(grid, brute, atol=2e-3)
    mass = at_end.repair_mass(0.05)
    assert mass[PERFECT] == pytest.approx(w[rho >= 0.95].sum(), abs=1e-3)
    assert mass[MINIMAL] == pytest.approx(w[rho <= 0.05].sum(), abs=5e-4)
    # mass piles against perfect repair: much more than the prior's 5%
    assert mass[PERFECT] > 0.2


def test_flat_prior_peak_is_the_maximum_likelihood_fit():
    from surpyval.bayes.engines.grid import find_mode
    from surpyval.bayes.result import bind, log_posterior

    x, i, c = INSIDE
    model = ARA.fit_bayes(x, i, c=c, m=2)
    bound, _ = bind(model.target, None)
    t0 = bound.transforms[0]
    log_post = log_posterior(model.target, bound)

    # the density on rho's own scale (uniform: flat) and the life's log
    # scales (flat): the likelihood
    def natural(phi):
        phi = np.atleast_2d(phi)
        return log_post(phi) - t0.log_jacobian(phi[:, 0])

    start = bound.to_phi(model.target.start()[None, :])[0]
    peak, _, _ = find_mode(natural, [start])
    np.testing.assert_allclose(
        bound.full_theta(peak[None, :])[0], model.mle.params, rtol=2e-4
    )


def test_kijima_one_is_ara_memory_one():
    x, i, c = AT_END
    life = {k: PROPER[k] for k in ("alpha", "beta")}
    ara = ARA.fit_bayes(x, i, c=c, m=1, priors=PROPER)
    grp = GeneralizedRenewal.fit_bayes(
        x, i, c=c, priors={"q": sp.prior.uniform(0, 1), **life}
    )
    levels = [0.025, 0.1, 0.5, 0.9, 0.975]
    rho = ara.posterior.quantile("rho", levels)
    q = grp.posterior.quantile("q", levels[::-1])
    np.testing.assert_allclose(1 - q, rho, atol=5e-3)
    for name in ("alpha", "beta"):
        np.testing.assert_allclose(
            grp.param_cri(name), ara.param_cri(name), rtol=5e-3
        )
    assert grp.repair_mass(0.05)[PERFECT] == pytest.approx(
        ara.repair_mass(0.05)[PERFECT], abs=3e-3
    )
    assert grp.log_evidence == pytest.approx(ara.log_evidence, abs=1e-3)


@pytest.mark.parametrize("within", [0.01, 0.05])
def test_mass_near_the_end_is_stable_when_the_grid_is_refined(within):
    x, i, c = AT_END
    coarse = ARA.fit_bayes(x, i, c=c, m=1, priors=PROPER)
    fine = ARA.fit_bayes(x, i, c=c, m=1, priors=PROPER, points=121)
    assert fine.diagnostics["points_per_dim"][0] >= 121
    a, b = coarse.repair_mass(within), fine.repair_mass(within)
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=2e-3)


def test_the_grid_reaches_into_the_tail_at_the_end(at_end):
    # the walk along logit(rho) goes on to where the density is 1e-8 of
    # the peak's, far past the curvature's few standard deviations
    assert at_end.diagnostics["reach_sd"][1] > 10
    top = at_end.posterior.points[:, 0].max()
    assert 1 - top < 1e-4


# -- results --------------------


@pytest.mark.parametrize(
    "fitter, params, r, kw",
    [
        (GeneralizedRenewal, [10.0, 2.5], 0.3, {}),
        (GeneralizedRenewal, [10.0, 2.5], 0.5, {"kijima": "ii"}),
        (ARA, [10.0, 2.5], 0.7, {"m": 2}),
        (ARI, [5.0, 1.8], 0.4, {"m": 2}),
    ],
)
def test_next_failure_is_the_models_at_each_point(fitter, params, r, kw):
    x, i, c = _data(fitter, params, r, 40.0, 3, 2, **kw)
    model = fitter.fit_bayes(x, i, c=c, **kw)
    t = np.array([0.0, 1.0, 4.0])
    rows = model._sf_rows(t, None)
    new = model._sf_rows(t, [0.0, 3.0])
    states = model.mle._states()
    for k in np.argsort(model.posterior.weights)[-4:]:
        at = model.target.with_params(model.posterior.points[k])
        own = at._state_of(None, states.failures, states.now)
        np.testing.assert_allclose(
            rows[k], at.next_failure_sf(t, states=own), atol=1e-12
        )
        np.testing.assert_allclose(
            new[k], at.next_failure_sf(t, age=[0.0, 3.0]), atol=1e-12
        )
    sf = model.next_failure_sf(t)
    assert sf.shape == (3, 3)
    np.testing.assert_allclose(
        sf, np.einsum("p,pux->ux", model.posterior.weights, rows), rtol=1e-12
    )
    median = model.next_failure_qf(0.5)
    for u in range(3):
        assert model.next_failure_sf(median[u])[u] == pytest.approx(
            0.5, abs=1e-8
        )
    lo, hi = np.moveaxis(model.next_failure_cri(4.0), -1, 0)
    assert np.all((lo < sf[:, 2]) & (sf[:, 2] < hi))


def test_count_is_reproducible_and_sums_the_units(at_end):
    a = at_end.count(10.0, draws=40, items=5)
    b = at_end.count(10.0, draws=40, items=5)
    assert a.mean == b.mean and a.replicates == 200 and a.units == 3
    assert a.mean == pytest.approx(a.unit_means.sum())
    lo, hi = a.interval(0.1)
    assert lo <= a.mean <= hi
    assert float(a.cdf(hi)) >= 0.95
    new = at_end.count(10.0, age=[0.0], draws=40, items=5)
    assert new.units == 1


def test_defaults_summary_and_refusals():
    x, i, c = AT_END
    model = GeneralizedRenewal.fit_bayes(x, i, c=c)
    assert model.default_priors == {"q": "Uniform(0, 2)"}
    assert isinstance(model, BayesianRenewal) and model.method == "grid"
    text = model.summary()
    assert "Repair (q), posterior probability within 0.05:" in text
    assert "that q < 1" in text
    with pytest.raises(ValueError, match="param_cri"):
        model.param_cb("q")
    with pytest.raises(ValueError, match="flat prior"):
        GeneralizedRenewal.fit_bayes(x, i, c=c, priors={"q": sp.prior.flat()})
    with pytest.raises(ValueError, match="from_data"):
        GeneralizedRenewal.fit_bayes(
            x, i, c=c, priors=sp.prior.from_data([5.0, 9.0, 12.0])
        )
    with pytest.raises(ValueError, match="duration"):
        model.count(-1.0)


def test_an_end_is_not_a_prior_conflict(recwarn):
    # rho's estimate is 1, outside the uniform prior's central interval,
    # but the prior gives that end as much weight as its middle
    x, i, c = AT_END
    model = ARA.fit_bayes(x, i, c=c, m=1, priors=PROPER)
    assert model.diagnostics["prior_conflict"] == []
    assert not [w for w in recwarn if "disagree" in str(w.message)]


def test_life_prior_from_a_non_repairable_fit():
    x, i, c = AT_END
    life = sp.Weibull.fit([8.0, 12, 9, 15, 11, 7, 10, 13, 9.5, 14])
    flat = GeneralizedRenewal.fit_bayes(x, i, c=c)
    informed = GeneralizedRenewal.fit_bayes(
        x, i, c=c, priors=sp.prior.from_model(life, weight=0.5)
    )
    assert "from an earlier Weibull fit" in informed.summary()
    width = np.diff(informed.param_cri("beta"))[0]
    assert width < np.diff(flat.param_cri("beta"))[0]


def test_laplace_on_request():
    x, i, c = INSIDE
    model = ARA.fit_bayes(x, i, c=c, m=2, method="laplace")
    assert model.method == "laplace"


def test_saved_and_pickled(at_end):
    back = sp.from_dict(json.loads(json.dumps(at_end.to_dict())))
    assert isinstance(back, BayesianRenewal)
    np.testing.assert_allclose(back.params, at_end.params, rtol=1e-12)
    assert back.repair_mass() == at_end.repair_mass()
    again = pickle.loads(pickle.dumps(at_end))
    np.testing.assert_allclose(
        again.next_failure_sf(2.0), at_end.next_failure_sf(2.0), rtol=1e-12
    )


def test_criteria_take_a_system_as_the_unit(at_end):
    with warnings.catch_warnings():
        # (three units: WAIC may say it is unreliable)
        warnings.simplefilter("ignore")
        waic = at_end.waic()
    assert "3 units" in repr(waic)
    assert "3 units" in repr(at_end.loo())


def test_against_pymc():
    """The frozen PyMC reference (``references/renewal_ara_pymc.json``,
    made by ``scripts/reference/make_renewal_ara.py`` with the likelihood
    written from the model's definition), by the kit's rule."""
    from surpyval.tests.bayes import kit

    reference = kit.load_reference("renewal_ara_pymc")
    x, i, c = AT_END
    data = reference["data"]
    np.testing.assert_allclose(data["x"], x, rtol=1e-12)
    np.testing.assert_array_equal(data["c"], c)
    model = ARA.fit_bayes(x, i, c=c, m=1, priors=PROPER)
    kit.check_reference(model, reference)
    # the mass near perfect repair, within four of the sampler's
    # standard errors of a proportion (its draws are correlated: the
    # effective size is taken as a quarter of the 40,000)
    p = reference["perfect_within_0.05"]
    se = np.sqrt(p * (1 - p) / 10_000)
    assert model.repair_mass(0.05)[PERFECT] == pytest.approx(p, abs=4 * se)
