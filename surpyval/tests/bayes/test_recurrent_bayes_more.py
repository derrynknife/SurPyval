"""Bayesian fits of the other NHPP forms, the proportional intensity
models and the cause-specific intensities
(``surpyval.recurrent.bayes``).

Checked against:

- the maximum-likelihood fit (with no priors the posterior's peak);
- closed forms: a proportional intensity with a constant baseline and a
  two-level covariate is two separate rates, each Gamma(events, time
  watched) with priors flat on their logs; cause-specific constant
  rates with Gamma priors are Gamma posteriors, and their total (the
  same rate parameter) a Gamma too;
- the grid against the Laplace approximation on small cases;
- coverage, in ``tests/calibration/test_sbc_recurrent_bayes.py``.
"""

import json
import pickle
import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
import surpyval.recurrent as rc
from surpyval.recurrent.bayes import (
    BayesianCauseSpecific,
    BayesianProportionalIntensity,
)

X = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
C = [0] * 11 + [1]
#: A prior on a coefficient wide enough to be flat over the posterior.
WIDE = {"coef_0": sp.Normal.from_params([0.0, 1000.0])}


def fleet(seed, systems=12, beta=1.0, coef=(0.7,), alpha=60.0, end=300.0):
    """Power-law systems with covariates: two-level ones for up to two
    covariates, normal ones beyond."""
    rng = np.random.default_rng(seed)
    x, Z, i, c = [], [], [], []
    for k in range(systems):
        if len(coef) < 3:
            z = [float((k >> j) & 1) for j in range(len(coef))]
        else:
            z = list(rng.normal(size=len(coef)))
        total = (end / alpha) ** beta * np.exp(np.dot(coef, z))
        m = rng.poisson(total)
        u = np.sort((end / alpha) ** beta * rng.uniform(size=m))
        x += [*(alpha * u ** (1 / beta)), end]
        Z += [z] * (m + 1)
        i += [k] * (m + 1)
        c += [0] * m + [1]
    return x, Z, i, c


def _worst(grid, other):
    """The largest difference between two fits' 95% intervals, as a
    share of the first's width."""
    worst = 0.0
    for name in grid.parameter_names:
        g = grid.param_cri(name)
        d = np.max(np.abs(other.param_cri(name) - g)) / (g[1] - g[0])
        worst = max(worst, float(d))
    return worst


# -- the other NHPP forms ---------------------------------------


@pytest.mark.parametrize("name", ["Duane", "CoxLewis"])
def test_other_intensities_peak_at_the_mle_with_no_priors(name):
    fitter = getattr(rc, name)
    model = fitter.fit_bayes(X, c=C)
    assert model.method == "grid"
    mle = fitter.fit(X, c=C)
    np.testing.assert_allclose(
        model.diagnostics["mode"], mle.params, rtol=2e-4
    )
    lo, hi = model.mtbf_cri()
    assert lo < float(model.mtbf(400.0)) < hi
    assert model.count(duration=100.0).mean > 0
    assert model.target.family == f"{fitter.name} process"


def test_the_power_law_in_two_forms_gives_one_mtbf():
    # Duane's intensity is Crow-AMSAA's in other parameters: with the
    # same belief about the MTBF and the shape, the same posterior of
    # the MTBF
    shape = sp.prior.between(0.5, 1.2)
    mtbf = {400.0: sp.prior.between(30.0, 70.0)}
    crow = rc.CrowAMSAA.fit_bayes(
        X,
        c=C,
        priors=sp.prior.on_quantities(rc.CrowAMSAA, mtbf_at=mtbf, beta=shape),
    )
    duane = rc.Duane.fit_bayes(
        X,
        c=C,
        priors=sp.prior.on_quantities(rc.Duane, mtbf_at=mtbf, alpha=shape),
    )
    np.testing.assert_allclose(duane.mtbf_cri(), crow.mtbf_cri(), rtol=2e-3)
    np.testing.assert_allclose(
        duane.param_cri("alpha"), crow.param_cri("beta"), rtol=2e-3
    )


def test_a_finite_fault_model_needs_priors_on_growing_data():
    # these data show no levelling off, so a model with a finite number
    # of faults has no maximum in it without a prior
    with pytest.raises(ValueError, match="no finite mass|no prior"):
        rc.GoelOkumoto.fit_bayes(X, c=C)
    model = rc.GoelOkumoto.fit_bayes(
        X,
        c=C,
        priors={
            "a": sp.prior.between(12, 40),
            "b": sp.prior.between(0.001, 0.01),
        },
    )
    assert model.method == "grid"
    lo, hi = model.param_cri("a")
    assert 8 < lo < hi < 60


# -- proportional intensity ---------------------------------------


def test_pi_hpp_with_two_levels_is_two_gamma_rates():
    x, Z, i, c = fleet(1)
    model = rc.ProportionalIntensityHPP.fit_bayes(x, Z, i, c, priors=WIDE)
    assert isinstance(model, BayesianProportionalIntensity)
    assert model.method == "grid"
    xa, za, ca = np.array(x), np.array(Z)[:, 0], np.array(c)
    for level in (0.0, 1.0):
        events = np.sum((za == level) & (ca == 0))
        watched = 300.0 * 6
        exact = stats.gamma(events, scale=1 / watched)
        np.testing.assert_allclose(
            model.cri(100.0, [level], on="iif"),
            exact.ppf([0.025, 0.975]),
            rtol=3e-3,
        )
        np.testing.assert_allclose(
            model.cri(100.0, [level], on="mtbf"),
            1 / exact.ppf([0.975, 0.025]),
            rtol=3e-3,
        )
        assert model.mtbf_probability(40.0, [level]) == pytest.approx(
            exact.cdf(1 / 40.0), abs=5e-3
        )
        nxt = model.count([level], duration=100.0, systems=2)
        nb = stats.nbinom(events, watched / (watched + 200.0))
        np.testing.assert_allclose(
            nxt.pmf(np.arange(12)), nb.pmf(np.arange(12)), atol=2e-4
        )
    assert xa.size == len(x)


def test_pi_default_priors_and_the_mle_at_the_peak():
    x, Z, i, c = fleet(1)
    model = rc.ProportionalIntensityNHPP.fit_bayes(
        x, Z, i, c, baseline=rc.CrowAMSAA, priors=WIDE
    )
    assert model.method == "grid"
    assert model.parameter_names == ["alpha", "beta", "coef_0"]
    np.testing.assert_allclose(
        model.diagnostics["mode"], model.mle._mle, rtol=2e-4
    )
    plain = rc.ProportionalIntensityHPP.fit_bayes(x, Z, i, c)
    assert list(plain.default_priors) == ["coef_0"]
    assert "default: Normal(0, 5), which is 2.5 over" in plain.summary()
    assert plain.summary().startswith(
        "Proportional intensity (Homogeneous Poisson Process baseline): "
        "Bayesian fit, 12 systems"
    )
    stated = rc.ProportionalIntensityHPP.fit_bayes(
        x, Z, i, c, priors={"coef_0": sp.Normal.from_params([0.5, 0.2])}
    )
    assert stated.default_priors == {}
    assert np.diff(stated.param_cri("coef_0")) < np.diff(
        plain.param_cri("coef_0")
    )


@pytest.mark.parametrize(
    "fitter, kw, allowed",
    [
        ("ProportionalIntensityHPP", {}, 0.08),
        ("ProportionalIntensityNHPP", {"baseline": "CrowAMSAA"}, 0.05),
    ],
)
def test_pi_grid_against_laplace(fitter, kw, allowed):
    x, Z, i, c = fleet(1)
    kw = {k: getattr(rc, v) for k, v in kw.items()}
    f = getattr(rc, fitter)
    grid = f.fit_bayes(x, Z, i, c, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        laplace = f.fit_bayes(x, Z, i, c, method="laplace", **kw)
        reweighted = f.fit_bayes(
            x, Z, i, c, method="laplace", importance=True, **kw
        )
    assert grid.method == "grid" and laplace.method == "laplace"
    assert laplace.diagnostics["approximation"] == "good"
    print(
        fitter,
        f"Laplace {_worst(grid, laplace):.3f}, "
        f"reweighted {_worst(grid, reweighted):.3f} of the 95% width",
    )
    assert _worst(grid, laplace) < allowed
    assert _worst(grid, reweighted) < 0.03


def test_pi_beyond_three_parameters_is_laplace_with_its_checks():
    x, Z, i, c = fleet(2, systems=30, coef=(0.5, -0.4, 0.3))
    model = rc.ProportionalIntensityNHPP.fit_bayes(
        x, Z, i, c, baseline=rc.CrowAMSAA
    )
    assert model.method == "laplace"
    d = model.diagnostics
    assert d["approximation"] == "good" and d["pareto_k"] < 0.5
    assert sorted(model.default_priors) == ["coef_0", "coef_1", "coef_2"]
    for k, truth in enumerate((0.5, -0.4, 0.3)):
        lo, hi = model.param_cri(f"coef_{k}", alpha_ci=0.01)
        assert lo < truth < hi
    assert model.diagnose().ok
    with pytest.raises(ValueError, match="the grid takes 1 to 3"):
        rc.ProportionalIntensityNHPP.fit_bayes(
            x, Z, i, c, baseline=rc.CrowAMSAA, method="grid"
        )


def test_pi_units_checks_saving_and_earlier_fits():
    x, Z, i, c = fleet(1)
    model = rc.ProportionalIntensityHPP.fit_bayes(x, Z, i, c)
    target = model.target
    theta = np.array([model.params, model.params * 1.1])
    np.testing.assert_allclose(
        target.unit_log_lik(theta).sum(axis=1),
        [-target.neg_ll(t) for t in theta],
        rtol=1e-10,
    )
    assert model.loo().units == 12
    without = rc.ProportionalIntensityHPP.fit_bayes(
        x, Z, i, c, priors={"coef_0": sp.prior.fixed(0.0)}
    )
    table = sp.bayes.compare({"with": model, "without": without})
    assert table.best == "with"
    assert model.diagnose().ok
    one = model.at_params()
    assert float(one.cif(100.0, [[1.0]])) == pytest.approx(
        100 * model.params[0] * np.exp(model.params[1])
    )
    again = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert isinstance(again, BayesianProportionalIntensity)
    np.testing.assert_array_equal(
        again.cri(100.0, [1.0]), model.cri(100.0, [1.0])
    )
    pickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        pickled.mtbf_cri([1.0]), model.mtbf_cri([1.0])
    )
    # an earlier fleet as the prior: at weight 1, the two fleets together
    xa, Za, ia, ca = (np.asarray(v) for v in (x, Z, i, c))
    old, new = ia < 6, ia >= 6
    slope = {"coef_0": sp.Normal.from_params([0.0, 2.0])}
    earlier = rc.ProportionalIntensityHPP.fit(
        xa[old], Za[old], ia[old], ca[old]
    )
    pooled = rc.ProportionalIntensityHPP.fit_bayes(x, Z, i, c, priors=slope)
    power = rc.ProportionalIntensityHPP.fit_bayes(
        xa[new],
        Za[new],
        ia[new],
        ca[new],
        priors=[sp.prior.from_fit(earlier), slope],
    )
    for name in pooled.parameter_names:
        np.testing.assert_allclose(
            power.param_cri(name), pooled.param_cri(name), rtol=2e-3
        )
    normal = rc.ProportionalIntensityHPP.fit_bayes(
        xa[new], Za[new], ia[new], ca[new], priors=sp.prior.from_model(earlier)
    )
    assert normal.default_priors == {}


def test_pi_refusals_say_what_to_do():
    x, Z, i, c = fleet(1)
    model = rc.ProportionalIntensityHPP.fit_bayes(x, Z, i, c)
    with pytest.raises(ValueError, match="one covariate set"):
        model.cif(100.0, [[0.0], [1.0]])
    with pytest.raises(ValueError, match="on must be one of"):
        model.cri(100.0, [1.0], on="sf")
    with pytest.raises(ValueError, match="positive MTBF"):
        model.mtbf_probability(-1.0, [1.0])
    with pytest.raises(ValueError, match="Leave the column out"):
        rc.ProportionalIntensityHPP.fit_bayes(x, np.ones((len(x), 1)), i, c)


# -- cause-specific intensities ---------------------------------------


def _causes():
    rng = np.random.default_rng(7)
    x, i, c, e = [], [], [], []
    for k in range(4):
        end = 250.0
        for cause, rate in (("pump", 0.012), ("seal", 0.03)):
            m = rng.poisson(rate * end)
            x += list(end * rng.uniform(size=m))
            i += [k] * m
            c += [0] * m
            e += [cause] * m
        x.append(end)
        i.append(k)
        c.append(1)
        e.append(None)
    order = np.lexsort((x, i))
    pick = lambda v: [v[j] for j in order]  # noqa: E731
    return pick(x), pick(i), pick(c), pick(e)


def test_cause_specific_rates_are_separate_gamma_posteriors():
    x, i, c, e = _causes()
    prior = {"lambda": sp.Gamma.from_params([2.0, 100.0])}
    model = rc.CauseSpecificNHPP.fit_bayes(
        x, i, c, e=e, dist=rc.HPP, priors=prior
    )
    assert isinstance(model, BayesianCauseSpecific)
    assert model.event_types == ["pump", "seal"]
    counts = {k: sum(1 for v in e if v == k) for k in ("pump", "seal")}
    for cause, n in counts.items():
        assert model.method[cause] == "conjugate"
        exact = stats.gamma(2.0 + n, scale=1 / (100.0 + 1000.0))
        np.testing.assert_allclose(
            model.param_cri(cause, "lambda"),
            exact.ppf([0.025, 0.975]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.cri(100.0, cause, on="iif"),
            exact.ppf([0.025, 0.975]),
            rtol=1e-12,
        )
        assert model.mtbf_probability(40.0, cause) == pytest.approx(
            exact.cdf(1 / 40.0), rel=1e-12
        )
    # all causes: independent gammas with one rate parameter sum to a
    # gamma (joint draws, so to Monte Carlo accuracy)
    total = stats.gamma(4.0 + sum(counts.values()), scale=1 / 1100.0)
    lo, hi = total.ppf([0.025, 0.975])
    np.testing.assert_allclose(
        model.cri(100.0, on="iif"), [lo, hi], atol=0.03 * (hi - lo)
    )
    assert model.mtbf_probability(20.0) == pytest.approx(
        total.cdf(1 / 20.0), abs=0.02
    )
    assert float(model.iif(100.0)) == pytest.approx(total.mean(), rel=1e-9)
    nxt = model.count(duration=100.0)
    assert nxt.mean == pytest.approx(100 * total.mean(), rel=0.02)
    assert model.count("seal", duration=100.0).mean < nxt.mean


def test_cause_specific_priors_by_cause_and_the_power_law():
    x, i, c, e = _causes()
    belief = sp.prior.around(1.0, between=(0.7, 1.4))
    model = rc.CauseSpecificNHPP.fit_bayes(
        x, i, c, e=e, priors={"seal": {"beta": belief}}
    )
    assert model.method == {"pump": "grid", "seal": "grid"}
    seal, pump = model.models["seal"], model.models["pump"]
    assert "beta: about 1" in seal.summary()
    assert "beta: no prior given" in pump.summary()
    # each cause's fit is the process's own fit of that cause's events
    rows = model.mle.models["seal"].data
    alone = rc.CrowAMSAA.fit_bayes(rows, priors={"beta": belief})
    np.testing.assert_array_equal(
        alone.param_cri("beta"), model.param_cri("seal", "beta")
    )
    every = rc.CauseSpecificNHPP.fit_bayes(
        x, i, c, e=e, priors={"beta": belief}
    )
    assert "beta: about 1" in every.models["pump"].summary()
    assert "Cause 'pump'" in model.summary()
    assert set(model.diagnose()) == {"pump", "seal"}
    again = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(
        again.cri(200.0, "seal", on="mtbf"),
        model.cri(200.0, "seal", on="mtbf"),
    )
    with pytest.raises(ValueError, match="No cause 'valve'"):
        model.cif(100.0, "valve")
    with pytest.raises(ValueError, match="mixes causes and parameter"):
        rc.CauseSpecificNHPP.fit_bayes(
            x, i, c, e=e, priors={"seal": {"beta": belief}, "beta": belief}
        )


def _three_systems(with_Z=False, with_e=False):
    from surpyval.tests.conformance.registry_fixtures import recurrent_data

    return recurrent_data(with_Z, with_e)


def test_a_point_of_no_weight_adds_nothing_to_a_mean():
    """The grid's outer cells reach a scale whose cumulative intensity is
    infinite; with no weight there, the means are numbers (they were NaN:
    0 times inf)."""
    d = _three_systems(with_e=True)
    model = rc.CauseSpecificNHPP.fit_bayes(
        d["x"], d["i"], d["c"], d["n"], e=d["e"]
    )
    total = model.cif(55.0)
    assert np.isfinite(total)
    assert total == pytest.approx(4.0927, abs=1e-3)
    assert np.isfinite(model.cif(55.0, event="a"))
    assert np.isfinite(model.iif(30.0, event="b"))


def test_a_covariate_in_very_small_units_gives_the_same_fit():
    """A direction curving less than 1e-8 at the peak is still given its
    own width, so the coefficient's very wide posterior is reached."""
    d = _three_systems(with_Z=True)
    base = rc.ProportionalIntensityHPP.fit_bayes(
        d["x"], d["Z"], d["i"], d["c"], d["n"]
    )
    small = rc.ProportionalIntensityHPP.fit_bayes(
        d["x"], d["Z"] * 1e-7, d["i"], d["c"], d["n"]
    )
    assert small.cif(30.0, [0.5e-7]) == pytest.approx(
        base.cif(30.0, [0.5]), rel=1e-5
    )
    assert small.params[1] * 1e-7 == pytest.approx(base.params[1], rel=1e-4)


def test_a_missing_time_or_covariate_gives_nan():
    d = _three_systems()
    model = rc.CrowAMSAA.fit_bayes(d["x"], d["i"], d["c"])
    out = model.mtbf(np.array([np.nan, 30.0]))
    assert np.isnan(out[0]) and np.isfinite(out[1])
    lower, upper = model.cri(np.array([np.nan, 30.0]), on="cif")
    assert np.isnan(lower[0]) and np.isfinite(upper[1])
    hpp = rc.HPP.fit_bayes(
        d["x"],
        d["i"],
        d["c"],
        priors={"lambda": sp.Gamma.from_params([2, 20])},
    )
    assert np.isnan(hpp.iif(np.array([np.nan]))[0])
    dz = _three_systems(with_Z=True)
    pi = rc.ProportionalIntensityHPP.fit_bayes(
        dz["x"], dz["Z"], dz["i"], dz["c"], dz["n"]
    )
    assert np.all(np.isnan(pi.cri(np.array([10.0, 30.0]), [np.nan])))
