"""The shared Bayesian layer (``surpyval.bayes.target`` and
``surpyval.bayes.result``) on a family small enough to check by hand:
Poisson counts with a rate, where the gamma prior's posterior is known
in closed form, and a log-linear rate, where the engines are checked
against each other."""

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.bayes import BayesianResult, Target, fit_target
from surpyval.bayes.engines.conjugate import ConjugateResult, _prior_family
from surpyval.bayes.target import TARGETS, register_target
from surpyval.tests.bayes import kit


class _Mle(SimpleNamespace):
    maximum = "verified"

    def covariance(self):
        return self.cov

    def param_cb(self, name, alpha_ci=0.05):
        i = self.names.index(name)
        z = stats.norm.ppf(1 - alpha_ci / 2)
        se = np.sqrt(self.cov[i, i])
        return np.array([self.params[i] - z * se, self.params[i] + z * se])


@register_target
class RateTarget(Target):
    """Counts ``n_i`` over exposures ``e_i`` at one rate."""

    tag = "test-rate"
    family = "Poisson rate"
    names = ["rate"]
    bounds = [(0, None)]

    def __init__(self, counts, exposure):
        self.counts = np.asarray(counts, dtype=float)
        self.exposure = np.asarray(exposure, dtype=float)
        self.data = (self.counts, self.exposure)
        self.fixed = {}
        N, E = self.counts.sum(), self.exposure.sum()
        self.mle = _Mle(
            params=np.array([N / E]),
            cov=np.array([[N / E**2]]),
            names=["rate"],
        )

    def unit_log_lik(self, theta):
        mean = np.atleast_2d(theta)[:, :1] * self.exposure[None, :]
        return stats.poisson.logpmf(self.counts[None, :], mean)

    def unit_counts(self):
        return np.ones(len(self.counts))

    def neg_ll(self, theta):
        return -float(self.unit_log_lik(np.atleast_2d(theta)).sum())

    def with_params(self, theta):
        return SimpleNamespace(rate=float(np.ravel(theta)[0]))

    def simulate(self, theta, rng):
        return rng.poisson(theta[0] * self.exposure), self.exposure

    def statistics(self):
        return {
            "total count": lambda d: float(np.sum(d[0])),
            "largest count": lambda d: float(np.max(d[0])),
        }

    def conjugate(self, bound):
        prior = bound.spec.single.get("rate")
        fam = None if prior is None else _prior_family(prior)
        if fam is None or fam[0] != "gamma" or bound.spec.joint:
            return None
        a, b = fam[1]
        N, E = self.counts.sum(), self.exposure.sum()
        return ConjugateResult(
            "rate", "gamma", (a + N, b + E), 0.0, "Gamma prior on the rate"
        )

    def describe_data(self):
        return f"{len(self.counts)} units, {self.counts.sum():g} events"

    def same_data(self, other):
        return np.array_equal(self.counts, other.counts) and np.array_equal(
            self.exposure, other.exposure
        )

    def to_dict(self):
        return {
            "target": self.tag,
            "counts": self.counts.tolist(),
            "exposure": self.exposure.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["counts"], d["exposure"])


class LogLinearTarget(Target):
    """Counts with rate ``exp(a + b'z)`` a unit of exposure."""

    tag = "test-loglinear"
    family = "Poisson log-linear rate"

    def __init__(self, counts, exposure, Z):
        from scipy.optimize import minimize

        self.counts = np.asarray(counts, dtype=float)
        self.exposure = np.asarray(exposure, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.data = (self.counts, self.exposure)
        k = self.Z.shape[1]
        self.names = ["a"] + [f"b{j}" for j in range(k)]
        self.bounds = [(None, None)] * (k + 1)
        self.fixed = {}
        res = minimize(self.neg_ll, np.zeros(k + 1), method="BFGS")
        self.mle = _Mle(params=res.x, cov=res.hess_inv, names=self.names)

    def unit_log_lik(self, theta):
        theta = np.atleast_2d(theta)
        eta = theta[:, :1] + theta[:, 1:] @ self.Z.T
        return stats.poisson.logpmf(
            self.counts[None, :], np.exp(eta) * self.exposure[None, :]
        )

    def unit_counts(self):
        return np.ones(len(self.counts))

    def neg_ll(self, theta):
        return -float(self.unit_log_lik(np.atleast_2d(theta)).sum())

    def log_lik(self, theta):
        return self.unit_log_lik(theta).sum(axis=1)

    def with_params(self, theta):
        theta = np.asarray(theta, dtype=float)
        return SimpleNamespace(
            rate_at=lambda z: float(np.exp(theta[0] + theta[1:] @ z))
        )

    def simulate(self, theta, rng):
        eta = theta[0] + self.Z @ theta[1:]
        return rng.poisson(np.exp(eta) * self.exposure), self.exposure

    def statistics(self):
        return {"total count": lambda d: float(np.sum(d[0]))}

    def describe_data(self):
        return f"{len(self.counts)} units"

    def same_data(self, other):
        return np.array_equal(self.counts, other.counts) and np.array_equal(
            self.Z, other.Z
        )


COUNTS = [3, 5, 2, 8, 4, 6]
EXPOSURE = [10.0, 12.0, 8.0, 20.0, 11.0, 15.0]
GAMMA = {"rate": sp.Gamma.from_params([2.0, 5.0])}


def _loglinear(k, seed=3, units=40):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(units, k))
    exposure = rng.uniform(5, 15, size=units)
    beta = np.array([0.4, -0.3, 0.2, 0.1])[:k]
    counts = rng.poisson(np.exp(-1.0 + Z @ beta) * exposure)
    return LogLinearTarget(counts, exposure, Z)


def _normal_priors(target):
    return {n: sp.Normal.from_params([0.0, 2.0]) for n in target.names}


# -- the engine rule ----------------------------------------


def test_auto_is_closed_form_where_conjugate():
    model = fit_target(RateTarget(COUNTS, EXPOSURE), GAMMA)
    assert model.method == "conjugate"
    # Gamma(2 + 28, 5 + 76)
    exact = stats.gamma(30.0, scale=1 / 81.0)
    np.testing.assert_allclose(
        model.param_cri("rate"), exact.ppf([0.025, 0.975]), rtol=1e-12
    )
    assert model.params[0] == pytest.approx(exact.ppf(0.5), rel=1e-12)
    assert "closed form" in model.summary()


def test_auto_is_the_exact_grid_up_to_three():
    target = RateTarget(COUNTS, EXPOSURE)
    grid = fit_target(target, GAMMA, method="grid")
    assert grid.method == "grid"
    exact = stats.gamma(30.0, scale=1 / 81.0)
    np.testing.assert_allclose(
        grid.param_cri("rate"), exact.ppf([0.025, 0.975]), rtol=2e-4
    )
    # no conjugate prior: auto goes to the grid
    auto = fit_target(target, {"rate": sp.prior.between(0.1, 1.0)})
    assert auto.method == "grid"
    assert auto.diagnostics["approximate"] is False
    assert _loglinear(2).names == ["a", "b0", "b1"]
    assert fit_target(_loglinear(2)).method == "grid"


def test_auto_is_laplace_beyond_three_with_its_checks():
    target = _loglinear(3)
    model = fit_target(target, _normal_priors(target))
    assert model.method == "laplace"
    d = model.diagnostics
    assert d["approximate"] is True
    assert d["approximation"] == "good" and d["pareto_k"] < 0.5
    assert "Pareto k" in model.summary()


def test_grid_and_laplace_agree_on_a_near_normal_posterior():
    target = _loglinear(2, units=80)
    priors = _normal_priors(target)
    grid = fit_target(target, priors)
    laplace = fit_target(target, priors, method="laplace")
    for name in target.names:
        g = grid.param_cri(name)
        lap = laplace.param_cri(name)
        assert np.max(np.abs(g - lap)) < 0.03 * (g[1] - g[0])


def test_flat_priors_put_the_mode_at_the_mle():
    target = _loglinear(2)
    model = fit_target(target)
    np.testing.assert_allclose(
        model.diagnostics["mode"], target.mle.params, atol=2e-5
    )


def test_method_and_importance_are_checked():
    target = RateTarget(COUNTS, EXPOSURE)
    with pytest.raises(ValueError, match="method must be one of"):
        fit_target(target, GAMMA, method="mcmc")
    with pytest.raises(ValueError, match="No closed form"):
        fit_target(
            target, {"rate": sp.prior.between(0.1, 1)}, method="conjugate"
        )
    with pytest.raises(ValueError, match="importance=True reweights"):
        fit_target(target, GAMMA, importance=True)
    big = _loglinear(3)
    with pytest.raises(ValueError, match="the grid takes 1 to 3"):
        fit_target(big, _normal_priors(big), method="grid")
    with pytest.raises(ValueError, match="nothing to fit"):
        fit_target(target, {"rate": sp.prior.fixed(0.3)})


# -- the result ----------------------------------------


def test_function_intervals_and_probabilities():
    model = fit_target(RateTarget(COUNTS, EXPOSURE), GAMMA, method="grid")
    exact = stats.gamma(30.0, scale=1 / 81.0)
    lo, hi = model.function_cri(lambda m: 1.0 / m.rate)
    np.testing.assert_allclose(
        [lo, hi], 1 / exact.ppf([0.975, 0.025]), rtol=2e-3
    )
    p = model.probability(lambda m: 1.0 / m.rate >= 2.5)
    assert p == pytest.approx(exact.cdf(0.4), abs=2e-3)
    assert len(model.draw_models(5, random_state=1)) == 5
    assert model.at_params().rate == pytest.approx(model.params[0])
    with pytest.raises(ValueError, match="alpha_ci must be between"):
        model.function_cri(lambda m: m.rate, alpha_ci=2)


def test_summary_names_the_engine_the_defaults_and_a_conflict():
    target = RateTarget(COUNTS, EXPOSURE)
    with pytest.warns(UserWarning, match="disagree with the prior"):
        model = fit_target(target, {"rate": sp.prior.between(2.0, 3.0)})
    text = model.summary()
    assert text.startswith("Poisson rate: Bayesian fit, 6 units, 28 events")
    assert "Computed exactly on two offset" in text
    assert "The data disagree with the prior on rate" in text
    assert model.diagnostics["prior_conflict"]
    calm = fit_target(target, GAMMA)
    assert calm.diagnostics["prior_conflict"] == []
    table = calm.compare().table["rate"]
    assert table["mle"] == pytest.approx(28 / 76)
    assert table["mle_interval"][0] < table["mle"] < table["mle_interval"][1]


def test_waic_and_loo_are_per_unit_and_match_refitting():
    target = RateTarget(COUNTS, EXPOSURE)
    model = fit_target(target, GAMMA)
    loo = model.loo()
    assert loo.units == 6
    # leave each unit out exactly: Gamma(2 + N - n_i, 5 + E - e_i), whose
    # predictive for the unit left out is negative binomial
    counts, exposure = np.array(COUNTS), np.array(EXPOSURE)
    expected = []
    for i in range(6):
        a = 2.0 + counts.sum() - counts[i]
        b = 5.0 + exposure.sum() - exposure[i]
        expected.append(
            stats.nbinom.logpmf(counts[i], a, b / (b + exposure[i]))
        )
    np.testing.assert_allclose(loo.pointwise, expected, atol=2e-3)
    waic = model.waic()
    assert abs(waic.elpd - loo.elpd) < 0.2
    grid = fit_target(target, GAMMA, method="grid")
    assert grid.loo().elpd == pytest.approx(sum(expected), abs=2e-3)


def test_compare_ranks_fits_of_the_same_data():
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(60, 1))
    exposure = np.full(60, 10.0)
    counts = rng.poisson(np.exp(-1.0 + 0.8 * Z[:, 0]) * exposure)
    with_z = fit_target(LogLinearTarget(counts, exposure, Z))
    without = fit_target(
        LogLinearTarget(counts, exposure, Z), {"b0": sp.prior.fixed(0.0)}
    )
    table = sp.bayes.compare({"with": with_z, "without": without})
    assert table.best == "with"
    other = fit_target(LogLinearTarget(counts + 1, exposure, Z))
    with pytest.raises(ValueError, match="different data"):
        sp.bayes.compare({"a": with_z, "b": other})


def test_predictive_check_and_diagnose():
    target = RateTarget(COUNTS, EXPOSURE)
    model = fit_target(target, GAMMA)
    check = model.ppc(random_state=4)
    assert check.statistic == "total count" and check.observed == 28
    assert check.ok and check.interval[0] < 28 < check.interval[1]
    again = model.ppc(random_state=4)
    np.testing.assert_array_equal(check.replicated, again.replicated)
    assert "reproduces it" in repr(check)
    # a model that cannot make the data: one unit far from the rate
    odd = RateTarget([3, 5, 2, 60], [10.0, 12.0, 8.0, 10.0])
    bad = fit_target(odd, GAMMA).ppc("largest count", random_state=4)
    assert not bad.ok and bad.p_value < 0.025
    report = model.diagnose()
    assert report.ok and len(report.checks) == 4
    assert "predictive check, total count" in repr(report)
    with pytest.raises(ValueError, match="No statistic"):
        model.ppc("median")
    with pytest.raises(ValueError, match="at least 20"):
        model.ppc(replicates=5)


def test_saving_and_loading():
    model = fit_target(RateTarget(COUNTS, EXPOSURE), GAMMA, method="grid")
    d = json.loads(json.dumps(model.to_dict()))
    again = sp.from_dict(d)
    assert isinstance(again, BayesianResult)
    np.testing.assert_array_equal(again.params, model.params)
    assert again.summary() == model.summary()
    assert "test-rate" in TARGETS


# -- priors for any family ----------------------------------------


def test_a_prior_on_a_function_changes_variables():
    # a belief about the mean time between events, 1 / rate: the same
    # posterior as the same belief stated on the rate itself
    target = RateTarget(COUNTS, EXPOSURE)
    on_fn = sp.prior.on_quantities(
        None,
        sp.prior.on_function(
            lambda m: 1.0 / m.rate, sp.prior.between(2.0, 5.0), name="MTBE"
        ),
    )
    a = fit_target(target, on_fn)
    b = fit_target(target, {"rate": sp.prior.between(0.2, 0.5)})
    np.testing.assert_allclose(
        a.param_cri("rate"), b.param_cri("rate"), rtol=1e-4
    )
    assert "MTBE 90% between 2 and 5" in a.summary()
    # saved without its function, it still describes itself
    again = sp.from_dict(json.loads(json.dumps(a.to_dict())))
    assert "MTBE" in again.summary()
    with pytest.raises(ValueError, match="saved without"):
        fit_target(target, again.priors)


def test_statements_in_place_of_some_parameters_only():
    target = _loglinear(1)
    belief = sp.prior.on_quantities(
        None,
        sp.prior.on_function(
            lambda m: m.rate_at(np.zeros(1)),
            sp.prior.between(0.2, 0.6),
            name="rate at z = 0",
        ),
        instead_of="a",
    )
    slope = {"b0": sp.Normal.from_params([0.0, 1.0])}
    a = fit_target(target, [belief, slope])
    # the rate at z = 0 is exp(a): the same belief on a, on the log scale
    direct = {
        "a": sp.prior.NormalOnScale(
            loc=np.log(np.sqrt(0.2 * 0.6)),
            scale=np.log(0.6 / 0.2) / (2 * stats.norm.ppf(0.95)),
            kind="identity",
        ),
        **slope,
    }
    b = fit_target(target, direct)
    for name in target.names:
        np.testing.assert_allclose(
            a.param_cri(name), b.param_cri(name), atol=2e-4
        )
    with pytest.raises(ValueError, match="one statement"):
        fit_target(
            target,
            sp.prior.on_quantities(
                None,
                sp.prior.on_function(lambda m: 1.0, sp.prior.between(1, 2)),
                instead_of=["a", "b0"],
            ),
        )
    with pytest.raises(ValueError, match="quantities of a life distribution"):
        fit_target(
            target,
            [
                sp.prior.on_quantities(
                    sp.Weibull, B10=sp.prior.between(1, 2), instead_of="a"
                )
            ],
        )


def test_the_old_fit_as_the_prior():
    old = RateTarget([4, 6, 3], [10.0, 14.0, 9.0])
    new = RateTarget(COUNTS, EXPOSURE)
    # a power prior at weight 1 is fitting the two together
    together = fit_target(
        RateTarget([4, 6, 3] + COUNTS, [10.0, 14.0, 9.0] + EXPOSURE),
        method="grid",
    )
    power = fit_target(new, sp.prior.from_fit(old), method="grid")
    np.testing.assert_allclose(
        power.param_cri("rate"), together.param_cri("rate"), rtol=1e-6
    )
    # at weight 0.5 the old counts and exposure count for half: with a
    # flat prior on log rate the posterior is Gamma(N, E)
    half = fit_target(new, sp.prior.from_fit(old, weight=0.5), method="grid")
    exact = stats.gamma(28 + 6.5, scale=1 / (76 + 16.5))
    np.testing.assert_allclose(
        half.param_cri("rate"), exact.ppf([0.025, 0.975]), rtol=3e-4
    )
    # an earlier Bayesian fit as the prior: exact, its prior and its data
    first = fit_target(old, GAMMA)
    second = fit_target(new, first.as_prior(), method="grid")
    exact = stats.gamma(2 + 13 + 28, scale=1 / (5 + 33 + 76))
    np.testing.assert_allclose(
        second.param_cri("rate"), exact.ppf([0.025, 0.975]), rtol=3e-4
    )
    assert "the earlier fit's data at weight 1" in second.summary()
    # ... and through prior.from_model, with a weight
    third = fit_target(
        new, sp.prior.from_model(first, weight=0.5), method="grid"
    )
    exact = stats.gamma(2 + 6.5 + 28, scale=1 / (5 + 16.5 + 76))
    np.testing.assert_allclose(
        third.param_cri("rate"), exact.ppf([0.025, 0.975]), rtol=3e-4
    )
    # a summary of one parameter is a normal on its unconstrained scale
    summary = first.as_prior(param="rate", widen=2.0)
    assert type(summary).__name__ == "NormalOnScale"
    with pytest.raises(ValueError, match="same model for the old"):
        fit_target(_loglinear(1), sp.prior.from_fit(old))
    with pytest.raises(ValueError, match="holds its data"):
        sp.prior.from_fit(sp.Weibull.from_params([10, 2]))


# -- the validation kit ----------------------------------------


def test_the_kit_catches_a_wrong_posterior():
    exposure = np.array(EXPOSURE)
    prior = stats.gamma(2.0, scale=1 / 5.0)

    def truth(rng):
        return {"rate": float(prior.rvs(random_state=rng))}

    def simulate(t, rng):
        return rng.poisson(t["rate"] * exposure)

    good = kit.sbc(
        truth,
        simulate,
        lambda counts: fit_target(RateTarget(counts, exposure), GAMMA),
        ["rate"],
        reps=300,
        seed=7300,
    )
    kit.check_sbc(good)
    # the same fits with the exposure doubled in the likelihood: a wrong
    # posterior, which the rules reject
    wrong = kit.sbc(
        truth,
        simulate,
        lambda counts: fit_target(RateTarget(counts, 2 * exposure), GAMMA),
        ["rate"],
        reps=300,
        seed=7300,
    )
    with pytest.raises(AssertionError):
        kit.check_sbc(wrong)


def test_the_kit_compares_with_a_reference():
    model = fit_target(RateTarget(COUNTS, EXPOSURE), GAMMA, method="grid")
    exact = stats.gamma(30.0, scale=1 / 81.0)
    q = exact.ppf([0.025, 0.5, 0.975])
    reference = {"parameters": {"rate": {"q": q.tolist(), "mcse": [1e-4] * 3}}}
    assert kit.check_reference(model, reference)
    off = {
        "parameters": {"rate": {"q": (1.2 * q).tolist(), "mcse": [1e-4] * 3}}
    }
    with pytest.raises(AssertionError):
        kit.check_reference(model, off)


def test_no_new_warnings_from_a_plain_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_target(RateTarget(COUNTS, EXPOSURE), GAMMA)
        fit_target(RateTarget(COUNTS, EXPOSURE), method="grid")
