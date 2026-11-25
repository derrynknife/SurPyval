"""Bayesian fits of the Crow-AMSAA process and the HPP
(``surpyval.recurrent.bayes``).

Checked against:

- the gamma-Poisson closed form for the HPP (posterior, predictive count
  as a negative binomial, probability of meeting an MTBF);
- the closed form for the Crow-AMSAA shape with no priors: for systems
  observed from 0 to a common end ``T``, with ``N`` failures and
  ``S = sum log(T / t)``, the likelihood in ``lambda = alpha^-beta`` is
  ``lambda^N beta^N prod t^(beta - 1) exp(-k lambda T^beta)``. A prior
  flat on ``log alpha`` and ``log beta`` is ``1 / (lambda beta^2)`` in
  ``(lambda, beta)``; integrating ``lambda`` out leaves ``beta^(N - 2)
  exp(-beta S)``: ``beta`` is Gamma(N - 1, rate S);
- the maximum-likelihood fit (with no priors the posterior's peak on the
  log scale is the maximum-likelihood estimate);
- frozen PyMC references (``references/crow_amsaa_*.json``, made by
  ``scripts/reference/make_crow_amsaa.py``), by the kit's rule;
- simulation-based calibration is in
  ``tests/calibration/test_sbc_recurrent_bayes.py``.
"""

import json
import pickle
import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.bayes.prior import NormalOnScale
from surpyval.recurrent import HPP, CrowAMSAA
from surpyval.recurrent.bayes import BayesianRecurrence
from surpyval.tests.bayes import kit

TIMES = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390]
X = TIMES + [400]
C = [0] * 11 + [1]
GAMMA = {"lambda": sp.Gamma.from_params([2.0, 100.0])}


def _three_systems():
    times = [30, 110, 240, 420, 55, 190, 380, 610, 820, 95, 300]
    ids = [1] * 4 + [2] * 5 + [3] * 2
    ends = [500.0, 900.0, 350.0]
    x = times + ends
    i = ids + [1, 2, 3]
    c = [0] * 11 + [1, 1, 1]
    return x, i, c


# -- the HPP: gamma-Poisson --------------------


def test_hpp_gamma_prior_is_the_closed_form():
    x = [20, 55, 61, 130, 200, 30, 90, 250]
    i = [1, 1, 1, 1, 1, 2, 2, 2]
    c = [0, 0, 0, 0, 1, 0, 0, 1]
    model = HPP.fit_bayes(x, i, c, priors=GAMMA)
    assert isinstance(model, BayesianRecurrence)
    assert model.method == "conjugate"
    # 6 failures in 200 + 250 = 450 of time: Gamma(2 + 6, 100 + 450)
    exact = stats.gamma(8.0, scale=1 / 550.0)
    np.testing.assert_allclose(
        model.param_cri("lambda"), exact.ppf([0.025, 0.975]), rtol=1e-12
    )
    assert model.params[0] == pytest.approx(exact.ppf(0.5), rel=1e-12)
    assert float(model.cif(100.0)) == pytest.approx(100 * 8 / 550, rel=1e-12)
    assert float(model.iif(100.0)) == pytest.approx(8 / 550, rel=1e-12)
    np.testing.assert_allclose(
        model.cri(50.0, on="mtbf"), 1 / exact.ppf([0.975, 0.025]), rtol=1e-12
    )
    assert model.mtbf_probability(50.0) == pytest.approx(
        exact.cdf(1 / 50.0), rel=1e-12
    )
    # the grid gives the same posterior
    grid = HPP.fit_bayes(x, i, c, priors=GAMMA, method="grid")
    np.testing.assert_allclose(
        grid.param_cri("lambda"), exact.ppf([0.025, 0.975]), rtol=3e-4
    )
    assert grid.mtbf_probability(50.0) == pytest.approx(
        exact.cdf(1 / 50.0), abs=1e-3
    )


def test_hpp_predicted_count_is_negative_binomial():
    model = HPP.fit_bayes(
        [20, 55, 61, 130, 200], c=[0, 0, 0, 0, 1], priors=GAMMA
    )
    # Gamma(6, 300): the count over 150 of time is negative binomial
    nxt = model.count(duration=150.0)
    exact = stats.nbinom(6, 300 / (300 + 150))
    k = np.arange(0, 15)
    np.testing.assert_allclose(nxt.pmf(k), exact.pmf(k), rtol=1e-12)
    np.testing.assert_allclose(nxt.cdf(k), exact.cdf(k), rtol=1e-12)
    assert nxt.mean == pytest.approx(exact.mean(), rel=1e-12)
    # the same from the grid's posterior, a Poisson mixture over its points
    grid = HPP.fit_bayes(
        [20, 55, 61, 130, 200], c=[0, 0, 0, 0, 1], priors=GAMMA, method="grid"
    ).count(duration=150.0)
    np.testing.assert_allclose(grid.pmf(k), exact.pmf(k), atol=2e-5)
    assert nxt.interval(0.1) == tuple(exact.ppf([0.05, 0.95]))
    assert float(nxt.pmf(-1)) == 0 and float(nxt.pmf(2.5)) == 0
    three = model.count(200.0, 250.0, systems=3)
    assert three.mean == pytest.approx(3 * 50 * 6 / 300, rel=1e-12)
    assert "Predicted events in (200, 250] over 3 systems" in repr(three)
    with pytest.raises(ValueError, match="end or its duration"):
        model.count()
    with pytest.raises(ValueError, match="must end after it starts"):
        model.count(300.0, 200.0)


def test_hpp_with_no_failures_rests_on_the_prior():
    # the maximum-likelihood fit refuses data with no events; with a
    # proper prior the posterior is Gamma(a, b + time watched)
    model = HPP.fit_bayes([100.0], c=[1], priors=GAMMA)
    assert model.mle is None and model.method == "conjugate"
    exact = stats.gamma(2.0, scale=1 / 200.0)
    np.testing.assert_allclose(
        model.param_cri("lambda"), exact.ppf([0.025, 0.975]), rtol=1e-12
    )
    assert "0 events" in model.summary()
    with pytest.raises(ValueError, match="no prior"):
        HPP.fit_bayes([100.0], c=[1])
    grid = HPP.fit_bayes(
        [100.0], c=[1], priors={"lambda": sp.prior.between(0.002, 0.05)}
    )
    assert grid.method == "grid" and grid.mle is None


def test_hpp_counts_in_windows_are_conjugate_too():
    # interval-censored counts: 3 in (0, 50], 5 in (50, 150]
    x = [[0, 50], [50, 150]]
    model = HPP.fit_bayes(x, c=[2, 2], n=[3, 5], priors=GAMMA)
    exact = stats.gamma(2.0 + 8, scale=1 / (100.0 + 150))
    assert model.method == "conjugate"
    np.testing.assert_allclose(
        model.param_cri("lambda"), exact.ppf([0.025, 0.975]), rtol=1e-12
    )


# -- Crow-AMSAA: the grid --------------------


def test_crow_amsaa_flat_priors_give_the_gamma_posterior_of_beta():
    model = CrowAMSAA.fit_bayes(X, c=C)
    assert model.method == "grid"
    assert model.diagnostics["approximate"] is False
    S = float(np.sum(np.log(400.0 / np.array(TIMES))))
    exact = stats.gamma(11 - 1, scale=1 / S)
    np.testing.assert_allclose(
        model.param_cri("beta"), exact.ppf([0.025, 0.975]), rtol=2e-3
    )
    assert model.params[1] == pytest.approx(exact.ppf(0.5), rel=1e-3)
    # ... and for two systems to a common end, the same form
    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
    i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    two = CrowAMSAA.fit_bayes(x, i, c)
    t = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44])
    exact = stats.gamma(9 - 1, scale=1 / np.sum(np.log(60.0 / t)))
    np.testing.assert_allclose(
        two.param_cri("beta"), exact.ppf([0.025, 0.975]), rtol=2e-3
    )


def test_crow_amsaa_flat_priors_peak_at_the_mle():
    model = CrowAMSAA.fit_bayes(X, c=C)
    mle = CrowAMSAA.fit(X, c=C)
    np.testing.assert_allclose(
        model.diagnostics["mode"], mle.params, rtol=1e-5
    )
    np.testing.assert_allclose(model.mle.params, mle.params, rtol=1e-12)


@pytest.mark.parametrize(
    "name", ["crow_amsaa_one_system", "crow_amsaa_three_systems"]
)
def test_crow_amsaa_against_pymc(name):
    if name == "crow_amsaa_one_system":
        data = dict(x=X, c=C)
        priors = {
            "alpha": NormalOnScale(loc=np.log(40.0), scale=1.0, kind="log"),
            "beta": NormalOnScale(loc=np.log(0.7), scale=0.35, kind="log"),
        }
    else:
        x, i, c = _three_systems()
        data = dict(x=x, i=i, c=c)
        priors = {
            "alpha": NormalOnScale(loc=np.log(100.0), scale=2.0, kind="log"),
            "beta": NormalOnScale(loc=0.0, scale=0.5, kind="log"),
        }
    model = CrowAMSAA.fit_bayes(**data, priors=priors)
    assert model.method == "grid"
    reference = kit.load_reference(name)
    assert reference["sampler"]["divergences"] == 0
    for line in kit.check_reference(model, reference):
        print(name, line)


def test_crow_amsaa_laplace_is_close_to_the_grid():
    x, i, c = _three_systems()
    priors = {"beta": sp.prior.around(1.0, between=(0.5, 2.0))}
    grid = CrowAMSAA.fit_bayes(x, i, c, priors=priors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        laplace = CrowAMSAA.fit_bayes(
            x, i, c, priors=priors, method="laplace", importance=True
        )
    assert laplace.method == "laplace"
    assert "pareto_k" in laplace.diagnostics
    for name in ("alpha", "beta"):
        g = grid.param_cri(name)
        assert np.max(np.abs(laplace.param_cri(name) - g)) < 0.08 * (
            g[1] - g[0]
        )


# -- priors: the growth rate, the MTBF, an earlier phase --------------------


def test_a_prior_on_the_growth_rate():
    belief = {"beta": sp.prior.around(0.6, between=(0.5, 0.72))}
    flat = CrowAMSAA.fit_bayes(X, c=C)
    model = CrowAMSAA.fit_bayes(X, c=C, priors=belief)
    assert np.diff(model.param_cri("beta")) < np.diff(flat.param_cri("beta"))
    assert model.params[1] < flat.params[1]
    assert "beta: about 0.6" in model.summary()
    fixed = CrowAMSAA.fit_bayes(X, c=C, priors={"beta": sp.prior.fixed(1.0)})
    # beta = 1 is the HPP with rate 1 / alpha: Gamma(N, T) for the rate
    # with a prior flat on its log
    exact = stats.gamma(11.0, scale=1 / 400.0)
    np.testing.assert_allclose(
        1 / fixed.param_cri("alpha")[::-1],
        exact.ppf([0.025, 0.975]),
        rtol=5e-4,
    )


def test_a_prior_on_the_mtbf_at_the_end_of_the_test():
    # with beta held at 1 the MTBF at any time is alpha: a belief about
    # it is the same belief about alpha
    hold = {"beta": sp.prior.fixed(1.0)}
    on_mtbf = sp.prior.on_quantities(
        CrowAMSAA, mtbf_at={400.0: sp.prior.between(30.0, 60.0)}
    )
    a = CrowAMSAA.fit_bayes(X, c=C, priors=[on_mtbf, hold])
    b = CrowAMSAA.fit_bayes(
        X, c=C, priors={"alpha": sp.prior.between(30.0, 60.0), **hold}
    )
    np.testing.assert_allclose(
        a.param_cri("alpha"), b.param_cri("alpha"), rtol=1e-5
    )
    # with both free: the MTBF's belief with one about beta, a statement
    # for each parameter
    both = sp.prior.on_quantities(
        CrowAMSAA,
        mtbf_at={400.0: sp.prior.between(45.0, 55.0)},
        beta=sp.prior.between(0.5, 1.0),
    )
    model = CrowAMSAA.fit_bayes(X, c=C, priors=both)
    lo, hi = model.mtbf_cri()
    flat = CrowAMSAA.fit_bayes(X, c=C).mtbf_cri()
    assert flat[0] < lo < hi < flat[1]
    assert 40 < lo and hi < 62
    assert "MTBF at 400 90% between 45 and 55" in model.summary()
    # ... or the MTBF's belief in place of the prior on alpha only
    partly = sp.prior.on_quantities(
        CrowAMSAA,
        mtbf_at={400.0: sp.prior.between(45.0, 55.0)},
        instead_of="alpha",
    )
    model = CrowAMSAA.fit_bayes(X, c=C, priors=partly)
    assert "beta: no prior given" in model.summary()
    # the intensity is the MTBF's reciprocal: the same belief either way
    on_rate = sp.prior.on_quantities(
        CrowAMSAA,
        intensity_at={400.0: sp.prior.between(1 / 55.0, 1 / 45.0)},
        instead_of="alpha",
    )
    other = CrowAMSAA.fit_bayes(X, c=C, priors=on_rate)
    np.testing.assert_allclose(
        other.param_cri("beta"), model.param_cri("beta"), rtol=1e-4
    )
    with pytest.raises(ValueError, match="does not know 'mttf_at'"):
        CrowAMSAA.fit_bayes(
            X,
            c=C,
            priors=sp.prior.on_quantities(
                CrowAMSAA,
                mttf_at={400.0: sp.prior.between(45.0, 55.0)},
                instead_of="alpha",
            ),
        )


def test_an_earlier_phase_as_the_prior():
    # phase 1: one system to 400; phase 2: a second system to 300
    x2 = [22, 70, 135, 210, 280, 300]
    c2 = [0, 0, 0, 0, 0, 1]
    phase1 = CrowAMSAA.fit(X, c=C)
    # the earlier phase's data at weight 1: the two systems together
    together = CrowAMSAA.fit_bayes(X + x2, [1] * 12 + [2] * 6, C + c2)
    power = CrowAMSAA.fit_bayes(x2, c=c2, priors=sp.prior.from_fit(phase1))
    for name in ("alpha", "beta"):
        np.testing.assert_allclose(
            power.param_cri(name), together.param_cri(name), rtol=2e-3
        )
    assert "at weight 1" in power.summary()
    # discounted, the posterior is wider and nearer phase 2's own
    half = CrowAMSAA.fit_bayes(
        x2, c=c2, priors=sp.prior.from_fit(phase1, weight=0.3)
    )
    assert np.diff(half.param_cri("beta")) > np.diff(power.param_cri("beta"))
    # the earlier fit's estimates and covariance as a normal prior
    normal = CrowAMSAA.fit_bayes(x2, c=c2, priors=sp.prior.from_model(phase1))
    assert "from an earlier Crow-AMSAA process fit" in normal.summary()
    width = np.diff(together.param_cri("beta"))[0]
    assert abs(normal.params[1] - together.params[1]) < 0.3 * width
    # an earlier Bayesian fit: its priors and its data, exactly
    belief = {"beta": sp.prior.around(0.7, between=(0.4, 1.2))}
    first = CrowAMSAA.fit_bayes(X, c=C, priors=belief)
    second = CrowAMSAA.fit_bayes(x2, c=c2, priors=sp.prior.from_model(first))
    direct = CrowAMSAA.fit_bayes(
        X + x2, [1] * 12 + [2] * 6, C + c2, priors=belief
    )
    for name in ("alpha", "beta"):
        np.testing.assert_allclose(
            second.param_cri(name), direct.param_cri(name), rtol=2e-3
        )
    with pytest.raises(ValueError, match="prior.from_fit"):
        CrowAMSAA.fit_bayes(x2, c=c2, priors=sp.prior.from_data(TIMES))
    with pytest.raises(ValueError, match="same model for the old"):
        HPP.fit_bayes(x2, c=c2, priors=sp.prior.from_fit(phase1))


# -- results --------------------


def test_current_and_projected_mtbf_and_the_requirement():
    model = CrowAMSAA.fit_bayes(
        X, c=C, priors={"beta": sp.prior.around(0.7, between=(0.4, 1.2))}
    )
    assert model.end == 400.0
    now = model.mtbf_cri()
    later = model.mtbf_cri(800.0)
    np.testing.assert_array_equal(now, model.cri(400.0, on="mtbf"))
    assert now[0] < float(model.mtbf(400.0)) < now[1]
    # growth (beta below 1): the projected MTBF is higher
    assert float(model.mtbf(800.0)) > float(model.mtbf(400.0))
    assert later[1] > now[1]
    # the probability of meeting a requirement is the posterior's share
    # above it: it falls as the requirement rises, and is a half at the
    # median
    p = [model.mtbf_probability(r) for r in (20.0, 40.0, 60.0, 90.0)]
    assert all(a > b for a, b in zip(p, p[1:]))
    median = float(model.mtbf(400.0))
    assert model.mtbf_probability(median) == pytest.approx(0.5, abs=0.01)
    assert model.mtbf_probability(40.0, at=800.0) > p[1]
    assert model.probability(lambda m: m.mtbf(400.0) >= 40.0) == pytest.approx(
        p[1], abs=2e-3
    )
    # the general interval agrees with the vectorised one
    np.testing.assert_allclose(
        model.function_cri(lambda m: m.mtbf(400.0)), now, rtol=5e-3
    )
    lower = model.cri([400.0, 800.0], on="mtbf", alpha_ci=0.1, bound="lower")
    assert lower.shape == (2,)
    assert model.cri(400.0, on="cif").shape == (2,)
    assert model.cri(400.0, on="iif")[0] == pytest.approx(1 / now[1])
    with pytest.raises(ValueError, match="on must be one of"):
        model.cri(400.0, on="sf")
    with pytest.raises(ValueError, match="use mtbf_cri"):
        model.mtbf_cb(400.0)
    with pytest.raises(ValueError, match="positive MTBF"):
        model.mtbf_probability(0.0)


def test_the_predicted_count_is_wider_than_a_poisson_at_the_estimates():
    model = CrowAMSAA.fit_bayes(X, c=C)
    nxt = model.count(duration=200.0)
    plug_in = float(model.mle.cif(600.0) - model.mle.cif(400.0))
    assert nxt.mean == pytest.approx(
        float(model.cif(600.0) - model.cif(400.0)), rel=1e-9
    )
    lo, hi = nxt.interval(0.1)
    p_lo, p_hi = stats.poisson(plug_in).ppf([0.05, 0.95])
    assert lo <= p_lo and hi > p_hi
    assert float(nxt.cdf(1e9)) == 1.0
    assert abs(float(np.sum(nxt.pmf(np.arange(0, 400)))) - 1) < 1e-9


def test_summary_compare_and_saving():
    model = CrowAMSAA.fit_bayes(
        X, c=C, priors={"beta": sp.prior.around(0.7, between=(0.4, 1.2))}
    )
    text = model.summary()
    assert text.startswith(
        "Crow-AMSAA process: Bayesian fit, 1 system, 11 events, observed "
        "to 400"
    )
    assert "alpha: no prior given; flat on the log scale" in text
    assert "Computed exactly on two offset" in text
    table = model.compare().table
    assert table["beta"]["mle"] == pytest.approx(model.mle.params[1])
    again = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert isinstance(again, BayesianRecurrence)
    np.testing.assert_array_equal(again.params, model.params)
    np.testing.assert_array_equal(again.mtbf_cri(), model.mtbf_cri())
    assert again.summary() == text
    pickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(pickled.mtbf_cri(), model.mtbf_cri())
    hours = CrowAMSAA.fit_bayes(X, c=C, unit="hours")
    assert hours.unit == "hours"


# -- criteria and checks: the unit is one system --------------------


def _fleet(seed=5, systems=8, beta=1.4):
    rng = np.random.default_rng(seed)
    xs, ids, cs = [], [], []
    for k in range(systems):
        end = 300.0
        total = (end / 60.0) ** beta
        m = rng.poisson(total)
        times = 60.0 * np.sort(total * rng.uniform(size=m)) ** (1 / beta)
        xs += [*times, end]
        ids += [k] * (m + 1)
        cs += [0] * m + [1]
    return xs, ids, cs


def test_loo_and_waic_are_per_system():
    x, i, c = _fleet()
    model = CrowAMSAA.fit_bayes(x, i, c)
    loo = model.loo()
    assert loo.units == 8 and len(loo.pointwise) == 8
    # the systems' log-likelihoods sum to the whole
    target = model.target
    theta = np.array([model.params, model.params * 1.1])
    np.testing.assert_allclose(
        target.unit_log_lik(theta).sum(axis=1),
        [-target.neg_ll(t) for t in theta],
        rtol=1e-10,
    )
    # leaving a system out exactly: refit without it and evaluate its
    # predictive density from that posterior
    first = np.asarray(i) == 0
    rest = CrowAMSAA.fit_bayes(
        np.asarray(x)[~first], np.asarray(i)[~first], np.asarray(c)[~first]
    )
    only = CrowAMSAA.fit(np.asarray(x)[first], c=np.asarray(c)[first])
    from surpyval.recurrent.bayes import RecurrenceTarget

    held = RecurrenceTarget(CrowAMSAA, only.data, only)
    ll = held.log_lik(rest.posterior.points)
    exact = float(np.log(rest.posterior.weights @ np.exp(ll)))
    assert loo.pointwise[0] == pytest.approx(exact, abs=5e-3)
    assert abs(model.waic().elpd - loo.elpd) < 0.5
    hpp = HPP.fit_bayes(x, i, c)
    table = sp.bayes.compare({"power law": model, "constant": hpp})
    assert table.best == "power law"


def test_predictive_checks_see_a_trend_the_model_lacks():
    x, i, c = _fleet()
    good = CrowAMSAA.fit_bayes(x, i, c)
    report = good.diagnose()
    assert report.ok
    assert [name for name, _, _ in report.checks][2:] == [
        "predictive check, number of events",
        "predictive check, share of events in the later half of "
        "observation",
        "predictive check, most events on one system",
    ]
    # a constant rate cannot put this many of the events late
    flat = HPP.fit_bayes(x, i, c)
    check = flat.ppc(
        "share of events in the later half of observation", random_state=1
    )
    assert not check.ok and check.p_value < 0.025
    assert flat.ppc("number of events", random_state=1).ok


# -- what is refused --------------------


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match="starts from the maximum-lik"):
        CrowAMSAA.fit_bayes([-5.0, 10.0, 20.0], c=[0, 0, 1])
    with pytest.raises(ValueError, match="no prior"):
        CrowAMSAA.fit_bayes([100.0], c=[1])
    with pytest.raises(ValueError, match="No parameter 'rate'"):
        HPP.fit_bayes(X, c=C, priors={"rate": sp.prior.between(0.01, 0.1)})
    with pytest.raises(ValueError, match="method must be one of"):
        CrowAMSAA.fit_bayes(X, c=C, method="nuts")
    with pytest.raises(ValueError, match="importance=True reweights"):
        CrowAMSAA.fit_bayes(X, c=C, importance=True)
    with pytest.raises(ValueError, match="quantities of a life distribution"):
        CrowAMSAA.fit_bayes(
            X,
            c=C,
            priors=sp.prior.on_quantities(
                CrowAMSAA,
                B10=sp.prior.between(10, 20),
                beta=sp.prior.between(0.5, 1),
            ),
        )
    with pytest.warns(UserWarning, match="disagree with the prior"):
        model = CrowAMSAA.fit_bayes(
            X, c=C, priors={"beta": sp.prior.between(2.5, 3.0)}
        )
    assert "The data disagree with the prior on beta" in model.summary()
