"""Bayesian fits of the Wiener and gamma degradation processes
(``WienerProcess.fit_bayes``, ``GammaProcess.fit_bayes``)."""

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.degradation import (
    GammaProcess,
    GammaProcessModel,
    WienerProcess,
    WienerProcessModel,
)
from surpyval.degradation.bayes import BayesianProcessModel, ProcessLife


def _readings(kind, units=5, reads=11, seed=1, start=0.0):
    rng = np.random.default_rng(seed)
    t = np.tile(np.arange(0, 10.0 * reads, 10.0), units)
    i = np.repeat(np.arange(units), reads)
    if kind == "wiener":
        s = rng.normal(0.5 * 10, 1.0 * np.sqrt(10), size=(units, reads - 1))
    else:
        s = rng.gamma(shape=2.0 * 10, scale=0.25, size=(units, reads - 1))
    y = np.hstack([np.r_[start, start + np.cumsum(r)] for r in s])
    return t, y, i


@pytest.fixture(scope="module")
def wiener():
    t, y, i = _readings("wiener")
    return WienerProcess.fit_bayes(t, y, i, threshold=100)


@pytest.fixture(scope="module")
def gamma():
    t, y, i = _readings("gamma")
    return GammaProcess.fit_bayes(t, y, i, threshold=100)


def test_no_prior_wiener_is_the_normal_inverse_gamma(wiener):
    # Flat on mu and log sigma is p(mu, sigma^2) ~ 1 / sigma^2: sigma^2
    # is S / chi2(n - 1), and mu a Student t(n - 1) about the
    # maximum-likelihood drift with scale sqrt(S / ((n - 1) sum dt)).
    t, y, i = _readings("wiener")
    dt = np.tile(np.diff(t[:11]), 5)
    dy = np.concatenate([np.diff(y[i == u]) for u in range(5)])
    n = dy.size
    mu_hat = dy.sum() / dt.sum()
    S = np.sum((dy - mu_hat * dt) ** 2 / dt)
    q = np.array([0.025, 0.5, 0.975])
    mu_exact = stats.t.ppf(q, n - 1, mu_hat, np.sqrt(S / ((n - 1) * dt.sum())))
    sigma_exact = np.sqrt(S / stats.chi2.ppf(1 - q, n - 1))
    assert wiener.method == "grid"
    width_mu = mu_exact[2] - mu_exact[0]
    width_s = sigma_exact[2] - sigma_exact[0]
    got_mu = wiener.posterior.quantile("mu", q)
    got_s = wiener.posterior.quantile("sigma", q)
    np.testing.assert_allclose(got_mu, mu_exact, atol=2e-3 * width_mu)
    np.testing.assert_allclose(got_s, sigma_exact, atol=2e-3 * width_s)


@pytest.mark.parametrize(
    "model",
    [
        WienerProcessModel(0.5, 0.9, 100.0, y0=3.0),
        WienerProcessModel(-0.02, 0.9, 20.0),
        GammaProcessModel(2.0, 4.0, 100.0, y0=5.0),
        GammaProcessModel(0.3, 0.2, 10.0),
    ],
)
def test_the_life_at_every_point_is_the_models(model):
    from surpyval.degradation.bayes import _target_of_fit

    target = _target_of_fit(model)
    life = ProcessLife(target, model.params[None, :], None, None)
    t = np.array([0.0, 5.0, 50.0, 150.0, 200.0, 400.0, 1e5, np.inf])
    np.testing.assert_allclose(life.sf(t)[0], model.sf(t), rtol=1e-12)
    np.testing.assert_allclose(
        life.ff(t)[0], model.ff(t), rtol=1e-12, atol=1e-300
    )
    for p in (0.01, 0.1, 0.5, 0.9):
        if p < float(model.ff(np.inf)):
            np.testing.assert_allclose(
                life.quantile(p)[0], model.qf(p), rtol=1e-8
            )
        else:
            assert np.isposinf(life.quantile(p)[0])
    if float(model.ff(np.inf)) < 1:
        # a unit that may never fail has no finite mean life
        assert np.isposinf(life.mean()[0])
    else:
        np.testing.assert_allclose(life.mean()[0], model.mean(), rtol=1e-7)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_the_likelihood_is_the_fits(kind):
    t, y, i = _readings(kind)
    y = np.round(y, 0) if kind == "gamma" else y  # zero increments too
    fitter = WienerProcess if kind == "wiener" else GammaProcess
    stress = np.where(i < 3, 1.0, 2.0)
    for Z in (None, stress):
        model = fitter.fit_bayes(t, y, i, threshold=100, Z=Z)
        target = model.target
        assert np.isclose(
            target.neg_ll(target.start()), model.mle.neg_ll(), rtol=1e-10
        )
        theta = np.stack([target.start(), target.start() * 1.05])
        np.testing.assert_allclose(
            target.unit_log_lik(theta).sum(axis=1),
            target.log_lik(theta),
            rtol=1e-12,
        )
    assert model.method == "grid"
    assert model.parameter_names[-1] == "gamma_0"


def test_predictions(wiener):
    x = np.array([150.0, 200.0, 250.0])
    sf = wiener.sf(x)
    np.testing.assert_allclose(sf + wiener.ff(x), 1.0, atol=1e-12)
    band = wiener.cri(x)
    assert np.all((band[:, 0] < sf) & (sf < band[:, 1]))
    lo, hi = wiener.quantile_cri(0.1)
    b10 = wiener.quantile_cri(0.1, point=True)
    assert lo < b10 < hi
    q = wiener.qf(0.1)
    assert np.isclose(wiener.ff(q), 0.1, atol=1e-9)
    # the predictive quantile carries the parameters' uncertainty: wider
    # than the B10 life at the posterior medians
    assert q < b10
    lo, hi = wiener.mean_cri()
    assert lo < wiener.mean() < hi
    assert wiener.cri(200.0, bound="lower") < wiener.cri(200.0)[1]


def test_remaining_life(gamma):
    rul = gamma.predict_rul(50.0)
    assert (
        rul.interval[0]
        < rul.median_cri[0]
        < rul.median
        < rul.median_cri[1]
        < rul.interval[1]
    )
    # a fresh passage over the distance that is left
    assert np.isclose(rul.median, gamma.qf(0.5, y0=50.0))
    assert np.isclose(float(rul.sf(rul.median)), 0.5, atol=1e-9)
    assert rul.cri([10.0, 100.0]).shape == (2, 2)
    with pytest.raises(ValueError, match="past the threshold"):
        gamma.predict_rul(120.0)


def test_drift_from_an_earlier_fleet():
    t, y, i = _readings("wiener", seed=1)
    t2, y2, i2 = _readings("wiener", units=3, seed=2)
    old = WienerProcess.fit(t, y, i, threshold=100)
    flat = WienerProcess.fit_bayes(t2, y2, i2, threshold=100)
    drift = sp.prior.from_model(old, param="mu", weight=0.5)
    informed = WienerProcess.fit_bayes(
        t2, y2, i2, threshold=100, priors={"mu": drift}
    )
    w_flat = np.diff(flat.param_cri("mu"))[0]
    w_informed = np.diff(informed.param_cri("mu"))[0]
    assert w_informed < 0.8 * w_flat


def test_earlier_fleets_readings_at_full_weight_are_pooling():
    # A power prior of weight 1 from the old fleet is the same as fitting
    # both fleets' readings together.
    t, y, i = _readings("gamma", seed=1)
    t2, y2, i2 = _readings("gamma", units=3, seed=2)
    old = GammaProcess.fit_bayes(t, y, i, threshold=100)
    both = GammaProcess.fit_bayes(
        np.r_[t, t2], np.r_[y, y2], np.r_[i, i2 + 10], threshold=100
    )
    new = GammaProcess.fit_bayes(
        t2, y2, i2, threshold=100, priors=[sp.prior.from_fit(old)]
    )
    for name in ("alpha", "beta"):
        a, b = new.param_cri(name), both.param_cri(name)
        np.testing.assert_allclose(a, b, atol=5e-3 * (b[1] - b[0]))


def test_a_belief_about_the_life():
    t, y, i = _readings("wiener", units=2, reads=4)
    belief = sp.prior.on_quantities(
        WienerProcess,
        B10=sp.prior.around(150, between=(130, 175)),
        instead_of=["mu"],
    )
    flat = WienerProcess.fit_bayes(t, y, i, threshold=100)
    held = WienerProcess.fit_bayes(t, y, i, threshold=100, priors=[belief])
    lo, hi = held.quantile_cri(0.1)
    f_lo, f_hi = flat.quantile_cri(0.1)
    assert hi - lo < f_hi - f_lo
    assert 120 < lo < hi < 190
    gamma_belief = sp.prior.on_quantities(
        GammaProcess,
        mean=sp.prior.around(200, between=(180, 220)),
        instead_of=["alpha"],
    )
    tg, yg, ig = _readings("gamma", units=2, reads=4)
    fit = GammaProcess.fit_bayes(tg, yg, ig, 100, priors=[gamma_belief])
    lo, hi = fit.mean_cri()
    assert 170 < lo < hi < 235


def test_stress():
    t, y, i = _readings("wiener", units=6)
    Z = np.where(i < 3, 1.0, 2.0)
    # the second group's clock runs exp(0.5) times faster
    y = np.where(i < 3, y, y * np.exp(0.5))
    model = WienerProcess.fit_bayes(t, y, i, threshold=100, Z=Z)
    assert model.is_accelerated and model.method == "grid"
    lo, hi = model.param_cri("gamma_0")
    assert lo < 0.5 < hi
    slow, fast = model.sf(150.0, [1.0]), model.sf(150.0, [2.0])
    assert fast < slow
    with pytest.raises(ValueError, match="pass Z"):
        model.sf(150.0)


def test_refusals():
    t, y, i = _readings("gamma")
    with pytest.raises(ValueError, match="gauge"):
        GammaProcess.fit_bayes(t, y, i, threshold=100, gauge=1.0)
    with pytest.raises(ValueError, match="no prior"):
        WienerProcess.fit_bayes([0, 1, 2], [0, 1, 3], [0, 0, 0], 10)
    model = WienerProcess.fit_bayes(t, y, i, threshold=100)
    with pytest.raises(ValueError, match="without stress"):
        model.sf(10.0, [1.0])
    with pytest.raises(ValueError, match="must be one of"):
        model.cri(10.0, on="hf")


def test_saved_and_loaded(gamma, wiener):
    for model in (gamma, wiener):
        back = sp.from_dict(model.to_dict())
        assert isinstance(back, BayesianProcessModel)
        np.testing.assert_allclose(
            back.sf([150.0, 200.0]), model.sf([150.0, 200.0])
        )
        np.testing.assert_allclose(
            back.quantile_cri(0.1), model.quantile_cri(0.1)
        )


def test_criteria_and_checks(wiener):
    loo = wiener.loo()
    assert np.isfinite(loo.elpd)
    check = wiener.ppc(random_state=3)
    assert 0 <= check.p_value <= 1
