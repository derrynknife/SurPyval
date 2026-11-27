"""Bayesian fits of the Wiener and gamma degradation processes
(``WienerProcess.fit_bayes``, ``GammaProcess.fit_bayes``)."""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import gammainc

import surpyval as sp
from surpyval.degradation import (
    GammaProcess,
    GammaProcessModel,
    WienerProcess,
    WienerProcessModel,
)
from surpyval.degradation.bayes import (
    BayesianProcessModel,
    ProcessLife,
    _gamma_partial_integral,
    _gamma_passage_integral,
    _wiener_partial_life,
    _wiener_sf,
)


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
    with pytest.raises(ValueError, match="only meaningful with gauge"):
        GammaProcess.fit_bayes(t, y, i, threshold=100, exact_start=True)
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


# ---------------------------------------------------------------------------
# Gauge readings: the quantised likelihood
# ---------------------------------------------------------------------------
def _gauge_readings(units=4, reads=7, seed=1, step=2.0):
    """Gamma wear (alpha 2, beta 4: a mean of 5 and a spread of about 1.1
    an interval of 10) read off a gauge of ``step``, and the exact
    readings."""
    t, y, i = _readings("gamma", units=units, reads=reads, seed=seed)
    return t, np.round(y / step) * step, i, y


GAUGES = [
    {"step": 2.0, "method": "independent"},
    {"step": 2.0, "method": "exact"},
    {"step": 2.0, "method": "exact", "exact_start": True},
    {"step": 2.0, "method": "exact", "rounding": "floor", "exact_start": True},
]


@pytest.mark.parametrize("gauge", GAUGES)
@pytest.mark.parametrize("stress", [False, True])
def test_the_gauge_likelihood_is_the_quantised_fits(gauge, stress):
    from surpyval.degradation.bayes import ProcessTarget
    from surpyval.degradation.process_models import (
        _increments,
        _quantised_log_likelihood,
    )

    t, y, i, _ = _gauge_readings()
    Z = np.where(i < 2, 1.0, 2.0) if stress else None
    target = ProcessTarget.from_readings(
        GammaProcess, t, y, i, 100.0, Z=Z, gauge=gauge
    )
    mle = target.mle
    # the target at the estimates is the likelihood the fit maximised
    assert np.isclose(target.neg_ll(target.start()), mle.neg_ll(), rtol=1e-10)
    rng = np.random.default_rng(4)
    theta = target.start()[None, :] * np.exp(
        rng.normal(0.0, 0.2, (7, target.start().size))
    )
    by_unit = target.unit_log_lik(theta)
    np.testing.assert_allclose(
        by_unit.sum(axis=1), target.log_lik(theta), rtol=1e-12
    )
    # one point at a time, through the scalar likelihood fit() uses
    m = GammaProcess._GAUGE_CELLS if gauge["method"] == "exact" else 1
    start = None
    if gauge.get("exact_start"):
        start = 1.0 if gauge.get("rounding", "nearest") == "nearest" else 0.0
    dt, dy = _increments(t, y, i)
    one = _quantised_log_likelihood(dy, i, 2.0, m, start)
    for row, ll in zip(theta, target.log_lik(theta)):
        dtau = dt if Z is None else dt * np.exp(target.z @ row[2:])
        assert np.isclose(one(row[0] * dtau, row[1]), ll, rtol=1e-12)


@pytest.fixture(scope="module")
def gauged():
    t, y, i, _ = _gauge_readings(units=5, reads=11)
    return GammaProcess.fit_bayes(t, y, i, threshold=100, gauge=2.0)


def test_flat_priors_peak_at_the_gauge_mle(gauged):
    # No priors is flat on log alpha and log beta, where the posterior is
    # the likelihood: its peak is the quantised maximum-likelihood fit.
    assert gauged.method == "grid"
    np.testing.assert_allclose(
        gauged.diagnostics["mode"], gauged.mle.params, rtol=1e-5
    )
    t, y, i, _ = _gauge_readings(units=5, reads=11)
    direct = GammaProcess.fit(t, y, i, 100, gauge=2.0)
    np.testing.assert_allclose(gauged.mle.params, direct.params, rtol=1e-12)


def test_a_coarse_gauge_ignored_is_biased_and_modelled_is_not(gauged):
    # Seeded: the true alpha is 2. Rounding to a step of 2 makes the
    # increments look more scattered (and some zero), so the fit that
    # takes the readings as exact puts alpha well below 2; the quantised
    # likelihood recovers it.
    t, y, i, _ = _gauge_readings(units=5, reads=11)
    naive = GammaProcess.fit_bayes(t, y, i, threshold=100)
    lo, hi = naive.param_cri("alpha")
    assert hi < 2.0
    lo, hi = gauged.param_cri("alpha")
    assert lo < 2.0 < hi
    lo, hi = gauged.param_cri("beta")
    assert lo < 4.0 < hi


@pytest.mark.parametrize("method", ["independent", "exact"])
def test_a_fine_gauge_is_the_exact_readings(method):
    # As the gauge step goes to zero the quantised posterior becomes the
    # posterior of the exact readings: the gap, in widths of the 95%
    # interval, shrinks with the step and is small at the finest.
    # Both sides get the same weak priors: with 15 increments and a gauge
    # of step 1, the independent approximation leaves a long tail in
    # alpha that a flat prior lets the grid chase.
    t, _, i, exact = _gauge_readings(units=3, reads=6)
    weak = {
        "alpha": sp.LogNormal.from_params([np.log(2.0), 1.0]),
        "beta": sp.LogNormal.from_params([np.log(4.0), 1.0]),
    }
    plain = GammaProcess.fit_bayes(t, exact, i, threshold=100, priors=weak)
    q = [0.05, 0.5, 0.95]
    gaps = []
    for step in (1.0, 0.1, 0.01):
        y = np.round(exact / step) * step
        fit = GammaProcess.fit_bayes(
            t,
            y,
            i,
            threshold=100,
            gauge=step,
            gauge_method=method,
            priors=weak,
        )
        gap = 0.0
        for name in ("alpha", "beta"):
            ref = plain.posterior.quantile(name, q)
            got = fit.posterior.quantile(name, q)
            gap = max(
                gap, float(np.max(np.abs(got - ref)) / (ref[2] - ref[0]))
            )
        gaps.append(gap)
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.02


def test_gauge_replicates_are_gauge_readings(gauged):
    rng = np.random.default_rng(0)
    target = gauged.target
    dy = target.simulate(target.start(), rng)
    assert dy.shape == target.observed().shape
    np.testing.assert_allclose(dy / 2.0, np.round(dy / 2.0), atol=1e-12)
    assert np.all(dy >= 0)
    check = gauged.ppc(random_state=3)
    assert 0 <= check.p_value <= 1
    assert "gauge of step 2" in target.describe_data()


def test_a_gauge_fit_is_saved_and_loaded(gauged):
    back = sp.from_dict(gauged.to_dict())
    assert back.target.gauge == gauged.target.gauge
    theta = gauged.posterior.points[:5]
    np.testing.assert_allclose(
        back.target.log_lik(theta), gauged.target.log_lik(theta), rtol=1e-12
    )
    np.testing.assert_allclose(back.sf([150.0]), gauged.sf([150.0]))


# ---------------------------------------------------------------------------
# Stress profiles (a StepSchedule as Z)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["wiener", "gamma"])
def stressed(request):
    kind = request.param
    t, y, i = _readings(kind, units=6)
    Z = np.where(i < 3, 1.0, 2.0)
    # the second group's clock runs exp(0.5) times faster
    y = np.where(i < 3, y, y * np.exp(0.5))
    fitter = WienerProcess if kind == "wiener" else GammaProcess
    return fitter.fit_bayes(t, y, i, threshold=100, Z=Z)


PROFILE = sp.StepSchedule.from_changepoints([0, 60, 120], [1.0, 2.0, 1.5])
X = np.array([0.0, 30.0, 60.0, 90.0, 150.0, 250.0, np.inf])


def test_a_profile_at_every_point_is_the_fitted_models(stressed):
    # The slow route: the fitted model at each posterior point, through
    # its own StressClock. The vectorised clocks give the same numbers.
    points = stressed.posterior.points
    sub = points[:: max(1, len(points) // 2000)]
    life = ProcessLife(stressed.target, sub, PROFILE, None)
    models = [stressed.target.with_params(row) for row in sub]
    slow = np.stack([m.sf(X, PROFILE) for m in models])
    np.testing.assert_allclose(life.sf(X), slow, rtol=1e-12, atol=1e-15)
    slow_ff = np.stack([m.ff(X, PROFILE) for m in models])
    np.testing.assert_allclose(life.ff(X), slow_ff, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(
        stressed.sf(X, PROFILE),
        stressed.posterior.expectation(
            ProcessLife(
                stressed.target, stressed.posterior.points, PROFILE, None
            ).sf(X)
        ),
        rtol=1e-14,
    )
    # B-lives, a few hundred points through the fitted model's qf
    sub = points[:: max(1, len(points) // 300)]
    life = ProcessLife(stressed.target, sub, PROFILE, None)
    for p in (0.01, 0.1, 0.5, 0.9):
        slow_q = [
            float(stressed.target.with_params(row).qf(p, PROFILE))
            for row in sub
        ]
        np.testing.assert_allclose(life.quantile(p), slow_q, rtol=1e-9)


def test_a_cyclic_profile(stressed):
    cyc = sp.StepSchedule.cyclic([0.0, 5.0], [[1.0], [2.0]], period=10.0)
    points = stressed.posterior.points
    sub = points[:: max(1, len(points) // 100)]
    life = ProcessLife(stressed.target, sub, cyc, None)
    x = np.array([3.0, 47.0, 151.0, 503.0])
    slow = np.stack(
        [stressed.target.with_params(row).sf(x, cyc) for row in sub]
    )
    np.testing.assert_allclose(life.sf(x), slow, rtol=1e-12, atol=1e-15)
    slow_q = [float(stressed.target.with_params(r).qf(0.5, cyc)) for r in sub]
    np.testing.assert_allclose(life.quantile(0.5), slow_q, rtol=1e-9)


def test_the_credible_band_under_a_profile(stressed):
    x = X[1:-1]
    band = stressed.cri(x, PROFILE)
    assert np.all(band[:, 0] <= band[:, 1])
    life = ProcessLife(
        stressed.target, stressed.posterior.points, PROFILE, None
    )
    np.testing.assert_allclose(
        band[:, 0], stressed.posterior.wquantile(life.sf(x), 0.025)
    )
    q = stressed.qf(0.1, PROFILE)
    assert np.isclose(stressed.ff(q, PROFILE), 0.1, atol=1e-9)
    lo, hi = stressed.quantile_cri(0.1, PROFILE)
    assert lo < stressed.quantile_cri(0.1, PROFILE, point=True) < hi
    rul = stressed.predict_rul(60.0, Z=PROFILE)
    assert np.isclose(rul.median, stressed.qf(0.5, PROFILE, y0=60.0))
    assert rul.interval[0] < rul.median < rul.interval[1]
    # more stress after 60 than the constant 1: a shorter life
    assert stressed.qf(0.5, PROFILE) < stressed.qf(0.5, [1.0])


def test_a_constant_profile_is_the_constant_stress(stressed):
    const = sp.StepSchedule.constant([2.0])
    x = X[1:]
    np.testing.assert_allclose(
        stressed.sf(x, const), stressed.sf(x, [2.0]), rtol=1e-13
    )
    np.testing.assert_allclose(
        stressed.cri(x, const), stressed.cri(x, [2.0]), rtol=1e-13
    )
    np.testing.assert_allclose(
        stressed.qf([0.1, 0.5], const),
        stressed.qf([0.1, 0.5], [2.0]),
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        stressed.quantile_cri(0.1, const),
        stressed.quantile_cri(0.1, [2.0]),
        rtol=1e-11,
    )
    a = stressed.predict_rul(40.0, Z=const)
    b = stressed.predict_rul(40.0, Z=[2.0])
    np.testing.assert_allclose(
        [a.median, *a.interval, *a.median_cri],
        [b.median, *b.interval, *b.median_cri],
        rtol=1e-11,
    )


def test_a_tight_posterior_is_the_fitted_model_under_a_profile(stressed):
    mle = stressed.mle
    # a posterior of one point, the estimates, is the fitted model
    life = ProcessLife(
        stressed.target, mle._estimates()[None, :], PROFILE, None
    )
    np.testing.assert_allclose(life.sf(X)[0], mle.sf(X, PROFILE), rtol=1e-12)
    for p in (0.1, 0.5):
        np.testing.assert_allclose(
            life.quantile(p)[0], mle.qf(p, PROFILE), rtol=1e-9
        )
    # and many readings narrow the posterior onto the fitted model: the
    # gap from the fit is a small part of the credible band
    kind = stressed.kind
    t, y, i = _readings(kind, units=60, seed=7)
    Z = np.where(i < 30, 1.0, 2.0)
    y = np.where(i < 30, y, y * np.exp(0.5))
    fitter = WienerProcess if kind == "wiener" else GammaProcess
    big = fitter.fit_bayes(t, y, i, threshold=100, Z=Z)
    x = np.array([90.0, 120.0, 150.0])
    band = big.cri(x, PROFILE)
    gap = np.abs(big.sf(x, PROFILE) - big.mle.sf(x, PROFILE))
    assert np.all(gap <= 0.1 * (band[:, 1] - band[:, 0]) + 1e-12)
    gap = abs(float(big.qf(0.5, PROFILE)) - float(big.mle.qf(0.5, PROFILE)))
    lo, hi = big.quantile_cri(0.5, PROFILE)
    assert gap < 0.1 * (hi - lo)


# The mean life under a profile. Rules fixed before the first run: the
# Wiener partial life and the gamma partial integral against adaptive
# quadrature of the survival (1e-9 relative); the mean at each posterior
# point against the fitted model's own mean through the profile (a
# quadrature of its sf, 1e-7 relative, the issue's tolerance), the
# fixed and the cyclic profile; a constant profile against the constant
# stress (1e-8: the gamma plain mean is tabulated to about 1e-9).
@pytest.mark.parametrize("mu", [0.5, 2.0, -0.3])
@pytest.mark.parametrize("sigma", [1.2, 0.4])
@pytest.mark.parametrize("u", [0.5, 5.0, 20.0, 80.0])
def test_the_wiener_partial_life_is_its_survival_integrated(mu, sigma, u):
    D = 10.0
    m, s = np.array([mu]), np.array([sigma])

    def sf(t):
        return float(_wiener_sf(np.array([t]), m, s, D)[0])

    # split where the survival falls, for the quadrature's sake
    cuts = [0.0] + [c for c in (D / abs(mu), 2 * D / abs(mu)) if c < u] + [u]
    ref = sum(
        quad(sf, a, b, epsabs=0, epsrel=1e-13, limit=500)[0]
        for a, b in zip(cuts[:-1], cuts[1:])
    )
    got = float(_wiener_partial_life(np.array([u]), m, s, D)[0])
    assert got == pytest.approx(ref, rel=1e-9)


def test_the_wiener_partial_life_runs_to_the_mean():
    D = 10.0
    m, s = np.array([0.5, 2.0, -0.3]), np.array([1.2, 0.4, 1.0])
    got = _wiener_partial_life(np.full(3, np.inf), m, s, D)
    np.testing.assert_allclose(got[:2], D / m[:2], rtol=1e-15)
    assert np.isposinf(got[2])
    # and far out, the partial life is the mean
    far = _wiener_partial_life(np.array([1e4, 1e4]), m[:2], s[:2], D)
    np.testing.assert_allclose(far, D / m[:2], rtol=1e-13)


@pytest.mark.parametrize("c", [0.5, 8.0, 200.0])
@pytest.mark.parametrize("share", [0.1, 1.0, 1.3, 50.0])
def test_the_gamma_partial_integral_is_its_survival_integrated(c, share):
    v = share * c
    cuts = sorted({0.0, max(c - 3 * np.sqrt(c), 0.0), c, v})
    cuts = [x for x in cuts if x <= v]
    ref = sum(
        quad(lambda t: gammainc(t, c), a, b, epsabs=0, epsrel=1e-13)[0]
        for a, b in zip(cuts[:-1], cuts[1:])
    )
    got = float(_gamma_partial_integral(np.array([c]), np.array([[v]]))[0, 0])
    assert got == pytest.approx(ref, rel=1e-9)
    whole = _gamma_partial_integral(np.array([c]), np.array([[np.inf]]))
    assert whole[0, 0] == pytest.approx(
        float(_gamma_passage_integral(np.array([c]))[0]), rel=1e-9
    )


CYCLIC = sp.StepSchedule.cyclic([0.0, 5.0], [[1.0], [2.0]], period=10.0)


@pytest.mark.parametrize("profile", ["steps", "cyclic"])
def test_the_mean_under_a_profile_is_the_fitted_models(stressed, profile):
    """At each posterior point the mean through the profile equals the
    fitted model's own mean there (a quadrature of its sf through its
    StressClock): what function_cri(lambda m: m.mean(profile)) reads,
    point by point."""
    prof = PROFILE if profile == "steps" else CYCLIC
    points = stressed.posterior.points
    # (the fitted model's mean through a cyclic profile is a quadrature
    # over hundreds of its steps, seconds a point: eight points for it)
    sub = points[:: max(1, len(points) // (40 if profile == "steps" else 8))]
    got = ProcessLife(stressed.target, sub, prof, None).mean()
    slow = np.array(
        [float(stressed.target.with_params(r).mean(prof)) for r in sub]
    )
    assert np.array_equal(np.isinf(got), np.isinf(slow))
    ok = np.isfinite(slow)
    np.testing.assert_allclose(got[ok], slow[ok], rtol=1e-7)
    if profile == "cyclic":
        # (the whole grid through a cyclic profile takes a minute or two
        # for the gamma process: the steps profile checks the summaries)
        return
    life = ProcessLife(stressed.target, points, prof, None).mean()
    assert stressed.mean(prof) == pytest.approx(
        float(stressed.posterior.wquantile(life, 0.5)), rel=1e-14
    )
    lo, hi = stressed.mean_cri(prof)
    assert lo < stressed.mean(prof) < hi


def test_a_cyclic_mean_lies_between_its_stresses(stressed):
    # half the time at 1, half at 2: at every point, strictly between
    # the constant-stress means
    points = stressed.posterior.points
    sub = points[:: max(1, len(points) // 500)]
    m = ProcessLife(stressed.target, sub, CYCLIC, None).mean()
    hi = ProcessLife(stressed.target, sub, [1.0], None).mean()
    lo = ProcessLife(stressed.target, sub, [2.0], None).mean()
    ok = np.isfinite(hi)
    assert np.all(np.isinf(m[~ok]))
    # (which stress is the faster depends on the sign of the point's
    # stress coefficient)
    moving = ok & (sub[:, -1] != 0)
    assert np.all(
        (np.minimum(lo, hi)[moving] < m[moving])
        & (m[moving] < np.maximum(lo, hi)[moving])
    )


def test_a_constant_profile_has_the_constant_stress_mean(stressed):
    const = sp.StepSchedule.constant([2.0])
    assert stressed.mean(const) == pytest.approx(
        stressed.mean([2.0]), rel=1e-8
    )
    np.testing.assert_allclose(
        stressed.mean_cri(const), stressed.mean_cri([2.0]), rtol=1e-8
    )


def test_a_wrong_profile_is_refused(stressed):
    with pytest.raises(ValueError, match="covariate"):
        stressed.sf(10.0, sp.StepSchedule.from_changepoints([0], [[1.0, 2.0]]))


def _calendar_mean(target, point, profile, horizon):
    """An independent mean life at one point: its calendar survival
    (through the clock, sf) integrated over each step of the profile, cut
    into pieces of at most 5, by a 20-point Gauss-Legendre rule, to
    ``horizon`` (past which the survival is below 1e-15)."""
    from surpyval.univariate.regression.tvc_schedule import (
        segments_from_origin,
    )

    starts, ends, _ = segments_from_origin(profile, horizon)
    cuts = [0.0]
    for a, b in zip(starts, ends):
        pieces = max(1, int(np.ceil((b - a) / 5.0)))
        cuts += list(np.linspace(a, b, pieces + 1)[1:])
    cuts = np.array(cuts)
    v, w = np.polynomial.legendre.leggauss(20)
    lo, hi = cuts[:-1], cuts[1:]
    t = (lo[:, None] + (hi - lo)[:, None] * 0.5 * (v + 1)).ravel()
    life = ProcessLife(target, point[None, :], profile, None)
    S = life.sf(t)[0].reshape(len(lo), 20)
    assert S[-1, -1] < 1e-15
    return float(np.sum((hi - lo) * 0.5 * (S @ w)))


IDLE = sp.StepSchedule.from_changepoints([0, 60, 70], [1.0, -30.0, 1.0])
IDLE_CYCLIC = sp.StepSchedule.cyclic([0.0, 5.0], [[1.0], [-30.0]], period=10.0)


@pytest.mark.parametrize("profile", ["fixed", "cyclic"])
def test_a_step_with_the_clock_all_but_stopped(stressed, profile):
    """A step at a stress so low that the clock barely moves (an idle or
    storage period): the step's calendar length counts in full. At the
    posterior point with the largest stress coefficient (the tail, where
    the clock is slowest there) the mean equals an independent
    integration of the calendar survival to 1e-9; the difference of the
    clock survival's integral over the step would lose the step (10 of
    1432 for the Wiener fixture, a 7e-3 error)."""
    prof = IDLE if profile == "fixed" else IDLE_CYCLIC
    points = stressed.posterior.points
    point = points[np.argmax(points[:, -1])]
    got = float(
        ProcessLife(stressed.target, point[None], prof, None).mean()[0]
    )
    horizon = 100.0 * float(stressed.target.with_params(point).mean([1.0]))
    ref = _calendar_mean(stressed.target, point, prof, horizon)
    assert got == pytest.approx(ref, rel=1e-9)


def test_the_wiener_partial_life_at_and_near_no_drift():
    D = 10.0
    s = np.array([1.2])
    for mu in (0.0, 1e-12, 1e-9, -1e-9, 1e-7, 1e-6, 1e-4):
        m = np.array([mu])

        def sf(t, m=m):
            return float(_wiener_sf(np.array([t]), m, s, D)[0])

        for u in (5.0, 80.0):
            ref = sum(
                quad(sf, a, b, epsabs=0, epsrel=1e-13, limit=500)[0]
                for a, b in ((0.0, 1.0), (1.0, u))
            )
            got = float(_wiener_partial_life(np.array([u]), m, s, D)[0])
            assert got == pytest.approx(ref, rel=1e-9), (mu, u)


def test_the_cyclic_summaries_over_the_whole_grid(stressed):
    """mean(CYCLIC) and mean_cri(CYCLIC) are the posterior's median and
    quantiles of the per-point means over every point of the grid (no
    subset: the per-point means are what the other tests check)."""
    if stressed.kind == "gamma":
        pytest.skip(
            "the gamma grid through a cyclic profile takes about two "
            "minutes; the Wiener grid checks the same summaries"
        )
    life = ProcessLife(
        stressed.target, stressed.posterior.points, CYCLIC, None
    ).mean()
    assert stressed.mean(CYCLIC) == pytest.approx(
        float(stressed.posterior.wquantile(life, 0.5)), rel=1e-14
    )
    lo, hi = stressed.mean_cri(CYCLIC)
    assert lo == pytest.approx(
        float(stressed.posterior.wquantile(life, 0.025)), rel=1e-14
    )
    assert hi == pytest.approx(
        float(stressed.posterior.wquantile(life, 0.975)), rel=1e-14
    )
    assert lo < stressed.mean(CYCLIC) < hi


IDLE_FIRST = sp.StepSchedule.cyclic([0.0, 5.0], [[-30.0], [1.0]], period=10.0)


@pytest.mark.parametrize("arrangement", ["idle second", "idle first"])
def test_a_cyclic_profile_with_idle_spells_at_points_across_the_posterior(
    stressed, arrangement
):
    """Half of each cycle at a stress that all but stops the clock,
    either half first, at four posterior points from the tail of the
    stress coefficient to its lower quartile: the mean equals a direct
    integration of the calendar survival (rule 1e-9). Where the horizon
    ends in an idle step, the clock survival left beyond it over that
    step's slope (about 1e-14) is not the calendar remainder: before the
    remainder was taken at the cycle's own slope, one gamma point was
    off by 1.5e-4."""
    prof = IDLE_CYCLIC if arrangement == "idle second" else IDLE_FIRST
    points = stressed.posterior.points
    order = np.argsort(points[:, -1])
    n = len(order)
    for j in (order[-1], order[3 * n // 4], order[n // 2], order[n // 4]):
        point = points[j]
        got = float(
            ProcessLife(stressed.target, point[None], prof, None).mean()[0]
        )
        mean1 = float(stressed.target.with_params(point).mean([1.0]))
        ref = _calendar_mean(stressed.target, point, prof, 100.0 * mean1)
        assert got == pytest.approx(ref, rel=1e-9)
