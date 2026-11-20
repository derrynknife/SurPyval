"""The shared-frailty Poisson process: ``fit(..., frailty=...)`` of the
intensity models and of the proportional-intensity regression.

The likelihood is checked against scipy's adaptive quadrature of each
cluster's frailty integral, its ``theta -> 0`` limit against the plain
fit, and the fit against simulated fleets. The simulations' designs,
seeds, sizes and tolerances were fixed before they were first run.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats
from scipy.special import gammaln

import surpyval
from surpyval.recurrent import (
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityNHPP,
    SharedFrailtyModel,
)
from surpyval.recurrent.shared_frailty import (
    frailty_moments,
    mixed_poisson_log_moment,
)

SEED = 468_113


def _log_density(w, theta, family):
    """The log-density of the frailty ``u = e^w``."""
    if family == "gamma":
        k = 1 / theta
        return (k - 1) * w - k * np.exp(w) + k * np.log(k) - gammaln(k)
    return -w - w**2 / (2 * theta) - 0.5 * np.log(2 * np.pi * theta)


def _quad_log_moment(D, A, theta, family):
    """``log E[u^D e^{-A u}]`` by scipy's adaptive quadrature in ``w = log
    u``, the integrand scaled by its peak. The gamma frailty's integrand
    falls only as ``e^{w / theta}`` to the left, so the range is long
    there."""

    def log_f(w):
        return D * w - A * np.exp(w) + _log_density(w, theta, family) + w

    grid = np.linspace(-60, 12, 30001)
    values = log_f(grid)
    peak = grid[np.argmax(values)]
    top = values.max()
    reach = 45 * max(1.0, theta)
    total = 0.0
    for low, high in ((peak - reach, peak), (peak, peak + 30)):
        value, _ = integrate.quad(
            lambda w: np.exp(log_f(w) - top),
            low,
            high,
            epsabs=0,
            epsrel=1e-13,
            limit=400,
        )
        total += value
    return top + np.log(total)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize(
    "D, A, theta",
    [
        (0, 0.4, 0.3),
        (1, 2.5, 0.05),
        (3, 2.5, 1.0),
        (7, 4.0, 0.5),
        (12, 30.0, 2.0),
        (40, 25.0, 0.2),
        (0, 12.0, 3.0),
    ],
)
def test_frailty_expectation_against_quadrature(D, A, theta, family):
    got = float(mixed_poisson_log_moment(D, A, theta, family))
    assert got == pytest.approx(
        _quad_log_moment(D, A, theta, family), abs=1e-10
    )


def test_gamma_expectation_is_the_gamma_function_form():
    D, A, theta = np.array([0, 2, 5, 9]), np.array([0.5, 1.0, 7.0, 3.0]), 0.7
    k = 1 / theta
    exact = (
        gammaln(D + k)
        - gammaln(k)
        + D * np.log(theta)
        - (D + k) * np.log1p(theta * A)
    )
    np.testing.assert_allclose(
        mixed_poisson_log_moment(D, A, theta), exact, rtol=0, atol=1e-12
    )


def test_expectation_without_frailty_is_poisson():
    A = np.array([0.5, 3.0])
    np.testing.assert_array_equal(mixed_poisson_log_moment(4, A, 0.0), -A)
    # ... and is approached: the first-order term is theta/2 ((D-A)^2 - D)
    for family in ("gamma", "lognormal"):
        tiny = mixed_poisson_log_moment(4, A, 1e-7, family)
        first = (4 - A) ** 2 - (4 if family == "gamma" else A)
        np.testing.assert_allclose(tiny + A, 0.5e-7 * first, atol=1e-12)


def _simulate(rng, units, theta, family, coef=0.0, alpha=20.0, beta=1.3):
    """A fleet from the definition: each unit's frailty, then a Poisson
    count over (entry, end] and the times given the count."""
    x, i, c, Z, tl = [], [], [], [], []
    for unit in range(units):
        if family == "gamma":
            u = rng.gamma(1 / theta, theta)
        else:
            u = np.exp(rng.normal(0.0, np.sqrt(theta)))
        z = float(unit % 2)
        entry = 0.0 if unit % 3 else 15.0
        end = 80.0 + 40.0 * (unit % 4) / 3
        low, high = (entry / alpha) ** beta, (end / alpha) ** beta
        n = rng.poisson(u * np.exp(coef * z) * (high - low))
        times = alpha * np.sort(rng.uniform(low, high, n)) ** (1 / beta)
        x += [*times, end]
        i += [unit] * (n + 1)
        c += [0] * n + [1]
        Z += [z] * (n + 1)
        tl += [entry] * (n + 1)
    return {
        "x": np.array(x),
        "i": np.array(i),
        "c": np.array(c),
        "tl": np.array(tl),
    }, np.array(Z)[:, None]


@pytest.fixture(scope="module")
def fleet():
    rows, Z = _simulate(np.random.default_rng([SEED, 0]), 40, 0.6, "gamma")
    return rows, Z


def _independent_log_likelihood(model, rows, Z, cluster=None):
    """The marginal log-likelihood from the model's own functions and
    scipy's quadrature of each cluster's frailty integral."""
    x, i, c, tl = rows["x"], rows["i"], rows["c"], rows["tl"]
    regression = model.n_covariates > 0
    total = 0.0
    groups = {}
    for unit in np.unique(i):
        own = i == unit
        kw = {"frailty": 1.0}
        if regression:
            kw["Z"] = Z[own][0]
        events = x[own][c[own] == 0]
        total += np.sum(np.log(model.iif(events, **kw)))
        span = float(
            model.cif(x[own].max(), **kw) - model.cif(tl[own][0], **kw)
        )
        key = unit if cluster is None else cluster[unit]
        n, L = groups.get(key, (0, 0.0))
        groups[key] = (n + events.size, L + span)
    for n, L in groups.values():
        total += _quad_log_moment(n, L, model.theta, model.family)
    return total


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("fitter", [CrowAMSAA, Duane, CoxLewis])
def test_log_likelihood_against_quadrature(fleet, fitter, family):
    rows, Z = fleet
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fitter.fit(**rows, frailty=family)
    assert isinstance(model, SharedFrailtyModel)
    assert model.maximum == "verified"
    assert model.log_likelihood == pytest.approx(
        _independent_log_likelihood(model, rows, Z), abs=1e-8
    )


def test_regression_and_cluster_log_likelihood_against_quadrature():
    rows, Z = _simulate(
        np.random.default_rng([SEED, 1]), 40, 0.6, "gamma", coef=0.5
    )
    site = {unit: f"site {unit // 4}" for unit in range(40)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ProportionalIntensityNHPP.fit(
            rows["x"],
            Z,
            rows["i"],
            rows["c"],
            tl=rows["tl"],
            baseline=CrowAMSAA,
            frailty="gamma",
            cluster=site,
        )
    assert model.parameter_names == ["alpha", "beta", "theta", "coef_0"]
    assert len(model.cluster_ids) == 10
    assert model.log_likelihood == pytest.approx(
        _independent_log_likelihood(model, rows, Z, site), abs=1e-8
    )
    # one label per row is the mapping
    per_row = np.array([site[unit] for unit in rows["i"]])
    again = ProportionalIntensityNHPP.fit(
        rows["x"],
        Z,
        rows["i"],
        rows["c"],
        tl=rows["tl"],
        baseline=CrowAMSAA,
        frailty="gamma",
        cluster=per_row,
    )
    np.testing.assert_allclose(again.params, model.params, rtol=1e-10)


def test_likelihood_approaches_the_plain_one_as_theta_goes_to_zero(fleet):
    rows, _ = fleet
    plain = CrowAMSAA.fit(**rows)
    for family in ("gamma", "lognormal"):
        model = CrowAMSAA.fit(**rows, frailty=family)
        at = np.r_[plain.params, 0.0]
        assert -model._neg_ll(at) == pytest.approx(
            plain.log_likelihood, abs=1e-9
        )
        # The plain fit is a stationary point in the intensity; the score
        # in theta there is the first-order term.
        near = np.r_[plain.params, 1e-8]
        gap = -model._neg_ll(near) - plain.log_likelihood
        assert abs(gap) < 1e-4
        wider = -model._neg_ll(np.r_[plain.params, 1e-6])
        assert gap == pytest.approx(
            1e-2 * (wider - plain.log_likelihood), rel=1e-3
        )


def _regular_fleet():
    """Units whose counts are all equal: less spread than Poisson."""
    x, i, c = [], [], []
    for unit in range(6):
        x += [10 + unit, 25 + unit, 40 - unit, 55 + unit, 60]
        i += [unit] * 5
        c += [0, 0, 0, 0, 1]
    return np.array(x, dtype=float), np.array(i), np.array(c)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
@pytest.mark.parametrize("fitter", [CrowAMSAA, CoxLewis])
def test_theta_on_its_boundary_is_the_plain_fit(fitter, family):
    x, i, c = _regular_fleet()
    plain = fitter.fit(x, i, c)
    with pytest.warns(UserWarning, match="theta is estimated at its boundary"):
        model = fitter.fit(x, i, c, frailty=family)
    assert model.theta == 0.0
    assert model.maximum == "verified"
    # The same maximum: the log-likelihoods agree to 1e-8. The plain
    # fit's simplex search stops within about 1e-5 of it in the
    # parameters (its gradient there is 2e-2 for Cox-Lewis, against 2e-9
    # at this fit's), so the parameters are compared to 1e-4.
    assert model.log_likelihood == pytest.approx(
        plain.log_likelihood, abs=1e-8
    )
    np.testing.assert_allclose(model.params[:2], plain.params, rtol=1e-4)
    np.testing.assert_allclose(
        model.cif([10.0, 50.0]), plain.cif([10.0, 50.0]), rtol=1e-4
    )
    test = model.frailty_test()
    assert test["statistic"].iloc[0] == 0.0 and test["p"].iloc[0] == 1.0
    frail = model.frailties()
    assert np.all(frail["frailty"] == 1.0)
    assert np.all(frail.iloc[:, 3:] == 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(model.standard_errors()[2])
        assert np.all(np.isnan(model.param_cb("theta")))
    assert np.isfinite(model.param_cb("beta")).all()
    # theta is not counted as a parameter of the fit
    assert model.aic() == pytest.approx(plain.aic(), abs=1e-7)
    # ... and the forecast is the Poisson one
    ahead = model.forecast(20.0)
    poisson = surpyval.forecast(plain, horizon=20.0)
    np.testing.assert_allclose(ahead.expected, poisson.expected, rtol=1e-4)
    np.testing.assert_array_equal(ahead.lower, poisson.lower)
    np.testing.assert_array_equal(ahead.upper, poisson.upper)


# -- recovery ---------------------------------------------------------------
# 20 fleets of 100 units (gamma) and 10 of 100 (log-normal), the truth
# alpha = 20, beta = 1.3, theta = 0.5, coefficient 0.4. The mean estimate
# must be within 10% of alpha, 5% of beta, 15% of theta and 0.1 of the
# coefficient.
@pytest.mark.parametrize("family, fleets", [("gamma", 20), ("lognormal", 10)])
def test_recovery_over_simulated_fleets(family, fleets):
    estimates = []
    for r in range(fleets):
        rng = np.random.default_rng([SEED, 7, r, len(family)])
        rows, Z = _simulate(rng, 100, 0.5, family, coef=0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = ProportionalIntensityNHPP.fit(
                rows["x"],
                Z,
                rows["i"],
                rows["c"],
                tl=rows["tl"],
                baseline=CrowAMSAA,
                frailty=family,
            )
        assert model.maximum == "verified"
        estimates.append(model.params)
    mean = np.mean(estimates, axis=0)
    assert mean[0] == pytest.approx(20.0, rel=0.10)
    assert mean[1] == pytest.approx(1.3, rel=0.05)
    assert mean[2] == pytest.approx(0.5, rel=0.15)
    assert mean[3] == pytest.approx(0.4, abs=0.1)


def test_ignoring_the_frailty_understates_the_errors(fleet):
    rows, _ = fleet
    plain = CrowAMSAA.fit(**rows)
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    assert model.theta > 0.2
    assert model.standard_errors()[0] > plain.standard_errors()[0]
    assert model.aic() < plain.aic()
    assert model.frailty_test()["p"].iloc[0] < 1e-6
    lo, hi = model.cif_cb(100.0)
    plo, phi = plain.cif_cb(100.0)
    assert hi - lo > phi - plo


# -- frailties and the forecast -------------------------------------------
def test_frailties_are_the_gamma_posterior(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    table = model.frailties(alpha_ci=0.1)
    n, L = table["events"].to_numpy(), table["expected"].to_numpy()
    k = 1 / model.theta
    np.testing.assert_allclose(table["frailty"], (n + k) / (L + k))
    np.testing.assert_allclose(
        table["lower 90%"], stats.gamma.ppf(0.05, n + k, scale=1 / (L + k))
    )
    np.testing.assert_allclose(
        table["upper 90%"], stats.gamma.ppf(0.95, n + k, scale=1 / (L + k))
    )
    assert list(table.index) == sorted(np.unique(rows["i"]).tolist())
    assert n.sum() == np.sum(rows["c"] == 0)
    # each cluster's expected count at frailty 1, from the model
    unit = table.index[3]
    own = rows["i"] == unit
    span = model.cif(rows["x"][own].max(), frailty=1.0) - model.cif(
        rows["tl"][own][0], frailty=1.0
    )
    assert L[3] == pytest.approx(float(span), rel=1e-10)


def test_lognormal_frailties_against_quadrature(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="lognormal")
    table = model.frailties()
    theta = model.theta
    for row in (0, 5, 17):
        n, L = table["events"].iloc[row], table["expected"].iloc[row]
        mean = np.exp(
            _quad_log_moment(n + 1, L, theta, "lognormal")
            - _quad_log_moment(n, L, theta, "lognormal")
        )
        assert table["frailty"].iloc[row] == pytest.approx(mean, rel=1e-8)
        # the interval's ends hold 2.5% and 97.5% of the posterior
        norm = np.exp(_quad_log_moment(n, L, theta, "lognormal"))

        def below(u):
            value, _ = integrate.quad(
                lambda v: v**n
                * np.exp(-L * v)
                * stats.lognorm.pdf(v, np.sqrt(theta)),
                0,
                u,
                epsabs=0,
                epsrel=1e-11,
                limit=400,
            )
            return value / norm

        assert below(table["lower 95%"].iloc[row]) == pytest.approx(
            0.025, abs=1e-5
        )
        assert below(table["upper 95%"].iloc[row]) == pytest.approx(
            0.975, abs=1e-5
        )
    mean, variance = frailty_moments([0.0], [0.0], theta, "lognormal")
    assert mean[0] == pytest.approx(np.exp(theta / 2), rel=1e-9)
    assert variance[0] == pytest.approx(
        np.expm1(theta) * np.exp(theta), rel=1e-8
    )
    assert model.mean_frailty == pytest.approx(np.exp(theta / 2))
    assert float(model.cif(50.0)) == pytest.approx(
        np.exp(theta / 2) * float(model.cif(50.0, frailty=1.0))
    )


def test_forecast_of_a_new_unit_is_negative_binomial(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    ahead = model.forecast([20.0, 50.0], age=[100.0], alpha_ci=0.1)
    mu = np.array(
        [float(model.cif(100.0 + h) - model.cif(100.0)) for h in (20.0, 50.0)]
    )
    k = 1 / model.theta
    np.testing.assert_allclose(ahead.expected, mu, rtol=1e-12)
    np.testing.assert_allclose(
        ahead.variance, mu + model.theta * mu**2, rtol=1e-12
    )
    np.testing.assert_array_equal(
        ahead.lower, stats.nbinom.ppf(0.05, k, k / (k + mu))
    )
    np.testing.assert_array_equal(
        ahead.upper, stats.nbinom.ppf(0.95, k, k / (k + mu))
    )
    np.testing.assert_allclose(
        ahead.probability[0], 1 - stats.nbinom.pmf(0, k, k / (k + mu))
    )
    # wider than the Poisson interval of the same mean
    assert ahead.upper[1] > stats.poisson.ppf(0.95, mu[1])
    # Two new units that share a frailty: one cluster, twice the mean
    pair = model.forecast(50.0, age=[100.0, 100.0], cluster=["a", "a"])
    np.testing.assert_allclose(
        pair.variance, 2 * mu[1] + model.theta * (2 * mu[1]) ** 2
    )
    np.testing.assert_array_equal(
        pair.upper, stats.nbinom.ppf(0.975, k, k / (k + 2 * mu[1]))
    )
    # ... and two that do not: the sum of two independent counts
    apart = model.forecast(50.0, age=[100.0, 100.0])
    np.testing.assert_allclose(
        apart.variance, 2 * (mu[1] + model.theta * mu[1] ** 2)
    )
    np.testing.assert_array_equal(
        apart.upper, stats.nbinom.ppf(0.975, 2 * k, k / (k + mu[1]))
    )
    assert apart.upper[0] < pair.upper[0]


def test_forecast_of_the_fitted_units_uses_their_histories(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    ahead = model.forecast(30.0)
    table = model.frailties()
    ends = np.array([rows["x"][rows["i"] == u].max() for u in table.index])
    mu = np.asarray(
        model.cif(ends + 30.0, frailty=1.0) - model.cif(ends, frailty=1.0)
    )
    np.testing.assert_array_equal(ahead.units, table.index.to_numpy())
    np.testing.assert_allclose(
        ahead.per_unit[:, 0], mu * table["frailty"].to_numpy(), rtol=1e-10
    )
    k = 1 / model.theta
    shape = table["events"].to_numpy() + k
    rate = table["expected"].to_numpy() + k
    variance = np.sum(mu * shape / rate + mu**2 * shape / rate**2)
    assert ahead.variance[0] == pytest.approx(variance, rel=1e-10)
    # One fitted cluster alone, by its label: its own negative binomial
    one = model.forecast(30.0, age=[ends[3]], cluster=[table.index[3]])
    np.testing.assert_array_equal(
        one.upper,
        stats.nbinom.ppf(0.975, shape[3], rate[3] / (rate[3] + mu[3])),
    )
    # The fleet's interval against a simulation of the clusters' counts
    rng = np.random.default_rng([SEED, 3])
    draws = rng.poisson(
        rng.gamma(shape, 1 / rate, size=(200_000, shape.size)) * mu
    ).sum(axis=1)
    assert ahead.expected[0] == pytest.approx(draws.mean(), rel=0.01)
    assert abs(ahead.lower[0] - np.quantile(draws, 0.025)) <= 1
    assert abs(ahead.upper[0] - np.quantile(draws, 0.975)) <= 1


def test_lognormal_forecast_against_simulation(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="lognormal")
    ahead = model.forecast(40.0, age=[90.0, 60.0], cluster=["new", "new"])
    mu = float(
        np.sum(
            model.cif([130.0, 100.0], frailty=1.0)
            - model.cif([90.0, 60.0], frailty=1.0)
        )
    )
    rng = np.random.default_rng([SEED, 4])
    u = np.exp(rng.normal(0, np.sqrt(model.theta), 400_000))
    draws = rng.poisson(u * mu)
    assert ahead.expected[0] == pytest.approx(mu * np.exp(model.theta / 2))
    assert ahead.variance[0] == pytest.approx(draws.var(), rel=0.03)
    assert abs(ahead.lower[0] - np.quantile(draws, 0.025)) <= 1
    assert abs(ahead.upper[0] - np.quantile(draws, 0.975)) <= 1


# -- entry points, saving ---------------------------------------------------
def test_entry_points_agree(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    data = surpyval.handle_xicn(rows["x"], rows["i"], rows["c"], tl=rows["tl"])
    for other in (
        CrowAMSAA.fit(data, frailty="gamma"),
        CrowAMSAA.fit_from_recurrent_data(data, frailty="gamma"),
        CrowAMSAA.fit_from_df(
            pd.DataFrame(rows), "x", "i", "c", tl_col="tl", frailty="gamma"
        ),
    ):
        np.testing.assert_allclose(other.params, model.params, rtol=1e-9)
    # row order does not matter
    order = np.random.default_rng([SEED, 5]).permutation(rows["x"].size)
    by_unit = np.argsort(rows["i"][order], kind="stable")
    shuffled = {k: v[order][by_unit] for k, v in rows.items()}
    again = CrowAMSAA.fit(**shuffled, frailty="gamma")
    np.testing.assert_allclose(again.params, model.params, rtol=1e-8)


def test_observation_windows_share_the_items_frailty():
    events = {
        1: [5.0, 12.0, 31.0, 44.0],
        2: [8.0, 38.0],
        3: [3.0, 9.0, 14.0, 33.0, 41.0, 47.0],
    }
    windows = {k: [(0.0, 20.0), (30.0, 50.0)] for k in events}
    x = np.concatenate(list(events.values()))
    i = np.concatenate([[k] * len(v) for k, v in events.items()])
    with warnings.catch_warnings():
        # (three items' counts: theta is on its boundary here)
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(x, i, windows=windows, frailty="gamma")
    assert model.cluster_ids == [1, 2, 3]
    assert model.frailties()["events"].tolist() == [4.0, 2.0, 6.0]
    span = float(
        model.cif(50.0, frailty=1.0)
        - model.cif(30.0, frailty=1.0)
        + model.cif(20.0, frailty=1.0)
    )
    np.testing.assert_allclose(model.frailties()["expected"], span)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_round_trips(fleet, family):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty=family)
    for again in (
        SharedFrailtyModel.from_dict(model.to_dict()),
        surpyval.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert isinstance(again, SharedFrailtyModel)
        np.testing.assert_array_equal(again.params, model.params)
        np.testing.assert_allclose(
            again.cif_cb([10.0, 50.0]), model.cif_cb([10.0, 50.0]), rtol=1e-9
        )
        pd.testing.assert_frame_equal(again.frailties(), model.frailties())
        np.testing.assert_allclose(
            again.param_cb("theta"), model.param_cb("theta"), rtol=1e-9
        )
        ahead = again.forecast(10.0, age=[50.0])
        np.testing.assert_array_equal(
            ahead.upper, model.forecast(10.0, age=[50.0]).upper
        )
    restored = SharedFrailtyModel.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="use method='wald'"):
        restored.param_cb("theta", method="lr")
    with pytest.raises(ValueError, match="age is required"):
        restored.forecast(10.0)


def test_profile_likelihood_interval_holds_the_estimate(fleet):
    rows, _ = fleet
    model = CrowAMSAA.fit(**rows, frailty="gamma")
    lo, hi = model.param_cb("theta", method="lr")
    assert lo < model.theta < hi
    # the deviance at each end is the chi-square critical value
    from scipy.optimize import minimize

    for end in (lo, hi):
        res = minimize(
            lambda q: model._neg_ll(np.r_[np.exp(q), end]),
            np.log(model.params[:2]),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12},
        )
        deviance = 2 * (res.fun + model.log_likelihood)
        assert deviance == pytest.approx(stats.chi2.ppf(0.95, 1), abs=2e-3)


# -- refusals -----------------------------------------------------------------
def test_refusals_say_what_to_do(fleet):
    rows, Z = fleet
    x, i, c = rows["x"], rows["i"], rows["c"]
    with pytest.raises(ValueError, match="PanelCountRegression"):
        CrowAMSAA.fit(
            [[0, 10], [10, 20], [0, 10], [10, 20]],
            i=[1, 1, 2, 2],
            c=[2, 2, 2, 2],
            n=[3, 1, 2, 2],
            frailty="gamma",
        )
    with pytest.raises(ValueError, match="right truncation"):
        CrowAMSAA.fit(x, i, c, tr=200.0, frailty="gamma")
    with pytest.raises(ValueError, match="at least 2 clusters"):
        CrowAMSAA.fit([1.0, 5.0, 9.0, 12.0], frailty="gamma")
    with pytest.raises(ValueError, match="at least 2 clusters"):
        CrowAMSAA.fit(
            x, i, c, frailty="gamma", cluster={u: "all" for u in np.unique(i)}
        )
    with pytest.raises(ValueError, match="how='MSE' has no"):
        CrowAMSAA.fit(x, i, c, how="MSE", frailty="gamma")
    with pytest.raises(ValueError, match="'frailty' must be one of"):
        CrowAMSAA.fit(x, i, c, frailty="weibull")
    with pytest.raises(ValueError, match="needs 'frailty'"):
        CrowAMSAA.fit(x, i, c, cluster={u: 1 for u in np.unique(i)})
    with pytest.raises(ValueError, match="needs 'frailty'"):
        ProportionalIntensityNHPP.fit(
            x, Z, i, c, cluster={u: 1 for u in np.unique(i)}
        )
    with pytest.raises(ValueError, match="names no cluster for 39 of the 40"):
        CrowAMSAA.fit(x, i, c, frailty="gamma", cluster={0: "a"})
    changing = np.arange(x.size) % 2
    with pytest.raises(ValueError, match="changes within item"):
        CrowAMSAA.fit(x, i, c, frailty="gamma", cluster=changing)
    with pytest.raises(ValueError, match="one label per row"):
        CrowAMSAA.fit(x, i, c, frailty="gamma", cluster=[1, 2, 3])
    with pytest.raises(ValueError, match="'init' must have 3 values"):
        CrowAMSAA.fit(x, i, c, frailty="gamma", init=[20.0, 1.3])
    model = CrowAMSAA.fit(x, i, c, frailty="gamma", init=[15.0, 1.0, 1.0])
    assert model.maximum == "verified"
    with pytest.raises(ValueError, match="'Z' is not taken"):
        model.cif(10.0, Z=[1.0])
    with pytest.raises(ValueError, match="n is not taken"):
        surpyval.forecast(model, age=[10.0], horizon=5.0, n=[3])
    with pytest.raises(ValueError, match="this model has no frailty"):
        surpyval.forecast(
            CrowAMSAA.fit(x, i, c), age=[10.0], horizon=5.0, cluster=[1]
        )
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.param_cb("gamma")
