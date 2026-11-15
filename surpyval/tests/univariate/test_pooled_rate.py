"""``pooled_rate``: the two-stage (hierarchical) pooling of event counts
from several sources.

References: hand-worked values; the BUGS "Pumps" example (Gaver and
O'Muircheartaigh 1987; Gelfand and Smith 1990), whose posterior means of
the population's parameters are about 0.70 and 0.93, and whose per-pump
means are checked against an independent numerical integration here; the
negative binomial and beta-binomial MLEs of scipy.stats; and a six-source
worked example of failures to start per demand (inputs below).
"""

import json
import warnings

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from surpyval import PooledRate, pooled_rate
from surpyval.univariate import pooled_rate as pr_module

# The BUGS "Pumps" data: failures and operating time (1000s of hours).
PUMP_X = np.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22], float)
PUMP_T = np.array(
    [94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.048, 1.048, 2.096, 10.48]
)
# Six sources of failures to start: failures and demands.
FTS_X = [227, 137, 25, 3, 42, 6]
FTS_N = [410593, 49000, 2100, 260, 3700, 2400]


def _quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pooled_rate(*args, **kwargs)


# --- worked and published examples -----------------------------------------


@pytest.mark.parametrize("kind", ["probability", "rate"])
def test_worked_example_six_sources(kind):
    # Median about 0.0046, 90% range about 0.00017 to 0.026 for a new
    # source, the average about 0.0066 (0.0033 to 0.017); the one-pot
    # estimate 440 / 468,053. Below 0.012 per demand the binomial and
    # Poisson forms agree to about 1%.
    res = pooled_rate(FTS_X, FTS_N, kind, alpha_ci=0.1)
    p, a = res.predictive, res.average
    tol = 0.01 if kind == "rate" else 0.02
    assert p["median"] == pytest.approx(0.00457, rel=tol)
    assert p["lower"] == pytest.approx(0.000169, rel=0.03)
    assert p["upper"] == pytest.approx(0.0263, rel=tol)
    assert p["mean"] == pytest.approx(0.0083, rel=0.03)
    assert a["estimate"] == pytest.approx(0.0066, rel=0.02)
    assert a["lower"] == pytest.approx(0.0034, rel=0.03)
    assert a["upper"] == pytest.approx(0.0166, rel=0.03)
    assert res.common["estimate"] == pytest.approx(440 / 468053, rel=1e-12)
    # the stored one-distribution summary: shape 0.71, rate (or b) 92
    pp = res.population_params
    shape = pp["shape"] if kind == "rate" else pp["a"]
    scale = pp["rate"] if kind == "rate" else pp["b"]
    assert shape == pytest.approx(0.71, abs=0.01)
    assert scale == pytest.approx(92, rel=0.01)
    # it has the predictive median and upper quantile
    pop = res.population
    assert float(pop.qf(0.5)) == pytest.approx(p["median"], rel=1e-6)
    assert float(pop.qf(0.95)) == pytest.approx(p["upper"], rel=1e-6)


def _pumps_prior(fam, loc, u):
    # BUGS: alpha ~ Exp(1), beta ~ Gamma(0.1, 1), written on (log mean,
    # log alpha): the Jacobian of (alpha, beta) is alpha * beta.
    a = np.exp(u)
    b = a / np.exp(loc)
    return -a + 0.1 * np.log(b) - b + u


def _pumps_by_quadrature(f):
    """E[f(alpha, beta)] under the Pumps posterior, by 2-d quadrature in
    (log alpha, log beta)."""

    def post(lb, la):
        a, b = np.exp(la), np.exp(lb)
        ll = np.sum(stats.nbinom.logpmf(PUMP_X, a, b / (b + PUMP_T)))
        return np.exp(ll - a - 0.9 * np.log(b) - b + la + lb + 25.0)

    lims = (np.log(0.02), np.log(20.0), np.log(0.01), np.log(30.0))
    z = integrate.dblquad(post, *lims, epsrel=1e-9)[0]
    num = integrate.dblquad(
        lambda lb, la: post(lb, la) * f(np.exp(la), np.exp(lb)),
        *lims,
        epsrel=1e-9,
    )[0]
    return num / z


def test_pumps_posterior_matches_published_and_quadrature():
    fam = pr_module._Poisson
    L, U, w = pr_module._bayes_grid(fam, PUMP_X, PUMP_T, _pumps_prior)
    a, b = fam.params(L, U)
    # published (BUGS, MCMC): alpha 0.70, beta 0.93
    assert np.sum(w * a) == pytest.approx(0.70, abs=0.01)
    assert np.sum(w * b) == pytest.approx(0.93, abs=0.01)
    assert np.sum(w * a) == pytest.approx(
        _pumps_by_quadrature(lambda a, b: a), rel=2e-3
    )
    for i in (0, 4, 9):
        grid = np.sum(w * (a + PUMP_X[i]) / (b + PUMP_T[i]))
        quad = _pumps_by_quadrature(
            lambda a, b, i=i: (a + PUMP_X[i]) / (b + PUMP_T[i])
        )
        assert grid == pytest.approx(quad, rel=2e-3)


def test_mle_matches_scipy_negative_binomial():
    res = pooled_rate(PUMP_X, PUMP_T, method="MLE", random_state=1)

    def nll(p):
        a, m = np.exp(p)
        return -np.sum(stats.nbinom.logpmf(PUMP_X, a, a / (a + m * PUMP_T)))

    fit = optimize.minimize(
        nll,
        [0.0, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12},
    )
    a_hat, m_hat = np.exp(fit.x)
    pp = res.population_params
    assert pp["shape"] == pytest.approx(a_hat, rel=1e-4)
    assert pp["mean"] == pytest.approx(m_hat, rel=1e-4)
    assert res.average["estimate"] == pytest.approx(m_hat, rel=1e-4)
    # the profile interval: the deviance is chi2(0.95, 1) at each end
    best = -fit.fun
    for end in (res.average["lower"], res.average["upper"]):
        prof = -optimize.minimize_scalar(
            lambda la: nll([la, np.log(end)]),
            bounds=(-9, 18),
            method="bounded",
            options={"xatol": 1e-9},
        ).fun
        assert 2 * (best - prof) == pytest.approx(
            stats.chi2.ppf(0.95, 1), abs=1e-4
        )
    # the empirical-Bayes shrunk estimate is the plug-in posterior mean
    s = res.sources
    np.testing.assert_allclose(
        s["shrunk"], (a_hat + PUMP_X) / (a_hat / m_hat + PUMP_T), rtol=1e-4
    )


def test_mle_matches_scipy_beta_binomial():
    x = np.array([3, 0, 7, 2, 12, 1], float)
    n = np.array([200, 150, 300, 90, 400, 120], float)
    res = pooled_rate(x, n, "probability", method="MLE", random_state=1)

    def nll(p):
        a, b = np.exp(p)
        return -np.sum(stats.betabinom.logpmf(x, n, a, b))

    fit = optimize.minimize(
        nll,
        [0.0, 3.0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
    )
    a_hat, b_hat = np.exp(fit.x)
    pp = res.population_params
    assert pp["a"] == pytest.approx(a_hat, rel=1e-3)
    assert pp["b"] == pytest.approx(b_hat, rel=1e-3)
    assert pp["mean"] == pytest.approx(a_hat / (a_hat + b_hat), rel=1e-4)


# --- hand-worked ------------------------------------------------------------


def test_heterogeneity_chi_square_by_hand():
    # one rate 10 / 200 = 0.05; expected 5 and 5; (3^2 + 3^2) / 5 = 3.6
    res = _quiet([2, 8], [100, 100])
    h = res.heterogeneity
    assert h["chi2"] == pytest.approx(3.6, abs=1e-12)
    assert h["df"] == 1
    assert h["p_value"] == pytest.approx(stats.chi2.sf(3.6, 1), rel=1e-12)
    assert h["I2"] == pytest.approx((3.6 - 1) / 3.6, rel=1e-12)
    # binomial: the 2 x 2 table [[2, 98], [8, 92]]
    resb = _quiet([2, 8], [100, 100], "probability")
    table = np.array([[2, 98], [8, 92]])
    expected = stats.chi2_contingency(table, correction=False)[0]
    assert resb.heterogeneity["chi2"] == pytest.approx(expected, rel=1e-12)


def test_one_source_is_its_own_jeffreys_interval():
    with pytest.warns(UserWarning, match=r"1 source given.*at least 4"):
        res = pooled_rate([3], [1000.0], alpha_ci=0.1)
    assert not res.pooled
    assert res.population is None and res.predictive is None
    s = res.sources.iloc[0]
    lo, hi = stats.gamma.ppf([0.05, 0.95], 3.5, scale=1 / 1000.0)
    assert s["own"] == 0.003
    assert s["own_lower"] == pytest.approx(lo, rel=1e-12)
    assert s["own_upper"] == pytest.approx(hi, rel=1e-12)
    assert res.heterogeneity["chi2"] is None
    with pytest.raises(ValueError, match="no population"):
        res.update(1, 100)
    # a probability: Beta(x + 1/2, n - x + 1/2)
    resb = _quiet([0], [50], "probability")
    lo, hi = stats.beta.ppf([0.025, 0.975], 0.5, 50.5)
    assert resb.sources.iloc[0]["own_lower"] == pytest.approx(lo, rel=1e-10)
    assert resb.sources.iloc[0]["own_upper"] == pytest.approx(hi, rel=1e-10)


def test_fewer_than_min_sources_are_shown_side_by_side():
    with pytest.warns(UserWarning, match="3 sources given"):
        res = pooled_rate([463, 83, 149], [3510.0, 672.1, 235.4])
    assert not res.pooled
    assert res.average is None and res.population_params is None
    assert res.heterogeneity["p_value"] < 1e-10
    assert np.all(np.isnan(res.sources["shrunk"]))
    assert "No population" in repr(res)
    # the threshold is the caller's
    assert pooled_rate(
        [463, 83, 149], [3510.0, 672.1, 235.4], min_sources=3
    ).pooled


def test_no_events_anywhere_is_refused_with_own_intervals():
    with pytest.warns(UserWarning, match="No source had an event"):
        res = pooled_rate([0, 0, 0, 0], [100.0, 200.0, 50.0, 400.0])
    assert not res.pooled
    assert np.all(res.sources["own_upper"] > 0)


def test_zero_event_source_is_shrunk_up():
    res = pooled_rate([0, 4, 6, 3, 5], [500.0, 1000.0, 1200.0, 800.0, 900.0])
    s = res.sources.iloc[0]
    assert s["own"] == 0.0
    assert 0 < s["shrunk_lower"] < s["shrunk"] < s["shrunk_upper"]
    assert s["shrunk"] > 0.5 / 500 * 0.5


def test_very_unequal_exposures():
    x = [3000, 2, 0, 5, 1]
    T = [1e9, 1e3, 50.0, 2e3, 400.0]
    for method in ("Bayes", "MLE"):
        res = pooled_rate(x, T, method=method, random_state=0)
        s = res.sources
        assert np.all(
            np.isfinite(
                s[["shrunk", "shrunk_lower", "shrunk_upper"]].to_numpy()
            )
        )
        # the huge source pins its own rate
        assert s["shrunk"].iloc[0] == pytest.approx(3e-6, rel=0.02)
        p = res.predictive
        assert 0 < p["lower"] < p["median"] < p["upper"]


def test_homogeneous_sources_have_no_spread_under_mle():
    x = np.array([10, 20, 30, 40], float)
    T = 1000.0 * x  # every source at exactly 0.001
    res = pooled_rate(x, T, method="MLE", random_state=0)
    pp = res.population_params
    assert np.isinf(pp["shape"]) and pp["cv"] == 0.0
    assert pp["mean"] == pytest.approx(0.001, rel=1e-12)
    assert res.population is None
    assert res.heterogeneity["chi2"] == pytest.approx(0.0, abs=1e-12)
    # the new source's range is then that of the mean
    p, a = res.predictive, res.average
    assert p["lower"] < 0.001 < p["upper"]
    assert p["upper"] / p["lower"] < 2.0
    assert a["lower"] < 0.001 < a["upper"]
    # Bayes still allows some spread, but not much
    rb = pooled_rate(x, T)
    assert rb.population_params["cv"] < 0.5


# --- properties -------------------------------------------------------------


def test_order_of_sources_does_not_matter():
    x, T = [0, 4, 6, 3, 5], [500.0, 1000.0, 1200.0, 800.0, 900.0]
    res = pooled_rate(x, T, labels=list("abcde"))
    perm = [3, 0, 4, 1, 2]
    rp = pooled_rate(
        np.take(x, perm), np.take(T, perm), labels=np.take(list("abcde"), perm)
    )
    for key in res.predictive:
        assert rp.predictive[key] == pytest.approx(
            res.predictive[key], rel=1e-9
        )
    assert rp.sources.loc[list("abcde")].to_numpy() == pytest.approx(
        res.sources.to_numpy(), rel=1e-9
    )


def test_rates_scale_with_the_unit_of_exposure():
    x, T = np.array([0, 4, 6, 3, 5]), np.array([500.0, 1e3, 1.2e3, 800, 900])
    res = pooled_rate(x, T)
    per_k = pooled_rate(x, T / 1000.0)  # exposure in thousands
    for key in res.predictive:
        assert per_k.predictive[key] == pytest.approx(
            1000.0 * res.predictive[key], rel=1e-3
        )
    assert per_k.population_params["shape"] == pytest.approx(
        res.population_params["shape"], rel=1e-3
    )


def test_update_is_the_shrunk_estimate_of_one_more_source():
    x, T = [0, 4, 6, 3, 5], [500.0, 1000.0, 1200.0, 800.0, 900.0]
    res = pooled_rate(x, T)
    new = res.update(2, 300.0)
    more = pooled_rate(x + [2], T + [300.0]).sources.iloc[-1]
    assert new["mean"] == pytest.approx(more["shrunk"], rel=5e-3)
    assert new["lower"] == pytest.approx(more["shrunk_lower"], rel=5e-3)
    assert new["upper"] == pytest.approx(more["shrunk_upper"], rel=5e-3)
    # no data of its own: the new source is the predictive distribution
    tiny = res.update(0, 1e-12)
    assert tiny["median"] == pytest.approx(res.predictive["median"], rel=1e-6)


def test_mle_bootstrap_is_seeded():
    a = pooled_rate(PUMP_X, PUMP_T, method="MLE", n_boot=50, random_state=3)
    b = pooled_rate(PUMP_X, PUMP_T, method="MLE", n_boot=50, random_state=3)
    assert a.predictive == b.predictive


@pytest.mark.parametrize("method", ["Bayes", "MLE"])
@pytest.mark.parametrize("kind", ["rate", "probability"])
def test_round_trip(kind, method):
    res = pooled_rate(
        FTS_X,
        FTS_N,
        kind,
        method=method,
        n_boot=30,
        random_state=0,
        labels=list("ABCDEF"),
    )
    text = json.dumps(res.to_dict(), allow_nan=False)
    back = PooledRate.from_dict(json.loads(text))
    assert back.predictive == res.predictive
    assert back.average == res.average
    assert back.population_params == res.population_params
    assert back.heterogeneity == res.heterogeneity
    assert back.sources.equals(res.sources)
    assert back.update(1, 400) == res.update(1, 400)
    assert repr(back) == repr(res)
    assert back.assumptions == res.assumptions


def test_round_trip_unpooled_and_infinite_shape():
    res = _quiet([3], [1000.0])
    back = PooledRate.from_dict(json.loads(json.dumps(res.to_dict())))
    assert back.message == res.message and not back.pooled
    x = np.array([10, 20, 30, 40], float)
    res = pooled_rate(x, 1000.0 * x, method="MLE", random_state=0)
    text = json.dumps(res.to_dict(), allow_nan=False)
    back = PooledRate.from_dict(json.loads(text))
    assert np.isinf(back.population_params["shape"])
    with pytest.raises(ValueError, match="not a PooledRate"):
        PooledRate.from_dict({"result": "PersonYears"})


def test_text_outputs():
    res = pooled_rate(FTS_X, FTS_N, "probability", alpha_ci=0.1)
    text = repr(res)
    assert "A new source like these" in text and "90%" in text
    assert "beta-binomial" in res.method_description
    assert any("independent" in line for line in res.assumptions)


# --- refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"events": [1, 2], "exposure": [1.0]}, "one value per source"),
        ({"events": [-1, 2], "exposure": [1.0, 2.0]}, "negative"),
        ({"events": [1, 2], "exposure": [0.0, 2.0]}, "more than 0"),
        ({"events": [1, np.nan], "exposure": [1.0, 2.0]}, "finite"),
        ({"events": [1, 2], "exposure": [1.0, 2.0], "kind": "time"}, "kind"),
        ({"events": [1, 2], "exposure": [1.0, 2.0], "method": "EB"}, "method"),
        (
            {"events": [1, 2], "exposure": [1.0, 2.0], "alpha_ci": 1.5},
            "alpha_ci",
        ),
        (
            {"events": [1, 2], "exposure": [1.0, 2.0], "min_sources": 1},
            "min_sources",
        ),
        (
            {"events": [1, 2], "exposure": [1.0, 2.0], "labels": ["a"]},
            "labels",
        ),
        (
            {"events": [5, 2], "exposure": [3, 2], "kind": "probability"},
            "cannot exceed",
        ),
        (
            {"events": [1.5, 2], "exposure": [3, 2], "kind": "probability"},
            "whole numbers",
        ),
    ],
)
def test_refusals_name_the_argument(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pooled_rate(**kwargs)
