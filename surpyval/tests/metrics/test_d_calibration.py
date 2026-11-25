"""The PIT histogram and D-calibration (``surpyval.metrics.d_calibration``)."""

import json

import numpy as np
import pytest
from scipy.stats import binom, chi2, weibull_min

import surpyval as sp
from surpyval.metrics import DCalibration, d_calibration, pit_values


def _censored_weibull(seed, n=300, shape=1.7, scale=10.0, cmax=14.0):
    rng = np.random.default_rng(seed)
    t = scale * rng.weibull(shape, n)
    cz = cmax * rng.uniform(size=n)
    return np.minimum(t, cz), (cz < t).astype(int)


# SurvivalEVAL 0.9.0's ``d_calibration`` (Haider et al.'s reference
# implementation, rewritten) on the data of ``_censored_weibull(2020)`` with
# predictions from a Weibull(scale 11, shape 1.5), given there as survival
# probabilities ``1 - F`` and event indicators ``1 - c``. Its bins run on the
# survival scale from 1 down to 0, so its first bin is ``u`` in [0, 0.1),
# the same order as ours.
SURVIVALEVAL = {
    10: (
        10.281033375830983,
        0.32821679480109645,
        [
            27.071294159770996,
            34.25888631897546,
            37.1287362002644,
            33.829698419850885,
            40.33658195272425,
            24.688433296404426,
            23.51028782187859,
            26.34257753509006,
            26.416752147520487,
            26.416752147520487,
        ],
    ),
    5: (
        5.023840830605973,
        0.2848600133593929,
        [
            61.330180478746456,
            70.95843462011528,
            65.02501524912867,
            49.85286535696864,
            52.833504295040974,
        ],
    ),
}


@pytest.mark.parametrize("bins", sorted(SURVIVALEVAL))
def test_matches_survivaleval(bins):
    x, c = _censored_weibull(2020)
    F = weibull_min.cdf(x, 1.5, scale=11.0)
    stat, p, hist = SURVIVALEVAL[bins]
    result = d_calibration(F=F, c=c, bins=bins)
    np.testing.assert_allclose(result.observed, hist, rtol=1e-10)
    assert result.statistic == pytest.approx(stat, rel=1e-10)
    assert result.p_value == pytest.approx(p, rel=1e-9)
    assert result.df == bins - 1


def test_censored_case_spread_by_hand():
    # One event at u = 0.3 and one censored at u = 0.5, four bins: the
    # censored case's weight is spread over (0.5, 1], half in each of the
    # last two bins.
    r = d_calibration(F=[0.3, 0.5], c=[0, 1], bins=4)
    np.testing.assert_allclose(r.events, [0, 1, 0, 0])
    np.testing.assert_allclose(r.censored, [0, 0, 0.5, 0.5])
    # Censored at u = 0.6: 0.15/0.4 in bin 3, 0.25/0.4 in bin 4.
    r = d_calibration(F=[0.6], c=[1], bins=4)
    np.testing.assert_allclose(r.censored, [0, 0, 0.375, 0.625])
    # At u = 1 all of it in the last bin; at u = 0 evenly everywhere.
    np.testing.assert_allclose(
        d_calibration(F=[1.0], c=[1], bins=4).censored, [0, 0, 0, 1]
    )
    np.testing.assert_allclose(
        d_calibration(F=[0.0], c=[1], bins=4).censored, [0.25] * 4
    )
    # Pearson's statistic by hand: observed [0, 1, .5, .5], expected .5.
    r = d_calibration(F=[0.3, 0.5], c=[0, 1], bins=4)
    assert r.statistic == pytest.approx((0.25 + 0.25 + 0 + 0) / 0.5)
    assert r.p_value == pytest.approx(chi2.sf(1.0, 3))


def test_no_censoring_is_the_pit_histogram():
    rng = np.random.default_rng(7)
    u = rng.beta(2, 3, 250)
    r = d_calibration(F=u, bins=10)
    counts, _ = np.histogram(u, bins=np.linspace(0, 1, 11))
    np.testing.assert_array_equal(r.observed, counts)
    assert r.n_censored == 0
    np.testing.assert_array_equal(r.censored, 0)


def test_counts_weight_cases():
    F = np.array([0.1, 0.45, 0.8, 0.3])
    c = np.array([0, 1, 0, 1])
    n = np.array([3, 2, 1, 4])
    weighted = d_calibration(F=F, c=c, n=n, bins=5)
    expanded = d_calibration(F=np.repeat(F, n), c=np.repeat(c, n), bins=5)
    np.testing.assert_allclose(weighted.observed, expanded.observed)
    assert weighted.n == 10 and weighted.n_censored == 6


def test_model_route_matches_arrays():
    x, c = _censored_weibull(11, n=120)
    model = sp.Weibull.from_params([10.0, 1.7])
    a = d_calibration(x, c, model=model)
    b = d_calibration(F=model.ff(x), c=c)
    np.testing.assert_allclose(a.observed, b.observed)
    np.testing.assert_allclose(pit_values(model, x), model.ff(x))


def test_regression_model_with_covariates():
    rng = np.random.default_rng(5)
    n = 200
    Z = rng.normal(size=(n, 1))
    t = rng.weibull(1.5, n) * 10 * np.exp(0.5 * Z[:, 0])
    cz = rng.uniform(0, 25, n)
    x, c = np.minimum(t, cz), (cz < t).astype(int)
    model = sp.WeibullPH.fit(x=x, Z=Z, c=c)
    u = pit_values(model, x, Z)
    expected = np.array(
        [
            1 - model.sf(np.array([xi]), Z[i : i + 1])[0]
            for i, xi in enumerate(x)
        ]
    )
    np.testing.assert_allclose(u, expected, rtol=1e-12)
    r = d_calibration(x, c, model=model, Z=Z)
    assert r.n == pytest.approx(n) and r.p_value > 0.01


def _rejections(F_of, seeds, n=200, cmax=15.0, randomise=False):
    hits = 0
    for seed in seeds:
        x, c = _censored_weibull(seed, n=n, shape=2.0, scale=10.0, cmax=cmax)
        r = d_calibration(F=F_of(x), c=c, randomise=randomise, seed=[7, seed])
        hits += r.p_value < 0.05
    return hits


def _true(x):
    return weibull_min.cdf(x, 2.0, scale=10.0)


# 400 data sets of 200 cases under the true model: a 5% test with its
# stated level rejects about 20; the 99.9% binomial range is [8, 35].
LO, HI = binom.ppf([0.0005, 0.9995], 400, 0.05)


def test_type_one_error_nominal_without_censoring():
    hits = _rejections(_true, range(1000, 1400), cmax=np.inf)
    assert hits == 21
    assert LO <= hits <= HI


def test_spread_is_conservative_with_censoring():
    # About 30% censored: Haider et al.'s spread histogram varies less than
    # the chi-square assumes, so it rejects a right model less than 5% of
    # the time (4 of 400 here).
    hits = _rejections(_true, range(1000, 1400), cmax=30.0)
    assert hits == 4
    assert hits <= HI


@pytest.mark.parametrize("cmax, expected", [(30.0, 21), (15.0, 15), (8.0, 19)])
def test_randomised_has_nominal_level_with_censoring(cmax, expected):
    # 30%, 57% and 82% censored.
    hits = _rejections(_true, range(1000, 1400), cmax=cmax, randomise=True)
    assert hits == expected
    assert LO <= hits <= HI


def test_misspecified_model_fails():
    # The shape wrong (1.2 for 2.0): the predicted spread is too wide.
    seeds = range(2000, 2100)
    hits = _rejections(lambda x: weibull_min.cdf(x, 1.2, scale=10.0), seeds)
    assert hits >= 90, hits


def test_randomised_pit_for_discrete_outcomes():
    # Geometric outcomes: the CDF jumps at each value, so the plain PIT
    # piles up and fails; the randomised one is uniform under the true
    # model.
    rng = np.random.default_rng(9)
    p = 0.3
    y = rng.geometric(p, 2000)
    F = 1 - (1 - p) ** y
    F_left = 1 - (1 - p) ** (y - 1)
    plain = d_calibration(F=F)
    rand = d_calibration(F=F, F_left=F_left, randomise=True, seed=1)
    assert plain.p_value < 1e-6
    assert rand.p_value > 0.01
    assert rand.randomised


def test_randomise_on_continuous_model_changes_nothing():
    # With no jump at an outcome there is nothing to draw: outcomes keep
    # their u. Censored cases are drawn, and counted as outcomes.
    x, c = _censored_weibull(3, n=80)
    model = sp.Weibull.from_params([10.0, 1.7])
    a = d_calibration(x[c == 0], model=model)
    b = d_calibration(x[c == 0], model=model, randomise=True, seed=0)
    np.testing.assert_allclose(a.observed, b.observed, atol=1e-9)
    r = d_calibration(x, c, model=model, randomise=True, seed=0)
    assert r.n_censored == c.sum() and np.all(r.censored == 0)
    assert r.events.sum() == pytest.approx(80)


def test_randomise_on_step_model_uses_left_limit():
    # A Kaplan-Meier estimate jumps at each failure: at its own data the
    # randomised PIT draws between the CDF before and at the jump.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    km = sp.KaplanMeier.fit(x)
    r = d_calibration(x, model=km, bins=4, randomise=True, seed=0)
    # Each jump is 0.25 wide and is the bin itself: one draw in each bin.
    np.testing.assert_allclose(r.observed, [1, 1, 1, 1])


def test_by_groups():
    rng = np.random.default_rng(4)
    F = rng.uniform(size=90)
    c = (rng.uniform(size=90) < 0.2).astype(int)
    h = np.repeat(["1 y", "2 y", "5 y"], 30)
    r = d_calibration(F=F, c=c, by=h, bins=5)
    assert list(r.groups) == ["1 y", "2 y", "5 y"]
    total = sum(g.observed for g in r.groups.values())
    np.testing.assert_allclose(total, r.observed)
    assert all(g.n == 30 for g in r.groups.values())
    assert "by group" in repr(r)


def test_round_trip_and_table():
    x, c = _censored_weibull(8, n=50)
    r = d_calibration(
        x,
        c,
        model=sp.Weibull.from_params([10, 1.7]),
        by=np.arange(50) % 2,
        bins=5,
    )
    d = json.loads(json.dumps(r.to_dict()))
    back = DCalibration.from_dict(d)
    np.testing.assert_allclose(back.observed, r.observed)
    assert back.p_value == pytest.approx(r.p_value)
    assert set(back.groups) == {"0", "1"}
    t = r.table
    assert list(t.columns) == [
        "bin",
        "u_from",
        "u_to",
        "events",
        "censored",
        "observed",
        "expected",
    ]
    assert t["observed"].sum() == pytest.approx(50)


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = d_calibration(F=[0.1, 0.6, 0.9], c=[0, 1, 0], bins=4).plot(ax=ax)
    assert out is ax and len(ax.patches) == 8
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs, words",
    [
        ({}, "either model="),
        ({"F": [0.2], "model": sp.Weibull.from_params([1, 1])}, "not both"),
        ({"F": [0.2, 0.4], "c": [0, -1]}, "left and interval"),
        ({"F": [0.2, 1.4]}, "outside"),
        ({"F": [0.2, np.nan]}, "missing"),
        ({"F": [0.2], "bins": 1}, "at least 2"),
        ({"F": [0.2], "bins": [0, 0.3, 1]}, "equal in width"),
        ({"F": [0.2], "F_left": [0.1]}, "randomise=True"),
        ({"F": [0.2], "F_left": [0.3], "randomise": True}, "must not exceed"),
        ({"F": [0.2], "Z": [[1.0]]}, "'Z' is used with model="),
        ({"F": [0.2, 0.3], "by": [1]}, "one label per case"),
        ({"model": sp.Weibull.from_params([1, 1])}, "needs 'x'"),
    ],
)
def test_refusals_say_what_to_do(kwargs, words):
    with pytest.raises(ValueError, match=words):
        d_calibration(**kwargs)


def test_top_level_names():
    assert sp.d_calibration is d_calibration
    assert sp.pit_values is pit_values
