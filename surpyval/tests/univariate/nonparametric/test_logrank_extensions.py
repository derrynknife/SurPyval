"""The max-combo, trend, permutation and maximally selected log-rank tests:
their agreement with the log-rank test where they must agree, brute-force
checks of the permutation and cut-point searches, refusals and JSON."""

import itertools
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

import surpyval as sp
from surpyval.univariate.nonparametric.logrank import _logrank_parts
from surpyval.univariate.nonparametric.logrank_maxcombo import _max_normal_sf
from surpyval.univariate.nonparametric.logrank_permutation import (
    logrank_scores,
)

AML_X = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
         5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]  # fmt: skip
AML_C = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # fmt: skip
AML_Z = [1] * 11 + [2] * 12


def _data(seed=0, n=40, k=2):
    rng = np.random.default_rng(seed)
    t = rng.weibull(1.5, n * k) * np.repeat(np.linspace(10, 14, k), n)
    cens = rng.uniform(0, 20, n * k)
    x = np.round(np.minimum(t, cens), 1)
    c = (cens < t).astype(int)
    return x, c, np.repeat(np.arange(k), n)


def _strict(d):
    return json.loads(json.dumps(d, allow_nan=False))


# --- max-combo ------------------------------------------------------------


def test_maxcombo_one_component_is_the_logrank_test():
    x, c, Z = _data(1)
    for rho, gamma in [(0, 0), (1, 0), (0, 1), (2, 0.5)]:
        res = sp.maxcombo_logrank(x, Z, c=c, rho=[rho], gamma=[gamma])
        lr = sp.logrank(
            x, Z, c=c, weighting="fleming-harrington", rho=rho, gamma=gamma
        )
        assert_allclose(res.z[0] ** 2, lr.statistic, rtol=1e-12)
        assert_allclose(res.p_value, lr.p_value, rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strata": np.tile([0, 1], 40)},
        {"t0": 8.0},
        {"tl": "late"},
    ],
)
def test_maxcombo_components_follow_logrank_options(kwargs):
    x, c, Z = _data(2)
    if kwargs.get("tl") == "late":
        x = x + 0.1
        kwargs = {"tl": np.where(np.arange(80) % 2 == 0, 0.0, 0.5 * x)}
    res = sp.maxcombo_logrank(x, Z, c=c, **kwargs)
    for j, (rho, gamma) in enumerate(zip(res.rho, res.gamma)):
        lr = sp.logrank(
            x, Z, c=c, weighting="fleming-harrington", rho=rho, gamma=gamma,
            **kwargs,
        )  # fmt: skip
        assert_allclose(res.z[j] ** 2, lr.statistic, rtol=1e-10)


def test_maxcombo_p_value_allows_for_the_search():
    x, c, Z = _data(3)
    res = sp.maxcombo_logrank(x, Z, c=c)
    assert min(res.p_values) <= res.p_value <= res.p_bonferroni
    greater = sp.maxcombo_logrank(x, Z, c=c, alternative="greater")
    less = sp.maxcombo_logrank(x, Z, c=c, alternative="less")
    assert greater.p_value <= res.p_value and less.p_value >= 0.5


def test_maxcombo_counts_equal_copies():
    x = [3, 5, 7, 8, 10, 2, 3, 4, 6]
    Z = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    n = [1, 2, 1, 3, 1, 2, 1, 1, 2]
    res = sp.maxcombo_logrank(x, Z, n=n)
    rep = sp.maxcombo_logrank(np.repeat(x, n), np.repeat(Z, n))
    assert_allclose(res.z, rep.z, rtol=1e-12)
    assert_allclose(res.p_value, rep.p_value, rtol=1e-12)


def test_maxcombo_nothing_to_compare():
    res = sp.maxcombo_logrank([1, 2, 3, 4], [0, 0, 1, 1], c=[1, 1, 1, 1])
    assert res.p_value == 1.0 and res.statistic == 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"Z": [0, 0, 1, 1, 2, 2]}, "two groups"),
        ({"w": [1.0] * 6}, "sampling weights"),
        ({"rho": [0, 0], "gamma": [1, 1]}, "repeat"),
        ({"rho": [0, 1], "gamma": [0]}, "same"),
        ({"rho": [-1], "gamma": [0]}, "non-negative"),
        ({"alternative": "two.sided"}, "alternative"),
    ],
)
def test_maxcombo_refusals(kwargs, match):
    args = {"x": [1, 2, 3, 4, 5, 6], "Z": [0, 0, 0, 1, 1, 1]}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sp.maxcombo_logrank(**args)


def test_maxcombo_to_dict_is_strict_json():
    x, c, Z = _data(4)
    d = _strict(sp.maxcombo_logrank(x, Z, c=c).to_dict())
    assert len(d["components"]) == 4 and len(d["correlation"]) == 4
    assert "Max-Combo" in repr(sp.maxcombo_logrank(x, Z, c=c))


def test_max_normal_sf_closed_forms():
    R = np.eye(3)
    m = 2.1
    assert_allclose(
        _max_normal_sf(m, R, "two-sided"),
        1 - (1 - 2 * norm.sf(m)) ** 3,
        rtol=3e-3,
    )
    assert_allclose(
        _max_normal_sf(m, R, "greater"), 1 - norm.cdf(m) ** 3, rtol=3e-3
    )
    assert_allclose(
        _max_normal_sf(-m, R, "less"), 1 - norm.sf(-m) ** 3, rtol=3e-3
    )
    # Perfectly correlated components are one.
    assert_allclose(
        _max_normal_sf(m, np.ones((3, 3)), "two-sided"),
        2 * norm.sf(m),
        rtol=1e-9,
    )


# --- trend ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"weighting": "gehan"},
        {"weighting": "fleming-harrington", "rho": 1, "gamma": 1},
        {"strata": np.tile([0, 1], 40)},
        {"w": np.linspace(0.5, 2, 80)},
        {"t0": 9.0},
    ],
)
def test_trend_of_two_groups_is_the_logrank_test(kwargs):
    x, c, Z = _data(5)
    trend = sp.trend_logrank(x, Z, c=c, **kwargs)
    lr = sp.logrank(x, Z, c=c, **kwargs)
    assert_allclose(trend.statistic, lr.statistic, rtol=1e-10)
    assert_allclose(trend.p_value, lr.p_value, rtol=1e-9)


def test_trend_depends_only_on_the_spacing_of_the_scores():
    x, c, Z = _data(6, k=3)
    a = sp.trend_logrank(x, Z, c=c, scores=[1, 2, 5])
    b = sp.trend_logrank(x, Z, c=c, scores=[10, 30, 90])
    d = sp.trend_logrank(x, Z, c=c, scores={0: 1, 1: 2, 2: 5})
    assert_allclose([a.z, d.z], [b.z, b.z], rtol=1e-12)
    flipped = sp.trend_logrank(x, Z, c=c, scores=[5, 2, 1 - 3])
    assert np.sign(flipped.z) != np.sign(a.z) or a.z == 0


def test_trend_one_sided():
    x, c, Z = _data(7, k=3)  # the scale rises with the group
    two = sp.trend_logrank(x, Z, c=c)
    up = sp.trend_logrank(x, Z, c=c, alternative="increasing")
    down = sp.trend_logrank(x, Z, c=c, alternative="decreasing")
    assert_allclose(up.p_value + down.p_value, 1.0, rtol=1e-12)
    assert_allclose(min(up.p_value, down.p_value) * 2, two.p_value)


def test_trend_is_more_powerful_for_a_steady_trend():
    rng = np.random.default_rng(8)
    Z = np.repeat([0, 1, 2, 3], 30)
    x = rng.exponential(1.0 / np.exp(0.25 * Z))
    assert sp.trend_logrank(x, Z).p_value < sp.logrank(x, Z).p_value


@pytest.mark.parametrize(
    "scores, match",
    [
        ([1, 2], "one score per group"),
        ([1, 1, 1], "all equal"),
        ([1, np.nan, 2], "finite"),
        ({0: 1, 1: 2}, "no score"),
        ({0: 1, 1: 2, 2: 3, 9: 4}, "not in 'Z'"),
    ],
)
def test_trend_refuses_bad_scores(scores, match):
    x, c, Z = _data(9, k=3)
    with pytest.raises(ValueError, match=match):
        sp.trend_logrank(x, Z, c=c, scores=scores)


def test_trend_to_dict_and_nothing_to_compare():
    x, c, Z = _data(10, k=3)
    _strict(sp.trend_logrank(x, Z, c=c).to_dict())
    with pytest.warns(UserWarning, match="single member") as caught:
        res = sp.trend_logrank([1, 2, 3], [0, 1, 2], c=[1, 1, 1])
    # The warning points at the caller's line.
    assert caught[0].filename == __file__
    assert res.dof == 0 and res.p_value == 1.0
    with pytest.raises(ValueError, match="alternative"):
        sp.trend_logrank(x, Z, alternative="greater")


# --- permutation ----------------------------------------------------------


def test_scores_sum_to_observed_minus_expected():
    x, c, Z = _data(11, k=3)
    for weighting, rho, gamma in [
        ("log-rank", 0, 0),
        ("gehan", 0, 0),
        ("tarone-ware", 0, 0),
        ("peto-peto", 0, 0),
        ("fleming-harrington", 1, 0.5),
    ]:
        a = logrank_scores(x.astype(float), c, weighting, rho, gamma)
        U = np.bincount(Z, weights=a)
        z, *_ = _logrank_parts(
            x, Z, c, None, weighting, rho, gamma, None, None, None, False
        )
        assert_allclose(U, z, atol=1e-10)


def test_exact_matches_brute_force_relabelling():
    x = np.array([3, 5, 7, 2, 3, 4, 6, 7, 8.0])
    c = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0])
    Z = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    res = sp.permutation_logrank(x, Z, c=c)
    assert res.method == "exact" and res.n_permutations == 1680
    # Every one of the 9! orderings of the labels (each relabelling 216
    # times), with the statistic computed afresh.
    a = logrank_scores(x, c)
    N = 9
    Sigma = (
        np.sum((a - a.mean()) ** 2)
        / (N - 1)
        * (np.diag([3.0, 3, 3]) - np.full((3, 3), 1.0))
    )
    inv = np.linalg.pinv(Sigma)
    stats = []
    for perm in itertools.permutations(Z):
        U = np.bincount(np.array(perm), weights=a, minlength=3) - 3 * a.mean()
        stats.append(U @ inv @ U)
    stats = np.array(stats)
    assert_allclose(
        res.p_value, np.mean(stats >= res.statistic - 1e-9), rtol=1e-12
    )


def test_monte_carlo_agrees_with_exact_and_is_seeded():
    exact = sp.permutation_logrank(AML_X, AML_Z, c=AML_C, method="exact",
                                   max_exact=2_000_000)  # fmt: skip
    mc = sp.permutation_logrank(
        AML_X, AML_Z, c=AML_C, n_permutations=20000, random_state=5
    )
    assert mc.method == "monte-carlo"
    assert abs(mc.p_value - exact.p_value) < 4 * mc.p_value_se
    again = sp.permutation_logrank(
        AML_X, AML_Z, c=AML_C, n_permutations=20000, random_state=5
    )
    assert again.p_value == mc.p_value
    np.random.seed(3)
    g1 = sp.permutation_logrank(AML_X, AML_Z, c=AML_C, n_permutations=500)
    np.random.seed(3)
    g2 = sp.permutation_logrank(AML_X, AML_Z, c=AML_C, n_permutations=500)
    assert g1.p_value == g2.p_value


def test_permutation_stratified_exact_counts_the_product():
    x = np.array([3, 5, 7, 2, 3, 4, 6, 7, 8, 1.0])
    Z = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    s = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    res = sp.permutation_logrank(x, Z, strata=s)
    assert res.method == "exact" and res.n_permutations == 6 * 20
    assert res.strata == 2


def test_permutation_counts_are_units():
    x = [3, 5, 7, 2, 3, 4]
    Z = [0, 0, 0, 1, 1, 1]
    n = [2, 1, 1, 1, 2, 1]
    res = sp.permutation_logrank(x, Z, n=n)
    rep = sp.permutation_logrank(np.repeat(x, n), np.repeat(Z, n))
    assert res.p_value == rep.p_value and res.n_permutations == 70


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tl": [0.5] * 6}, "delayed entry"),
        ({"w": [1.0] * 6}, "sampling weights"),
        ({"n": [1.5] * 6}, "whole numbers"),
        ({"method": "exact", "max_exact": 5}, "max_exact"),
        ({"method": "approximate"}, "method"),
        ({"n_permutations": 0}, "n_permutations"),
    ],
)
def test_permutation_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sp.permutation_logrank([3, 5, 7, 2, 3, 4], [0, 0, 0, 1, 1, 1],
                               **kwargs)  # fmt: skip


def test_permutation_to_dict_is_strict_json():
    res = sp.permutation_logrank(AML_X, AML_Z, c=AML_C, random_state=1,
                                 n_permutations=999)  # fmt: skip
    d = _strict(res.to_dict())
    assert d["method"] == "monte-carlo" and d["p_value_se"] > 0
    assert "Monte Carlo s.e." in repr(res)


# --- maximally selected statistic -----------------------------------------


def _brute_force(x, c, Z, minprop=0.1, maxprop=0.9):
    a = logrank_scores(np.asarray(x, float), np.asarray(c))
    N = a.size
    best = (-1.0, None)
    for cut in np.unique(Z):
        below = Z <= cut
        m = below.sum()
        if not (max(1, np.floor(N * minprop)) <= m <= np.floor(N * maxprop)):
            continue
        E = m / N * a.sum()
        V = m * (N - m) / (N**2 * (N - 1)) * (N * np.sum(a**2) - a.sum() ** 2)
        stat = abs(a[below].sum() - E) / np.sqrt(V)
        if stat > best[0] + 1e-12:
            best = (stat, cut)
    return best


def test_maxstat_matches_brute_force_search():
    rng = np.random.default_rng(12)
    Z = np.round(rng.uniform(0, 10, 80), 1)
    t = rng.exponential(np.where(Z > 6.5, 1.0, 2.5))
    cens = rng.uniform(0, 5, 80)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    res = sp.maxstat_logrank(x, Z, c=c)
    stat, cut = _brute_force(x, c, Z)
    assert_allclose(res.statistic, stat, rtol=1e-12)
    assert res.cutpoint == cut
    assert res.n_below == np.sum(Z <= cut)
    # A monotone transformation of Z moves the cut, not the statistic.
    res_log = sp.maxstat_logrank(x, np.exp(Z), c=c)
    assert_allclose(res_log.statistic, res.statistic, rtol=1e-12)
    assert_allclose(res_log.cutpoint, np.exp(res.cutpoint))


def test_maxstat_p_values_order():
    rng = np.random.default_rng(13)
    Z = rng.uniform(0, 10, 100)
    x = rng.exponential(np.where(Z > 5, 1.0, 1.6))
    p = {
        m: sp.maxstat_logrank(x, Z, method=m, random_state=1).p_value
        for m in ("exact", "lausen94", "permutation")
    }
    res = sp.maxstat_logrank(x, Z)
    assert res.naive_p_value < p["exact"] <= p["lausen94"]
    assert abs(p["permutation"] - p["exact"]) < 0.03


def test_maxstat_one_cut_is_the_two_sided_normal():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    Z = [0, 0, 0, 0, 1, 1, 1, 1]
    res = sp.maxstat_logrank(x, Z)
    assert res.cuts.size == 1
    assert_allclose(res.p_value, res.naive_p_value, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tl": [0.1] * 10}, "delayed entry"),
        ({"w": [1.0] * 10}, "sampling weights"),
        ({"Z": ["a"] * 10}, "numeric covariate"),
        ({"Z": [1.0] * 9 + [np.nan]}, "missing"),
        ({"Z": [1.0] * 10}, "No cut"),
        ({"minprop": 0.6, "maxprop": 0.4}, "minprop"),
        ({"method": "Lau94"}, "method"),
        ({"c": [2] * 10}, "right censored"),
    ],
)
def test_maxstat_refusals(kwargs, match):
    args = {"x": np.arange(1.0, 11), "Z": np.arange(10.0)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sp.maxstat_logrank(**args)


def test_maxstat_to_dict_is_strict_json():
    rng = np.random.default_rng(14)
    Z = rng.uniform(0, 10, 50)
    x = rng.exponential(1.0, 50)
    res = sp.maxstat_logrank(x, Z, method="permutation", random_state=2,
                             n_permutations=499)  # fmt: skip
    d = _strict(res.to_dict())
    assert len(d["cuts"]) == res.cuts.size and d["n_permutations"] == 499
    assert "not valid" in repr(res)
