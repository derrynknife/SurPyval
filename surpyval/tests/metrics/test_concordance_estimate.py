"""Uno's C, the jackknife standard error of Harrell's and Uno's C, and the
comparison of two scores' C (``concordance_estimate``,
``compare_concordance``).

The values R gives are checked in
``surpyval/tests/reference/test_concordance_references.py``. Here the fast
counts are checked against an O(n^2) definition of every subject's share
of the weighted pairs (``pairwise_shares``) on data with every kind of tie,
and the estimates against their properties: invariances, symmetry, the
limits, and the standard error against the spread of C over simulated
samples.
"""

import json
import time
from math import isclose

import numpy as np
import pytest

import surpyval as sp
from surpyval.metrics import (
    compare_concordance,
    concordance_estimate,
    concordance_index,
)
from surpyval.utils.ipcw import censoring_survival, step_left_limit


def pairwise_shares(x, c, s, weighting="harrell", tau=None, ties="therneau"):
    """Each subject's weighted score sum N_i and weight sum M_i over the
    usable pairs it belongs to, pair by pair (the definitions of
    ``concordance_index`` and R's ``timewt``)."""
    x = np.asarray(x, float)
    c = np.asarray(c)
    s = np.asarray(s, float)
    n = x.size
    if weighting == "uno":
        times, g = censoring_survival(x, c == 1, ties="events_first")
        w = 1 / step_left_limit(times, g, x, before=1.0) ** 2
    else:
        w = np.ones(n)
    N = np.zeros(n)
    M = np.zeros(n)
    for i in range(n):
        if c[i] != 0 or (tau is not None and x[i] > tau):
            continue
        for j in range(n):
            if j == i:
                continue
            tied = isclose(s[i], s[j], abs_tol=1e-8)
            if x[j] > x[i]:
                score = 1.0 if s[i] > s[j] else 0.5 if tied else 0.0
            elif x[j] == x[i] and c[j] == 1:
                score = 0.5 if tied else 1.0 if s[i] > s[j] else 0.0
            elif x[j] == x[i] and c[j] == 0 and ties == "harrell":
                if j < i:
                    continue  # the pair is counted once, from its first
                score = 1.0 if tied else 0.5
            else:
                continue
            for k in (i, j):
                N[k] += w[i] * score
                M[k] += w[i]
    return N, M


def _sample(rng, n, kind):
    if kind in ("time_ties", "all_ties"):
        x = rng.integers(1, max(3, n // 4), n).astype(float)
    else:
        x = rng.exponential(10.0, n)
    c = (rng.random(n) < rng.uniform(0.0, 0.6)).astype(int)
    if kind in ("score_ties", "all_ties"):
        s = rng.integers(0, 5, n).astype(float)
    elif kind == "near_ties":
        s = np.round(rng.normal(size=n), 1)
        s = s + rng.choice([0.0, 4e-9, -4e-9, 3e-8], n)
    else:
        s = rng.normal(size=n)
    return x, c, s


KINDS = ("plain", "time_ties", "score_ties", "all_ties", "near_ties")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ties", ["therneau", "harrell"])
@pytest.mark.parametrize("weighting", ["harrell", "uno"])
def test_agrees_with_the_pairwise_definition(kind, ties, weighting):
    rng = np.random.default_rng(2611)
    checked = 0
    for n in (2, 3, 5, 8, 13, 30, 61):
        x, c, s = _sample(rng, n, kind)
        tau = None if n % 2 else float(np.median(x))
        N, M = pairwise_shares(x, c, s, weighting, tau, ties)
        if M.sum() == 0:
            continue
        C = N.sum() / M.sum()
        d = (N - C * M) / (M.sum() / 2)
        res = concordance_estimate(
            x, c, s, weighting=weighting, tau=tau, ties=ties
        )
        assert res.estimate == pytest.approx(C, abs=1e-12)
        assert res.se == pytest.approx(np.sqrt(np.sum(d**2)), abs=1e-12)
        np.testing.assert_allclose(res.influence, d, atol=1e-12)
        checked += 1
    assert checked >= 4


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ties", ["therneau", "harrell"])
def test_harrell_estimate_is_concordance_index(kind, ties):
    rng = np.random.default_rng(1105)
    for n in (4, 50, 500):
        x, c, s = _sample(rng, n, kind)
        try:
            expected = concordance_index(x, c, s, ties=ties)
        except ValueError:
            continue
        res = concordance_estimate(x, c, s, ties=ties)
        assert res.estimate == pytest.approx(expected, abs=1e-12)


def test_uno_is_harrell_without_censoring():
    # G = 1 at every time, so every pair weighs 1.
    rng = np.random.default_rng(7)
    x = rng.exponential(5.0, 80)
    s = rng.normal(size=80) - 0.1 * x
    c = np.zeros(80, int)
    h = concordance_estimate(x, c, s)
    u = concordance_estimate(x, c, s, weighting="uno")
    assert u.estimate == pytest.approx(h.estimate, abs=1e-14)
    assert u.se == pytest.approx(h.se, abs=1e-14)


def test_invariant_to_monotone_transforms_of_score_and_time():
    rng = np.random.default_rng(8)
    x, c, s = _sample(rng, 150, "all_ties")
    base = concordance_estimate(x, c, s, weighting="uno", tau=20.0)
    moved = concordance_estimate(
        3.0 * x, c, np.exp(s) + 4.0, weighting="uno", tau=60.0
    )
    assert moved.estimate == pytest.approx(base.estimate, abs=1e-12)
    assert moved.se == pytest.approx(base.se, abs=1e-12)


@pytest.mark.parametrize("weighting", ["harrell", "uno"])
def test_reversed_scores_give_one_minus_c_with_the_same_se(weighting):
    # Tied scores count half either way, so the reversal is exact.
    rng = np.random.default_rng(9)
    x, c, s = _sample(rng, 120, "all_ties")
    a = concordance_estimate(x, c, s, weighting=weighting)
    b = concordance_estimate(x, c, -s, weighting=weighting)
    assert a.estimate + b.estimate == pytest.approx(1.0, abs=1e-12)
    assert a.se == pytest.approx(b.se, abs=1e-12)


def test_tau_keeps_only_the_pairs_whose_event_is_at_or_before_it():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 0, 0, 1])
    s = np.array([0.1, 0.9, 0.8, 0.7, 0.6])
    # tau = 1: only subject 1's pairs, all discordant
    assert concordance_estimate(x, c, s, tau=1.0).estimate == 0.0
    # tau = 1.5 is the same pairs; tau = 2 adds subject 2's three
    assert concordance_estimate(x, c, s, tau=1.5).estimate == 0.0
    res = concordance_estimate(x, c, s, tau=2.0)
    assert (res.concordant, res.pairs) == (3.0, 7.0)
    with pytest.raises(ValueError, match="at or before tau"):
        concordance_estimate(x, c, s, tau=0.5)


def test_the_standard_error_matches_the_spread_over_samples():
    # Fixed seed and size, set before the result was seen: the mean
    # jackknife SE is within 15% of the standard deviation of C over 400
    # samples of 200 (the Monte Carlo error of that SD is about 3.5%).
    rng = np.random.default_rng(20261105)
    for weighting in ("harrell", "uno"):
        est, se = [], []
        for _ in range(400):
            z = rng.normal(size=200)
            t = rng.exponential(np.exp(-0.7 * z))
            cens = rng.exponential(1.5, 200)
            x = np.minimum(t, cens)
            c = (cens < t).astype(int)
            res = concordance_estimate(x, c, z, weighting=weighting, tau=1.5)
            est.append(res.estimate)
            se.append(res.se)
        ratio = np.mean(se) / np.std(est, ddof=1)
        assert 0.85 < ratio < 1.15, (weighting, ratio)


def test_intervals():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    c = [0, 1, 0, 1, 0, 0]
    s = [0.9, 0.5, 0.7, 0.6, 0.2, 0.4]
    normal = concordance_estimate(x, c, s, alpha=0.1)
    z = 1.6448536269514722
    assert normal.ci == pytest.approx(
        (normal.estimate - z * normal.se, normal.estimate + z * normal.se)
    )
    logit = concordance_estimate(x, c, s, ci_method="logit")
    assert 0 < logit.ci[0] < logit.estimate < logit.ci[1] < 1
    assert normal.ci[1] > 1  # what the logit scale avoids
    perfect = concordance_estimate(x, c, -np.array(x), ci_method="logit")
    assert perfect.estimate == 1.0 and perfect.ci == (1.0, 1.0)


def test_comparison():
    rng = np.random.default_rng(11)
    x, c, s = _sample(rng, 200, "time_ties")
    other = s + rng.normal(size=200)
    res = compare_concordance(x, c, s, other, weighting="uno", tau=10.0)
    a = concordance_estimate(x, c, s, weighting="uno", tau=10.0)
    b = concordance_estimate(x, c, other, weighting="uno", tau=10.0)
    assert res.estimates == pytest.approx([a.estimate, b.estimate])
    assert res.cov[0, 0] == pytest.approx(a.se**2)
    assert res.cov[1, 1] == pytest.approx(b.se**2)
    assert res.se == pytest.approx(
        np.sqrt(np.sum((a.influence - b.influence) ** 2))
    )
    swapped = compare_concordance(x, c, other, s, weighting="uno", tau=10.0)
    assert swapped.difference == pytest.approx(-res.difference)
    assert swapped.se == pytest.approx(res.se)
    assert swapped.p_value == pytest.approx(res.p_value)
    # A score against itself: no difference, no spread
    same = compare_concordance(x, c, s, s)
    assert same.difference == 0.0 and same.se == 0.0


def test_results_are_strict_json():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    c = [0, 1, 0, 1, 0, 0]
    s = [0.9, 0.5, 0.7, 0.6, 0.2, 0.4]
    one = concordance_estimate(x, c, s, weighting="uno", tau=5)
    d = json.loads(json.dumps(one.to_dict(), allow_nan=False))
    assert d["estimate"] == pytest.approx(one.estimate)
    assert d["tau"] == 5.0 and d["weighting"] == "uno"
    assert float(one) == one.estimate
    assert "Uno's C" in repr(one)
    two = compare_concordance(x, c, s, s[::-1])
    d = json.loads(json.dumps(two.to_dict(), allow_nan=False))
    assert d["difference"] == pytest.approx(two.difference)
    same = compare_concordance(x, c, s, s)
    assert same.to_dict()["statistic"] is None  # 0 / 0
    assert "Comparison" in repr(two)


def test_refusals_and_missing_values():
    x = [1.0, 2.0, 3.0]
    c = [0, 0, 1]
    s = [0.3, 0.2, 0.1]
    with pytest.raises(ValueError, match="weighting"):
        concordance_estimate(x, c, s, weighting="ipcw")
    with pytest.raises(ValueError, match="ci_method"):
        concordance_estimate(x, c, s, ci_method="wald")
    with pytest.raises(ValueError, match="'tau' must be a positive"):
        concordance_estimate(x, c, s, tau=0.0)
    with pytest.raises(ValueError, match="'alpha'"):
        concordance_estimate(x, c, s, alpha=1.5)
    with pytest.raises(ValueError, match="same length"):
        concordance_estimate(x, c, s[:2])
    with pytest.raises(ValueError, match="same length"):
        concordance_estimate(x, c[:2], s)
    with pytest.raises(ValueError, match="missing"):
        concordance_estimate(x, c, [0.3, np.nan, 0.1])
    with pytest.raises(ValueError, match="right censored"):
        concordance_estimate(x, [0, -1, 1], s)
    with pytest.raises(ValueError, match="No usable pairs"):
        concordance_estimate([1.0, 1.0], [0, 0], [0.1, 0.2])
    res = concordance_estimate([1.0, np.nan, 3.0], c, s)
    assert np.isnan(res.estimate) and np.isnan(res.se)
    with pytest.raises(ValueError, match="missing"):
        compare_concordance([1.0, np.nan, 3.0], c, s, s)


def test_fifty_thousand_subjects_in_seconds():
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.exponential(10.0, n).round(2)
    c = (rng.random(n) < 0.3).astype(int)
    s = rng.normal(size=n).round(3)
    start = time.perf_counter()
    res = concordance_estimate(x, c, s, weighting="uno", tau=20.0)
    assert time.perf_counter() - start < 10.0
    assert 0.45 < res.estimate < 0.55


# ---------------------------------------------------------------------------
# On the regression models
# ---------------------------------------------------------------------------
def _data(n=120, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    t = sp.Weibull.random(n, 10.0, 1.5, random_state=rng) * np.exp(
        -(0.8 * z[:, 0] - 0.4 * z[:, 1]) / 1.5
    )
    cens = rng.uniform(0, 25, n)
    x = np.minimum(t, cens).round(3)
    c = (cens < t).astype(int)
    return x, c, z


@pytest.mark.parametrize("fitter", [sp.CoxPH, sp.WeibullPH])
def test_model_estimate_is_that_of_its_linear_predictor(fitter):
    x, c, z = _data()
    model = fitter.fit(x, z, c)
    beta = np.asarray(getattr(model, "beta", None))
    if beta.ndim == 0:
        beta = np.asarray(model.params)[model.k_dist :]
    for kwargs in ({}, {"weighting": "uno", "tau": 12.0}):
        expected = concordance_estimate(x, c, z @ beta, **kwargs)
        res = model.concordance_estimate(**kwargs)
        assert res.estimate == pytest.approx(expected.estimate, abs=1e-12)
        assert res.se == pytest.approx(expected.se, abs=1e-12)
        assert model.concordance(**kwargs) == pytest.approx(
            expected.estimate, abs=1e-12
        )
    x2, c2, z2 = _data(60, seed=9)
    res = model.concordance_estimate(x2, c2, z2, weighting="uno")
    expected = concordance_estimate(x2, c2, z2 @ beta, weighting="uno")
    assert res.estimate == pytest.approx(expected.estimate)


def test_model_estimate_refusals():
    x, c, z = _data()
    strata = np.tile(["a", "b"], x.size // 2)
    model = sp.CoxPH.fit(x, z, c, strata=strata)
    with pytest.raises(ValueError, match="stratified"):
        model.concordance_estimate()
    with pytest.raises(ValueError, match="stratified"):
        model.concordance(weighting="uno")
    plain = sp.CoxPH.fit(x, z, c)
    with pytest.raises(ValueError, match="together"):
        plain.concordance_estimate(x, c)
    z_missing = z.copy()
    z_missing[0, 0] = np.nan
    with pytest.raises(ValueError, match="covariate is missing"):
        plain.concordance_estimate(x, c, z_missing)
