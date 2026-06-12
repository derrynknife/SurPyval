import numpy as np
import pytest
from scipy.stats import CensoredData
from scipy.stats import logrank as scipy_logrank

import surpyval


def _to_scipy(obs, c):
    # surpyval c==1 is right censored; scipy wants uncensored/right
    return CensoredData(uncensored=obs[c == 0], right=obs[c == 1])


def test_logrank_matches_scipy_two_groups():
    rng = np.random.default_rng(0)
    oa = np.minimum(rng.exponential(10, 40), rng.exponential(15, 40))
    da = rng.integers(0, 2, 40)
    ob = np.minimum(rng.exponential(7, 35), rng.exponential(15, 35))
    db = rng.integers(0, 2, 35)
    # build censoring flags (1 == censored)
    ca = 1 - da
    cb = 1 - db

    sp = scipy_logrank(_to_scipy(oa, ca), _to_scipy(ob, cb))

    x = np.concatenate([oa, ob])
    c = np.concatenate([ca, cb])
    Z = np.array([0] * 40 + [1] * 35)
    res = surpyval.logrank(x, Z, c=c)

    # scipy reports a z statistic; the chi-squared statistic is z**2
    assert np.isclose(res.statistic, sp.statistic**2, atol=1e-6)
    assert np.isclose(res.p_value, sp.pvalue, atol=1e-6)
    assert res.dof == 1


def test_logrank_known_example():
    x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
    x += [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    c += [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Z = [1] * 11 + [2] * 12
    res = surpyval.logrank(x, Z, c=c)
    assert np.isclose(res.statistic, 3.3964, atol=1e-3)
    assert np.isclose(res.p_value, 0.0653, atol=1e-3)


def test_logrank_fh_zero_equals_standard():
    rng = np.random.default_rng(1)
    x = rng.exponential(5, 60)
    Z = rng.integers(0, 2, 60)
    standard = surpyval.logrank(x, Z)
    fh = surpyval.logrank(x, Z, weighting="fleming-harrington", rho=0, gamma=0)
    assert np.isclose(standard.statistic, fh.statistic, atol=1e-9)


def test_logrank_weightings_run_and_differ():
    rng = np.random.default_rng(2)
    x = rng.exponential(5, 80)
    c = rng.integers(0, 2, 80)
    Z = rng.integers(0, 2, 80)
    stats = {}
    for w in ["log-rank", "gehan", "tarone-ware", "fleming-harrington"]:
        res = surpyval.logrank(x, Z, c=c, weighting=w)
        assert res.statistic >= 0
        assert 0 <= res.p_value <= 1
        stats[w] = res.statistic
    # Gehan weights early events more heavily, so it should differ from
    # the unweighted log-rank in general.
    assert stats["gehan"] != stats["log-rank"]


def test_logrank_three_groups_dof():
    rng = np.random.default_rng(3)
    x = rng.exponential(5, 90)
    Z = rng.integers(0, 3, 90)
    res = surpyval.logrank(x, Z)
    assert res.dof == 2


def test_logrank_identical_groups_small_statistic():
    x = np.array([1.0, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    Z = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    res = surpyval.logrank(x, Z)
    assert res.statistic < 1e-9
    assert res.p_value > 0.99


def test_logrank_rejects_interval_censoring():
    x = np.array([1.0, 2, 3, 4])
    c = np.array([0, 2, 0, 0])
    Z = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        surpyval.logrank(x, Z, c=c)


def test_logrank_requires_two_groups():
    x = np.array([1.0, 2, 3])
    Z = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        surpyval.logrank(x, Z)


def test_logrank_bad_weighting():
    x = np.array([1.0, 2, 3, 4])
    Z = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        surpyval.logrank(x, Z, weighting="not-a-weighting")
