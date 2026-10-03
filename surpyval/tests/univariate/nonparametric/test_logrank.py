import numpy as np
import pytest
from scipy.stats import CensoredData
from scipy.stats import logrank as scipy_logrank

import surpyval
import surpyval as sp


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


# ---------------------------------------------------------------------------
# Groups never at risk, missing labels and length checks.
# ---------------------------------------------------------------------------


def test_logrank_ignores_a_group_never_at_risk():
    base = sp.logrank([1, 2, 3, 4, 5, 6], list("bbbccc"))
    extra = sp.logrank(
        [1, 2, 3, 4, 5, 6, 0.5], list("bbbccca"), c=[0] * 6 + [1]
    )
    assert extra.dof == 1
    assert extra.statistic == pytest.approx(base.statistic)
    assert extra.p_value == pytest.approx(base.p_value)
    assert extra.p_value == pytest.approx(0.0246, abs=1e-4)


def test_logrank_without_informative_groups():
    # Group 1 is censored before the first event, so every risk set holds
    # group 0 only and there is nothing to compare.
    res = sp.logrank([2, 3, 1, 1.5], [0, 0, 1, 1], c=[0, 0, 1, 1])
    assert (res.statistic, res.dof, res.p_value) == (0.0, 0, 1.0)


@pytest.mark.parametrize("label", [np.nan, None])
def test_logrank_refuses_missing_group_labels(label):
    with pytest.raises(ValueError, match="missing"):
        sp.logrank([1, 2, 3, 4], np.array([0, 0, 1, label], dtype=object))


def test_logrank_refuses_missing_strata_labels():
    with pytest.raises(ValueError, match="strata"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], strata=[0, np.nan, 0, np.nan])


@pytest.mark.parametrize("arg", ["c", "n"])
def test_logrank_checks_c_and_n_lengths(arg):
    with pytest.raises(ValueError, match=f"'{arg}'"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], **{arg: [0, 0, 1]})


# -- delayed entry, tl (#576) ------------------------------------------------
# R's survdiff takes right censored data only; with delayed entry the
# log-rank test is the score test of the Cox model of the group, which R
# computes: coxph(Surv(tl, x, 1 - c) ~ Z, ties = "exact")$score (ties
# "exact" is the hypergeometric variance survdiff uses; survival 3.5.8).
AML_X = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
AML_X += [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
AML_C = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
AML_Z = [1] * 11 + [2] * 12
AML_TL = [6, 6, 6, 6] + [0] * 7 + [0, 0, 6, 6, 6, 6] + [0] * 6


def test_576_logrank_with_entry_times_is_r_cox_score_test():
    res = sp.logrank(AML_X, AML_Z, c=AML_C, tl=AML_TL)
    np.testing.assert_allclose(res.statistic, 3.365711, rtol=1e-6)
    # Without delayed entry it is survdiff's 3.396389, as before
    plain = sp.logrank(AML_X, AML_Z, c=AML_C)
    assert sp.logrank(AML_X, AML_Z, c=AML_C, tl=0).statistic == pytest.approx(
        plain.statistic, rel=1e-15
    )
    np.testing.assert_allclose(plain.statistic, 3.396389, rtol=1e-6)


def test_576_logrank_with_entry_times_untied_r():
    # Untied data, where every tie method's score test is the log-rank:
    # R gives 0.698339098460187.
    tl = np.array(
        "0.336 1.615 0.770 0.655 1.204 1.209 0.249 0.589 1.155 1.262 1.024 "
        "1.010 1.068 1.114 1.736 1.659 0.223 1.407 1.795 0.559 0.456 0.031 "
        "0.258 0.187 0.474 1.582 1.199 1.820 1.121 1.511".split(),
        float,
    )
    x = np.array(
        "0.941 2.801 1.644 1.527 1.274 3.256 0.659 1.162 2.040 1.403 1.709 "
        "1.774 2.719 3.849 2.211 2.260 0.990 1.734 2.532 1.191 1.878 0.564 "
        "0.714 0.726 0.515 2.725 2.479 2.340 1.382 1.868".split(),
        float,
    )
    s = np.array(
        "1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1".split(),
        int,
    )
    g = np.arange(1, 31) % 2
    res = sp.logrank(x, g, c=1 - s, tl=tl)
    np.testing.assert_allclose(res.statistic, 0.698339098460187, rtol=1e-10)
    # The same with the rows in counts and strata of one
    strat = sp.logrank(x, g, c=1 - s, tl=tl, strata=np.zeros(30))
    np.testing.assert_allclose(strat.statistic, res.statistic, rtol=1e-12)


def test_576_logrank_checks_entry_times():
    with pytest.raises(ValueError, match="'tl'"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], tl=[0, 0, 1])
    # An entry at or after the exit is refused as everywhere else
    with pytest.raises(ValueError, match="left truncated"):
        sp.logrank([1, 2, 3, 4], [0, 0, 1, 1], tl=[0, 2, 0, 0])
