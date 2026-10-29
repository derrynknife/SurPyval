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


# -- Peto-Peto weights and the t0 cut-off ------------------------------------

# The AML maintenance data (R's survival::aml): group 1 maintained.
_AML_X = np.array(
    [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
    + [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45],
    float,
)
_AML_C = np.array(
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1] + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
)
_AML_Z = np.array([1] * 11 + [2] * 12)


def _two_group_by_hand(x, c, Z, weight):
    """The two-sample weighted log-rank statistic written out from Klein
    and Moeschberger (2003), eq. 7.3.3-7.3.4, with ``weight(r, d, S_peto)``
    at each distinct event time; an oracle independent of logrank."""
    num, var = _two_group_parts(x, c, Z, weight)
    return num**2 / var


def _two_group_parts(x, c, Z, weight):
    """The numerator (weighted observed less expected) and variance of
    :func:`_two_group_by_hand`, with the Peto survival formed from these
    rows alone."""
    times = np.unique(x[c == 0])
    num, var, s_peto = 0.0, 0.0, 1.0
    for t in times:
        r = float(np.sum(x >= t))
        r1 = float(np.sum((x >= t) & (Z == Z.min())))
        d = float(np.sum((x == t) & (c == 0)))
        d1 = float(np.sum((x == t) & (c == 0) & (Z == Z.min())))
        s_peto *= 1 - d / (r + 1)
        w = weight(r, d, s_peto)
        num += w * (d1 - d * r1 / r)
        if r > 1:
            var += w**2 * d * (r1 / r) * (1 - r1 / r) * (r - d) / (r - 1)
    return num, var


def test_peto_peto_matches_lifelines_on_aml():
    # lifelines 0.30.3: logrank_test(..., weightings="peto") on the AML
    # data gives 2.7080350199857874, p = 0.099843906985543.
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, weighting="peto-peto")
    np.testing.assert_allclose(res.statistic, 2.7080350199857874, rtol=1e-10)
    np.testing.assert_allclose(res.p_value, 0.099843906985543, rtol=1e-10)
    assert res.weighting == "peto-peto"


@pytest.mark.parametrize(
    "weighting, weight",
    [
        ("peto-peto", lambda r, d, s: s),
        ("modified-peto-peto", lambda r, d, s: s * r / (r + 1)),
    ],
)
def test_peto_weights_match_the_formula_by_hand(weighting, weight):
    expected = _two_group_by_hand(_AML_X, _AML_C, _AML_Z, weight)
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, weighting=weighting)
    np.testing.assert_allclose(res.statistic, expected, rtol=1e-12)


def test_peto_peto_on_lung_three_groups_matches_lifelines():
    # lifelines 0.30.3, multivariate_logrank_test(weightings="peto") on
    # the lung data by ECOG score 0, 1, 2: 20.26560446072507 on 2 df.
    df = sp.datasets.load_lung().dropna(subset=["ph.ecog"])
    df = df[df["ph.ecog"] < 3]
    res = sp.logrank(
        df["time"], df["ph.ecog"], c=1 - df["status"], weighting="peto-peto"
    )
    np.testing.assert_allclose(res.statistic, 20.26560446072507, rtol=1e-9)
    assert res.dof == 2


@pytest.mark.parametrize(
    "t0, statistic, p_value",
    # lifelines 0.30.3, logrank_test(..., t_0=t0) on the AML data.
    [
        (30, 2.1301545637467254, 0.14442619197316434),
        (23, 0.7934106317369176, 0.3730708137573371),
    ],
)
def test_t0_matches_lifelines_on_aml(t0, statistic, p_value):
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, t0=t0)
    np.testing.assert_allclose(res.statistic, statistic, rtol=1e-10)
    np.testing.assert_allclose(res.p_value, p_value, rtol=1e-10)


def test_t0_with_peto_peto_matches_lifelines():
    # lifelines 0.30.3: weightings="peto", t_0=30: 2.2437492037616167.
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, weighting="peto-peto", t0=30)
    np.testing.assert_allclose(res.statistic, 2.2437492037616167, rtol=1e-10)


def test_t0_is_censoring_at_t0_by_hand():
    # An event exactly at t0 stays an event; later ones are censored.
    x = np.minimum(_AML_X, 23.0)
    c = np.where(_AML_X > 23, 1, _AML_C)
    by_hand = sp.logrank(x, _AML_Z, c=c)
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, t0=23)
    assert res.statistic == by_hand.statistic
    # Past the last time it changes nothing.
    whole = sp.logrank(_AML_X, _AML_Z, c=_AML_C)
    late = sp.logrank(_AML_X, _AML_Z, c=_AML_C, t0=1000)
    assert late.statistic == whole.statistic


def test_t0_drops_the_rows_that_enter_after_it():
    tl = np.zeros(_AML_X.size)
    tl[[10, 21]] = [40, 35]  # enter after the window
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, tl=tl, t0=30)
    keep = tl < 30
    by_hand = sp.logrank(
        _AML_X[keep], _AML_Z[keep], c=_AML_C[keep], tl=tl[keep], t0=30
    )
    assert res.statistic == by_hand.statistic


def test_t0_with_strata_and_counts():
    strata = np.arange(_AML_X.size) % 2
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, strata=strata, t0=30)
    x = np.minimum(_AML_X, 30.0)
    c = np.where(_AML_X > 30, 1, _AML_C)
    by_hand = sp.logrank(x, _AML_Z, c=c, strata=strata)
    assert res.statistic == by_hand.statistic
    assert res.strata == 2


@pytest.mark.parametrize("t0", ["soon", np.nan, None])
def test_t0_must_be_a_number(t0):
    if t0 is None:
        sp.logrank(_AML_X, _AML_Z, c=_AML_C, t0=t0)  # the default
        return
    with pytest.raises(ValueError, match="'t0'"):
        sp.logrank(_AML_X, _AML_Z, c=_AML_C, t0=t0)


def test_t0_does_not_hide_interval_censoring():
    c = _AML_C.copy()
    c[0] = -1
    with pytest.raises(ValueError, match="right censored"):
        sp.logrank(_AML_X, _AML_Z, c=c, t0=30)


# -- gaps found by the 0.26 test audit ---------------------------------------


def test_scalar_entry_time_with_t0_is_the_same_entry_for_every_row():
    """Audit C10. A scalar ``tl`` is the entry time of every row, also
    when the comparison is cut at ``t0``."""
    tl = np.full(_AML_X.size, 0.5)
    scalar = sp.logrank(_AML_X, _AML_Z, c=_AML_C, tl=0.5, t0=30)
    array = sp.logrank(_AML_X, _AML_Z, c=_AML_C, tl=tl, t0=30)
    assert scalar.statistic == array.statistic
    assert scalar.p_value == array.p_value
    assert scalar.dof == array.dof


def test_t0_drops_a_row_entering_exactly_at_t0():
    """Audit C10. A row with ``tl >= t0`` is never at risk before ``t0``
    and is dropped; at the boundary ``tl == t0`` too. The others keep
    the scalar entry time."""
    tl = np.full(_AML_X.size, 0.5)
    tl[10] = 30.0  # x = 161, enters exactly at t0
    res = sp.logrank(_AML_X, _AML_Z, c=_AML_C, tl=tl, t0=30)
    keep = np.arange(_AML_X.size) != 10
    rest = sp.logrank(
        _AML_X[keep], _AML_Z[keep], c=_AML_C[keep], tl=0.5, t0=30
    )
    assert res.statistic == rest.statistic


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": [1, 2, 3, 4], "Z": [0, 0, 1, 1], "c": [1, 1, 1, 1]},
        {"x": _AML_X, "Z": _AML_Z, "c": _AML_C, "tl": 30, "t0": 30},
    ],
)
def test_nothing_left_to_compare_has_zero_degrees_of_freedom(kwargs):
    """Found while closing audit C10. As
    ``test_logrank_without_informative_groups``: with no event to compare,
    the statistic is 0 on 0 degrees of freedom and the p-value 1."""
    res = sp.logrank(**kwargs)
    assert (res.statistic, res.dof, res.p_value) == (0.0, 0, 1.0)


def test_peto_peto_with_counts_is_the_rows_repeated():
    """Audit C11 (a). Peto-Peto weights from counts ``n`` are those of the
    rows repeated ``n`` times: the Peto survival counts each unit."""
    n = 1 + np.arange(_AML_X.size) % 3
    counted = sp.logrank(_AML_X, _AML_Z, c=_AML_C, n=n, weighting="peto-peto")
    repeated = sp.logrank(
        np.repeat(_AML_X, n),
        np.repeat(_AML_Z, n),
        c=np.repeat(_AML_C, n),
        weighting="peto-peto",
    )
    assert counted.statistic == repeated.statistic
    assert counted.p_value == repeated.p_value


def test_stratified_peto_peto_resets_the_peto_survival_per_stratum():
    """Audit C11 (b). The stratified Peto-Peto statistic is (sum of the
    strata's numerators)^2 / (sum of their variances), with each stratum's
    weights the Peto survival of that stratum's rows alone (Klein and
    Moeschberger 2003, section 7.5), worked by hand."""
    strata = np.arange(_AML_X.size) % 2
    num, var = 0.0, 0.0
    for s in (0, 1):
        rows = strata == s
        num_s, var_s = _two_group_parts(
            _AML_X[rows], _AML_C[rows], _AML_Z[rows], lambda r, d, S: S
        )
        num += num_s
        var += var_s
    res = sp.logrank(
        _AML_X, _AML_Z, c=_AML_C, strata=strata, weighting="peto-peto"
    )
    np.testing.assert_allclose(res.statistic, num**2 / var, rtol=1e-12)
    assert res.strata == 2
