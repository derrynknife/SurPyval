"""Properties of ``interval_logrank`` (the R checks are in
``tests/reference/test_interval_logrank.py``)."""

import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.nonparametric.logrank import _logrank_z_v

inf = np.inf

# The AML maintenance data (R's survival::aml).
AML_X = np.array(
    [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
    + [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45],
    dtype=float,
)
AML_C = np.array(
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1] + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
)
AML_Z = np.array([1] * 11 + [2] * 12)

XL = np.array([0, 2, 4, 1, 5, 6, 8, 7, 3, 3, 0, 9], dtype=float)
XR = np.array([3, 5, 6, 4, 9, 10, 12, 11, 3, inf, 2, inf])
Z = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0])


@pytest.mark.parametrize("weighting", ["log-rank"])
def test_sun_scores_on_right_censored_data_are_observed_minus_expected(
    weighting,
):
    """For exact and right censored data Sun's score sums are the
    log-rank test's observed-minus-expected events."""
    res = sp.interval_logrank(AML_X, AML_Z, c=AML_C)
    z, _, _ = _logrank_z_v(
        AML_X, AML_Z, AML_C, None, np.unique(AML_Z), weighting, 0, 0
    )
    assert_allclose(list(res.U.values()), z, rtol=1e-8)


def test_the_data_forms_agree():
    """xl/xr, a two-column x with flags, and one-column x with c = -1, 1
    for the one-sided rows give the same test."""
    a = sp.interval_logrank(xl=XL, xr=XR, Z=Z)
    x2 = np.column_stack([XL, XR])
    c = np.where(XL == XR, 0, 2)
    c[np.isinf(XR)] = 1
    x2[np.isinf(XR), 1] = XL[np.isinf(XR)]
    b = sp.interval_logrank(x2, Z, c=c)
    # (0, 2] is left censored at 2 for positive times.
    xl = XL.copy()
    xl[10] = -inf
    d = sp.interval_logrank(xl=xl, xr=XR, Z=Z)
    for other in (b, d):
        assert_allclose(other.statistic, a.statistic, rtol=1e-10)
        assert_allclose(other.p_value, a.p_value, rtol=1e-10)


def test_counts_are_copies():
    n = np.array([1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1])
    a = sp.interval_logrank(xl=XL, xr=XR, Z=Z, n=n)
    b = sp.interval_logrank(
        xl=np.repeat(XL, n), xr=np.repeat(XR, n), Z=np.repeat(Z, n)
    )
    assert_allclose(a.statistic, b.statistic, rtol=1e-8)
    assert_allclose(list(a.U.values()), list(b.U.values()), rtol=1e-8)
    pa = sp.interval_logrank(
        xl=XL, xr=XR, Z=Z, n=n, method="permutation", random_state=3
    )
    pb = sp.interval_logrank(
        xl=np.repeat(XL, n),
        xr=np.repeat(XR, n),
        Z=np.repeat(Z, n),
        method="permutation",
        random_state=3,
    )
    assert pa.p_value == pb.p_value


@pytest.mark.parametrize("scores", ["log-rank", "finkelstein", "wilcoxon"])
def test_invariant_to_row_order_and_labels(scores):
    a = sp.interval_logrank(xl=XL, xr=XR, Z=Z, scores=scores)
    order = np.random.default_rng(0).permutation(XL.size)
    labels = np.where(Z[order] == 0, "y", "x")
    b = sp.interval_logrank(
        xl=XL[order], xr=XR[order], Z=labels, scores=scores
    )
    assert_allclose(b.statistic, a.statistic, rtol=1e-8)
    # Two groups: the score sums are equal and opposite.
    assert_allclose(sum(a.U.values()), 0.0, atol=1e-8)


def test_invariant_to_a_monotone_change_of_time():
    a = sp.interval_logrank(xl=XL, xr=XR, Z=Z, scores="wilcoxon")
    b = sp.interval_logrank(
        xl=np.exp(XL / 3), xr=np.exp(XR / 3), Z=Z, scores="wilcoxon"
    )
    assert_allclose(b.statistic, a.statistic, rtol=1e-8)


def test_r_score_names_are_accepted():
    for r_name, name in (
        ("logrank1", "log-rank"),
        ("logrank2", "finkelstein"),
        ("wmw", "wilcoxon"),
    ):
        a = sp.interval_logrank(xl=XL, xr=XR, Z=Z, scores=r_name)
        b = sp.interval_logrank(xl=XL, xr=XR, Z=Z, scores=name)
        assert a.scores == name
        assert a.statistic == b.statistic


def test_permutation_is_reproducible_and_follows_the_global_seed():
    kw = dict(xl=XL, xr=XR, Z=Z, method="permutation", n_permutations=499)
    a = sp.interval_logrank(**kw, random_state=7)
    b = sp.interval_logrank(**kw, random_state=7)
    assert a.p_value == b.p_value
    assert a.n_permutations == 499
    np.random.seed(11)
    c = sp.interval_logrank(**kw)
    np.random.seed(11)
    d = sp.interval_logrank(**kw)
    assert c.p_value == d.p_value
    # (1 + count) / (1 + B): never 0, at most 1.
    assert 1 / 500 <= a.p_value <= 1


def test_permutation_p_value_is_close_to_the_asymptotic_one():
    rng = np.random.default_rng(231)
    t = rng.weibull(1.5, 120) * np.repeat([10.0, 13.0], 60)
    xl = np.floor(t / 2) * 2
    xr = xl + 2
    g = np.repeat(["a", "b"], 60)
    asym = sp.interval_logrank(xl=xl, xr=xr, Z=g)
    perm = sp.interval_logrank(
        xl=xl, xr=xr, Z=g, method="permutation", random_state=1
    )
    assert perm.statistic == asym.statistic
    assert abs(perm.p_value - asym.p_value) < 0.02


def test_size_under_the_null():
    """200 null data sets (fixed seed): the 5% test rejects between 1%
    and 10% of them."""
    rng = np.random.default_rng(2026)
    rejected = 0
    for _ in range(200):
        t = rng.weibull(1.2, 40) * 10
        start = rng.uniform(0, 3, 40)
        xl = np.maximum(np.floor((t - start) / 3) * 3 + start, 0)
        xl = np.where(t < start, 0, xl)
        xr = np.where(t < start, start, xl + 3)
        g = rng.permutation(np.repeat([0, 1], 20))
        rejected += sp.interval_logrank(xl=xl, xr=xr, Z=g).p_value < 0.05
    assert 0.01 <= rejected / 200 <= 0.10


def test_nothing_to_compare():
    # Every unit right censored at the same time: every score is equal.
    res = sp.interval_logrank(
        xl=[5.0, 5, 5, 5], xr=[inf, inf, inf, inf], Z=[0, 0, 1, 1]
    )
    assert (res.statistic, res.dof, res.p_value) == (0.0, 0, 1.0)


def test_three_groups_dof():
    g = np.array(["a", "b", "c"] * 4)
    res = sp.interval_logrank(xl=XL, xr=XR, Z=g)
    assert res.dof == 2
    assert set(res.U) == {"a", "b", "c"}


def test_to_dict_is_strict_json():
    res = sp.interval_logrank(
        xl=XL, xr=XR, Z=Z, method="permutation", random_state=0
    )
    d = res.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["dof"] == 1 and d["n_permutations"] == 9999
    assert [u["group"] for u in d["U"]] == [0, 1]
    assert "Permutations" in repr(res)
    assert "Wilcoxon" in repr(
        sp.interval_logrank(xl=XL, xr=XR, Z=Z, scores="wilcoxon")
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(tl=1.0), "'tl'"),
        (dict(tr=20.0), "'tr'"),
        (dict(scores="gehan"), "'scores'"),
        (dict(method="pclt"), "'method'"),
        (dict(method="permutation", n_permutations=0), "'n_permutations'"),
        (dict(method="permutation", n_permutations=2.5), "'n_permutations'"),
    ],
)
def test_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sp.interval_logrank(xl=XL, xr=XR, Z=Z, **kwargs)


def test_infinite_truncation_is_no_truncation():
    a = sp.interval_logrank(xl=XL, xr=XR, Z=Z, tl=-inf, tr=inf)
    b = sp.interval_logrank(xl=XL, xr=XR, Z=Z)
    assert a.statistic == b.statistic


def test_group_refusals():
    with pytest.raises(ValueError, match="'Z'"):
        sp.interval_logrank(xl=XL, xr=XR)
    with pytest.raises(ValueError, match="'Z'"):
        sp.interval_logrank(xl=XL, xr=XR, Z=Z[:-1])
    with pytest.raises(ValueError, match="'Z'"):
        sp.interval_logrank(xl=XL, xr=XR, Z=np.where(Z == 0, np.nan, 1.0))
    with pytest.raises(ValueError, match="two groups"):
        sp.interval_logrank(xl=XL, xr=XR, Z=np.zeros(XL.size))


def test_non_convergence_warns():
    with pytest.warns(UserWarning, match="max_iter"):
        sp.interval_logrank(xl=XL, xr=XR, Z=Z, max_iter=1)


def test_logrank_points_to_the_interval_test():
    c = AML_C.copy()
    c[0] = 2
    x = np.column_stack([AML_X, AML_X])
    x[0, 1] += 1
    with pytest.raises(ValueError, match="interval_logrank"):
        sp.logrank(x, AML_Z, c=c)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.interval_logrank(x, AML_Z, c=c)
