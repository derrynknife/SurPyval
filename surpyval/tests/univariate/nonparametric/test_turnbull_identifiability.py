"""Whether the truncated Turnbull NPMLE exists, decided from the data (#327).

The warnings used to be driven by a tuned cut-off (more than 90% of the
fitted mass on pieces some observation "gains" from). They are now driven
by a verdict computed before the EM runs and exposed as ``model.npmle``;
see ``surpyval/univariate/nonparametric/_turnbull_npmle.py`` for the
derivation. Each case below was also checked against what the EM does
from several starting points (it drifts to the boundary, settles, or
settles in different places with the same likelihood).
"""

import itertools
import time
import warnings

import numpy as np
import pytest

from surpyval import Turnbull
from surpyval.univariate.nonparametric import _turnbull_npmle as nm


def _fit(**kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = Turnbull.fit(
            turnbull_estimator="Kaplan-Meier", max_iter=5000, **kwargs
        )
    return model, [str(w.message) for w in record]


SIX_X = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
SIX_C = np.array([-1, 0, 0, -1, 0, 0])


# -- verdicts on the reproducers ---------------------------------------------


def test_six_point_reproducer_does_not_exist():
    # The #308 reproducer: the piece after the first entry (0.1, 0.28] is
    # inside only the first left-censored support, which ends at 2 while
    # exact failures come later. Nothing penalises the hazard there, so the
    # likelihood rises as it goes to one: the NPMLE does not exist. The EM
    # confirms it: the first window's mass falls tenfold for every tenfold
    # increase in iterations (2.5e-3 at 2,000, 2.5e-4 at 20,000).
    model, messages = _fit(x=SIX_X, c=SIX_C, tl=np.linspace(0.1, 1.0, 6))
    assert model.npmle == "does not exist"
    assert any("NPMLE does not exist" in m for m in messages)
    assert any("0.28" in m for m in messages)
    assert not model.converged


@pytest.mark.parametrize(
    "tl",
    [np.zeros(6), np.full(6, 0.5), None],
    ids=["common_zero", "common_half", "untruncated"],
)
def test_common_entry_exists(tl):
    kwargs = {} if tl is None else {"tl": tl}
    model, messages = _fit(x=SIX_X, c=SIX_C, **kwargs)
    assert model.npmle == "exists"
    assert messages == []


def test_distinct_entries_without_left_censoring_exist():
    model, messages = _fit(
        x=SIX_X, c=np.zeros(6, dtype=int), tl=np.linspace(0.1, 1.0, 6)
    )
    assert model.npmle == "exists"
    assert messages == []


def test_kaplan_meier_hitting_zero_before_a_later_entry_does_not_exist():
    # Everyone at risk at 1.0 fails there, and two units enter at 2: the
    # delayed-entry Kaplan-Meier drops to zero at 1.0. The old code only
    # said the EM "did not converge"; the fit is structurally unattainable.
    model, messages = _fit(x=[0.5, 1.0, 3.0, 4.0], tl=[0, 0, 2, 2])
    assert model.npmle == "does not exist"
    assert any("NPMLE does not exist" in m for m in messages)
    assert not any("did not converge" in m for m in messages)


def test_gap_after_an_open_support_is_not_unique():
    # Nobody entered by 1.5 is known to survive past it (one is censored at
    # 1.0, one is interval censored in (1.1, 5]) and the next unit enters
    # at 1.5. Every support containing the gap runs past the last support
    # start, so the likelihood is flat across it: the maximum is attained
    # but not unique. From four random starts the EM converges to curves
    # 0.023 apart with log-likelihoods equal to 4e-16. The old code said
    # nothing.
    model, messages = _fit(
        x=[[0.5, 0.5], [1.0, 1.0], [1.1, 5.0], [3.0, 3.0], [4.0, 4.0]],
        c=[0, 1, 2, 0, 0],
        tl=[0, 0, 0, 1.5, 2],
    )
    assert model.npmle == "not unique"
    assert model.converged
    assert len(messages) == 1
    assert "not unique" in messages[0] and "t = 1.5" in messages[0]


def test_right_truncation_mirror_does_not_exist():
    # The mirror image of the Kaplan-Meier case: observable only up to 2.5,
    # the two early failures say nothing about mass above 2.5, and the two
    # later failures (observable up to 10) cannot tell mass below 5 from
    # none. The Lynden-Bell estimate is not attained.
    model, messages = _fit(x=[1.0, 2.0, 5.0, 6.0], tr=[2.5, 2.5, 10, 10])
    assert model.npmle == "does not exist"
    assert any("NPMLE does not exist" in m for m in messages)


def test_lynden_bell_healthy_case_exists():
    model, messages = _fit(x=[7, 3, 5, 2, 7, 6], tr=[11, 3, 7, 5, 7, 6])
    assert model.npmle == "exists"
    assert messages == []


def test_disjoint_windows_are_not_unique():
    # Vardi (1985): two groups whose windows never overlap. Each is fitted
    # perfectly, but nothing links their total masses. The old code was
    # silent.
    model, messages = _fit(
        x=[1.0, 1.5, 5.0, 5.5], tl=[0, 0, 4, 4], tr=[2, 2, 6, 6]
    )
    assert model.npmle == "not unique"
    assert len(messages) == 1
    assert "do not overlap" in messages[0]


def test_vardi_connected_but_not_strongly_connected_does_not_exist():
    # Exact data, two-sided windows: the unit failing at 1.0 is observable
    # only on (0, 1.6], which contains the failure at 1.5 but not the later
    # ones, and so on up: each window reaches the next failure but no
    # window reaches back. The old code only reported non-convergence.
    model, messages = _fit(
        x=[1.0, 1.5, 2.0, 2.5], tl=[0, 0.8, 1.2, 1.2], tr=[1.6, 1.6, 3, 3]
    )
    assert model.npmle == "does not exist"
    assert any("NPMLE does not exist" in m for m in messages)


def test_untruncated_interval_data_exists():
    model, messages = _fit(
        x=[[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]], c=[2, 2, 2, 2, 2]
    )
    assert model.npmle == "exists"


def test_verdict_does_not_depend_on_the_estimator():
    # The verdict is a property of the data, computed before the EM.
    for est in ("Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Turnbull.fit(
                x=SIX_X,
                c=SIX_C,
                tl=np.linspace(0.1, 1.0, 6),
                turnbull_estimator=est,
            )
        assert model.npmle == "does not exist"


def test_zero_count_rows_are_ignored():
    # A row with no count adds no term to the likelihood (``fit`` refuses
    # such counts, but the helper is also called directly). Two exact
    # failures, the second entering after the first: a gap before the last
    # entry inside a support that ends before the last support start. A
    # zero-count row that would fill the gap changes nothing.
    lo, hi = np.array([0, 2, 1]), np.array([0, 2, 2])
    wl, wh = np.array([0, 1, 0]), np.array([2, 2, 2])
    assert nm.npmle_existence(lo, hi, wl, wh, np.array([1, 1, 0]), 3)[0] == (
        "does not exist"
    )
    assert nm.npmle_existence(lo, hi, wl, wh, np.array([1, 1, 1]), 3)[0] == (
        "exists"
    )


# -- the helper against brute force ------------------------------------------


def _random_rows(rng, K, N, kind):
    rows = []
    for _ in range(N):
        a, b = sorted(rng.integers(0, K, 2))
        if rng.random() < 0.5:
            b = a
        wl = 0 if kind == "right" else int(rng.integers(0, a + 1))
        wh = K - 1 if kind == "left" else int(rng.integers(b, K))
        rows.append((a, b, wl, wh))
    r = np.array(rows)
    return r[:, 0], r[:, 1], r[:, 2], r[:, 3]


def _condition_e(lo, hi, wl, wh, K):
    # Every non-empty closed set meets every window, by enumeration.
    for size in range(1, K + 1):
        for members in itertools.combinations(range(K), size):
            A = np.isin(np.arange(K), members)
            meets_w = np.array([A[a : b + 1].any() for a, b in zip(wl, wh)])
            meets_s = np.array([A[a : b + 1].any() for a, b in zip(lo, hi)])
            if (meets_w & ~meets_s).any():
                continue
            if (~meets_w).any():
                return False
    return True


def test_stable_run_search_matches_enumeration():
    rng = np.random.default_rng(1)
    checked = 0
    for trial in range(600):
        K = int(rng.integers(2, 7))
        kind = ("left", "right", "both")[trial % 3]
        lo, hi, wl, wh = _random_rows(rng, K, int(rng.integers(1, 6)), kind)
        if not (nm._coverage(lo, hi, K) > 0).all():
            continue
        checked += 1
        assert (not nm._has_stable_run(lo, hi, wl, wh, K)) == _condition_e(
            lo, hi, wl, wh, K
        ), (lo, hi, wl, wh)
    assert checked > 200


def test_one_sided_verdict_agrees_with_condition_e():
    # With one-sided windows, "exists" is exactly condition E.
    rng = np.random.default_rng(5)
    for trial in range(2000):
        K = int(rng.integers(2, 9))
        kind = ("left", "right")[trial % 2]
        lo, hi, wl, wh = _random_rows(rng, K, int(rng.integers(1, 7)), kind)
        if not (nm._coverage(lo, hi, K) > 0).all():
            continue
        if wl.min() != 0 or wh.max() != K - 1:
            continue
        verdict = nm._block(lo, hi, wl, wh, K)[0]
        assert (verdict == "exists") == (
            not nm._has_stable_run(lo, hi, wl, wh, K)
        )


def test_paint_matches_a_direct_reduction():
    rng = np.random.default_rng(3)
    for _ in range(200):
        size = int(rng.integers(1, 40))
        start = rng.integers(0, size, 15)
        end = np.minimum(start + rng.integers(0, size, 15), size - 1)
        value = rng.integers(0, 100, 15)
        got = nm._paint(start, end, value, size, np.minimum, 1000)
        want = np.full(size, 1000)
        for s, e, v in zip(start, end, value):
            want[s : e + 1] = np.minimum(want[s : e + 1], v)
        np.testing.assert_array_equal(got, want)


def test_criterion_is_cheap_on_a_large_sample():
    # The criterion runs on every truncated fit, so it must cost little
    # next to the EM: a few vectorised O((N + M) log M) passes. About 3 ms
    # here, against ~0.5 ms for each of the EM's (up to max_iter)
    # iterations at this size; the bound is loose to stay robust.
    rng = np.random.default_rng(0)
    N, M = 5000, 15000
    lo = np.sort(rng.integers(0, M, N))
    hi = np.minimum(lo + rng.integers(0, 50, N), M - 1)
    wl = np.maximum(lo - rng.integers(0, 50, N), 0)
    wh = np.minimum(hi + rng.integers(0, 50, N), M - 1)
    start = time.perf_counter()
    nm.npmle_existence(lo, hi, wl, wh, np.ones(N), M)
    assert time.perf_counter() - start < 1.0
