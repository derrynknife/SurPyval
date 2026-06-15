"""
Tests for delayed-entry (left-truncation) handling in the nonparametric
recurrent risk set.

``RecurrentEventData.to_xrd`` used to build the at-risk set assuming every
item enters observation at ``t=0``, ignoring each item's left-truncation
bound ``tl``. A delayed-entry item must only join the risk set once
observation begins at its ``tl``; counting it earlier gives the wrong risk
set and a biased MCF (and the same error propagated to ``CauseSpecificMCF``,
which shares this risk set).
"""

import numpy as np

from surpyval import RecurrentEventData
from surpyval.recurrent.competing_risks import CauseSpecificMCF
from surpyval.recurrent.nonparametric import NonParametricCounting
from surpyval.utils.recurrent_utils import handle_xicn


def _delayed_entry_data():
    # Item 1 enters at t=0 with events at 2, 5 and is right-censored at 10.
    # Item 2 has a *delayed entry* at t=4 with events at 6, 8 and is
    # right-censored at 12. Item 2 must not be in the risk set before t=4.
    x = np.array([2.0, 5.0, 10.0, 6.0, 8.0, 12.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    c = np.array([0, 0, 1, 0, 0, 1])
    tl = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 4.0])
    return x, i, c, tl


def test_to_xrd_respects_left_truncation():
    x, i, c, tl = _delayed_entry_data()
    data = handle_xicn(x, i, c=c, tl=tl, as_recurrent_data=True)

    x_unique, r, d = data.to_xrd()

    # Risk set: item 2 only enters at t=4, so at t=2 only item 1 is at risk.
    np.testing.assert_array_equal(x_unique, [2.0, 5.0, 6.0, 8.0, 10.0, 12.0])
    np.testing.assert_array_equal(r, [1, 2, 2, 2, 2, 1])
    np.testing.assert_array_equal(d, [1, 1, 1, 1, 0, 0])


def test_left_truncation_changes_risk_set_vs_naive_entry():
    # Without the fix every item is assumed present from t=0, so the risk set
    # at the first event would be 2 instead of 1. Confirm the delayed entry
    # genuinely shrinks the early risk set relative to the all-enter-at-0 case.
    x, i, c, tl = _delayed_entry_data()

    truncated = handle_xicn(x, i, c=c, tl=tl, as_recurrent_data=True)
    naive = handle_xicn(x, i, c=c, as_recurrent_data=True)

    _, r_trunc, _ = truncated.to_xrd()
    _, r_naive, _ = naive.to_xrd()

    # Same event grid, but the delayed-entry item is excluded before t=4.
    assert r_trunc[0] == 1
    assert r_naive[0] == 2
    assert np.all(r_trunc <= r_naive)


def test_mcf_uses_truncated_risk_set():
    x, i, c, tl = _delayed_entry_data()
    model = NonParametricCounting.fit(x, i, c=c, tl=tl)

    # MCF = cumsum(d / r) over the truncation-aware risk set.
    expected = np.cumsum(
        np.array([1, 1, 1, 1, 0, 0]) / np.array([1, 2, 2, 2, 2, 1])
    )
    np.testing.assert_allclose(model.mcf_hat, expected)
    # Non-decreasing, as an MCF must be.
    assert np.all(np.diff(model.mcf_hat) >= -1e-12)


def test_default_fit_unchanged_without_truncation():
    # The default (no tl) path must be byte-for-byte the old behaviour: every
    # item enters at the origin.
    x = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    c = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])

    data = RecurrentEventData(x, i, c, np.ones_like(x))
    x_unique, r, d = data.to_xrd()

    np.testing.assert_array_equal(x_unique, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(r, [2, 2, 2, 2, 2])


def test_cause_specific_mcf_shares_truncated_risk_set():
    # The cause-specific MCF reuses the shared risk set, so left truncation
    # must flow through to every cause.
    x = np.array([2.0, 5.0, 10.0, 6.0, 8.0, 12.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    c = np.array([0, 0, 1, 0, 0, 1])
    e = ["a", "b", None, "a", "b", None]
    tl = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 4.0])

    model = CauseSpecificMCF.fit(x, i, c=c, e=e, tl=tl)

    # The shared at-risk grid matches the truncation-aware to_xrd.
    np.testing.assert_array_equal(
        model.x, [2.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    )
    np.testing.assert_array_equal(model.r, [1, 2, 2, 2, 2, 1])


def test_cause_specific_mcf_scalar_truncation_broadcasts():
    # A scalar tl is broadcast across all rows (handle_xicn is bypassed for
    # marked data, so the fit must broadcast itself).
    x = np.array([3.0, 7.0, 4.0, 8.0])
    i = np.array([1, 1, 2, 2])
    c = np.array([0, 1, 0, 1])
    e = ["a", None, "a", None]

    model = CauseSpecificMCF.fit(x, i, c=c, e=e, tl=2.0)
    assert np.all(model.data.tl == 2.0)
