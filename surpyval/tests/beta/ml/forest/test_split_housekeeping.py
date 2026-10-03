"""Split-machinery housekeeping (#193).

1. ``log_rank_split`` refuses data with no risk sets (left or interval
   censoring, right truncation) itself, not only through the tree's
   ``parse_kind``.
2. ``min_leaf_failures`` counts failures ``n``-weighted in every split --
   the log-rank, Turnbull-score and conditional-inference paths as the
   deviance split always has -- so a row with count ``n`` and ``n``
   identical rows are split alike.
"""

import numpy as np
import pytest

from surpyval.beta.ml.forest.conditional_inference import ctree_select
from surpyval.beta.ml.forest.deviance_split import deviance_split
from surpyval.beta.ml.forest.log_rank_split import log_rank_split
from surpyval.beta.ml.forest.turnbull_score_split import (
    turnbull_score_split,
)
from surpyval.utils.surpyval_data import SurpyvalData


@pytest.mark.parametrize(
    "x, c, t",
    [
        ([[1, 2], [2, 3], [3, 4], [4, 5]], [2, 2, 2, 2], None),
        ([1, 2, 3, 4], [-1, 0, 0, 0], None),
        ([1, 2, 3, 4], [0, 0, 0, 0], [[0, 10]] * 4),
    ],
    ids=["interval", "left-censored", "right-truncated"],
)
def test_log_rank_split_refuses_data_without_risk_sets(x, c, t):
    data = SurpyvalData(x, c, t=t, group_and_sort=False)
    Z = np.arange(4.0).reshape(-1, 1)
    with pytest.raises(ValueError, match="no risk sets"):
        log_rank_split(data, Z, 1, 1, [0])


def _counted():
    # Two early-failing rows of count 5 against four later single rows:
    # by rows the left child has 2 failures, n-weighted it has 10.
    x = np.array([1.0, 2.0, 10.0, 11.0, 12.0, 13.0])
    n = np.array([5, 5, 1, 1, 1, 1])
    Z = np.array([[0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    return x, n, Z


def _expanded(x, n, Z, c=None):
    c = np.zeros(x.size, dtype=int) if c is None else c
    idx = np.repeat(np.arange(x.size), n)
    return x[idx], c[idx], Z[idx]


def test_log_rank_min_leaf_failures_is_n_weighted():
    x, n, Z = _counted()
    data = SurpyvalData(x, n=n, group_and_sort=False)
    # 10 n-weighted failures on the left: the split is allowed (the row
    # count, 2, used to refuse it).
    assert log_rank_split(data, Z, 1, 3, [0]) == (0, 0.0)
    # ... as the deviance split always allowed it
    assert deviance_split(data, Z, 1, 3, [0], model="exponential") == (
        0,
        0.0,
    )
    # and as the expanded rows are split
    xe, ce, Ze = _expanded(x, n, Z)
    expanded = SurpyvalData(xe, ce, group_and_sort=False)
    assert log_rank_split(expanded, Ze, 1, 3, [0]) == (0, 0.0)


def test_turnbull_min_leaf_failures_is_n_weighted():
    x, n, Z = _counted()
    c = np.array([2, 2, 2, 2, 2, 2])
    xi = np.column_stack([x - 0.5, x + 0.5])
    data = SurpyvalData(xi, c, n, group_and_sort=False)
    assert turnbull_score_split(data, Z, 1, 3, [0]) == (0, 0.0)


def test_ctree_min_leaf_failures_is_n_weighted():
    x, n, Z = _counted()
    data = SurpyvalData(x, n=n, group_and_sort=False)
    chosen, p_value = ctree_select(data, Z, "non-parametric", 1, 3, [0])
    assert chosen == 0
    assert p_value < 1.0


def test_count_and_expanded_rows_split_alike():
    # A row with count n is n identical rows (Design Principle 5): the
    # failure constraint used to count the collapsed data's rows.
    rng = np.random.default_rng(3)
    x = np.round(rng.exponential(10, 30), 1) + 0.1
    c = (rng.uniform(size=30) < 0.3).astype(int)
    n = rng.integers(1, 4, 30)
    Z = rng.uniform(0, 1, (30, 2))
    data = SurpyvalData(x, c, n, group_and_sort=False)
    xe, ce, Ze = _expanded(x, n, Z, c)
    expanded = SurpyvalData(xe, ce, group_and_sort=False)
    for min_leaf_failures in [1, 6, 12]:
        # min_leaf_samples counts rows, so it is left at 1 here
        assert log_rank_split(
            data, Z, 1, min_leaf_failures, [0, 1]
        ) == log_rank_split(expanded, Ze, 1, min_leaf_failures, [0, 1])
