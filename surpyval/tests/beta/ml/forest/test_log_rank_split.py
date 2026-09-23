import numpy as np
import pytest

from surpyval.beta.ml.forest.log_rank_split import (
    at_risk_on_grid,
    log_rank,
    log_rank_split,
)
from surpyval.utils.surpyval_data import SurpyvalData


def log_rank_split_xZc(x, Z, c, min_leaf_failures, feature_indices_in):
    """Call log_rank_split() from x-Z-c arrays."""
    data = SurpyvalData(
        np.asarray(x, dtype=float), np.asarray(c), group_and_sort=False
    )
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return log_rank_split(
        data,
        Z,
        min_leaf_samples=1,
        min_leaf_failures=min_leaf_failures,
        feature_indices_in=feature_indices_in,
    )


def test_log_rank_split_one_binary_feature():
    """Simplest case. One feature and two samples."""
    x = np.array(
        [10, 12, 8, 9, 11, 12, 13, 9, 10, 10]
        + [50, 60, 40, 45, 55, 60, 65, 45, 50, 50]
    )
    Z = np.array([0] * 10 + [1] * 10)
    c = np.array([0] * len(x))

    lrs = log_rank_split_xZc(
        x,
        Z,
        c,
        min_leaf_failures=6,
        feature_indices_in=[0],
    )

    # Assert feature 0 (the only feature) is returned
    assert lrs[0] == 0

    # Assert feature 0 value 0 (left children have Z_0 <= 0)
    assert lrs[1] == 0


def test_log_rank_split_one_feature_four_samples():
    x = np.array([10, 11] + [24, 25])
    Z = np.array([0, 0.2] + [15.1, 15])
    c = np.array([0] * len(x))

    lrs = log_rank_split_xZc(
        x,
        Z,
        c,
        min_leaf_failures=1,
        feature_indices_in=[0],
    )

    assert lrs[0] == 0
    assert lrs[1] == 0


def test_log_rank_split_two_features_two_samples():
    """
    Idea is to have two features, one with basically no predictive ability
    and one with plenty of predictive ability.
    """
    x = np.array([15, 75])
    Z = np.array([[0.3, 1], [0.3, 3]])  # Feature 1 should be selected
    c = np.array([0] * len(x))

    lrs = log_rank_split_xZc(
        x,
        Z,
        c,
        min_leaf_failures=1,
        feature_indices_in=[0, 1],
    )

    assert lrs[0] == 1
    assert lrs[1] == 1


def test_log_rank_split_min_leaf_failures():
    """
    Make sure there are min_leaf_failures, not just min leaf samples, at the
    leaf.
    """
    # Case A: a split is possible
    min_leaf_failures = 3
    x = np.array([15, 17, 16, 75, 78, 77])
    Z = np.array([0, 0.1, 0, 3, 3.1, 3.2])
    c_A = np.array([0] * len(x))

    lrsA = log_rank_split_xZc(
        x,
        Z,
        c_A,
        min_leaf_failures=min_leaf_failures,
        feature_indices_in=[0],
    )
    assert lrsA[0] == 0
    assert lrsA[1] == 0.1

    # Case B: all samples are censored, a split is not possible
    c_B = np.array([1] * len(x))
    lrsA = log_rank_split_xZc(
        x, Z, c_B, min_leaf_failures=min_leaf_failures, feature_indices_in=[0]
    )
    assert lrsA[0] == -1
    assert lrsA[1] == float("-Inf")

    # Case C: Not enough uncensored samples to make a split
    c_C = np.array([0, 1, 0, 0, 0, 0])
    lrsA = log_rank_split_xZc(
        x, Z, c_C, min_leaf_failures=min_leaf_failures, feature_indices_in=[0]
    )
    assert lrsA[0] == -1
    assert lrsA[1] == float("-Inf")


# ---------------------------------------------------------------------------
# The statistic itself (#287)
# ---------------------------------------------------------------------------
# The tests above check which split is chosen. These check the value the
# choice is made on, which was wrong by factors of several: the left
# child's at-risk count was expanded onto the pooled event-time grid by
# forward-filling its own risk ladder, which carried a count past the
# deaths and censorings that had already left, and kept a phantom at risk
# for ever after a child ending in a censored observation.


def _log_rank_of_split(xl, cl, xr, cr, tll=None, tlr=None):
    """``log_rank`` for an explicit left/right partition."""
    x = np.array(list(xl) + list(xr), dtype=float)
    c = np.array(list(cl) + list(cr), dtype=int)
    tl = np.array(
        list(tll if tll is not None else [-np.inf] * len(xl))
        + list(tlr if tlr is not None else [-np.inf] * len(xr)),
        dtype=float,
    )
    t = np.vstack([tl, np.full(x.size, np.inf)]).T
    data = SurpyvalData(x=x, c=c, t=t, group_and_sort=False)
    Z = np.array([[0.0]] * len(xl) + [[1.0]] * len(xr))
    return log_rank(0, 0.0, data, Z)


def _log_rank_by_definition(xl, cl, xr, cr, tll=None, tlr=None):
    """The same statistic, summed straight from its definition.

    Deliberately naive: at each event time count who is at risk by
    testing every observation, rather than maintaining a ladder. Slow,
    and independent of the code under test.
    """
    xl, cl, xr, cr = map(np.asarray, (xl, cl, xr, cr))
    tll = np.asarray(
        tll if tll is not None else [-np.inf] * len(xl), dtype=float
    )
    tlr = np.asarray(
        tlr if tlr is not None else [-np.inf] * len(xr), dtype=float
    )
    numerator = 0.0
    variance = 0.0
    for tau in np.unique(np.concatenate([xl, xr])):
        # (entry, exit]: entering exactly at an event time is not at risk
        # for that event.
        Y_L = ((xl >= tau) & (tll < tau)).sum()
        Y = Y_L + ((xr >= tau) & (tlr < tau)).sum()
        d_L = ((xl == tau) & (cl == 0)).sum()
        d = d_L + ((xr == tau) & (cr == 0)).sum()
        if Y <= 1:
            continue
        numerator += d_L - Y_L * d / Y
        variance += (Y_L / Y) * (1 - Y_L / Y) * ((Y - d) / (Y - 1)) * d
    if variance <= 0:
        # No separation to measure -- e.g. every event time has the whole
        # risk set on one side. The caller skips these.
        return float("nan")
    return abs(numerator) / np.sqrt(variance)


@pytest.mark.parametrize(
    "xl, cl, xr, cr, tll, tlr",
    [
        # The two cases reported in #287, checked there against
        # lifelines' logrank_test. Before the fix these came out as
        # 1.856 and 0.276 -- note they were not merely wrong but
        # swapped in rank, so the split with the weaker separation
        # looked like the stronger one.
        ([1.0, 3.0, 4.0], [0, 0, 1], [2.0, 5.0], [0, 0], None, None),
        ([1.0, 2.0], [0, 1], [3.0, 4.0, 5.0], [0, 0, 0], None, None),
        # A child whose last observation is censored: the tail used to
        # subtract only the deaths, leaving someone at risk for ever.
        ([1.0, 4.0], [0, 1], [2.0, 5.0], [0, 1], None, None),
        # Left truncation, where at-risk is (entry, exit] rather than
        # everyone with a later event time.
        (
            [1.0, 3.0, 4.0],
            [0, 0, 1],
            [2.0, 5.0],
            [0, 0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0],
        ),
        # Ties, including a death time shared across both children.
        ([2.0, 2.0, 5.0], [0, 0, 1], [2.0, 3.0], [0, 0], None, None),
        # A child with no observation before the other's first event, so
        # the pooled grid starts outside its range.
        ([6.0, 7.0], [0, 0], [1.0, 2.0, 8.0], [0, 0, 1], None, None),
    ],
)
def test_log_rank_matches_the_definition(xl, cl, xr, cr, tll, tlr):
    assert _log_rank_of_split(xl, cl, xr, cr, tll, tlr) == pytest.approx(
        _log_rank_by_definition(xl, cl, xr, cr, tll, tlr)
    )


def test_log_rank_matches_the_definition_on_random_partitions():
    rng = np.random.default_rng(287)
    for _ in range(200):
        n_l = int(rng.integers(2, 12))
        n_r = int(rng.integers(2, 12))
        xl = rng.integers(1, 12, n_l).astype(float)
        xr = rng.integers(1, 12, n_r).astype(float)
        cl = rng.integers(0, 2, n_l)
        cr = rng.integers(0, 2, n_r)
        if (cl == 0).sum() + (cr == 0).sum() < 2:
            continue  # no events, statistic undefined
        mine = _log_rank_of_split(xl, cl, xr, cr)
        theirs = _log_rank_by_definition(xl, cl, xr, cr)
        if not np.isfinite(theirs):
            continue
        assert mine == pytest.approx(theirs), (xl, cl, xr, cr)


def test_at_risk_count_matches_the_shared_xrd_conversion():
    """``at_risk_on_grid`` must agree with ``xcnt_to_xrd``.

    The pooled ``Y`` in the statistic comes from ``to_xrd``, and the
    left child's ``Y_L`` from ``at_risk_on_grid``. If the two disagree
    about what "at risk" means -- most easily under truncation -- then
    ``Y_L / Y`` is not a proportion of anything.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_obs = int(rng.integers(5, 40))
        x = rng.integers(1, 15, n_obs).astype(float)
        c = rng.integers(0, 2, n_obs)
        tl = np.where(
            rng.random(n_obs) < 0.5,
            0.0,
            np.maximum(0.0, x - rng.integers(1, 5, n_obs)),
        )
        t = np.vstack([tl, np.full(n_obs, np.inf)]).T
        data = SurpyvalData(x=x, c=c, t=t, group_and_sort=False)
        grid, r, _ = data.to_xrd()
        assert at_risk_on_grid(data, grid) == pytest.approx(r)
