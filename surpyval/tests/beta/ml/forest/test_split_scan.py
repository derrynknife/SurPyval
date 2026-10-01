"""The split searches give the same splits as the per-candidate
reference implementations they replaced (#193, #190, #518).

``log_rank_split`` scores every threshold of a feature from cumulative
at-risk and death counts in one pass; it must reproduce, bit for bit, the
statistic :func:`log_rank` computes for each candidate subset, and so
choose exactly the same split, ties included.
"""

import numpy as np
import pytest

from surpyval.beta.ml.forest.log_rank_split import (
    _LogRankScan,
    log_rank,
    log_rank_split,
)
from surpyval.utils.surpyval_data import SurpyvalData


def reference_log_rank_split(
    data, Z, min_leaf_samples, min_leaf_failures, feature_indices_in
):
    """The per-candidate search the scan replaced: a new subset and a
    full log-rank per threshold (with #193's n-weighted failures)."""
    event_weight = data.n * (data.c != 1)
    best, best_u, best_v = -np.inf, -1, -np.inf
    for u in feature_indices_in:
        Z_u = Z[:, u]
        for v in np.unique(Z_u):
            mask = Z_u <= v
            if (
                mask.sum() < min_leaf_samples
                or (~mask).sum() < min_leaf_samples
                or event_weight[mask].sum() < min_leaf_failures
                or event_weight[~mask].sum() < min_leaf_failures
            ):
                continue
            value = log_rank(u, v, data, Z)
            if value > best:
                best, best_u, best_v = value, u, v
    return best_u, best_v


def _case(seed):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(5, 120))
    Z = rng.uniform(0, 1, (N, int(rng.integers(1, 4))))
    if seed % 3 == 0:
        Z = np.round(Z * 4) / 4  # tied feature values
    if seed % 5 == 0:
        Z = np.column_stack([Z, Z[:, 0]])  # a repeated column: exact ties
    x = rng.exponential(10, N) * np.exp(Z[:, 0])
    if seed % 2 == 0:
        x = np.round(x) + 1  # tied times
    c = (rng.uniform(size=N) < 0.3).astype(int)
    n = rng.integers(1, 4, N) if seed % 7 == 0 else None
    t = None
    if seed % 4 == 1:  # delayed entry
        tl = rng.uniform(0, 0.8, N) * x
        tl[rng.uniform(size=N) < 0.5] = 0
        t = np.column_stack([tl, np.full(N, np.inf)])
    data = SurpyvalData(x, c, n, t, group_and_sort=False)
    return data, Z, int(rng.integers(1, 6)), int(rng.integers(0, 4))


@pytest.mark.parametrize("seed", range(60))
def test_scan_matches_per_candidate_search(seed):
    data, Z, mls, mlf = _case(seed)
    features = list(range(Z.shape[1]))
    new = log_rank_split(data, Z, mls, mlf, features)
    old = reference_log_rank_split(data, Z, mls, mlf, features)
    assert (int(new[0]), float(new[1])) == (int(old[0]), float(old[1]))


@pytest.mark.parametrize("seed", range(0, 60, 7))
@pytest.mark.parametrize("block", [None, 5])
def test_scan_statistics_bit_identical(seed, block, monkeypatch):
    import surpyval.beta.ml.forest.log_rank_split as module

    if block is not None:
        # Blocks of a few rows: the carried running totals
        monkeypatch.setattr(module, "_SCAN_BLOCK", block)
    data, Z, _, _ = _case(seed)
    Z_u = Z[:, 0]
    values = np.unique(Z_u)[:-1]  # both children non-empty
    if values.size == 0:
        pytest.skip("one feature value")
    order = np.argsort(Z_u, kind="stable")
    n_left = np.searchsorted(Z_u[order], values, side="right")
    scan = _LogRankScan(data).statistics(order, n_left)
    direct = np.array([log_rank(0, v, data, Z) for v in values])
    np.testing.assert_array_equal(
        scan, np.where(np.isnan(direct), -np.inf, direct)
    )
