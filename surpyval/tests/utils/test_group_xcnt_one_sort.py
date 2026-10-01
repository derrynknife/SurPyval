"""``group_xcnt`` groups with one sort of the full key (#515).

It sorted three times per call -- the full key, ``x`` alone and
``(x, c)`` -- and a fit calls it three times: 36% of a tied Weibull fit at
1e5 rows. The ``x`` and ``(x, c)`` groups are prefixes of the full key, so
they are now read off the one sort. The three-sort version is kept here as
the reference: the grouping, its order and the counts must be identical,
and so must the fits built on them.
"""

import importlib

import numpy as np
import pytest

from surpyval import LogNormal, Weibull
from surpyval.utils import group_xcnt

utils_module = importlib.import_module("surpyval.utils")
INF = np.inf


def _group_ids(key):
    order = np.lexsort(key.T[::-1])
    ordered = key[order]
    starts = np.empty(len(ordered), dtype=bool)
    if len(ordered) > 0:
        starts[0] = True
    if len(ordered) > 1:
        starts[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    group = np.empty(len(ordered), dtype=np.intp)
    group[order] = np.cumsum(starts) - 1
    return group, order[starts]


def _three_sorts(x, c, n, t):
    """``group_xcnt`` before #515, verbatim, as the reference."""
    leading = x if x.ndim == 1 else x[:, 0]
    if np.unique(leading).size == leading.size:
        return x, c, n, t
    x_columns = x.reshape(-1, 1) if x.ndim == 1 else x
    group, first_full = _group_ids(np.column_stack([x_columns, c, t]))
    by_x, first_x = _group_ids(x_columns)
    by_xc, first_xc = _group_ids(np.column_stack([x_columns, c]))
    representative = first_full
    order = np.lexsort(
        (
            first_full,
            first_xc[by_xc[representative]],
            first_x[by_x[representative]],
        )
    )
    representative = representative[order]
    totals = np.bincount(group, weights=n, minlength=first_full.size)[order]
    totals = totals.astype(n.dtype)
    return x[representative], c[representative], totals, t[representative]


def _assert_same(got, want):
    for field, a, b in zip("xcnt", got, want):
        assert a.dtype == b.dtype, field
        np.testing.assert_array_equal(a, b, err_msg=field)


def _random_case(rng):
    size = int(rng.integers(2, 300))
    two_column = rng.random() < 0.25
    if two_column:
        lo = rng.choice(np.arange(1.0, 8.0), size=size)
        x = np.column_stack([lo, lo + rng.choice([0.0, 0.5, 1.0], size=size)])
    else:
        x = rng.choice(np.arange(1.0, 1.0 + rng.integers(1, 30)), size=size)
    # Constant or varying c and truncation, so the sort drops or keeps
    # each column.
    c = rng.choice([0, 1, -1, 2][: 4 if two_column else 3], size=size)
    if rng.random() < 0.4:
        c = np.zeros(size, dtype=int)
    tl = rng.choice([-INF, 0.0, 0.5], size=size)
    tr = rng.choice([INF, 20.0, 30.0], size=size)
    if rng.random() < 0.5:
        tl = np.full(size, -INF)
    if rng.random() < 0.5:
        tr = np.full(size, INF)
    t = np.column_stack([tl, tr])
    if rng.random() < 0.1:
        t[rng.integers(0, size), 0] = np.nan
    if rng.random() < 0.1 and not two_column:
        x[rng.integers(0, size, 2)] = np.nan
    n = rng.integers(1, 5, size=size).astype(np.int64)
    if rng.random() < 0.2:
        n = rng.uniform(0.5, 3.0, size)
    return x, c, n, t


def test_matches_the_three_sort_version():
    rng = np.random.default_rng(515)
    for _ in range(600):
        case = _random_case(rng)
        _assert_same(group_xcnt(*case), _three_sorts(*case))


@pytest.mark.parametrize(
    "x,c,t",
    [
        # Every row identical: no column varies.
        ([2.0] * 4, [0] * 4, [[-INF, INF]] * 4),
        # Only c varies.
        ([2.0] * 4, [0, 1, 0, 1], [[-INF, INF]] * 4),
        # Only the right truncation varies (the order-sensitive case).
        ([1.0] * 3, [0] * 3, [[-INF, 10.0], [-INF, 5.0], [-INF, 7.0]]),
        # -0.0 and 0.0 are one value.
        ([1.0, 1.0], [0, 0], [[-0.0, INF], [0.0, INF]]),
    ],
)
def test_degenerate_keys(x, c, t):
    case = (
        np.asarray(x),
        np.asarray(c),
        np.ones(len(x), dtype=np.int64),
        np.asarray(t, dtype=float),
    )
    _assert_same(group_xcnt(*case), _three_sorts(*case))


def test_sorts_the_key_once(monkeypatch):
    # The regression: one sort of the key and one to order the groups
    # (the three-sort version made four lexsort calls).
    calls = []
    lexsort = np.lexsort

    def counting(keys, *args, **kwargs):
        calls.append(1)
        return lexsort(keys, *args, **kwargs)

    monkeypatch.setattr(np, "lexsort", counting)
    rng = np.random.default_rng(0)
    x = rng.choice(np.arange(1.0, 20.0), 500)
    c = rng.choice([0, 1], 500)
    t = np.column_stack([rng.choice([-INF, 0.0], 500), np.full(500, INF)])
    group_xcnt(x, c, np.ones(500, dtype=np.int64), t)
    assert len(calls) <= 2


@pytest.mark.parametrize("dist", [Weibull, LogNormal])
def test_tied_fits_bit_identical(monkeypatch, dist):
    rng = np.random.default_rng(1)
    x = np.ceil(rng.weibull(2.0, 3000) * 100) / 10
    c = (rng.uniform(size=3000) < 0.2).astype(int)
    tl = rng.choice([0.0, 0.05], 3000)
    new = dist.fit(x, c, tl=tl)
    with monkeypatch.context() as patch:
        patch.setattr(utils_module, "group_xcnt", _three_sorts)
        old = dist.fit(x, c, tl=tl)
    np.testing.assert_array_equal(new.params, old.params)
    for field in "xcnt":
        np.testing.assert_array_equal(new.data[field], old.data[field])
