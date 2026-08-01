"""``group_xcnt`` -- collapsing duplicate ``(x, c, t)`` rows.

This used to be a triple-nested ``defaultdict`` loop over every
observation, which dominated fitting cost at scale. The vectorised
replacement has to reproduce it *exactly*, including the group ordering:
``xcnt_sort`` runs straight afterwards and is a stable sort, so any rows
tying on its keys keep whatever order grouping produced. The original
implementation is kept here as the oracle so the two cannot drift.
"""

from collections import defaultdict

import numpy as np
import pytest

from surpyval.utils import group_xcnt, xcnt_sort

INF = np.inf


def dict_form(x, c, n, t):
    """The original implementation, verbatim, as the reference."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if x.ndim == 2:
        for vx, vc, vn, vt in zip(x, c, n, t):
            grouped[tuple(vx)][vc][tuple(vt)] += vn
    else:
        for vx, vc, vn, vt in zip(x, c, n, t):
            grouped[vx][vc][tuple(vt)] += vn

    x_out, c_out, n_out, t_out = [], [], [], []
    for xv, level2 in grouped.items():
        for cv, level3 in level2.items():
            for tv, nv in level3.items():
                x_out.append(xv)
                c_out.append(cv)
                n_out.append(nv)
                t_out.append(tv)
    return np.array(x_out), np.array(c_out), np.array(n_out), np.array(t_out)


def assert_matches_oracle(x, c, n, t):
    expected = dict_form(x, c, n, t)
    actual = group_xcnt(x, c, n, t)
    for field, want, got in zip("xcnt", expected, actual):
        np.testing.assert_array_equal(
            got, want, err_msg=f"field {field!r} differs"
        )
    # The count dtype is load-bearing: integer counts must stay integer
    # (np.bincount would hand back float64).
    assert actual[2].dtype == expected[2].dtype


def test_docstring_case():
    x = np.array([1, 1, 3, 3])
    c = np.zeros(4, dtype=int)
    n = np.ones(4, dtype=int)
    t = np.vstack([np.full(4, -INF), np.full(4, INF)]).T
    x_g, c_g, n_g, _ = group_xcnt(x, c, n, t)
    np.testing.assert_array_equal(x_g, [1, 3])
    np.testing.assert_array_equal(c_g, [0, 0])
    np.testing.assert_array_equal(n_g, [2, 2])


def test_preserves_order_when_sort_keys_tie():
    # The subtle one. These three rows share x and c, and all have
    # t.min() == -inf, so they tie on every key `xcnt_sort` uses and it
    # cannot reorder them. Their order is therefore whatever grouping
    # produced -- a plain sorted np.unique would give 5, 7, 10.
    x = np.array([1.0, 1.0, 1.0])
    c = np.zeros(3, dtype=int)
    n = np.ones(3, dtype=int)
    t = np.column_stack([np.full(3, -INF), [10.0, 5.0, 7.0]])
    assert_matches_oracle(x, c, n, t)
    _, _, _, t_out = xcnt_sort(*group_xcnt(x, c, n, t))
    np.testing.assert_array_equal(t_out[:, 1], [10.0, 5.0, 7.0])


def test_preserves_nested_x_major_order():
    # The dictionary iterated x-major, so (x=1, c=1) precedes (x=2, c=0)
    # even though the latter appears first in the input.
    x = np.array([1.0, 2.0, 1.0])
    c = np.array([0, 0, 1])
    n = np.ones(3, dtype=int)
    t = np.column_stack([np.full(3, -INF), np.full(3, INF)])
    assert_matches_oracle(x, c, n, t)
    x_out, c_out, _, _ = group_xcnt(x, c, n, t)
    assert list(zip(x_out, c_out)) == [(1.0, 0), (1.0, 1), (2.0, 0)]


EDGE_CASES = {
    "single row": (
        [4.0],
        [0],
        [1],
        [[-INF, INF]],
    ),
    "all identical": (
        [2.0] * 5,
        [0] * 5,
        [1] * 5,
        [[-INF, INF]] * 5,
    ),
    "no duplicates": (
        [1.0, 2.0, 3.0],
        [0, 0, 0],
        [1, 1, 1],
        [[-INF, INF]] * 3,
    ),
    "same x different c": (
        [1.0, 1.0, 1.0],
        [0, 1, -1],
        [1, 1, 1],
        [[-INF, INF]] * 3,
    ),
    "same x different tl": (
        [3.0, 3.0],
        [0, 0],
        [1, 1],
        [[0.0, INF], [1.0, INF]],
    ),
    "weights above one": (
        [1.0, 1.0, 2.0],
        [0, 0, 0],
        [3, 4, 5],
        [[-INF, INF]] * 3,
    ),
    # nan entry times are accepted upstream; nan != nan so each row stays
    # its own group, and the replacement must agree.
    "nan truncation": (
        [1.0, 1.0, 3.0],
        [0, 0, 0],
        [1, 1, 1],
        [[np.nan, INF]] * 3,
    ),
    "nan x": (
        [np.nan, np.nan, 3.0],
        [0, 0, 0],
        [1, 1, 1],
        [[-INF, INF]] * 3,
    ),
}


@pytest.mark.parametrize(
    "case", list(EDGE_CASES.values()), ids=list(EDGE_CASES)
)
def test_edge_cases_match_oracle(case):
    x, c, n, t = case
    assert_matches_oracle(
        np.asarray(x, dtype=float),
        np.asarray(c, dtype=int),
        np.asarray(n, dtype=np.int64),
        np.asarray(t, dtype=float),
    )


def test_interval_censored_two_dimensional_x():
    xl = np.array([1.0, 1.0, 2.0, 2.0])
    xr = np.array([3.0, 3.0, 4.0, 5.0])
    x = np.column_stack([xl, xr])
    c = np.full(4, 2)
    n = np.ones(4, dtype=np.int64)
    t = np.column_stack([np.full(4, -INF), np.full(4, INF)])
    assert_matches_oracle(x, c, n, t)


def test_float_counts_stay_float():
    x = np.array([1.0, 1.0, 2.0])
    c = np.zeros(3, dtype=int)
    n = np.array([0.5, 1.5, 2.0])
    t = np.column_stack([np.full(3, -INF), np.full(3, INF)])
    _, _, n_out, _ = group_xcnt(x, c, n, t)
    assert n_out.dtype == n.dtype
    np.testing.assert_array_equal(n_out, [2.0, 2.0])


def test_randomised_differential_sweep():
    # Every shape of input the handler can produce, checked against the
    # original implementation.
    rng = np.random.default_rng(90210)
    for _ in range(400):
        n_obs = int(rng.integers(1, 40))
        if rng.random() < 0.25:
            lo = rng.choice(np.arange(1.0, 6.0), size=n_obs)
            x = np.column_stack(
                [lo, lo + rng.choice([0.5, 1.0, 2.0], size=n_obs)]
            )
            c = rng.choice([0, 1, -1, 2], size=n_obs)
        else:
            x = rng.choice(np.arange(1.0, 6.0), size=n_obs)
            c = rng.choice([0, 1, -1], size=n_obs)
        n = rng.integers(1, 4, size=n_obs).astype(np.int64)
        t = np.column_stack(
            [
                rng.choice([-INF, 0.0, 0.5], size=n_obs),
                rng.choice([INF, 5.0, 7.0, 10.0], size=n_obs),
            ]
        )
        assert_matches_oracle(x, c, n, t)


def test_totals_are_conserved():
    rng = np.random.default_rng(11)
    x = rng.choice(np.arange(1.0, 5.0), size=200)
    c = rng.choice([0, 1], size=200)
    n = rng.integers(1, 6, size=200).astype(np.int64)
    t = np.column_stack([np.full(200, -INF), np.full(200, INF)])
    _, _, n_out, _ = group_xcnt(x, c, n, t)
    assert n_out.sum() == n.sum()


def test_scales_to_large_samples():
    # 200k distinct observations took ~4 s under the Python loop.
    n_obs = 200_000
    x = np.arange(1.0, n_obs + 1.0)
    c = np.zeros(n_obs, dtype=int)
    n = np.ones(n_obs, dtype=np.int64)
    t = np.column_stack([np.full(n_obs, -INF), np.full(n_obs, INF)])
    x_out, _, n_out, _ = group_xcnt(x, c, n, t)
    assert x_out.shape == (n_obs,)
    assert n_out.sum() == n_obs
