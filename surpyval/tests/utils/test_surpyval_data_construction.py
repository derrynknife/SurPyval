"""Building a ``SurpyvalData`` gives the same data, faster.

Two costs dominated building one from 100,000 rows, and through it every
fit's set-up:

- the distinct truncation windows, and the distinct non-right-censored
  values ``_check_identifiable`` counts, came from ``np.unique(...,
  axis=0)``, which sorts the rows as structured records; ``unique_pairs``
  ranks each column and then the pairs of ranks as integers;
- a list input was converted to an array several times over (to look
  for durations, for missing values, to read floats); it is now
  converted once (``_numeric_list_as_array``) where numpy reads it as
  flat numbers.

``np.unique(axis=0)`` is kept here as the reference, and an array input as
the reference for a list.
"""

import datetime as dt

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.utils.data_formats import _numeric_list_as_array
from surpyval.utils.numeric import unique_pairs
from surpyval.utils.surpyval_data import SurpyvalData

INF = np.inf

ATTRIBUTES = [
    "x", "c", "n", "t",
    "x_o", "n_o", "x_r", "n_r", "x_l", "n_l",
    "x_il", "x_ir", "n_i",
    "x_tl", "x_tr", "n_t",
    "tl_unique", "tr_unique", "n_t_unique",
    "mask_o", "mask_r", "mask_l", "mask_i", "truncated_mask",
]  # fmt: skip


def _reference_pairs(a, b):
    unique, inverse = np.unique(
        np.column_stack([a, b]), axis=0, return_inverse=True
    )
    return unique[:, 0], unique[:, 1], inverse.ravel()


@pytest.mark.parametrize("seed", range(20))
def testunique_pairs_is_numpys_unique_rows(seed):
    rng = np.random.default_rng(seed)
    pool = np.array([-INF, INF, 0.0, -0.0, 1.0, 2.5, 1e300, -3.0])
    size = int(rng.integers(1, 200))
    a = rng.choice(pool[: rng.integers(1, pool.size + 1)], size)
    b = rng.choice(pool[: rng.integers(1, pool.size + 1)], size)
    for got, want in zip(unique_pairs(a, b), _reference_pairs(a, b)):
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype


def testunique_pairs_of_nothing():
    a, b, inverse = unique_pairs(np.array([]), np.array([]))
    assert a.size == b.size == inverse.size == 0


def testunique_pairs_keeps_each_nan_row_apart():
    # as np.unique(axis=0) does, since nan != nan
    a = np.array([np.nan, np.nan, 1.0, 1.0])
    b = np.array([1.0, 1.0, np.nan, 2.0])
    pairs = unique_pairs(a, b)
    assert pairs[0].size == _reference_pairs(a, b)[0].size == 4
    np.testing.assert_array_equal(
        np.column_stack(pairs[:2])[pairs[2]], np.column_stack([a, b])
    )


def _random_xcnt(rng, size):
    x = rng.integers(0, 8, size) + 0.5
    width = rng.choice([0, 1, 2], size)
    c = np.where(width > 0, 2, rng.choice([-1, 0, 1], size))
    return dict(
        x=np.column_stack([x, x + width]),
        c=c,
        n=rng.integers(1, 4, size),
        t=np.column_stack(
            [
                rng.choice([-INF, 0.0, 0.25], size),
                rng.choice([INF, 60.0, 100.0], size),
            ]
        ),
    )


@pytest.mark.parametrize("seed", range(10))
def test_the_truncation_windows_are_as_numpys_unique_rows(seed):
    data = SurpyvalData(**_random_xcnt(np.random.default_rng(seed), 300))
    tl, tr, inverse = _reference_pairs(data.x_tl, data.x_tr)
    np.testing.assert_array_equal(data.tl_unique, tl)
    np.testing.assert_array_equal(data.tr_unique, tr)
    np.testing.assert_array_equal(
        data.n_t_unique, np.bincount(inverse, weights=data.n_t)
    )


def _as_lists(kwargs, container):
    return {
        k: (
            container(v.tolist())
            if v.ndim == 1
            else [container(row) for row in v.tolist()]
        )
        for k, v in kwargs.items()
    }


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("seed", range(5))
def test_a_list_input_gives_the_data_its_array_gives(seed, container):
    rng = np.random.default_rng(seed)
    arrays = _random_xcnt(rng, 300)
    arrays["tl"], arrays["tr"] = arrays.pop("t").T
    arrays["xl"], arrays["xr"] = arrays.pop("x").T
    from_arrays = SurpyvalData(**arrays)
    from_lists = SurpyvalData(**_as_lists(arrays, container))
    for name in ATTRIBUTES:
        got, want = getattr(from_lists, name), getattr(from_arrays, name)
        np.testing.assert_array_equal(got, want, err_msg=name)
        assert got.dtype == want.dtype, name


@pytest.mark.parametrize(
    "value",
    [
        [1, 2.5, 3],
        (1, 2, 3),
        [True, False],
        [np.float32(1.5), np.float32(2.5)],
    ],
)
def test_flat_numeric_lists_become_arrays(value):
    arr = _numeric_list_as_array(value)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.asarray(value))


@pytest.mark.parametrize(
    "value",
    [
        [1, [2, 3], 4],  # a scalar among pairs
        [[1, 2], [3, 4]],  # pairs: the handler reads them row by row
        [[1], [2]],
        [1, None, 3],  # a missing value, said as missing
        [1, 2, 2**70],  # beyond int64
        ["1", "2"],
        [dt.timedelta(days=1)],  # refused as a duration
        [np.timedelta64(1, "D")],
        5,
        np.array([1, 2]),
        None,
    ],
)
def test_anything_else_is_left_as_given(value):
    assert _numeric_list_as_array(value) is value


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(x=[1, 2, 3], n=[1, None, 1]), "'n' cannot contain NaN"),
        (dict(x=[1, 2, 3], c=[0, None, 1]), "'c' cannot contain NaN"),
        (dict(x=[1, 2, float("nan")]), "'x' cannot contain NaN"),
        (dict(x=[dt.timedelta(days=1)]), "'x' holds durations"),
        (dict(x=[1, 2], tl=[np.timedelta64(1, "D")] * 2), "'tl' holds"),
        (dict(x=[1, 2], tl=[0, float("nan")]), "must not contain NaN"),
        (dict(x=[[1, 2, 3], [4, 5, 6]]), "no more than length 2"),
        (dict(xl=[1, 2], xr=[2]), "the same length"),
    ],
)
def test_list_inputs_are_refused_as_before(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SurpyvalData(**kwargs)


@pytest.mark.parametrize(
    "x, c, refused",
    [
        ([[1, 2], [1, 2], [1, 2]], [2, 2, 2], True),
        ([[1, 2], [3, 4], [3, 4]], [2, 2, 2], False),
        ([10, 10, 12, 12], [0, 0, 1, 1], True),
        ([10, 10, 12, 12], [0, 0, -1, 1], False),
    ],
)
def test_the_identifiability_check_counts_distinct_values(x, c, refused):
    if refused:
        with pytest.raises(ValueError, match="distinct non-right-censored"):
            Weibull.fit(x, c)
    else:
        Weibull.fit(x, c)
