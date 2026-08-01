"""``xcnt_to_xrd`` and its at-risk-entry helper.

The entry count used to be built as an ``N x K`` comparison matrix, which
was quadratic in time and memory (a ``MemoryError`` past ~50k
observations). These tests pin the values it must produce -- including the
``(entry, exit]`` convention from #260 -- against the direct matrix form,
so the linear-time replacement cannot drift from it.
"""

import numpy as np
import pytest

from surpyval.utils import _entered_before, xcnt_to_xrd

INF = np.inf


def matrix_form(tl, x, n):
    """The original quadratic expression, kept here as the oracle."""
    return ((tl[:, np.newaxis] < x[np.newaxis, :]) * n[:, np.newaxis]).sum(0)


EDGE_CASES = [
    ("no truncation", [-INF, -INF, -INF], [1.0, 2.0, 3.0], [1, 1, 1]),
    ("no truncation weighted", [-INF, -INF], [1.0, 2.0], [5, 3]),
    ("entry equals event time", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1, 1, 1]),
    ("common entry time", [1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [2, 2, 2]),
    (
        "mixed -inf and finite",
        [-INF, 1.0, -INF, 3.0],
        [2.0, 4.0, 5.0],
        [1, 2, 3, 4],
    ),
    ("ties in entry times", [1.0, 1.0, 2.0, 2.0], [2.0, 3.0], [1, 1, 1, 1]),
    ("entries after all events", [5.0, 6.0], [1.0, 2.0], [1, 1]),
    ("entries before all events", [0.0, 1.0], [9.0, 10.0], [2, 3]),
    ("single observation", [-INF], [1.0], [7]),
    ("large weights", [-INF, 0.0], [1.0, 2.0], [10**6, 10**7]),
    # nan entry times are accepted upstream; `nan < x` is False, and the
    # replacement must reproduce that rather than silently differ.
    ("all nan", [np.nan, np.nan], [1.0, 2.0], [1, 1]),
    ("mixed nan", [0.0, np.nan, 1.0], [1.0, 2.0, 3.0], [1, 2, 3]),
]


@pytest.mark.parametrize(
    "tl,x,n", [c[1:] for c in EDGE_CASES], ids=[c[0] for c in EDGE_CASES]
)
def test_entered_before_matches_matrix_form(tl, x, n):
    tl = np.asarray(tl, dtype=float)
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=np.int64)
    expected = matrix_form(tl, x, n)
    actual = _entered_before(tl, x, n)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_entered_before_matches_matrix_form_randomised():
    # A broad differential sweep: the two forms must agree exactly on
    # every shape of truncation, not just the tidy cases above.
    rng = np.random.default_rng(4242)
    for _ in range(300):
        n_obs = int(rng.integers(1, 25))
        x = np.unique(
            rng.choice(
                np.arange(0.0, 12.0, 0.5), size=int(rng.integers(1, 12))
            )
        )
        n = rng.integers(1, 6, size=n_obs).astype(np.int64)
        mode = rng.integers(0, 4)
        if mode == 0:
            tl = np.full(n_obs, -INF)
        elif mode == 1:
            tl = rng.choice(np.arange(-2.0, 12.0, 0.5), size=n_obs)
        elif mode == 2:
            tl = np.where(
                rng.random(n_obs) < 0.5,
                -INF,
                rng.choice(np.arange(0.0, 12.0, 0.5), size=n_obs),
            )
        else:
            tl = np.full(n_obs, float(rng.choice(np.arange(0.0, 6.0, 0.5))))
        np.testing.assert_array_equal(
            _entered_before(tl, x, n), matrix_form(tl, x, n)
        )


def test_docstring_examples():
    x, r, d = xcnt_to_xrd(np.array([1, 2, 3, 4, 5]), np.array([0, 1, 1, 0, 0]))
    np.testing.assert_array_equal(x, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(r, [5, 4, 3, 2, 1])
    np.testing.assert_array_equal(d, [1, 0, 0, 1, 1])


def test_entry_exit_convention_preserved():
    # (entry, exit]: a subject entering exactly at an event time is NOT at
    # risk for that event (#260). Each subject here enters at the previous
    # subject's event time, so exactly one is at risk at each time.
    x, r, d = xcnt_to_xrd(
        x=np.array([1, 2, 3, 4, 5]), tl=np.array([0, 1, 2, 3, 4])
    )
    np.testing.assert_array_equal(r, [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(d, [1, 1, 1, 1, 1])


def test_return_dtypes():
    x, r, d = xcnt_to_xrd(x=[1.0, 2.0, 3.0], c=[0, 1, 0])
    assert x.dtype == np.float64
    assert r.dtype == int
    assert d.dtype == int


def test_scales_beyond_the_quadratic_wall():
    # 60k distinct observations needed a 28 GB intermediate under the
    # matrix form and raised MemoryError; it must now simply work.
    n_obs = 60_000
    x = np.arange(1.0, n_obs + 1.0)
    x_out, r, d = xcnt_to_xrd(x=x)
    assert x_out.shape == (n_obs,)
    # No censoring or truncation: the risk set falls by one at each time.
    np.testing.assert_array_equal(r, np.arange(n_obs, 0, -1))
    np.testing.assert_array_equal(d, np.ones(n_obs, dtype=int))


def test_scales_with_truncation():
    # The searchsorted branch must scale too, not just the constant one.
    n_obs = 60_000
    x = np.arange(1.0, n_obs + 1.0)
    tl = x - 1.0
    x_out, r, d = xcnt_to_xrd(x=x, tl=tl)
    assert x_out.shape == (n_obs,)
    # Every subject enters at the previous event time, so exactly one is
    # at risk at each event.
    np.testing.assert_array_equal(r, np.ones(n_obs, dtype=int))
