"""Royston-Parmar predictions do not depend on where a query point sits.

The spline at the query times was a matrix-vector product (``B @
gamma``), which BLAS sums in an order that depends on a row's place in
the matrix: on Linux x86-64 a permuted query gave results that differed
in the last bit (``test_query_order_equivariance[RoystonParmar]``, and,
at longer queries, the regressions too). The sums now run in a fixed
order per point (``utils.linalg.fixed_order_dot``). The tests here use
longer queries than the conformance suite's, where the remainder blocks
of the product fall on different rows."""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    call,
    calls,
    cases_for,
    fitted,
    query,
)
from surpyval.utils.linalg import fixed_order_dot

NAMES = ("RoystonParmar", "RoystonParmarPH", "RoystonParmarPO")
CASES = [ps for ps in cases_for("query_order") if ps.id in NAMES]


def test_the_cases_are_found():
    assert sorted(ps.id for ps in CASES) == sorted(NAMES)


@pytest.mark.parametrize("length", [17, 33, 64])
@pytest.mark.parametrize("case", CASES)
def test_long_query_order_equivariance(case, length):
    model = fitted(case)
    rng = np.random.default_rng(20261029)
    for fname, event in calls(case):
        x0 = np.asarray(query(case, fname))
        x = x0[np.arange(length) % len(x0)]
        Z = None
        if case.Z is not None and fname != "qf":
            Z = np.asarray(case.Z)[np.arange(length) % len(case.Z)]
        ref = np.asarray(call(case, model, fname, x, Z, event), float)
        for _ in range(3):
            perm = rng.permutation(length)
            got = np.asarray(
                call(
                    case,
                    model,
                    fname,
                    x[perm],
                    None if Z is None else Z[perm],
                    event,
                ),
                float,
            )
            np.testing.assert_array_equal(got, ref[perm], err_msg=fname)


def test_royston_parmar_band_is_order_equivariant():
    (ps,) = (ps for ps in CASES if ps.id == "RoystonParmar")
    model = fitted(ps.values[0])
    x = np.tile(np.asarray(ps.values[0].x), 6)[:64]
    ref = model.cb(x)
    perm = np.random.default_rng(1).permutation(x.size)
    np.testing.assert_array_equal(model.cb(x[perm]), ref[perm])


def test_fixed_order_dot_sums_each_row_alone():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(65, 5))
    g = rng.normal(size=5)
    out = fixed_order_dot(B, g)
    # Left to right, one row at a time
    by_row = np.array(
        [
            (((r[0] * g[0] + r[1] * g[1]) + r[2] * g[2]) + r[3] * g[3])
            + r[4] * g[4]
            for r in B
        ]
    )
    np.testing.assert_array_equal(out, by_row)
    perm = rng.permutation(65)
    np.testing.assert_array_equal(fixed_order_dot(B[perm], g), out[perm])
    np.testing.assert_allclose(out, B @ g, rtol=1e-13, atol=1e-15)
    assert fixed_order_dot(np.zeros((3, 0)), np.zeros(0)).shape == (3,)
