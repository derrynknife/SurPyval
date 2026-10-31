"""Permuting a long query permutes the result, bit for bit (principle 4).

``test_vectorisation.test_query_order_equivariance`` permutes each case's
short default query. A prediction that sums with a BLAS matrix product
(``B @ g``, ``np.dot``) can give a result that depends on where a query
point sits in the matrix: the remainder blocks of the product are summed
in a different order, so on Linux x86-64 a permuted query can differ in
the last bit. Short queries do not show it. The Royston-Parmar models were
fixed this way (``utils.linalg.fixed_order_dot``, and
``univariate/regression/test_royston_parmar_query_order.py``); this
module asks the same of every case, at query lengths that put the points
in different remainder blocks (17, 33, 64 and 129).

A failure here on one platform and not another is a real finding (a
prediction whose value depends on its neighbours in the query), not a
flaky test: the comparison is exact and nothing in it is random but the
seeded permutations.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    call,
    calls,
    cases_for,
    fitted,
    query,
)

CASES = cases_for("query_order")
LENGTHS = (17, 33, 64, 129)
PERMUTATIONS = 2
# The two cases whose predictions are slowest (a posterior integral, a
# joint model's marker path) run their longest query in the full suite
# only; every other case runs every length on a pull request.
_SLOW_AT_129 = frozenset({"Weibull[bayes]", "WeibullJointPH"})


def _params():
    return [
        pytest.param(
            *ps.values,
            length,
            id=f"{ps.id}-{length}",
            marks=list(ps.marks)
            + (
                [pytest.mark.slow]
                if length == 129 and ps.id in _SLOW_AT_129
                else []
            ),
        )
        for ps in CASES
        for length in LENGTHS
    ]


def test_the_cases_are_found():
    ids = {ps.id for ps in CASES}
    for name in ("AalenAdditive", "WeibullDistributional", "RoystonParmar"):
        assert name in ids, name
    assert len(CASES) >= 150


def _tiled(case, fname, length):
    x0 = np.asarray(query(case, fname))
    x = x0[np.arange(length) % len(x0)]
    Z = None
    if case.Z is not None and fname != "qf":
        Z = np.asarray(case.Z)[np.arange(length) % len(case.Z)]
    return x, Z


@pytest.mark.parametrize("case, length", _params())
def test_long_query_order_equivariance(case, length):
    model = fitted(case)
    rng = np.random.default_rng(20261030)
    for fname, event in calls(case):
        x, Z = _tiled(case, fname, length)
        ref = np.asarray(call(case, model, fname, x, Z, event), float)
        for _ in range(PERMUTATIONS):
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
            np.testing.assert_array_equal(
                got,
                ref[perm],
                err_msg=f"{case.name}.{fname}[{event}], length {length}",
            )
