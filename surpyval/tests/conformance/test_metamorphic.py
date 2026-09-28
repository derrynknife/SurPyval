"""Metamorphic invariances of the fits (#379).

Each property rewrites the fixture in a way that should not change the
answer, refits, and compares every function of the two models:

- **units**: times multiplied by ``K`` give the same model in the new
  unit -- probabilities at ``K x`` equal those at ``x``, rates are
  divided by ``K``, quantiles multiplied by it ("fits depended on the
  data's units");
- **row order**: a permutation of the data rows (the input-order bugs);
- **counts**: a count ``n`` gives what that many repeated rows give.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.checks import (
    compare,
    expanded,
    permuted,
    rescaled,
)
from surpyval.tests.conformance.registry import (
    cases_for,
    fitted,
    predictions,
    refit,
)

K = 7.3  # an awkward unit change, so it cannot hide in rounding


@pytest.mark.parametrize("case", cases_for("units"))
def test_change_of_units(case):
    ref = predictions(case, fitted(case))
    model = refit(case, rescaled(case, case.data(), K))
    got = predictions(case, model, x=case.x * K)
    compare(case, got, ref, scale=K)


@pytest.mark.parametrize("case", cases_for("row_order"))
def test_data_row_order(case):
    ref = predictions(case, fitted(case))
    got = predictions(case, refit(case, permuted(case, case.data())))
    compare(case, got, ref)


@pytest.mark.parametrize("case", cases_for("counts"))
def test_counts_equal_repeated_rows(case):
    data = case.data()
    assert np.any(np.asarray(data["n"]) > 1), "the fixture needs counts"
    ref = predictions(case, fitted(case))
    got = predictions(case, refit(case, expanded(case, data)))
    compare(case, got, ref)
