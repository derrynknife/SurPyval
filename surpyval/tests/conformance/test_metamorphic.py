"""Metamorphic invariances of the fits (#379).

Each property rewrites the fixture in a way that should not change the
answer, refits, and compares every function of the two models:

- **units**: times multiplied by ``K`` give the same model in the new
  unit -- probabilities at ``K x`` equal those at ``x``, rates are
  divided by ``K``, quantiles multiplied by it ("fits depended on the
  data's units");
- **row order**: a permutation of the data rows (the input-order bugs);
- **counts**: a count ``n`` gives what that many repeated rows give;
- **covariate origin**: for the Cox-based models, a constant added to a
  covariate column (and to the query rows) changes nothing: the partial
  likelihood sees only differences within a risk set (#459).
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
    CASES,
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


# The models whose covariates enter through a Cox partial likelihood,
# which is invariant to the origin of each covariate (#459). The
# parametric regressions and Fine-Gray do not centre yet (#463).
_COX_BASED = (
    "CoxPH",
    "CoxPH[strata]",
    "CompetingRisksProportionalHazards[Cox]",
)
# Far enough from 0 that exp(beta'Z) on the raw values overflows.
_SHIFTS = (1e5, -3e4)


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in CASES if c.name in _COX_BASED],
)
def test_covariate_origin(case):
    data = case.data()
    shift = np.array(_SHIFTS[: np.shape(data["Z"])[1]])
    moved = dict(data, Z=np.asarray(data["Z"], dtype=float) + shift)
    ref = predictions(case, fitted(case))
    got = predictions(
        case, refit(case, moved), Z=np.asarray(case.Z, dtype=float) + shift
    )
    compare(case, got, ref)
