"""Metamorphic invariances of the fits (#379).

Each property rewrites the fixture in a way that should not change the
answer, refits, and compares every function of the two models:

- **units**: times multiplied by ``K`` give the same model in the new
  unit -- probabilities at ``K x`` equal those at ``x``, rates are
  divided by ``K``, quantiles multiplied by it ("fits depended on the
  data's units");
- **row order**: a permutation of the data rows (the input-order bugs);
- **counts**: a count ``n`` gives what that many repeated rows give;
- **covariate origin**: a constant added to a covariate column (and to
  the query rows) changes nothing: for every model fitted with
  ``center=True`` (the baseline at the covariate means), and by default
  (the baseline at Z = 0) for the Cox and Fine-Gray partial likelihoods,
  which see only differences within a risk set, and the log-linear
  parametric families whose baseline maps exactly between the two
  (#459, #463).
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
    BASELINES,
    CASES,
    cases_for,
    fitted,
    predictions,
    refit,
)
from surpyval.univariate.regression._fit_skeleton import ORIGIN_MAPS

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


# Every model whose fit takes ``center``: those whose covariates enter
# through a partial likelihood, which is invariant to the origin of each
# covariate, and the parametric regressions (#459, #463).
_PARTIAL_LIKELIHOOD = (
    "CoxPH",
    "CoxPH[strata]",
    "CompetingRisksProportionalHazards[Cox]",
    "CompetingRisksProportionalHazards[Fine-Gray]",
    "FineGray",
)
_PARAMETRIC = tuple(
    base + kind for kind in ("PH", "AFT", "PO", "AH") for base in BASELINES
)
# The parametric families whose baseline maps exactly between the
# covariate means and Z = 0 (ORIGIN_MAPS): by default they are the same
# model wherever the covariates' zero is. For the others (and the additive
# hazards models) the origin is part of the default model, whose baseline
# is at Z = 0; with center=True it is at the means for every family.
_KIND = {
    "Proportional Hazard": "PH",
    "Accelerated Failure Time": "AFT",
    "Proportional Odds": "PO",
}
_MAPPED = tuple(
    name
    for name in (dist + _KIND[kind] for kind, dist in ORIGIN_MAPS)
    if name in _PARAMETRIC
)
# Far enough from 0 that exp(beta'Z) on the raw values overflows: only a
# baseline at the means (center=True) is representable there.
_SHIFTS = (1e5, -3e4)
# Near enough that the default baseline at Z = 0 is representable.
_MODERATE_SHIFTS = (30.0, -20.0)


def _origin_cases(names):
    return [
        pytest.param(
            c,
            id=c.name,
            marks=[pytest.mark.slow] if c.is_slow("units") else [],
        )
        for c in CASES
        if c.name in names
    ]


def _moved(case, data, shifts):
    shift = np.array(shifts[: np.shape(data["Z"])[1]])
    moved = dict(data, Z=np.asarray(data["Z"], dtype=float) + shift)
    return moved, np.asarray(case.Z, dtype=float) + shift


@pytest.mark.parametrize(
    "case", _origin_cases(_PARTIAL_LIKELIHOOD + _PARAMETRIC)
)
def test_covariate_origin(case):
    # With the baseline at the covariate means, far shifts included.
    data = dict(case.data(), center=True)
    moved, Z = _moved(case, data, _SHIFTS)
    ref = predictions(case, refit(case, data))
    got = predictions(case, refit(case, moved), Z=Z)
    compare(case, got, ref)


@pytest.mark.parametrize("case", _origin_cases(_PARTIAL_LIKELIHOOD + _MAPPED))
def test_covariate_origin_by_default(case):
    # With the baseline at Z = 0 (the default), where that is the same
    # model wherever the origin is and the shift keeps it representable.
    data = case.data()
    moved, Z = _moved(case, data, _MODERATE_SHIFTS)
    ref = predictions(case, fitted(case))
    got = predictions(case, refit(case, moved), Z=Z)
    compare(case, got, ref)
