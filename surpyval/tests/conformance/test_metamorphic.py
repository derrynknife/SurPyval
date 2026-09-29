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
  the query rows) changes nothing, for the models where the origin is
  not part of the model: the Cox-based and Fine-Gray partial likelihoods
  see only differences within a risk set (#459, #463), and the
  log-linear parametric families whose baseline maps exactly between
  origins fit on centred covariates (#463).
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


# The models whose covariates enter through a partial likelihood, which
# is invariant to the origin of each covariate (#459, #463), and the
# parametric regressions. Those fit on centred covariates (#463), which
# is exact for the log-linear families whose baseline maps between
# origins; for the others the covariates' origin is part of the model
# (pinned below).
_PARTIAL_LIKELIHOOD = (
    "CoxPH",
    "CoxPH[strata]",
    "CompetingRisksProportionalHazards[Cox]",
    "CompetingRisksProportionalHazards[Fine-Gray]",
    "FineGray",
)
_PARAMETRIC = tuple(
    base + kind
    for kind in ("PH", "AFT", "PO", "AH")
    for base in BASELINES
)
# Far enough from 0 that exp(beta'Z) on the raw values overflows.
_SHIFTS = (1e5, -3e4)
_NOT_CLOSED = (
    "#463: the {} baseline is not closed under the change of origin "
    "(exp(beta'c) times its {} is not a {} {}), so the model "
    "with its baseline at Z = 0 differs from the one at the covariate "
    "means (the maximum log-likelihood moves with a shift of 1 already); "
    "it is fitted as defined, on the covariates as given, and a shift of "
    "1e5 wrecks that fit"
)
_ORIGIN_DEPENDENT: dict[str, str] = {
    **{
        f"{base}PH": _NOT_CLOSED.format(
            base, "cumulative hazard", base, "cumulative hazard"
        )
        for base in ("LogNormal", "Gamma", "Normal", "Logistic")
    },
    **{
        f"{base}PO": _NOT_CLOSED.format(
            base, "survival odds", base, "survival odds"
        )
        for base in ("Weibull", "LogNormal", "Exponential", "Gamma")
        + ("Normal", "Gumbel")
    },
    **{
        f"{base}AH": "#463: the additive hazards models are not centred: "
        "h0(x) + beta'c is not a hazard of the baseline's family (but for "
        "the Exponential, whose positivity bound then moves with c), so "
        "the origin is part of the model, and a shift of 1e5 changes the "
        "fit"
        for base in BASELINES
    },
}


def _origin_cases():
    out = []
    for case in CASES:
        if case.name not in _PARTIAL_LIKELIHOOD + _PARAMETRIC:
            continue
        marks = []
        if case.name in _ORIGIN_DEPENDENT:
            marks.append(
                pytest.mark.xfail(
                    strict=True, reason=_ORIGIN_DEPENDENT[case.name]
                )
            )
        if case.is_slow("units"):
            marks.append(pytest.mark.slow)
        out.append(pytest.param(case, id=case.name, marks=marks))
    return out


@pytest.mark.parametrize("case", _origin_cases())
def test_covariate_origin(case):
    data = case.data()
    shift = np.array(_SHIFTS[: np.shape(data["Z"])[1]])
    moved = dict(data, Z=np.asarray(data["Z"], dtype=float) + shift)
    ref = predictions(case, fitted(case))
    got = predictions(
        case, refit(case, moved), Z=np.asarray(case.Z, dtype=float) + shift
    )
    compare(case, got, ref)
