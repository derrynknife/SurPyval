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
  (#459, #463);
- **covariate scale**: a covariate column multiplied by a constant (and
  the query rows with it) divides its coefficient by the constant and
  changes no prediction, by ``fit`` and by ``fit_tvc`` (#577).
"""

import numpy as np
import pytest

from surpyval.tests.conformance.checks import (
    compare,
    expanded,
    permuted,
    rescaled,
)
from surpyval.tests.conformance.leaks import quiet
from surpyval.tests.conformance.registry import (
    BASELINES,
    CASES,
    cases_for,
    fitted,
    predictions,
    refit,
    tvc_path,
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


# ---------------------------------------------------------------------------
# Covariate scale (#577)
# ---------------------------------------------------------------------------
# A covariate's unit is a reparameterisation: multiplying a column by a
# constant divides its coefficient by it, and the maximum of the
# likelihood is the same number. Field and test data carry covariates in
# their engineering units -- 1/T in kelvin (about 3e-3), a load in
# newtons, a pressure in pascals -- and a search or a convergence test in
# absolute terms meets a coefficient's gradient long before the optimum:
# #577 is a step-stress fit with Z = 1/T that stopped at its start,
# coefficient 0, and reported a verified maximum.
#
# The oracle is the likelihood reached: it cannot depend on the units.
# Models whose fit reports no ``neg_ll`` are compared on their
# predictions instead. A failure is keyed ``covariate_scale[<label>]``,
# or ``covariate_scale[tvc <label>]`` for the fit_tvc path, in
# KNOWN_FAILURES.
SCALES = {
    "1-731": 1 / 731.0,  # an awkward unit change, below the data's scale
    "731": 731.0,  # and above it
    "1e-6": 1e-6,  # coefficients of order 1e6
}
# The maximised log-likelihood may differ by the optimiser's tolerance
# between the two parameterisations, and no more.
NEG_LL_ATOL = 1e-5


def _shrunk(case, data, k):
    key = case.covariates
    return dict(data, **{key: np.asarray(data[key], dtype=float) * k})


def _neg_ll(model):
    value = getattr(model, "neg_ll", None)
    try:
        value = value() if callable(value) else value
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _scale_params(tvc):
    params = []
    for case in CASES:
        if not case.applies("covariate_scale") or case.covariates is None:
            continue
        if tvc and tvc_path(case) is None:
            continue
        for label, k in SCALES.items():
            # The fit_tvc path is a second refit of every case: left to
            # the full run, as are the slow cases' refits.
            slow = tvc or case.is_slow("units")
            marks = [pytest.mark.slow] if slow else []
            key = f"tvc {label}" if tvc else label
            reason = case.xfail.get(f"covariate_scale[{key}]")
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(
                pytest.param(case, k, id=f"{case.name}-{label}", marks=marks)
            )
    return params


def _same_model(case, ref_model, model, k):
    a, b = _neg_ll(ref_model), _neg_ll(model)
    if a is not None and b is not None:
        assert abs(a - b) <= NEG_LL_ATOL, (
            f"neg_ll {a:.8f} in the covariate's units, {b:.8f} with it "
            f"multiplied by {k:g} (the fit says {model.maximum!r})"
        )
        rtol = max(case.rtol, 1e-2)  # the optimum's flat directions
    else:
        # Without a likelihood to compare, the predictions carry the
        # optimiser's tolerance in either parameterisation.
        rtol = max(case.rtol, 1e-4)
    ref = predictions(case, ref_model)
    got = predictions(case, model, Z=np.asarray(case.Z, dtype=float) * k)
    compare(case, got, ref, rtol=rtol)


@pytest.mark.parametrize("case, k", _scale_params(tvc=False))
def test_covariate_scale(case, k):
    model = refit(case, _shrunk(case, case.data(), k))
    _same_model(case, fitted(case), model, k)


@pytest.mark.parametrize("case, k", _scale_params(tvc=True))
def test_covariate_scale_tvc(case, k):
    fit_tvc = tvc_path(case)
    with quiet():
        ref = fit_tvc(case.data())
        model = fit_tvc(_shrunk(case, case.data(), k))
    _same_model(case, ref, model, k)
