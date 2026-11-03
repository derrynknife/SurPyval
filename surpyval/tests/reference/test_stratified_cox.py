"""The residuals, proportional-hazards test, robust standard errors and
concordance of a stratified Cox model against stored R ``survival``
results (``scripts/reference/reference_r_stratified_cox.R``).

R's ``lung`` with ``strata(sex)`` under Efron and Breslow ties, and the
start-stop ``heart`` data with ``strata(surgery)`` (fitted with
``fit_tvc``, the robust variance clustered by patient). Every residual is
within its stratum's risk sets; the coefficients agree with R's to about
1e-9, so the residuals and statistics, smooth functions of them, agree to
``rtol=1e-6``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    added_covariate_score_test,
    compute_residuals,
    zph_score_test,
)

from ._data import fixture, values

DERIVED = dict(rtol=1e-6, atol=1e-8)
SOURCE = "r_stratified_cox"


def _lung(ties):
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["ph_ecog"]])[keep]
    model = sp.CoxPH.fit(
        d["time"][keep],
        Z,
        c=d["c"][keep],
        strata=d["sex"][keep],
        tie_method=ties,
    )
    return model, d["time"][keep], d["sex"][keep], d["c"][keep]


def _heart():
    d = fixture("heart")
    Z = np.column_stack([d["age"], d["year"], d["transplant"]])
    model = sp.CoxPH.fit_tvc(
        d["id"], d["start"], d["stop"], d["c"], Z, strata=d["surgery"]
    )
    return model, d["stop"], d["surgery"], d["c"]


CASES = {
    "lung_strata_sex_efron": lambda: _lung("efron"),
    "lung_strata_sex_breslow": lambda: _lung("breslow"),
    "heart_strata_surgery_efron": _heart,
}


@pytest.mark.parametrize("ref_id", list(CASES))
def test_287_stratified_residuals_match_r(ref_id):
    model, x, strata, c = CASES[ref_id]()
    ref = values(SOURCE, ref_id)
    assert_allclose(model.beta, ref["coef"], rtol=1e-7)
    for kind in ("martingale", "deviance", "score", "dfbeta"):
        assert_allclose(compute_residuals(model, kind), ref[kind], **DERIVED)
    # R orders the event residuals by stratum, then time (ties in the
    # data's order); SurPyval keeps the event rows' order.
    events = np.flatnonzero(c == 0)
    order = np.lexsort((x[events], strata[events]))
    assert_allclose(x[events][order], ref["schoenfeld_time"])
    for kind in ("schoenfeld", "scaled_schoenfeld"):
        res = compute_residuals(model, kind)[order]
        assert_allclose(res, ref[kind], **DERIVED)


@pytest.mark.parametrize("ref_id", list(CASES))
@pytest.mark.parametrize("transform", ["km", "rank", "identity"])
def test_287_stratified_zph_matches_r(ref_id, transform):
    model = CASES[ref_id]()[0]
    ref = values(SOURCE, ref_id)
    res = zph_score_test(model, transform)
    stats = [e["statistic"] for e in res["per_covariate"]]
    stats.append(res["global"]["statistic"])
    p = [e["p_value"] for e in res["per_covariate"]]
    p.append(res["global"]["p_value"])
    assert_allclose(stats, ref[f"zph_{transform}_chisq"], **DERIVED)
    assert_allclose(p, ref[f"zph_{transform}_p"], **DERIVED)


@pytest.mark.parametrize("ref_id", list(CASES))
def test_287_stratified_robust_standard_errors_match_r(ref_id):
    model = CASES[ref_id]()[0]
    ref = values(SOURCE, ref_id)
    se = np.sqrt(np.diag(model.robust_covariance()))
    assert_allclose(se, ref["robust_se"], **DERIVED)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_287_stratified_concordance_and_square_test_match_r(ties):
    model = _lung(ties)[0]
    ref = values(SOURCE, f"lung_strata_sex_{ties}")
    assert_allclose(model.concordance(), ref["concordance"], rtol=1e-12)
    d = fixture("lung")
    age = d["age"][~np.isnan(d["ph_ecog"])]
    res = added_covariate_score_test(model, age**2)
    assert_allclose(res["statistic"], ref["square_age"], **DERIVED)
