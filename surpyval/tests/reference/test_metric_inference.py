"""Standard errors, intervals and model comparisons of the prediction
metrics against R (``score_predictions``).

The stored values come from scripts/reference/reference_r_metric_inference.R:
``riskRegression::Score(se.fit = TRUE, cens.model = "km")`` for the Brier
score and the time-dependent AUC, their standard errors, intervals and the
differences between two models with their tests; the integrated Brier
score's standard error from the same trapezoid combination of Score's
influence functions; and ``timeROC(iid = TRUE)`` with ``compare()`` on the
untied data. Each entry stores both models' predictions, so only the
metric is compared.

Every quantity is a closed-form weighted sum, so the values must agree to
rounding: ``rtol=1e-9`` (observed differences are below 1e-13), far
inside the 1e-6 (estimates) and 1e-4 (standard errors) the comparison
needs.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, reference, values

EXACT = dict(rtol=1e-9, atol=1e-12)
SETS = ["prediction_ties", "prediction_continuous", "veteran"]


def _score(name):
    ref = values("r_metric_inference", "score_" + name)
    d = fixture(name)
    if name == "veteran":
        x, c = d["time"], 1 - d["status"]
    else:
        x, c = d["x"], d["c"]
    models = [k for k, v in ref.items() if isinstance(v, dict)]
    models.remove("contrast")
    preds = {m: np.array(ref[m]["survival"], dtype=float) for m in models}
    result = sp.score_predictions(x, c, preds, ref["times"])
    return ref, models, result


@pytest.mark.parametrize("name", SETS)
def test_scores_match_riskregression(name):
    ref, models, result = _score(name)
    table = result.scores
    for m in models:
        for metric in ("brier", "auc"):
            rows = table[(table.metric == metric) & (table.model == m)]
            assert_allclose(rows.time, ref["times"])
            assert_allclose(rows.estimate, ref[m][metric], **EXACT)
            assert_allclose(rows.se, ref[m][metric + "_se"], **EXACT)
            assert_allclose(rows.lower, ref[m][metric + "_lower"], **EXACT)
            assert_allclose(rows.upper, ref[m][metric + "_upper"], **EXACT)


@pytest.mark.parametrize("name", SETS)
def test_integrated_brier_score_matches_riskregression_influence(name):
    ref, models, result = _score(name)
    for m in models:
        assert_allclose(result.estimate("ibs", m), ref[m]["ibs"], **EXACT)
        assert_allclose(result.se("ibs", m), ref[m]["ibs_se"], **EXACT)


@pytest.mark.parametrize("name", SETS)
def test_comparisons_match_riskregression_contrasts(name):
    ref, models, result = _score(name)
    con = ref["contrast"]
    table = result.comparisons
    for metric in ("brier", "auc", "ibs"):
        rows = table[table.metric == metric]
        assert set(rows.model) == {con["model"]}
        assert set(rows.reference) == {con["reference"]}
        assert_allclose(rows.difference, np.atleast_1d(con[metric]), **EXACT)
        assert_allclose(rows.se, np.atleast_1d(con[metric + "_se"]), **EXACT)
        if metric != "ibs":
            assert_allclose(rows.lower, con[metric + "_lower"], **EXACT)
            assert_allclose(rows.upper, con[metric + "_upper"], **EXACT)
            assert_allclose(rows.p_value, con[metric + "_p"], **EXACT)


def test_auc_and_comparison_match_timeroc_without_ties():
    ref = values("r_metric_inference", "timeroc_prediction_continuous")
    _, _, result = _score("prediction_continuous")
    assert_allclose(result.estimate("auc", "m"), ref["auc"], **EXACT)
    assert_allclose(result.se("auc", "m"), ref["auc_se"], **EXACT)
    assert_allclose(result.estimate("auc", "m2"), ref["auc_m2"], **EXACT)
    assert_allclose(result.se("auc", "m2"), ref["auc_se_m2"], **EXACT)
    p = result.comparisons.query("metric == 'auc'").p_value
    assert_allclose(p, ref["compare_p"], **EXACT)


@pytest.mark.parametrize("name", SETS)
def test_estimates_are_those_of_the_plain_functions(name):
    ref, models, result = _score(name)
    entry = reference("r_metric_inference", "score_" + name)
    d = fixture(entry["fixture"])
    x, c = (
        (d["time"], 1 - d["status"]) if name == "veteran" else (d["x"], d["c"])
    )
    for m in models:
        S = np.array(ref[m]["survival"], dtype=float)
        _, bs = sp.brier_score(x, c, S, ref["times"])
        _, auc = sp.auc_td(x, c, 1 - S, ref["times"])
        ibs = sp.integrated_brier_score(x, c, S, ref["times"])
        assert_allclose(result.estimate("brier", m), bs, **EXACT)
        assert_allclose(result.estimate("auc", m), auc, **EXACT)
        assert_allclose(result.estimate("ibs", m), ibs, **EXACT)
