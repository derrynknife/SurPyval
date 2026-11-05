"""The Brier score, integrated Brier score and time-dependent AUC of one
cause with competing risks, with their standard errors, intervals and
model comparisons, against R.

The stored values come from
scripts/reference/reference_r_competing_metrics.R:
``riskRegression::Score(cause = k, se.fit = TRUE, cens.model = "km")`` on
the seeded synthetic ``competing`` fixture (causes 1 and 2, whole-number
times, so events tie with each other and with censorings) and on the PBC
trial (death, with transplant competing), for two models each whose
predicted cumulative incidences are stored; and ``timeROC(cause = 2,
weighting = "marginal", iid = TRUE)``'s ``AUC_1`` and ``AUC_2`` on PBC.

Every estimate is a closed-form weighted sum, so the values agree to
rounding: ``rtol=1e-9`` (observed differences are below 1e-11). Two
exceptions, both on R's side:

* riskRegression's AUC standard errors for a cause other than 1 depend on
  the cause's label (the same data with the cause coded 1 give others,
  and those agree with timeROC's); SurPyval's match the relabelled call
  to 1e-9, and the test pins the label-dependent ones as different.
* timeROC's ``AUC_2`` standard errors differ from Score's and SurPyval's
  by up to about 2e-4 relative (its estimates and its ``AUC_1`` standard
  errors agree to 1e-9), so they are compared at ``rtol=1e-3``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, reference, values

EXACT = dict(rtol=1e-9, atol=1e-12)
SETS = ["competing_cause1", "competing_cause2", "pbc_death"]


def _data(name):
    entry = reference("r_competing_metrics", "score_" + name)
    d = fixture(entry["fixture"])
    x = d["x"] if "x" in d else d["years"]
    e = [None if k == 0 else int(k) for k in d["cause"]]
    return x, e


def _score(name, ref_id=None):
    ref = values("r_competing_metrics", ref_id or "score_" + name)
    x, e = _data(name)
    models = [k for k, v in ref.items() if isinstance(v, dict)]
    models.remove("contrast")
    preds = {m: np.array(ref[m]["cif"], dtype=float) for m in models}
    result = sp.score_predictions(
        x, None, preds, ref["times"], e=e, cause=int(ref["cause"])
    )
    return ref, models, result


def _auc_reference(name):
    """Score's AUC standard errors: from the call with the scored cause
    coded 1 where it is not 1 already (see the module notes)."""
    ref_id = "score_" + name
    if values("r_competing_metrics", ref_id)["cause"] != 1:
        ref_id += "_as_cause1"
    return values("r_competing_metrics", ref_id)


@pytest.mark.parametrize("name", SETS)
def test_brier_scores_match_riskregression(name):
    ref, models, result = _score(name)
    table = result.scores
    for m in models:
        rows = table[(table.metric == "brier") & (table.model == m)]
        assert_allclose(rows.time, ref["times"])
        assert_allclose(rows.estimate, ref[m]["brier"], **EXACT)
        assert_allclose(rows.se, ref[m]["brier_se"], **EXACT)
        assert_allclose(rows.lower, ref[m]["brier_lower"], **EXACT)
        assert_allclose(rows.upper, ref[m]["brier_upper"], **EXACT)


@pytest.mark.parametrize("name", SETS)
def test_auc_matches_riskregression(name):
    ref, models, result = _score(name)
    auc_ref = _auc_reference(name)
    table = result.scores
    for m in models:
        rows = table[(table.metric == "auc") & (table.model == m)]
        assert_allclose(rows.estimate, ref[m]["auc"], **EXACT)
        assert_allclose(rows.estimate, auc_ref[m]["auc"], **EXACT)
        assert_allclose(rows.se, auc_ref[m]["auc_se"], **EXACT)
        assert_allclose(rows.lower, auc_ref[m]["auc_lower"], **EXACT)
        assert_allclose(rows.upper, auc_ref[m]["auc_upper"], **EXACT)


@pytest.mark.parametrize("name", ["competing_cause2", "pbc_death"])
def test_riskregression_auc_se_depends_on_the_cause_label(name):
    """Pins the R behaviour the module notes describe: if a later
    riskRegression drops the label dependence, this fails and the
    relabelled entries can go."""
    ref, models, _ = _score(name)
    relabelled = _auc_reference(name)
    for m in models:
        assert_allclose(ref[m]["auc"], relabelled[m]["auc"], **EXACT)
        assert not np.allclose(ref[m]["auc_se"], relabelled[m]["auc_se"])


@pytest.mark.parametrize("name", SETS)
def test_integrated_brier_score_matches_riskregression_influence(name):
    ref, models, result = _score(name)
    for m in models:
        assert_allclose(result.estimate("ibs", m), ref[m]["ibs"], **EXACT)
        assert_allclose(result.se("ibs", m), ref[m]["ibs_se"], **EXACT)


@pytest.mark.parametrize("name", SETS)
def test_comparisons_match_riskregression_contrasts(name):
    ref, models, result = _score(name)
    table = result.comparisons
    for metric in ("brier", "auc", "ibs"):
        con = (_auc_reference(name) if metric == "auc" else ref)["contrast"]
        rows = table[table.metric == metric]
        assert set(rows.model) == {con["model"]}
        assert set(rows.reference) == {con["reference"]}
        diff = np.atleast_1d(con[metric])
        assert_allclose(rows.difference, diff, **EXACT)
        assert_allclose(rows.se, np.atleast_1d(con[metric + "_se"]), **EXACT)
        if metric != "ibs":
            assert_allclose(rows.lower, con[metric + "_lower"], **EXACT)
            assert_allclose(rows.upper, con[metric + "_upper"], **EXACT)
            # Where both models rank every subject alike the AUCs are
            # equal, the difference and its standard error 0, and the
            # p-value undefined: NA in Score, nan here.
            p_ref = np.asarray(con[metric + "_p"], dtype=float)
            p = rows.p_value.to_numpy()
            assert_allclose(p, p_ref, **EXACT)
            assert (np.isnan(p) == np.isnan(p_ref)).all()


@pytest.mark.parametrize("name", SETS)
def test_estimates_are_those_of_the_plain_functions(name):
    ref, models, result = _score(name)
    x, e = _data(name)
    k = int(ref["cause"])
    for m in models:
        F = np.array(ref[m]["cif"], dtype=float)
        _, bs = sp.brier_score(x, None, F, ref["times"], e=e, cause=k)
        _, auc = sp.auc_td(x, None, F, ref["times"], e=e, cause=k)
        ibs = sp.integrated_brier_score(x, None, F, ref["times"], e=e, cause=k)
        assert_allclose(result.estimate("brier", m), bs, **EXACT)
        assert_allclose(result.estimate("auc", m), auc, **EXACT)
        assert_allclose(result.estimate("ibs", m), ibs, **EXACT)


@pytest.mark.parametrize(
    "controls, key, se_tol",
    [
        ("event_free", "auc_1", EXACT),
        ("cause_free", "auc_2", dict(rtol=1e-3)),
    ],
)
def test_auc_definitions_match_timeroc(controls, key, se_tol):
    ref = values("r_competing_metrics", "timeroc_pbc_death")
    d = fixture("pbc")
    e = [None if k == 0 else int(k) for k in d["cause"]]
    x = ref["x"]  # the PBC times with ties broken, as timeROC was given
    marker = ref["marker"]
    times = ref["times"]
    _, auc = sp.auc_td(x, None, marker, times, e=e, cause=2, controls=controls)
    assert_allclose(auc, ref[key], **EXACT)
    result = sp.score_predictions(
        x,
        None,
        np.tile(marker[:, None], (1, len(times))),
        times,
        metrics=("auc",),
        e=e,
        cause=2,
        controls=controls,
    )
    assert_allclose(result.estimate("auc"), ref[key], **EXACT)
    assert_allclose(result.se("auc"), ref[key + "_se"], **se_tol)
