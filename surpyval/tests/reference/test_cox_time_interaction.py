"""Cox models with time-varying coefficients, ``CoxPH.fit(...,
time_interaction=)``, against stored R ``survival`` results
(``scripts/reference/reference_r_cox_tt.R``).

R fits ``coxph(..., tt = function(x, t, ...) x * g(t))`` on ``veteran``
(Karnofsky score, the classic non-proportional effect) and ``lung``
(ECOG score, with and without ``strata(sex)``), under Efron and Breslow
ties, each checked in R against the ``survSplit`` form. The coefficients,
their covariance (so the standard errors) and the log partial likelihood
agree to ``rtol=1e-6``; beta(t) and its standard error, and the
cumulative hazard of a patient with Karnofsky score 60 (``survfit`` of the
split fit along that patient's rows), agree to the same tolerance.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

SOURCE = "r_cox_tt"
TOL = dict(rtol=1e-6, atol=1e-10)


def _veteran(g, ties, columns=("karno",), w=None):
    d = fixture("veteran")
    Z = np.column_stack([d[k] for k in columns])
    j = list(columns).index("karno")
    return sp.CoxPH.fit(
        d["time"],
        Z,
        c=1 - d["status"],
        tie_method=ties,
        time_interaction={j: g},
        w=w,
    )


def _lung(ties, strata=False):
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["ph_ecog"]])[keep]
    return sp.CoxPH.fit(
        d["time"][keep],
        Z,
        c=d["c"][keep],
        tie_method=ties,
        strata=d["sex"][keep] if strata else None,
        time_interaction={1: "log"},
    )


def _log_t20(t):
    return np.log(t + 20)


CASES = {
    "veteran_karno_t_efron": (lambda: _veteran("t", "efron"), 0),
    "veteran_karno_t_breslow": (lambda: _veteran("t", "breslow"), 0),
    "veteran_karno_log_efron": (lambda: _veteran("log", "efron"), 0),
    "veteran_trt_prior_karno_log_t20_efron": (
        lambda: _veteran(_log_t20, "efron", ("trt", "prior", "karno")),
        2,
    ),
    "lung_ph_ecog_log_efron": (lambda: _lung("efron"), 1),
    "lung_ph_ecog_log_breslow": (lambda: _lung("breslow"), 1),
    "lung_ph_ecog_log_efron_strata_sex": (lambda: _lung("efron", True), 1),
}


@pytest.mark.parametrize("ref_id", list(CASES))
def test_coefficients_covariance_and_loglik_match_r(ref_id):
    build, _ = CASES[ref_id]
    model = build()
    ref = values(SOURCE, ref_id)
    assert_allclose(model.params, ref["coef"], **TOL)
    assert_allclose(model.covariance(), ref["vcov"], **TOL)
    assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(ref["vcov"])), **TOL
    )
    assert_allclose(-model.neg_ll(), ref["loglik"], rtol=1e-9)


@pytest.mark.parametrize("ref_id", list(CASES))
def test_beta_of_t_and_its_se_match_r(ref_id):
    build, j = CASES[ref_id]
    ref = values(SOURCE, ref_id)
    effect = build().time_varying_effect(j, ref["beta_t_times"])
    assert_allclose(effect.beta, ref["beta_t"], **TOL)
    assert_allclose(effect.se, ref["beta_t_se"], **TOL)


def test_prediction_at_fixed_covariates_matches_survfit_of_the_split_fit():
    model = _veteran("t", "efron")
    ref = values(SOURCE, "veteran_karno_t_efron")
    H = model.Hf(ref["Hf_karno60_times"], [60.0])
    assert_allclose(H, ref["Hf_karno60"], **TOL)
    assert_allclose(
        model.sf(ref["Hf_karno60_times"], [60.0]),
        np.exp(-ref["Hf_karno60"]),
        **TOL,
    )


def test_weighted_fit_matches_r_robust_variance_clustered_by_patient():
    ref = values(SOURCE, "veteran_karno_t_efron_weighted")
    model = _veteran("t", "efron", w=ref["w"])
    assert_allclose(model.params, ref["coef"], **TOL)
    assert_allclose(model.covariance(), ref["vcov"], **TOL)
