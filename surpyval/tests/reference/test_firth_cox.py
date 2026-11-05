"""Firth's penalised Cox fit, ``CoxPH.fit_firth``, against R's
``coxphf`` 1.13.4 (``scripts/reference/reference_r_firth_cox.R``).

Six fits, all with Breslow ties (``coxphf``'s only ones): the seeded
``separation`` data, where every event has ``z1 = 1`` and the ordinary
coefficient of ``z1`` is infinite, at the 95% and the 90% level; ``lung``
(age and sex), where Firth's fit is close to the ordinary one; the seeded
``ties`` data; the seeded ``left_truncation`` data (``Surv(tl, x,
status)``); and the start-stop ``heart`` data. ``coxphf`` was run to
convergence (``epsilon = gconv = 1e-12``).

Tolerances: 1e-6 (absolute) on the coefficients, their covariance and
standard errors, the Wald bounds and p-values; 1e-5 on the profile
penalised-likelihood bounds; 1e-6 on the penalised likelihood-ratio
p-values, the log-likelihoods and the likelihood-ratio statistic.
SurPyval agrees to about 1e-10 on every one.

``coxphf`` divides each covariate by its standard deviation before
fitting, which adds ``sum(log(sd))`` to its log-likelihoods (not to the
tests); SurPyval reports them with the information in the covariates' own
units, so the stored ``log_sd_sum`` is subtracted before comparing.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

SOURCE = "r_firth_cox"
COEF = dict(rtol=0, atol=1e-6)
BOUND = dict(rtol=0, atol=1e-5)

# ref id: (fixture, covariates, time column, entry column)
CASES = {
    "separation": ("separation", ["z1", "z2"], "x", None),
    "separation_90": ("separation", ["z1", "z2"], "x", None),
    "lung_age_sex": ("lung", ["age", "sex"], "time", None),
    "ties": ("ties", ["z1", "z2"], "x", None),
    "left_truncation": ("left_truncation", ["z"], "x", "tl"),
    "heart_start_stop": (
        "heart",
        ["age", "year", "transplant"],
        "stop",
        "start",
    ),
}


def _fit(ref_id, **kwargs):
    name, covariates, time, entry = CASES[ref_id]
    d = fixture(name)
    Z = np.column_stack([d[k] for k in covariates])
    tl = None if entry is None else d[entry]
    ref = values(SOURCE, ref_id)
    model = sp.CoxPH.fit_firth(
        d[time], Z, c=d["c"], tl=tl, alpha_ci=ref["alpha"], **kwargs
    )
    return model, ref


@pytest.mark.parametrize("ref_id", list(CASES))
def test_firth_matches_coxphf(ref_id):
    model, ref = _fit(ref_id)
    firth = model.firth
    assert model.maximum == "verified"
    assert_allclose(model.params, ref["coef"], **COEF)
    # (one covariate's 1 x 1 matrix is stored as a list of one)
    var = np.reshape(ref["var"], model.covariance().shape)
    assert_allclose(model.covariance(), var, **COEF)
    assert_allclose(model.standard_errors(), np.sqrt(np.diag(var)), **COEF)
    # The profile penalised-likelihood bounds and tests (pl = TRUE)
    assert_allclose(firth.lower, ref["ci_lower"], **BOUND)
    assert_allclose(firth.upper, ref["ci_upper"], **BOUND)
    assert_allclose(firth.p_values, ref["p"], **COEF)
    assert_allclose(model.p_values, ref["p"], **COEF)
    # The Wald ones (pl = FALSE)
    assert_allclose(firth.wald_lower, ref["wald_ci_lower"], **COEF)
    assert_allclose(firth.wald_upper, ref["wald_ci_upper"], **COEF)
    assert_allclose(firth.wald_p, ref["wald_p"], **COEF)
    # The penalised log-likelihoods and the likelihood-ratio test
    offset = ref["log_sd_sum"]
    assert_allclose(firth.log_likelihood - offset, ref["loglik"], **COEF)
    assert_allclose(
        firth.log_likelihood_null - offset, ref["loglik_null"], **COEF
    )
    test = model.likelihood_ratio_test()
    assert_allclose(test["statistic"], ref["lr_statistic"], **COEF)
    assert test["df"] == ref["df"]


def test_firth_wald_option_matches_coxphf_pl_false():
    model, ref = _fit("separation", profile=False)
    assert_allclose(model.firth.lower, ref["wald_ci_lower"], **COEF)
    assert_allclose(model.firth.upper, ref["wald_ci_upper"], **COEF)
    assert_allclose(model.p_values, ref["wald_p"], **COEF)
    assert np.all(np.isnan(model.firth.chisq))


def test_firth_summary_shows_the_profile_bounds():
    model, ref = _fit("separation_90")
    table = model.summary()
    assert_allclose(table["coef lower 90%"], ref["ci_lower"], **BOUND)
    assert_allclose(
        table["exp(coef) upper 90%"], np.exp(ref["ci_upper"]), rtol=1e-4
    )
    assert_allclose(table["p"], ref["p"], **COEF)
