"""``ParametricCompetingRisksRegression`` against stored R ``flexsurv``
results (``scripts/reference/
reference_r_parametric_competing_risks_regression.R``).

Each cause's ``flexsurvreg(Surv(time, cause == k) ~ ..., dist =
"weibullPH")``, the other causes censored: the maximised log-likelihood,
the shape, the scale, the coefficients and their covariance. flexsurv's
``weibullPH`` has the hazard ``scale * shape * t^(shape - 1) *
exp(coefficients' Z)`` and SurPyval's ``WeibullPH`` ``(beta / alpha) (t /
alpha)^(beta - 1) exp(coefficients' Z)``: the shape is ``beta``, the scale
``alpha^(-beta)``, and the coefficients are the same. And the cumulative
incidences of the competing-risks multi-state model built from the causes'
fits, ``pmatrix.fs`` at two covariate rows, out to several times the
follow-up. On ``pbc`` (transplant and death; also with a lognormal
accelerated failure time model for death) and the synthetic ``competing``
set (integer times, ties).

The fits are two optimisers' answers. They agree to 5e-7 (relative) in
the parameters, 1e-12 in the log-likelihoods, 2e-6 in the covariances and
1e-7 (absolute) in the incidences, which flexsurv gets from an ODE solver;
the tolerances below, set before the comparison was run, are looser.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import (
    LogNormalAFT,
    ParametricCompetingRisksRegression,
    WeibullPH,
)

from ._data import fixture, values

SOURCE = "r_parametric_competing_regression"
COVARIATES = {
    "pbc": ("years", ("age", "log_bili", "female")),
    "competing": ("x", ("group", "z")),
}


def _data(name):
    d = fixture(name)
    time, covariates = COVARIATES[name]
    e = np.array(
        [None if k == 0 else int(k) for k in d["cause"]], dtype=object
    )
    return d[time], np.column_stack([d[k] for k in covariates]), e


@pytest.fixture(scope="module")
def fits():
    out = {
        name: ParametricCompetingRisksRegression.fit(*_data(name))
        for name in COVARIATES
    }
    out["pbc_mixed"] = ParametricCompetingRisksRegression.fit(
        *_data("pbc"), model={1: WeibullPH, 2: LogNormalAFT}
    )
    return out


@pytest.mark.parametrize("name", sorted(COVARIATES))
@pytest.mark.parametrize("cause", [1, 2])
def test_each_cause_matches_flexsurvreg(fits, name, cause):
    ref = values(SOURCE, f"{name}_weibullPH_cause{cause}")
    model = fits[name].models[cause]
    assert model.maximum == "verified"
    alpha, beta = model.params[:2]
    coef = model.params[2:]
    shape, scale = ref["est"][:2]
    assert_allclose(-model.neg_ll(), ref["loglik"], rtol=0, atol=1e-5)
    assert_allclose(beta, shape, rtol=1e-4)
    assert_allclose(alpha ** (-beta), scale, rtol=1e-3)
    assert_allclose(coef, ref["est"][2:], rtol=1e-3, atol=1e-5)
    # The covariance on flexsurv's scale (log shape, log scale, the
    # coefficients), by the delta method: log scale = -beta log(alpha).
    p = coef.size
    J = np.zeros((2 + p, 2 + p))
    J[0, 1] = 1 / beta
    J[1, 0] = -beta / alpha
    J[1, 1] = -np.log(alpha)
    J[2:, 2:] = np.eye(p)
    assert_allclose(J @ model.covariance() @ J.T, ref["cov"], rtol=2e-3)


@pytest.mark.parametrize(
    "fit_id, ref_id",
    [
        ("pbc", "pbc_weibullPH_cif"),
        ("competing", "competing_weibullPH_cif"),
        ("pbc_mixed", "pbc_mixed_cif"),
    ],
)
def test_cumulative_incidences_match_the_multi_state_model(
    fits, fit_id, ref_id
):
    ref = values(SOURCE, ref_id)
    model = fits[fit_id]
    t = np.asarray(ref["times"], dtype=float)
    for entry in ref["rows"]:
        row = np.asarray(entry["row"], dtype=float)
        assert_allclose(model.cif(t, row, 1), entry["cause1"], atol=1e-4)
        assert_allclose(model.cif(t, row, 2), entry["cause2"], atol=1e-4)
        assert_allclose(model.sf(t, row), entry["survival"], atol=1e-4)


def test_the_lognormal_cause_matches_flexsurvreg(fits):
    """``dist = "lnorm"``: flexsurv's ``meanlog = mu + coefficients' Z``.
    SurPyval's ``LogNormalAFT`` has the sign of every SurPyval regression
    (a positive coefficient shortens the life), so its coefficients are
    flexsurv's negated."""
    ref = values(SOURCE, "pbc_lnorm_cause2")
    model = fits["pbc_mixed"].models[2]
    assert_allclose(-model.neg_ll(), ref["loglik"], rtol=0, atol=1e-5)
    mu, sigma = model.params[:2]
    assert_allclose(mu, ref["est"][0], rtol=1e-4)
    assert_allclose(sigma, ref["est"][1], rtol=1e-4)
    assert_allclose(model.params[2:], -np.asarray(ref["est"][2:]), rtol=1e-3)
