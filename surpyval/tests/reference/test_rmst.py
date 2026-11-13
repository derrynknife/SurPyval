"""The parametric restricted mean survival time against R.

``scripts/reference/reference_r_rmst.R`` writes ``data/r_rmst.json``: on
survival's ``lung`` and ``ovarian`` data, R's fit (``survreg``; the Gamma
by ``optim`` on its likelihood), the RMST as ``integrate()`` of the fitted
survival function to ``tau``, its delta-method standard error (central
differences, ``vcov`` the inverse observed information) and the Wald
interval on ``logit(rmst / tau)``; for two AFT regressions the same at two
covariate values, and their difference with the delta method on the
difference. flexsurv 2.3.2's own fit and ``summary(fit, type="rmst")``
are stored beside them (its intervals are simulated, so they are not).

Tolerances. The values at the R estimate depend on SurPyval's fit
agreeing with R's, which it does to the optimisers' stopping rules
(log-likelihoods within 1e-9 here): the RMST and its interval to
``rtol=2e-6`` (largest seen 5.4e-7, the Gamma, whose R fit is by a
general-purpose optimiser), the standard error to ``rtol=2e-5`` (largest
seen 1.9e-6). SurPyval's RMST at flexsurv's own estimate
(``from_params``) must equal flexsurv's to ``rtol=1e-7``: the same
function at the same parameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, reference

POINT = dict(rtol=2e-6)
SE = dict(rtol=2e-5)

UNIVARIATE = {
    "weibull": sp.Weibull,
    "exponential": sp.Exponential,
    "lognormal": sp.LogNormal,
    "loglogistic": sp.LogLogistic,
    "gamma": sp.Gamma,
}


def _flexsurv_params(name, stored):
    """flexsurv's estimate in SurPyval's parameters."""
    if name == "exponential":
        return [float(np.ravel(stored)[0])]
    if name in ("weibull", "loglogistic"):
        return [stored["scale"], stored["shape"]]
    if name == "lognormal":
        return [stored["meanlog"], stored["sdlog"]]
    return [stored["shape"], stored["rate"]]


@pytest.mark.parametrize("name", sorted(UNIVARIATE))
def test_univariate_rmst_matches_r(name):
    ref = reference("r_rmst", f"lung_{name}")
    v = ref["values"]
    lung = fixture("lung")
    model = UNIVARIATE[name].fit(lung["time"], c=lung["c"])
    assert -model.neg_ll() == pytest.approx(v["loglik"], abs=1e-6)
    taus = ref["settings"]["tau"]
    got = [model.rmst(tau) for tau in taus]
    for key, tol in (("rmst", POINT), ("se", SE)):
        assert_allclose([g[key] for g in got], v[key], **tol)
    for key in ("lower", "upper"):
        assert_allclose([g[key] for g in got], v[key], **POINT)
    # rmst_cb's Wald bound is the interval of rmst
    assert_allclose(
        model.rmst_cb(taus),
        np.column_stack([v["lower"], v["upper"]]),
        **POINT,
    )


@pytest.mark.parametrize("name", sorted(UNIVARIATE))
def test_univariate_rmst_matches_flexsurv_at_its_estimate(name):
    ref = reference("r_rmst", f"lung_{name}")
    v = ref["values"]
    params = _flexsurv_params(name, v["flexsurv_params"])
    model = UNIVARIATE[name].from_params(params)
    got = [model.mean(tau=tau) for tau in ref["settings"]["tau"]]
    assert_allclose(got, v["flexsurv_rmst"], rtol=1e-7)


REGRESSION = {
    "lung_weibull_sex": (sp.WeibullAFT, "lung", "time", "c", "sex"),
    "ovarian_lognormal_rx": (
        sp.LogNormalAFT,
        "ovarian",
        "futime",
        "fustat",
        "rx",
    ),
}


@pytest.mark.parametrize("ref_id", sorted(REGRESSION))
def test_regression_rmst_matches_r(ref_id):
    fitter, data_name, t_col, c_col, z_col = REGRESSION[ref_id]
    ref = reference("r_rmst", ref_id)
    v = ref["values"]
    tau = ref["settings"]["tau"]
    zs = ref["settings"]["z"]
    data = fixture(data_name)
    c = data["c"] if c_col == "c" else 1 - data[c_col]
    model = fitter.fit(data[t_col], Z=data[z_col].reshape(-1, 1), c=c)
    rows = [[z] for z in zs]
    got = model.rmst(tau, rows)
    for key in ("rmst", "lower", "upper"):
        assert_allclose(got[key], v[key], **POINT)
    assert_allclose(got["se"], v["se"], **SE)
    # flexsurv's AFT fit gives the same RMST at each covariate value
    assert_allclose(got["rmst"], v["flexsurv_rmst"], rtol=1e-6)
    diff = model.rmst_diff(tau, rows[0], rows[1])
    assert_allclose(diff["difference"], v["difference"], **POINT)
    assert_allclose(diff["se"], v["difference_se"], **SE)
