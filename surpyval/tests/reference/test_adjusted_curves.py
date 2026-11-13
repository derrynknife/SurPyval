"""Directly adjusted survival curves (``surpyval.direct_adjustment``)
against stored R results
(``scripts/reference/reference_r_adjusted_curves.R``), on the ``lung``
data without the row whose ``ph.ecog`` is missing; the group is ``sex``.

* ``cox_by_hand``: ``rowMeans(survfit(coxph, newdata)$surv)`` with the
  group set to each level, and the area under each averaged step curve.
  The coefficients agree with R's to about 1e-8; tested to ``rtol=1e-6``.
* ``weibull_by_hand``: the same averages of a ``survreg`` Weibull fit,
  with delta-method standard errors from a central-difference gradient
  and ``vcov``; with the covariates fixed, and with the spread of the
  units' predictions added. The two fits' maxima agree to about 1e-6
  and both gradients are numerical: the values are tested to
  ``rtol=1e-5`` and the standard errors to ``rtol=1e-3``.
* ``cox_ate``: ``riskRegression::ate``, the G-formula with
  influence-function standard errors. Its point estimates are the same
  quantity (``rtol=1e-6``). Its standard errors use the Cox model's
  robust influence function where ``direct_adjustment`` uses the
  model-based covariance, so they are not the same estimator: they are
  compared to ``rtol=0.1``, which they do not meet on these data (a
  strict xfail, with the numbers).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

SOURCE = "r_adjusted_curves"


def _lung():
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["sex"], d["age"], d["ph_ecog"]])[keep]
    return d["time"][keep], d["c"][keep], Z


@pytest.fixture(scope="module")
def cox():
    x, c, Z = _lung()
    return sp.direct_adjustment(sp.CoxPH.fit(x, Z, c=c), group=0)


@pytest.fixture(scope="module")
def weibull():
    x, c, Z = _lung()
    model = sp.WeibullAFT.fit(x, Z, c=c)
    return (
        sp.direct_adjustment(model, group=0),
        sp.direct_adjustment(model, group=0, covariates="fixed"),
    )


def test_cox_curves_are_survfit_row_means(cox):
    ref = values(SOURCE, "cox_by_hand")
    assert cox.levels == [1.0, 2.0]
    for g in (1, 2):
        assert_allclose(cox.sf(ref["times"], g), ref[f"sf_{g}"], rtol=1e-6)
        got = cox.rmst(ref["tau"], g)["rmst"]
        assert_allclose(got, ref[f"rmst_{g}"], rtol=1e-6)


def test_weibull_curves_and_fixed_covariate_errors(weibull):
    sampled, fixed = weibull
    ref = values(SOURCE, "weibull_by_hand")
    t, tau = ref["times"], ref["tau"]
    z = 1.959963984540054
    for g in (1, 2):
        assert_allclose(fixed.sf(t, g), ref[f"sf_{g}"], rtol=1e-5)
        res = fixed.rmst(tau, g)
        assert_allclose(res["rmst"], ref[f"rmst_{g}"], rtol=1e-5)
        assert_allclose(res["se"], ref[f"rmst_se_fixed_{g}"], rtol=1e-3)
        res = sampled.rmst(tau, g)
        assert_allclose(res["se"], ref[f"rmst_se_sampled_{g}"], rtol=1e-3)
        # the standard error of sf, read back from the logit bounds
        for adjusted, key in ((fixed, "fixed"), (sampled, "sampled")):
            S = adjusted.sf(t, g)
            cb = adjusted.cb(t, g)
            logit = np.log(cb / (1 - cb))
            se = (logit[:, 1] - logit[:, 0]) / (2 * z) * S * (1 - S)
            assert_allclose(se, ref[f"sf_se_{key}_{g}"], rtol=1e-3)


def test_weibull_differences(weibull):
    sampled, fixed = weibull
    ref = values(SOURCE, "weibull_by_hand")
    t, tau = ref["times"], ref["tau"]
    for adjusted, key in ((fixed, "fixed"), (sampled, "sampled")):
        res = adjusted.sf_diff(t, 2, 1)
        assert_allclose(res["difference"], ref["sf_diff"], rtol=1e-5)
        assert_allclose(res["se"], ref[f"sf_diff_se_{key}"], rtol=1e-3)
        res = adjusted.rmst_diff(tau, 2, 1)
        assert_allclose(res["difference"], ref["rmst_diff"], rtol=1e-5)
        assert_allclose(res["se"], ref[f"rmst_diff_se_{key}"], rtol=1e-3)


def test_cox_curves_against_riskregression_ate(cox):
    ref = values(SOURCE, "cox_ate")
    t = ref["times"]
    for g in (1, 2):
        assert_allclose(cox.ff(t, g), ref[f"risk_{g}"], rtol=1e-6)
    # ate's difference is of the risks, level 2 minus level 1
    res = cox.sf_diff(t, 2, 1)
    assert_allclose(-res["difference"], ref["risk_diff"], rtol=1e-6)


def _risk_se(adjusted, t, g):
    """The standard error of the adjusted survival (= of the risk) of
    level ``g``, from the width of its logit bounds."""
    z = 1.959963984540054
    S = adjusted.sf(t, g)
    cb = adjusted.cb(t, g)
    logit = np.log(cb / (1 - cb))
    return (logit[:, 1] - logit[:, 0]) / (2 * z) * S * (1 - S)


def test_cox_one_unit_errors_are_survfits():
    # The model's part of the variance, for a population of one unit, is
    # survfit's model-based standard error of that unit's curve.
    ref = values(SOURCE, "cox_se_components")
    t = ref["times"]
    x, c, Z = _lung()
    model = sp.CoxPH.fit(x, Z, c=c)
    row = [[1.0, ref["row_age"], ref["row_ph_ecog"]]]
    one = sp.direct_adjustment(
        model, group=0, levels=[1, 2], Z=row, covariates="fixed"
    )
    for g in (1, 2):
        assert_allclose(one.sf(t, g), ref[f"unit_sf_{g}"], rtol=1e-6)
        se = _risk_se(one, t, g)
        assert_allclose(se, ref[f"unit_survfit_se_{g}"], rtol=1e-4)


def test_cox_covariate_term_is_ates(cox):
    # The sampling of the covariates: ate's influence function carries it
    # as (F_i(t) - mean F(t)) / n, the "sampled" term here.
    ref = values(SOURCE, "cox_se_components")
    t = ref["times"]
    x, c, Z = _lung()
    fixed = sp.direct_adjustment(
        sp.CoxPH.fit(x, Z, c=c), group=0, covariates="fixed"
    )
    for g in (1, 2):
        term = np.sqrt(_risk_se(cox, t, g) ** 2 - _risk_se(fixed, t, g) ** 2)
        assert_allclose(term, ref[f"covariate_se_{g}"], rtol=1e-4)


@pytest.mark.xfail(
    strict=True,
    reason="A different estimator of the model's part, not an error: "
    "ate's influence function is the sum of the covariates' term (equal "
    "here, test_cox_covariate_term_is_ates) and the Cox model's own "
    "empirical influence (coefficients and Breslow baseline, a sandwich "
    "variance), where direct_adjustment uses survfit's model-based "
    "variance (test_cox_one_unit_errors_are_survfits). The two agree when "
    "the model holds and differ when it fits less well: on lung, one "
    "unit's predictCox(se = TRUE) is 10% below survfit's std.err for sex "
    "2 at t = 700 (0.0495 against 0.0549) and 8% above it for sex 1 "
    "(0.0294 against 0.0274), the same pattern as the averaged curves "
    "(4% to 13% apart; the difference at t = 700, 0.0489 against "
    "0.0435), past the 10% asked.",
)
def test_cox_errors_against_riskregression_ate(cox):
    ref = values(SOURCE, "cox_ate")
    t = ref["times"]
    z = 1.959963984540054
    for g in (1, 2):
        S = cox.sf(t, g)
        cb = cox.cb(t, g)
        logit = np.log(cb / (1 - cb))
        se = (logit[:, 1] - logit[:, 0]) / (2 * z) * S * (1 - S)
        assert_allclose(se, ref[f"risk_se_{g}"], rtol=0.1)
    res = cox.sf_diff(t, 2, 1)
    assert_allclose(res["se"], ref["risk_diff_se"], rtol=0.1)
