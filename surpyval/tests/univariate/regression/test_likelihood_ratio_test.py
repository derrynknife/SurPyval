"""The global likelihood-ratio test against the model with no covariates,
``likelihood_ratio_test()``."""

import numpy as np
import pytest

import surpyval as sp
from surpyval.datasets import load_rossi_static

SEVEN = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi_static()
    return df["week"].to_numpy(), 1 - df["arrest"].to_numpy(), df


def test_cox_matches_r_and_lifelines_on_rossi(rossi):
    # R: summary(coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar +
    # paro + prio, data = Rossi)) reports "Likelihood ratio test= 33.27 on 7
    # df, p=2e-05" (Fox and Weisberg, "Cox proportional-hazards regression
    # for survival data in R", appendix to An R Companion to Applied
    # Regression). lifelines 0.30.3's log_likelihood_ratio_test():
    # 33.26594580156939, p = 2.362045053699918e-05.
    x, c, df = rossi
    model = sp.CoxPH.fit(x, df[SEVEN].to_numpy(), c=c)
    res = model.likelihood_ratio_test()
    np.testing.assert_allclose(res["statistic"], 33.26594580156939, rtol=1e-8)
    np.testing.assert_allclose(
        res["p_value"], 2.362045053699918e-05, rtol=1e-6
    )
    assert res["df"] == 7
    assert res["log_likelihood"] == model.log_likelihood


def test_cox_stratified_matches_lifelines(rossi):
    # lifelines 0.30.3, CoxPHFitter(strata=["wexp"]) on the other six:
    # 23.77474748490863 on 6 df.
    x, c, df = rossi
    Z = df[[v for v in SEVEN if v != "wexp"]].to_numpy()
    model = sp.CoxPH.fit(x, Z, c=c, strata=df["wexp"].to_numpy())
    res = model.likelihood_ratio_test()
    np.testing.assert_allclose(res["statistic"], 23.77474748490863, rtol=1e-8)
    assert res["df"] == 6


def test_cox_null_is_the_fit_without_covariates_by_hand(rossi):
    # The null partial likelihood (beta = 0) with Efron ties, written out.
    x, c, df = rossi
    model = sp.CoxPH.fit(x, df[["fin", "prio"]].to_numpy(), c=c)
    ll0 = 0.0
    for t in np.unique(x[c == 0]):
        r = np.sum(x >= t)
        d = np.sum((x == t) & (c == 0))
        ll0 -= np.sum(np.log(r - np.arange(d)))
    res = model.likelihood_ratio_test()
    np.testing.assert_allclose(res["log_likelihood_null"], ll0, rtol=1e-12)


@pytest.mark.parametrize(
    "fitter, covariates, statistic",
    [
        # lifelines 0.30.3, WeibullAFTFitter / LogNormalAFTFitter
        # .log_likelihood_ratio_test() on the same covariates.
        (sp.WeibullAFT, SEVEN, 33.415666000417104),
        (sp.LogNormalAFT, ["fin", "age", "prio"], 20.938779887553665),
    ],
)
def test_parametric_matches_lifelines(rossi, fitter, covariates, statistic):
    x, c, df = rossi
    model = fitter.fit(x, df[covariates].to_numpy(float), c=c)
    res = model.likelihood_ratio_test()
    np.testing.assert_allclose(res["statistic"], statistic, rtol=1e-6)
    assert res["df"] == len(covariates)


def test_weibull_ph_and_aft_are_one_model(rossi):
    # An equivalent parameterisation: the Weibull PH and AFT models are the
    # same family, so they reach the same likelihood and the same test.
    x, c, df = rossi
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    ph = sp.WeibullPH.fit(x, Z, c=c).likelihood_ratio_test()
    aft = sp.WeibullAFT.fit(x, Z, c=c).likelihood_ratio_test()
    np.testing.assert_allclose(ph["statistic"], aft["statistic"], rtol=1e-6)


@pytest.mark.parametrize(
    "fitter", [sp.WeibullPH, sp.WeibullPO, sp.ExponentialAH, sp.LogNormalAFT]
)
def test_parametric_null_is_the_baseline_fit(rossi, fitter):
    x, c, df = rossi
    Z = df[["fin"]].to_numpy(float) * 1e-3
    model = fitter.fit(x, Z, c=c)
    res = model.likelihood_ratio_test()
    base = model.distribution.fit(x, c=c)
    np.testing.assert_allclose(
        res["log_likelihood_null"], base.log_likelihood, rtol=1e-9
    )
    assert res["df"] == 1 and res["statistic"] >= 0


def test_fixed_parameters_are_not_counted(rossi):
    x, c, df = rossi
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    model = sp.WeibullPH.fit(x, Z, c=c, fixed={"beta": 1.4})
    res = model.likelihood_ratio_test()
    base = sp.Weibull.fit(x, c=c, fixed={"beta": 1.4})
    np.testing.assert_allclose(
        res["log_likelihood_null"], base.log_likelihood, rtol=1e-9
    )
    assert res["df"] == 3


def test_tvc_fits_compare_with_the_same_rows(rossi):
    # One row per subject from 0 is the ordinary fit.
    x, c, df = rossi
    Z = df[["fin", "age", "prio"]].to_numpy(float)
    ids, xl = np.arange(x.size), np.zeros(x.size)
    for fitter in (sp.CoxPH, sp.WeibullPH):
        tvc = fitter.fit_tvc(ids, xl, x, c, Z).likelihood_ratio_test()
        plain = fitter.fit(x, Z, c=c).likelihood_ratio_test()
        np.testing.assert_allclose(
            tvc["statistic"], plain["statistic"], rtol=1e-8
        )


def test_refusals_say_what_to_do(rossi):
    x, c, df = rossi
    Z = df[["fin", "prio"]].to_numpy(float)
    for model in (sp.CoxPH.fit(x, Z, c=c), sp.WeibullPH.fit(x, Z, c=c)):
        restored = sp.from_dict(model.to_dict())
        with pytest.raises(ValueError, match="refit"):
            restored.likelihood_ratio_test()
    stress = np.where(df["fin"].to_numpy() == 1, 2.0, 1.0)
    al = sp.AcceleratedLife(sp.Weibull, sp.life_models.Power).fit(
        x, stress, c=c
    )
    with pytest.raises(ValueError, match="accelerated life"):
        al.likelihood_ratio_test()
    held = sp.WeibullPH.fit(x, Z[:, :1], c=c, fixed={"coef_0": 0.1})
    with pytest.raises(ValueError, match="nothing to test"):
        held.likelihood_ratio_test()
