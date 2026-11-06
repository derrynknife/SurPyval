"""``model.at(Z)``: a regression's lifetime distribution at one covariate
row, as a plain life model (:class:`LifeAtCovariates`).

Against R: ``scripts/reference/reference_r_regression_at.R`` prints
flexsurv 2.3.2's ``summary(fit, newdata = data.frame(sex = 2), type =
...)`` for Weibull AFT and PH regressions of survival's lung data on sex,
and survival 3.8.12's ``predict(survreg, type = "quantile")``, each with
R's estimate. The values are typed in below. SurPyval's model is built at
R's estimate with ``with_params`` (the parameters mapped as the script
says), so the comparison is of one function at one set of parameters:
to ``rtol=1e-9`` (flexsurv's quantiles are found by a root search, to
about 1e-10 relative; its mean by integration).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.tests.reference._data import fixture
from surpyval.univariate.regression import LifeAtCovariates

TIMES = [100.0, 365.0, 730.0]
PROBS = [0.1, 0.5, 0.9]

# reference_r_regression_at.R, R 4.6.2, flexsurv 2.3.2, survival 3.8.12
FLEXSURV_AFT = {
    "params": [241.914353104164, 1.32434892591292, -0.395578007441066],
    "sf": [0.896858096716195, 0.546245236017343, 0.219971581643749],
    "hf": [0.00144165481499126, 0.00219401894565159, 0.00274712947857585],
    "Hf": [0.108857627078716, 0.604687253860082, 1.51425691532009],
    "qf": [97.5645656058974, 404.636952854234, 1001.75392545138],
    "mean": 491.080303830645,
}
FLEXSURV_PH = {
    "params": [241.914345580752, 1.32434893327669, -0.523883308254494],
    "sf": [0.896858093132105, 0.546245220742239, 0.21997156453964],
    "hf": [0.0014416548759319, 0.00219401905931361, 0.00274712963491383],
    "qf": [97.5645629147787, 404.63693849272, 1001.75388484744],
}
SURVREG_AFT = {
    "params": [241.914353104236, 1.32434892591891, -0.395578007455515],
    "qf": [97.5645656094961, 404.636952866554, 1001.75392547777],
}
RTOL = 1e-9


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return d["time"], d["sex"].reshape(-1, 1), d["c"]


def _at_r_estimate(fitter, lung, params):
    x, Z, c = lung
    return fitter.fit(x, Z, c=c).with_params(params).at([2.0])


def test_weibull_aft_at_sex_2_matches_flexsurv(lung):
    life = _at_r_estimate(sp.WeibullAFT, lung, FLEXSURV_AFT["params"])
    for f in ("sf", "hf", "Hf"):
        assert_allclose(getattr(life, f)(TIMES), FLEXSURV_AFT[f], rtol=RTOL)
    assert_allclose(life.qf(PROBS), FLEXSURV_AFT["qf"], rtol=RTOL)
    assert_allclose(life.mean(), FLEXSURV_AFT["mean"], rtol=RTOL)


def test_weibull_ph_at_sex_2_matches_flexsurv(lung):
    life = _at_r_estimate(sp.WeibullPH, lung, FLEXSURV_PH["params"])
    for f in ("sf", "hf"):
        assert_allclose(getattr(life, f)(TIMES), FLEXSURV_PH[f], rtol=RTOL)
    assert_allclose(life.qf(PROBS), FLEXSURV_PH["qf"], rtol=RTOL)


def test_weibull_aft_quantiles_match_survreg_predict(lung):
    life = _at_r_estimate(sp.WeibullAFT, lung, SURVREG_AFT["params"])
    assert_allclose(life.qf(PROBS), SURVREG_AFT["qf"], rtol=RTOL)


def test_own_fit_agrees_with_flexsurv_to_the_fit(lung):
    # SurPyval's own estimate, not R's: the two fits agree to their
    # optimisers' tolerances.
    x, Z, c = lung
    life = sp.WeibullAFT.fit(x, Z, c=c).at([2.0])
    assert_allclose(life.sf(TIMES), FLEXSURV_AFT["sf"], rtol=1e-5)


# -- the view ---------------------------------------------------------------

FITTERS = [
    sp.WeibullPH,
    sp.LogNormalAFT,
    sp.WeibullPO,
    sp.ExponentialAH,
    sp.WeibullDistributional,
    sp.CoxPH,
    sp.ProportionalOdds,
    sp.AdditiveHazards,
]


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    return df


@pytest.mark.parametrize("fitter", FITTERS, ids=lambda f: type(f).__name__)
def test_every_function_is_the_regressions_at_the_row(rossi, fitter):
    x = rossi["week"].values.astype(float)
    Z = rossi[["fin", "prio"]].values.astype(float)
    model = fitter.fit(x, Z, c=rossi["censored"].values)
    life = model.at(Z[4])
    t = np.array([3.0, 20.0, 45.0, 52.0])
    for f in ("sf", "ff", "df", "hf", "Hf"):
        assert np.array_equal(
            getattr(life, f)(t), getattr(model, f)(t, Z[4:5]), equal_nan=True
        ), f
    p = np.array([0.05, 0.2])
    assert np.array_equal(life.qf(p), model.qf(p, Z[4:5]), equal_nan=True)
    # random is qf(U), U drawn as Parametric.random draws it
    u = np.random.default_rng(11).uniform(size=200)
    expected = np.asarray(life.qf(u), dtype=float)
    expected = np.where(np.isnan(expected), np.inf, expected)
    assert_allclose(life.random(200, random_state=11), expected)
    np.random.seed(3)
    first = life.random(5)
    np.random.seed(3)
    assert_allclose(life.random(5), first)
    assert isinstance(life, LifeAtCovariates)
    assert "LifeAtCovariates" in repr(life)


def test_random_matches_a_univariate_model_draw_for_draw():
    # A regression with every coefficient 0 at Z is its baseline: the
    # same seed gives the same lifetimes as the univariate model.
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    Z = np.array([0.0, 1, 0, 1, 0, 1, 1, 0])
    model = sp.WeibullPH.fit(x, Z)
    life = model.with_params([10.0, 2.0, 0.0]).at([1.0])
    uni = sp.Weibull.from_params([10.0, 2.0])
    assert_allclose(
        life.random(6, random_state=4), uni.random(6, random_state=4)
    )
    np.random.seed(9)
    a = life.random(6)
    np.random.seed(9)
    assert_allclose(a, uni.random(6), rtol=1e-12)


def test_several_rows_are_refused_saying_what_to_do(rossi):
    x = rossi["week"].values.astype(float)
    Z = rossi[["fin", "prio"]].values.astype(float)
    model = sp.WeibullPH.fit(x, Z, c=rossi["censored"].values)
    with pytest.raises(ValueError, match="once per row"):
        model.at(Z[:3])
    one = sp.WeibullPH.fit(x, Z[:, :1], c=rossi["censored"].values)
    # A 1-D Z of a one-covariate model is one value per row
    with pytest.raises(ValueError, match="got 2 rows"):
        one.at([0.0, 1.0])
    assert one.at(1.0).sf(10.0) == one.sf(10.0, [1.0])


def test_bounds_mean_and_rmst_are_the_regressions(rossi):
    x = rossi["week"].values.astype(float)
    Z = rossi[["fin", "prio"]].values.astype(float)
    model = sp.WeibullAFT.fit(x, Z, c=rossi["censored"].values)
    life = model.at([1, 3])
    t = np.array([20.0, 52.0])
    assert_allclose(life.cb(t), model.cb(t, [1, 3]))
    assert_allclose(
        life.cb(t, on="Hf", alpha_ci=0.1),
        model.cb(t, [1, 3], on="Hf", alpha_ci=0.1),
    )
    assert_allclose(life.quantile_cb(0.1), model.quantile_cb(0.1, [1, 3]))
    assert life.mean() == model.mean([1, 3])
    assert life.rmst(52.0) == model.rmst(52.0, [1, 3])

    cox = sp.CoxPH.fit(x, Z, c=rossi["censored"].values)
    assert_allclose(cox.at([1, 3]).cb(t), cox.cb(t, [1, 3]))
    with pytest.raises(ValueError, match="rmst"):
        cox.at([1, 3]).mean()
    odds = sp.ProportionalOdds.fit(x, Z, c=rossi["censored"].values)
    with pytest.raises(ValueError, match="param_cb"):
        odds.at([1, 3]).cb(t)


def test_stratified_cox_and_dataframe_rows(rossi):
    x = rossi["week"].values.astype(float)
    Z = rossi[["fin", "prio"]].values.astype(float)
    strata = rossi["wexp"].values
    cox = sp.CoxPH.fit(x, Z, c=rossi["censored"].values, strata=strata)
    t = np.array([20.0, 52.0])
    life = cox.at([1, 3], stratum=1)
    assert_allclose(life.sf(t), cox.sf(t, [1, 3], stratum=1))
    assert "stratum=1" in repr(life)

    model = sp.WeibullPH.fit_from_df(
        rossi, x_col="week", c_col="censored", formula="fin + C(race)"
    )
    row = rossi.iloc[[3]]
    assert isinstance(row, pd.DataFrame)
    assert_allclose(model.at(row).sf(t), model.sf(t, row))
    with pytest.raises(ValueError, match="got 2 rows"):
        model.at(rossi.iloc[:2])


def test_at_a_with_params_model(rossi):
    # The two compose: a covariance draw, at a row, as a life model
    x = rossi["week"].values.astype(float)
    Z = rossi[["fin", "prio"]].values.astype(float)
    model = sp.WeibullPH.fit(x, Z, c=rossi["censored"].values, center=True)
    draw = np.random.default_rng(5).multivariate_normal(
        model._estimates(), model.covariance()
    )
    life = model.with_params(draw).at([1, 3])
    t = np.array([20.0, 52.0])
    assert_allclose(life.sf(t), model.with_params(draw).sf(t, [1, 3]))
    with pytest.raises(ValueError, match="from_params models"):
        life.cb(t)
