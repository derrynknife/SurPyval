"""The coefficient table of the Cox and parametric regression models
(#484): ``summary()`` and the ``repr`` that prints it.

The repr printed ``beta_0 ... beta_6`` and the values alone, even for a
model fitted from a DataFrame with named columns, and a parametric model
printed the Weibull shape ``beta`` next to the coefficients ``beta_i``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval import Weibull
from surpyval.tests._helpers import rossi_with_censoring
from surpyval.univariate.regression import AcceleratedLife
from surpyval.univariate.regression.accelerated_life import Power

COLS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
COLUMNS = [
    "coef",
    "exp(coef)",
    "se(coef)",
    "coef lower 95%",
    "coef upper 95%",
    "exp(coef) lower 95%",
    "exp(coef) upper 95%",
    "z",
    "p",
]


def _cox():
    return sp.CoxPH.fit_from_df(
        rossi_with_censoring(), x_col="week", c_col="censored", Z_cols=COLS
    )


def test_cox_summary_matches_r():
    # R: summary(coxph(Surv(week, arrest) ~ fin + age + race + wexp + mar
    # + paro + prio, data = rossi)) -- the censoring flag is
    # 1 - arrest (#479).
    table = _cox().summary()
    assert list(table.columns) == COLUMNS
    assert list(table.index) == COLS
    fin = table.loc["fin"]
    assert fin["coef"] == pytest.approx(-0.37942, abs=1e-5)
    assert fin["exp(coef)"] == pytest.approx(0.68426, abs=1e-5)
    assert fin["se(coef)"] == pytest.approx(0.19138, abs=1e-5)
    assert fin["z"] == pytest.approx(-1.983, abs=1e-3)
    assert fin["p"] == pytest.approx(0.04742, abs=1e-5)
    assert fin["exp(coef) lower 95%"] == pytest.approx(0.4702, abs=1e-4)
    assert fin["exp(coef) upper 95%"] == pytest.approx(0.9957, abs=1e-4)
    q = norm.ppf(0.975)
    np.testing.assert_allclose(
        table["coef lower 95%"], table["coef"] - q * table["se(coef)"]
    )
    np.testing.assert_allclose(
        table["exp(coef) upper 95%"], np.exp(table["coef upper 95%"])
    )


def test_cox_robust_summary():
    model = _cox()
    robust = model.summary(robust=True)
    np.testing.assert_allclose(
        robust["se(coef)"], model.robust_summary()["se"], rtol=1e-12
    )
    assert not np.allclose(robust["se(coef)"], model.summary()["se(coef)"])
    level = model.summary(alpha_ci=0.1)
    assert "coef lower 90%" in level.columns


def test_cox_repr_names_the_covariates():
    model = _cox()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = repr(model)
    assert "coef_0" not in text
    assert "exp(coef) is the hazard ratio" in text
    fin = next(
        line for line in text.splitlines() if line.split()[:1] == ["fin"]
    )
    # coef, exp(coef), se, lower, upper, z, p
    assert fin.split() == [
        "fin",
        "-0.3794",
        "0.6843",
        "0.1914",
        "-0.7545",
        "-0.004325",
        "-1.983",
        "0.04742",
    ]
    x = rossi_with_censoring()
    unnamed = sp.CoxPH.fit(x.week, x[COLS].to_numpy(), x.censored)
    assert "coef_6" in repr(unnamed)


def test_cox_standard_errors_are_saved():
    model = _cox()
    back = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    pd.testing.assert_frame_equal(back.summary(), model.summary())
    assert repr(back) == repr(model)
    # A dict saved before they were: they follow from the p-values.
    old = model.to_dict()
    del old["se"]
    np.testing.assert_allclose(
        sp.from_dict(old).summary()["se(coef)"],
        model.summary()["se(coef)"],
        rtol=1e-6,
    )


def test_parametric_summary_and_repr_separate_the_baseline():
    df = rossi_with_censoring()
    model = sp.WeibullPH.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    )
    table = model.summary()
    assert list(table.columns) == COLUMNS
    assert list(table.index) == [
        ("baseline", "alpha"),
        ("baseline", "beta"),
        ("coefficients", "fin"),
        ("coefficients", "age"),
        ("coefficients", "prio"),
    ]
    se = model.standard_errors()
    np.testing.assert_allclose(table["se(coef)"], se, rtol=1e-12)
    np.testing.assert_allclose(table["coef"], model.params, rtol=1e-12)
    # The baseline's interval is param_cb's, inside the support.
    np.testing.assert_allclose(
        table.loc[("baseline", "alpha"), ["coef lower 95%", "coef upper 95%"]],
        model.param_cb("alpha"),
    )
    assert np.isnan(table.loc["baseline", "p"]).all()
    coefs = table.loc["coefficients"]
    np.testing.assert_allclose(coefs["exp(coef)"], np.exp(coefs["coef"]))
    np.testing.assert_allclose(
        coefs["p"], 2 * norm.sf(np.abs(coefs["coef"] / coefs["se(coef)"]))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = repr(model)
    baseline, coefficients = text.split("Coefficients")
    # The Weibull shape is in the baseline block, the covariates below.
    assert "    beta " in baseline and "fin" not in baseline
    assert "fin" in coefficients and "beta" not in coefficients
    assert "exp(coef) is the hazard ratio" in coefficients


def test_parametric_links_without_a_ratio():
    df = rossi_with_censoring()
    x, c = df.week.to_numpy(), df.censored.to_numpy()
    additive = sp.WeibullAH.fit(x, df[["fin"]].to_numpy(), c)
    assert np.isnan(additive.summary()["exp(coef)"]).all()
    assert "exp(coef)" not in repr(additive).split("Coefficients")[1]
    aft = sp.WeibullAFT.fit(x, df[["fin"]].to_numpy(), c)
    assert "exp(coef) is the acceleration factor" in repr(aft)
    rng = np.random.default_rng(0)
    V = rng.choice([1.0, 2.0, 4.0], 100)
    life = AcceleratedLife(Weibull, Power).fit(
        10 * V**-1.5 * rng.weibull(2, 100), Z=V
    )
    table = life.summary()
    assert list(table.index.get_level_values(0).unique()) == [
        "baseline",
        "life model",
    ]
    text = repr(life)
    assert "Life model" in text and "Coefficients" not in text


def test_fixed_and_aliased_parameters():
    df = rossi_with_censoring()
    x, c = df.week.to_numpy(), df.censored.to_numpy()
    model = sp.WeibullPH.fit(x, df[["fin"]].to_numpy(), c, fixed={"beta": 1.4})
    row = model.summary().loc[("baseline", "beta")]
    assert row["coef"] == 1.4 and np.isnan(row["se(coef)"])
    assert "Fixed               : beta = 1.4" in repr(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aliased = sp.WeibullPH.fit(
            x, np.c_[df[["fin"]].to_numpy(), np.ones(len(x))], c
        )
    assert aliased.summary().loc[("coefficients", "coef_1")].isna().all()


def _frailty():
    df = rossi_with_censoring()
    df["grp"] = np.arange(len(df)) % 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.WeibullFrailty.fit_from_df(
            df,
            x_col="week",
            c_col="censored",
            Z_cols=["fin", "age"],
            group_col="grp",
        )


def test_frailty_summary_is_a_table():
    # FrailtyModel.summary() returned the text of its repr; it is now the
    # table the other regression models give.
    model = _frailty()
    table = model.summary()
    assert list(table.columns) == COLUMNS
    assert list(table.index) == [
        ("baseline", "alpha"),
        ("baseline", "beta"),
        ("coefficients", "fin"),
        ("coefficients", "age"),
        ("frailty", "theta"),
    ]
    np.testing.assert_allclose(table["se(coef)"], model.standard_errors())
    np.testing.assert_allclose(
        table["coef"],
        np.concatenate([model.dist_params, model.beta, [model.theta]]),
    )
    # The baseline's intervals are param_cb's, which stay positive.
    np.testing.assert_allclose(
        table.loc[("baseline", "alpha"), ["coef lower 95%", "coef upper 95%"]],
        model.param_cb("alpha"),
    )
    assert "fin" in repr(model) and "Frailty variance" in repr(model)


def test_frailty_summary_without_covariance():
    model = _frailty()
    model._covariance = None
    table = model.summary()
    assert table["se(coef)"].isna().all()
    assert np.isfinite(table["coef"]).all()
    repr(model)


# ---------------------------------------------------------------------------
# #662: p_values, summary() and the information criteria on every family
# ---------------------------------------------------------------------------


def test_662_parametric_and_frailty_p_values_match_the_summary():
    df = rossi_with_censoring()
    x, c = df.week.to_numpy(), df.censored.to_numpy()
    model = sp.WeibullPH.fit(x, df[["fin", "age"]].to_numpy(), c)
    p = model.p_values
    assert np.isnan(p[:2]).all()
    np.testing.assert_allclose(p[2:], model.summary()["p"].iloc[2:])
    se = model.standard_errors()[2:]
    np.testing.assert_allclose(
        p[2:], 2 * norm.sf(np.abs(model.params[2:] / se)), rtol=1e-12
    )
    frailty = _frailty()
    np.testing.assert_allclose(
        frailty.p_values, frailty.summary()["p"].to_numpy(), equal_nan=True
    )
    assert np.isfinite(frailty.p_values[2:4]).all()
    assert np.isnan(frailty.p_values[[0, 1, 4]]).all()


def test_662_accelerated_life_summary_tests_the_unbounded_parameters():
    # The exp(coef), bounds, z and p columns were all nan; a power (or an
    # activation energy) is tested against 0, no stress effect, and a
    # positive constant is not.
    rng = np.random.default_rng(0)
    V = rng.choice([1.0, 2.0, 4.0], 100)
    life = AcceleratedLife(Weibull, Power).fit(
        10 * V**-1.5 * rng.weibull(2, 100), Z=V
    )
    table = life.summary()
    assert list(table.columns) == COLUMNS
    n_row = table.loc[("life model", "n")]
    assert n_row["z"] == pytest.approx(n_row["coef"] / n_row["se(coef)"])
    assert n_row["p"] == pytest.approx(life.p_values[3])
    assert np.isnan(table.loc[("life model", "a"), ["z", "p"]]).all()
    assert np.isnan(table.loc[("baseline", "beta"), ["z", "p"]]).all()
    assert np.isnan(life.p_values[[0, 1, 2]]).all()


def test_662_semi_parametric_summaries_are_tables():
    df = rossi_with_censoring()
    x, c = df.week.to_numpy(), df.censored.to_numpy()
    Z = df[["fin", "age"]].to_numpy()
    ah = sp.AdditiveHazards.fit(x, Z, c=c)
    table = ah.summary()
    assert list(table.columns) == COLUMNS
    np.testing.assert_allclose(table["p"], ah.p_values)
    np.testing.assert_allclose(table["se(coef)"], ah.standard_errors())
    assert table["exp(coef)"].isna().all()  # an excess hazard, not a ratio
    bj = sp.BuckleyJames.fit(x, Z, c=c)
    table = bj.summary()
    assert list(table.columns) == COLUMNS
    np.testing.assert_allclose(table["exp(coef)"], np.exp(bj.beta))
    assert table[["se(coef)", "z", "p", "coef lower 95%"]].isna().all().all()
    boot = bj.summary(n_boot=20, random_state=0)
    np.testing.assert_allclose(
        boot[["coef lower 95%", "coef upper 95%"]],
        bj.bootstrap_ci(n_boot=20, random_state=0),
    )
    rp = sp.RoystonParmar.fit(x, c=c, df=2)
    table = rp.summary()
    assert list(table.columns) == COLUMNS
    np.testing.assert_allclose(table["se(coef)"], rp.standard_errors())
    assert repr(rp).startswith("Royston-Parmar Flexible Parametric Model")


@pytest.mark.parametrize("fitter", [sp.AdditiveHazards, sp.BuckleyJames])
def test_662_no_likelihood_says_why(fitter):
    df = rossi_with_censoring()
    x, c = df.week.to_numpy(), df.censored.to_numpy()
    model = fitter.fit(x, df[["fin"]].to_numpy(), c=c)
    for name in ("neg_ll", "aic", "bic", "aic_c"):
        with pytest.raises(NotImplementedError, match="no likelihood"):
            getattr(model, name)()
    with pytest.raises(NotImplementedError, match="no likelihood"):
        model.log_likelihood
    assert not hasattr(model, "log_likelihood")


def test_662_cox_frailty_param_cb_takes_method():
    df = rossi_with_censoring()
    df["grp"] = np.arange(len(df)) % 40
    model = sp.CoxFrailty.fit(
        df.week, Z=df[["fin"]].to_numpy(), c=df.censored, groups=df.grp
    )
    wald = model.param_cb("coef_0")
    np.testing.assert_array_equal(model.param_cb("coef_0", method=None), wald)
    np.testing.assert_array_equal(
        model.param_cb("coef_0", method="wald"), wald
    )
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.param_cb("coef_0", method="lr")
    with pytest.raises(ValueError, match="method"):
        model.param_cb("coef_0", method="wold")
