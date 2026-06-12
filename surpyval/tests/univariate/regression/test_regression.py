"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import numpy as np

from surpyval.datasets import load_lung, load_rossi_static
from surpyval.univariate.regression import CoxPH


def test_coxph_against_ll_rossi_static():
    ll_answer = np.array(
        [
            -0.37942216,
            -0.05743772,
            0.31389978,
            -0.14979572,
            -0.43370385,
            -0.08487107,
            0.09149708,
        ]
    )

    rossi = load_rossi_static()
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="arrest",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        method="efron",
    )

    assert np.allclose(model.beta, ll_answer)


# Examples taken from:
# http://www.sthda.com/english/wiki/cox-proportional-hazards-model


def test_coxph_against_r_lung_1():
    r_answer = np.array([-0.5310235])

    lung = load_lung()
    x = lung["time"].values
    c = lung["status"].values
    Z = lung[["sex"]].values

    model = CoxPH.fit(x=x, Z=Z, c=c, method="efron")

    assert np.allclose(model.beta, r_answer)


def test_coxph_against_r_lung_2():
    r_answer = np.array([0.01106676, -0.55261240, 0.46372848])

    lung = load_lung()
    model = CoxPH.fit_from_df(
        lung,
        x_col="time",
        c_col="status",
        Z_cols=["age", "sex", "ph.ecog"],
        method="efron",
    )

    assert np.allclose(model.beta, r_answer)


def test_breslow_betas_rossi():
    # Hardcoded breslow answer for the Rossi dataset.
    # lifelines does not support breslow for this dataset, so the expected
    # values were generated from surpyval and kept as a regression guard.
    expected = np.array(
        [
            -0.37902189,
            -0.05724593,
            0.31412977,
            -0.15111460,
            -0.43278257,
            -0.08498284,
            0.09111154,
        ]
    )

    rossi = load_rossi_static()
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="arrest",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        method="breslow",
    )

    assert np.allclose(model.beta, expected)


def test_breslow_p_values_rossi():
    # Breslow method returns a p_value per covariate; Efron returns None.
    rossi = load_rossi_static()
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="arrest",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        method="breslow",
    )

    assert model.p_values is not None
    assert model.p_values.shape == (7,)
    assert np.all((model.p_values >= 0) & (model.p_values <= 1))


def test_formula_interface_matches_Z_cols():
    # fit_from_df with formula= must give the same betas as Z_cols=.
    # Formulaic preserves the order covariates appear in the formula, so
    # the betas can be compared directly.
    rossi = load_rossi_static()
    Z_cols = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

    model_z = CoxPH.fit_from_df(
        rossi, x_col="week", c_col="arrest", Z_cols=Z_cols, method="efron"
    )
    model_f = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="arrest",
        formula="fin + age + race + wexp + mar + paro + prio",
        method="efron",
    )

    assert np.allclose(model_z.beta, model_f.beta)


def test_parametric_ph_aic_bic():
    # ParametricRegressionModel.aic/bic crashed: the PH fitter never set
    # model.k, and bic/aic_c indexed SurpyvalData like a dict.
    from surpyval import Weibull, WeibullPH

    np.random.seed(1)
    n = 100
    Z = np.random.binomial(1, 0.5, n).reshape(-1, 1)
    x = Weibull.random(n, 10, 2) * np.exp(-0.5 * Z[:, 0])
    model = WeibullPH.fit(x=x, Z=Z)

    assert model.k == 3
    assert np.isfinite(model.aic())
    assert np.isfinite(model.aic_c())
    assert np.isfinite(model.bic())
    assert model.aic() < model.aic_c()
