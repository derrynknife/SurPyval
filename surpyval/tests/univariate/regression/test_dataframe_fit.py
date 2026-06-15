"""
Tests for fitting and predicting parametric regression models directly from
pandas DataFrames, retaining covariate names and supporting formulas.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import AFT, PO, AcceleratedLife, Power, Weibull, WeibullPH
from surpyval.univariate.regression import CoxPH


def _make_df(seed=0, n=200):
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    sex = rng.choice(["M", "F"], n)
    beta = 0.03 * (age - 50) + np.where(sex == "M", 0.4, 0.0)
    x = Weibull.random(n, 10, 2) * np.exp(-beta / 2)
    c = np.zeros(n)
    return pd.DataFrame({"time": x, "age": age, "sex": sex, "censored": c})


PARAMETRIC_FITTERS = [
    WeibullPH,
    AFT(Weibull),
    PO(Weibull),
]


@pytest.mark.parametrize("fitter", PARAMETRIC_FITTERS)
def test_fit_from_df_keeps_feature_names(fitter):
    df = _make_df()
    model = fitter.fit_from_df(
        df, x_col="time", Z_cols=["age"], c_col="censored"
    )
    assert model.feature_names == ["age"]
    assert model.formula is None


@pytest.mark.parametrize("fitter", PARAMETRIC_FITTERS)
def test_predict_from_df_matches_array(fitter):
    df = _make_df()
    model = fitter.fit_from_df(
        df, x_col="time", Z_cols=["age"], c_col="censored"
    )
    sub = df[["age"]].head(3)
    from_df = model.sf([5, 10, 15], sub)
    from_arr = model.sf([5, 10, 15], sub.values)
    assert np.allclose(from_df, from_arr)


@pytest.mark.parametrize("fitter", PARAMETRIC_FITTERS)
def test_predict_from_df_selects_correct_columns(fitter):
    # The predict DataFrame has extra columns and a different order: the
    # model must still select the column it was trained on.
    df = _make_df()
    model = fitter.fit_from_df(
        df, x_col="time", Z_cols=["age"], c_col="censored"
    )
    reordered = df[["age"]].head(3)
    scrambled = pd.DataFrame(
        {
            "junk": [1, 2, 3],
            "age": reordered["age"].values,
            "sex": ["M", "F", "M"],
        }
    )
    assert np.allclose(model.sf([7], reordered), model.sf([7], scrambled))


def test_formula_with_categorical():
    df = _make_df()
    model = WeibullPH.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="censored"
    )
    assert model.formula == "age + sex"
    # Categorical 'sex' should be expanded into dummy columns.
    assert "age" in model.feature_names
    assert any(name.startswith("sex") for name in model.feature_names)

    # Prediction on a fresh DataFrame uses the same encoding.
    new_df = pd.DataFrame({"age": [60.0, 40.0], "sex": ["M", "F"]})
    sf = model.sf([5, 10], new_df)
    assert sf.shape == (2,)
    assert np.all((sf >= 0) & (sf <= 1))


def test_formula_and_z_cols_are_mutually_exclusive():
    df = _make_df()
    with pytest.raises(ValueError):
        WeibullPH.fit_from_df(df, x_col="time", Z_cols=["age"], formula="age")


def test_requires_z_cols_or_formula():
    df = _make_df()
    with pytest.raises(ValueError):
        WeibullPH.fit_from_df(df, x_col="time")


def test_unknown_column_raises():
    df = _make_df()
    with pytest.raises(ValueError):
        WeibullPH.fit_from_df(
            df, x_col="time", Z_cols=["not_a_column"], c_col="censored"
        )


def test_predicting_df_on_array_fit_model_raises():
    df = _make_df()
    model = WeibullPH.fit(
        x=df["time"].values,
        Z=df[["age"]].values,
        c=df["censored"].values,
    )
    with pytest.raises(ValueError):
        model.sf([5], df[["age"]].head(2))


def test_single_string_z_col():
    df = _make_df()
    model = WeibullPH.fit_from_df(
        df, x_col="time", Z_cols="age", c_col="censored"
    )
    assert model.feature_names == ["age"]
    assert np.isfinite(model.sf([5], df[["age"]].head(1))).all()


def test_accelerated_life_fit_from_df():
    stresses = np.repeat([1.0, 2.0, 3.0, 4.0], 40)
    x = Weibull.random(len(stresses), 100 / stresses, 3)
    df = pd.DataFrame(
        {"time": x, "stress": stresses, "censored": np.zeros_like(x)}
    )
    model = AcceleratedLife(Weibull, Power).fit_from_df(
        df, x_col="time", Z_cols=["stress"], c_col="censored"
    )
    assert model.feature_names == ["stress"]
    sub = df[["stress"]].head(2)
    assert np.allclose(model.sf([50], sub), model.sf([50], sub.values))


def test_coxph_formula_predict_from_df():
    df = _make_df()
    model = CoxPH.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="censored"
    )
    assert "age" in model.feature_names
    new_df = pd.DataFrame({"age": [60.0, 40.0], "sex": ["M", "F"]})
    sf = model.sf([5, 10], new_df)
    assert sf.shape == (2,)
    assert np.all((sf >= 0) & (sf <= 1))
