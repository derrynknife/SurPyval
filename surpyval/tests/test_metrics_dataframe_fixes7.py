"""
``survival_probability`` with a DataFrame of covariates (#375 item 8c).

The helper cast ``Z`` to float before calling ``model.sf``, so a model fitted
with a formula over a string-valued factor could not be scored at all ("could
not convert string to float"). A DataFrame is now passed to ``model.sf``
untouched; arrays are unchanged.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.metrics import integrated_brier_score, survival_probability

TIMES = np.array([2.0, 5.0, 8.0])


def _df(seed=0, n=150):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], size=n)
    eff = 0.5 * z + np.where(g == "b", 0.6, np.where(g == "c", -0.6, 0.0))
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    cens = rng.uniform(5, 30, n)
    return pd.DataFrame(
        {
            "x": np.minimum(t, cens),
            "c": (cens < t).astype(int),
            "z": z,
            "g": g,
        }
    )


def _by_time(model, Z, times):
    # The matrix built directly from the model, one time at a time
    n = len(Z)
    return np.column_stack([model.sf(np.full(n, t), Z) for t in times])


@pytest.mark.parametrize("fitter", ["WeibullPH", "CoxPH", "AdditiveHazards"])
def test_formula_fit_with_string_levels_is_scored(fitter):
    train, test = _df(0), _df(1, n=40)
    model = getattr(sp, fitter).fit_from_df(
        train, x_col="x", c_col="c", formula="z + g"
    )
    Z = test[["z", "g"]]
    # Old code: ValueError, could not convert string to float
    S = survival_probability(model, Z, TIMES)
    assert S.shape == (40, TIMES.size)
    np.testing.assert_allclose(S, _by_time(model, Z, TIMES))
    ibs = integrated_brier_score(test["x"], test["c"], S, TIMES)
    assert np.isfinite(ibs)


def test_formula_fit_missing_level_scores_nan_row():
    train = _df(0)
    model = sp.WeibullPH.fit_from_df(
        train, x_col="x", c_col="c", formula="z + g"
    )
    Z = pd.DataFrame(
        {"z": [0.1, 0.2, np.nan], "g": pd.Series([None, "b", "c"])}
    )
    S = survival_probability(model, Z, TIMES)
    assert np.isnan(S[[0, 2]]).all()
    np.testing.assert_allclose(S[1], _by_time(model, Z.iloc[[1]], TIMES)[0])


def test_named_column_fit_reads_dataframe_by_name():
    train, test = _df(0), _df(1, n=30)
    train["w"] = 1.0
    test["w"] = 1.0
    model = sp.WeibullPH.fit_from_df(
        train, x_col="x", c_col="c", Z_cols=["z", "w"]
    )
    # Columns in another order, and an extra one: read by name
    S = survival_probability(model, test[["g", "w", "z"]], TIMES)
    np.testing.assert_allclose(
        S, survival_probability(model, test[["z", "w"]].values, TIMES)
    )


def test_arrays_unchanged():
    df = _df(0)
    Z = df[["z"]].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullPH.fit(df["x"].values, Z, df["c"].values)
    S = survival_probability(model, Z[:10], TIMES)
    np.testing.assert_allclose(S, _by_time(model, Z[:10], TIMES))
    # A 1-D array is one covariate, and a list works too
    np.testing.assert_allclose(
        survival_probability(model, Z[:10, 0], TIMES), S
    )
    np.testing.assert_allclose(
        survival_probability(model, Z[:10].tolist(), TIMES), S
    )
