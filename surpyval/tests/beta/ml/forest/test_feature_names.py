"""Feature names on the survival tree and forest (#192).

A tree or forest fitted from a DataFrame (``fit_from_df`` with ``Z_cols``
or a ``formula``, or ``fit`` with a DataFrame ``Z``) keeps the covariate
names as ``feature_names``, as the regression models do; its split
descriptions, printout and permutation importances are named by them, it
predicts from a DataFrame by them, and they survive serialisation. Fitted
from an array it has none (``None``), and the features are shown as
``Z0``, ``Z1``, ...
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "temp": rng.uniform(0, 100, n),
            "load": rng.uniform(0, 1, n),
            "site": rng.choice(["a", "b"], n),
        }
    )
    df["x"] = rng.weibull(2.0, n) * np.where(df["temp"] > 42, 5.0, 10.0)
    df["c"] = (df["x"] > 9).astype(int)
    df["x"] = np.minimum(df["x"], 9)
    return df


def _tree(df, **options):
    options = {"max_depth": 1, "n_features_split": "all", **options}
    return SurvivalTree.fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["temp", "load"], **options
    )


@pytest.mark.parametrize("kind", ["weibull", "exponential", "non-parametric"])
def test_split_description_uses_the_feature_name(kind):
    df = _frame()
    tree = _tree(df, kind=kind)
    assert tree.feature_names == ["temp", "load"]
    root = tree._root
    assert root.split_feature_index == 0
    rule = root.describe(tree.feature_labels)
    assert rule.startswith("temp <= ")
    assert abs(float(rule.split("<=")[1]) - 42) < 3
    text = repr(tree)
    assert "|--- temp <= " in text and "|--- temp >  " in text
    assert "Z0" not in text


def test_array_fit_has_no_names_and_shows_column_labels():
    df = _frame()
    tree = SurvivalTree.fit(
        df["x"].to_numpy(),
        df[["temp", "load"]].to_numpy(),
        c=df["c"].to_numpy(),
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    assert tree.feature_names is None
    assert tree.feature_labels == ["Z0", "Z1"]
    assert "|--- Z0 <= " in repr(tree)


def test_fit_with_a_dataframe_Z_is_fit_from_df():
    df = _frame()
    a = _tree(df, kind="exponential")
    b = SurvivalTree.fit(
        df["x"],
        df[["temp", "load"]],
        c=df["c"],
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    assert b.feature_names == ["temp", "load"]
    assert repr(a) == repr(b)


def test_dataframe_prediction_reads_columns_by_name():
    df = _frame()
    tree = _tree(df, kind="exponential")
    query = pd.DataFrame({"load": [0.5, 0.5], "temp": [10.0, 90.0]})
    expected = tree.sf([2.0, 5.0], [[10.0, 0.5], [90.0, 0.5]])
    np.testing.assert_array_equal(tree.sf([2.0, 5.0], query), expected)


def test_formula_features():
    df = _frame()
    tree = SurvivalTree.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        formula="temp + C(site)",
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    assert tree.feature_names[0] == "temp"
    assert len(tree.feature_names) == 2
    assert tree.formula == "temp + C(site)"
    # Predicts from the raw columns through the formula
    query = df.head(3)
    expected = tree.sf(
        2.0, np.column_stack([query["temp"], query["site"] == "b"])
    )
    np.testing.assert_array_equal(tree.sf(2.0, query), expected)
    restored = surpyval.from_dict(json.loads(json.dumps(tree.to_dict())))
    assert restored.feature_names == tree.feature_names
    np.testing.assert_array_equal(restored.sf(2.0, query), expected)


def test_interval_columns():
    df = _frame()
    df["xl"] = df["x"] * 0.9
    df["xr"] = df["x"] * 1.1
    df["ci"] = 2
    tree = SurvivalTree.fit_from_df(
        df,
        xl_col="xl",
        xr_col="xr",
        c_col="ci",
        Z_cols=["temp", "load"],
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    same = SurvivalTree.fit(
        xl=df["xl"],
        xr=df["xr"],
        c=df["ci"],
        Z=df[["temp", "load"]],
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    assert repr(tree) == repr(same)


@pytest.mark.parametrize(
    "columns",
    [
        {"x_col": "x", "xl_col": "x", "xr_col": "x"},
        {"xl_col": "x"},
        {},
    ],
)
def test_times_given_exactly_once(columns):
    with pytest.raises(ValueError, match="times exactly once"):
        SurvivalTree.fit_from_df(_frame(), Z_cols="temp", **columns)


def test_covariates_given_exactly_once():
    with pytest.raises(ValueError, match="covariates exactly once"):
        SurvivalTree.fit_from_df(_frame(), x_col="x")
    with pytest.raises(ValueError, match="covariates exactly once"):
        SurvivalTree.fit_from_df(
            _frame(), x_col="x", Z_cols="temp", formula="temp"
        )


def test_names_survive_serialisation():
    df = _frame()
    tree = _tree(df, kind="weibull")
    restored = surpyval.from_dict(json.loads(json.dumps(tree.to_dict())))
    assert restored.feature_names == ["temp", "load"]
    # The printout, leaf sizes included, survives too
    assert restored.describe() == tree.describe()
    assert "units" in restored.describe()


def test_restored_array_tree_labels():
    df = _frame()
    tree = SurvivalTree.fit(
        df["x"].to_numpy(),
        df[["temp", "load"]].to_numpy(),
        max_depth=1,
        n_features_split="all",
        kind="exponential",
    )
    restored = SurvivalTree.from_dict(tree.to_dict())
    assert restored.feature_names is None
    assert "|--- Z0 <= " in repr(restored)


def test_forest_importances_keyed_by_name():
    df = _frame()
    forest = RandomSurvivalForest.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        Z_cols=["temp", "load"],
        n_trees=10,
        max_depth=1,
        kind="exponential",
        random_state=0,
    )
    assert forest.feature_names == ["temp", "load"]
    assert all(tree.feature_names == ["temp", "load"] for tree in forest.trees)
    importances = forest.feature_importances(random_state=1)
    assert isinstance(importances, pd.Series)
    assert list(importances.index) == ["temp", "load"]
    assert importances["temp"] > 0.05 > abs(importances["load"])
    assert "temp" in repr(forest)

    # Same forest from arrays: the same numbers, keyed Z0, Z1
    plain = RandomSurvivalForest.fit(
        df["x"].to_numpy(),
        df[["temp", "load"]].to_numpy(),
        c=df["c"].to_numpy(),
        n_trees=10,
        max_depth=1,
        kind="exponential",
        random_state=0,
    )
    plain_importances = plain.feature_importances(random_state=1)
    assert list(plain_importances.index) == ["Z0", "Z1"]
    np.testing.assert_array_equal(
        plain_importances.to_numpy(), importances.to_numpy()
    )

    query = pd.DataFrame({"load": [0.5], "temp": [90.0]})
    np.testing.assert_array_equal(
        forest.sf([1.0, 3.0], query), plain.sf([1.0, 3.0], [[90.0, 0.5]])
    )
    restored = RandomSurvivalForest.from_dict(
        json.loads(json.dumps(forest.to_dict()))
    )
    assert restored.feature_names == ["temp", "load"]
    assert restored.feature_labels == ["temp", "load"]
    np.testing.assert_array_equal(
        restored.sf([1.0, 3.0], query), forest.sf([1.0, 3.0], query)
    )
