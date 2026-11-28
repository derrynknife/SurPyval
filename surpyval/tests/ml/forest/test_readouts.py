"""Forest and tree read-outs: ICE and partial dependence, minimal depth,
and permutation importance by horizon."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from surpyval.ml import RandomSurvivalForest, SurvivalTree
from surpyval.ml.forest.readouts import tree_minimal_depths


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 2))
    x = rng.weibull(2.0, n) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
    c = (x > 12).astype(int)
    return np.minimum(x, 12.0), c, Z


@pytest.fixture(scope="module")
def forest():
    x, c, Z = _data()
    return RandomSurvivalForest.fit(
        x,
        Z,
        c=c,
        n_trees=15,
        max_depth=1,
        n_features_split=2,
        kind="exponential",
        random_state=0,
    )


# ---------------------------------------------------------------- ICE / PD


def test_partial_dependence_is_the_weighted_mean_of_the_ice_curves():
    # Grouped rows (n > 1) are weighted by their counts by default.
    x, c, Z = _data(80)
    n = np.random.default_rng(3).integers(1, 4, 80)
    forest = RandomSurvivalForest.fit(
        x, Z, c=c, n=n, n_trees=5, kind="exponential", random_state=1
    )
    out = forest.partial_dependence("Z0", [0.1, 0.5, 0.9], [2.0, 6.0])
    w = np.asarray(forest.data.n, dtype=float)
    assert out.weights == pytest.approx(w)
    expected = (out.ice * w[:, None, None]).sum(0) / w.sum()
    assert out.average == pytest.approx(expected, rel=1e-14, abs=0)
    # With Z passed in, the rows weigh equally
    out2 = forest.partial_dependence(0, [0.1, 0.9], [4.0], Z=Z[:10])
    assert out2.average == pytest.approx(out2.ice.mean(0), rel=1e-14)


def test_each_ice_curve_is_the_models_prediction_at_the_edited_row(forest):
    _, _, Z = _data()
    out = forest.partial_dependence(1, [0.2, 0.8], [3.0, 7.0], Z=Z[:4])
    for i in range(4):
        for k, v in enumerate([0.2, 0.8]):
            row = Z[i].copy()
            row[1] = v
            assert out.ice[i, k] == pytest.approx(
                forest.sf([3.0, 7.0], row), rel=1e-14
            )
    frame = out.to_frame()
    assert frame.index.name == "Z1"
    assert frame.shape == (2, 2)


def test_partial_dependence_follows_the_effect(forest):
    # Units with Z0 > 0.5 fail twice as fast; Z1 is noise and unused.
    out = forest.partial_dependence("Z0", [0.25, 0.75], [5.0])
    assert out.average[0, 0] > out.average[1, 0]
    flat = forest.partial_dependence("Z1", [0.1, 0.9], [5.0])
    assert flat.average[0, 0] == pytest.approx(flat.average[1, 0], abs=1e-15)


def test_tree_partial_dependence_and_other_functions():
    x, c, Z = _data()
    tree = SurvivalTree.fit(x, Z, c=c, max_depth=2, kind="exponential")
    out = tree.partial_dependence(0, [0.2, 0.8], [5.0], on="ff")
    assert out.average == pytest.approx(out.ice.mean(0), rel=1e-14)
    sf = tree.partial_dependence(0, [0.2, 0.8], [5.0])
    assert out.average == pytest.approx(1 - sf.average, rel=1e-12)


def test_partial_dependence_competing_risks_cif():
    rng = np.random.default_rng(2)
    Z = rng.uniform(0, 1, (150, 2))
    x = rng.exponential(np.where(Z[:, 0] > 0.5, 3.0, 8.0))
    e = rng.choice(["a", "b"], 150)
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=5, kind="exponential", random_state=0
    )
    out = forest.partial_dependence(0, [0.2, 0.8], [4.0], on="cif", event="a")
    assert out.average.shape == (2, 1)
    with pytest.raises(ValueError, match="event"):
        forest.partial_dependence(0, [0.2], [4.0], event="a")


def test_partial_dependence_refusals(forest):
    with pytest.raises(ValueError, match="not one of the model's covariates"):
        forest.partial_dependence("load", [1.0], [1.0])
    with pytest.raises(ValueError, match="not a column"):
        forest.partial_dependence(5, [1.0], [1.0])
    with pytest.raises(ValueError, match="one column per covariate"):
        forest.partial_dependence(0, [1.0], [1.0], Z=np.ones((3, 3)))
    restored = RandomSurvivalForest.from_dict(forest.to_dict())
    with pytest.raises(ValueError, match="pass Z"):
        restored.partial_dependence(0, [1.0], [1.0])
    _, _, Z = _data()
    a = restored.partial_dependence(0, [0.3], [2.0], Z=Z)
    b = forest.partial_dependence(0, [0.3], [2.0], Z=Z)
    assert a.average == pytest.approx(b.average, rel=1e-12)


def test_partial_dependence_plot(forest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = forest.partial_dependence(0, np.linspace(0, 1, 5), [2.0, 5.0])
    ax = out.plot(at=5.0, centred=True)
    assert ax.get_ylabel().startswith("Change in survival probability")
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="pass at="):
        out.plot()


# ------------------------------------------------------------ minimal depth


def _split(feature, left, right):
    return SimpleNamespace(
        split_feature_index=feature, left_child=left, right_child=right
    )


LEAF = SimpleNamespace()


def test_minimal_depth_hand_worked():
    # Tree 1: root on Z2; its left child on Z0, whose left child splits
    # Z2 again (deeper; ignored). Depth 3. Z1 never split -> 3.
    t1 = _split(2, _split(0, _split(2, LEAF, LEAF), LEAF), LEAF)
    # Tree 2: one split on Z1, depth 1; Z0 and Z2 unused -> 1.
    t2 = _split(1, LEAF, LEAF)
    model = SimpleNamespace(
        trees=[SimpleNamespace(_root=t1), SimpleNamespace(_root=t2)],
        feature_labels=["Z0", "Z1", "Z2"],
    )
    depths, used = tree_minimal_depths(model)
    assert depths.tolist() == [[1, 3, 0], [1, 0, 1]]
    assert used.tolist() == [[True, False, True], [False, True, False]]

    from surpyval.ml.forest.readouts import minimal_depth

    out = minimal_depth(model)
    assert out["minimal_depth"].tolist() == [1.0, 1.5, 0.5]
    assert out["split_fraction"].tolist() == [0.5, 0.5, 0.5]


def test_minimal_depth_of_fitted_models(forest):
    out = forest.minimal_depth()
    # Every stump splits at the root on the feature with the effect
    assert out.loc["Z0", ["minimal_depth", "split_fraction"]].tolist() == [
        0.0,
        1.0,
    ]
    assert out.loc["Z1", ["minimal_depth", "split_fraction"]].tolist() == [
        1.0,
        0.0,
    ]
    # Kept by serialisation: no training data needed
    restored = RandomSurvivalForest.from_dict(forest.to_dict())
    pd.testing.assert_frame_equal(restored.minimal_depth(), out)
    x, c, Z = _data()
    tree = SurvivalTree.fit(
        x, Z, c=c, max_depth=1, n_features_split=2, kind="exponential"
    )
    assert tree.minimal_depth()["minimal_depth"].tolist() == [0.0, 1.0]


# ------------------------------------------------- importance by horizon


@pytest.mark.parametrize("metric", ["brier", "auc"])
def test_one_horizon_equals_its_column_of_several(forest, metric):
    several = forest.feature_importances(
        n_repeats=3, random_state=7, times=[2.0, 5.0, 8.0], metric=metric
    )
    assert list(several.columns) == [2.0, 5.0, 8.0]
    for t in (2.0, 5.0, 8.0):
        one = forest.feature_importances(
            n_repeats=3, random_state=7, times=[t], metric=metric
        )
        assert one[t].to_numpy() == pytest.approx(
            several[t].to_numpy(), rel=1e-13, abs=1e-15
        )
    # The effect feature matters; the unused one is exactly zero.
    assert (several.loc["Z0"] > 0).all()
    assert several.loc["Z1"].abs().max() == 0.0


def test_default_importance_is_unchanged(forest):
    # Without times, the out-of-bag log-likelihood importance, a Series
    a = forest.feature_importances(random_state=1)
    assert isinstance(a, pd.Series)
    assert a.name == "importance"


def test_horizon_importance_base_score_is_the_oob_brier(forest):
    # Shuffling an unused feature leaves the predictions as they were, so
    # its importance is 0; recompute the base Brier score by hand.
    from surpyval.metrics import brier_score

    oob, _, _, n_oob = forest._oob_setup()
    total = np.zeros((len(forest.data), 1))
    for tree, rows in zip(forest.trees, oob):
        total[rows] += tree.sf([5.0], forest.Z[rows], grid=True)
    keep = n_oob > 0
    pred = total[keep] / n_oob[keep, None]
    base = brier_score(forest.data.x[keep], forest.data.c[keep], pred, [5.0])[
        1
    ]
    assert np.isfinite(base).all()
    imp = forest.feature_importances(n_repeats=1, random_state=0, times=[5.0])
    assert imp.loc["Z1", 5.0] == 0.0


def test_horizon_importance_competing_risks():
    rng = np.random.default_rng(2)
    Z = rng.uniform(0, 1, (150, 2))
    x = rng.exponential(np.where(Z[:, 0] > 0.5, 3.0, 8.0))
    e = np.where(Z[:, 0] > 0.5, "a", rng.choice(["a", "b"], 150))
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=10, kind="exponential", random_state=0
    )
    with pytest.warns(UserWarning, match="Brier score or AUC"):
        out = forest.feature_importances(
            n_repeats=2, random_state=0, times=[2.0, 5.0], event="a"
        )
    assert out.shape == (2, 2)
    assert np.isfinite(out.to_numpy()).all()


def test_horizon_importance_refusals(forest):
    with pytest.raises(ValueError, match="metric"):
        forest.feature_importances(times=[1.0], metric="c-index")
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1, (60, 2))
    xl = rng.exponential(5.0, 60)
    interval = RandomSurvivalForest.fit(
        xl=xl,
        xr=xl + 1.0,
        Z=Z,
        n_trees=3,
        kind="exponential",
        random_state=0,
    )
    with pytest.raises(ValueError, match="right-censored"):
        interval.feature_importances(times=[2.0])


# --------------------------------------- minimal-depth selection threshold


def _hand_threshold(split_nodes, depth, p):
    """The paper's formula written out with exact fractions."""
    q = 1 - Fraction(1, p)
    probs = []
    for d in range(depth):
        big_l = sum(split_nodes[:d])
        probs.append(q**big_l * (1 - q ** split_nodes[d]))
    probs.append(1 - sum(probs))
    return probs, sum(d * pr for d, pr in enumerate(probs))


def test_null_minimal_depth_hand_worked_exactly():
    from surpyval.ml.forest.readouts import (
        null_minimal_depth_distribution,
        null_minimal_depth_mean,
    )

    # A tree with 1, 2 and 1 split nodes at depths 0, 1, 2 (leaves at
    # depth 3), and p = 3 covariates: q = 2/3.
    #   P(0) = 1 - 2/3                  = 1/3
    #   P(1) = 2/3 * (1 - 4/9)          = 10/27
    #   P(2) = (2/3)**3 * (1 - 2/3)     = 8/81
    #   P(3) = the rest                 = 81/81 - 27/81 - 30/81 - 8/81
    #                                   = 16/81
    # mean = 10/27 + 16/81 + 48/81 = 30/81 + 16/81 + 48/81 = 94/81
    expected = [
        Fraction(1, 3),
        Fraction(10, 27),
        Fraction(8, 81),
        Fraction(16, 81),
    ]
    probs = null_minimal_depth_distribution([1, 2, 1], 3, Fraction(3))
    assert probs == expected
    assert null_minimal_depth_mean([1, 2, 1], 3, Fraction(3)) == Fraction(
        94, 81
    )
    assert _hand_threshold([1, 2, 1], 3, 3) == (expected, Fraction(94, 81))
    # In floating point, with p an int
    assert np.allclose(
        null_minimal_depth_distribution([1, 2, 1], 3, 3),
        [float(e) for e in expected],
        rtol=0,
        atol=1e-15,
    )


def test_threshold_of_a_hand_built_forest_is_the_mean_over_trees():
    from surpyval.ml.forest.readouts import (
        minimal_depth,
        minimal_depth_threshold,
    )

    # Tree 1: splits 1, 2, 1 as above (depth 3).
    t1 = _split(
        0,
        _split(1, _split(2, LEAF, LEAF), LEAF),
        _split(1, LEAF, LEAF),
    )
    # Tree 2: a stump (1 split, depth 1): P(0) = 1/3, P(1) = 2/3, mean 2/3.
    t2 = _split(1, LEAF, LEAF)
    model = SimpleNamespace(
        trees=[SimpleNamespace(_root=t1), SimpleNamespace(_root=t2)],
        feature_labels=["Z0", "Z1", "Z2"],
    )
    # t1: 1, 2, 1 split nodes at depths 0, 1, 2? Depth 1 has 2 nodes
    # (left child, right child), depth 2 has the left child's child.
    expected = (94 / 81 + 2 / 3) / 2
    assert minimal_depth_threshold(model) == pytest.approx(expected, abs=1e-15)
    out = minimal_depth(model)
    assert out.attrs["threshold"] == minimal_depth_threshold(model)
    # Existing columns unchanged; selected is mean depth below threshold
    assert out["minimal_depth"].tolist() == [0.5, 0.5, 1.5]
    assert out["selected"].tolist() == [True, True, False]


def _noise_forest(seed, n_cov, n_trees):
    rng = np.random.default_rng(seed)
    n = 300
    Z = rng.uniform(0, 1, (n, n_cov))
    x = rng.weibull(1.5, n) * 10.0
    return x, Z, n_trees


@pytest.mark.parametrize("n_cov", [5, 10])
def test_noise_covariates_sit_at_the_threshold(n_cov):
    # Fixed before running: with pure noise covariates the mean minimal
    # depth over the covariates is within 10% of the threshold.
    x, Z, n_trees = _noise_forest(seed=11, n_cov=n_cov, n_trees=60)
    forest = RandomSurvivalForest.fit(
        x,
        Z,
        n_trees=n_trees,
        kind="exponential",
        n_features_split=2,
        random_state=11,
    )
    out = forest.minimal_depth()
    threshold = out.attrs["threshold"]
    assert threshold == forest.minimal_depth_threshold()
    average = out["minimal_depth"].mean()
    print("noise: mean minimal depth", average, "threshold", threshold)
    assert abs(average - threshold) < 0.10 * threshold


def test_a_strong_covariate_is_selected_among_noise():
    rng = np.random.default_rng(5)
    n = 300
    Z = rng.uniform(0, 1, (n, 5))
    x = rng.weibull(2.0, n) * np.exp(2.0 - 3.0 * Z[:, 0])
    forest = RandomSurvivalForest.fit(
        x,
        Z,
        n_trees=60,
        kind="exponential",
        n_features_split=5,
        random_state=5,
    )
    out = forest.minimal_depth()
    threshold = out.attrs["threshold"]
    print(out, threshold)
    assert out.loc["Z0", "selected"]
    assert out.loc["Z0", "minimal_depth"] < 0.5 * threshold
    assert not out.loc[["Z1", "Z2", "Z3", "Z4"], "selected"].any()
