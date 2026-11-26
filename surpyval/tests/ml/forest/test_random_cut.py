"""Extremely randomised survival trees: ``splitrule="random"``.

- the helper draws one cut a feature, uniformly on the feature's range at
  the node, read as the largest observed value at or below the draw;
- a tree built with it makes exactly that cut (checked against a draw
  from the same seeded stream), at an observed value inside the range;
- the same seed grows the same tree, another seed another, and the
  default (``"searched"``) is the tree there was;
- every kind, ``selection="ctree"``, interval data and competing risks
  take it; a forest keeps its bootstrap (so out-of-bag scores exist) and
  saves and restores the rule;
- an unknown rule is refused, saying what the choices are.
"""

import contextlib
import io

import numpy as np
import pytest

from surpyval.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.ml.forest.random_cut import random_cuts


def _data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 10, (n, 3))
    T = 20 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 5, 0.4, 1.0)
    C = rng.uniform(10, 40, n)
    return np.minimum(T, C), Z, (C < T).astype(int)


def _tree(**kw):
    x, Z, c = _data()
    return SurvivalTree.fit(x, Z, c=c, **kw)


def test_random_cuts_indicator_and_range():
    rng = np.random.default_rng(5)
    Z = np.array([[1.0, 7.0], [2.0, 7.0], [4.0, 7.0], [9.0, 7.0]])
    Zr, cuts = random_cuts(Z, [0, 1], np.random.default_rng(5))
    drawn = float(rng.uniform(1.0, 9.0))
    expected = float(Z[:, 0][Z[:, 0] <= drawn].max())
    assert cuts == {0: expected}  # column 1 is constant: no cut
    np.testing.assert_array_equal(Zr[:, 0], (Z[:, 0] > expected).astype(float))
    np.testing.assert_array_equal(Zr[:, 1], 0.0)
    assert Z[0, 0] == 1.0  # the covariates are not changed


def test_the_root_cut_is_the_seeded_draw():
    x, Z, c = _data()
    tree = SurvivalTree.fit(
        x,
        Z[:, :1],
        c=c,
        kind="non-parametric",
        n_features_split="all",
        max_depth=1,
        splitrule="random",
        random_state=3,
    )
    rng = np.random.default_rng(3)
    # the order of draws in a node: its features, then each one's cut
    rng.choice(1, size=1, replace=False)
    lo, hi = Z[:, 0].min(), Z[:, 0].max()
    drawn = float(rng.uniform(lo, hi))
    expected = float(Z[:, 0][Z[:, 0] <= drawn].max())
    assert tree._root.split_feature_index == 0
    assert tree._root.split_feature_value == expected
    assert lo <= expected < hi
    # and it is an observed value, so "<=" sends the same rows left
    assert expected in Z[:, 0]


def test_a_seed_gives_a_tree_and_another_seed_another():
    a = _tree(random_state=1, splitrule="random").describe()
    b = _tree(random_state=1, splitrule="random").describe()
    c = _tree(random_state=2, splitrule="random").describe()
    assert a == b
    assert a != c


def test_the_default_is_the_searched_tree():
    a = _tree(random_state=4)
    b = _tree(random_state=4, splitrule="searched")
    assert a.describe().replace("searched", "") == b.describe().replace(
        "searched", ""
    )
    assert a.splitrule == "searched"
    assert a.describe() != _tree(random_state=4, splitrule="random").describe()


@pytest.mark.parametrize("kind", ["weibull", "exponential", "non-parametric"])
def test_every_kind_grows_and_predicts(kind):
    tree = _tree(kind=kind, splitrule="random", random_state=0)
    assert tree.splitrule == "random"
    sf = tree.sf([5.0, 10.0], np.array([[1.0, 2.0, 3.0]]))
    assert np.all((sf >= 0) & (sf <= 1))
    assert tree._root.__class__.__name__ == "IntermediateNode"


def test_ctree_selection_and_interval_data_take_it():
    x, Z, c = _data()
    tree = SurvivalTree.fit(
        x, Z, c=c, selection="ctree", splitrule="random", random_state=0
    )
    assert tree.splitrule == "random"
    xl = np.where(c == 0, np.maximum(x - 1.0, 0.1), x)
    xr = np.where(c == 0, x + 1.0, x)
    cc = np.where(c == 0, 2, 1)
    tree = SurvivalTree.fit(
        xl=xl,
        xr=xr,
        c=cc,
        Z=Z,
        kind="non-parametric",
        splitrule="random",
        random_state=0,
    )
    assert tree.describe().count("<=") >= 1


def test_competing_risks_take_it():
    x, Z, c = _data()
    e = np.where(c == 0, np.where(Z[:, 1] > 5, "a", "b"), None)
    tree = SurvivalTree.fit(
        x, Z, e=e, kind="non-parametric", splitrule="random", random_state=0
    )
    cif = tree.cif([5.0], np.array([[1.0, 2.0, 3.0]]), event="a")
    assert np.all(np.isfinite(cif))


def test_forest_keeps_its_bootstrap_and_saves_the_rule():
    x, Z, c = _data(120)
    with contextlib.redirect_stderr(io.StringIO()):
        forest = RandomSurvivalForest.fit(
            x,
            Z,
            c=c,
            n_trees=6,
            kind="exponential",
            splitrule="random",
            random_state=0,
        )
    assert forest.bootstrap_indices is not None
    assert np.isfinite(forest.oob_log_likelihood())
    assert "splitrule='random'" in repr(forest)
    restored = RandomSurvivalForest.from_dict(forest.to_dict())
    assert restored.splitrule == "random"
    assert all(t.splitrule == "random" for t in forest.trees)
    # a forest saved before the rule existed reads as searched
    old = forest.to_dict()
    del old["splitrule"]
    assert RandomSurvivalForest.from_dict(old).splitrule == "searched"


def test_unknown_splitrule_is_refused_with_the_choices():
    x, Z, c = _data(60)
    with pytest.raises(ValueError, match="splitrule.*searched.*random"):
        SurvivalTree.fit(x, Z, c=c, splitrule="best")
    with pytest.raises(ValueError, match="splitrule"):
        RandomSurvivalForest.fit(x, Z, c=c, n_trees=2, splitrule="best")
