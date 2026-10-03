"""``random_state`` for the survival tree and forest (issue #471).

The forest draws its bootstrap samples, and every tree the features it
considers at each split, at random. Both take ``random_state`` under the
package's seed rule (Design Principle 19):

- ``None`` draws from numpy's global stream, exactly as before the
  argument existed, so ``np.random.seed`` reproduces a fit bit for bit;
- an int or a ``Generator`` gives a stream of its own, reproducible, and
  neither depending on nor advancing the global one.
"""

import numpy as np
import pytest

from surpyval.beta.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.beta.ml.forest.node import IntermediateNode, TerminalNode

X_QUERY = np.array([1.0, 4.0, 9.0])


def _data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 4))
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, 0.5, 1.0)
    C = rng.uniform(3, 25, n)
    return dict(x=np.minimum(T, C), c=(C < T).astype(int), Z=Z)


def _forest(**kwargs):
    kwargs = {"n_trees": 4, "max_depth": 2, "kind": "exponential", **kwargs}
    return RandomSurvivalForest.fit(**_data(), **kwargs)


def _tree(**kwargs):
    kwargs = {"max_depth": 3, "n_features_split": 1, **kwargs}
    return SurvivalTree.fit(**_data(), kind="exponential", **kwargs)


def _splits(node):
    # The (feature, value) of every split, depth first.
    if isinstance(node, TerminalNode):
        return []
    assert isinstance(node, IntermediateNode)
    return (
        [(node.split_feature_index, node.split_feature_value)]
        + _splits(node.left_child)
        + _splits(node.right_child)
    )


def _forest_fingerprint(forest):
    return (
        [idx.tolist() for idx in forest.bootstrap_indices],
        [_splits(tree._root) for tree in forest.trees],
        forest.sf(X_QUERY, _data()["Z"][:5]),
    )


def _assert_same(a, b):
    assert a[0] == b[0]
    assert a[1] == b[1]
    np.testing.assert_array_equal(a[2], b[2])


def test_seeded_forest_is_reproducible():
    a = _forest_fingerprint(_forest(random_state=3))
    b = _forest_fingerprint(_forest(random_state=3))
    _assert_same(a, b)
    c = _forest_fingerprint(_forest(random_state=4))
    assert a[0] != c[0], "the seed is ignored"


def test_seeded_forest_accepts_a_generator():
    a = _forest_fingerprint(_forest(random_state=3))
    b = _forest_fingerprint(_forest(random_state=np.random.default_rng(3)))
    _assert_same(a, b)


def test_seeded_forest_leaves_the_global_stream_alone():
    np.random.seed(1)
    a = _forest_fingerprint(_forest(random_state=3))
    after = np.random.uniform(size=4)
    np.random.seed(1)
    untouched = np.random.uniform(size=4)
    np.testing.assert_array_equal(after, untouched)
    # ... nor depends on it
    np.random.seed(2)
    _assert_same(a, _forest_fingerprint(_forest(random_state=3)))


def test_unseeded_forest_follows_np_random_seed():
    np.random.seed(5)
    a = _forest_fingerprint(_forest())
    np.random.seed(5)
    b = _forest_fingerprint(_forest())
    _assert_same(a, b)
    np.random.seed(6)
    c = _forest_fingerprint(_forest())
    assert a[0] != c[0], "np.random.seed is ignored"


def test_unseeded_forest_draws_as_it_always_did():
    # random_state=None is numpy's global stream itself, drawn in the
    # same order as before the argument existed: the bootstraps first,
    # with np.random.choice, then the trees. So an existing
    # np.random.seed script gets the same forest bit for bit.
    n_rows = len(_data()["x"])
    np.random.seed(7)
    expected = [np.random.choice(n_rows, n_rows, replace=True) for _ in "abcd"]
    np.random.seed(7)
    forest = _forest()
    for idx, want in zip(forest.bootstrap_indices, expected):
        np.testing.assert_array_equal(idx, want)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_forest_trees_have_their_own_streams(bootstrap):
    # Without the bootstrap every tree sees the same rows, so trees that
    # differ can only differ by their feature draws: each tree gets a
    # child stream of the forest's seed, not a copy of one stream.
    forest = _forest(
        random_state=0, bootstrap=bootstrap, n_features_split=1, n_trees=6
    )
    splits = [tuple(_splits(tree._root)) for tree in forest.trees]
    assert len(set(splits)) > 1


def test_seeded_tree_is_reproducible_and_leaves_the_global_stream_alone():
    np.random.seed(1)
    a = _splits(_tree(random_state=3)._root)
    after = np.random.uniform(size=4)
    np.random.seed(1)
    np.testing.assert_array_equal(after, np.random.uniform(size=4))
    np.random.seed(2)
    assert _splits(_tree(random_state=3)._root) == a
    assert _splits(_tree(random_state=np.random.default_rng(3))._root) == a
    # The features drawn at each split depend on the seed.
    draws = {tuple(_splits(_tree(random_state=s)._root)) for s in range(8)}
    assert len(draws) > 1


def test_unseeded_tree_follows_np_random_seed():
    np.random.seed(5)
    a = _splits(_tree()._root)
    np.random.seed(5)
    assert _splits(_tree()._root) == a
    trees = set()
    for s in range(8):
        np.random.seed(s)
        trees.add(tuple(_splits(_tree()._root)))
    assert len(trees) > 1


# -- #546: quiet by default; n_jobs ------------------------------------------


def test_546_forest_fit_prints_nothing(capfd):
    # joblib's progress log ("[Parallel(n_jobs=1)]: Done ...") was printed
    # on every fit (verbose=1 was hard-coded).
    _forest(random_state=0)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_546_n_jobs_does_not_change_a_seeded_forest(capfd):
    a = _forest_fingerprint(_forest(random_state=3))
    b = _forest_fingerprint(_forest(random_state=3, n_jobs=2))
    _assert_same(a, b)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_546_unseeded_parallel_forest_follows_np_random_seed():
    np.random.seed(5)
    a = _forest_fingerprint(_forest(n_jobs=2))
    np.random.seed(5)
    b = _forest_fingerprint(_forest(n_jobs=2))
    _assert_same(a, b)
    # The bootstraps are drawn first, as with n_jobs=1
    np.random.seed(5)
    assert a[0] == _forest_fingerprint(_forest())[0]


@pytest.mark.parametrize("n_jobs", [0, 1.5, True, "2"])
def test_546_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        _forest(random_state=0, n_jobs=n_jobs)
