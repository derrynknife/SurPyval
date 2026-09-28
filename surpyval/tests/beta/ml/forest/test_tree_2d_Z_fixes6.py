"""
Prediction with a matrix of covariate vectors (issue #369).

``SurvivalTree`` routed a 2-D ``Z`` by ``Z[split_index]`` -- a *row* -- so
with one covariate every subject got the first subject's prediction and
with several the routing raised. A 2-D ``Z`` must give the same result as
evaluating each row on its own and stacking, for the tree, the forest and
every prediction method, and ``survival_probability`` must work for both.
"""

import contextlib
import io

import numpy as np
import pytest

from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree
from surpyval.metrics import survival_probability

FUNCTIONS = ["sf", "ff", "df", "hf", "Hf"]
XS = np.array([2.0, 5.0, 8.0])


def _data(n_features):
    # Life halves when z0 > 0.5, and again when z1 > 0.5 (if present)
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1, (200, n_features))
    scale = np.where(Z[:, 0] > 0.5, 5.0, 10.0)
    if n_features > 1:
        scale = scale * np.where(Z[:, 1] > 0.5, 0.5, 1.0)
    x = rng.weibull(2.0, 200) * scale
    c = (x > 12).astype(int)
    return np.minimum(x, 12), c, Z


def _query(n_features):
    # Rows on both sides of every split, repeated so equal rows must get
    # equal predictions wherever they sit in the matrix
    rng = np.random.default_rng(1)
    Zq = rng.uniform(0, 1, (6, n_features))
    Zq[:3, 0], Zq[3:, 0] = 0.2, 0.8
    return np.vstack([Zq, Zq[::-1]])


def _stack(model, fn, x, Zq):
    return np.vstack([getattr(model, fn)(x, Zq[i]) for i in range(len(Zq))])


@pytest.fixture(scope="module", params=[1, 3], ids=["p1", "p3"])
def n_features(request):
    return request.param


@pytest.fixture(
    scope="module", params=["weibull", "exponential", "non-parametric"]
)
def tree(request, n_features):
    x, c, Z = _data(n_features)
    np.random.seed(0)
    return SurvivalTree.fit(
        x, Z, c=c, max_depth=2, n_features_split="all", kind=request.param
    )


@pytest.fixture(scope="module")
def forest(n_features):
    x, c, Z = _data(n_features)
    np.random.seed(0)
    with contextlib.redirect_stderr(io.StringIO()):
        return RandomSurvivalForest.fit(
            x, Z, c=c, n_trees=5, max_depth=2, kind="exponential"
        )


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_tree_matrix_equals_stacked_rows(tree, n_features, fn):
    Zq = _query(n_features)
    expected = _stack(tree, fn, XS, Zq)
    # The query rows really reach different leaves (the old code gave
    # every row the first row's leaf)
    assert not np.allclose(expected[0], expected[3])

    got = getattr(tree, fn)(XS, Zq)
    assert got.shape == (len(Zq), XS.size)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_tree_matrix_x_conventions(tree, n_features):
    Zq = _query(n_features)
    # Scalar x: one value per row, (n_rows,) + x.shape (principle 7)
    got = tree.sf(5.0, Zq)
    assert got.shape == (len(Zq),)
    np.testing.assert_array_equal(got, _stack(tree, "sf", 5.0, Zq)[:, 0])
    # One subject (1-D Z) is unchanged: values shaped like x
    assert tree.sf(XS, Zq[0]).shape == XS.shape
    assert tree.sf(5.0, Zq[0]).shape == ()
    # A matrix with a single row is a one-row grid
    np.testing.assert_array_equal(
        tree.sf(XS, Zq[:1]), tree.sf(XS, Zq[0])[None, :]
    )
    # No rows: an empty grid
    assert tree.sf(XS, Zq[:0]).shape == (0, XS.size)


def test_tree_one_feature_scalar_Z(tree, n_features):
    if n_features != 1:
        pytest.skip("scalar Z only names a subject for a one-feature tree")
    np.testing.assert_array_equal(tree.sf(XS, 0.8), tree.sf(XS, [0.8]))
    np.testing.assert_array_equal(
        tree.sf(XS, [[0.2], [0.8]]),
        np.vstack([tree.sf(XS, 0.2), tree.sf(XS, 0.8)]),
    )


def test_tree_rejects_3d_Z(tree, n_features):
    with pytest.raises(ValueError, match="dimensions"):
        tree.sf(XS, np.zeros((2, 2, n_features)))


def test_restored_tree_routes_matrix(tree, n_features):
    Zq = _query(n_features)
    restored = SurvivalTree.from_dict(tree.to_dict())
    np.testing.assert_allclose(restored.sf(XS, Zq), _stack(tree, "sf", XS, Zq))


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_forest_matrix_equals_stacked_rows(forest, n_features, fn):
    Zq = _query(n_features)
    got = getattr(forest, fn)(XS, Zq)
    assert got.shape == (len(Zq), XS.size)
    np.testing.assert_allclose(got, _stack(forest, fn, XS, Zq), rtol=1e-14)


def test_forest_other_methods_row_by_row(forest, n_features):
    Zq = _query(n_features)
    np.testing.assert_allclose(
        forest.sf(XS, Zq, ensemble_method="Hf"),
        np.vstack([forest.sf(XS, z, ensemble_method="Hf") for z in Zq]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        forest.mortality(XS, Zq),
        np.concatenate([forest.mortality(XS, z) for z in Zq]),
        rtol=1e-14,
    )
    assert forest.sf(5.0, Zq).shape == (len(Zq),)
    assert forest.sf(XS, Zq[0]).shape == XS.shape


def test_survival_probability_tree(tree, n_features):
    # survival_probability passes x = np.full(n, t) with the whole matrix
    # and reads the (n, n) grid's first column
    Zq = _query(n_features)
    S = survival_probability(tree, Zq, XS)
    np.testing.assert_allclose(S, _stack(tree, "sf", XS, Zq), rtol=0, atol=0)


def test_survival_probability_forest(forest, n_features):
    Zq = _query(n_features)
    S = survival_probability(forest, Zq, XS)
    np.testing.assert_allclose(S, _stack(forest, "sf", XS, Zq), rtol=1e-14)
