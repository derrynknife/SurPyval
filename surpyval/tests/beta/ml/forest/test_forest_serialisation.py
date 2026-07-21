"""Serialisation round-trips for the survival tree and forest (issue #191).

The tree/forest were deliberately excluded from the original serialisation
campaign while the forest was crash-prone. These tests lock in that a fitted
tree or forest survives ``to_dict``/``from_dict`` (and the JSON/BSON forms)
with byte-for-byte identical predictions, dispatches through the package-level
``surpyval.from_dict``, and stays BSON-native.
"""

import json
import os
import tempfile

import numpy as np
import pytest

import surpyval
from surpyval import Weibull
from surpyval.beta.ml.forest.forest import RandomSurvivalForest
from surpyval.beta.ml.forest.node import TerminalNode, node_from_dict
from surpyval.beta.ml.forest.tree import SurvivalTree
from surpyval.serialisation import SCHEMA_VERSION
from surpyval.univariate.parametric import NeverOccurs
from surpyval.utils.surpyval_data import SurpyvalData


def _data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    lin = Z @ np.array([0.9, -0.6, 0.3])
    x = rng.exponential(np.exp(-lin)) * 15 + 1
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return x, Z, c


def _assert_bson_native(obj, path="$"):
    """Every key a str, every value a BSON-encodable native type."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-string key {k!r} at {path}"
            _assert_bson_native(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            _assert_bson_native(v, f"{path}[{j}]")
    else:
        assert obj is None or isinstance(
            obj, (str, bool, int, float)
        ), f"non-native {type(obj).__name__} at {path}"
        assert not isinstance(obj, np.generic), f"numpy scalar at {path}"


KINDS = ["weibull", "exponential", "non-parametric"]


@pytest.mark.parametrize("kind", KINDS)
def test_tree_round_trip_predictions(kind):
    x, Z, c = _data()
    np.random.seed(1)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, kind=kind, max_depth=4)

    restored = SurvivalTree.from_dict(tree.to_dict())

    xq = np.linspace(1, 30, 12)
    for zq in ([0.5, -0.5, 0.2], [-1.0, 1.0, 0.0]):
        for fn in ("sf", "ff", "Hf"):
            a = getattr(tree, fn)(xq, zq)
            b = getattr(restored, fn)(xq, zq)
            assert np.allclose(a, b, equal_nan=True)


@pytest.mark.parametrize("kind", KINDS)
def test_tree_dict_is_bson_native(kind):
    x, Z, c = _data(seed=3)
    np.random.seed(2)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, kind=kind, max_depth=4)
    d = tree.to_dict()
    _assert_bson_native(d)
    assert d["model"] == "SurvivalTree"
    assert d["schema"] == SCHEMA_VERSION


def test_tree_json_file_round_trip():
    x, Z, c = _data(seed=4)
    np.random.seed(3)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, kind="weibull", max_depth=4)
    fp = tempfile.mktemp(suffix=".json")
    try:
        tree.to_json(fp)
        restored = SurvivalTree.from_json(fp)
    finally:
        if os.path.exists(fp):
            os.remove(fp)
    xq = np.linspace(1, 30, 10)
    assert np.allclose(
        tree.sf(xq, [0.3, 0.3, 0.3]),
        restored.sf(xq, [0.3, 0.3, 0.3]),
        equal_nan=True,
    )


def test_forest_round_trip_predictions():
    x, Z, c = _data(seed=5)
    np.random.seed(4)
    forest = RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=8, kind="weibull", max_depth=4
    )
    d = forest.to_dict()
    _assert_bson_native(d)
    restored = RandomSurvivalForest.from_dict(json.loads(json.dumps(d)))

    xq = np.linspace(1, 30, 12)
    for zq in ([0.5, -0.5, 0.2], [-1.0, 1.0, 0.0]):
        assert np.allclose(forest.sf(xq, zq), restored.sf(xq, zq))
        assert np.allclose(forest.Hf(xq, zq), restored.Hf(xq, zq))
    assert restored.n_trees == forest.n_trees
    assert len(restored.trees) == len(forest.trees)


def test_package_from_dict_dispatch():
    x, Z, c = _data(seed=6)
    np.random.seed(5)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, kind="weibull", max_depth=3)
    forest = RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=4, kind="exponential", max_depth=3
    )

    assert isinstance(surpyval.from_dict(tree.to_dict()), SurvivalTree)
    assert isinstance(
        surpyval.from_dict(forest.to_dict()), RandomSurvivalForest
    )


def test_never_occurs_leaf_round_trip():
    # An all-censored leaf carries no event information, so its model is the
    # parameterless NeverOccurs; it must serialise as a sentinel and restore
    # to the same degenerate leaf.
    data = SurpyvalData(
        x=[5.0, 6.0, 7.0], c=[1, 1, 1], n=[1, 1, 1], group_and_sort=False
    )
    leaf = TerminalNode(data, kind="weibull")
    assert leaf.model is NeverOccurs

    d = leaf.to_dict()
    assert d["leaf"] == "NeverOccurs"
    _assert_bson_native(d)

    restored = node_from_dict(d)
    assert restored.model is NeverOccurs
    xq = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        leaf.apply_model_function("sf", xq, None),
        restored.apply_model_function("sf", xq, None),
    )


def test_from_dict_rejects_tag_mismatch():
    with pytest.raises(ValueError, match="SurvivalTree"):
        SurvivalTree.from_dict({"model": "RandomSurvivalForest"})
    with pytest.raises(ValueError, match="RandomSurvivalForest"):
        RandomSurvivalForest.from_dict({"model": "SurvivalTree"})
    with pytest.raises(ValueError, match="node"):
        node_from_dict({"node": "not-a-node"})


def test_parametric_offset_to_dict_is_bson_native():
    # Regression guard for the leak surfaced by the forest work: an offset
    # Weibull's ``gamma`` and every model's ``_neg_ll`` were stored as numpy
    # scalars, which MongoDB's BSON encoder rejects.
    for model in (
        Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0, 8.0], offset=True),
    ):
        _assert_bson_native(model.to_dict())
