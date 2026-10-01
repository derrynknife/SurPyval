"""``min_split_gain``: a likelihood stop for the deviance splits (#189).

A ``"weibull"`` or ``"exponential"`` node splits only if its best cut
raises the maximised log-likelihood by more than ``min_split_gain``:
a number, ``"aic"`` (the kind's degrees of freedom ``k``) or ``"bic"``
(``k log(d) / 2``, ``d`` the node's n-weighted failures). The default, 0,
keeps the old behaviour (any gain beyond the optimiser's noise).
"""

import json

import numpy as np
import pytest

import surpyval as sp
from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree
from surpyval.beta.ml.forest.deviance_split import (
    deviance_split,
    split_gain_threshold,
)
from surpyval.utils.surpyval_data import SurpyvalData


def _data(seed, effect, n=120):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 1))
    x = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, effect, 1.0)
    c = (x > 15).astype(int)
    return np.minimum(x, 15), c, Z


def _exp_ll(x, c):
    # The exponential maximum log-likelihood, in closed form
    r = np.sum(c == 0)
    return r * np.log(r / x.sum()) - r


def _leaves(node):
    if hasattr(node, "left_child"):
        return _leaves(node.left_child) + _leaves(node.right_child)
    return 1


@pytest.mark.parametrize("seed", range(4))
def test_exponential_gain_threshold_brackets_the_gain(seed):
    x, c, Z = _data(seed, 0.7)
    data = SurpyvalData(x, c, group_and_sort=False)
    u, v = deviance_split(data, Z, 5, 2, [0], model="exponential")
    assert u == 0
    left = Z[:, 0] <= v
    gain = (
        _exp_ll(x[left], c[left]) + _exp_ll(x[~left], c[~left]) - _exp_ll(x, c)
    )
    split = deviance_split(
        data, Z, 5, 2, [0], model="exponential", min_split_gain=gain - 1e-3
    )
    assert split == (u, v)
    stop = deviance_split(
        data, Z, 5, 2, [0], model="exponential", min_split_gain=gain + 1e-3
    )
    assert stop == (-1, -np.inf)


@pytest.mark.parametrize("model", ["exponential", "weibull"])
def test_weibull_gain_threshold_brackets_the_gain(model):
    x, c, Z = _data(1, 0.6)
    data = SurpyvalData(x, c, group_and_sort=False)
    u, v = deviance_split(data, Z, 5, 2, [0], model=model)
    left = Z[:, 0] <= v
    dist = sp.Weibull if model == "weibull" else sp.Exponential
    gain = (
        dist.fit(x, c).neg_ll()
        - dist.fit(x[left], c[left]).neg_ll()
        - dist.fit(x[~left], c[~left]).neg_ll()
    )
    for delta, expected in [(-1e-3, (u, v)), (1e-3, (-1, -np.inf))]:
        assert (
            deviance_split(
                data,
                Z,
                5,
                2,
                [0],
                model=model,
                min_split_gain=gain + delta,
            )
            == expected
        )


def test_aic_and_bic_are_the_kinds_penalties():
    x, c, Z = _data(2, 0.8)
    data = SurpyvalData(x, c, group_and_sort=False)
    d = float(np.sum(c == 0))
    assert split_gain_threshold("aic", data, "exponential") == 1.0
    assert split_gain_threshold("aic", data, "weibull") == 2.0
    assert split_gain_threshold("bic", data, "weibull") == pytest.approx(
        np.log(d)
    )
    assert split_gain_threshold("bic", data, "exponential") == pytest.approx(
        np.log(d) / 2
    )
    # The default keeps the old numerical floor
    assert split_gain_threshold(0.0, data, "weibull") == 1e-6
    # n-weighted failures: a row of count 2 is two rows
    counted = SurpyvalData(x, c, np.full(x.size, 2), group_and_sort=False)
    assert split_gain_threshold("bic", counted, "weibull") == pytest.approx(
        np.log(2 * d)
    )


def test_aic_tree_stops_on_noise_and_keeps_the_effect():
    noise, effect = [], []
    for seed in range(6):
        x, c, Z = _data(seed, 1.0)
        Z = np.column_stack([Z, np.random.default_rng(seed).uniform(size=120)])
        options = dict(kind="exponential", n_features_split="all")
        greedy = SurvivalTree.fit(x, Z, c=c, **options)
        aic = SurvivalTree.fit(x, Z, c=c, min_split_gain="aic", **options)
        noise.append((_leaves(greedy._root), _leaves(aic._root)))
        x, c, Z1 = _data(seed, 0.4)
        aic = SurvivalTree.fit(
            x, Z1, c=c, min_split_gain="aic", max_depth=1, **options
        )
        effect.append(getattr(aic._root, "split_feature_index", None))
    # Without "aic" every noise tree grows; with it they stay small
    assert all(g > 5 for g, _ in noise), noise
    assert sum(a for _, a in noise) < sum(g for g, _ in noise) / 3, noise
    assert effect == [0] * 6


def test_default_is_unchanged():
    x, c, Z = _data(3, 0.7)
    a = SurvivalTree.fit(x, Z, c=c, kind="exponential", random_state=0)
    b = SurvivalTree.fit(
        x, Z, c=c, kind="exponential", random_state=0, min_split_gain=0
    )
    assert repr(a) == repr(b)


@pytest.mark.parametrize("value", ["aic", "bic", 0.5])
def test_non_parametric_refuses_a_gain(value):
    x, c, Z = _data(0, 0.7)
    with pytest.raises(ValueError, match="selection='ctree'"):
        SurvivalTree.fit(
            x, Z, c=c, kind="non-parametric", min_split_gain=value
        )
    with pytest.raises(ValueError, match="selection='ctree'"):
        RandomSurvivalForest.fit(
            x, Z, c=c, kind="non-parametric", min_split_gain=value, n_trees=1
        )
    # The default is accepted
    SurvivalTree.fit(x, Z, c=c, kind="non-parametric", max_depth=1)


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan, "AICc", None, True])
def test_invalid_gain_raises(value):
    x, c, Z = _data(0, 0.7)
    with pytest.raises(ValueError, match="min_split_gain"):
        SurvivalTree.fit(x, Z, c=c, kind="exponential", min_split_gain=value)


def test_gain_is_serialised():
    x, c, Z = _data(0, 0.7)
    tree = SurvivalTree.fit(
        x, Z, c=c, kind="exponential", min_split_gain="bic", max_depth=2
    )
    restored = SurvivalTree.from_dict(json.loads(json.dumps(tree.to_dict())))
    assert restored.min_split_gain == "bic"
    forest = RandomSurvivalForest.fit(
        x,
        Z,
        c=c,
        kind="exponential",
        min_split_gain=1.5,
        n_trees=2,
        max_depth=1,
        random_state=0,
    )
    assert forest.min_split_gain == 1.5
    assert all(t.min_split_gain == 1.5 for t in forest.trees)
    restored_forest = RandomSurvivalForest.from_dict(forest.to_dict())
    assert restored_forest.min_split_gain == 1.5
