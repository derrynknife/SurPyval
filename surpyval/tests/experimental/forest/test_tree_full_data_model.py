"""The survival tree under the full SurPyval data model.

The risk-set log-rank split only exists for observed / right-censored
(optionally left-truncated) data, so the tree adds a full-likelihood
exponential deviance split (Davis & Anderson, 1989) and switches to it
automatically when the data contains left censoring, interval censoring
or right truncation. These tests pin:

- the exponential MLE inside the deviance split against SurPyval's own
  ``Exponential.fit`` on every data type (the correctness anchor);
- that a tree builds on every data type and, given all features, splits
  on the signal feature and predicts the right survival ordering;
- the auto rule selection and the guard for forcing log-rank on data it
  cannot handle;
- Turnbull nonparametric leaves for data Nelson-Aalen cannot fit;
- the forest passthrough.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from surpyval import Exponential
from surpyval.experimental.forest import RandomSurvivalForest, SurvivalTree
from surpyval.experimental.forest.deviance_split import (
    _exp_neg_ll,
    _exp_neg_ll_parts,
    needs_full_likelihood_split,
)
from surpyval.utils.surpyval_data import SurpyvalData


def _signal_data(n=120, seed=2):
    """Feature 0 is a strong binary signal (3x scale), feature 1 noise."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.integers(0, 2, n).astype(float), rng.normal(0, 1, n)]
    )
    scale = np.where(Z[:, 0] > 0.5, 4.0, 12.0)
    x = scale * rng.weibull(1.5, n)
    c = (rng.random(n) < 0.25).astype(int)
    return Z, x, c


def _intervalise(x, c, every=3):
    """Turn every ``every``-th observed value into an interval."""
    x_out = [
        [v * 0.75, v * 1.25] if (i % every == 0 and c[i] == 0) else v
        for i, v in enumerate(x)
    ]
    c_out = np.where((np.arange(len(x)) % every == 0) & (c == 0), 2, c)
    return x_out, c_out


def _fit_all_features(**kwargs):
    np.random.seed(7)
    return SurvivalTree.fit(
        n_features_split="all", min_leaf_samples=10, **kwargs
    )


def _assert_signal_recovered(tree, mid_time=5.0):
    assert tree._root.split_feature_index == 0
    s_fast = float(tree.sf(mid_time, np.array([1.0, 0.0]))[0])
    s_slow = float(tree.sf(mid_time, np.array([0.0, 0.0]))[0])
    assert s_fast < s_slow


# -- exponential MLE anchor ------------------------------------------------


def _deviance_mle(data):
    parts = _exp_neg_ll_parts(data)
    res = minimize_scalar(
        _exp_neg_ll,
        args=(parts,),
        bounds=(-20, 5),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(np.exp(res.x))


@pytest.mark.parametrize(
    "case",
    ["right", "left", "interval", "left_trunc", "right_trunc", "mixed"],
)
def test_deviance_mle_matches_exponential_fit(case):
    rng = np.random.default_rng(0)
    n = 100
    x = rng.exponential(10, n)
    c = (rng.random(n) < 0.3).astype(int)
    tl = np.full(n, -np.inf)
    tr = np.full(n, np.inf)
    x_in: list | np.ndarray = x
    if case == "left":
        c = np.where(rng.random(n) < 0.2, -1, c)
    elif case == "interval":
        x_in, c = _intervalise(x, c)
    elif case == "left_trunc":
        c = np.zeros(n)
        tl = np.minimum(x * 0.3, 2.0)
    elif case == "right_trunc":
        c = np.zeros(n)
        tr = x * 1.5 + 5.0
    elif case == "mixed":
        x_in, c = _intervalise(x, c)
        c = np.where((np.arange(n) % 7 == 0) & (c != 2), -1, c)
        tl = np.minimum(x * 0.1, 0.5)

    data = SurpyvalData(x_in, c, tl=tl, tr=tr, group_and_sort=False)
    lam_surpyval = float(Exponential.fit_from_surpyval_data(data).params[0])
    assert np.isclose(_deviance_mle(data), lam_surpyval, rtol=1e-3)


# -- tree building on every data type --------------------------------------


def test_tree_right_censored_uses_log_rank():
    Z, x, c = _signal_data()
    tree = _fit_all_features(x=x, Z=Z, c=c)
    assert tree.split_rule == "log_rank"
    _assert_signal_recovered(tree)


def test_tree_interval_censored():
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in)
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


def test_tree_left_censored():
    Z, x, c = _signal_data()
    rng = np.random.default_rng(3)
    c_in = np.where((rng.random(len(x)) < 0.2) & (c == 0), -1, c)
    tree = _fit_all_features(x=x, Z=Z, c=c_in)
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


def test_tree_left_truncated_keeps_log_rank():
    Z, x, _ = _signal_data()
    n = len(x)
    tl = np.minimum(x * 0.3, 1.0)
    tree = _fit_all_features(
        x=x, Z=Z, c=np.zeros(n), tl=tl, tr=np.full(n, np.inf)
    )
    assert tree.split_rule == "log_rank"
    _assert_signal_recovered(tree)


def test_tree_right_truncated():
    Z, x, _ = _signal_data()
    n = len(x)
    tree = _fit_all_features(
        x=x,
        Z=Z,
        c=np.zeros(n),
        tl=np.full(n, -np.inf),
        tr=np.full(n, float(x.max()) * 1.1),
    )
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


def test_tree_everything_at_once():
    # Interval + left + right censoring with left truncation on every
    # row and right truncation on the event rows. (A right-censored row
    # cannot carry finite right truncation: right-truncation sampling
    # only admits units whose event was seen before tr.)
    Z, x, c = _signal_data()
    n = len(x)
    x_in, c_in = _intervalise(x, c)
    c_in = np.where((np.arange(n) % 7 == 0) & (c_in != 2), -1, c_in)
    tr = np.where(c_in == 1, np.inf, float(x.max()) * 1.2)
    tree = _fit_all_features(
        x=x_in,
        Z=Z,
        c=c_in,
        tl=np.minimum(x * 0.1, 0.5),
        tr=tr,
    )
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


def test_tree_xl_xr_convenience():
    # pure interval-censored input via xl/xr
    rng = np.random.default_rng(4)
    n = 80
    Z = np.column_stack(
        [rng.integers(0, 2, n).astype(float), rng.normal(0, 1, n)]
    )
    scale = np.where(Z[:, 0] > 0.5, 4.0, 12.0)
    x = scale * rng.weibull(1.5, n)
    tree = _fit_all_features(xl=x * 0.8, xr=x * 1.2, Z=Z)
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


# -- split-rule selection and guards ---------------------------------------


def test_forced_deviance_on_classic_data():
    Z, x, c = _signal_data()
    tree = _fit_all_features(x=x, Z=Z, c=c, split_rule="deviance")
    assert tree.split_rule == "deviance"
    _assert_signal_recovered(tree)


def test_forced_log_rank_on_interval_data_raises():
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    with pytest.raises(ValueError, match="undefined"):
        SurvivalTree.fit(x=x_in, Z=Z, c=c_in, split_rule="log-rank")


def test_invalid_split_rule_raises():
    Z, x, c = _signal_data()
    with pytest.raises(ValueError, match="split_rule"):
        SurvivalTree.fit(x=x, Z=Z, c=c, split_rule="bogus")


def test_needs_full_likelihood_split_detection():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert not needs_full_likelihood_split(
        SurpyvalData(x, np.array([0, 0, 1, 0, 1]), group_and_sort=False)
    )
    assert needs_full_likelihood_split(
        SurpyvalData(x, np.array([0, 0, -1, 0, 1]), group_and_sort=False)
    )
    assert needs_full_likelihood_split(
        SurpyvalData(
            x,
            np.zeros(5),
            tl=np.full(5, -np.inf),
            tr=np.full(5, 10.0),
            group_and_sort=False,
        )
    )


# -- leaves ----------------------------------------------------------------


def test_nonparametric_turnbull_leaves_on_interval_data():
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in, leaf_type="non-parametric")
    # predictions are finite probabilities and decreasing in time
    s = tree.sf(np.array([2.0, 6.0, 12.0]), np.array([1.0, 0.0]))
    s = np.asarray(s, dtype=float)
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(np.diff(s) <= 1e-9)


def test_parametric_leaves_on_interval_data():
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in, leaf_type="parametric")
    s = np.asarray(
        tree.sf(np.array([2.0, 6.0, 12.0]), np.array([0.0, 0.0])),
        dtype=float,
    )
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(np.diff(s) <= 1e-9)


# -- forest passthrough ----------------------------------------------------


def test_forest_on_interval_censored_data():
    Z, x, c = _signal_data(n=90, seed=6)
    x_in, c_in = _intervalise(x, c)
    np.random.seed(11)
    forest = RandomSurvivalForest.fit(
        x=x_in,
        Z=Z,
        c=c_in,
        n_trees=5,
        min_leaf_samples=10,
        n_features_split="all",
    )
    assert all(tree.split_rule == "deviance" for tree in forest.trees)
    s_fast = float(np.atleast_1d(forest.sf(5.0, np.array([1.0, 0.0])))[0])
    s_slow = float(np.atleast_1d(forest.sf(5.0, np.array([0.0, 0.0])))[0])
    assert s_fast < s_slow
