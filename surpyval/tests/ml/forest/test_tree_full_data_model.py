"""The survival tree under the full SurPyval data model.

The tree's ``kind`` couples a split criterion with its matching leaf
model: ``"weibull"`` (Weibull deviance split + Weibull leaves, the
default), ``"exponential"`` (exponential deviance split + Exponential
leaves) and ``"non-parametric"`` (risk-set log-rank + Nelson-Aalen;
observed/right-censored data only). These tests pin:

- the exponential and Weibull MLEs inside the deviance split against
  SurPyval's own ``Exponential.fit`` / ``Weibull.fit`` on every data
  configuration (the correctness anchors);
- that both parametric kinds build on every data type and, given all
  features, split on the signal feature and predict the right survival
  ordering;
- that the Weibull kind detects a *shape-only* (crossing-hazards)
  signal that the exponential deviance is nearly blind to;
- kind validation: non-parametric raises on data its log-rank split
  cannot express, unknown kinds raise;
- that parametric kinds stay parametric all the way down (leaves);
- the forest passthrough.
"""

import numpy as np
import pytest

from surpyval import Exponential, Weibull
from surpyval.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.ml.forest.deviance_split import (
    _LOG_BETA_BOUNDS,
    _exp_max_ll,
    _exp_theta0,
    _wei_max_ll,
    needs_full_likelihood_split,
)
from surpyval.ml.forest.node import TerminalNode
from surpyval.univariate.nonparametric.nonparametric import NonParametric
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


def _all_data_cases(seed=0, n=100):
    """One SurpyvalData per data configuration, from a common sample."""
    rng = np.random.default_rng(seed)
    x = 10 * rng.weibull(2.2, n)
    c = (rng.random(n) < 0.3).astype(int)
    c_left = np.where(rng.random(n) < 0.2, -1, c)
    x_int = [[v * 0.8, v * 1.3] if i % 3 == 0 else v for i, v in enumerate(x)]
    c_int = np.where(np.arange(n) % 3 == 0, 2, c)
    x_rt = 10 * rng.weibull(2.2, 3 * n)
    x_rt = x_rt[x_rt < 12][:n]
    inf = np.full(n, np.inf)
    ninf = np.full(n, -np.inf)
    return {
        "right": SurpyvalData(x, c, group_and_sort=False),
        "left": SurpyvalData(x, c_left, group_and_sort=False),
        "interval": SurpyvalData(x_int, c_int, group_and_sort=False),
        "left_trunc": SurpyvalData(
            x,
            np.zeros(n),
            tl=np.minimum(x * 0.3, 2.0),
            tr=inf,
            group_and_sort=False,
        ),
        "right_trunc": SurpyvalData(
            x_rt,
            np.zeros(x_rt.size),
            tl=np.full(x_rt.size, -np.inf),
            tr=np.full(x_rt.size, 12.0),
            group_and_sort=False,
        ),
        "mixed": SurpyvalData(
            x_int,
            c_int,
            tl=np.minimum(x * 0.1, 0.5),
            tr=inf,
            group_and_sort=False,
        ),
        "_ninf": ninf,
    }


_CASE_NAMES = [
    "right",
    "left",
    "interval",
    "left_trunc",
    "right_trunc",
    "mixed",
]


# -- MLE anchors ------------------------------------------------------------


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_exponential_mle_matches_exponential_fit(case):
    data = _all_data_cases()[case]
    theta0 = _exp_theta0(data)
    ll = _exp_max_ll(data, (theta0 - 15.0, theta0 + 15.0))
    lam_surpyval = float(Exponential.fit_from_surpyval_data(data).params[0])
    # the attained likelihood is at least surpyval's (same likelihood)
    from surpyval.ml.forest.deviance_split import (
        _exp_neg_ll,
        _exp_neg_ll_parts,
    )

    ll_surpyval = -_exp_neg_ll(
        float(np.log(lam_surpyval)), _exp_neg_ll_parts(data)
    )
    assert ll >= ll_surpyval - 1e-4


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_weibull_mle_matches_weibull_fit(case):
    data = _all_data_cases()[case]
    theta0 = _exp_theta0(data)
    log_alpha0 = -theta0
    box = ((log_alpha0 - 15.0, log_alpha0 + 15.0), _LOG_BETA_BOUNDS)
    _, theta = _wei_max_ll(data, box, np.array([log_alpha0, 0.0]))
    alpha, beta = np.exp(theta)
    a_sp, b_sp = Weibull.fit_from_surpyval_data(data).params
    assert np.isclose(alpha, a_sp, rtol=1e-2)
    assert np.isclose(beta, b_sp, rtol=1e-2)


# -- shape sensitivity: the reason the Weibull kind exists ------------------


def test_weibull_split_sees_crossing_hazards_exponential_does_not():
    # Weibull(10, 0.8) vs Weibull(10, 3): nearly equal means, crossing
    # hazards. The 2-d.f. Weibull deviance gain is enormous; the
    # exponential (rate-only) gain is near its chi2_1 noise floor.
    rng = np.random.default_rng(11)
    n = 150
    g = rng.integers(0, 2, n)
    x = np.where(g == 1, 10 * rng.weibull(3.0, n), 10 * rng.weibull(0.8, n))
    data = SurpyvalData(x, np.zeros(n), group_and_sort=False)
    mask = g == 1

    theta0 = _exp_theta0(data)
    log_alpha0 = -theta0
    box = ((log_alpha0 - 15.0, log_alpha0 + 15.0), _LOG_BETA_BOUNDS)
    parent_ll, parent_theta = _wei_max_ll(
        data, box, np.array([log_alpha0, 0.0])
    )
    gain_wei = 2 * (
        _wei_max_ll(data[mask], box, parent_theta)[0]
        + _wei_max_ll(data[~mask], box, parent_theta)[0]
        - parent_ll
    )

    bounds = (theta0 - 15.0, theta0 + 15.0)
    gain_exp = 2 * (
        _exp_max_ll(data[mask], bounds)
        + _exp_max_ll(data[~mask], bounds)
        - _exp_max_ll(data, bounds)
    )

    assert gain_wei > 20.0  # far beyond the chi2_2 5% critical value
    assert gain_exp < gain_wei / 10.0

    # and the fitted Weibull tree recovers the shape split with distinct
    # leaf shapes
    Z = np.column_stack([g.astype(float), rng.normal(0, 1, n)])
    np.random.seed(13)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=np.zeros(n),
        n_features_split="all",
        min_leaf_samples=25,
        max_depth=1,
        kind="weibull",
    )
    assert tree._root.split_feature_index == 0
    beta_left = tree._root.left_child.model.params[1]
    beta_right = tree._root.right_child.model.params[1]
    assert min(beta_left, beta_right) < 1.2
    assert max(beta_left, beta_right) > 2.0


# -- tree building on every data type, both parametric kinds ----------------


@pytest.mark.parametrize("kind", ["weibull", "exponential"])
def test_tree_right_censored(kind):
    Z, x, c = _signal_data()
    tree = _fit_all_features(x=x, Z=Z, c=c, kind=kind)
    assert tree.kind == kind
    _assert_signal_recovered(tree)


@pytest.mark.parametrize("kind", ["weibull", "exponential"])
def test_tree_interval_censored(kind):
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in, kind=kind)
    _assert_signal_recovered(tree)


def test_tree_left_censored():
    Z, x, c = _signal_data()
    rng = np.random.default_rng(3)
    c_in = np.where((rng.random(len(x)) < 0.2) & (c == 0), -1, c)
    tree = _fit_all_features(x=x, Z=Z, c=c_in)
    _assert_signal_recovered(tree)


def test_tree_left_truncated():
    Z, x, _ = _signal_data()
    n = len(x)
    tree = _fit_all_features(
        x=x,
        Z=Z,
        c=np.zeros(n),
        tl=np.minimum(x * 0.3, 1.0),
        tr=np.full(n, np.inf),
    )
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
    _assert_signal_recovered(tree)


def test_tree_everything_at_once():
    # Interval + left + right censoring with left truncation on every
    # row and right truncation on the event rows. (A right-censored row
    # cannot carry finite right truncation: right-truncation sampling
    # only admits units whose event was seen before tr. The bound sits
    # well above the data -- a common bound just above the maximum
    # leaves the Weibull shape barely identified; see the degenerate
    # smoke test below.)
    Z, x, c = _signal_data()
    n = len(x)
    x_in, c_in = _intervalise(x, c)
    c_in = np.where((np.arange(n) % 7 == 0) & (c_in != 2), -1, c_in)
    tr = np.where(c_in == 1, np.inf, float(x.max()) * 3.0)
    tree = _fit_all_features(
        x=x_in,
        Z=Z,
        c=c_in,
        tl=np.minimum(x * 0.1, 0.5),
        tr=tr,
    )
    _assert_signal_recovered(tree)


def test_tree_degenerate_truncation_smoke():
    # A common right-truncation bound just above the observed maximum
    # leaves the shape parameter near-unidentified (the conditional
    # likelihood is maximised in a degenerate direction). The tree must
    # degrade gracefully: build without crashing and return valid
    # probabilities.
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tr = np.where(c_in == 1, np.inf, float(x.max()) * 1.05)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in, tr=tr)
    s = np.asarray(
        tree.sf(np.array([2.0, 6.0, 12.0]), np.array([1.0, 0.0])),
        dtype=float,
    )
    assert np.all((s >= 0) & (s <= 1))


def test_tree_xl_xr_convenience():
    rng = np.random.default_rng(4)
    n = 80
    Z = np.column_stack(
        [rng.integers(0, 2, n).astype(float), rng.normal(0, 1, n)]
    )
    scale = np.where(Z[:, 0] > 0.5, 4.0, 12.0)
    x = scale * rng.weibull(1.5, n)
    tree = _fit_all_features(xl=x * 0.8, xr=x * 1.2, Z=Z)
    _assert_signal_recovered(tree)


# -- kind selection and guards ----------------------------------------------


def test_non_parametric_kind_on_classic_data():
    Z, x, c = _signal_data()
    tree = _fit_all_features(x=x, Z=Z, c=c, kind="non-parametric")
    assert tree.kind == "non-parametric"
    s_fast = float(tree.sf(5.0, np.array([1.0, 0.0]))[0])
    s_slow = float(tree.sf(5.0, np.array([0.0, 0.0]))[0])
    assert s_fast < s_slow


def test_non_parametric_kind_on_interval_data_raises():
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    with pytest.raises(ValueError, match="non-parametric"):
        SurvivalTree.fit(x=x_in, Z=Z, c=c_in, kind="non-parametric")


def test_invalid_kind_raises():
    Z, x, c = _signal_data()
    with pytest.raises(ValueError, match="kind"):
        SurvivalTree.fit(x=x, Z=Z, c=c, kind="bogus")


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


# -- leaves: parametric all the way down ------------------------------------


def _leaves(node):
    if isinstance(node, TerminalNode):
        return [node]
    return _leaves(node.left_child) + _leaves(node.right_child)


@pytest.mark.parametrize("kind", ["weibull", "exponential"])
def test_parametric_kind_has_no_nonparametric_leaves(kind):
    Z, x, c = _signal_data()
    x_in, c_in = _intervalise(x, c)
    tree = _fit_all_features(x=x_in, Z=Z, c=c_in, kind=kind)
    for leaf in _leaves(tree._root):
        assert not isinstance(leaf.model, NonParametric)


def test_non_parametric_kind_has_nonparametric_leaves():
    Z, x, c = _signal_data()
    tree = _fit_all_features(x=x, Z=Z, c=c, kind="non-parametric")
    for leaf in _leaves(tree._root):
        assert isinstance(leaf.model, NonParametric)


# -- forest passthrough ------------------------------------------------------


def test_forest_on_interval_censored_data():
    Z, x, c = _signal_data(n=90, seed=6)
    x_in, c_in = _intervalise(x, c)
    np.random.seed(11)
    forest = RandomSurvivalForest.fit(
        x=x_in,
        Z=Z,
        c=c_in,
        n_trees=5,
        min_leaf_samples=15,
        n_features_split="all",
        kind="weibull",
    )
    assert forest.kind == "weibull"
    assert all(tree.kind == "weibull" for tree in forest.trees)
    s_fast = float(np.atleast_1d(forest.sf(5.0, np.array([1.0, 0.0])))[0])
    s_slow = float(np.atleast_1d(forest.sf(5.0, np.array([0.0, 0.0])))[0])
    assert s_fast < s_slow
