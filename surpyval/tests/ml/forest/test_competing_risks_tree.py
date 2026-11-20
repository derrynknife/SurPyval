"""A survival tree grown with the cause of each failure (``e``)."""

import json

import numpy as np
import pandas as pd
import pytest

from surpyval import Weibull
from surpyval.ml import SurvivalTree
from surpyval.ml.forest.cause_specific_split import cause_specific_split
from surpyval.ml.forest.log_rank_split import log_rank_split
from surpyval.ml.forest.node import (
    CompetingRisksLeaf,
    IntermediateNode,
    TerminalNode,
)
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    ParametricCompetingRisks,
)
from surpyval.utils.surpyval_data import SurpyvalData

TIMES = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 7.5])


def planted(seed=1, size=400):
    """Cause "a" is five times as likely where Z0 > 0.5; cause "b" is the
    same everywhere, and Z1 does nothing. Censored at 8."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (size, 3))
    t_a = rng.exponential(np.where(Z[:, 0] > 0.5, 2.0, 10.0))
    t_b = rng.exponential(np.where(Z[:, 1] > 2.0, 1.0, 6.0))
    x = np.minimum(np.minimum(t_a, t_b), 8.0)
    e = np.where(t_a < t_b, "a", "b").astype(object)
    e[x == 8.0] = None
    return x, Z, e


def weibull_ll(x, c, alpha, beta):
    """The Weibull log-likelihood of exact (c = 0) and right-censored
    times."""
    failed = c == 0
    z = x / alpha
    return float(
        np.sum(np.log(beta / alpha) + (beta - 1) * np.log(z[failed]))
        - np.sum(z**beta)
    )


def splits(node):
    """The tree's cuts, in order: (feature, value) pairs, None per leaf."""
    if isinstance(node, IntermediateNode):
        return (
            [(int(node.split_feature_index), float(node.split_feature_value))]
            + splits(node.left_child)
            + splits(node.right_child)
        )
    return [None]


def leaves(node):
    if isinstance(node, IntermediateNode):
        return leaves(node.left_child) + leaves(node.right_child)
    return [node]


# -- depth 0 is the plain competing-risks fit --------------------------------


def test_depth_zero_non_parametric_is_aalen_johansen():
    x, Z, e = planted()
    tree = SurvivalTree.fit(x, Z, e=e, kind="non-parametric", max_depth=0)
    plain = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    z = Z[0]
    for cause in ("a", "b"):
        np.testing.assert_array_equal(
            tree.cif(TIMES, z, cause), plain.cif(TIMES, cause)
        )
    np.testing.assert_array_equal(tree.sf(TIMES, z), plain.sf(TIMES))
    np.testing.assert_allclose(
        tree.total_cif(TIMES, z), 1 - plain.sf(TIMES), atol=1e-15
    )


def test_depth_zero_with_delayed_entry_is_aalen_johansen():
    x, Z, e = planted()
    tl = np.where(np.arange(x.size) % 3 == 0, np.minimum(0.4, x / 2), 0.0)
    tree = SurvivalTree.fit(
        x, Z, e=e, tl=tl, kind="non-parametric", max_depth=0
    )
    plain = CompetingRisks.fit(x, e, how="Kaplan-Meier", tl=tl)
    np.testing.assert_array_equal(
        tree.cif(TIMES, Z[0], "a"), plain.cif(TIMES, "a")
    )


def test_depth_zero_weibull_is_parametric_competing_risks():
    """Each cause's leaf is the plain Weibull tree's leaf on the
    cause-specific view (exactly), assembled as ParametricCompetingRisks;
    it is the maximum ParametricCompetingRisks.fit finds, found a little
    more precisely (its log-likelihood is no lower; the two differ by
    about 1e-6 relative, the optimiser's tolerance)."""
    x, Z, e = planted()
    tree = SurvivalTree.fit(x, Z, e=e, kind="weibull", max_depth=0)
    models = {}
    for cause in ("a", "b"):
        c_k = np.where(e == cause, 0, 1)
        models[cause] = SurvivalTree.fit(
            x, Z, c=c_k, kind="weibull", max_depth=0
        )._root.model
        fitted = Weibull.fit(x, c_k)
        leaf = tree._root.model.models[cause]
        assert weibull_ll(x, c_k, *leaf.params) >= weibull_ll(
            x, c_k, *fitted.params
        )
    assembled = ParametricCompetingRisks.from_fitted(models)
    for cause in ("a", "b"):
        np.testing.assert_array_equal(
            tree.cif(TIMES, Z[0], cause), assembled.cif(TIMES, cause)
        )
    np.testing.assert_array_equal(tree.sf(TIMES, Z[0]), assembled.sf(TIMES))


# -- the split rule ----------------------------------------------------------


@pytest.mark.parametrize("kind", ["non-parametric", "weibull"])
def test_cause_specific_rule_finds_a_one_cause_effect(kind):
    x, Z, e = planted()
    tree = SurvivalTree.fit(
        x, Z, e=e, kind=kind, max_depth=1, n_features_split="all"
    )
    root = tree._root
    assert isinstance(root, IntermediateNode)
    assert root.split_feature_index == 0
    assert abs(root.split_feature_value - 0.5) < 0.05
    low, high = [0.2, 0.5, 0.5], [0.8, 0.5, 0.5]
    # Cause "a" is the one that changes, and by about the planted amount
    # (exponential causes: CIF_a(5) = 0.5 (1 - e^-1) against
    # 0.75 (1 - e^-3.33))
    assert tree.cif(5.0, low, "a") == pytest.approx(0.316, abs=0.06)
    assert tree.cif(5.0, high, "a") == pytest.approx(0.725, abs=0.06)


def test_gray_rule_finds_the_nominated_cause_effect():
    x, Z, e = planted()
    tree = SurvivalTree.fit(
        x,
        Z,
        e=e,
        kind="non-parametric",
        max_depth=1,
        n_features_split="all",
        split="gray",
        event="a",
    )
    assert tree._root.split_feature_index == 0
    assert abs(tree._root.split_feature_value - 0.5) < 0.05


def test_gray_split_statistic_is_gray_test():
    """The best cut's statistic is gray_test's on its two children."""
    from surpyval.ml.forest import cause_specific_split as css
    from surpyval.univariate.competing_risks import gray_test

    x, Z, e = planted(size=120)
    data = SurpyvalData(x, np.where(e == None, 1, 0), e=e)  # noqa: E711
    seen = {}
    real = css.score_and_variance

    def spy(Y, d1, d2, rho):
        U, V = real(Y, d1, d2, rho)
        seen[float(Y[0, 0])] = U[0] ** 2 / V[0, 0]
        return U, V

    css.score_and_variance = spy
    try:
        u, v = css.gray_split(data, Z, 5, 2, [0], ["a", "b"], "a")
    finally:
        css.score_and_variance = real
    left = Z[:, u] <= v
    res = gray_test(x, e, left.astype(int), event="a")
    assert seen[float(left.sum())] == pytest.approx(res.statistic, rel=1e-10)


def test_one_cause_is_the_plain_log_rank_tree():
    x, Z, e = planted()
    one = np.where(e == None, None, "a").astype(object)  # noqa: E711
    c = (one == None).astype(int)  # noqa: E711
    kwargs = dict(kind="non-parametric", max_depth=3, random_state=4)
    with_cause = SurvivalTree.fit(x, Z, e=one, **kwargs)
    plain = SurvivalTree.fit(x, Z, c=c, **kwargs)
    assert splits(with_cause._root) == splits(plain._root)
    # And the split function itself agrees with log_rank_split
    data = SurpyvalData(x, c, e=one)
    assert cause_specific_split(data, Z, 5, 2, [0, 1, 2], ["a"]) == (
        log_rank_split(data, Z, 5, 2, [0, 1, 2])
    )


def test_parametric_children_keep_min_failures_of_each_cause():
    x, Z, e = planted()
    tree = SurvivalTree.fit(
        x, Z, e=e, kind="weibull", min_leaf_failures=6, random_state=0
    )
    # Every cut gives both children 6 failures of each cause the node had
    # 12 of; a rarer cause may fall to one side
    checked = 0
    stack = [(tree._root, tree.data.e)]
    while stack:
        node, e_node = stack.pop()
        if not isinstance(node, IntermediateNode):
            continue
        # The rows each child got, from the leaves under it
        left = np.concatenate(
            [leaf.data.e for leaf in leaves(node.left_child)]
        )
        right = np.concatenate(
            [leaf.data.e for leaf in leaves(node.right_child)]
        )
        for cause in ("a", "b"):
            if (e_node == cause).sum() >= 12:
                assert (left == cause).sum() >= 6
                assert (right == cause).sum() >= 6
                checked += 1
        stack += [(node.left_child, left), (node.right_child, right)]
    assert checked > 4


# -- leaves -------------------------------------------------------------------


def test_absent_cause_has_zero_incidence_and_is_described():
    rng = np.random.default_rng(3)
    Z = rng.uniform(0, 1, (200, 1))
    x = rng.exponential(5.0, 200)
    # Cause "b" occurs only where Z > 0.5
    e = np.where(
        (Z[:, 0] > 0.5) & (rng.uniform(size=200) < 0.5), "b", "a"
    ).astype(object)
    tree = SurvivalTree.fit(
        x, Z, e=e, kind="non-parametric", max_depth=1, min_leaf_failures=1
    )
    assert tree._root.split_feature_value == pytest.approx(0.5, abs=0.05)
    left = tree._root.left_child
    assert left.absent == ["b"]
    np.testing.assert_array_equal(tree.cif(TIMES, [0.1], "b"), 0.0)
    assert tree.cif(5.0, [0.9], "b") > 0
    assert "zero incidence (no failures here): 'b'" in tree.describe()


def test_leaf_with_no_failures():
    x = np.array([7.0, 8.0, 9.0, 1.0, 2.0, 3.0])
    Z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    e = np.array([None, None, None, "a", "b", "a"], dtype=object)
    tree = SurvivalTree.fit(
        x,
        Z,
        e=e,
        kind="non-parametric",
        min_leaf_samples=1,
        min_leaf_failures=0,
        max_depth=1,
    )
    assert isinstance(tree._root, IntermediateNode)
    np.testing.assert_array_equal(tree.sf([1, 9], [0.0]), [1.0, 1.0])
    np.testing.assert_array_equal(tree.cif([1, 9], [0.0], "a"), 0.0)
    assert "no failures" in tree.describe()
    restored = SurvivalTree.from_dict(json.loads(json.dumps(tree.to_dict())))
    np.testing.assert_array_equal(restored.sf([1, 9], [0.0]), [1.0, 1.0])


def test_shapes_and_missing_covariates():
    x, Z, e = planted()
    tree = SurvivalTree.fit(x, Z, e=e, kind="non-parametric", max_depth=2)
    assert np.ndim(tree.cif(5.0, Z[0], "a")) == 0
    assert tree.cif(TIMES, Z[:6], "a").shape == (6,)
    assert tree.cif(TIMES, Z[:4], "a", grid=True).shape == (4, TIMES.size)
    out = tree.cif([1.0, 2.0], [[np.nan, 0, 0], [0.5, 0.5, 0.5]], "a")
    assert np.isnan(out[0]) and np.isfinite(out[1])


# -- seeds and saving ---------------------------------------------------------


def test_random_state_reproduces_the_tree():
    x, Z, e = planted()
    kwargs = dict(e=e, max_depth=3, n_features_split=1)
    one = SurvivalTree.fit(x, Z, random_state=7, **kwargs)
    two = SurvivalTree.fit(x, Z, random_state=7, **kwargs)
    assert one.to_dict() == two.to_dict()
    state = np.random.get_state()
    SurvivalTree.fit(x, Z, random_state=7, **kwargs)
    after = np.random.get_state()
    assert state[1].tolist() == after[1].tolist() and state[2] == after[2]


@pytest.mark.parametrize("kind", ["non-parametric", "weibull", "exponential"])
def test_to_dict_round_trip(kind):
    x, Z, e = planted()
    tree = SurvivalTree.fit(x, Z, e=e, kind=kind, max_depth=2, random_state=0)
    saved = json.loads(json.dumps(tree.to_dict()))
    restored = SurvivalTree.from_dict(saved)
    assert restored.causes == ["a", "b"]
    assert restored.split == "cause-specific"
    for cause in ("a", "b"):
        np.testing.assert_array_equal(
            restored.cif(TIMES, Z[:6], cause), tree.cif(TIMES, Z[:6], cause)
        )
    np.testing.assert_array_equal(
        restored.sf(TIMES, Z[:6]), tree.sf(TIMES, Z[:6])
    )
    assert restored.describe() == tree.describe()


def test_round_trip_keeps_nominated_event_and_tuple_labels():
    x, Z, e = planted()
    labels = np.array([None if v is None else (v, 1) for v in e], dtype=object)
    tree = SurvivalTree.fit(
        x,
        Z,
        e=labels,
        kind="non-parametric",
        max_depth=1,
        split="gray",
        event=("a", 1),
    )
    restored = SurvivalTree.from_dict(json.loads(json.dumps(tree.to_dict())))
    assert restored.event == ("a", 1)
    np.testing.assert_array_equal(
        restored.cif(TIMES, Z[0], ("b", 1)), tree.cif(TIMES, Z[0], ("b", 1))
    )


# -- without causes nothing changes; refusals ---------------------------------


def test_without_causes_the_tree_is_unchanged():
    x, Z, e = planted()
    c = (e == None).astype(int)  # noqa: E711
    tree = SurvivalTree.fit(x, Z, c=c, kind="non-parametric", max_depth=2)
    assert tree.causes is None
    assert "causes" not in tree.to_dict()
    assert all(type(leaf) is TerminalNode for leaf in leaves(tree._root))
    with pytest.raises(ValueError, match="grown without causes"):
        tree.cif(1.0, Z[0], "a")


def test_fit_from_df_reads_the_causes():
    x, Z, e = planted()
    df = pd.DataFrame({"x": x, "z0": Z[:, 0], "z1": Z[:, 1], "z2": Z[:, 2]})
    df["cause"] = e
    tree = SurvivalTree.fit_from_df(
        df,
        x_col="x",
        Z_cols=["z0", "z1", "z2"],
        e_col="cause",
        kind="non-parametric",
        max_depth=1,
        random_state=0,
    )
    direct = SurvivalTree.fit(
        x, Z, e=e, kind="non-parametric", max_depth=1, random_state=0
    )
    np.testing.assert_array_equal(
        tree.cif(TIMES, Z[:6], "a"), direct.cif(TIMES, Z[:6], "a")
    )


@pytest.mark.parametrize(
    "options, match",
    [
        ({"selection": "ctree"}, "selection='ctree' is not built"),
        ({"min_split_gain": "aic"}, "min_split_gain"),
        ({"split": "gray"}, "name it with event"),
        ({"split": "both"}, "'split' must be one of"),
        ({"event": "c"}, "Unknown cause 'c'"),
    ],
)
def test_refusals(options, match):
    x, Z, e = planted(size=60)
    with pytest.raises(ValueError, match=match):
        SurvivalTree.fit(x, Z, e=e, **options)


def test_refuses_interval_censoring_and_masked_causes():
    x, Z, e = planted(size=60)
    c = (e == None).astype(int)  # noqa: E711
    c[0] = 2
    xi = np.column_stack([x, x])
    xi[0] = [x[0] / 2, x[0]]
    if e[0] is None:
        e = e.copy()
        e[0] = "a"
    with pytest.raises(ValueError, match="left- or interval-censored"):
        SurvivalTree.fit(xi, Z, c=c, e=e)
    masked = e.copy()
    masked[np.flatnonzero(masked != None)[0]] = None  # noqa: E711
    c_masked = (e == None).astype(int)  # noqa: E711
    with pytest.warns(UserWarning, match="masked causes"):
        with pytest.raises(ValueError, match="have no cause"):
            SurvivalTree.fit(x, Z, c=c_masked, e=masked)


def test_gray_refuses_delayed_entry_and_options_need_causes():
    x, Z, e = planted(size=60)
    with pytest.raises(ValueError, match="delayed entry"):
        SurvivalTree.fit(
            x,
            Z,
            e=e,
            tl=np.full(x.size, 0.5) * (x > 0.5),
            split="gray",
            event="a",
        )
    with pytest.raises(ValueError, match="pass the causes as e"):
        SurvivalTree.fit(x, Z, split="gray")


def test_leaf_class():
    x, Z, e = planted(size=60)
    tree = SurvivalTree.fit(x, Z, e=e, max_depth=0)
    assert isinstance(tree._root, CompetingRisksLeaf)
    assert isinstance(tree._root.model, ParametricCompetingRisks)
