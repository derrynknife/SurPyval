"""A random survival forest grown with the cause of each failure (``e``)."""

import json

import numpy as np
import pandas as pd
import pytest

from surpyval.metrics.concordance import concordance_index
from surpyval.ml import RandomSurvivalForest, SurvivalTree
from surpyval.ml.forest.forest import _cause_outcomes
from surpyval.ml.forest.oob import _CauseLeafCurve

TIMES = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 7.5])


def planted(seed=1, size=400):
    """Cause "a" is five times as likely where Z0 > 0.5; cause "b" is the
    same everywhere, and Z1 and Z2 do nothing. Censored at 8."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (size, 3))
    t_a = rng.exponential(np.where(Z[:, 0] > 0.5, 2.0, 10.0))
    t_b = rng.exponential(6.0, size)
    x = np.minimum(np.minimum(t_a, t_b), 8.0)
    e = np.where(t_a < t_b, "a", "b").astype(object)
    e[x == 8.0] = None
    return x, Z, e


@pytest.mark.parametrize("kind", ["non-parametric", "weibull", "exponential"])
def test_one_tree_without_bootstrap_is_the_tree(kind):
    x, Z, e = planted(size=150)
    forest = RandomSurvivalForest.fit(
        x,
        Z,
        e=e,
        n_trees=1,
        bootstrap=False,
        max_depth=2,
        n_features_split="all",
        kind=kind,
        random_state=0,
    )
    tree = SurvivalTree.fit(
        x, Z, e=e, max_depth=2, n_features_split="all", kind=kind
    )
    for cause in ("a", "b"):
        np.testing.assert_allclose(
            forest.cif(TIMES, Z[:10], cause, grid=True),
            tree.cif(TIMES, Z[:10], cause, grid=True),
            rtol=1e-12,
            atol=1e-15,
        )
    np.testing.assert_allclose(
        forest.sf(TIMES, Z[:10], grid=True),
        tree.sf(TIMES, Z[:10], grid=True),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        forest.total_cif(TIMES, Z[:10], grid=True),
        tree.total_cif(TIMES, Z[:10], grid=True),
        rtol=1e-12,
        atol=1e-15,
    )


def test_cif_is_the_average_of_the_trees():
    x, Z, e = planted(size=150)
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=4, max_depth=2, kind="weibull", random_state=3
    )
    expected = np.mean(
        [tree.cif(TIMES, Z[:5], "a", grid=True) for tree in forest.trees],
        axis=0,
    )
    np.testing.assert_allclose(
        forest.cif(TIMES, Z[:5], "a", grid=True), expected, rtol=1e-12
    )
    # The causes add up to the all-cause failure probability
    np.testing.assert_allclose(
        forest.total_cif(TIMES, Z[:5], grid=True),
        forest.ff(TIMES, Z[:5], grid=True),
        rtol=1e-6,
        atol=1e-9,
    )


# A tiny data set, by hand: three trees of depth 0 with Exponential
# leaves, so each tree's leaf has the closed-form rates d_k / exposure.
TINY_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
TINY_E = np.array(["a", "b", None, "a", "b", "a", None, "b"], dtype=object)
TINY_Z = np.arange(8.0).reshape(-1, 1)


def _tiny_forest(n_trees=3):
    return RandomSurvivalForest.fit(
        TINY_X,
        TINY_Z,
        e=TINY_E,
        n_trees=n_trees,
        max_depth=0,
        min_leaf_failures=1,
        min_leaf_samples=1,
        kind="exponential",
        random_state=0,
    )


def _hand_oob(forest, event=None):
    rates = []
    for idx in forest.bootstrap_indices:
        exposure = TINY_X[idx].sum()
        rates.append(
            {k: float(np.sum(TINY_E[idx] == k)) / exposure for k in ("a", "b")}
        )
    lls = []
    for i in range(TINY_X.size):
        trees = [
            t for t, idx in enumerate(forest.bootstrap_indices) if i not in idx
        ]
        if not trees:
            continue
        x, cause = TINY_X[i], TINY_E[i]
        terms = []
        for t in trees:
            lam = rates[t]
            if event is None:
                total = lam["a"] + lam["b"]
                rate = 1.0 if cause is None else lam[cause]
                terms.append(rate * np.exp(-total * x))
            else:
                rate = lam[event] if cause == event else 1.0
                terms.append(rate * np.exp(-lam[event] * x))
        lls.append(np.log(np.mean(terms)))
    return float(np.mean(lls))


@pytest.mark.parametrize("event", [None, "a", "b"])
def test_oob_log_likelihood_by_hand(event):
    forest = _tiny_forest()
    assert forest.oob_log_likelihood(event) == pytest.approx(
        _hand_oob(forest, event), rel=1e-10
    )


def test_one_parametric_tree_restricted_scores_add_up():
    # With one tree the ensemble is its leaf, whose likelihood is the
    # product of the causes' cause-specific likelihoods.
    forest = _tiny_forest(n_trees=1)
    full = forest.oob_log_likelihood()
    parts = forest.oob_log_likelihood("a") + forest.oob_log_likelihood("b")
    assert np.isfinite(full)
    assert parts == pytest.approx(full, rel=1e-10)


def test_step_leaf_sub_densities_add_up_to_the_all_cause_density():
    x, Z, e = planted(size=120)
    tree = SurvivalTree.fit(x, Z, e=e, kind="non-parametric", max_depth=0)
    curve = _CauseLeafCurve(tree._root, 0.0)
    # Points inside the linear segments and in the tail
    points = np.concatenate(
        [
            (curve.knots_x[:-1] + curve.knots_x[1:]) / 2,
            curve.knots_x[-1] + np.array([0.5, 3.0]),
        ]
    )
    h = 1e-7
    density = (curve.sf(points - h) - curve.sf(points + h)) / (2 * h)
    total = curve.sub_df(points, "a") + curve.sub_df(points, "b")
    np.testing.assert_allclose(total, density, rtol=1e-5)
    # At the knots the joined incidences are the Aalen-Johansen ones
    knots = curve.knots_x[1:]
    np.testing.assert_allclose(
        curve.knots_F["a"][1:], tree._root.model.cif(knots, "a"), rtol=1e-12
    )


def test_importances_pick_out_the_cause_specific_feature():
    x, Z, e = planted(seed=1, size=400)
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=30, max_depth=3, kind="exponential", random_state=2
    )
    imp_a = forest.feature_importances(event="a", random_state=0)
    imp_b = forest.feature_importances(event="b", random_state=0)
    imp = forest.feature_importances(random_state=0)
    noise_a = max(abs(imp_a["Z1"]), abs(imp_a["Z2"]))
    assert imp_a["Z0"] > 5 * noise_a
    assert imp_a["Z0"] > 5 * abs(imp_b["Z0"])
    assert imp["Z0"] > 5 * max(abs(imp["Z1"]), abs(imp["Z2"]))


def test_n_jobs_does_not_change_the_forest():
    x, Z, e = planted(size=150)
    kwargs = dict(e=e, n_trees=4, max_depth=2, kind="weibull", random_state=5)
    one = RandomSurvivalForest.fit(x, Z, n_jobs=1, **kwargs)
    two = RandomSurvivalForest.fit(x, Z, n_jobs=2, **kwargs)
    np.testing.assert_array_equal(
        one.cif(TIMES, Z[:6], "a", grid=True),
        two.cif(TIMES, Z[:6], "a", grid=True),
    )
    assert one.oob_log_likelihood("b") == two.oob_log_likelihood("b")


def test_a_cause_missing_from_a_bootstrap_sample():
    # One failure of cause "c": most bootstrap samples miss it, and each
    # tree is still grown for it, giving it zero incidence.
    x, Z, e = planted(size=60)
    e = e.copy()
    e[np.flatnonzero(e == "a")[0]] = "c"
    forest = RandomSurvivalForest.fit(
        x,
        Z,
        e=e,
        n_trees=6,
        max_depth=1,
        kind="non-parametric",
        random_state=0,
    )
    assert forest.causes == ["a", "b", "c"]
    assert all(tree.causes == ["a", "b", "c"] for tree in forest.trees)
    values = forest.cif(TIMES, Z[:4], "c", grid=True)
    assert np.all((values >= 0) & (values < 0.2))
    # The one failure from "c", out of bag, lands only in trees that never
    # saw the cause: zero probability, said with a count.
    with pytest.warns(UserWarning, match="1 of .* zero probability"):
        assert forest.oob_log_likelihood("c") == -np.inf


def test_serialisation_round_trip():
    x, Z, e = planted(size=120)
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=3, max_depth=2, kind="weibull", random_state=1
    )
    restored = RandomSurvivalForest.from_dict(
        json.loads(json.dumps(forest.to_dict()))
    )
    assert restored.causes == ["a", "b"]
    assert restored.split == "cause-specific"
    for cause in ("a", "b"):
        np.testing.assert_allclose(
            restored.cif(TIMES, Z[:5], cause, grid=True),
            forest.cif(TIMES, Z[:5], cause, grid=True),
            rtol=1e-12,
        )
    assert "causes=['a', 'b']" in repr(restored)


def test_fit_from_df_with_e_col():
    x, Z, e = planted(size=120)
    df = pd.DataFrame({"x": x, "e": e, "z0": Z[:, 0], "z1": Z[:, 1]})
    forest = RandomSurvivalForest.fit_from_df(
        df,
        x_col="x",
        e_col="e",
        Z_cols=["z0", "z1"],
        n_trees=3,
        max_depth=1,
        kind="non-parametric",
        random_state=0,
    )
    assert forest.causes == ["a", "b"]
    assert forest.cif(5.0, df[["z0", "z1"]].iloc[:3], "a").shape == (3,)


def test_cause_specific_mortality_and_score():
    x, Z, e = planted(size=150)
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=5, max_depth=2, kind="exponential", random_state=0
    )
    c = np.array([v is None for v in e], dtype=int)
    np.testing.assert_allclose(
        forest.mortality(TIMES, Z[:4], "a"),
        forest.cif(TIMES, Z[:4], "a", grid=True).sum(1),
    )
    x_k, c_k = _cause_outcomes(x, c, e, "a")
    expected = concordance_index(x_k, c_k, forest.mortality(x, Z, "a"))
    assert forest.score(x, Z, c, e=e, event="a") == pytest.approx(expected)
    # A failure from another cause never fails from "a": later than all
    assert np.all(np.isinf(x_k[(c == 0) & (e != "a")]))
    assert np.all(c_k[e == "a"] == 0) and np.all(c_k[e != "a"] == 1)
    # The all-cause index is unchanged
    assert forest.score(x, Z, c) == pytest.approx(
        concordance_index(x, c, forest.mortality(x, Z))
    )


def test_refusals_say_what_to_do():
    x, Z, e = planted(size=80)
    c = np.array([v is None for v in e], dtype=int)
    plain = RandomSurvivalForest.fit(
        x, Z, c=c, n_trees=2, max_depth=1, random_state=0
    )
    with pytest.raises(ValueError, match="grown without causes"):
        plain.cif(1.0, Z[:2], "a")
    with pytest.raises(ValueError, match="grown without causes"):
        plain.oob_log_likelihood("a")
    with pytest.raises(ValueError, match="for a forest grown with the cause"):
        RandomSurvivalForest.fit(x, Z, c=c, n_trees=2, event="a")
    forest = RandomSurvivalForest.fit(
        x, Z, e=e, n_trees=2, max_depth=1, random_state=0
    )
    with pytest.raises(ValueError, match="'z'"):
        forest.cif(1.0, Z[:2], "z")
    with pytest.raises(ValueError, match="'z'"):
        forest.feature_importances(event="z")
    with pytest.raises(ValueError, match="both e"):
        forest.score(x, Z, c, e=e)
    with pytest.raises(ValueError, match="selection='ctree'.*forest"):
        RandomSurvivalForest.fit(x, Z, e=e, n_trees=2, selection="ctree")
    with pytest.raises(ValueError, match="name it with event"):
        RandomSurvivalForest.fit(x, Z, e=e, n_trees=2, split="gray")
