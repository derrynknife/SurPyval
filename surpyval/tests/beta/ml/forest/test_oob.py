"""Out-of-bag log-likelihood and permutation importance (issue #186).

``RandomSurvivalForest.oob_log_likelihood`` scores every training row
with the trees whose bootstrap sample left it out, by its full
likelihood (density, ``S``, ``F``, interval probability, over the
truncation probability), and ``feature_importances`` reports the drop in
that score when a feature is shuffled among the out-of-bag rows. These
check:

- a small case against the closed-form exponential MLE of each tree;
- the continuous reading of a step-function leaf (Nelson-Aalen) by hand;
- every censoring type and truncation, against a row-by-row evaluation
  through the trees' own ``sf`` and ``df``;
- an informative feature ranks above noise features;
- the seed rule for the shuffles;
- the warning for rows no tree left out, and a restored forest.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import NelsonAalen
from surpyval.beta.ml.forest import RandomSurvivalForest
from surpyval.beta.ml.forest.oob import _LeafCurve

# A small forest leaves a few rows in every tree's sample, which the
# out-of-bag methods leave out with a warning (tested below).
IN_EVERY_SAMPLE = pytest.mark.filterwarnings(
    "ignore:.*in the sample of every tree"
)


def _fit(**kwargs):
    # The forest reports progress through joblib on stderr.
    with contextlib.redirect_stderr(io.StringIO()):
        return RandomSurvivalForest.fit(**kwargs)


def _signal_data(n=150, seed=0):
    """Feature 0 halves the life above 0.5; features 1 and 2 are noise."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 3))
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, 0.4, 1.0)
    C = rng.uniform(5, 25, n)
    return np.minimum(T, C), Z, (C < T).astype(int)


def _oob_trees(forest, i):
    return [
        tree
        for tree, idx in zip(forest.trees, forest.bootstrap_indices)
        if i not in idx
    ]


def test_hand_checked_exponential_stumps():
    # max_depth=0: each tree is one exponential leaf fitted to its
    # bootstrap sample, whose MLE rate is failures / total time.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    c = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    Z = np.arange(8.0)
    np.random.seed(3)
    forest = _fit(x=x, Z=Z, c=c, n_trees=6, max_depth=0, kind="exponential")
    rates = [
        (c[idx] == 0).sum() / x[idx].sum() for idx in forest.bootstrap_indices
    ]

    expected = []
    for i in range(8):
        lam = np.array(
            [
                rate
                for rate, idx in zip(rates, forest.bootstrap_indices)
                if i not in idx
            ]
        )
        if lam.size == 0:
            continue
        if c[i] == 0:
            # the density of the averaged exponentials
            expected.append(np.log(np.mean(lam * np.exp(-lam * x[i]))))
        else:
            expected.append(np.log(np.mean(np.exp(-lam * x[i]))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got = forest.oob_log_likelihood()
    assert got == pytest.approx(np.mean(expected), rel=1e-6)


def test_step_leaf_is_read_as_a_continuous_distribution():
    # Nelson-Aalen of 1, 2, 2+, 3, 5+: drops at 1 (1/5), 2 (1/4) and
    # 3 (1/2), so S = exp(-0.2), exp(-0.45), exp(-0.95) there.
    model = NelsonAalen.fit([1, 2, 2, 3, 5], c=[0, 0, 1, 0, 1])
    curve = _LeafCurve(model, 0.0)
    S1, S2, S3 = np.exp(-0.2), np.exp(-0.45), np.exp(-0.95)
    rate = 0.95 / 3  # the average hazard up to the last drop
    q = np.array([-1.0, 0.5, 1.0, 1.5, 2.5, 3.0, 4.0, np.inf])
    np.testing.assert_allclose(
        curve.sf(q),
        [
            1.0,
            1 - (1 - S1) / 2,
            S1,
            (S1 + S2) / 2,
            (S2 + S3) / 2,
            S3,
            S3 * np.exp(-rate),
            0.0,
        ],
    )
    np.testing.assert_allclose(
        curve.df(q),
        [
            0.0,
            1 - S1,
            1 - S1,
            S1 - S2,
            S2 - S3,
            S2 - S3,
            rate * S3 * np.exp(-rate),
            0.0,
        ],
    )
    # A proper distribution: the density integrates to one.
    grid = np.linspace(0, 300, 300001)
    assert np.trapezoid(curve.df(grid), grid) == pytest.approx(1, abs=1e-4)


def _full_data_model(n=60, seed=3):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 2))
    T = rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, 4.0, 10.0)
    c = np.resize([0, 1, -1, 2], n)
    xl = np.where(c == 2, 0.7 * T, T)
    xr = np.where(c == 2, 1.3 * T, T)
    tl = np.where(rng.uniform(size=n) < 0.3, 0.1 * T, -np.inf)
    tr = np.where(rng.uniform(size=n) < 0.3, 3.0 * T, np.inf)
    return np.column_stack([xl, xr]), Z, c, tl, tr


def test_every_censoring_type_and_truncation():
    x, Z, c, tl, tr = _full_data_model()
    np.random.seed(1)
    forest = _fit(
        x=x, Z=Z, c=c, tl=tl, tr=tr, n_trees=8, max_depth=1, kind="exponential"
    )
    lls = []
    for i in range(len(c)):
        trees = _oob_trees(forest, i)
        if not trees:
            continue

        def S(v, trees=trees, i=i):
            if np.isinf(v):
                return 1.0 if v < 0 else 0.0
            return np.mean([float(t.sf(v, Z[i])) for t in trees])

        lo, hi = x[i]
        if c[i] == 0:
            num = np.mean([float(t.df(lo, Z[i])) for t in trees])
        elif c[i] == 1:
            num = S(max(lo, tl[i])) - S(tr[i])
        elif c[i] == -1:
            num = S(tl[i]) - S(min(hi, tr[i]))
        else:
            num = S(max(lo, tl[i])) - S(min(hi, tr[i]))
        lls.append(np.log(num) - np.log(S(tl[i]) - S(tr[i])))
    assert set(c) == {0, 1, -1, 2}
    with warnings.catch_warnings():
        # a rare row in every sample is left out, with a UserWarning;
        # no raw numerical warning may escape
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        got = forest.oob_log_likelihood()
    assert np.isfinite(got)
    assert got == pytest.approx(np.mean(lls), rel=1e-9)


@IN_EVERY_SAMPLE
def test_non_parametric_kind_with_left_truncation():
    x, Z, c = _signal_data(n=120)
    tl = np.where(np.arange(120) % 3 == 0, 0.2 * x, 0.0)
    np.random.seed(0)
    forest = _fit(
        x=x, Z=Z, c=c, tl=tl, n_trees=10, max_depth=2, kind="non-parametric"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isfinite(forest.oob_log_likelihood())


@IN_EVERY_SAMPLE
@pytest.mark.parametrize("kind", ["non-parametric", "exponential"])
def test_splits_beat_the_pooled_fit_out_of_bag(kind):
    x, Z, c = _signal_data()
    scores = {}
    for depth in (0, 2):
        np.random.seed(0)
        forest = _fit(
            x=x,
            Z=Z,
            c=c,
            n_trees=10,
            max_depth=depth,
            kind=kind,
            n_features_split="all",
        )
        scores[depth] = forest.oob_log_likelihood()
    assert scores[2] > scores[0] + 0.05, scores


def test_informative_feature_ranks_above_noise():
    x, Z, c = _signal_data()
    np.random.seed(0)
    forest = _fit(
        x=x, Z=Z, c=c, n_trees=15, max_depth=2, kind="non-parametric"
    )
    importances = forest.feature_importances(random_state=0)
    assert importances.shape == (3,)
    assert importances["Z0"] > 0.1
    assert np.abs(importances.iloc[1:]).max() < 0.05
    assert importances["Z0"] > importances.iloc[1:].max() + 0.05


@IN_EVERY_SAMPLE
def test_importance_seed_rule():
    x, Z, c = _signal_data(n=80)
    np.random.seed(0)
    forest = _fit(x=x, Z=Z, c=c, n_trees=6, max_depth=2, kind="non-parametric")
    a = forest.feature_importances(n_repeats=2, random_state=1)
    b = forest.feature_importances(n_repeats=2, random_state=1)
    d = forest.feature_importances(n_repeats=2, random_state=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d)

    # None draws from the global stream; an explicit seed leaves it alone
    np.random.seed(5)
    e = forest.feature_importances(n_repeats=2)
    np.random.seed(5)
    f = forest.feature_importances(n_repeats=2)
    np.testing.assert_array_equal(e, f)
    np.random.seed(5)
    forest.feature_importances(n_repeats=2, random_state=1)
    after_explicit = np.random.random()
    np.random.seed(5)
    assert np.random.random() == after_explicit


@pytest.mark.parametrize("n_repeats", [0, -1, 1.5, True, "2"])
def test_bad_n_repeats_raises(n_repeats):
    x, Z, c = _signal_data(n=40)
    np.random.seed(0)
    forest = _fit(x=x, Z=Z, c=c, n_trees=2, max_depth=0, kind="exponential")
    with pytest.raises(ValueError, match="n_repeats"):
        forest.feature_importances(n_repeats=n_repeats)


def test_no_bootstrap_has_no_out_of_bag_rows():
    x, Z, c = _signal_data(n=40)
    forest = _fit(
        x=x, Z=Z, c=c, n_trees=2, bootstrap=False, kind="exponential"
    )
    with pytest.warns(UserWarning, match="40 of 40 rows") as record:
        assert np.isnan(forest.oob_log_likelihood())
    assert len(record) == 1
    assert record[0].filename == __file__


def test_rows_in_every_sample_are_left_out_with_one_warning():
    x, Z, c = _signal_data(n=40)
    np.random.seed(0)
    forest = _fit(x=x, Z=Z, c=c, n_trees=2, max_depth=0, kind="exponential")
    in_all = set(forest.bootstrap_indices[0]) & set(
        forest.bootstrap_indices[1]
    )
    assert in_all  # this seed has some
    with pytest.warns(UserWarning, match=f"{len(in_all)} of 40 rows") as rec:
        assert np.isfinite(forest.oob_log_likelihood())
    assert len(rec) == 1


def test_counts_weight_the_mean():
    # A row with n=3 counts three times in the mean.
    x, Z, c = _signal_data(n=60)
    n = np.where(np.arange(60) % 4 == 0, 3, 1)
    np.random.seed(0)
    forest = _fit(
        x=x, Z=Z, c=c, n=n, n_trees=10, max_depth=1, kind="exponential"
    )
    oob, terms, origin, n_oob = forest._oob_setup()
    ll = forest._oob_rows_log_likelihood(oob, terms, origin, n_oob, {})
    has = ~np.isnan(ll)
    assert forest.oob_log_likelihood() == pytest.approx(
        np.sum(n[has] * ll[has]) / np.sum(n[has])
    )


def test_restored_forest_predicts_but_cannot_score_out_of_bag():
    x, Z, c = _signal_data(n=60)
    np.random.seed(0)
    forest = _fit(x=x, Z=Z, c=c, n_trees=4, max_depth=2, kind="non-parametric")
    restored = surpyval.from_dict(forest.to_dict())
    np.testing.assert_array_equal(
        restored.sf([2.0, 5.0], Z[:5]), forest.sf([2.0, 5.0], Z[:5])
    )
    # the bootstrap samples are not serialised: they are no use without
    # the training data, which is not either
    assert "bootstrap_indices" not in forest.to_dict()
    assert restored.to_dict() == forest.to_dict()
    with pytest.raises(ValueError, match="from_dict"):
        restored.oob_log_likelihood()
    with pytest.raises(ValueError, match="from_dict"):
        restored.feature_importances()
