"""Regression tests for the third docs-review bug-fix round (competing risks).

- ``gray_test`` is Gray's (1988) construction, with each group's
  subdistribution risk set (and so its censoring distribution) estimated
  from the group's own data and Gray's asymptotic variance: it stays
  calibrated when the groups are censored differently (the pooled censoring
  estimate rejected a true null up to 89% of the time), and it refuses
  invalid input instead of silently mis-reading it.
- ``ParametricCompetingRisks.cif`` / ``probability_of_cause`` integrate each
  query accurately (over the cause's probability scale) instead of on one
  fixed 4,000-point grid.
- Cause and group labels of mixed types are ordered by one shared helper,
  tuple labels work and serialise, and several bad inputs raise clear errors.
"""

import json

import numpy as np
import pytest
from scipy.stats import norm

import surpyval
from surpyval import Exponential, LogNormal, Weibull, gray_test
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)
from surpyval.univariate.competing_risks.labels import ordered_labels

# -- Gray's test --------------------------------------------------------------


def _sim_cr(rng, n, h1, h2, cens_mean):
    t1 = rng.exponential(1 / h1, n)
    t2 = rng.exponential(1 / h2, n)
    cz = rng.exponential(cens_mean, n)
    x = np.minimum.reduce([t1, t2, cz])
    e = np.where(x == cz, None, np.where(t1 < t2, 1, 2)).astype(object)
    return x, e


@pytest.mark.parametrize("cause", [1, 2])
def test_gray_calibrated_under_unequal_censoring(cause):
    # Identical cause-specific hazards, censoring means 2 and 50. With one
    # pooled censoring estimate about half of these tests rejected at 5%.
    rng = np.random.default_rng(4)
    stats = []
    for _ in range(150):
        x0, e0 = _sim_cr(rng, 400, 0.1, 0.2, 2.0)
        x1, e1 = _sim_cr(rng, 400, 0.1, 0.2, 50.0)
        res = gray_test(
            np.concatenate([x0, x1]),
            np.concatenate([e0, e1]),
            np.repeat([0, 1], 400),
            cause=cause,
        )
        stats.append(res.statistic)
    stats = np.array(stats)
    assert np.mean(stats > 3.841) < 0.1
    # a chi-square(1) statistic has mean 1
    assert 0.75 < stats.mean() < 1.3


def test_gray_equal_censoring_close_to_pooled_version():
    # The docstring example: the pooled-censoring version gave 19.878.
    rng = np.random.default_rng(0)
    group = rng.binomial(1, 0.5, 200)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * group)))
    t_b = rng.exponential(1 / 0.05, 200)
    t_c = rng.uniform(0, 20, 200)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    first = np.where(t_a < t_b, "a", "b")
    e = np.where(t_c < np.minimum(t_a, t_b), None, first)
    res = gray_test(x, e, group, cause="a")
    assert res.statistic == pytest.approx(19.962, abs=1e-3)


def test_gray_invariances_hold():
    rng = np.random.default_rng(3)
    x, e = _sim_cr(rng, 240, 0.1, 0.2, 10.0)
    g = np.repeat([0, 1, 2], 80)
    n = rng.integers(1, 4, 240)
    base = gray_test(x, e, g, cause=1, n=n)
    expanded = gray_test(
        np.repeat(x, n), np.repeat(e, n), np.repeat(g, n), cause=1
    )
    relabelled = gray_test(x, e, np.array(["b", "c", "a"])[g], cause=1, n=n)
    assert expanded.statistic == pytest.approx(base.statistic)
    assert relabelled.statistic == pytest.approx(base.statistic)


X6 = [1, 2, 3, 4, 5, 6]
E6 = [1, 2, 1, 2, 1, 2]
G6 = [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"c": [-1, 0, 0, 0, 0, 0]}, "c must be 0"),
        ({"c": [2, 0, 0, 0, 0, 0]}, "c must be 0"),
        ({"n": [0, 1, 1, 1, 1, 1]}, "n must be finite and positive"),
        ({"n": [-1, 1, 1, 1, 1, 1]}, "n must be finite and positive"),
        ({"n": [1, 1]}, "one count per time"),
        ({"rho": np.nan}, "rho"),
    ],
)
def test_gray_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gray_test(X6, E6, G6, cause=1, **kwargs)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_gray_rejects_non_finite_times(bad):
    with pytest.raises(ValueError, match="finite"):
        gray_test([bad, 2, 3, 4, 5, 6], E6, G6, cause=1)


@pytest.mark.parametrize("missing", [None, np.nan])
def test_gray_rejects_missing_group_labels(missing):
    with pytest.raises(ValueError, match="missing label"):
        gray_test(X6, E6, [missing, 0, 0, 1, 1, 1], cause=1)


def test_gray_rejects_length_mismatches():
    with pytest.raises(ValueError, match="one label per time"):
        gray_test(X6, E6, G6[:-1], cause=1)
    with pytest.raises(ValueError, match="one cause per time"):
        gray_test(X6, E6[:-1], G6, cause=1)
    with pytest.raises(ValueError, match="empty"):
        gray_test([], [], [], cause=1)


def test_gray_mixed_and_tuple_labels():
    res = gray_test(X6, E6, [0, "a", 0, "a", 0, "a"], cause=1)
    assert res.groups == [0, "a"]
    causes = [("a", 1), ("b", 2)] * 3
    tup = gray_test(X6, causes, G6, cause=("a", 1))
    assert tup.statistic == pytest.approx(gray_test(X6, E6, G6, 1).statistic)


# -- ParametricCompetingRisks: accurate incidence ----------------------------


@pytest.fixture
def _no_runtime_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@pytest.mark.usefixtures("_no_runtime_warnings")
def test_parametric_cif_accurate_over_a_wide_query():
    model = ParametricCompetingRisks.from_fitted(
        {
            "a": Exponential.from_params([0.1]),
            "b": Exponential.from_params([0.3]),
        }
    )
    t = np.array([0.01, 0.1, 1.0, 1e4])
    true = 0.1 / 0.4 * (1 - np.exp(-0.4 * t))
    np.testing.assert_allclose(model.cif(t, "a"), true, rtol=1e-9)
    # every value is independent of the other requested times
    alone = [model.cif(ti, "a") for ti in t]
    np.testing.assert_allclose(model.cif(t, "a"), alone, rtol=1e-12)


@pytest.mark.usefixtures("_no_runtime_warnings")
def test_parametric_cifs_sum_to_ff_with_infinite_density_at_zero():
    model = ParametricCompetingRisks.from_fitted(
        {
            "a": Weibull.from_params([10, 0.5]),
            "b": Weibull.from_params([20, 0.7]),
        }
    )
    t = np.array([0.01, 1.0, 10.0, 100.0])
    total = model.cif(t, "a") + model.cif(t, "b")
    np.testing.assert_allclose(total, model.ff(t), rtol=1e-9)
    p = model.probability_of_cause("a") + model.probability_of_cause("b")
    assert p == pytest.approx(1.0, abs=1e-9)


@pytest.mark.usefixtures("_no_runtime_warnings")
def test_probability_of_cause_heavy_tailed_lognormals():
    model = ParametricCompetingRisks.from_fitted(
        {
            "a": LogNormal.from_params([0, 3]),
            "b": LogNormal.from_params([1, 3]),
        }
    )
    pa = model.probability_of_cause("a")
    pb = model.probability_of_cause("b")
    # P(T_a < T_b) for independent lognormals
    assert pa == pytest.approx(norm.cdf(1 / np.sqrt(18)), abs=1e-9)
    assert pa + pb == pytest.approx(1.0, abs=1e-9)


def test_parametric_cif_shapes_and_edges():
    model = ParametricCompetingRisks.from_fitted(
        {
            "a": Weibull.from_params([10, 2]),
            "b": Exponential.from_params([0.05]),
        }
    )
    assert isinstance(model.cif(5.0, "a"), float)
    grid = np.array([[1.0, 5.0], [10.0, 20.0]])
    out = model.cif(grid, "a")
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.ravel(), model.cif(grid.ravel(), "a"))
    assert model.cif([-5.0, 0.0], "a").tolist() == [0.0, 0.0]
    assert model.cif(np.inf, "a") == pytest.approx(
        model.probability_of_cause("a")
    )


def test_parametric_cure_fraction_probabilities():
    cured = Weibull.from_params([5, 2], p=0.4)
    model = ParametricCompetingRisks.from_fitted(
        {"a": cured, "b": Weibull.from_params([8, 3], p=0.5)}
    )
    total = model.probability_of_cause("a") + model.probability_of_cause("b")
    # 1 - P(never fails) = 1 - 0.6 * 0.5
    assert total == pytest.approx(0.7, abs=1e-9)


def test_parametric_fit_dist_dict_must_cover_every_cause():
    with pytest.raises(ValueError, match="no distribution for the cause"):
        ParametricCompetingRisks.fit(
            [1, 2, 3, 4], ["a", "b", "a", "b"], dist={"a": Weibull}
        )


# -- labels -----------------------------------------------------------------


def test_ordered_labels_helper():
    assert ordered_labels([2, None, 1, 10]) == [1, 2, 10]
    assert ordered_labels(["b", 1, None, "a"]) == [1, "a", "b"]


X8 = [1, 2, 3, 4, 5, 6, 7, 8]
Z8 = [[0], [1], [0], [1], [1], [0], [0], [1]]
LABELS = {
    "mixed": [1, "b", None, 1, "b", 1, "b", 1],
    "tuple": [("a", 1), ("b", 2), None, ("a", 1)] * 2,
}


def _round_trip(model):
    return surpyval.from_dict(json.loads(json.dumps(model.to_dict())))


@pytest.mark.parametrize("name", LABELS)
def test_nonparametric_labels(name):
    e = LABELS[name]
    model = CompetingRisks.fit(X8, e)
    restored = _round_trip(model)
    assert list(restored.event_idx_map) == list(model.event_idx_map)
    for k in model.event_idx_map:
        np.testing.assert_allclose(restored.cif(X8, k), model.cif(X8, k))


@pytest.mark.parametrize("name", LABELS)
def test_parametric_labels(name):
    e = LABELS[name]
    model = ParametricCompetingRisks.fit(X8, e, dist=Exponential)
    restored = _round_trip(model)
    assert restored.causes == model.causes
    for k in model.causes:
        np.testing.assert_allclose(restored.cif(X8, k), model.cif(X8, k))


@pytest.mark.parametrize("name", LABELS)
@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_labels(name, how):
    e = LABELS[name]
    model = CompetingRisksProportionalHazards.fit(X8, Z8, e, how=how)
    restored = _round_trip(model)
    for k in model.event_idx_map:
        np.testing.assert_allclose(
            restored.cif(X8, [1], k), model.cif(X8, [1], k)
        )


def test_fine_gray_tuple_cause_round_trips():
    e = LABELS["tuple"]
    model = FineGray.fit(X8, Z8, e, cause=("a", 1))
    restored = _round_trip(model)
    assert restored.cause == ("a", 1)
    np.testing.assert_allclose(restored.cif(X8, [1]), model.cif(X8, [1]))


# -- clear errors and shapes ------------------------------------------------


def _crph_data(seed=1, n=120):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    t1 = rng.exponential(1 / (0.2 * np.exp(Z @ [0.5, -0.3])))
    t2 = rng.exponential(1 / (0.1 * np.exp(Z @ [-0.4, 0.2])))
    tc = rng.exponential(8, n)
    x = np.minimum.reduce([t1, t2, tc])
    e = np.where(tc < np.minimum(t1, t2), None, np.where(t1 < t2, 1, 2))
    return x, Z, e.astype(object)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
@pytest.mark.parametrize("event", [3, None])
def test_crph_cif_unknown_event_is_a_clear_error(how, event):
    x, Z, e = _crph_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, how=how)
    with pytest.raises(ValueError, match="one of the fitted causes"):
        model.cif([1.0], [0, 0], event)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_cif_pairs_covariate_rows_with_times(how):
    x, Z, e = _crph_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, how=how)
    paired = model.cif([1.0, 2.0], [[0, 0], [1, 1]], 1)
    single = [
        model.cif([1.0], [0, 0], 1)[0],
        model.cif([2.0], [1, 1], 1)[0],
    ]
    np.testing.assert_allclose(paired, single)
    with pytest.raises(ValueError, match="rows for 3 times"):
        model.cif([1.0, 2.0, 3.0], [[0, 0], [1, 1]], 1)


def test_fine_gray_cif_pairs_covariate_rows():
    x, Z, e = _crph_data()
    fg = FineGray.fit(x, Z, e, cause=1)
    paired = fg.cif([1.0, 2.0], [[0, 0], [1, 1]])
    np.testing.assert_allclose(
        paired, [fg.cif([1.0], [0, 0])[0], fg.cif([2.0], [1, 1])[0]]
    )


def test_wrong_number_of_covariate_rows_is_a_clear_error():
    x, Z, e = _crph_data()
    with pytest.raises(ValueError, match="row"):
        CompetingRisksProportionalHazards.fit(x, Z[:-1], e)
    with pytest.raises(ValueError, match="row"):
        FineGray.fit(x, Z[:-1], e, cause=1)


def test_nonparametric_sf_keeps_the_query_shape():
    model = CompetingRisks.fit([1, 2, 3], ["a", "b", "a"])
    query = np.array([[1, 2], [3, 4]])
    out = model.sf(query)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.ravel(), model.sf(query.ravel()))
    assert model.cif(query, "a").shape == (2, 2)


def test_empty_data_is_refused():
    with pytest.raises(ValueError):
        CompetingRisks.fit([], [])
