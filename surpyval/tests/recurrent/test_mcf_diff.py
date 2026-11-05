"""The two-sample MCF comparison, ``mcf_diff``: its properties, its
refusals and its sums against a direct (items x times) computation. The
values against R's ``reda::mcfDiff`` are in
``tests/reference/test_mcf_diff.py``."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.recurrent import MCFDiffResult, NonParametricCounting, mcf_diff

A = dict(x=[2, 5, 9, 10, 4, 10], i=[1, 1, 1, 1, 2, 2], c=[0, 0, 0, 1, 0, 1])
B = dict(x=[6, 10, 3, 8, 10], i=[1, 1, 2, 2, 2], c=[0, 1, 0, 0, 1])


def _fit(d, **kwargs):
    return NonParametricCounting.fit(d["x"], i=d["i"], c=d["c"], **kwargs)


def _poisson_groups(rng, items, rate, end):
    """Each item a homogeneous Poisson process with rate ``rate``,
    observed over ``(0, end)``."""
    x, i, c = [], [], []
    for item in range(items):
        times = np.cumsum(rng.exponential(1 / rate, size=60))
        times = times[times < end]
        x += times.tolist() + [end]
        i += [item] * (times.size + 1)
        c += [0] * times.size + [1]
    return dict(x=x, i=i, c=c)


def _direct(model_a, model_b, weight, variance):
    """The score and its variance by the definitions, item by item and
    time by time, for windows and late entry too."""
    grid = np.union1d(model_a.x, model_b.x)
    parts = []
    for model in (model_a, model_b):
        data = model.data
        entry, exit_ = data.item_observation_windows()
        window_map = data.window_map or {}
        owner = [window_map[k][0] if k in window_map else k for k in data.ids]
        at_risk = (entry[:, None] < grid) & (grid <= exit_[:, None])
        counts = np.zeros(at_risk.shape)
        rows = data.item_rows()[1]
        for row in np.flatnonzero(np.asarray(data.c) == 0):
            counts[rows[row], np.searchsorted(grid, data.x[row])] += 1
        parts.append((at_risk, counts, np.array(owner)))
    r = [p[0].sum(axis=0) for p in parts]
    keep = (r[0] > 0) & (r[1] > 0)
    grid = grid[keep]
    r = [v[keep] for v in r]
    dL = [p[1][:, keep].sum(axis=0) / rk for p, rk in zip(parts, r)]
    w = r[0] * r[1] / (r[0] + r[1])
    if weight == "linear":
        w = w * (grid.max() - grid) / grid.max()
    score = np.sum(w * (dL[0] - dL[1]))
    if variance == "poisson":
        return score, np.sum(w**2 * (dL[0] / r[0] + dL[1] / r[1]))
    total = 0.0
    for (at_risk, counts, owner), rk, dLk in zip(parts, r, dL):
        res = (at_risk[:, keep] / rk * (counts[:, keep] - dLk)) @ w
        for key in np.unique(owner):
            total += np.sum(res[owner == key]) ** 2
    return score, total


def test_docstring_example_by_hand():
    result = mcf_diff(_fit(A), _fit(B))
    # MCF a: 0.5 at 2, 1 at 4, 1.5 at 5, 2 at 9; MCF b: 0.5 at 3, 1 at 6
    # and 8 (half each), so at 3 the difference is 0.5 - 0.5.
    assert_allclose(
        result.diff([1, 2, 3, 4, 5, 6, 8, 9, 10]),
        [0, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 0.5],
    )
    assert result.n_items == [2, 2]
    assert result.n_events == [4, 3]
    assert np.isnan(result.diff([-1, 11, np.nan])).all()


def test_swapping_the_groups_negates_the_difference_and_keeps_the_tests():
    a, b = _fit(A), _fit(B)
    ab, ba = mcf_diff(a, b), mcf_diff(b, a)
    assert_allclose(ab.difference, -ba.difference)
    assert_allclose(ab.var, ba.var)
    for w in ("constant", "linear"):
        assert_allclose(ab.tests[w]["score"], -ba.tests[w]["score"])
        assert_allclose(ab.tests[w]["statistic"], ba.tests[w]["statistic"])
    assert_allclose(ab.diff_cb([5]), -ba.diff_cb([5])[:, ::-1])


def test_identical_groups_differ_by_nothing():
    result = mcf_diff(_fit(A), _fit(A))
    assert_allclose(result.difference, 0, atol=1e-15)
    for w in ("constant", "linear"):
        assert result.tests[w]["score"] == pytest.approx(0, abs=1e-15)
        assert result.tests[w]["p_value"] == pytest.approx(1)


@pytest.mark.parametrize("variance", ["robust", "poisson"])
def test_the_tests_do_not_depend_on_the_time_scale(variance):
    a, b = _fit(A), _fit(B)
    scaled = [
        NonParametricCounting.fit(np.array(d["x"]) * 37.5, i=d["i"], c=d["c"])
        for d in (A, B)
    ]
    one = mcf_diff(a, b, test_variance=variance)
    two = mcf_diff(*scaled, test_variance=variance)
    for w in ("constant", "linear"):
        assert_allclose(two.tests[w]["statistic"], one.tests[w]["statistic"])
    assert_allclose(two.difference, one.difference)


@pytest.mark.parametrize("variance", ["robust", "poisson"])
@pytest.mark.parametrize("weight", ["constant", "linear"])
def test_sums_match_the_direct_computation_with_late_entry(weight, variance):
    rng = np.random.default_rng(20261105)
    a = _poisson_groups(rng, 9, 0.4, 20.0)
    b = _poisson_groups(rng, 7, 0.6, 16.0)
    # Late entry on some items of each group (no event at or before it).
    tl_a = np.where(np.arange(9) % 3 == 0, 0.5, 0.0)
    tl_b = np.where(np.arange(7) % 2 == 0, 0.25, 0.0)
    for d, tl in ((a, tl_a), (b, tl_b)):
        keep = np.array(d["x"]) > tl[np.array(d["i"])]
        for k in ("x", "i", "c"):
            d[k] = np.array(d[k])[keep]
    ma = _fit(a, tl=tl_a)
    mb = _fit(b, tl=tl_b)
    result = mcf_diff(ma, mb, test_variance=variance)
    score, var = _direct(ma, mb, weight, variance)
    assert_allclose(result.tests[weight]["score"], score, rtol=1e-12)
    assert_allclose(result.tests[weight]["score_var"], var, rtol=1e-12)


@pytest.mark.parametrize("weight", ["constant", "linear"])
def test_windowed_items_are_one_cluster(weight):
    a = NonParametricCounting.fit(
        [1, 3, 12, 15, 2, 7, 18],
        i=[1, 1, 1, 1, 2, 2, 2],
        windows={1: [(0, 5), (10, 20)], 2: [(0, 20)]},
    )
    b = _fit(
        dict(
            x=[4, 9, 14, 20, 6, 20], i=[1, 1, 1, 1, 2, 2], c=[0, 0, 0, 1, 0, 1]
        )
    )
    result = mcf_diff(a, b)
    score, var = _direct(a, b, weight, "robust")
    assert_allclose(result.tests[weight]["score"], score, rtol=1e-12)
    assert_allclose(result.tests[weight]["score_var"], var, rtol=1e-12)
    assert result.n_items == [2, 2]


def test_pointwise_bounds_are_wald_on_the_summed_variance():
    a, b = _fit(A), _fit(B)
    result = mcf_diff(a, b, alpha_ci=0.1)
    q = [2, 5, 9]

    def var_at(model):
        idx = np.searchsorted(model.x, q, side="right") - 1
        return np.where(idx < 0, 0.0, model.var[np.maximum(idx, 0)])

    # b has no event before 3: its variance at 2 is 0.
    var = var_at(a) + var_at(b)
    centre = a.mcf(q) - b.mcf(q)
    z = 1.6448536269514722
    assert_allclose(
        result.diff_cb(q),
        np.column_stack(
            [centre - z * np.sqrt(var), centre + z * np.sqrt(var)]
        ),
    )
    assert_allclose(
        result.diff_cb(q, "upper"), centre + 1.2815515655446004 * np.sqrt(var)
    )
    assert_allclose(
        result.diff_cb(q, "lower", alpha_ci=0.1),
        centre - 1.2815515655446004 * np.sqrt(var),
    )
    assert result.diff_cb(5.0).shape == (2,)
    assert result.diff(5.0).shape == ()


def test_round_trip_through_strict_json():
    result = mcf_diff(_fit(A), _fit(B), test_variance="poisson")
    text = json.dumps(result.to_dict(), allow_nan=False)
    again = MCFDiffResult.from_dict(json.loads(text))
    assert_allclose(again.difference, result.difference)
    assert_allclose(again.var, result.var)
    assert again.tests == result.tests
    assert again.test_variance == "poisson"
    assert repr(again) == repr(result)


def test_repr_reads_plainly():
    text = repr(mcf_diff(_fit(A), _fit(B)))
    assert "Two-sample MCF comparison" in text
    assert "Constant weight" in text and "Linear weight" in text


def test_plot_draws_the_difference_and_its_band():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = mcf_diff(_fit(A), _fit(B)).plot(ax=ax)
    assert out is ax
    assert ax.get_ylabel().startswith("Difference in MCF")
    plt.close(fig)


def test_refusals_name_the_argument():
    a = _fit(A)
    with pytest.raises(TypeError, match="model_b"):
        mcf_diff(a, [1, 2, 3])
    with pytest.raises(ValueError, match="model_b.*NonParametricCounting.fit"):
        mcf_diff(a, NonParametricCounting.from_xrd([1.0], [3], [1]))
    with pytest.raises(ValueError, match="test_variance"):
        mcf_diff(a, a, test_variance="naive")
    with pytest.raises(ValueError, match="alpha_ci"):
        mcf_diff(a, a, alpha_ci=1.5)
    late = NonParametricCounting.fit([40, 50], i=[1, 1], c=[0, 1], tl=30)
    with pytest.raises(ValueError, match="never have items"):
        mcf_diff(a, late)


def test_size_under_the_null_with_frailty():
    """Both groups from the same mixed Poisson process (an exponential
    frailty, coefficient of variation 1, so the Poisson variance is
    wrong), 20 items per group, 300 draws; the seed, the count and the
    bound were fixed before the result was seen.

    The robust tests come out somewhat liberal at this size: 26 and 25
    rejections of 300 (8.7% and 8.3%), at the edge of the bound. That is
    the method's, not the code's (the numbers match ``reda`` to rounding):
    the robust variance has no small-sample correction. A separate run of
    1000 draws gave 8.0% and 7.1% with 20 items per group and 5.0% and
    5.1% with 80. The Poisson test rejects 113 times (38%)."""
    rng = np.random.default_rng(235)
    reps = 300
    rejections = {"constant": 0, "linear": 0}
    poisson_rejections = 0
    for _ in range(reps):
        groups = []
        for items in (20, 20):
            x, i, c = [], [], []
            for item in range(items):
                rate = 0.5 * rng.gamma(1.0, 1.0)
                end = rng.uniform(5, 10)
                times = np.cumsum(rng.exponential(1, size=60)) / rate
                times = times[times < end]
                x += times.tolist() + [end]
                i += [item] * (times.size + 1)
                c += [0] * times.size + [1]
            groups.append(NonParametricCounting.fit(x, i=i, c=c))
        result = mcf_diff(*groups)
        for w in rejections:
            rejections[w] += result.tests[w]["p_value"] < 0.05
        poisson_rejections += (
            mcf_diff(*groups, test_variance="poisson").p_value < 0.05
        )
    for w, count in rejections.items():
        # 300 draws: a binomial(300, 0.05) count is in [6, 26] with
        # probability above 0.99.
        assert 6 <= count <= 26, (w, count)
    # With the frailty the Poisson variance is too small, so its test
    # rejects too often.
    assert poisson_rejections > max(rejections.values())
