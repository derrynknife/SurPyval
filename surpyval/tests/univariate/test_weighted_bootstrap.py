"""The case bootstrap of fits with sampling weights ``w``
(``method="bootstrap"``); the conformance suite checks every weighted
fitter's bounds for finiteness and the weight scale, and the calibration
study their coverage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data
from surpyval.utils.weights import WeightedRefusal

BOOT = {"method": "bootstrap", "n_boot": 30, "random_state": 4}


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def test_resamples_the_units_with_their_weights(data):
    """The bound is the refits' percentiles over resamples of the units,
    each with its weight: drawn here by hand with the same generator, the
    refits' alphas span the bound."""
    x, c, w = data["x"], data["c"], data["w"]
    model = sp.Weibull.fit(x, c, w=w)
    lo, hi = model.param_cb("alpha", **BOOT)
    rng = np.random.default_rng(4)
    alphas = []
    for _ in range(30):
        k = rng.integers(0, x.size, size=x.size)
        alphas.append(quietly(sp.Weibull.fit, x[k], c[k], w=w[k]).params[0])
    alphas = np.sort(alphas)
    assert alphas[0] <= lo < hi <= alphas[-1]
    # Not the unweighted bootstrap's
    plain = sp.Weibull.fit(x, c).param_cb("alpha", **BOOT)
    assert not np.allclose(plain, [lo, hi])


def test_same_seed_same_bounds_and_kept(data):
    model = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    first = model.cb([2.0, 5.0], **BOOT)
    assert_allclose(model.cb([2.0, 5.0], **BOOT), first, rtol=0, atol=0)
    assert np.all(first[:, 0] <= model.sf([2.0, 5.0]))
    assert np.all(model.sf([2.0, 5.0]) <= first[:, 1])


def test_interval_censored_weighted_data_are_resampled(data):
    """Nothing is simulated, so interval censoring (which the parametric
    bootstrap refuses) is resampled as it is."""
    x, c, w = data["x"], data["c"], data["w"]
    xi = np.column_stack([np.where(c == 0, 0.8 * x, x), x])
    ci = np.where(c == 0, 2, 1)
    model = sp.Weibull.fit(xi, ci, w=w)
    b = quietly(model.param_cb, "beta", **BOOT)
    assert np.isfinite(b).all() and b[0] < model.params[1] < b[1]
    with pytest.raises(ValueError, match="interval-censored"):
        sp.Weibull.fit(xi, ci).param_cb("beta", **BOOT)


def test_regression_case_bootstrap(data):
    x, c, w, Z = data["x"], data["c"], data["w"], data["Z"]
    model = quietly(sp.WeibullAFT.fit, x, Z, c=c, w=w)
    lo, hi = quietly(model.param_cb, model.parameter_names[-1], **BOOT)
    assert lo < model.params[-1] < hi
    with pytest.raises(WeightedRefusal, match="likelihood-ratio"):
        model.param_cb(model.parameter_names[-1], method="lr")


def test_likelihood_ratio_still_refused(data):
    model = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    with pytest.raises(WeightedRefusal, match="Wald"):
        model.cb([2.0], method="lr")


def test_limited_failure_with_weights_refuses_the_bootstrap(data):
    # Without weights a limited-failure model has bootstrap (BCa) bounds;
    # with weights its jackknife is not built, so the bootstrap refuses
    # and points to the Wald bounds.
    x, c, w = data["x"], data["c"], data["w"]
    model = quietly(sp.Weibull.fit, x, c, w=w, lfp=True)
    with pytest.raises(WeightedRefusal, match="Wald"):
        model.param_cb("lfp_p", **BOOT)
    with pytest.raises(WeightedRefusal, match="limited-failure"):
        model.cb([2.0], **BOOT)


def _grouped(data):
    """``data`` recorded to whole time units and grouped: rows with
    counts ``n``, each row one weight."""
    x, c, w = np.ceil(data["x"]), data["c"], data["w"]
    w = np.round(w, 1)
    rows, counts = np.unique(
        np.column_stack([x, c, w]), axis=0, return_counts=True
    )
    return rows[:, 0], rows[:, 1].astype(int), counts, rows[:, 2]


def test_grouped_rows_are_resampled_with_multinomial_counts(data):
    """Drawing the units with replacement draws each row a multinomial
    number of times; the refits get the rows with those counts (as many
    rows as the data, not every unit), summing to the units."""
    from surpyval.univariate.parametric._bootstrap import _CaseDesign

    x, c, n, w = _grouped(data)
    model = sp.Weibull.fit(x, c, n, w=w)
    design = _CaseDesign(model)
    assert design.grouped and design.x.size == np.asarray(model.data["n"]).size
    rng = np.random.default_rng(0)
    for _ in range(200):
        out = design.resample(rng)
        assert out["n"].sum() == n.sum() and np.all(out["n"] > 0)
        assert out["x"].size <= design.x.size
        assert np.all(np.isin(out["w"], design.w))
    # each row is drawn n_i / N of the time: its mean count is its n
    assert_allclose(design.share * design.units, model.data["n"])
    counts = np.array(
        [rng.multinomial(design.units, design.share) for _ in range(4000)]
    )
    assert_allclose(counts.mean(axis=0), model.data["n"], rtol=0.1, atol=0.5)


def test_ungrouped_weighted_data_are_still_drawn_unit_by_unit(data):
    from surpyval.univariate.parametric._bootstrap import _CaseDesign

    model = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    assert np.all(np.asarray(model.data["n"]) == 1)
    assert not _CaseDesign(model).grouped


def test_grouped_and_expanded_bounds_agree_in_distribution(data):
    """The grouped data's bootstrap is the expanded data's in
    distribution, not draw for draw: over seeds the bounds' spread
    matches (the regressions keep expanded rows as given, so the
    expanded fit draws unit by unit)."""
    x, c, n, w = _grouped(data)
    Z = (x > np.median(x)).astype(float)[:, None]
    grouped = quietly(sp.WeibullAFT.fit, x, Z, c=c, n=n, w=w)
    expanded = quietly(
        sp.WeibullAFT.fit,
        np.repeat(x, n),
        np.repeat(Z, n, axis=0),
        c=np.repeat(c, n),
        w=np.repeat(w, n),
    )
    assert_allclose(grouped.params, expanded.params, rtol=1e-6)
    name = grouped.parameter_names[-1]
    kw = {"method": "bootstrap", "n_boot": 40}
    g = np.array(
        [
            quietly(grouped.param_cb, name, random_state=s, **kw)
            for s in range(8)
        ]
    )
    e = np.array(
        [
            quietly(expanded.param_cb, name, random_state=s, **kw)
            for s in range(8)
        ]
    )
    assert not np.allclose(g, e)
    # the means over 8 seeds agree within three standard errors
    se = np.sqrt((g.var(axis=0, ddof=1) + e.var(axis=0, ddof=1)) / 8)
    assert np.all(np.abs(g.mean(axis=0) - e.mean(axis=0)) < 3 * se)
