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
