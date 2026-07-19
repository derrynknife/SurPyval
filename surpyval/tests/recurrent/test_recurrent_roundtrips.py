"""Round-trip and closed-form coverage for the recurrent intensity models.

The likelihood / AIC / BIC / standard-error behaviour of the parametric
counting processes is covered elsewhere; this module fills the thinner spots:

* the closed-form intensity identities of every NHPP baseline (``HPP``,
  ``CrowAMSAA``, ``Duane``, ``CoxLewis``) -- ``iif`` is the derivative of
  ``cif``, ``exp(log_iif) == iif``, ``inv_cif`` inverts ``cif``, and the
  fitted model's ``mcf`` equals its ``cif``;
* the nonparametric ``NonParametricCounting`` MCF and its Greenwood
  confidence bounds against a hand-computed risk set;
* the covariate scaling and inverse round-trips of the proportional-intensity
  ``HPP`` / ``NHPP`` regression models.
"""

import numpy as np
import pytest

from surpyval.recurrent import (
    CoxLewis,
    CrowAMSAA,
    Duane,
    HPP,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)

# One representative parameter set per closed-form NHPP baseline.
NHPP_CASES = {
    "CrowAMSAA": (CrowAMSAA, [5.0, 1.3]),
    "Duane": (Duane, [1.2, 0.8]),
    "CoxLewis": (CoxLewis, [0.2, 0.1]),
}


# --- closed-form intensity identities (dist level) --------------------------


@pytest.mark.parametrize("name", list(NHPP_CASES))
def test_iif_is_derivative_of_cif(name):
    dist, params = NHPP_CASES[name]
    xs = np.array([0.5, 1.0, 2.5, 4.0])
    h = 1e-6
    numeric = (dist.cif(xs + h, *params) - dist.cif(xs - h, *params)) / (2 * h)
    assert np.allclose(numeric, dist.iif(xs, *params), rtol=1e-4)


@pytest.mark.parametrize("name", list(NHPP_CASES))
def test_log_iif_is_log_of_iif(name):
    dist, params = NHPP_CASES[name]
    xs = np.array([0.5, 1.0, 2.5, 4.0])
    assert np.allclose(
        np.exp(dist.log_iif(xs, *params)), dist.iif(xs, *params)
    )


@pytest.mark.parametrize("name", list(NHPP_CASES))
def test_inv_cif_inverts_cif(name):
    dist, params = NHPP_CASES[name]
    xs = np.array([0.5, 1.0, 2.5, 4.0])
    # x -> cif -> x
    assert np.allclose(dist.inv_cif(dist.cif(xs, *params), *params), xs)
    # N -> inv_cif -> N
    ns = np.array([0.25, 1.0, 3.0])
    assert np.allclose(dist.cif(dist.inv_cif(ns, *params), *params), ns)


@pytest.mark.parametrize("name", list(NHPP_CASES))
def test_cif_matches_known_closed_form(name):
    dist, params = NHPP_CASES[name]
    x = 3.0
    if name == "CrowAMSAA":
        alpha, beta = params
        expected = (x / alpha) ** beta
    elif name == "Duane":
        a, b = params
        expected = b * x**a
    else:  # CoxLewis
        alpha, beta = params
        expected = np.exp(alpha) / beta * (np.exp(beta * x) - 1.0)
    assert np.isclose(float(dist.cif(x, *params)), expected)


# --- fitted-model round-trips (model level) ---------------------------------


@pytest.mark.parametrize("name", list(NHPP_CASES))
def test_from_params_model_mcf_equals_cif_and_matches_dist(name):
    dist, params = NHPP_CASES[name]
    model = dist.from_params(params)
    xs = np.array([1.0, 2.0, 3.0])
    # the counting-process MCF is a closed-form alias of the CIF
    assert np.allclose(model.mcf(xs), model.cif(xs))
    # and the model delegates to the underlying closed form
    assert np.allclose(model.cif(xs), dist.cif(xs, *params))
    assert np.allclose(model.iif(xs), dist.iif(xs, *params))
    # inverse round-trip through the model
    assert np.allclose(model.inv_cif(model.cif(xs)), xs)


def test_hpp_intensity_is_constant_and_cif_linear():
    # A homogeneous Poisson process fitted to one long sequence: the intensity
    # is a constant rate and the CIF is exactly rate * t.
    rng = np.random.default_rng(0)
    gaps = rng.exponential(scale=1.0 / 2.0, size=400)
    x = np.cumsum(gaps)
    model = HPP.fit(x)
    rate = model.params[0]
    assert np.isclose(rate, 2.0, rtol=0.15)
    ts = np.array([1.0, 3.0, 7.0])
    assert np.allclose(model.iif(ts), rate)
    assert np.allclose(model.cif(ts), rate * ts)
    assert np.allclose(model.mcf(ts), model.cif(ts))
    assert np.allclose(model.inv_cif(model.cif(ts)), ts)


# --- nonparametric MCF and confidence bounds --------------------------------


def _hand_mcf_model():
    # item 1: events at 1, 2 then right-censored at 4
    # item 2: event at 3 then right-censored at 5
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    i = np.array([1, 1, 1, 2, 2])
    c = np.array([0, 0, 1, 0, 1])
    return NonParametricCounting.fit(x, i, c)


def test_mcf_matches_hand_computed_risk_set():
    model = _hand_mcf_model()
    # risk set: both items at risk through t=4, only item 2 after; the
    # Nelson-Aalen MCF increments by d / r at each event time.
    assert np.array_equal(model.x, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(model.r, [2, 2, 2, 2, 1])
    assert np.allclose(model.mcf_hat, [0.5, 1.0, 1.5, 1.5, 1.5])
    # step interpolation reads the last increment at or below x
    assert np.isclose(model.mcf(1.5)[0], 0.5)
    assert np.isclose(model.mcf(3.0)[0], 1.5)
    assert np.isclose(model.mcf(4.9)[0], 1.5)
    # linear interpolation rises between event times
    assert 0.5 < model.mcf(2.5, interp="linear")[0] < 1.5


def test_mcf_is_non_decreasing():
    model = _hand_mcf_model()
    assert np.all(np.diff(model.mcf_hat) >= -1e-12)


def test_mcf_confidence_bounds_bracket_estimate():
    model = _hand_mcf_model()
    x = np.array([2.0, 3.0])
    two = model.mcf_cb(x, bound="two-sided")
    assert two.shape == (2, 2)
    est = model.mcf(x)
    lo = two.min(axis=1)
    hi = two.max(axis=1)
    # the estimate lies within its own two-sided band, which stays positive
    assert np.all(lo <= est) and np.all(est <= hi)
    assert np.all(two > 0)  # exponential Greenwood bounds cannot go negative


def test_mcf_one_sided_bounds_are_ordered():
    model = _hand_mcf_model()
    x = np.array([2.0, 3.0])
    est = model.mcf(x)
    upper = model.mcf_cb(x, bound="upper")
    lower = model.mcf_cb(x, bound="lower")
    assert upper.shape == x.shape and lower.shape == x.shape
    assert np.all(lower <= est) and np.all(est <= upper)


def test_mcf_cb_rejects_bad_options():
    model = _hand_mcf_model()
    with pytest.raises(ValueError, match="bound_type"):
        model.mcf_cb(np.array([2.0]), bound_type="nonsense")
    with pytest.raises(ValueError, match="'dist' must be 'z'"):
        model.mcf_cb(np.array([2.0]), dist="t")


# --- proportional-intensity regression round-trips --------------------------

Z = np.array([[0.1], [0.1], [0.5], [0.5], [0.9], [0.9]])
I = np.array([1, 1, 2, 2, 3, 3])
X = np.array([5.0, 8.0, 6.0, 10.0, 7.0, 9.0])
C = np.array([0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize(
    "fitter", [ProportionalIntensityHPP, ProportionalIntensityNHPP]
)
def test_proportional_intensity_cif_is_baseline_times_covariate(fitter):
    model = fitter.fit(X, Z, i=I, c=C)
    Zt = np.array([0.4])
    xs = np.array([3.0, 6.0, 9.0])
    baseline = model.dist.cif(xs, *model.params)
    scale = np.exp(Zt @ model.coeffs)
    assert np.allclose(model.cif(xs, Zt), baseline * scale)
    assert np.allclose(
        model.iif(xs, Zt), model.dist.iif(xs, *model.params) * scale
    )


@pytest.mark.parametrize(
    "fitter", [ProportionalIntensityHPP, ProportionalIntensityNHPP]
)
def test_proportional_intensity_inv_cif_round_trip(fitter):
    model = fitter.fit(X, Z, i=I, c=C)
    Zt = np.array([0.4])
    xs = np.array([3.0, 6.0, 9.0])
    assert np.allclose(model.inv_cif(model.cif(xs, Zt), Zt), xs)


@pytest.mark.parametrize(
    "fitter", [ProportionalIntensityHPP, ProportionalIntensityNHPP]
)
def test_proportional_intensity_covariate_monotonicity(fitter):
    # A larger covariate value scales the whole intensity up or down by a
    # constant factor, so the CIF ratio between two covariate levels is the
    # same at every time and equals exp((z2 - z1) * beta).
    model = fitter.fit(X, Z, i=I, c=C)
    xs = np.array([2.0, 5.0, 8.0])
    z1, z2 = np.array([0.2]), np.array([0.7])
    ratio = model.cif(xs, z2) / model.cif(xs, z1)
    assert np.allclose(ratio, np.exp((z2 - z1) @ model.coeffs))
