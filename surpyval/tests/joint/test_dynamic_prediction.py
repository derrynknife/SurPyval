"""DynamicPrediction: one unit's survival given its readings (#891).

The prediction is checked against Monte Carlo truth: random effects drawn
from their distribution, weighted by the likelihood of the unit's readings
and its survival to now (importance sampling), each unit's survival by
quadrature in time. Its band's coverage is a calibration study
(``tests/calibration/test_coverage_joint.py``).
"""

import warnings

import numpy as np
import pytest

from surpyval import ItemHistory
from surpyval.joint import DynamicPrediction, JointModel

PARAMS = {
    "alpha": 1000.0,
    "beta": 1.5,
    "load": 0.4,
    "association": 0.8,
    "path_intercept": 3.0,
    "path_slope": 0.004,
    "sd_intercept": 0.5,
    "sd_slope": 0.002,
    "corr": 0.3,
    "sigma": 0.5,
}
CENTRE = 3.0
T_READ = np.array([0.0, 100.0, 200.0])
Y_READ = np.array([3.1, 3.6, 4.4])


@pytest.fixture(scope="module")
def model():
    return JointModel.from_params(PARAMS, marker_centre=CENTRE)


def _monte_carlo(t, y, load, age, horizons, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.array([PARAMS["sd_intercept"], PARAMS["sd_slope"]])
    rho = PARAMS["corr"]
    u = rng.standard_normal((n, 2))
    b0 = sd[0] * u[:, 0]
    b1 = sd[1] * (rho * u[:, 0] + np.sqrt(1 - rho**2) * u[:, 1])
    a0 = PARAMS["path_intercept"] + b0
    a1 = PARAMS["path_slope"] + b1
    z, w = np.polynomial.legendre.leggauss(80)

    def H(lo, hi):
        s = lo + 0.5 * (hi - lo) * (z + 1.0)
        m = a0[:, None] + a1[:, None] * s
        k, alpha = PARAMS["beta"], PARAMS["alpha"]
        h = (
            (k / alpha)
            * (s / alpha) ** (k - 1)
            * np.exp(
                PARAMS["load"] * load + PARAMS["association"] * (m - CENTRE)
            )
        )
        return 0.5 * (hi - lo) * h @ w

    resid = y - (a0[:, None] + a1[:, None] * t)
    log_w = -0.5 * np.sum(resid**2, 1) / PARAMS["sigma"] ** 2 - H(0.0, age)
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    return np.array([weights @ np.exp(-H(age, x)) for x in horizons])


@pytest.mark.parametrize("age", [200.0, 350.0])
def test_the_prediction_is_the_monte_carlo_truth(model, age):
    horizons = age + np.array([100.0, 250.0, 400.0, 700.0])
    pred = model.predict(T_READ, Y_READ, Z=[1.0], age=age)
    truth = _monte_carlo(T_READ, Y_READ, 1.0, age, horizons)
    np.testing.assert_allclose(pred.sf(horizons), truth, atol=2e-3)


def test_without_readings_it_is_a_new_unit_still_running(model):
    pred = model.predict([], [], Z=[0.0], age=300.0)
    x = np.array([400.0, 700.0, 1000.0])
    expected = model.sf(x, Z=[0.0]) / model.sf(300.0, Z=[0.0])
    # (two quadratures: adaptive over the posterior, Gauss-Hermite over
    # the distribution)
    np.testing.assert_allclose(pred.sf(x), expected, rtol=1e-5)
    fresh = model.predict([], [], Z=[0.0], age=0.0)
    np.testing.assert_allclose(fresh.sf(x), model.sf(x, Z=[0.0]), rtol=1e-5)


def test_its_functions_agree(model):
    pred = model.predict(T_READ, Y_READ, Z=[1.0])
    assert isinstance(pred, DynamicPrediction)
    x = np.array([250.0, 400.0, 700.0])
    assert pred.sf(150.0) == 1.0 and pred.ff(200.0) == 0.0
    assert pred.hf(100.0) == 0.0 and pred.df(150.0) == 0.0
    np.testing.assert_allclose(pred.sf(x) + pred.ff(x), 1.0)
    np.testing.assert_allclose(pred.Hf(x), -np.log(pred.sf(x)))
    np.testing.assert_allclose(pred.df(x), pred.hf(x) * pred.sf(x))
    h = 1e-3 * x
    numeric = (pred.ff(x + h) - pred.ff(x - h)) / (2 * h)
    np.testing.assert_allclose(pred.df(x), numeric, rtol=1e-4)
    p = np.array([0.05, 0.5, 0.95])
    np.testing.assert_allclose(pred.ff(pred.qf(p)), p, rtol=1e-8)
    assert pred.qf(0.0) == pred.age
    np.testing.assert_allclose(pred.remaining_life(p), pred.qf(p) - 200.0)
    assert np.isnan(pred.sf(np.nan))
    assert pred.sf([[300.0, 400.0]]).shape == (1, 2)
    with pytest.warns(UserWarning, match="outside"):
        assert np.isnan(pred.qf(1.5))


def test_a_unit_whose_marker_rises_faster_is_at_more_risk(model):
    calm = model.predict(T_READ, [3.1, 3.2, 3.3], Z=[1.0])
    worn = model.predict(T_READ, Y_READ, Z=[1.0])
    assert worn.sf(500.0) < calm.sf(500.0)
    assert worn.remaining_life() < calm.remaining_life()
    # A new reading updates it.
    worse = model.predict(np.r_[T_READ, 250.0], np.r_[Y_READ, 5.2], Z=[1.0])
    assert worse.sf(500.0) < worn.sf(500.0)


def test_the_marker(model):
    pred = model.predict(T_READ, Y_READ, Z=[1.0])
    smooth = pred.marker(T_READ)
    assert np.all(np.abs(smooth - Y_READ) < 0.5)
    band = pred.marker_interval([0.0, 400.0])
    assert np.all(band[:, 0] < pred.marker([0.0, 400.0]))
    assert band[1, 1] - band[1, 0] > band[0, 1] - band[0, 0]


def test_draws_follow_the_prediction(model):
    pred = model.predict(T_READ, Y_READ, Z=[1.0])
    draws = pred.random(3000, random_state=7)
    assert np.all(draws > pred.age)
    for x in (300.0, 450.0, 700.0):
        assert np.mean(draws > x) == pytest.approx(pred.sf(x), abs=0.03)
    np.testing.assert_array_equal(
        pred.random(4, random_state=1), pred.random(4, random_state=1)
    )
    np.random.seed(3)
    first = pred.random(4)
    np.random.seed(3)
    np.testing.assert_array_equal(first, pred.random(4))


def test_an_item_history_is_a_unit(model):
    item = ItemHistory(
        "pump",
        right_censored=200.0,
        covariates=[1.0],
        markers=dict(zip(T_READ, Y_READ)),
    )
    a = model.predict(item)
    b = model.predict(T_READ, Y_READ, Z=[1.0])
    assert a.sf(500.0) == pytest.approx(b.sf(500.0), rel=1e-12)


def test_bounds(fitted):
    pred = fitted.predict([0.0, 30.0, 60.0, 90.0], [3.3, 3.8, 4.1, 4.6])
    x = np.array([60.0, 200.0, 400.0])
    band = pred.cb(x)
    sf = pred.sf(x)
    assert np.all((band[:, 0] <= sf) & (sf <= band[:, 1]))
    assert np.all((band >= 0) & (band <= 1))
    np.testing.assert_allclose(pred.cb(x, on="ff")[:, ::-1], 1 - band)
    np.testing.assert_allclose(
        pred.cb(x, bound="lower", alpha_ci=0.025), band[:, 0]
    )
    wide = pred.cb(x, alpha_ci=0.01)
    assert np.all(wide[:, 0] <= band[:, 0]) and np.all(
        wide[:, 1] >= band[:, 1]
    )
    assert np.all(pred.cb([30.0]) == 1.0)


@pytest.fixture(scope="module")
def fitted():
    truth = JointModel.from_params(
        {k: v for k, v in PARAMS.items() if k != "load"},
        marker_centre=CENTRE,
    )
    df = truth.simulate(150, 30.0, 730.0, random_state=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return JointModel.fit_from_df(df, c_col="c")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"age": 100.0}, "before the last reading"),
        ({"marker_y": [1.0, np.nan, 2.0]}, "must be finite"),
        ({"marker_y": [1.0, 2.0]}, "same length"),
        ({"Z": [[1.0], [0.0]]}, "one unit"),
        ({"Z": None}, "depends on Z"),
        ({"Z": [np.nan]}, "must not be missing"),
        ({"age": -1.0}, "at least 0"),
    ],
)
def test_refusals(model, kwargs, match):
    args = {"marker_x": T_READ, "marker_y": Y_READ, "Z": [1.0], **kwargs}
    with pytest.raises(ValueError, match=match):
        model.predict(**args)


def test_repr(model):
    text = repr(model.predict(T_READ, Y_READ, Z=[1.0]))
    assert "Age now" in text and "Median life left" in text
