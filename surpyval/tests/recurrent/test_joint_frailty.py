"""The joint frailty model of recurrent and terminal events
(``surpyval.recurrent.JointFrailty``): its frailty integral, likelihood,
fits, predictions, simulation and refusals."""

import warnings

import numpy as np
import pandas as pd
import pytest
from autograd import grad
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist

from surpyval.recurrent import (
    JointFrailty,
    JointFrailtyModel,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.joint_frailty import log_frailty_integral


def _truth():
    return JointFrailty.from_params(
        (10.0, 1.2), (20.0, 1.5), 0.5, 1.0, beta=[0.5], gamma=[0.3]
    )


def _rows(seed=1, size=150, theta=0.5, alpha=1.0, time_scale="calendar"):
    truth = JointFrailty.from_params(
        (10.0, 1.2),
        (20.0, 1.5),
        theta,
        alpha,
        beta=[0.5],
        gamma=[0.3],
        time_scale=time_scale,
    )
    Z = (np.arange(size) % 2)[:, None].astype(float)
    return truth.random(size, end=15.0, Z=Z, random_state=seed)


@pytest.fixture(scope="module")
def fitted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return JointFrailty.fit(**_rows())


# ---------------------------------------------------------------------------
# The frailty integral
# ---------------------------------------------------------------------------
def _peaked(h, lo, hi):
    """``log int_lo^hi exp(h(w)) dw`` by scipy's adaptive quadrature on
    either side of the peak (scaled by it). The gamma frailty's left tail
    falls as ``exp((D + 1/theta) w)``, slowly for a large theta, hence the
    wide range."""
    with np.errstate(all="ignore"):
        grid = np.linspace(-60, 8, 68001)
        values = h(grid)
        peak = grid[np.nanargmax(values)]
        top = np.nanmax(values)
        total = 0.0
        for a, b in ((lo, peak), (peak, hi)):
            value, _ = quad(
                lambda w: np.exp(h(w) - top),
                a,
                b,
                epsabs=0,
                epsrel=1e-13,
                limit=4000,
            )
            total += value
    return top + np.log(total)


def _gamma_quad(D, A, B, alpha, theta):
    k = 1.0 / theta

    # in w = log u, scaled by the peak to avoid underflow
    def h(w):
        return (
            (D + k) * w
            - (A + k) * np.exp(w)
            - B * np.exp(alpha * w)
            + k * np.log(k)
            - gammaln(k)
        )

    return _peaked(h, -1000.0, 10.0)


def _lognormal_quad(D, A, B, alpha, theta):
    def h(w):
        return (
            D * w
            - A * np.exp(w)
            - B * np.exp(alpha * w)
            - w**2 / (2 * theta)
            - 0.5 * np.log(2 * np.pi * theta)
        )

    return _peaked(h, -60.0, 30.0)


CASES = [
    (0.0, 0.0, 1.0, 1.0),
    (3.0, 2.0, 0.5, 1.5),
    (10.0, 4.0, 1.0, -0.5),
    (1.0, 0.01, 3.0, 2.0),
    (40.0, 30.0, 2.0, 3.0),
    (0.0, 1e-4, 1e-3, 0.5),
]


@pytest.mark.parametrize("theta", [0.05, 0.5, 2.0, 5.0])
@pytest.mark.parametrize("D, A, B, alpha", CASES)
def test_integral_matches_adaptive_quadrature(D, A, B, alpha, theta):
    got = log_frailty_integral(D, A, B, alpha, theta, "gamma")
    assert got == pytest.approx(_gamma_quad(D, A, B, alpha, theta), abs=1e-10)
    got = log_frailty_integral(D, A, B, alpha, theta, "lognormal")
    want = _lognormal_quad(D, A, B, alpha, theta)
    assert got == pytest.approx(want, abs=1e-10)


@pytest.mark.parametrize("theta", [1e-9, 0.3, 4.0])
def test_gamma_integral_closed_form_at_alpha_one(theta):
    # E[u^D e^{-c u}] for a gamma frailty of mean 1, variance theta:
    # prod_{j < D} (1 + j theta) (1 + c theta)^-(1/theta + D), in a form
    # free of the cancellation of its gamma functions
    for D, c in [(0, 2.0), (3, 0.5), (12, 7.0)]:
        exact = np.sum(np.log1p(np.arange(D) * theta)) - (
            1.0 / theta + D
        ) * np.log1p(c * theta)
        got = log_frailty_integral(D, c / 2, c / 2, 1.0, theta)
        assert got == pytest.approx(exact, abs=1e-11, rel=1e-12)


def test_integral_tends_to_no_frailty():
    for family in ("gamma", "lognormal"):
        small = log_frailty_integral(4.0, 1.5, 0.7, 1.3, 1e-10, family)
        assert small == pytest.approx(-2.2, abs=1e-8)
        assert log_frailty_integral(4.0, 1.5, 0.7, 1.3, 0.0, family) == -2.2


def test_integral_gradient_matches_finite_differences():
    D = np.array([0.0, 2.0, 9.0])
    A = np.array([0.3, 1.0, 5.0])
    B = np.array([0.2, 0.4, 1.5])

    for family in ("gamma", "lognormal"):

        def f(v):
            return np.sum(
                log_frailty_integral(D, A * v[2], B, v[1], v[0], family)
            )

        v = np.array([0.7, 1.3, 1.1])
        g = grad(f)(v)
        h = 1e-6
        fd = [(f(v + h * e) - f(v - h * e)) / (2 * h) for e in np.eye(3)]
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
def test_likelihood_by_direct_integration(fitted):
    rows = _rows()
    params = fitted.params
    rs, rk, ts, tk, theta, alpha, b, g = params
    total = 0.0
    for unit in np.unique(rows["i"]):
        mine = rows["i"] == unit
        x, c, term = rows["x"][mine], rows["c"][mine], rows["terminal"][mine]
        z = rows["Z"][mine][0, 0]
        ev = x[c == 0]
        end = x[c == 1][0]
        delta = term[c == 1][0]
        er, et = np.exp(b * z), np.exp(g * z)

        def integrand(u):
            log_rec = np.sum(np.log(u * er * rk / rs * (ev / rs) ** (rk - 1)))
            log_rec -= u * er * (end / rs) ** rk
            haz = u**alpha * et * tk / ts * (end / ts) ** (tk - 1)
            log_term = delta * np.log(haz) - u**alpha * et * (end / ts) ** tk
            dens = gamma_dist.pdf(u, 1 / theta, scale=theta)
            return np.exp(log_rec + log_term) * dens

        value, _ = quad(integrand, 0, 60, epsabs=0, epsrel=1e-12, limit=500)
        total += np.log(value)
    assert fitted.log_likelihood == pytest.approx(total, abs=1e-7)


def test_left_truncation_conditions_on_survival_to_entry():
    # Units seen from age 2: the events before it are not seen, nor the
    # units whose observation ended by then.
    rows = _rows(seed=4, size=60)
    ends = {k: x for k, x, c in zip(rows["i"], rows["x"], rows["c"]) if c}
    keep = (rows["x"] > 2.0) & np.array([ends[k] > 2.0 for k in rows["i"]])
    rows = {k: v[keep] for k, v in rows.items()}
    ids = np.unique(rows["i"])
    entry = np.full(ids.size, 2.0)
    model = JointFrailty.fit(**rows, tl=entry)
    rs, rk, ts, tk, theta, alpha, b, g = model.params
    total = 0.0
    for k, unit in enumerate(ids):
        mine = rows["i"] == unit
        x, c, term = rows["x"][mine], rows["c"][mine], rows["terminal"][mine]
        z = rows["Z"][mine][0, 0]
        ev = x[c == 0]
        end = x[c == 1][0]
        delta = term[c == 1][0]
        e = entry[k]
        er, et = np.exp(b * z), np.exp(g * z)

        def integrand(u):
            log_rec = np.sum(np.log(u * er * rk / rs * (ev / rs) ** (rk - 1)))
            log_rec -= u * er * ((end / rs) ** rk - (e / rs) ** rk)
            haz = u**alpha * et * tk / ts * (end / ts) ** (tk - 1)
            # from age 0: the division by the chance of reaching the
            # entry age below leaves the window's hazard
            log_term = delta * np.log(haz) - u**alpha * et * (end / ts) ** tk
            return np.exp(log_rec + log_term) * gamma_dist.pdf(
                u, 1 / theta, scale=theta
            )

        def reached(u):
            return np.exp(
                -(u**alpha) * et * (e / ts) ** tk
            ) * gamma_dist.pdf(u, 1 / theta, scale=theta)

        num, _ = quad(integrand, 0, 60, epsabs=0, epsrel=1e-12, limit=500)
        den, _ = quad(reached, 0, 60, epsabs=0, epsrel=1e-12, limit=500)
        total += np.log(num) - np.log(den)
    assert model.log_likelihood == pytest.approx(total, abs=1e-7)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def test_fit_recovers_and_is_verified(fitted):
    assert fitted.maximum == "verified"
    assert fitted.parameter_names == [
        "recurrent_scale",
        "recurrent_shape",
        "terminal_scale",
        "terminal_shape",
        "theta",
        "alpha",
        "recurrent: coef_0",
        "terminal: coef_0",
    ]
    se = fitted.standard_errors()
    z = (fitted.params - _truth().params) / se
    assert np.all(np.abs(z) < 3.5)


def test_gap_time_scale_and_lognormal():
    rows = _rows(seed=2, time_scale="gap")
    model = JointFrailty.fit(**rows, time_scale="gap")
    assert model.maximum == "verified"
    assert model.time_scale == "gap"
    z = (model.params - _truth().params) / model.standard_errors()
    assert np.all(np.abs(z) < 3.5)
    with pytest.raises(ValueError, match="gap"):
        model.mcf(5.0, [1.0])
    ln = JointFrailty.fit(**_rows(seed=3), family="lognormal")
    assert ln.maximum == "verified" and ln.family == "lognormal"


def test_ignoring_the_terminal_event_biases_the_rate(fitted):
    rows = _rows()
    naive = ProportionalIntensityNHPP.fit(
        rows["x"], rows["Z"], i=rows["i"], c=rows["c"]
    )
    # The joint model's marginal recurrent rate at z = 1 against the
    # model that treats the terminal event as independent censoring: the
    # frail units leave early, so the naive fit sees too few events.
    joint = float(fitted.cif(15.0, [1.0]))
    assert float(naive.cif(15.0, [1.0])) < joint
    assert float(_truth().cif(15.0, [1.0])) == pytest.approx(joint, rel=0.25)


def test_units_and_covariate_origin(fitted):
    rows = _rows()
    scaled = JointFrailty.fit(**{**rows, "x": rows["x"] * 24.0})
    np.testing.assert_allclose(
        scaled.params[[0, 2]], fitted.params[[0, 2]] * 24.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.delete(scaled.params, [0, 2]),
        np.delete(fitted.params, [0, 2]),
        rtol=1e-5,
        atol=1e-7,
    )
    moved = JointFrailty.fit(**{**rows, "Z": rows["Z"] + 100.0})
    np.testing.assert_allclose(
        moved.params[4:], fitted.params[4:], rtol=1e-5, atol=1e-7
    )
    assert moved.log_likelihood == pytest.approx(
        fitted.log_likelihood, abs=1e-6
    )


def test_aliased_column():
    rows = _rows(seed=5, size=80)
    Z = np.c_[rows["Z"], rows["Z"] * 2]
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = JointFrailty.fit(**{**rows, "Z": Z})
    assert np.isnan(model.beta[1]) and np.isnan(model.gamma[1])
    np.testing.assert_array_equal(model.aliased, [1])
    assert np.isnan(model.standard_errors()[[7, 9]]).all()


def test_fit_from_df_names_coefficients(fitted):
    rows = _rows()
    df = pd.DataFrame(
        {
            "age": rows["x"],
            "unit": rows["i"],
            "c": rows["c"],
            "died": rows["terminal"],
            "load": rows["Z"][:, 0],
        }
    )
    model = JointFrailty.fit_from_df(
        df, "age", "unit", "c", "died", Z_cols="load"
    )
    assert model.parameter_names[-2:] == ["recurrent: load", "terminal: load"]
    np.testing.assert_allclose(model.params, fitted.params, rtol=1e-6)


def test_init_far_away_still_reaches_maximum(fitted):
    start = fitted.params.copy()
    start[0] *= 1e6
    model = JointFrailty.fit(**_rows(), init=start)
    np.testing.assert_allclose(model.params, fitted.params, rtol=1e-5)


def test_no_frailty_boundary():
    rows = _rows(seed=11, size=60, theta=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = JointFrailty.fit(**rows)
    if model.theta == 0.0:
        assert any("boundary" in str(w.message) for w in caught)
        assert np.isnan(model.alpha)
        tests = model.frailty_tests()
        assert tests.loc["theta = 0", "p"] == 1.0
    else:
        assert model.frailty_tests().loc["theta = 0", "p"] > 0.01


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def test_marginal_survival_closed_form():
    model = JointFrailty.from_params((10.0, 1.2), (20.0, 1.0), 0.5, 1.0)
    x = np.array([0.0, 5.0, 20.0, 60.0])
    H = x / 20.0
    np.testing.assert_allclose(model.sf(x), (1 + 0.5 * H) ** -2, rtol=1e-12)
    np.testing.assert_allclose(model.hf(x), 0.05 / (1 + 0.5 * H), rtol=1e-12)
    np.testing.assert_allclose(model.Hf(x), -np.log(model.sf(x)), rtol=1e-12)
    np.testing.assert_allclose(
        model.df(x), model.hf(x) * model.sf(x), rtol=1e-12
    )
    np.testing.assert_allclose(model.sf(x, frailty=2.0), np.exp(-2 * H))


def test_mcf_by_simulation():
    model = JointFrailty.from_params(
        (10.0, 1.2), (20.0, 1.5), 0.8, 1.5, beta=[0.4], gamma=[0.2]
    )
    sims = model.random(40000, end=12.0, Z=[1.0], random_state=7)
    events = sims["c"] == 0
    for t in (4.0, 8.0, 12.0):
        mean = np.sum(sims["x"][events] <= t) / 40000
        assert float(model.mcf(t, [1.0])) == pytest.approx(mean, rel=0.03)
    # The terminal event stops recurrences: the mcf is below the cif
    assert model.mcf(12.0, [1.0]) < model.cif(12.0, [1.0])
    died = sims["terminal"][sims["c"] == 1]
    assert float(model.ff(12.0, [1.0])) == pytest.approx(died.mean(), abs=0.01)


def test_conditional_predictions_and_frailties(fitted):
    assert len(fitted.frailties) == 150
    values = np.array(list(fitted.frailties.values()))
    # The posterior means average about 1 (the frailty's mean)
    assert np.mean(values) == pytest.approx(1.0, abs=0.15)
    u = values[0]
    assert float(fitted.cif(10.0, [1.0], frailty=u)) == pytest.approx(
        u * float(fitted.cif(10.0, [1.0]))
    )


def test_bounds_contain_estimates(fitted):
    for name in ("cif", "iif", "mcf", "sf"):
        cb = getattr(fitted, name + "_cb")([2.0, 8.0], [1.0])
        est = getattr(fitted, name)([2.0, 8.0], [1.0])
        assert np.all(cb[:, 0] < est) and np.all(est < cb[:, 1])
    for name in ("theta", "alpha"):
        wald = fitted.param_cb(name)
        lr = fitted.param_cb(name, method="lr")
        est = getattr(fitted, name)
        assert wald[0] < est < wald[1]
        assert lr[0] < est < lr[1]


def test_frailty_tests(fitted):
    tests = fitted.frailty_tests()
    assert list(tests.index) == ["theta = 0", "alpha = 0"]
    assert tests.loc["theta = 0", "p"] < 0.01
    assert (tests["statistic"] >= 0).all()


# ---------------------------------------------------------------------------
# Simulation and serialisation
# ---------------------------------------------------------------------------
def test_random_seeding():
    a = _truth().random(10, end=15.0, Z=[1.0], random_state=3)
    b = _truth().random(10, end=15.0, Z=[1.0], random_state=3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.random.seed(1)
    c = _truth().random(10, end=15.0, Z=[1.0])
    np.random.seed(1)
    d = _truth().random(10, end=15.0, Z=[1.0])
    np.testing.assert_array_equal(c["x"], d["x"])


def test_round_trip(fitted):
    again = JointFrailtyModel.from_dict(fitted.to_dict())
    x = np.array([1.0, 5.0, 12.0])
    for name in ("cif", "iif", "mcf", "sf", "hf"):
        np.testing.assert_array_equal(
            getattr(again, name)(x, [1.0]), getattr(fitted, name)(x, [1.0])
        )
    np.testing.assert_allclose(
        again.param_cb("theta"), fitted.param_cb("theta"), rtol=1e-12
    )
    assert again.frailties == fitted.frailties
    with pytest.raises(ValueError, match="data"):
        again.param_cb("theta", method="lr")
    with pytest.raises(ValueError, match="data"):
        again.frailty_tests()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _small():
    return {
        "x": [2.0, 5.0, 9.0, 3.0, 8.0],
        "i": [1, 1, 1, 2, 2],
        "c": [0, 0, 1, 0, 1],
        "terminal": [0, 0, 1, 0, 0],
    }


@pytest.mark.parametrize(
    "change, match",
    [
        ({"terminal": None}, "'terminal' is needed"),
        ({"i": None}, "'i' is needed"),
        ({"c": [0, 0, 0, 0, 1]}, "exactly one row with c=1"),
        ({"c": [0, 2, 1, 0, 1]}, "Left- and interval-censored"),
        ({"terminal": [1, 0, 1, 0, 0]}, "recurrent-event rows"),
        ({"x": [2.0, 50.0, 9.0, 3.0, 8.0]}, "after the end"),
        ({"x": [2.0, np.nan, 9.0, 3.0, 8.0]}, "missing values"),
        ({"terminal": [0, 0, 0, 0, 0]}, "No unit had the terminal event"),
        ({"Z": [0.0, 1.0, 0.0, 1.0, 1.0]}, "same on every row"),
        ({"family": "weibull"}, "'family'"),
        ({"time_scale": "age"}, "'time_scale'"),
        ({"init": [1.0, 2.0]}, "'init' must have 6"),
    ],
)
def test_refusals(change, match):
    with pytest.raises(ValueError, match=match):
        JointFrailty.fit(**{**_small(), **change})


def test_gap_refuses_entry_and_zero_gaps():
    with pytest.raises(ValueError, match="time_scale='gap'"):
        JointFrailty.fit(**_small(), tl=1.0, time_scale="gap")
    tied = {**_small(), "x": [2.0, 2.0, 9.0, 3.0, 8.0]}
    with pytest.raises(ValueError, match="gap of 0"):
        JointFrailty.fit(**tied, time_scale="gap")
