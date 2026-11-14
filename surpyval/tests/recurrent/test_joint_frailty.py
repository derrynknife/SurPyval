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


def _log_moment(D, H, alpha, theta, family):
    """``log E[u^D exp(-u^alpha H)]`` by scipy's adaptive quadrature over
    ``w = log u`` either side of the peak, found on a grid wide enough for
    the peak far out at a large ``H`` (``_peaked``'s ends at -60 and 8 are
    not)."""
    if family == "gamma":
        k = 1.0 / theta

        def log_g(w):
            return k * np.log(k) - gammaln(k) + k * w - k * np.exp(w)

    else:

        def log_g(w):
            return -(w**2) / (2 * theta) - 0.5 * np.log(2 * np.pi * theta)

    def h(w):
        with np.errstate(all="ignore"):
            return D * w + log_g(w) - np.exp(np.clip(alpha * w, -700, 700)) * H

    grid = np.linspace(-500.0, 30.0, 53001)
    values = h(grid)
    peak, top = grid[np.nanargmax(values)], np.nanmax(values)
    total = 0.0
    for a, b in ((-np.inf, peak), (peak, np.inf)):
        total += quad(
            lambda w: np.exp(h(w) - top),
            a,
            b,
            epsabs=0,
            epsrel=1e-13,
            limit=400,
        )[0]
    return top + np.log(total)


def _mcf_by_quadrature(model, x, frailty=None):
    """``E[N(x)]`` by scipy's adaptive quadrature over the log of age, with
    the inner expectation ``E[u exp(-u^alpha H)]`` by ``_log_moment`` (in
    log space): independent of ``model._mcf``."""
    rs, rk, ts, tk, theta, alpha = model.params[:6]

    def log_g(H):
        if frailty is not None:
            return np.log(frailty) - frailty**alpha * H
        return _log_moment(1.0, max(H, 1e-300), alpha, theta, model.family)

    def f(z):
        H = np.exp(tk * (z - np.log(ts)))
        return np.exp(np.log(rk) + rk * (z - np.log(rs)) + log_g(H))

    # where the integrand matters: up to the terminal scale and well past
    # it (it falls as a power of age, or faster, beyond)
    top = min(np.log(x), np.log(ts) + 40.0 / tk)
    cuts = [c for c in np.log(ts) + np.array([-3.0, 0.0, 3.0]) if c < top]
    total, lo = 0.0, -np.inf
    for hi in cuts + [top]:
        total += quad(f, lo, hi, epsabs=0, epsrel=1e-12, limit=400)[0]
        lo = hi
    return total


_MCF_MODELS = [
    ((10.0, 1.2), (20.0, 1.5), 0.5, 1.0, "gamma"),
    ((10.0, 1.2), (20.0, 1.5), 2.0, 0.3, "gamma"),
    ((10.0, 0.6), (20.0, 3.0), 0.5, 1.0, "gamma"),
    ((5.0, 2.0), (50.0, 0.8), 1.0, 1.0, "lognormal"),
    ((10.0, 1.2), (20.0, 1.5), 0.2, 2.5, "lognormal"),
]


@pytest.mark.parametrize("rec, term, theta, alpha, family", _MCF_MODELS)
def test_mcf_levels_off_past_the_terminal_event(
    rec, term, theta, alpha, family
):
    # Once every unit has reached the terminal event no more events
    # accrue: the expected count rises to a limit and stays there. Until
    # 0.28 a fixed rule over [0, x] lost the integrand far past the
    # terminal ages and the count went wrong there (3.682, 5.47 and 1.5e-4
    # in the conformance fit at 1e4, 1e6 and 1e8 times its terminal scale,
    # for a limit of 3.683).
    model = JointFrailty.from_params(rec, term, theta, alpha, family=family)
    x = term[0] * np.logspace(-4, 8, 400)
    for frailty in (None, 0.7):
        m = model.mcf(x, frailty=frailty)
        assert np.all(np.isfinite(m)) and np.all(np.diff(m) >= 0)
        assert m[-1] == m[-2]
        assert m[-1] == pytest.approx(
            _mcf_by_quadrature(model, 1e300, frailty), rel=1e-10
        )


@pytest.mark.parametrize("rec, term, theta, alpha, family", _MCF_MODELS)
def test_mcf_by_quadrature_in_the_ordinary_range(
    rec, term, theta, alpha, family
):
    # The ages of the data: the values of 0.27 to 1e-10 (it had them to
    # about 5e-15, but 7e-11 for the lognormal model with the steep
    # recurrent shape)
    model = JointFrailty.from_params(rec, term, theta, alpha, family=family)
    for t in np.array([0.05, 0.5, 1.0, 2.0]) * term[0]:
        assert float(model.mcf(t)) == pytest.approx(
            _mcf_by_quadrature(model, t), rel=1e-10
        )


def test_mcf_of_the_conformance_fit_far_out():
    from surpyval.tests.conformance.registry_fixtures import joint_frailty_data

    d = joint_frailty_data()
    model = JointFrailty.fit(
        x=d["x"], i=d["i"], c=d["c"], terminal=d["terminal"], Z=d["Z"]
    )
    scale = model.params[2]
    x = scale * np.logspace(-2, 8, 300)
    for z in ([0.0], [1.0]):
        m = model.mcf(x, z)
        assert np.all(np.diff(m) >= 0)
        # Within 1e-5 of the limit from 100 times the scale (the count
        # still creeps up, as a power of age, with a gamma frailty): it
        # was 5.47 and 1.5e-4 at 1e6 and 1e8 times the scale
        assert m[np.searchsorted(x, 100 * scale)] == pytest.approx(
            m[-1], rel=1e-5
        )
    assert float(model.mcf(1e8 * scale, [0.0])) == pytest.approx(
        3.682916, abs=1e-6
    )


def test_mcf_limit_by_simulation():
    # Followed until every unit has reached its terminal event: the mean
    # count of events is the mcf's limit.
    model = JointFrailty.from_params(
        (10.0, 1.2), (20.0, 1.5), 0.8, 1.5, beta=[0.4], gamma=[0.2]
    )
    sims = model.random(40000, end=1e7, Z=[1.0], random_state=11)
    assert np.all(sims["terminal"][sims["c"] == 1] == 1)
    mean = np.sum(sims["c"] == 0) / 40000
    assert float(model.mcf(1e7, [1.0])) == pytest.approx(mean, rel=0.03)
    assert float(model.mcf(1e300, [1.0])) == pytest.approx(
        float(model.mcf(1e7, [1.0])), rel=1e-5
    )


def test_terminal_functions_far_out():
    # sf, hf and df of the terminal event far past its ages, against the
    # frailty integrals by quadrature (in log space, so they do not
    # underflow)
    for family in ("gamma", "lognormal"):
        model = JointFrailty.from_params(
            (10.0, 1.2), (20.0, 1.5), 0.5, 1.3, family=family
        )
        for x in (200.0, 2e4, 2e8):
            H = (x / 20.0) ** 1.5
            h0 = 1.5 / 20.0 * (x / 20.0) ** 0.5
            log_s = _log_moment(0.0, H, 1.3, 0.5, family)
            hf = h0 * np.exp(_log_moment(1.3, H, 1.3, 0.5, family) - log_s)
            assert float(model.sf(x)) == pytest.approx(np.exp(log_s), rel=1e-9)
            assert float(model.Hf(x)) == pytest.approx(-log_s, rel=1e-10)
            assert float(model.hf(x)) == pytest.approx(hf, rel=1e-9)
            assert float(model.df(x)) == pytest.approx(
                hf * np.exp(log_s), rel=1e-9
            )


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


def test_well_identified_lr_bounds_are_unchanged(fitted):
    # The probe four Wald half-widths out finds the profile above the
    # critical value, and the searches give the bounds they gave in 0.27
    want = {
        ("theta", 0.05): [0.264095475012, 0.911073995334],
        ("theta", 0.01): [0.196886866437, 1.056309291055],
        ("alpha", 0.05): [0.732155138087, 3.92024961485],
        ("alpha", 0.01): [0.518119690297, 5.547221426304],
    }
    for (name, alpha_ci), bounds in want.items():
        got = fitted.param_cb(name, alpha_ci=alpha_ci, method="lr")
        np.testing.assert_allclose(got, bounds, rtol=1e-9)


@pytest.fixture(scope="module")
def flat_alpha():
    """The conformance fixture's fit (60 units), on which the profile of
    ``alpha`` is flat: its deviance rises to 2.96 and stays there, to
    1e-10, from ``alpha`` = 20 out to 1e10 (and from -2 out to -1e10)."""
    from surpyval.tests.conformance.registry_fixtures import joint_frailty_data

    d = joint_frailty_data()
    return JointFrailty.fit(
        x=d["x"], i=d["i"], c=d["c"], terminal=d["terminal"], Z=d["Z"]
    )


def test_flat_profile_gives_open_ends(flat_alpha):
    # Until 0.28 the search reported finite ends far out where the
    # likelihood could no longer be solved, which moved with the platform
    # and the interpreter (-1.6e31 and 9.3e6 at 95%; -4e12 or 1.6e31 at
    # 99% on x86-64 Linux, the 99% interval not containing the 95% one),
    # after a minute or more each. Where the critical value is above the
    # plateau both ends are open, with a warning saying so; where it is
    # below it (a one-sided 95% bound: 2.71) the profile crosses it.
    import time

    start = time.perf_counter()
    for alpha_ci in (0.01, 0.05):
        with pytest.warns(RuntimeWarning) as rec:
            got = flat_alpha.param_cb("alpha", alpha_ci=alpha_ci, method="lr")
        messages = [str(w.message) for w in rec]
        assert any("has no lower end" in m for m in messages)
        assert any("has no upper end" in m for m in messages)
        assert got[0] == -np.inf and got[1] == np.inf
    for bound, want in (("lower", -np.inf), ("upper", np.inf)):
        with pytest.warns(RuntimeWarning, match="has no"):
            got = flat_alpha.param_cb(
                "alpha", alpha_ci=0.01, bound=bound, method="lr"
            )
        assert got.tolist() == [want]
    # The runtime is bounded: the probes find the profile below the
    # critical value at a few Wald half-widths and the walks stop on its
    # plateau (about 3 s on an M1; one to twenty minutes per call before)
    assert time.perf_counter() - start < 60.0
    lower = flat_alpha.param_cb("alpha", bound="lower", method="lr")[0]
    upper = flat_alpha.param_cb("alpha", bound="upper", method="lr")[0]
    assert -1.0 < lower < flat_alpha.alpha < 9.0 < upper < 10.0


def test_lr_bound_on_the_edge_of_the_space_is_the_edge(flat_alpha):
    # theta's profile stays below the critical value down to theta = 0:
    # the lower bound is 0 itself (it was 2.2e-308, the end of the
    # search's coordinate); the upper bound is unchanged
    got = flat_alpha.param_cb("theta", alpha_ci=0.05, method="lr")
    assert got[0] == 0.0
    assert got[1] == pytest.approx(0.6489965790, rel=1e-8)


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
