"""JointModel: the fit, its likelihood, entry points and refusals (#891).

The likelihood is checked against an independent computation: each
unit's integral over its random effects on a dense grid, with each
cumulative hazard by a 64-point Gauss-Legendre rule in time, for every
kind of censoring and for truncation. The parameters are checked on a
simulated fleet; their coverage is a calibration study
(``tests/calibration/test_coverage_joint.py``).
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval.joint import JointModel
from surpyval.joint import _likelihood as lk
from surpyval.joint._data import joint_data as gather
from surpyval.joint._fit import prepare
from surpyval.joint._params import internal_from_natural
from surpyval.tests._helpers import no_warnings, quietly
from surpyval.tests.conformance.registry_fixtures import joint_data

PUMPS = {
    "alpha": 900.0,
    "beta": 1.4,
    "association": 0.9,
    "path_intercept": 2.8,
    "path_slope": 0.0045,
    "sd_intercept": 0.5,
    "sd_slope": 0.002,
    "corr": 0.0,
    "sigma": 0.6,
}
CENTRE = 3.5


@pytest.fixture(scope="module")
def truth():
    return JointModel.from_params(PUMPS, marker_centre=CENTRE)


@pytest.fixture(scope="module")
def fleet(truth):
    return truth.simulate(300, 30.0, 730.0, random_state=891)


@pytest.fixture(scope="module")
def model(fleet):
    return no_warnings(JointModel.fit_from_df, fleet, c_col="c")


def _arrays(df, Z_cols=None):
    units = df.groupby("i").first()
    out = {
        "x": units["x"].to_numpy(),
        "c": units["c"].to_numpy(),
        "i": units.index.to_numpy(),
        "marker_x": df["marker_x"].to_numpy(),
        "marker_y": df["marker_y"].to_numpy(),
        "marker_i": df["i"].to_numpy(),
    }
    if Z_cols:
        out["Z"] = units[Z_cols].to_numpy()
    return out


# ---------------------------------------------------------------------------
# The likelihood, against an independent computation
# ---------------------------------------------------------------------------
def _brute_force_neg_ll(params, data, centre):
    """Each unit's likelihood integrated over a dense grid of its random
    effects (to 8 standard deviations), with each cumulative hazard a
    64-point Gauss-Legendre sum in time; a truncated unit divided by the
    probability of its window, the same integral of ``P(tl < T <= tr |
    b)``."""
    alpha, beta, *rest = params
    p = data["Z"].shape[1] if "Z" in data else 0
    gamma = np.asarray(rest[:p])
    assoc, b0m, b1m, sd0, sd1, corr, sigma = rest[p:]
    g = np.linspace(-8.0, 8.0, 321)
    u0, u1 = np.meshgrid(g, g, indexing="ij")
    b0 = sd0 * u0
    b1 = sd1 * (corr * u0 + np.sqrt(1 - corr**2) * u1)
    dens = norm.pdf(u0) * norm.pdf(u1)
    du = (g[1] - g[0]) ** 2
    z, w = np.polynomial.legendre.leggauss(64)
    v, wv = 0.5 * (z + 1.0), 0.5 * w

    def H(lo, hi, zrow):
        # From 0, in s = t v^2, which smooths the s^(beta - 1) at 0.
        assert lo == 0.0
        s = hi * v**2
        m = (b0m + b0)[..., None] + (b1m + b1)[..., None] * s
        h = (
            (beta / alpha)
            * (s / alpha) ** (beta - 1)
            * np.exp(zrow @ gamma + assoc * (m - centre))
        )
        return np.sum(wv * 2.0 * hi * v * h, axis=-1)

    total = 0.0
    x, c = np.asarray(data["x"], float), np.asarray(data["c"])
    n = len(c)
    tl = np.broadcast_to(np.asarray(data.get("tl", 0.0), float), (n,))
    tr = np.broadcast_to(np.asarray(data.get("tr", np.inf), float), (n,))
    Zm = np.asarray(data["Z"], float) if p else np.zeros((n, 0))
    for u in range(n):
        uid = np.asarray(data["i"])[u]
        rows = np.asarray(data["marker_i"]) == uid
        ty = np.asarray(data["marker_x"])[rows]
        yy = np.asarray(data["marker_y"])[rows]
        m = (b0m + b0)[..., None] + (b1m + b1)[..., None] * ty
        f = np.prod(norm.pdf(yy, m, sigma), axis=-1)
        lo, hi = (x[u], x[u]) if np.ndim(x[u]) == 0 else x[u]
        if c[u] == 0:
            m_t = b0m + b0 + (b1m + b1) * hi
            haz = (
                (beta / alpha)
                * (hi / alpha) ** (beta - 1)
                * np.exp(Zm[u] @ gamma + assoc * (m_t - centre))
            )
            f = f * haz * np.exp(-H(0.0, hi, Zm[u]))
        elif c[u] == 1:
            f = f * np.exp(-H(0.0, hi, Zm[u]))
        elif c[u] == -1:
            f = f * -np.expm1(-H(0.0, hi, Zm[u]))
        else:
            f = f * (np.exp(-H(0.0, lo, Zm[u])) - np.exp(-H(0.0, hi, Zm[u])))
        total += np.log(np.sum(f * dens) * du)
        if tl[u] > 0 or np.isfinite(tr[u]):
            window = np.exp(-H(0.0, tl[u], Zm[u])) if tl[u] > 0 else 1.0
            if np.isfinite(tr[u]):
                window = window - np.exp(-H(0.0, tr[u], Zm[u]))
            total -= np.log(np.sum(window * dens) * du)
    return -total


def _small_data(kinds):
    """Five units with readings, one per kind of event in ``kinds``."""
    t = [np.array([0.0, 0.4, 0.8]), np.array([0.0, 0.5]), np.array([0.2])]
    t += [np.array([0.0, 0.3, 0.6, 0.9]), np.array([])]
    y = [
        np.array([1.0, 1.5, 2.1]),
        np.array([0.6, 0.7]),
        np.array([1.4]),
        np.array([0.8, 1.1, 1.0, 1.6]),
        np.array([]),
    ]
    events = {
        0: (1.0, 0),
        1: (1.2, 1),
        -1: (1.1, -1),
        2: ([0.9, 1.3], 2),
    }
    x, c = [], []
    for k in kinds:
        when, flag = events[k]
        x.append([when, when] if np.ndim(when) == 0 else when)
        c.append(flag)
    return {
        "x": np.array(x, dtype=float),
        "c": np.array(c),
        "Z": np.array([[0.0], [1.0], [0.5], [1.0], [0.0]]),
        "i": np.arange(5),
        "marker_x": np.concatenate(t),
        "marker_y": np.concatenate(y),
        "marker_i": np.repeat(np.arange(5), [a.size for a in t]),
    }


_PARAMS = [1.3, 1.6, 0.4, 0.8, 0.9, 0.7, 0.45, 0.6, 0.3, 0.2]


def _neg_ll(data, params, n_quad=11):
    d = gather(
        data["x"],
        data["c"],
        data["Z"],
        data["i"],
        data.get("tl"),
        data.get("tr"),
        None,
        None,
        data["marker_x"],
        data["marker_y"],
        data["marker_i"],
        None,
    )
    prepared = prepare(d, np.arange(1), np.arange(0), n_quad, 30)
    theta = internal_from_natural(
        params, prepared.prob.layout, prepared.scaling
    )
    from surpyval.joint._fit import natural_neg_ll

    value = lk.adaptive_neg_ll(theta, prepared.prob)
    return natural_neg_ll(value, prepared), prepared.scaling.y_c


@pytest.mark.parametrize(
    "kinds",
    [
        (0, 0, 0, 1, 0),
        (1, 0, 1, 1, 1),
        (-1, 0, -1, 1, 0),
        (2, 0, 2, 1, 2),
    ],
    ids=["observed", "right", "left", "interval"],
)
def test_the_likelihood_is_the_integral_over_the_random_effects(kinds):
    data = _small_data(kinds)
    value, centre = _neg_ll(data, _PARAMS)
    expected = _brute_force_neg_ll(_PARAMS, data, centre)
    assert value == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_a_truncated_unit_is_divided_by_the_chance_of_its_window():
    data = _small_data((0, 1, 2, 0, 1))
    data["tl"] = np.array([0.3, 0.0, 0.5, 0.0, 0.2])
    data["tr"] = np.array([np.inf, np.inf, 2.0, 1.5, np.inf])
    value, centre = _neg_ll(data, _PARAMS)
    expected = _brute_force_neg_ll(_PARAMS, data, centre)
    assert value == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_the_fit_reports_the_likelihood_at_its_estimate(model):
    assert model.neg_ll() == pytest.approx(
        model.neg_ll_of(model._estimates()), rel=1e-10
    )
    # and that is the one the dense-grid computation gives
    data = joint_data()
    fixture = quietly(JointModel.fit, **data)
    expected = _brute_force_neg_ll(fixture.params, data, fixture.marker_centre)
    assert fixture.neg_ll() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("kind", [0, 1, -1, 2])
def test_the_derivatives_in_b_are_those_of_the_integrand(kind):
    # The modes are found with analytic derivatives of the log integrand
    # in b; they must be its derivatives.
    data = _small_data((kind, 0, kind, kind, kind))
    d = gather(
        data["x"],
        data["c"],
        data["Z"],
        data["i"],
        None,
        None,
        None,
        None,
        data["marker_x"],
        data["marker_y"],
        data["marker_i"],
        None,
    )
    prepared = prepare(d, np.arange(1), np.arange(0), 7, 15)
    prob = prepared.prob
    theta = internal_from_natural(_PARAMS, prob.layout, prepared.scaling)
    blk = prob.units
    b = np.array([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.1], [0.0, -0.2]] * 2)[
        : blk.n
    ]
    parts = lk.unpack(theta, prob.layout)
    sig2 = float(np.exp(2 * parts.lsig))
    L = np.asarray(lk.chol_factor(np.asarray(parts.chol)))
    D_inv = np.linalg.inv(L @ L.T)
    gauss = np.einsum("njq,njr->nqr", blk.BR, blk.BR) / sig2 + D_inv
    grad, info = lk._gradient_and_information(
        theta, b, blk, prob.layout, prob.gl, gauss, D_inv
    )
    h = 1e-5
    for q in range(lk.Q):
        e = np.zeros(lk.Q)
        e[q] = h
        up = lk._value(theta, b + e, blk, prob.layout, prob.gl)
        down = lk._value(theta, b - e, blk, prob.layout, prob.gl)
        np.testing.assert_allclose(
            grad[:, q], (up - down) / (2 * h), rtol=1e-6
        )
        g_up, _ = lk._gradient_and_information(
            theta, b + e, blk, prob.layout, prob.gl, gauss, D_inv
        )
        g_down, _ = lk._gradient_and_information(
            theta, b - e, blk, prob.layout, prob.gl, gauss, D_inv
        )
        np.testing.assert_allclose(
            -info[:, :, q], (g_up - g_down) / (2 * h), rtol=1e-5, atol=1e-8
        )


def test_each_mode_is_a_maximum_of_its_integrand(model):
    state = model._fit_state
    prob = state.prepared.prob
    modes = state.nodes.modes
    value = lk._value(state.theta, modes.b, prob.units, prob.layout, prob.gl)
    for shift in ([1e-3, 0.0], [0.0, 1e-3], [-1e-3, 1e-3]):
        moved = lk._value(
            state.theta, modes.b + shift, prob.units, prob.layout, prob.gl
        )
        assert np.all(moved <= value + 1e-12)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def test_the_fit_recovers_the_fleet(model):
    assert model.maximum == "verified"
    se = dict(zip(model.covariance_names, model.standard_errors()))
    for name, value in PUMPS.items():
        if name == "alpha":
            continue  # its meaning moves with the marker centre
        assert abs(getattr(model, name) - value) < 3.5 * se[name], name
    assert model.marker_centre == pytest.approx(
        float(model.data.readings[0][1].mean()), rel=1
    )


def test_alpha_is_the_baseline_at_the_marker_centre(model):
    # The same hazard at another centre: alpha moves, nothing else.
    shifted = dict(zip(model.parameter_names, model.params))
    shifted["alpha"] = model.alpha * np.exp(
        model.association * (model.marker_centre - CENTRE) / model.beta
    )
    other = JointModel.from_params(shifted, marker_centre=CENTRE)
    x = [100.0, 400.0, 700.0]
    np.testing.assert_allclose(other.sf(x), model.sf(x), rtol=1e-10)


def test_the_time_unit_does_not_matter(fleet, model):
    hours = fleet.copy()
    hours[["x", "marker_x"]] *= 24.0
    in_hours = quietly(JointModel.fit_from_df, hours, c_col="c")
    scale = {"alpha": 24.0, "path_slope": 1 / 24.0, "sd_slope": 1 / 24.0}
    for name, value in zip(model.parameter_names, model.params):
        expected = value * scale.get(name, 1.0)
        assert getattr(in_hours, name) == pytest.approx(expected, rel=1e-4)
    x = np.array([100.0, 400.0])
    np.testing.assert_allclose(in_hours.sf(24 * x), model.sf(x), rtol=1e-4)


def test_the_marker_unit_does_not_matter(fleet, model):
    scaled = fleet.copy()
    scaled["marker_y"] = 1000.0 * scaled["marker_y"]
    in_um = quietly(JointModel.fit_from_df, scaled, c_col="c")
    assert in_um.association == pytest.approx(
        model.association / 1000.0, rel=1e-4
    )
    assert in_um.sigma == pytest.approx(1000.0 * model.sigma, rel=1e-4)
    np.testing.assert_allclose(
        in_um.sf([100.0, 400.0]), model.sf([100.0, 400.0]), rtol=1e-4
    )


def test_entry_points_agree(fleet, model):
    arrays = quietly(JointModel.fit, **_arrays(fleet))
    np.testing.assert_allclose(arrays.params, model.params, rtol=1e-8)
    items = []
    for i, g in fleet.groupby("i"):
        failed = g["c"].iat[0] == 0
        items.append(
            sp.ItemHistory(
                i,
                observed=g["x"].iat[0] if failed else None,
                right_censored=None if failed else g["x"].iat[0],
                markers=dict(zip(g["marker_x"], g["marker_y"])),
            )
        )
    from_items = quietly(JointModel.fit_from_items, items)
    np.testing.assert_allclose(from_items.params, model.params, rtol=1e-6)
    data = sp.EventHistoryData.from_items(items)
    from_data = quietly(JointModel.fit_from_items, data)
    np.testing.assert_allclose(from_data.params, model.params, rtol=1e-6)


def test_a_unit_without_readings_is_one_empty_row(fleet):
    units = fleet.groupby("i").first().reset_index()
    late = units.iloc[:20].copy()
    late["i"] = late["i"] + 10_000
    late[["marker_x", "marker_y"]] = np.nan
    df = pd.concat([fleet, late], ignore_index=True)
    model = quietly(JointModel.fit_from_df, df, c_col="c")
    assert model.data.n == 320
    assert model.data.n_readings == len(fleet)


def test_censoring_and_truncation_fit(fleet):
    # The same fleet with failures found at monthly inspections (interval
    # censored) and pumps entering at 30 days (left truncated): the fit
    # runs to a verified maximum near the exact one.
    d = _arrays(fleet)
    failed = d["c"] == 0
    lower = np.floor(d["x"] / 30.0) * 30.0
    upper = lower + 30.0
    x = np.column_stack([d["x"], d["x"]])
    x[failed] = np.column_stack([lower, upper])[failed]
    d2 = dict(d, x=x, c=np.where(failed, 2, d["c"]))
    exact = quietly(JointModel.fit, **d)
    inspected = no_warnings(JointModel.fit, **d2)
    assert inspected.maximum == "verified"
    se = dict(zip(exact.covariance_names, exact.standard_errors()))
    assert abs(inspected.association - exact.association) < se["association"]
    entered = d["x"] > 30.0
    late = {
        k: (v[entered] if k in ("x", "c", "i") else v) for k, v in d.items()
    }
    keep = np.isin(d["marker_i"], d["i"][entered])
    for k in ("marker_x", "marker_y", "marker_i"):
        late[k] = d[k][keep]
    truncated = no_warnings(JointModel.fit, **late, tl=30.0)
    assert truncated.maximum == "verified"


def test_covariates_in_both_parts(truth):
    model = JointModel.from_params(
        {
            **PUMPS,
            "heavy": 0.5,
            "path_intercept:heavy": 0.4,
            "path_slope:heavy": 0.002,
        },
        marker_centre=CENTRE,
    )
    heavy = np.tile([0.0, 1.0], 200)
    df = model.simulate(
        400, 30.0, 730.0, Z=heavy, path_Z=heavy, random_state=3
    )
    fit = no_warnings(
        JointModel.fit_from_df,
        df,
        c_col="c",
        Z_cols="heavy",
        path_Z_cols="path_heavy",
    )
    assert fit.parameter_names[2] == "heavy"
    assert "path_slope:path_heavy" in fit.parameter_names
    se = dict(zip(fit.covariance_names, fit.standard_errors()))
    assert abs(fit.heavy - 0.5) < 3.5 * se["heavy"]
    for name, value in (
        ("path_intercept:path_heavy", 0.4),
        ("path_slope:path_heavy", 0.002),
    ):
        k = fit.parameter_names.index(name)
        assert abs(fit.params[k] - value) < 3.5 * se[name], name


def test_aliased_columns_are_nan_and_named(fleet):
    d = _arrays(fleet)
    n = d["x"].size
    with pytest.warns(UserWarning, match="column\\(s\\) 1 of Z"):
        model = JointModel.fit(**d, Z=np.column_stack([np.arange(n) % 2] * 2))
    assert np.isnan(model.params[3]) and np.isfinite(model.params[2])
    assert model.aliased.tolist() == [1]
    with pytest.warns(UserWarning, match="of path_Z cannot be estimated"):
        model = JointModel.fit(**d, path_Z=np.ones(n))
    assert model.aliased_path.tolist() == [0]
    assert np.isnan(
        model.params[model.parameter_names.index("path_slope:w_0")]
    )
    # predictions take an aliased effect as 0
    assert np.isfinite(model.sf(100.0, path_Z=[5.0]))


# ---------------------------------------------------------------------------
# What the fit refuses, and what it says
# ---------------------------------------------------------------------------
def _first(key, value):
    def change(d):
        arr = np.array(d[key], dtype=object if key.endswith("i") else float)
        arr[0] = value
        return {**d, key: arr}

    return change


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: {**d, "c": np.ones_like(d["c"])}, "right censored"),
        (lambda d: {**d, "marker_x": None}, "marker_x, marker_y and"),
        (_first("x", np.nan), "missing \\(nan\\) times"),
        (_first("x", -1.0), "must be positive"),
        (_first("c", 3), "c must be 0"),
        (lambda d: {**d, "i": np.r_[1, d["i"][1:]]}, "more than once"),
        (_first("marker_i", "stranger"), "not a unit of i"),
        (_first("marker_y", np.nan), "missing or infinite"),
        (_first("marker_x", 1e9), "after their failure or censoring"),
        (
            lambda d: {**d, "Z": np.r_[[[np.nan]], d["Z"][1:]]},
            "missing or infinite values",
        ),
        (lambda d: {**d, "survival": "weibull_aft"}, "not available yet"),
        (lambda d: {**d, "link": "slope"}, "not available yet"),
        (lambda d: {**d, "path": "spline"}, "not available yet"),
        (
            lambda d: {**d, "marker_y": np.ones_like(d["marker_y"])},
            "the same value",
        ),
        (lambda d: {**d, "tl": 1e9}, "after its left truncation"),
        (lambda d: {**d, "Z": d["Z"][:5]}, "one row per unit"),
    ],
)
def test_refusals_say_what_to_do(change, match):
    with pytest.raises(ValueError, match=match):
        JointModel.fit(**change(joint_data()))


def test_units_with_one_reading_each_are_refused():
    data = joint_data()
    first = np.unique(data["marker_i"], return_index=True)[1]
    for key in ("marker_x", "marker_y", "marker_i"):
        data[key] = data[key][first]
    with pytest.raises(ValueError, match="at least two units with readings"):
        JointModel.fit(**data)


def test_a_starved_fit_says_so():
    with pytest.warns(UserWarning, match="did not reach a verified maximum"):
        model = JointModel.fit(**joint_data(), max_iter=1)
    assert model.maximum == "unverified"


def test_a_covariate_separating_the_failures_has_no_finite_maximum():
    data = joint_data()
    data["Z"] = (data["c"] == 1).astype(float)[:, None]
    with pytest.warns(UserWarning, match="No finite maximum: .*coef_0"):
        model = JointModel.fit(**data)
    assert model.maximum == "no finite maximum"


def test_the_long_frame_is_checked():
    df = pd.DataFrame(
        {
            "i": [1, 1, 2],
            "x": [5.0, 6.0, 7.0],
            "marker_x": [0.0, 1.0, 0.0],
            "marker_y": [1.0, 2.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="vary within a unit"):
        JointModel.fit_from_df(df)


# ---------------------------------------------------------------------------
# A new unit's failure time
# ---------------------------------------------------------------------------
def test_a_new_unit_is_the_average_over_the_random_effects(truth):
    # Monte Carlo over the random effects, each unit's survival by
    # quadrature in time.
    rng = np.random.default_rng(5)
    b = rng.standard_normal((20000, 2)) * [0.5, 0.002]
    z, w = np.polynomial.legendre.leggauss(64)
    t = np.array([250.0, 500.0, 1000.0])
    out = []
    for x in t:
        s = 0.5 * x * (z + 1.0)
        m = (2.8 + b[:, :1]) + (0.0045 + b[:, 1:]) * s
        h = 1.4 / 900 * (s / 900) ** 0.4 * np.exp(0.9 * (m - CENTRE))
        out.append(np.exp(-0.5 * x * h @ w).mean())
    np.testing.assert_allclose(truth.sf(t), out, atol=4e-3)


def test_the_functions_agree(truth):
    x = np.array([10.0, 100.0, 400.0, 900.0])
    np.testing.assert_allclose(truth.sf(x) + truth.ff(x), 1.0)
    np.testing.assert_allclose(truth.Hf(x), -np.log(truth.sf(x)))
    np.testing.assert_allclose(truth.df(x), truth.hf(x) * truth.sf(x))
    h = 1e-3 * x
    numeric = (truth.ff(x + h) - truth.ff(x - h)) / (2 * h)
    np.testing.assert_allclose(truth.df(x), numeric, rtol=1e-5)
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(truth.ff(truth.qf(p)), p, rtol=1e-8)
    assert truth.sf(-1.0) == 1.0 and truth.sf(np.inf) == 0.0
    assert np.isnan(truth.sf(np.nan))


def test_draws_follow_the_survival_function(truth):
    draws = truth.random(4000, random_state=2)
    for x in (250.0, 500.0, 1000.0):
        assert np.mean(draws > x) == pytest.approx(truth.sf(x), abs=0.025)
    np.random.seed(4)
    first = truth.random(5)
    np.random.seed(4)
    assert np.array_equal(first, truth.random(5))


def test_simulate_is_reproducible_and_reads_up_to_the_end(truth):
    a = truth.simulate(20, 50.0, 400.0, random_state=1)
    b = truth.simulate(20, 50.0, 400.0, random_state=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["marker_x"] < a["x"]).all() | (a["marker_x"] == 0).all()
    assert set(a["c"]) <= {0, 1}
    assert (a.loc[a["c"] == 1, "x"] == 400.0).all()


# ---------------------------------------------------------------------------
# Inference, reporting and saving
# ---------------------------------------------------------------------------
def test_bounds(model):
    lo, hi = model.param_cb("association")
    assert lo < model.association < hi
    lo_c, hi_c = model.param_cb("corr")
    assert -1 < lo_c < model.corr < hi_c < 1
    band = model.cb([100.0, 400.0])
    sf = model.sf([100.0, 400.0])
    assert np.all((band[:, 0] < sf) & (sf < band[:, 1]))
    np.testing.assert_allclose(
        model.cb([100.0, 400.0], on="ff")[:, ::-1], 1 - band, rtol=1e-12
    )
    with pytest.raises(ValueError, match="name"):
        model.param_cb("nope")


def test_summary_and_repr(model):
    table = model.summary()
    assert table.loc[("association", "association"), "coef"] == pytest.approx(
        model.association
    )
    assert table.loc[("association", "association"), "p"] < 1e-6
    text = repr(model)
    assert "Weibull proportional hazards" in text
    assert "300 (" in text and "maximum: verified" in text


def test_saving_and_loading_keeps_every_prediction(model):
    blob = model.to_dict()
    for restored in (sp.from_dict(blob), pickle.loads(pickle.dumps(model))):
        np.testing.assert_allclose(
            restored.sf([100.0, 400.0]), model.sf([100.0, 400.0]), rtol=0
        )
        np.testing.assert_allclose(
            restored.cb([100.0]), model.cb([100.0]), rtol=1e-12
        )
        assert restored.aic() == model.aic()
        a = restored.predict([0, 30, 60], [3.0, 3.4, 3.9]).sf(200.0)
        b = model.predict([0, 30, 60], [3.0, 3.4, 3.9]).sf(200.0)
        assert a == pytest.approx(b, rel=1e-12)
    with pytest.raises(ValueError, match="a joint model"):
        JointModel.from_dict({**blob, "model": "WienerProcessModel"})
    restored = JointModel.from_dict(blob)
    with pytest.raises(ValueError, match="keeps no data"):
        restored.neg_ll_of(restored._estimates())


def test_from_params_is_checked():
    with pytest.raises(ValueError, match="missing"):
        JointModel.from_params({"alpha": 1.0})
    with pytest.raises(ValueError, match="positive"):
        JointModel.from_params({**PUMPS, "sigma": 0.0})
    with pytest.raises(ValueError, match="strictly between -1 and 1"):
        JointModel.from_params({**PUMPS, "corr": 1.0})
    with pytest.raises(ValueError, match="path_slope:w"):
        JointModel.from_params({**PUMPS, "path_intercept:w": 1.0})
    model = JointModel.from_params(PUMPS, unit="day")
    assert model.unit == "day"
    with pytest.raises(ValueError, match="built from its parameters"):
        model.covariance()
    with pytest.raises(ValueError, match="no likelihood"):
        model.neg_ll()


def test_prediction_covariates_are_checked(truth):
    with pytest.raises(ValueError, match="fitted without Z"):
        truth.sf(1.0, Z=[1.0])
    with_z = JointModel.from_params({**PUMPS, "load": 0.3})
    with pytest.raises(ValueError, match="depends on Z"):
        with_z.sf(1.0)
    assert np.isnan(with_z.sf([1.0, 2.0], Z=[[np.nan], [1.0]])[0])


# ---------------------------------------------------------------------------
# A reference: the primary biliary cirrhosis data
# ---------------------------------------------------------------------------
def test_pbc_bilirubin_is_a_regression_check():
    # Log serum bilirubin with a random intercept and slope in years, and
    # death (transplantation censored) with a Weibull baseline: the
    # values this fit reached when it was written, to 1e-3, so that a
    # change to the likelihood or its quadrature that moves them is seen.
    # (No published fit of exactly this model was at hand to check
    # against; the quadrature and likelihood are checked independently
    # above.)
    from surpyval.datasets import load_pbc2

    df = load_pbc2()
    df["log_bilirubin"] = np.log(df["serBilir"])
    df["censored"] = 1 - df["status2"]
    model = no_warnings(
        JointModel.fit_from_df,
        df,
        x_col="years",
        c_col="censored",
        i_col="id",
        marker_x_col="year",
        marker_y_col="log_bilirubin",
    )
    assert model.maximum == "verified"
    expected = {
        "beta": 1.02173,
        "association": 1.2398,
        "path_intercept": 0.492899,
        "path_slope": 0.184956,
        "sd_intercept": 1.0025,
        "sd_slope": 0.180733,
        "corr": 0.424958,
        "sigma": 0.347134,
    }
    for name, value in expected.items():
        assert getattr(model, name) == pytest.approx(value, rel=1e-3), name
    assert model.neg_ll() == pytest.approx(1919.24, rel=1e-5)
