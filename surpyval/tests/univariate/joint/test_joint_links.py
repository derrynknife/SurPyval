"""The joint model's links: the hazard following the marker's current
value, its slope, the area under it, or more than one.

The checks: each built-in path's slope against central differences, and
the area under a path against its closed form; the cumulative hazard and
the hazard along a known path against their closed forms; the likelihood
of every kind of item, for each link, against an independent dense-grid
integral over the random effects; recovery of fleets simulated from each
link (``_fleet.simulate_links``, seed 5, chosen before the fits were
seen); the prediction against a brute-force integral; and the names,
the printout, the refusals and the saved form. The comparison with R's
JM on the PBC trial is in ``tests/reference/test_joint_links.py``.
"""

import functools
import pickle
import warnings
from typing import Any

import numpy as np
import pytest

import surpyval
from surpyval import JointModel, JointPH, Weibull, WeibullJointPH
from surpyval.degradation import LinkedPathModel, LloydLipowPath
from surpyval.degradation.path_models import PATH_MODELS, PathModel
from surpyval.tests.univariate.joint._fleet import simulate_links
from surpyval.tests.univariate.joint.test_joint_extensions import KINDS
from surpyval.univariate.joint._likelihood import (
    JointArrays,
    Layout,
    LinkPath,
    PathFunction,
    log_cumhaz,
    log_event,
)
from surpyval.univariate.joint.joint_ph import search_space

X = np.array([0.3, 1.0, 2.5, 7.0])
THETA = [1.3, 0.4, 0.2]

# -- the paths' slopes and areas ------------------------------------------


@pytest.mark.parametrize("key", sorted(PATH_MODELS))
def test_a_built_in_paths_slope_is_its_derivative(key):
    path = PATH_MODELS[key]
    theta = THETA[: len(path.parameter_names)]
    h = 1e-6
    numeric = (path.path(X + h, *theta) - path.path(X - h, *theta)) / (2 * h)
    np.testing.assert_allclose(path.slope(X, *theta), numeric, rtol=1e-7)
    # the base class's own (central differences) agrees
    np.testing.assert_allclose(
        PathModel.slope(path, X, *theta), path.slope(X, *theta), rtol=1e-7
    )
    jac = path.slope_jacobian(X, *theta)
    assert jac.shape == (X.size, len(theta))
    for j in range(len(theta)):
        up, down = list(theta), list(theta)
        up[j] += h
        down[j] -= h
        np.testing.assert_allclose(
            jac[:, j],
            (path.slope(X, *up) - path.slope(X, *down)) / (2 * h),
            rtol=1e-6,
            atol=1e-9,
        )
    np.testing.assert_allclose(
        PathModel.slope_jacobian(path, X, *theta), jac, rtol=1e-6, atol=1e-8
    )


def test_a_path_on_a_log_link_has_its_slope_on_the_link_scale():
    base = PATH_MODELS["exponential"]
    linked = LinkedPathModel(base, {"b": "log"})
    eta = [1.3, np.log(0.4)]
    np.testing.assert_allclose(linked.slope(X, *eta), base.slope(X, 1.3, 0.4))
    jac = base.slope_jacobian(X, 1.3, 0.4)
    # d / d log(b) = b d / d b
    np.testing.assert_allclose(
        linked.slope_jacobian(X, *eta), jac * np.array([1.0, 0.4])
    )


def _gauss_legendre_bound(f, points=15):
    """The error bound of ``points``-point Gauss-Legendre on (0, 1) for
    an ``f`` analytic in the Bernstein ellipses about it (Trefethen,
    Approximation Theory and Approximation Practice, theorem 19.3, halved
    for the interval's length): ``(64 / 15) M / ((rho^2 - 1) rho^(2n))``,
    ``M`` the largest ``|f|`` on the ellipse of parameter ``rho``, at the
    ``rho`` that gives the least."""
    theta = np.linspace(0.0, 2.0 * np.pi, 4001)
    best = np.inf
    for rho in np.linspace(1.1, 8.0, 300):
        t = (rho * np.exp(1j * theta) + np.exp(-1j * theta) / rho) / 2.0
        M = np.max(np.abs(f((t + 1.0) / 2.0)))
        best = min(
            best, 32.0 / 15.0 * M / ((rho**2 - 1.0) * rho ** (2 * points))
        )
    return best


def test_the_area_under_a_path_is_its_closed_form():
    a, b, c = 1.3, 0.4, 0.2
    quadratic = LinkPath(PATH_MODELS["quadratic"], "cumulative")
    np.testing.assert_allclose(
        quadratic.path(X, a, b, c),
        a * X + b * X**2 / 2 + c * X**3 / 3,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        quadratic.jacobian(X, *[np.full(X.size, v) for v in (a, b, c)]),
        np.column_stack([X, X**2 / 2, X**3 / 3]),
        rtol=1e-14,
    )
    # not a polynomial, so the rule is not exact: its error is held to
    # the Gauss-Legendre bound for an integrand analytic in a Bernstein
    # ellipse (below), plus rounding. It is 4e-12 at the oldest age,
    # 2e-17 or less at the others; the rule's actual error there is
    # 1.4e-12.
    exponential = LinkPath(PATH_MODELS["exponential"], "cumulative")
    exact = a * np.expm1(b * X) / b
    bound = np.array(
        [
            _gauss_legendre_bound(
                lambda w, x=x: 4 * a * x * w**3 * np.exp(b * x * w**4)
            )
            for x in X
        ]
    )
    assert np.all(
        np.abs(exponential.path(X, a, b) - exact) <= bound + 1e-14 * exact
    )
    # not polynomials, and singular at 0: a logarithmic path's area is
    # a t + b (t log t - t), a power path's a t^(b + 1) / (b + 1)
    logarithmic = LinkPath(PATH_MODELS["logarithmic"], "cumulative")
    np.testing.assert_allclose(
        logarithmic.path(X, a, b),
        a * X + b * (X * np.log(X) - X),
        rtol=1e-8,
    )
    power = LinkPath(PATH_MODELS["power"], "cumulative")
    np.testing.assert_allclose(
        power.path(X, a, b), a * X ** (b + 1) / (b + 1), rtol=1e-11
    )


# -- the hazard along a known path: closed forms ------------------------------


def _known_path(link, zeta, a, b, alpha, beta, t):
    """``log H(t)`` and ``log h(t)`` of an item whose line is ``a + b
    t``, by the likelihood's own functions."""
    links = (link,)
    layout = Layout(2, 0, (0, 1), links=links)
    path = PathFunction(PATH_MODELS["linear"], links)
    theta = {
        "alpha": alpha,
        "beta": beta,
        "gamma": np.zeros(0),
        "zeta": np.array([zeta]),
    }
    t = np.atleast_1d(np.asarray(t, dtype=float))
    arrays = JointArrays.build(
        np.ones(t.size),
        t,
        np.ones(t.size),
        np.zeros(t.size),
        np.zeros((t.size, 0)),
        [np.zeros(0)] * t.size,
        [np.zeros(0)] * t.size,
    )
    params = np.tile([a, b], (t.size, 1, 1))
    assert layout.n_links == 1
    return (
        np.asarray(log_cumhaz(theta, path, params, arrays, t, 0.0))[:, 0],
        np.asarray(log_event(theta, path, params, arrays, 0.0))[:, 0],
    )


def test_the_slope_of_a_line_scales_a_weibull():
    # h(t) = h0(t) exp(zeta b): a Weibull whose hazard is e^(zeta b) times
    t = np.array([0.5, 2.0, 6.0])
    log_H, log_h = _known_path("slope", 1.5, 3.0, 0.4, 4.0, 1.7, t)
    np.testing.assert_allclose(
        log_H, 1.5 * 0.4 + 1.7 * np.log(t / 4.0), rtol=0, atol=3e-6
    )
    np.testing.assert_allclose(
        log_h,
        np.log(1.7 / 4.0) + 0.7 * np.log(t / 4.0) + 1.5 * 0.4,
        rtol=0,
        atol=1e-12,
    )


def test_the_area_under_a_constant_marker_is_a_gompertz():
    # the area under a constant marker a is a t, so with beta = 1 the
    # hazard is exp(zeta a t) / alpha: H(t) = (exp(zeta a t) - 1) /
    # (alpha zeta a)
    t = np.array([0.5, 2.0, 6.0])
    zeta, a, alpha = 0.2, 3.0, 4.0
    log_H, log_h = _known_path("cumulative", zeta, a, 0.0, alpha, 1.0, t)
    np.testing.assert_allclose(
        log_H,
        np.log(np.expm1(zeta * a * t) / (alpha * zeta * a)),
        rtol=0,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        log_h, zeta * a * t - np.log(alpha), rtol=0, atol=1e-12
    )


def test_the_value_of_a_line_has_its_closed_form():
    # beta = 1: H(t) = exp(zeta a) (exp(zeta b t) - 1) / (alpha zeta b)
    t = np.array([0.5, 2.0, 6.0])
    zeta, a, b, alpha = 0.9, 3.0, 0.4, 4.0
    log_H, _ = _known_path("value", zeta, a, b, alpha, 1.0, t)
    np.testing.assert_allclose(
        log_H,
        zeta * a + np.log(np.expm1(zeta * b * t) / (alpha * zeta * b)),
        rtol=0,
        atol=3e-6,
    )


# -- the likelihood of each link against a dense grid ------------------------

#: The ages of ``KINDS`` in years (its readings' too), so a slope and an
#: area are of moderate size.
_KINDS: dict[str, Any] = dict(KINDS)
YEARS = dict(
    _KINDS,
    x=_KINDS["x"] / 365.0,
    t=_KINDS["t"] / 365.0,
    markers=[
        {age / 365.0: value for age, value in readings.items()}
        for readings in _KINDS["markers"]
    ],
)
LINK_FITS = {
    "slope": ("slope",),
    "cumulative": ("cumulative",),
    "value+slope": ("value", "slope"),
    "all": ("value", "slope", "cumulative"),
}


def _brute_force_neg_ll(model, data):
    """The negative log-likelihood of ``data`` by a dense grid over the
    random effects (161 points a dimension, 7 sds each way) and Simpson's
    rule in age (2001 points), from the model's parameters alone, the
    line's slope and area written out: no quadrature or code of the
    model's."""
    p = dict(zip(model.parameter_names, model.params))
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 161) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 161) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    log_prior = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(D))
    )
    cell = (g0[1] - g0[0]) * (g1[1] - g1[0])
    reference = model.marker_reference

    def log_sum(terms):
        top = terms.max()
        return top + np.log(np.sum(np.exp(terms - top)) * cell)

    total = 0.0
    for k, (lo, hi) in enumerate(data["x"]):
        m0, m1 = p["a"] + B0, p["b"] + B1

        def hazard(t, m0=m0, m1=m1):
            t = np.atleast_1d(t)[None, :]
            h0 = (p["beta"] / p["alpha"]) * (t / p["alpha"]) ** (p["beta"] - 1)
            eta = 0.0
            if "marker_effect" in p:
                eta = eta + p["marker_effect"] * (
                    m0[:, None] + m1[:, None] * t - reference
                )
            if "slope_effect" in p:
                eta = eta + p["slope_effect"] * m1[:, None]
            if "cumulative_effect" in p:
                eta = eta + p["cumulative_effect"] * (
                    m0[:, None] * t + m1[:, None] * t**2 / 2
                )
            return h0 * np.exp(eta)

        def survival(t, hazard=hazard):
            if not np.isfinite(t):
                return np.zeros(B0.size)
            if t <= 0:
                return np.ones(B0.size)
            s = np.linspace(0, t, 2001)
            w = np.ones(s.size)
            w[1:-1:2], w[2:-1:2] = 4, 2
            return np.exp(-hazard(s) @ w * (s[1] - s[0]) / 3)

        readings = data["markers"][k]
        ages = np.array(list(readings), dtype=float)
        ys = np.array(list(readings.values()), dtype=float)
        resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
        sd = p["sd_noise"]
        log_y = -0.5 * np.sum(
            resid**2, axis=1
        ) / sd**2 - ages.size * np.log(np.sqrt(2 * np.pi) * sd)
        tl, tr = data["t"][k]
        entry = max(tl, 0.0)
        c = data["c"][k]
        if c == 0:
            life = np.log(hazard(lo)[:, 0] * survival(lo))
        elif c == 1:
            life = np.log(survival(lo))
        elif c == 2:
            life = np.log(survival(lo) - survival(hi))
        else:
            life = np.log(survival(entry) - survival(hi))
        ll = log_sum(log_y + log_prior + life)
        if np.isfinite(tl) or np.isfinite(tr):
            ll -= log_sum(np.log(survival(entry) - survival(tr)) + log_prior)
        total += data["n"][k] * ll
    return -total


#: Parameters to evaluate each link's likelihood at (ages in years).
AT = {
    "a": 3.0,
    "b": 0.9,
    "sd_a": 0.6,
    "sd_b": 0.45,
    "corr_a_b": -0.3,
    "sd_noise": 0.5,
    "alpha": 2.5,
    "beta": 1.4,
    "marker_effect": 0.8,
    "slope_effect": 0.7,
    "cumulative_effect": 0.25,
}


def _model_at(link):
    """A model of ``link`` holding the parameters ``AT`` (not fitted)."""
    out = JointModel()
    out.dist = Weibull
    out.path_model = PATH_MODELS["linear"]
    out.varying = ["a", "b"]
    out.link = tuple(link)
    effects = {
        "value": "marker_effect",
        "slope": "slope_effect",
        "cumulative": "cumulative_effect",
    }
    out.parameter_names = list(AT)[:8] + [effects[name] for name in link]
    out.params = np.array([AT[name] for name in out.parameter_names])
    out.feature_names = []
    out.aliased = np.zeros(0, dtype=int)
    out.marker_reference = 4.0 if "value" in link else None
    out._centre = 3.5
    out._zbar = np.zeros(0)
    out._neg_ll = 0.0
    out.k = out.params.size
    return out


@pytest.mark.parametrize("name", sorted(LINK_FITS))
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(name):
    model = _model_at(LINK_FITS[name])
    data = {k: v for k, v in YEARS.items() if k != "Z"}
    f, u, _ = search_space(model, **data)
    with np.errstate(divide="ignore"):
        brute = _brute_force_neg_ll(model, data)
    # 8 rows, 11 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


# -- recovery ---------------------------------------------------------------

SIMULATED = {
    "slope": dict(alpha=3.0, beta=2.0, slope=1.0),
    "cumulative": dict(alpha=4.0, beta=1.5, cumulative=0.3),
    "value+slope": dict(alpha=2.5, beta=2.0, value=0.9, slope=0.8),
}


@functools.cache
def _data(name):
    return simulate_links(300, seed=5, **SIMULATED[name])


@functools.cache
def _fit(name):
    fitter = JointPH(Weibull, link=LINK_FITS[name])
    if "value" in LINK_FITS[name]:
        return fitter.fit(_data(name), marker_reference=4.0)
    return fitter.fit(_data(name))


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_recovers_a_fleet_simulated_from_each_link(name):
    model = _fit(name)
    assert model.maximum == "verified"
    truth = dict(
        a=3.0,
        b=1.0,
        sd_a=0.6,
        sd_b=0.5,
        corr_a_b=0.2,
        sd_noise=0.7,
        alpha=SIMULATED[name]["alpha"],
        beta=SIMULATED[name]["beta"],
        marker_effect=SIMULATED[name].get("value"),
        slope_effect=SIMULATED[name].get("slope"),
        cumulative_effect=SIMULATED[name].get("cumulative"),
    )
    target = np.array([truth[n] for n in model.parameter_names])
    z = (model.params - target) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


def test_the_wrong_link_fits_worse():
    # the slope fleet, fitted with each single link: the slope's wins
    data = _data("slope")
    own = _fit("slope").aic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = WeibullJointPH.fit(data).aic()
    assert own < value


# -- prediction ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_prediction_agrees_with_a_brute_force_integral(name):
    model = _fit(name)
    p = dict(zip(model.parameter_names, model.params))
    readings = {0.0: 3.2, 0.25: 3.6, 0.5: 4.1, 0.75: 4.2}
    given, x = 0.8, np.array([1.0, 1.5, 2.5])
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 241) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 241) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    m0, m1 = p["a"] + B0, p["b"] + B1
    ages = np.array(list(readings))
    ys = np.array(list(readings.values()))
    resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
    log_w = (
        -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
        - 0.5 * np.sum(resid**2, axis=1) / p["sd_noise"] ** 2
    )

    def survival(t):
        s = np.linspace(0, t, 2001)[None, :]
        w = np.ones(2001)
        w[1:-1:2], w[2:-1:2] = 4, 2
        eta = (
            p.get("marker_effect", 0.0) * (m0[:, None] + m1[:, None] * s - 4.0)
            + p.get("slope_effect", 0.0) * m1[:, None]
            + p.get("cumulative_effect", 0.0)
            * (m0[:, None] * s + m1[:, None] * s**2 / 2)
        )
        h = (p["beta"] / p["alpha"]) * (s / p["alpha"]) ** (p["beta"] - 1)
        return np.exp(-(h * np.exp(eta)) @ w * (t / 2000) / 3)

    weights = np.exp(log_w - log_w.max())
    base = weights @ survival(given)
    brute = np.array([weights @ survival(t) / base for t in x])
    np.testing.assert_allclose(
        model.sf(x, markers=readings, given=given), brute, rtol=2e-4
    )
    pred = model.predict(markers=readings, given=given)
    assert np.array_equal(pred.sf(x), model.sf(x, None, readings, given))
    band = model.cb(x, markers=readings, given=given)
    assert np.all(band[:, 0] < brute) and np.all(brute < band[:, 1])
    assert np.allclose(pred.ff(pred.qf([0.2, 0.6])), [0.2, 0.6])


def test_a_steeper_marker_is_at_more_risk_under_the_slope_link():
    model = _fit("slope")
    flat = {0.0: 3.5, 0.3: 3.5, 0.6: 3.6, 0.9: 3.5}
    steep = {0.0: 2.6, 0.3: 3.0, 0.6: 3.3, 0.9: 3.6}
    # the steep one ends where the flat one is, and has accumulated less
    assert model.ff(1.5, markers=steep, given=0.9) > model.ff(
        1.5, markers=flat, given=0.9
    )


def test_more_accumulated_exposure_is_more_risk_under_the_cumulative_link():
    model = _fit("cumulative")
    low = {0.0: 2.0, 0.3: 2.6, 0.6: 3.3, 0.9: 4.0}
    high = {0.0: 4.0, 0.3: 4.0, 0.6: 4.1, 0.9: 4.0}
    # both are at 4 now; the second has been there all along
    assert model.sf(1.5, markers=high, given=0.9) < model.sf(
        1.5, markers=low, given=0.9
    )


# -- names, printout, saving -------------------------------------------------


def test_each_link_names_its_effect():
    assert _fit("slope").parameter_names[-1] == "slope_effect"
    assert _fit("cumulative").parameter_names[-1] == "cumulative_effect"
    both = _fit("value+slope")
    assert both.parameter_names[-2:] == ["marker_effect", "slope_effect"]
    assert both.link == ("value", "slope")
    assert both.marker_reference == 4.0
    assert _fit("slope").marker_reference is None
    assert _fit("slope").k == 9 and both.k == 10


def test_summary_and_repr_say_the_link():
    slope, both = _fit("slope"), _fit("value+slope")
    parts = list(dict.fromkeys(slope.summary().index.get_level_values(0)))
    assert parts[-2:] == [
        "survival: Weibull PH, alpha at slope = 0",
        "link: slope",
    ]
    assert "in the marker's slope" in repr(slope)
    parts = list(dict.fromkeys(both.summary().index.get_level_values(0)))
    assert parts[-2:] == [
        "survival: Weibull PH, alpha at marker = 4, slope = 0",
        "link: current value, slope",
    ]
    assert "current value and slope" in repr(both)
    cumulative = _fit("cumulative")
    assert "alpha at area = 0" in repr(cumulative)
    assert "cumulative area" in cumulative.kind
    # the hazard ratio of each effect
    table = both.summary()
    row = table.xs("slope_effect", level="name")
    assert float(row["exp(coef)"].iloc[0]) == pytest.approx(
        np.exp(both.params[-1])
    )


def test_the_fitter_says_its_link():
    assert "link" not in repr(JointPH(Weibull))
    assert repr(JointPH(Weibull, link="slope")).endswith("link: slope)")
    # in the order of the effects, however given
    fitter = JointPH(Weibull, link=["slope", "value"])
    assert fitter.link == ("value", "slope")
    assert repr(fitter).endswith("link: value, slope)")


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_round_trips_through_json_and_pickle(name):
    model = _fit(name)
    readings = {0.0: 3.2, 0.5: 4.1}
    expected = model.sf([1.0, 2.0], markers=readings, given=0.6)
    for restored in (
        surpyval.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.link == model.link
        assert restored.parameter_names == model.parameter_names
        assert restored.marker_reference == model.marker_reference
        np.testing.assert_array_equal(
            restored.sf([1.0, 2.0], markers=readings, given=0.6), expected
        )
        np.testing.assert_array_equal(
            restored.cb([1.0, 2.0], markers=readings, given=0.6),
            model.cb([1.0, 2.0], markers=readings, given=0.6),
        )


def test_a_model_saved_before_the_links_follows_the_value():
    model = _fit("value+slope")
    saved = model.to_dict()
    assert saved["link"] == ["value", "slope"]
    old = WeibullJointPH.fit(_data("value+slope")).to_dict()
    assert old["link"] == ["value"]
    del old["link"]
    restored = JointModel.from_dict(old)
    assert restored.link == ("value",)
    assert restored.parameter_names[-1] == "marker_effect"


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "link", ["rate", [], ["value", "value"], ["slope", "area"], None]
)
def test_an_unknown_link_is_refused(link):
    with pytest.raises((ValueError, TypeError), match="link names what"):
        JointPH(Weibull, link=link)


def test_a_marker_reference_needs_the_value_link():
    with pytest.raises(ValueError, match="marker_reference.*leave it out"):
        JointPH(Weibull, link="slope").fit(
            _data("slope"), marker_reference=4.0
        )


def test_a_path_with_no_finite_area_cannot_be_accumulated():
    with pytest.raises(ValueError, match="no finite area"):
        JointPH(Weibull, LloydLipowPath, link="cumulative")
    with pytest.raises(ValueError, match="no finite area"):
        JointPH(
            Weibull,
            LinkedPathModel(LloydLipowPath, {"b": "log"}),
            link=["value", "cumulative"],
        )
    JointPH(Weibull, LloydLipowPath, link="slope")


def test_a_covariate_named_as_an_effect_is_refused():
    data = _data("slope")
    items = [
        item.clone(covariates={"slope_effect": float(k % 2)})
        for k, item in enumerate(data.to_items())
    ]
    with pytest.raises(ValueError, match="slope_effect"):
        JointPH(Weibull, link="slope").fit(
            surpyval.SurpyvalData.from_items(items)
        )
