"""The accelerated joint model, ``JointAFT`` and ``WeibullJointAFT``: the
marker speeds up the item's ageing in place of multiplying its hazard.

The checks: the effective age, the cumulative hazard and the hazard along
a known line against their closed forms; the likelihood of every kind of
item (failed, running, left and interval censored, left and right
truncated), with covariates in each part and for each link, against an
independent dense-grid integral over the random effects; with a marker
that never changes, the fit is the proportional hazards one with every
effect divided by ``beta``; recovery of fleets simulated from the model
(``_fleet.simulate_aft``, exact lifetimes, seeds 31 to 33 chosen before
the fits were seen); the prediction against a brute-force integral, its
hazard against the derivative of its survival and its quantiles against
its survival; and the names, the printout, the refusals and the saved
form.
"""

import functools
import pickle
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import (
    JointAFT,
    JointModel,
    JointPH,
    SurpyvalData,
    Weibull,
    WeibullJointAFT,
    WeibullJointPH,
    serialisation,
)
from surpyval.degradation import PathModel
from surpyval.degradation.path_models import PATH_MODELS
from surpyval.tests.univariate.joint._fleet import simulate_aft
from surpyval.tests.univariate.joint.test_joint_links import YEARS
from surpyval.univariate.joint import _aft
from surpyval.univariate.joint._likelihood import (
    JointArrays,
    PathFunction,
)
from surpyval.univariate.joint.joint_ph import search_space

# -- the hazard along a known line: closed forms ------------------------------


def _known_line(links, zetas, a, b, alpha, beta, t, centre=0.0):
    """``(log A, log H, log h)`` at ages ``t`` of an item whose line is
    ``a + b t``, by the accelerated part's own functions."""
    links = tuple(links)
    path = PathFunction(PATH_MODELS["linear"], links)
    theta = {
        "alpha": alpha,
        "beta": beta,
        "gamma": np.zeros(0),
        "zeta": np.asarray(zetas, dtype=float),
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
    return (
        np.asarray(_aft.log_age(theta, path, params, arrays, t, centre))[:, 0],
        np.asarray(_aft.log_cumhaz(theta, path, params, arrays, t, centre))[
            :, 0
        ],
        np.asarray(_aft.log_event(theta, path, params, arrays, centre))[:, 0],
        (theta, path, params, arrays),
    )


def test_the_value_of_a_line_ages_the_item_in_closed_form():
    # rate exp(zeta (a + b s - r)): A(t) = exp(zeta (a - r)) expm1(zeta b
    # t) / (zeta b); H = (A / alpha)^beta; h = h0(A) times the rate at t
    t = np.array([0.3, 1.0, 2.5, 7.0])
    zeta, a, b, r, alpha, beta = 0.5, 3.0, 0.4, 4.0, 5.0, 2.0
    log_A, log_H, log_h, _ = _known_line(
        ["value"], [zeta], a, b, alpha, beta, t, centre=r
    )
    A = np.exp(zeta * (a - r)) * np.expm1(zeta * b * t) / (zeta * b)
    np.testing.assert_allclose(log_A, np.log(A), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        log_H, beta * np.log(A / alpha), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        log_h,
        np.log(beta / alpha)
        + (beta - 1) * np.log(A / alpha)
        + zeta * (a + b * t - r),
        rtol=0,
        atol=1e-12,
    )


def test_a_constant_marker_rescales_the_clock():
    # a marker held at 5.4 with 0.5 per unit about 4.0 ages the item
    # e^0.7 times as fast: three years use up 3 e^0.7 years
    t = np.array([1.0, 3.0])
    log_A, log_H, _, _ = _known_line(
        ["value"], [0.5], 5.4, 0.0, 5.0, 2.0, t, centre=4.0
    )
    np.testing.assert_allclose(np.exp(log_A), t * np.exp(0.7), rtol=1e-13)
    np.testing.assert_allclose(
        np.exp(-np.exp(log_H)),
        Weibull.sf(t * np.exp(0.7), 5.0, 2.0),
        rtol=1e-13,
    )


def test_the_slope_of_a_line_is_a_constant_acceleration():
    t = np.array([0.5, 2.0, 6.0])
    log_A, log_H, log_h, _ = _known_line(
        ["slope"], [1.5], 3.0, 0.4, 4.0, 1.7, t
    )
    np.testing.assert_allclose(log_A, np.log(t) + 0.6, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        log_h,
        np.log(1.7 / 4.0) + 0.7 * (np.log(t) + 0.6 - np.log(4.0)) + 0.6,
        rtol=0,
        atol=1e-12,
    )


def test_an_interval_is_the_difference_of_two_survivals():
    zeta, a, b, alpha, beta = 0.5, 3.0, 0.4, 5.0, 2.0
    lo = np.array([0.0, 1.0, 6.0, 6.0])
    hi = np.array([2.0, 3.0, 6.5, 6.0 + 1e-6])
    _, _, _, (theta, path, params, arrays) = _known_line(
        ["value"], [zeta], a, b, alpha, beta, hi, centre=4.0
    )
    out = np.asarray(
        _aft.log_fails_between(theta, path, params, arrays, lo, hi, 4.0)
    )[:, 0]

    def H(t):
        A = np.exp(zeta * (a - 4.0)) * np.expm1(zeta * b * t) / (zeta * b)
        return (A / alpha) ** beta

    # log(1 - S(hi) / S(lo)) = log(-expm1(-(H(hi) - H(lo)))); the last
    # interval is narrow, its H(hi) - H(lo) from the hazard at lo
    gap = H(hi) - H(lo)
    np.testing.assert_allclose(
        out[:3], np.log(-np.expm1(-gap[:3])), rtol=1e-12
    )
    A_lo = np.exp(zeta * (a - 4.0)) * np.expm1(zeta * b * lo[3]) / (zeta * b)
    rate = np.exp(zeta * (a + b * 6.0000005 - 4.0))
    h = beta / alpha * (A_lo / alpha) ** (beta - 1) * rate
    np.testing.assert_allclose(out[3], np.log(h * 1e-6), rtol=0, atol=1e-6)


# -- the likelihood against a dense grid --------------------------------------

#: Parameters to evaluate the likelihood at (ages in years): ``site`` in
#: the ageing rate only, ``load`` in the line's rate only.
AT = {
    "a": 3.0,
    "b": 0.9,
    "b_load": 0.3,
    "sd_a": 0.6,
    "sd_b": 0.45,
    "corr_a_b": -0.3,
    "sd_noise": 0.5,
    "alpha": 2.5,
    "beta": 1.4,
    "site": 0.4,
    "marker_effect": 0.5,
    "slope_effect": 0.6,
}
EFFECTS = {"value": "marker_effect", "slope": "slope_effect"}


def _model_at(link, beta):
    """An accelerated model of ``link`` holding the parameters ``AT``
    with ``beta`` (not fitted)."""
    out = JointModel()
    out.dist = Weibull
    out.path_model = PATH_MODELS["linear"]
    out.varying = ["a", "b"]
    out.link = tuple(link)
    out.survival = "AFT"
    out.feature_names = ["site", "load"]
    out.path_covariates = {"a": [], "b": ["load"]}
    out.hazard_covariates = ["site"]
    out.parameter_names = list(AT)[:10] + [EFFECTS[name] for name in link]
    out.params = np.array(
        [beta if n == "beta" else AT[n] for n in out.parameter_names]
    )
    out.aliased = np.zeros(0, dtype=int)
    out.marker_reference = 4.0 if "value" in link else None
    out._centre = 3.5
    out._zbar = np.array([0.4, 0.6])
    out._neg_ll = 0.0
    out.k = out.params.size
    return out


def _brute_force_neg_ll(model, data):
    """The negative log-likelihood of ``data`` by a dense grid over the
    random effects (161 points a dimension, 7 sds each way) and Simpson's
    rule in age (2001 points) for the effective age, from the model's
    parameters alone: no quadrature or code of the model's."""
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
    alpha, beta = p["alpha"], p["beta"]

    def log_sum(terms):
        top = terms.max()
        return top + np.log(np.sum(np.exp(terms - top)) * cell)

    total = 0.0
    for k, (lo, hi) in enumerate(data["x"]):
        site, load = data["Z"][k]
        m0 = p["a"] + B0
        m1 = p["b"] + p["b_load"] * load + B1

        def rate(t, m0=m0, m1=m1, site=site):
            t = np.atleast_1d(t)[None, :]
            eta = p["site"] * site
            if "marker_effect" in p:
                eta = eta + p["marker_effect"] * (
                    m0[:, None] + m1[:, None] * t - model.marker_reference
                )
            if "slope_effect" in p:
                eta = eta + p["slope_effect"] * m1[:, None]
            return np.exp(eta) * np.ones_like(t)

        def age(t, rate=rate):
            s = np.linspace(0, t, 2001)
            w = np.ones(s.size)
            w[1:-1:2], w[2:-1:2] = 4, 2
            return rate(s) @ w * (s[1] - s[0]) / 3

        def survival(t, age=age):
            if not np.isfinite(t):
                return np.zeros(B0.size)
            if t <= 0:
                return np.ones(B0.size)
            return np.exp(-((age(t) / alpha) ** beta))

        def hazard(t, age=age, rate=rate):
            A = age(t)
            return beta / alpha * (A / alpha) ** (beta - 1) * rate(t)[:, 0]

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
            life = np.log(hazard(lo) * survival(lo))
        elif c == 1:
            life = np.log(survival(lo))
        elif c == 2:
            life = np.log(survival(lo) - survival(hi))
        else:  # failed by hi, having reached its entry
            life = np.log(survival(entry) - survival(hi))
        ll = log_sum(log_y + log_prior + life)
        if np.isfinite(tl) or np.isfinite(tr):
            ll -= log_sum(np.log(survival(entry) - survival(tr)) + log_prior)
        total += data["n"][k] * ll
    return -total


@pytest.mark.parametrize(
    "link, beta",
    [(("value",), 1.4), (("slope",), 0.8), (("value", "slope"), 1.4)],
    ids=["value", "slope-beta-below-1", "value+slope"],
)
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(link, beta):
    model = _model_at(link, beta)
    f, u, _ = search_space(model, **YEARS)
    with np.errstate(divide="ignore"):
        brute = _brute_force_neg_ll(model, YEARS)
    # 8 rows, 11 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


# -- a marker that never changes: the proportional hazards model --------------


class LevelPath(PathModel):
    """y = a: a marker that never changes (module level, to pickle)."""

    name = "Level"
    parameter_names = ["a"]

    def path(self, x, a):
        return a + 0.0 * np.asarray(x, dtype=float)

    def inv_path(self, y, a):
        return np.full(np.shape(y), np.nan)

    def _initial_guess(self, x, y):
        return [float(np.mean(y))]


def _levels(n=200, seed=41):
    """Items whose marker is a level of their own (3.0, sd 0.6), read
    every 0.25 years with noise 0.5, failing at a Weibull (2.5, 1.8) PH
    hazard with 0.9 per unit about 4.0; followed for two years."""
    rng = np.random.default_rng(seed)
    level = 3.0 + 0.6 * rng.normal(size=n)
    scale = 2.5 * np.exp(-0.9 * (level - 4.0) / 1.8)
    life = scale * rng.weibull(1.8, size=n)
    items = []
    for k in range(n):
        end = min(life[k], 2.0)
        ages = np.arange(0.0, end + 1e-9, 0.25)
        y = level[k] + 0.5 * rng.normal(size=ages.size)
        outcome = "observed" if life[k] <= 2.0 else "right_censored"
        items.append(
            surpyval.ItemLifetime(
                k,
                **{outcome: float(end)},
                markers=dict(zip(ages.tolist(), y.tolist())),
            )
        )
    return SurpyvalData.from_items(items)


def test_a_constant_marker_is_the_proportional_hazards_model():
    data = _levels()
    ph = JointPH(Weibull, LevelPath()).fit(data, marker_reference=4.0)
    aft = JointAFT(Weibull, LevelPath()).fit(data, marker_reference=4.0)
    assert ph.maximum == aft.maximum == "verified"
    assert aft.parameter_names == ph.parameter_names
    assert aft.neg_ll() == pytest.approx(ph.neg_ll(), rel=1e-9, abs=1e-6)
    p_ph = dict(zip(ph.parameter_names, ph.params))
    p_aft = dict(zip(aft.parameter_names, aft.params))
    for name in ("a", "sd_a", "sd_noise", "alpha", "beta"):
        assert p_aft[name] == pytest.approx(p_ph[name], rel=1e-4), name
    # the PH effect is beta times the accelerated one
    assert p_ph["marker_effect"] == pytest.approx(
        p_aft["beta"] * p_aft["marker_effect"], rel=1e-4
    )
    readings, x = {0.0: 3.4, 0.5: 3.9, 1.0: 3.6}, [1.5, 2.5, 4.0]
    np.testing.assert_allclose(
        aft.sf(x, markers=readings, given=1.0),
        ph.sf(x, markers=readings, given=1.0),
        rtol=1e-4,
    )


# -- recovery -----------------------------------------------------------------

#: The fleets, each fitted with its own link; seeds fixed before the fits.
SIMULATED = {
    "value": dict(seed=31, alpha=2.5, beta=2.0, value=0.8, gamma=0.5),
    "slope": dict(seed=32, alpha=2.5, beta=2.0, slope=0.7),
    "value+slope": dict(seed=33, alpha=2.5, beta=1.5, value=0.6, slope=0.5),
}
LINKS = {
    "value": ("value",),
    "slope": ("slope",),
    "value+slope": ("value", "slope"),
}


@functools.cache
def _data(name):
    return simulate_aft(300, **SIMULATED[name])


@functools.cache
def _fit(name):
    fitter = JointAFT(Weibull, link=LINKS[name])
    if "value" in LINKS[name]:
        return fitter.fit(_data(name), marker_reference=4.0)
    return fitter.fit(_data(name))


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_recovers_a_fleet_simulated_from_the_model(name):
    model = _fit(name)
    assert model.maximum == "verified"
    assert model.survival == "AFT"
    spec = SIMULATED[name]
    truth = dict(
        a=3.0,
        b=1.0,
        sd_a=0.6,
        sd_b=0.5,
        corr_a_b=0.2,
        sd_noise=0.7,
        alpha=spec["alpha"],
        beta=spec["beta"],
        z=spec.get("gamma"),
        a_z=0.0,
        b_z=0.0,
        marker_effect=spec.get("value"),
        slope_effect=spec.get("slope"),
    )
    target = np.array([truth[n] for n in model.parameter_names])
    z = (model.params - target) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


# -- prediction ---------------------------------------------------------------


def test_prediction_agrees_with_a_brute_force_integral():
    model = _fit("value+slope")
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
        # the effective age in closed form for a line
        k = p["marker_effect"] * m1
        r0 = np.exp(p["marker_effect"] * (m0 - 4.0) + p["slope_effect"] * m1)
        A = r0 * np.expm1(k * t) / k
        return np.exp(-((A / p["alpha"]) ** p["beta"]))

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


def test_the_hazard_is_the_derivative_of_the_survival_and_qf_inverts_it():
    pred = _fit("value").predict(
        {"z": 1.0}, markers={0.0: 3.0, 0.5: 3.8, 1.0: 4.4}, given=1.0
    )
    x, h = np.array([1.2, 1.8, 2.6]), 1e-5
    numeric = -(np.log(pred.sf(x + h)) - np.log(pred.sf(x - h))) / (2 * h)
    np.testing.assert_allclose(pred.hf(x), numeric, rtol=1e-5)
    np.testing.assert_allclose(pred.ff(pred.qf([0.2, 0.6])), [0.2, 0.6])
    assert pred.sf(1.0) == 1.0
    draws = pred.random(50, random_state=3)
    assert np.all(draws > 1.0)


def test_a_higher_marker_ages_the_item_faster():
    model = _fit("value")
    low = {0.0: 2.6, 0.5: 2.9, 1.0: 3.1}
    high = {0.0: 3.6, 0.5: 3.9, 1.0: 4.1}
    assert model.sf(2.0, {"z": 0.0}, high, 1.0) < model.sf(
        2.0, {"z": 0.0}, low, 1.0
    )


# -- names, printout, saving --------------------------------------------------


def test_the_fitter_says_it_is_accelerated():
    assert WeibullJointAFT.name == "WeibullJointAFT"
    assert repr(WeibullJointAFT) == (
        "WeibullJointAFT: joint model fitter (Weibull baseline, "
        "accelerated, Linear path, a + b t)"
    )
    assert JointAFT(Weibull, link=["slope", "value"]).link == (
        "value",
        "slope",
    )
    assert (
        WeibullJointPH.survival == "PH" and WeibullJointAFT.survival == "AFT"
    )


def test_summary_and_repr_say_the_survival_part():
    model = _fit("value")
    parts = list(dict.fromkeys(model.summary().index.get_level_values(0)))
    assert parts[-2:] == [
        "survival: Weibull AFT, alpha at marker = 4",
        "link: current value",
    ]
    text = repr(model)
    assert "accelerated failure time, ageing sped up" in text
    assert (
        model.kind == "Joint (Weibull accelerated failure time, current value)"
    )
    assert "proportional hazards" in WeibullJointPH.fit(_data("value")).kind
    row = model.summary().xs("marker_effect", level="name")
    assert float(row["exp(coef)"].iloc[0]) == pytest.approx(
        np.exp(model.params[-1])
    )


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_round_trips_through_json_and_pickle(name):
    model = _fit(name)
    readings = {0.0: 3.2, 0.5: 4.1}
    Z = {"z": 1.0} if "gamma" in SIMULATED[name] else None
    expected = model.sf([1.0, 2.0], Z, readings, 0.6)
    saved = model.to_dict()
    assert saved["survival"] == "AFT" and saved["schema"] == 5
    for restored in (
        surpyval.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.survival == "AFT"
        assert restored.link == model.link
        assert restored.parameter_names == model.parameter_names
        np.testing.assert_array_equal(
            restored.sf([1.0, 2.0], Z, readings, 0.6), expected
        )
        np.testing.assert_array_equal(
            restored.cb([1.0, 2.0], Z, readings, 0.6),
            model.cb([1.0, 2.0], Z, readings, 0.6),
        )


def test_a_proportional_hazards_model_keeps_its_schema(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = WeibullJointPH.fit(_data("slope"))
    saved = ph.to_dict()
    assert saved["survival"] == "PH" and saved["schema"] < 5
    # saved before the accelerated part: proportional hazards
    del saved["survival"]
    assert JointModel.from_dict(saved).survival == "PH"
    # an older reader refuses the accelerated model rather than misread it
    monkeypatch.setattr(serialisation, "SCHEMA_VERSION", 4)
    with pytest.raises(ValueError, match="Upgrade surpyval"):
        surpyval.from_dict(_fit("slope").to_dict())


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "link", ["cumulative", ["value", "cumulative"], ("slope", "cumulative")]
)
def test_the_cumulative_link_is_refused(link):
    with pytest.raises(ValueError, match="already accumulates the marker"):
        JointAFT(Weibull, link=link)


@pytest.mark.parametrize("link", ["rate", [], ["value", "value"], None])
def test_an_unknown_link_is_refused(link):
    with pytest.raises((ValueError, TypeError), match="link names what"):
        JointAFT(Weibull, link=link)


def test_another_baseline_is_refused():
    with pytest.raises(ValueError, match="JointAFT.Weibull, path_model"):
        JointAFT(surpyval.LogNormal)


def test_a_marker_reference_needs_the_value_link():
    with pytest.raises(ValueError, match="marker_reference.*leave it out"):
        JointAFT(Weibull, link="slope").fit(
            _data("slope"), marker_reference=4.0
        )


def test_a_dict_with_an_unknown_survival_part_is_refused():
    saved = _fit("slope").to_dict()
    with pytest.raises(ValueError, match="survival part"):
        JointModel.from_dict(dict(saved, survival="PO"))
    with pytest.raises(ValueError, match="area under the marker"):
        JointModel.from_dict(dict(saved, link=["cumulative"]))
