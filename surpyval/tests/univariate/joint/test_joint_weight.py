"""The joint model with a weight on its accumulated exposure: the hazard
follows ``int_0^t w(t - s) m(s) ds`` in place of the plain area, with
``w(u) = exp(-u / scale)`` (exposure that fades) or ``w(u) = 1`` for
``u`` up to a width and 0 beyond (only a window counts).

The checks: the weighted area and its Jacobian against closed forms (a
line, a quadratic, an exponential path) and the plain area; the
likelihood of every kind of item against an independent dense-grid
integral, with and without a lag; ``weight=None`` the model without
one, number for number; a window wider than every age the plain
cumulative link; recovery of a fleet simulated with each weight
(``_fleet.simulate_links``, seeds 31 and 37, chosen before the fits were
seen); the prediction against a brute-force integral; saving and
loading; the printouts; and the refusals. The recovery study over 200
fleets is in ``tests/calibration/test_coverage_joint_weight.py``.
"""

import functools
import pickle

import numpy as np
import pytest

import surpyval
from surpyval import JointAFT, JointPH, Weibull
from surpyval.degradation.path_models import PATH_MODELS
from surpyval.serialisation import required_schema
from surpyval.tests.univariate.joint import test_joint_links as links
from surpyval.tests.univariate.joint._fleet import line_area, simulate_links
from surpyval.univariate.joint import JointModel
from surpyval.univariate.joint._likelihood import (
    EXP_SPAN,
    LaggedPath,
    LinkPath,
    PathFunction,
    WeightedArea,
)
from surpyval.univariate.joint.compare import _candidates
from surpyval.univariate.joint.joint_ph import search_space

#: The weights of the dense-grid checks (years): a scale and a width
#: inside most of the items' lives, and a scale a fortieth of the
#: longest.
WEIGHTS = {
    "exponential": ("exponential", 0.5),
    "exponential-short": ("exponential", 0.05),
    "window": ("window", 0.5),
}
#: Ages over five decades of each scale: inside it, at it, just past
#: where the exponential sum stops reaching back to age 0 (30 scales),
#: and far beyond.
AGES = np.array([1e-3, 0.02, 0.3, 0.5, 0.5000001, 1.0, 3.0, 14.9, 15.1, 80.0])

# -- the weighted area --------------------------------------------------------


@pytest.mark.parametrize("weight", [("exponential", 0.5), ("window", 0.5)])
def test_the_weighted_area_of_a_line_is_its_closed_form(weight):
    area = WeightedArea(PATH_MODELS["linear"], weight)
    a, b = 3.0, 0.4
    expected = line_area(a, b, AGES, weight)
    np.testing.assert_allclose(area.path(AGES, a, b), expected, rtol=1e-9)
    # d/da is the weighted area of 1, d/db of s
    full = [np.full(AGES.size, v) for v in (a, b)]
    np.testing.assert_allclose(
        area.jacobian(AGES, *full),
        np.column_stack(
            [
                line_area(1.0, 0.0, AGES, weight),
                line_area(0.0, 1.0, AGES, weight),
            ]
        ),
        rtol=1e-9,
    )


def _faded_power(k, x, tau):
    """``int_0^x exp(-(x - s) / tau) s^k ds`` by its recurrence."""
    out = tau * -np.expm1(-x / tau)
    for j in range(1, k + 1):
        out = tau * x**j - j * tau * out
    return out


def test_the_weighted_area_of_a_quadratic_is_its_closed_form():
    a, b, c = 3.0, 0.4, -0.05
    # ages up to 30 scales: beyond, the recurrence cancels
    x = AGES[AGES < 15]
    faded = WeightedArea(PATH_MODELS["quadratic"], ("exponential", 0.5))
    np.testing.assert_allclose(
        faded.path(x, a, b, c),
        a * _faded_power(0, x, 0.5)
        + b * _faded_power(1, x, 0.5)
        + c * _faded_power(2, x, 0.5),
        rtol=1e-9,
    )
    window = WeightedArea(PATH_MODELS["quadratic"], ("window", 0.5))
    lo = np.maximum(AGES - 0.5, 0.0)
    np.testing.assert_allclose(
        window.path(AGES, a, b, c),
        a * (AGES - lo)
        + b * (AGES**2 - lo**2) / 2
        + c * (AGES**3 - lo**3) / 3,
        rtol=1e-9,
    )


def test_the_weighted_area_of_an_exponential_path_is_its_closed_form():
    # a exp(b s): the faded area is a (e^(b x) - e^(-x / tau)) / (b + 1/tau)
    a, b, tau = 1.5, 0.8, 0.5
    x = AGES[AGES < 15]
    faded = WeightedArea(PATH_MODELS["exponential"], ("exponential", tau))
    np.testing.assert_allclose(
        faded.path(x, a, b),
        a * (np.exp(b * x) - np.exp(-x / tau)) / (b + 1 / tau),
        rtol=1e-9,
    )
    window = WeightedArea(PATH_MODELS["exponential"], ("window", 0.5))
    lo = np.maximum(x - 0.5, 0.0)
    np.testing.assert_allclose(
        window.path(x, a, b),
        a * (np.exp(b * x) - np.exp(b * lo)) / b,
        rtol=1e-9,
    )


def test_exposure_older_than_30_scales_is_left_out():
    # what the exponential sum leaves out is below exp(-30) of the area
    area = WeightedArea(PATH_MODELS["linear"], ("exponential", 0.1))
    ages, weights = area.rule(np.array([10.0]))
    assert ages.min() > 10.0 - EXP_SPAN * 0.1
    assert weights.sum() == pytest.approx(0.1, rel=1e-10)


def test_a_window_back_to_age_0_and_a_slow_fade_are_the_plain_area():
    plain = LinkPath(PATH_MODELS["power"], "cumulative")
    x = np.array([0.01, 0.5, 2.0])
    window = WeightedArea(PATH_MODELS["power"], ("window", 2.0))
    np.testing.assert_allclose(
        window.path(x, 2.0, 0.4), plain.path(x, 2.0, 0.4), rtol=1e-14
    )
    slow = WeightedArea(PATH_MODELS["power"], ("exponential", 1e12))
    np.testing.assert_allclose(
        slow.path(x, 2.0, 0.4), plain.path(x, 2.0, 0.4), rtol=1e-9
    )


def test_a_lagged_weighted_area_is_the_weighted_area_a_lag_earlier():
    weight = ("window", 0.5)
    path = PathFunction(PATH_MODELS["linear"], ("cumulative",), 1.0, weight)
    ((_, followed, _),) = path.terms
    assert isinstance(followed, LaggedPath)
    x = np.array([0.2, 1.0, 1.3, 4.0])
    np.testing.assert_allclose(
        followed.path(x, 3.0, 0.4),
        line_area(3.0, 0.4, np.maximum(x - 1.0, 0.0), weight),
        atol=1e-12,
    )
    # the hazard's integrals are split at the lag and where the window
    # stops reaching back to age 0
    assert path.cuts == (1.0, 1.5)
    faded = PathFunction(
        PATH_MODELS["linear"], ("cumulative",), 1.0, ("exponential", 0.5)
    )
    assert faded.cuts == (1.0,)
    # no weight without the link it weights
    assert PathFunction(PATH_MODELS["linear"], ("value",)).weight is None


# -- the likelihood against a dense grid --------------------------------------


@pytest.mark.parametrize("lag", [0.0, 0.6])
@pytest.mark.parametrize("name", sorted(WEIGHTS))
@pytest.mark.parametrize("link", [("cumulative",), ("value", "cumulative")])
def test_every_kind_of_item_agrees_with_a_dense_grid_integral(link, name, lag):
    model = links._model_at(link)
    model.weight, model.lag = WEIGHTS[name], lag
    data = {k: v for k, v in links.YEARS.items() if k != "Z"}
    f, u, _ = search_space(model, **data)
    with np.errstate(divide="ignore"):
        brute = links._brute_force_neg_ll(
            model, data, lag=lag, weight=WEIGHTS[name]
        )
    # 8 rows, 11 items: to within 1e-5 an item
    assert float(f(u)) == pytest.approx(brute, abs=1.1e-4)


def test_a_window_wider_than_every_age_is_the_plain_cumulative_link():
    data = {k: v for k, v in links.YEARS.items() if k != "Z"}
    question = dict(markers={0.0: 3.2, 0.25: 3.6, 0.5: 4.1}, given=0.5)
    for link in (("cumulative",), ("value", "slope", "cumulative")):
        plain = links._model_at(link)
        wide = links._model_at(link)
        wide.weight = ("window", 1000.0)
        f, u, _ = search_space(plain, **data)
        g, v, _ = search_space(wide, **data)
        np.testing.assert_array_equal(u, v)
        assert float(g(v)) == pytest.approx(float(f(u)), rel=1e-10)
        np.testing.assert_allclose(
            wide.sf([0.8, 1.2, 2.0], **question),
            plain.sf([0.8, 1.2, 2.0], **question),
            rtol=1e-10,
        )


# -- fits ---------------------------------------------------------------------

SIMULATED = {
    "exponential": dict(seed=31, weight=("exponential", 0.5), cumulative=0.9),
    "window": dict(seed=37, weight=("window", 0.5), cumulative=0.9),
}


@functools.cache
def _data(name):
    """A fleet whose hazard follows the line's faded exposure, or its
    exposure over the last half year."""
    return simulate_links(300, alpha=3.0, beta=2.0, **SIMULATED[name])


@functools.cache
def _fit(name, weight):
    return JointPH(Weibull, link="cumulative", weight=weight).fit(_data(name))


def test_no_weight_is_the_model_without_one():
    data = simulate_links(80, seed=11, alpha=4.0, beta=1.5, cumulative=0.3)
    plain = JointPH(Weibull, link="cumulative").fit(data)
    none = JointPH(Weibull, link="cumulative", weight=None).fit(data)
    np.testing.assert_array_equal(none.params, plain.params)
    np.testing.assert_array_equal(none.covariance(), plain.covariance())
    assert none.neg_ll() == plain.neg_ll()
    assert none.weight is None
    assert "weight" not in none.to_dict()
    assert required_schema(none.to_dict()) < 5


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_recovers_a_fleet_simulated_with_each_weight(name):
    weight = SIMULATED[name]["weight"]
    model = _fit(name, weight)
    assert model.maximum == "verified"
    assert model.weight == weight
    truth = dict(
        a=3.0,
        b=1.0,
        sd_a=0.6,
        sd_b=0.5,
        corr_a_b=0.2,
        sd_noise=0.7,
        alpha=3.0,
        beta=2.0,
        cumulative_effect=0.9,
    )
    target = np.array([truth[n] for n in model.parameter_names])
    z = (model.params - target) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_weights_compare_by_aic_on_the_same_parameters(name):
    # The same parameters, so the criteria differ by the likelihood only
    # (which weight wins is reported in the calibration study).
    weighted, plain = _fit(name, SIMULATED[name]["weight"]), _fit(name, None)
    assert weighted.k == plain.k
    assert weighted.aic() - plain.aic() == pytest.approx(
        2 * (weighted.neg_ll() - plain.neg_ll())
    )


@pytest.mark.parametrize("name", sorted(WEIGHTS))
def test_prediction_agrees_with_a_brute_force_integral(name):
    weight = WEIGHTS[name]
    model = links._model_at(("value", "cumulative"))
    model.weight = weight
    p = dict(zip(model.parameter_names, model.params))
    readings = {0.0: 3.2, 0.25: 3.6, 0.5: 4.1}
    given, x = 0.5, np.array([0.8, 1.2, 2.0])
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 201) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 201) * np.sqrt(v1)
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

    def hazard(t):
        t = np.atleast_1d(t)[None, :]
        eta = p["marker_effect"] * (
            m0[:, None] + m1[:, None] * t - model.marker_reference
        ) + p["cumulative_effect"] * line_area(
            m0[:, None], m1[:, None], t, weight
        )
        return (
            (p["beta"] / p["alpha"])
            * (t / p["alpha"]) ** (p["beta"] - 1)
            * np.exp(eta)
        )

    def log_S(t):
        # split where the window stops reaching back to age 0
        return -links._simpson(hazard, 0.0, t, weight[1])

    base = log_w + log_S(given)
    top = base.max()
    expected = [
        np.sum(np.exp(log_w + log_S(t) - top)) / np.sum(np.exp(base - top))
        for t in x
    ]
    got = model.sf(x, markers=readings, given=given)
    np.testing.assert_allclose(got, expected, rtol=2e-5)


# -- saving, printing ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SIMULATED))
def test_round_trips_through_json_and_pickle(name):
    weight = SIMULATED[name]["weight"]
    model = _fit(name, weight)
    saved = model.to_dict()
    assert saved["weight"] == list(weight)
    restored = surpyval.from_dict(saved)
    assert restored.weight == weight
    question = dict(markers={0.0: 3.2, 0.5: 3.9}, given=0.6)
    expected = model.sf([1.0, 1.5], **question)
    np.testing.assert_allclose(restored.sf([1.0, 1.5], **question), expected)
    unpickled = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(unpickled.sf([1.0, 1.5], **question), expected)
    # the weight changes the prediction
    without = surpyval.from_dict(
        {k: v for k, v in saved.items() if k != "weight"}
    )
    assert without.weight is None
    assert not np.allclose(without.sf([1.0, 1.5], **question), expected)


def test_a_weighted_model_saves_as_schema_5_which_0_28_refuses():
    # SurPyval 0.28 reads schema 4 and below; it would ignore "weight"
    # and predict as if all the exposure counted equally.
    assert _fit("window", ("window", 0.5)).to_dict()["schema"] == 5
    plain = _fit("window", None).to_dict()
    assert plain["schema"] < 5
    assert required_schema(dict(plain, weight=["exponential", 0.5])) == 5


def test_a_saved_weight_that_is_not_one_is_refused():
    saved = _fit("window", ("window", 0.5)).to_dict()
    with pytest.raises(ValueError, match="'weight' says how much"):
        JointModel.from_dict(dict(saved, weight=["window", -1.0]))
    with pytest.raises(ValueError, match="'weight' says how much"):
        JointModel.from_dict(dict(saved, weight=["triangle", 1.0]))
    with pytest.raises(ValueError, match="weight is on its accumulated"):
        JointModel.from_dict(dict(saved, link=["value"]))


def test_the_printouts_say_the_weight():
    fitter = JointPH(Weibull, link="cumulative", weight=("exponential", 0.5))
    assert repr(fitter).endswith("weight: exponential 0.5)")
    assert "weight" not in repr(JointPH(Weibull, link="cumulative"))
    faded = _fit("exponential", ("exponential", 0.5))
    assert "the area fading over 0.5 (exponential weight)" in repr(faded)
    parts = faded.summary().index.get_level_values("part")
    assert (
        "link: cumulative area, the area fading over 0.5 (exponential "
        "weight)" in set(parts)
    )
    window = _fit("window", ("window", 0.5))
    assert "the area over the last 0.5 only (window)" in repr(window)
    assert "weight" not in repr(_fit("window", None))
    # the effect is still per unit of the marker times a unit of age
    assert window.effect_units == _fit("window", None).effect_units


def test_compare_links_keeps_the_weight_on_the_links_with_an_area():
    weight = ("window", 0.5)
    fitter = JointPH(Weibull, link="cumulative", weight=weight)
    fitters, _, _ = _candidates(fitter, None)
    assert {f.link: f.weight for f in fitters} == {
        f.link: (weight if "cumulative" in f.link else None) for f in fitters
    }
    assert any(f.weight for f in fitters)


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "weight",
    [
        "exponential",
        ("exponential",),
        ("exponential", 0.0),
        ("window", -1.0),
        ("window", np.nan),
        ("window", np.inf),
        ("window", [1.0, 2.0]),
        ("triangle", 1.0),
        (1.0, "window"),
        ("window", 1.0, 2.0),
        0.5,
    ],
)
def test_a_weight_that_is_not_a_kind_and_a_scale_above_0_is_refused(weight):
    with pytest.raises(ValueError, match="'weight' says how much.*got"):
        JointPH(Weibull, link="cumulative", weight=weight)


@pytest.mark.parametrize("link", ["value", "slope", ["value", "slope"]])
def test_a_weight_without_the_cumulative_link_is_refused(link):
    with pytest.raises(
        ValueError, match="no area to weight.*link='cumulative'"
    ):
        JointPH(Weibull, link=link, weight=("window", 1.0))


def test_an_accelerated_model_takes_no_weight():
    # it has no cumulative link (its effective age accumulates the marker)
    with pytest.raises(TypeError, match="weight"):
        JointAFT(Weibull, weight=("window", 1.0))
