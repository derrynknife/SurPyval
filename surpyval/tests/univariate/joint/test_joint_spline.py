"""Joint models on a spline path (``SplinePath``): the fleet's mean
curve a natural cubic spline, each item its own level and rate."""

import copy
import json
import pickle
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval
from surpyval import ItemLifetime, JointAFT, JointPH, SurpyvalData, Weibull
from surpyval.degradation import LinearPath, QuadraticPath, SplinePath
from surpyval.tests.conformance.registry_fixtures import joint_data
from surpyval.univariate.joint._likelihood import (
    LinkPath,
    PathFunction,
    lag_rule,
)

#: A knee: the marker stays near 3 and then climbs (the recovery study's
#: path, ``scripts/studies/joint_recovery/knee_study.py``).
KNEE = SplinePath(knots=[365.0], boundary=(0.0, 730.0))
TRUTH = {
    "level": 3.0,
    "rate": 0.0,
    "s1": 3.0,
    "sd_level": 0.5,
    "sd_rate": 0.3 / 365,
    "corr_level_rate": 0.0,
    "sd_noise": 0.4,
    "alpha": 800.0,
    "beta": 2.0,
    "marker_effect": 0.9,
}


def knee_fleet(n, seed, every=30.0):
    """``n`` items on the knee path, each with a level and rate of its
    own, read every ``every`` days to 730 or failure; the hazard Weibull
    PH (800, 2 at a marker of 4), 0.9 per unit of the marker."""
    from surpyval.tests.univariate.joint._fleet import simulate_path

    return simulate_path(
        KNEE,
        [3.0, 0.0, 3.0],
        [0.5, 0.3 / 365, 0.0],
        0.4,
        n=n,
        seed=seed,
        every=every,
    )


@pytest.fixture(scope="module")
def straight():
    return joint_data()


@pytest.mark.parametrize(
    "factory,link",
    [
        (JointPH, "value"),
        (JointPH, "slope"),
        (JointPH, "cumulative"),
        (JointPH, ["value", "slope"]),
        (JointAFT, "value"),
    ],
)
def test_no_interior_knots_is_the_linear_fit_bit_for_bit(
    straight, factory, link
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        line = factory(Weibull, LinearPath, link=link).fit(**straight)
        same = factory(Weibull, SplinePath(), link=link).fit(**straight)
    assert same.parameter_names[0] == "level"
    assert np.array_equal(same.params, line.params)
    assert np.array_equal(same.standard_errors(), line.standard_errors())
    assert same.neg_ll() == line.neg_ll()
    question = dict(markers={0: 3.0, 120: 3.4, 240: 3.9}, given=240.0)
    ages = np.array([300.0, 500.0, 900.0])
    assert np.array_equal(
        same.sf(ages, [1.0], **question), line.sf(ages, [1.0], **question)
    )


def test_the_hazard_is_integrated_piece_by_piece():
    """The integrals of the hazard are split at the knots (where the
    path's third derivative jumps), the lag later with a lag; a line has
    no pieces."""
    path = SplinePath(knots=[200.0, 400.0], boundary=(0.0, 700.0))
    assert PathFunction(path).cuts == (200.0, 400.0, 700.0)
    assert PathFunction(path, lag=50.0).cuts == (
        50.0,
        250.0,
        450.0,
        750.0,
    )
    assert PathFunction(SplinePath(boundary=(0.0, 700.0))).cuts == ()
    assert PathFunction(LinearPath).cuts == ()
    assert PathFunction(SplinePath.natural(df=3)).cuts == ()


@pytest.mark.parametrize("beta", [0.8, 2.0, 4.0])
@pytest.mark.parametrize("effect", [0.9, 2.0])
def test_the_split_rule_follows_a_spline_hazard(beta, effect):
    """The rule of the cumulative hazard, split at the knots, against
    adaptive quadrature: a knee whose hazard changes by up to e^5."""
    path = SplinePath(knots=[3.0, 6.0], boundary=(0.0, 9.0))
    t = np.linspace(0.0, 9.0, 19)
    theta = path.fit(t, np.where(t < 4, 2.0, 2.0 + 0.5 * (t - 4)))

    def hazard(s):
        return (
            beta
            * s ** (beta - 1)
            * np.exp(effect * np.asarray(path.path(s, *theta)))
        )

    cuts = PathFunction(path).cuts
    for upper in (3.5, 5.0, 8.0, 12.0):
        reference = quad(
            lambda s: float(hazard(s)),
            0.0,
            upper,
            points=[c for c in cuts if c < upper],
            epsabs=0,
            epsrel=1e-13,
            limit=400,
        )[0]
        ages, logw = lag_rule(np.array([0.0]), np.array([upper]), cuts)
        with np.errstate(all="ignore"):
            terms = np.exp(logw[0]) * hazard(ages[0])
        split = float(np.sum(terms[np.exp(logw[0]) > 0]))
        assert split == pytest.approx(reference, rel=1e-7)


def test_the_area_under_a_spline_is_its_closed_form():
    path = SplinePath(knots=[200.0, 400.0], boundary=(0.0, 700.0))
    theta = (3.0, 0.001, 2.0, -1.5)
    area = LinkPath(path, "cumulative")
    assert area.closed
    x = np.array([100.0, 300.0, 650.0, 900.0])
    for upper, value in zip(x, area.path(x, *theta)):
        reference = quad(
            lambda s: float(path.path(s, *theta)),
            0.0,
            upper,
            points=[200.0, 400.0, 700.0],
            epsabs=0,
            epsrel=1e-13,
        )[0]
        assert value == pytest.approx(reference, rel=1e-13)
    # the area is linear in the parameters: its Jacobian times them is
    # the area (finite differences of a value near 300 at a step of 1e-6
    # are good to 1e-8 only, so they are no reference here)
    np.testing.assert_allclose(
        area.jacobian(x, *theta) @ np.array(theta),
        area.path(x, *theta),
        rtol=1e-13,
    )
    assert np.array_equal(
        area.jacobian(x, *theta), area.jacobian(x, 1.0, 2.0, 3.0, 4.0)
    )
    # a line's area keeps the rule it had (exact for a polynomial)
    assert not LinkPath(SplinePath(), "cumulative").closed
    assert not LinkPath(LinearPath, "cumulative").closed


@pytest.fixture(scope="module")
def placed(straight):
    """The straight-line fixture fitted with a spline whose knot the fit
    places: the bend is estimated near 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return JointPH(
            Weibull, SplinePath.natural(df=2), varying=("level", "rate")
        ).fit(**straight)


def test_natural_knots_are_placed_from_the_readings_and_recorded(
    straight, placed
):
    ages = np.concatenate([list(m) for m in straight["markers"]])
    path = placed.path_model
    assert path.boundary == (float(ages.min()), float(ages.max()))
    assert path.knots == (float(np.quantile(ages, 0.5)),)
    assert placed.maximum == "verified"
    assert placed.parameter_names[:6] == [
        "level",
        "level_coef_0",
        "rate",
        "rate_coef_0",
        "s1",
        "s1_coef_0",
    ]
    assert "sd_level" in placed.parameter_names
    assert "sd_s1" not in placed.parameter_names
    # the data are straight lines: no bend
    k = placed.parameter_names.index("s1")
    assert abs(placed.params[k]) < 3 * placed.standard_errors()[k]
    assert "natural cubic spline, knots at" in str(placed)


def test_a_spline_model_saves_loads_pickles_and_predicts(placed):
    saved = json.loads(json.dumps(placed.to_dict()))
    assert saved["schema"] == 5
    assert saved["path_model"] == placed.path_model.to_dict()
    question = dict(markers={0: 3.0, 120: 3.4, 240: 3.9}, given=240.0)
    ages = np.array([300.0, 500.0, 900.0])
    expected = placed.sf(ages, [1.0], **question)
    assert np.all((expected > 0) & (expected < 1))
    assert np.all(np.diff(expected) < 0)
    for restored in (
        surpyval.from_dict(saved),
        pickle.loads(pickle.dumps(placed)),
    ):
        assert restored.path_model.knots == placed.path_model.knots
        assert np.array_equal(restored.params, placed.params)
        assert np.array_equal(restored.sf(ages, [1.0], **question), expected)
    prediction = placed.predict([1.0], **question)
    assert float(prediction.marker(240.0)) == pytest.approx(3.9, abs=0.5)
    lower, upper = placed.cb(ages, [1.0], **question).T
    assert np.all((lower < expected) & (expected < upper))
    with pytest.raises(ValueError, match="Unknown path model"):
        surpyval.from_dict({**saved, "path_model": "cubic"})


@pytest.mark.parametrize("link", ["slope", "cumulative"])
def test_the_other_links_fit_on_a_spline(straight, link):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(
            Weibull,
            SplinePath.natural(df=2),
            varying=("level", "rate"),
            link=link,
        ).fit(**straight)
        line = JointPH(Weibull, LinearPath, link=link).fit(**straight)
    assert model.maximum == "verified"
    assert np.all(np.isfinite(model.standard_errors()))
    # one more mean and one more covariate shift than the line, on data
    # that are straight lines: the likelihood rises, by little
    gain = line.neg_ll() - model.neg_ll()
    assert 0.0 <= gain < 10.0


def test_too_many_knots_for_the_readings_are_refused():
    items = [
        ItemLifetime(
            k,
            observed=100.0 + k,
            markers={0.0: 1.0, 40.0: 1.2 + 0.01 * k, 80.0: 1.5},
        )
        for k in range(12)
    ]
    data = SurpyvalData.from_items(items)
    with pytest.raises(ValueError) as refusal:
        JointPH(
            Weibull, SplinePath.natural(df=3), varying=("level", "rate")
        ).fit(data)
    message = str(refusal.value)
    assert "2 interior knot(s), so 4 parameters" in message
    assert "12 unit(s) with readings have a median of 3" in message
    assert "SplinePath.natural(df=2)" in message


@pytest.mark.calibration
def test_a_knee_fleet_is_recovered_where_a_line_is_not():
    """One fleet of the recovery study's design (150 items, seed 452,
    fixed before the fit): the spline's maximum is verified and every
    parameter is within 3 standard errors of the truth; a straight line
    on the same data fits far worse."""
    data = knee_fleet(150, 452)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spline = JointPH(Weibull, KNEE, varying=("level", "rate")).fit(
            data, marker_reference=4.0
        )
        line = JointPH(Weibull, LinearPath).fit(data, marker_reference=4.0)
    assert spline.maximum == "verified"
    z = {
        name: (spline.params[k] - TRUTH[name]) / spline.standard_errors()[k]
        for k, name in enumerate(spline.parameter_names)
    }
    assert max(abs(v) for v in z.values()) < 3, z
    assert spline.aic() < line.aic() - 10


def test_a_bound_with_a_path_parameter_common_to_all_items(straight):
    """``cb`` differentiates the prediction through the common path
    parameters' means in forward mode, which the path had no rule for
    (any path, not only the spline: here a quadratic's curvature)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, QuadraticPath, varying=("a", "b")).fit(
            **straight
        )
    question = dict(markers={0: 3.0, 120: 3.4, 240: 3.9}, given=240.0)
    ages = np.array([300.0, 500.0, 900.0])
    estimate = model.sf(ages, [1.0], **question)
    bounds = model.cb(ages, [1.0], **question)
    assert np.all((bounds[:, 0] < estimate) & (estimate < bounds[:, 1]))
    # the bound is the delta method's: against central differences
    params = np.asarray(model.params, dtype=float)
    cov = model.covariance()
    gradient = np.zeros((ages.size, params.size))
    for j in range(params.size):
        h = 1e-5 * max(abs(params[j]), 1e-3)
        values = []
        for sign in (1.0, -1.0):
            moved = params.copy()
            moved[j] += sign * h
            other = pickle.loads(pickle.dumps(model))
            other.params = moved
            values.append(np.log(-np.log(other.sf(ages, [1.0], **question))))
        gradient[:, j] = (values[0] - values[1]) / (2 * h)
    se = np.sqrt(np.einsum("mi,ij,mj->m", gradient, cov, gradient))
    centre = np.log(-np.log(estimate))
    lower = np.exp(-np.exp(centre + 1.959963984540054 * se))
    np.testing.assert_allclose(bounds[:, 0], lower, rtol=1e-4)


def test_varying_bends_default_to_their_own_blocks():
    """With a bend varying, the default structure is a block of the level
    and rate and each bend on its own; with only the level and rate, every
    pair, as for any other path."""
    path = SplinePath.natural(df=3)
    both = JointPH(Weibull, path, varying=("level", "rate", "s1", "s2"))
    assert both.random_effects == (("level", "rate"), ("s1",), ("s2",))
    assert JointPH(Weibull, path).random_effects == "full"
    one = JointPH(Weibull, path, varying=("level", "s2"))
    assert one.random_effects == (("level",), ("s2",)) or (
        one.random_effects == "diagonal"
    )
    asked = JointPH(
        Weibull, path, varying=("level", "rate", "s1"), random_effects="full"
    )
    assert asked.random_effects == "full"
    assert JointPH(Weibull, QuadraticPath).random_effects == "full"
    assert JointAFT(Weibull, path, varying=("rate", "s1")).random_effects in (
        "diagonal",
        (("rate",), ("s1",)),
    )


def test_a_bend_the_readings_do_not_reach_cannot_vary():
    """A unit's own ``s2`` acts only after the first interior knot, so the
    units' median number of readings there must reach the number of
    varying parameters."""
    items = [
        ItemLifetime(
            k,
            observed=230.0 + 5 * k,
            markers={
                age: 1.0 + 0.002 * age + 0.01 * ((k * 7 + j) % 5)
                for j, age in enumerate(np.arange(0.0, 230.0 + 5 * k, 30.0))
            },
        )
        for k in range(12)
    ]
    data = SurpyvalData.from_items(items)
    path = SplinePath(knots=[200.0, 400.0], boundary=(0.0, 520.0))
    with pytest.raises(ValueError) as refusal:
        JointPH(Weibull, path, varying=("level", "rate", "s2")).fit(data)
    message = str(refusal.value)
    assert "The bend 's2' varies between units" in message
    assert "after age 200" in message
    assert "12 unit(s) have a median of 2 reading(s) there" in message
    assert "the 3 varying parameters" in message
    assert "Leave 's2' out of varying" in message


@pytest.fixture(scope="module")
def bend_varying(straight):
    """The straight-line fixture fitted with the bend varying too, and the
    warnings the fit gave: the lines have no bend, so no spread in it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = JointPH(
            Weibull,
            SplinePath.natural(df=2),
            varying=("level", "rate", "s1"),
        ).fit(**straight)
    return model, [str(w.message) for w in caught]


def test_with_the_bend_spread_at_zero_it_is_the_common_bend_model(
    straight, placed, bend_varying
):
    """The likelihood with the bend varying, its spread held at (nearly)
    0 and every other parameter at the common-bend fit's, is the
    common-bend fit's likelihood."""
    from surpyval.univariate.joint.joint_ph import search_space

    model, _ = bend_varying
    assert model.random_effects_structure == (("level", "rate"), ("s1",))
    common = dict(zip(placed.parameter_names, placed.params))
    params = np.array(
        [
            common.get(name, 1e-7) if name != "sd_s1" else 1e-7
            for name in model.parameter_names
        ]
    )
    assert set(model.parameter_names) - set(common) == {"sd_s1"}
    held = copy.copy(model)
    held.params = params
    f1, u1, _ = search_space(placed, **straight)
    f2, u2, _ = search_space(held, **straight)
    assert float(f2(u2)) == pytest.approx(float(f1(u1)), rel=1e-9)


def test_a_bend_spread_at_its_lower_bound_is_warned_of(bend_varying):
    model, caught = bend_varying
    k = model.parameter_names.index("sd_s1")
    flat = [w for w in caught if "lower bound, 0" in w]
    assert len(flat) == 1, (model.params[k], caught)
    assert "'s1'" in flat[0]
    assert "Leave it out of varying" in flat[0]


def test_the_two_spline_fits_compare_by_aic(placed, bend_varying):
    """The fit with the bend varying has one more parameter (its sd) than
    the common-bend fit of the same data, and ranks beside it by
    ``aic()``; it saves its structure."""
    model, _ = bend_varying
    assert model.k == placed.k + 1
    assert np.isfinite(model.aic()) and np.isfinite(placed.aic())
    assert model.neg_ll() <= placed.neg_ll() + 1e-6
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.random_effects_structure == (("level", "rate"), ("s1",))
    assert np.array_equal(restored.params, model.params)
