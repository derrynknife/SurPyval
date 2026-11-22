"""``release_time`` and ``release_time_cb``: the cheapest time to stop
testing (Okumoto and Goel 1980), on a fitted Poisson process.

The optimum against its closed form for Goel-Okumoto, the cost against
the sum written out, the reliability-constrained case, the band of
near-best times, the bounds and the refusals. The bounds' coverage is in
``calibration/test_coverage_growth_answers.py``.
"""

import numpy as np
import pytest

from surpyval.recurrent import (
    HPP,
    CrowAMSAA,
    GoelOkumoto,
    MusaOkumoto,
    YamadaDelayedS,
)

A, B = 100.0, 0.05
C1, C2, C3 = 10.0, 500.0, 50.0  # fix, field, a day of testing


@pytest.fixture(scope="module")
def fitted():
    d = GoelOkumoto.from_params([A, B]).time_terminated_simulation_data(
        60.0, random_state=1
    )
    return GoelOkumoto.fit(d.x, c=d.c)


def _cost(t, life=np.inf):
    found = A * -np.expm1(-B * t)
    total = A * -np.expm1(-B * life)
    return C1 * found + C2 * (total - found) + C3 * t


def test_goel_okumoto_optimum_is_the_closed_form():
    model = GoelOkumoto.from_params([A, B])
    plan = model.release_time(C3, C1, C2)
    best = np.log(A * B * (C2 - C1) / C3) / B
    assert plan["time"] == pytest.approx(best, rel=1e-10)
    assert plan["cost"] == pytest.approx(_cost(best), rel=1e-12)
    assert plan["binding"] == "cost"
    assert plan["cost_time"] == plan["time"]
    assert plan["least_cost"] == plan["cost"]
    assert plan["reliability_time"] is None
    # the intensity there is the level at which testing stops paying
    assert float(model.iif(plan["time"])) == pytest.approx(
        C3 / (C2 - C1), rel=1e-10
    )
    # and it is the cheapest day
    days = np.linspace(0.0, 300.0, 3001)
    assert np.all(_cost(days) >= plan["cost"] - 1e-9)


def test_the_band_is_the_days_within_the_share_of_the_least_cost():
    model = GoelOkumoto.from_params([A, B])
    for within in (0.01, 0.1):
        plan = model.release_time(C3, C1, C2, within=within)
        lower, upper = plan["band"]
        assert lower < plan["time"] < upper
        ceiling = plan["cost"] * (1 + within)
        assert _cost(lower) == pytest.approx(ceiling, rel=1e-9)
        assert _cost(upper) == pytest.approx(ceiling, rel=1e-9)
        inside = np.linspace(lower, upper, 50)
        assert np.all(_cost(inside) <= ceiling * (1 + 1e-12))


def test_a_life_ends_the_field_cost():
    model = GoelOkumoto.from_params([A, B])
    plan = model.release_time(C3, C1, C2, life=100.0)
    # the optimum does not move (the faults found after the life cost
    # nothing either way), and the cost counts the field to day 100
    assert plan["time"] == pytest.approx(
        np.log(A * B * (C2 - C1) / C3) / B, rel=1e-10
    )
    assert plan["cost"] == pytest.approx(
        _cost(plan["time"], life=100.0), rel=1e-12
    )
    # testing that would still pay at the end of the life stops there
    short = model.release_time(0.5, C1, C2, life=60.0)
    assert short["time"] == 60.0 and short["binding"] == "life"
    assert short["band"][1] == 60.0


def test_testing_that_never_pays_releases_at_once():
    model = GoelOkumoto.from_params([A, B])
    # a day of testing costs more than the faults it could find: the
    # intensity starts at a b = 5, under 5000 / 490
    plan = model.release_time(5000.0, C1, C2)
    assert plan["time"] == 0.0 and plan["cost"] == C2 * A
    assert plan["band"][0] == 0.0


def test_s_shaped_curve_releases_on_the_way_down():
    model = YamadaDelayedS.from_params([A, 0.1])
    plan = model.release_time(C3, C1, C2)
    assert plan["time"] > 10.0  # past the peak of the intensity
    assert float(model.iif(plan["time"])) == pytest.approx(
        C3 / (C2 - C1), rel=1e-10
    )
    days = np.linspace(0.0, 300.0, 3001)
    cost = C1 * model.cif(days) + C2 * (A - model.cif(days)) + C3 * days
    assert np.all(cost >= plan["cost"] - 1e-9)


def test_reliability_target_delays_the_release():
    model = GoelOkumoto.from_params([A, B])
    free = model.release_time(C3, C1, C2)
    plan = model.release_time(C3, C1, C2, reliability=0.95, mission=10.0)
    met = model.time_to_reliability(0.95, 10.0)
    assert met > free["time"]
    assert plan["time"] == met and plan["binding"] == "reliability"
    assert plan["reliability_time"] == met
    assert plan["cost_time"] == free["time"]
    assert plan["cost"] == pytest.approx(_cost(met), rel=1e-12)
    assert plan["cost"] > plan["least_cost"] == free["cost"]
    # a target met before the cheapest day changes nothing
    easy = model.release_time(C3, C1, C2, reliability=0.2, mission=10.0)
    assert easy["time"] == free["time"] and easy["binding"] == "cost"
    assert easy["reliability_time"] < free["time"]


def test_infinite_failure_model_needs_a_life():
    model = MusaOkumoto.from_params([2.0, 0.05])
    with pytest.raises(ValueError) as error:
        model.release_time(C3, C1, C2)
    assert "expects failures without limit" in str(error.value)
    assert "give life=" in str(error.value)
    plan = model.release_time(C3, C1, C2, life=2000.0)
    # lambda0 / (1 + lambda0 theta T) = c3 / (c2 - c1)
    level = C3 / (C2 - C1)
    assert plan["time"] == pytest.approx(
        (2.0 / level - 1) / (2.0 * 0.05), rel=1e-10
    )


@pytest.mark.parametrize(
    "model, how",
    [
        (HPP.from_params([0.3]), "does not fall"),
        (CrowAMSAA.from_params([10.0, 1.5]), "is rising"),
    ],
)
def test_no_growth_has_no_release_time(model, how):
    with pytest.raises(ValueError) as error:
        model.release_time(C3, C1, C2, life=500.0)
    message = str(error.value)
    assert f"The fitted {model.dist.name} model has no best release" in message
    assert how in message and "0.102041" in message
    assert "more testing always pays" in message
    assert "trend_test()" in message


def test_costs_are_checked():
    model = GoelOkumoto.from_params([A, B])
    with pytest.raises(ValueError) as error:
        model.release_time(C3, 500.0, 10.0)
    assert "field_cost (10) is not above fix_cost (500)" in str(error.value)
    assert "no best release time" in str(error.value)
    with pytest.raises(ValueError, match="field_cost .* not above"):
        model.release_time(C3, 10.0, 10.0)
    with pytest.raises(ValueError, match="test_cost must be a number above 0"):
        model.release_time(0.0, C1, C2)
    with pytest.raises(
        ValueError, match="fix_cost must be a cost of at least"
    ):
        model.release_time(C3, -1.0, C2)
    with pytest.raises(ValueError, match="reliability and mission go"):
        model.release_time(C3, C1, C2, reliability=0.9)
    with pytest.raises(ValueError, match="reliability and mission go"):
        model.release_time(C3, C1, C2, mission=5.0)
    with pytest.raises(ValueError, match="reliability must be"):
        model.release_time(C3, C1, C2, reliability=1.5, mission=5.0)
    with pytest.raises(ValueError, match="within must be"):
        model.release_time(C3, C1, C2, within=0.0)
    with pytest.raises(ValueError, match="life must be"):
        model.release_time(C3, C1, C2, life=-5.0)


def test_wald_bounds_are_the_delta_method_on_the_log(fitted):
    a, b = fitted.params
    best = np.log(a * b * (C2 - C1) / C3) / b
    assert fitted.release_time(C3, C1, C2)["time"] == pytest.approx(best)
    # d T / d a = 1 / (a b); d T / d b = (1 - b T) / b**2
    grad = np.array([1 / (a * b), (1 - b * best) / b**2])
    se = np.sqrt(grad @ fitted.covariance() @ grad)
    z = 1.959963984540054
    expected = best * np.exp(np.array([-1.0, 1.0]) * z * se / best)
    np.testing.assert_allclose(
        fitted.release_time_cb(C3, C1, C2), expected, rtol=1e-6
    )
    lower = fitted.release_time_cb(C3, C1, C2, alpha_ci=0.025, bound="lower")
    assert lower == pytest.approx(expected[0], rel=1e-6)
    wide = fitted.release_time_cb(C3, C1, C2, alpha_ci=0.01)
    assert wide[0] < expected[0] and wide[1] > expected[1]


def test_bootstrap_bounds_are_seeded_and_contain_the_estimate(fitted):
    kw = {"method": "bootstrap", "n_boot": 40}
    first = fitted.release_time_cb(C3, C1, C2, random_state=3, **kw)
    again = fitted.release_time_cb(C3, C1, C2, random_state=3, **kw)
    np.testing.assert_array_equal(first, again)
    best = fitted.release_time(C3, C1, C2)["time"]
    assert first[0] < best < first[1]
    upper = fitted.release_time_cb(
        C3, C1, C2, random_state=3, bound="upper", alpha_ci=0.025, **kw
    )
    assert upper == first[1]
    with pytest.raises(ValueError, match="n_boot"):
        fitted.release_time_cb(C3, C1, C2, method="bootstrap", n_boot=5)
    with pytest.raises(ValueError, match="method"):
        fitted.release_time_cb(C3, C1, C2, method="lr")


def test_bootstrap_needs_failure_times():
    weeks = np.column_stack([np.arange(8.0), np.arange(1.0, 9.0)])
    model = GoelOkumoto.fit(weeks, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1])
    assert model.release_time_cb(0.5, 0.1, 5.0).shape == (2,)
    with pytest.raises(ValueError, match="exact event times"):
        model.release_time_cb(0.5, 0.1, 5.0, method="bootstrap")
    given = GoelOkumoto.from_params([A, B])
    with pytest.raises(ValueError):
        given.release_time_cb(C3, C1, C2)
    with pytest.raises(ValueError):
        given.release_time_cb(C3, C1, C2, method="bootstrap")


def test_release_time_does_not_depend_on_the_units():
    # days to hours (the cost of testing is per unit of time), and the
    # currency
    days = GoelOkumoto.from_params([A, B]).release_time(C3, C1, C2)
    hours = GoelOkumoto.from_params([A, B / 24.0]).release_time(
        C3 / 24.0, C1, C2
    )
    assert hours["time"] == pytest.approx(24.0 * days["time"], rel=1e-9)
    assert hours["cost"] == pytest.approx(days["cost"], rel=1e-9)
    cents = GoelOkumoto.from_params([A, B]).release_time(
        100 * C3, 100 * C1, 100 * C2
    )
    assert cents["time"] == pytest.approx(days["time"], rel=1e-12)
    assert cents["band"] == pytest.approx(days["band"], rel=1e-9)
