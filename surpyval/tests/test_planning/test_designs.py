"""One function per design: the quantity left None is solved for.

Each design function agrees, to the last digit, with the function of one
unknown it replaces (still reachable, deprecated), solves round trips,
and refuses to guess which quantity is wanted.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval
import surpyval.test_planning as tp
from surpyval.test_planning import (
    PowerCalculation,
    cox_design,
    cure_logrank_design,
    equivalence_design,
    logrank_design,
    noninferiority_design,
    rate_ratio_design,
    rmst_design,
)

LATENCY = surpyval.Weibull.from_params([2.0, 1.2])
CURVES = (
    surpyval.Exponential.from_params([1 / 10]),
    surpyval.Exponential.from_params([1 / 6]),
)
CURE = dict(accrual_time=1.0, follow_up=3.0)
RMST = dict(accrual_time=2.0, follow_up=4.0, dropout_rate=0.05)
RATE = dict(rate=30 / 10000, exposure_before=10000)


def old(name):
    """The deprecated function ``name``, which must warn as it is reached
    and name the design function that replaces it."""
    with pytest.warns(FutureWarning, match=name) as caught:
        function = getattr(tp, name)
    assert "_design(" in str(caught[0].message)
    assert "v0.30" in str(caught[0].message)
    return function


# Each old call and the new call solving for the same unknown.
SAME = [
    (
        lambda: old("logrank_events")(0.5),
        lambda: logrank_design(0.5, power=0.8).events,
    ),
    (
        lambda: old("logrank_events")(0.6, 0.9, ratio=2.0, method="freedman"),
        lambda: logrank_design(
            0.6, power=0.9, ratio=2.0, method="freedman"
        ).events,
    ),
    (
        lambda: old("logrank_sample_size")(
            0.7, (0.25, 0.35), alternative="one-sided"
        ),
        lambda: logrank_design(
            0.7,
            power=0.8,
            event_probability=(0.25, 0.35),
            alternative="one-sided",
        ).n,
    ),
    (
        lambda: old("logrank_power")(0.7, (200, 300), (0.25, 0.35)),
        lambda: logrank_design(
            0.7, n=(200, 300), event_probability=(0.25, 0.35)
        ).power,
    ),
    (
        lambda: old("cox_events")(1.5, sd=1.0, r_squared=0.3),
        lambda: cox_design(1.5, power=0.8, sd=1.0, r_squared=0.3).events,
    ),
    (
        lambda: old("cox_sample_size")(1.5, 0.4, 0.9, proportion=0.3),
        lambda: cox_design(
            1.5, power=0.9, event_probability=0.4, proportion=0.3
        ).n,
    ),
    (
        lambda: old("cox_power")(1.5, 120, 0.4, sd=0.7),
        lambda: cox_design(1.5, n=120, event_probability=0.4, sd=0.7).power,
    ),
    (
        lambda: old("cox_detectable")(200, 0.3, sd=1.0, direction="decrease"),
        lambda: cox_design(
            power=0.8,
            n=200,
            event_probability=0.3,
            sd=1.0,
            direction="decrease",
        ).value,
    ),
    (
        lambda: old("noninferiority_events")(1.3, hazard_ratio=0.9),
        lambda: noninferiority_design(1.3, power=0.8, hazard_ratio=0.9).events,
    ),
    (
        lambda: old("noninferiority_sample_size")(1.3, (0.4, 0.5), ratio=2),
        lambda: noninferiority_design(
            1.3, power=0.8, event_probability=(0.4, 0.5), ratio=2
        ).n,
    ),
    (
        lambda: old("noninferiority_power")(1.3, (571, 400), 0.4),
        lambda: noninferiority_design(
            1.3, n=(571, 400), event_probability=0.4
        ).power,
    ),
    (
        lambda: old("equivalence_events")((0.8, 1.3), alpha_ci=0.1),
        lambda: equivalence_design((0.8, 1.3), power=0.8, alpha_ci=0.1).events,
    ),
    (
        lambda: old("equivalence_sample_size")(1.25, 0.5, alpha_ci=0.1),
        lambda: equivalence_design(
            1.25, power=0.8, event_probability=0.5, alpha_ci=0.1
        ).n,
    ),
    (
        lambda: old("equivalence_power")(1.25, (688, 600), 0.5),
        lambda: equivalence_design(
            1.25, n=(688, 600), event_probability=0.5
        ).power,
    ),
    (
        lambda: old("rmst_sample_size")(CURVES, 5.0, **RMST),
        lambda: rmst_design(CURVES, 5.0, power=0.8, **RMST).n,
    ),
    (
        lambda: old("rmst_power")(CURVES, 5.0, (100, 120), **RMST),
        lambda: rmst_design(CURVES, 5.0, n=(100, 120), **RMST).power,
    ),
    (
        lambda: old("cure_logrank_sample_size")((0.45, 0.30), LATENCY, **CURE),
        lambda: cure_logrank_design(
            (0.45, 0.30), LATENCY, power=0.8, **CURE
        ).n,
    ),
    (
        lambda: old("cure_logrank_events")((0.45, 0.30), LATENCY, **CURE),
        lambda: cure_logrank_design(
            (0.45, 0.30), LATENCY, power=0.8, **CURE
        ).events,
    ),
    (
        lambda: old("cure_logrank_power")(
            (0.45, 0.30), LATENCY, (200, 250), **CURE
        ),
        lambda: cure_logrank_design(
            (0.45, 0.30), LATENCY, n=(200, 250), **CURE
        ).power,
    ),
    (
        lambda: old("rate_ratio_power")(0.5, 30 / 10000, 10000, 5000),
        lambda: rate_ratio_design(0.5, exposure_after=5000, **RATE).power,
    ),
    (
        lambda: old("rate_ratio_detectable")(30 / 10000, 10000, 20000),
        lambda: rate_ratio_design(
            exposure_after=20000, power=0.8, **RATE
        ).value,
    ),
    (
        lambda: old("rate_ratio_exposure")(0.5, 30 / 10000, 10000),
        lambda: rate_ratio_design(0.5, power=0.8, **RATE).value,
    ),
]


@pytest.mark.parametrize("before, after", SAME)
def test_each_old_function_is_the_design_solving_for_its_unknown(
    before, after
):
    assert before() == after()


def test_every_old_function_is_compared():
    compared = {
        name
        for name in tp._REPLACED
        if any(name in before.__code__.co_consts for before, _ in SAME)
    }
    assert compared == set(tp._REPLACED)


def test_arrays_are_answered_element_by_element():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        before = tp.logrank_events([0.5, 0.7], [0.8, 0.9])
        rates = tp.rate_ratio_power([0.3, 0.7], 30 / 10000, 10000, 20000)
    plan = logrank_design([0.5, 0.7], power=[0.8, 0.9])
    np.testing.assert_array_equal(plan.events, before)
    np.testing.assert_array_equal(
        rate_ratio_design([0.3, 0.7], exposure_after=20000, **RATE).power,
        rates,
    )


def test_the_answer_says_what_was_solved_for():
    assert logrank_design(0.5, power=0.8).solved_for == "events"
    plan = logrank_design(0.7, power=0.8, event_probability=0.3)
    assert (plan.solved_for, plan.value) == ("n", plan.n)
    assert plan.events == logrank_design(0.7, power=0.8).events
    power = logrank_design(0.7, n=(200, 200), event_probability=0.3)
    assert power.solved_for == "power" and power.value == power.power
    smallest = cox_design(power=0.8, n=200, event_probability=0.3, sd=1.0)
    assert smallest.solved_for == "hazard_ratio"
    exposure = rate_ratio_design(0.5, power=0.8, **RATE)
    assert exposure.solved_for == "exposure_after"
    assert exposure.inputs["exposure_after"] == exposure.value
    assert rmst_design(CURVES, 5.0, power=0.8, **RMST).events is None


# --- Round trips: the size found, then the power at it ---------------------


def test_round_trip_logrank_and_margins():
    for design, first in (
        (logrank_design, 0.7),
        (noninferiority_design, 1.3),
        (equivalence_design, 1.25),
    ):
        events = design(first, power=0.8).events
        assert design(first, events=events).power >= 0.8
        assert design(first, events=events - 1).power < 0.8
        n = design(first, power=0.8, event_probability=0.4).n
        at_n = design(first, n=n, event_probability=0.4).power
        assert 0.8 <= at_n < 0.8 + 0.01


def test_round_trip_cox():
    kw = dict(sd=1.0, r_squared=0.2)
    events = cox_design(1.5, power=0.8, **kw).events
    assert cox_design(1.5, events=events, **kw).power >= 0.8
    assert cox_design(1.5, events=events - 1, **kw).power < 0.8
    n = cox_design(1.5, power=0.8, event_probability=0.3, **kw).n
    assert cox_design(1.5, n=n, event_probability=0.3, **kw).power >= 0.8
    ratio = cox_design(power=0.8, events=events, **kw).value
    assert cox_design(ratio, events=events, **kw).power == pytest.approx(0.8)


def test_round_trip_rmst_cure_and_rate_ratio():
    n = rmst_design(CURVES, 5.0, power=0.8, **RMST).n
    assert rmst_design(CURVES, 5.0, n=n, **RMST).power >= 0.8
    n = cure_logrank_design((0.45, 0.3), LATENCY, power=0.8, **CURE).n
    found = cure_logrank_design((0.45, 0.3), LATENCY, n=n, **CURE).power
    assert found >= 0.8
    exposure = rate_ratio_design(0.5, power=0.8, **RATE).value
    power = rate_ratio_design(0.5, exposure_after=exposure, **RATE).power
    assert power >= 0.8


# --- Refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    "call, words",
    [
        (lambda: logrank_design(0.5), "power, n or events were all left"),
        (
            lambda: logrank_design(0.5, power=0.8, events=50),
            "were all given",
        ),
        (
            lambda: logrank_design(0.5, n=(10, 10), events=5),
            "give n or events, not both",
        ),
        (lambda: logrank_design(0.5, n=(10, 10)), "n needs event_probability"),
        (
            lambda: logrank_design(
                0.5, n=(10, 20), event_probability=0.5, ratio=2.0
            ),
            "ratio is read from n",
        ),
        (lambda: cox_design(power=0.8, sd=1.0), "hazard_ratio, n or events"),
        (
            lambda: cox_design(1.5, power=0.8, sd=1.0, direction="decrease"),
            "direction",
        ),
        (lambda: noninferiority_design(1.3), "noninferiority_design"),
        (lambda: equivalence_design(1.25), "equivalence_design"),
        (lambda: rmst_design(CURVES, 5.0), "power, n were all left"),
        (
            lambda: cure_logrank_design(0.3, LATENCY, follow_up=3.0),
            "cure_logrank_design",
        ),
        (
            lambda: rate_ratio_design(0.5, **RATE),
            "exposure_after, power were all left",
        ),
        (
            lambda: rate_ratio_design(
                0.5, exposure_after=10, power=0.8, **RATE
            ),
            "nothing is left to solve for",
        ),
    ],
)
def test_refusals_name_the_arguments(call, words):
    with pytest.raises(ValueError, match=words):
        call()


def test_an_out_of_range_argument_is_refused_as_before():
    with pytest.raises(ValueError, match="power must be between 0 and 1"):
        logrank_design(0.5, power=1.5)
    with pytest.raises(ValueError, match="events must be a whole number"):
        logrank_design(0.5, events=2.5)


# --- The answer as a result ------------------------------------------------


def test_to_dict_is_strict_json_and_repr_says_what_was_solved():
    plan = cure_logrank_design((0.45, 0.30), LATENCY, power=0.8, **CURE)
    assert isinstance(plan, PowerCalculation)
    d = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    assert d["result"] == "PowerCalculation"
    assert d["solved_for"] == "n" and d["value"] == [247, 247]
    assert d["inputs"]["latency"] == LATENCY.to_dict()
    text = repr(plan)
    assert "Solved for" in text and "n = (247, 247)" in text
    assert "Weibull" in text
    infinite = rate_ratio_design(
        0.5, rate=12 / 10000, exposure_before=10000, power=0.8
    )
    d = json.loads(json.dumps(infinite.to_dict(), allow_nan=False))
    assert d["value"] is None
