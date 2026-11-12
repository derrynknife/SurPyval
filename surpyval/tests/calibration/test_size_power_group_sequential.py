"""Size and power of group-sequential log-rank designs, by simulation.

The designs' boundaries, power and expected events come from the
normal approximation to the log-rank statistic with independent
increments (``logrank_group_sequential``); these studies run the designs
as a study would, on simulated survival data analysed with
``surpyval.logrank`` at each look, and check that they keep their
promises. Units enter uniformly over a year (calendar time 0 to 1) and
fail at exponential times, hazard 1 in the second group and the hazard
ratio times it in the first; look ``k`` is at the calendar time of the
design's ``events[k]``-th event, every unit then entered censored at its
follow-up to that time. The test is two-sided, so ``|Z|`` is the square
root of the chi-squared statistic.

The checks follow ``test_size_power.py``: the level has the slack 0.01 of
an asymptotic test (0.005 for the level spent by the second look, whose
target is 0.012); the power the 0.03 of Schoenfeld's approximation; the
expected events at the stop, under the alternative, 2%. Seeds, sizes and
tolerances were fixed before the first run.
"""

import numpy as np

import surpyval as sp
from surpyval.test_planning import logrank_group_sequential
from surpyval.tests.calibration._montecarlo import check_rate


def _run(design, hazard_ratio, n, rng):
    """One simulated study: the look it stopped at (1-based, 0 if it
    never crossed) and the events at the stop."""
    group = np.repeat([0, 1], n)
    entry = rng.uniform(0.0, 1.0, 2 * n)
    life = rng.exponential(1.0 / np.where(group == 0, hazard_ratio, 1.0))
    calendar = np.sort(entry + life)
    for k, events in enumerate(design.events):
        cut = calendar[events - 1]
        inside = entry <= cut
        follow = cut - entry[inside]
        x = np.minimum(life[inside], follow)
        c = (life[inside] > follow).astype(int)
        z = np.sqrt(sp.logrank(x, group[inside], c=c).statistic)
        if z >= design.efficacy_z[k]:
            return k + 1, events
    return 0, design.events[-1]


def _study(design, hazard_ratio, n, reps, seed):
    rng = np.random.default_rng(seed)
    looks = np.zeros(reps, dtype=int)
    events = np.zeros(reps)
    for r in range(reps):
        looks[r], events[r] = _run(design, hazard_ratio, n, rng)
    return looks, events


def test_obrien_fleming_spending_size():
    design = logrank_group_sequential(0.7, 3)
    looks, _ = _study(design, 1.0, 300, 20000, 601)
    check_rate(
        np.sum((looks >= 1) & (looks <= 2)),
        looks.size,
        design.alpha_spent[1],
        "group-sequential log-rank, level spent by look 2",
        slack=0.005,
    )
    check_rate(
        np.sum(looks > 0),
        looks.size,
        design.alpha_ci,
        "group-sequential log-rank (O'Brien-Fleming spending) size",
    )


def test_obrien_fleming_spending_power():
    design = logrank_group_sequential(0.7, 3)
    looks, events = _study(design, 0.7, 300, 10000, 602)
    check_rate(
        np.sum(looks > 0),
        looks.size,
        design.power,
        "group-sequential log-rank (O'Brien-Fleming spending) power",
        slack=0.03,
    )
    expected = design.expected_information["alternative"] * design.max_events
    mean = float(events.mean())
    print(f"expected events at the stop: {mean:.1f} (design {expected:.1f})")
    assert abs(mean - expected) <= 0.02 * expected


def test_pocock_size():
    design = logrank_group_sequential(0.7, 4, efficacy="pocock")
    looks, _ = _study(design, 1.0, 300, 20000, 603)
    check_rate(
        np.sum(looks > 0),
        looks.size,
        design.alpha_ci,
        "group-sequential log-rank (Pocock) size",
    )
