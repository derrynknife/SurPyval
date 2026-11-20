"""Coverage of the bounds on a fitted growth model's answers:
``remaining_faults_cb``, ``next_failure_sf_cb``, ``time_to_intensity_cb``,
``time_to_reliability_cb`` and ``release_time_cb`` (its delta-method
bound).

Data: 10 systems tested to T = 60, each a Goel-Okumoto process with a =
30 faults and b = 0.05 (95% of the faults found by the end, about 285
failures in all), simulated from the definition
(``_montecarlo.simulate_nhpp``). 1000 data sets, seed 411, nominal 95%.

The bounds are Wald bounds; the slack is the suite's 0.01. The design is
a well-informed one on purpose: with few failures, or the curve's
levelling off not yet seen, Wald bounds on the faults left are known to
cover less than they say (the guide says so), and a study of that would
measure the approximation, not the code.
"""

import warnings

import numpy as np

from surpyval.recurrent import GoelOkumoto
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    simulate_nhpp,
)

A, B, T_END, SYSTEMS = 30.0, 0.05, 60.0, 10
MISSION, RATE, RELIABILITY = 5.0, 0.02, 0.9
# The costs of a unit of testing, of a fault fixed in testing and of one
# found in use (a release at t = 78.5)
TEST, FIX, FIELD = 5.0, 10.0, 180.0


def test_growth_answers_coverage():
    rng = np.random.default_rng(411)
    truth = {
        "remaining_faults_cb": A * np.exp(-B * T_END),
        "next_failure_sf_cb": np.exp(
            -A * np.exp(-B * T_END) * -np.expm1(-B * MISSION)
        ),
        "time_to_intensity_cb": np.log(A * B / RATE) / B,
        "time_to_reliability_cb": np.log(
            A * -np.expm1(-B * MISSION) / -np.log(RELIABILITY)
        )
        / B,
        # the cheapest release: log(a b (c2 - c1) / c3) / b
        "release_time_cb": np.log(A * B * (FIELD - FIX) / TEST) / B,
    }
    reps = 1000
    bounds = {name: np.full((reps, 2), np.nan) for name in truth}
    for r in range(reps):
        x, i, c = simulate_nhpp(
            rng,
            lambda t: -A * np.expm1(-B * t),
            lambda n: -np.log1p(-n / A) / B,
            SYSTEMS,
            T_END,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = GoelOkumoto.fit(x, i, c=c)
            except ValueError:
                # refused as showing no growth: its bounds stay nan,
                # which count as misses
                continue
            bounds["remaining_faults_cb"][r] = model.remaining_faults_cb(T_END)
            bounds["next_failure_sf_cb"][r] = model.next_failure_sf_cb(
                MISSION, age=[T_END]
            )[0]
            bounds["time_to_intensity_cb"][r] = model.time_to_intensity_cb(
                RATE
            )
            bounds["time_to_reliability_cb"][r] = model.time_to_reliability_cb(
                RELIABILITY, MISSION
            )
            bounds["release_time_cb"][r] = model.release_time_cb(
                TEST, FIX, FIELD
            )
    for name, value in truth.items():
        check_coverage(
            bounds[name][:, 0], bounds[name][:, 1], value, 0.95, name
        )
