"""The size of ``change_point_test``: how often it rejects "no change"
when there is none.

Data: one system tested to T = 60, a Goel-Okumoto process with a = 40
faults and b = 0.04 (about 36 failures), with no change. 120 data sets,
seed 5364 (data set ``r`` from ``[5364, r]``, its test's bootstrap from
``[5364, r, 1]``); each test with 19 simulated data sets, so its p-value
is a multiple of 0.05.

A data set on which Goel-Okumoto is refused (no growth) or the change
point cannot be estimated has no test, and is left out of the rate: the
count is printed. The rate is checked at 10% and at 5% by the suite's
rule (3 Monte Carlo standard errors of the nominal rate, and 0.01).

The design, the seed and the rule were fixed before any run. As it came
out: 100 of the 120 data sets tested; 8% rejected at 10%, 3% at 5%.
"""

import os
import warnings

import numpy as np

from surpyval.recurrent import ChangePoint, GoelOkumoto, change_point_test
from surpyval.tests.calibration._montecarlo import check_rate, process_pool

A, B, T_END = 40.0, 0.04, 60.0
REPS, N_BOOT, SEED = 120, 19, 5364


def _p_value(r):
    seed = np.random.default_rng([SEED, r])
    truth = GoelOkumoto.from_params([A, B])
    data = truth.time_terminated_simulation_data(T_END, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ChangePoint(GoelOkumoto)._fit_from_recurrent_data(data)
            out = change_point_test(
                model,
                n_boot=N_BOOT,
                random_state=np.random.default_rng([SEED, r, 1]),
            )
        except ValueError:
            return np.nan
    return out["p_value"]


def test_size_of_the_test_of_no_change():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        p = np.array(list(pool.map(_p_value, range(REPS))))
    tested = int(np.sum(np.isfinite(p)))
    print(f"change_point_test: {tested} of {REPS} data sets tested")
    assert tested >= REPS // 2
    for level in (0.10, 0.05):
        check_rate(
            int(np.sum(p <= level + 1e-12)),
            tested,
            level,
            f"change_point_test size at {level:.0%}",
        )
