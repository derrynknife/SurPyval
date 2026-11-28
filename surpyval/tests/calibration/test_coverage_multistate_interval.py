"""Recovery and coverage of the parametric multi-state model with
interval-censored moves.

The truth: an illness-death model without recovery, Weibull hazards
(healthy -> ill scale 6 shape 1.5, healthy -> dead scale 25 shape 1.2,
ill -> dead scale 8 shape 2), on the time since the start
(``clock="forward"``) and, in a second design, on the time since
entering the state (``clock="reset"``). ``N = 300`` units start healthy
and are seen at 0 and every 1 time unit until a uniform (5, 10) end;
deaths are timed exactly, so the onset of illness is interval censored
(between the last visit healthy and the first ill; about 190 of the 300
units).

Checked over 200 replicates per design: the bias of each of the six
parameters against its spread, and the reported standard errors against
that spread (``check_bias``); and the 90% and 95% Wald bounds on the log
of each parameter, from the joint fit's covariance. The seed,
replicates, sizes and slack (0.01, the suite's rule) were fixed before
any run.
"""

import os
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from surpyval.multistate import ParametricMultiState, TransitionMatrix
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)
from surpyval.tests.multistate._inspected import TRANSITIONS, simulate

N = 300
REPS = 200
SEED = 70611
PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}
CLOCKS = ("forward", "reset")
TRUTH = np.concatenate([PARS[p] for p in TRANSITIONS])


def _one(args):
    clock, r = args
    x, e, state, tl, i = simulate(
        N, [SEED, CLOCKS.index(clock), r], PARS, clock
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricMultiState.fit(
            x,
            e,
            state,
            tl=tl,
            i=i,
            transitions=TransitionMatrix(TRANSITIONS),
            clock=clock,
        )
    return model.params, model.standard_errors()


@pytest.mark.parametrize("clock", CLOCKS)
def test_recovery_and_coverage(clock):
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_one, [(clock, r) for r in range(REPS)]))
    est = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    label = f"ParametricMultiState, interval-censored onset ({clock})"
    check_bias(est, TRUTH, label, standard_errors=se)
    for level in (0.90, 0.95):
        z = norm.ppf(0.5 + level / 2)
        w = z * se / est
        check_coverage(
            est * np.exp(-w),
            est * np.exp(w),
            TRUTH,
            level,
            f"{label} log-scale Wald bounds at {level:.0%}",
        )
