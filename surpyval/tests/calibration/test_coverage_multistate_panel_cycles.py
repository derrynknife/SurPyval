"""Recovery and coverage of ``ParametricPanelMultiState(clock="reset")``
with a way back: Weibull hazards of the time in the state, fitted to
states seen only at visits, at most ``max_moves = 3`` moves counted
between two visits.

The truth: an illness-death model with recovery, Weibull hazards of the
time in the current state (healthy -> ill scale 8 shape 1.3, healthy ->
dead scale 30 shape 1, ill -> healthy scale 4 shape 1.5, ill -> dead
scale 10 shape 1.2). ``N = 150`` units start healthy and are seen at 0
and every 1 time unit to 5; deaths are timed exactly, from a state not
seen (``exact_entry``).

Checked over 100 replicates: the bias of each of the eight parameters
against its spread and the reported standard errors against that spread
(``check_bias``); the 90% and 95% Wald bounds on the log of each
parameter. The truncation's own error is recorded, not checked: the
study is of the fit as offered (bound 3). The seed, replicates, sizes and
slack (0.01, the suite's rule) were fixed before any run. Each fit takes
minutes (the nested integrals of bound 3), so the study runs for hours.
"""

import os
import warnings

import numpy as np
from scipy.stats import norm

from surpyval.multistate import ParametricPanelMultiState, TransitionMatrix
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)
from surpyval.tests.multistate._inspected import NAMES, visits_recovery

N = 150
REPS = 100
SEED = 70751
PARS = {
    ("healthy", "ill"): (8.0, 1.3),
    ("healthy", "dead"): (30.0, 1.0),
    ("ill", "healthy"): (4.0, 1.5),
    ("ill", "dead"): (10.0, 1.2),
}
TM = TransitionMatrix(list(PARS))
TRUTH = np.concatenate([PARS[p] for p in TM.transitions])


def fit_one(seed, n=N):
    ids, times, states = visits_recovery(n, seed, PARS)
    s = np.array([NAMES[int(v)] for v in states], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricPanelMultiState.fit(
            times,
            s,
            ids,
            obstype=np.where(s == "dead", "exact_entry", "panel"),
            transitions=TM,
            clock="reset",
        )
    return model.params, model.standard_errors(), model.truncation_error


def _one(r):
    return fit_one([SEED, r])


def test_recovery_and_coverage():
    # SURPYVAL_CALIBRATION_POOL caps the processes (each fit holds over
    # 5 GB; the weekly run uses 2); the results do not depend on it.
    workers = int(
        os.environ.get("SURPYVAL_CALIBRATION_POOL")
        or min(os.cpu_count() or 1, 4)
    )
    with process_pool(max_workers=workers) as pool:
        out = list(pool.map(_one, range(REPS)))
    est = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    label = "ParametricPanelMultiState, Weibull hazards (reset, recovery)"
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
