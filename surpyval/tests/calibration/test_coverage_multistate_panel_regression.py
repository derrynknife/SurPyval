"""Recovery and coverage of ``ParametricPanelMultiStateRegression``:
Weibull proportional hazards fitted to states seen only at visits.

The truth: an illness-death model without recovery, Weibull baselines
(healthy -> ill scale 6 shape 1.5, healthy -> dead scale 25 shape 1.2,
ill -> dead scale 8 shape 2) on the time since the start
(``clock="forward"``) and, in a second design, on the time since
entering the state (``clock="reset"``). ``male`` (0 or 1, fixed for a
unit) has effects 0.5, 0.3 and -0.4 on the three transitions; ``w`` (0
or 1, drawn afresh at each visit and holding until the next) has 0.6 on
healthy -> ill and 0 on the others. ``N = 300`` units start healthy and
are seen at 0 and every 1 time unit until a uniform (5, 10) end; deaths
are timed exactly, from a state not seen (``exact_entry``).

Checked over 200 replicates per design: the bias of each of the twelve
parameters against its spread and the reported standard errors against
that spread (``check_bias``); the 90% and 95% Wald bounds, on the log of
each scale and shape and on each coefficient itself, from the joint
fit's covariance. The seed, replicates, sizes and slack (0.01, the
suite's rule) were fixed before any run.
"""

import os
import sys
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from surpyval.multistate import (
    ParametricPanelMultiStateRegression,
    TransitionMatrix,
)
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    process_pool,
)
from surpyval.tests.multistate._inspected import (
    NAMES,
    TRANSITIONS,
    visits_covariates,
)

N = 300
REPS = 200
SEED = 70741
PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}
BETAS = {
    ("healthy", "ill"): (0.5, 0.6),
    ("healthy", "dead"): (0.3, 0.0),
    ("ill", "dead"): (-0.4, 0.0),
}
CLOCKS = ("forward", "reset")
TRUTH = np.concatenate([[*PARS[p], *BETAS[p]] for p in TRANSITIONS])
# Scales and shapes are positive (log-scale bounds); coefficients are not.
POSITIVE = np.tile([True, True, False, False], 3)


def fit_one(clock, seed, n=N):
    ids, times, states, Z = visits_covariates(n, seed, PARS, BETAS, clock)
    s = np.array([NAMES[int(v)] for v in states], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricPanelMultiStateRegression.fit(
            times,
            Z,
            s,
            ids,
            obstype=np.where(s == "dead", "exact_entry", "panel"),
            transitions=TransitionMatrix(TRANSITIONS),
            clock=clock,
        )
    return model.params, model.standard_errors()


def _one(args):
    clock, r = args
    return fit_one(clock, [SEED, CLOCKS.index(clock), r])


def bounds(est, se, level):
    z = norm.ppf(0.5 + level / 2)
    with np.errstate(all="ignore"):
        w = z * se / np.abs(est)
        lower = np.where(POSITIVE, est * np.exp(-w), est - z * se)
        upper = np.where(POSITIVE, est * np.exp(w), est + z * se)
    return lower, upper


#: Misses on the 0.30 release run (x86-64 Linux), released as known
#: limitations (the maintainer's decision, 28 Nov 2026), to be fixed: the
#: standard error of the healthy -> dead scale (parameter 4) overstates
#: its spread over the replicates.
_RELEASE_MISSES = {
    "forward": "se / sd of the healthy -> dead scale 1.7581 (rule 1 +- "
    "0.25) on x86-64 Linux",
    "reset": "se / sd of the healthy -> dead scale 1.2652 (rule 1 +- "
    "0.25) on x86-64 Linux",
}


@pytest.mark.parametrize(
    "clock",
    [
        pytest.param(
            clock,
            marks=pytest.mark.xfail(
                sys.platform.startswith("linux"),
                strict=True,
                reason=_RELEASE_MISSES[clock],
            ),
        )
        for clock in CLOCKS
    ],
)
def test_recovery_and_coverage(clock):
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_one, [(clock, r) for r in range(REPS)]))
    est = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    label = f"ParametricPanelMultiStateRegression, Weibull PH ({clock})"
    check_bias(est, TRUTH, label, standard_errors=se)
    for level in (0.90, 0.95):
        lower, upper = bounds(est, se, level)
        check_coverage(
            lower,
            upper,
            TRUTH,
            level,
            f"{label} Wald bounds at {level:.0%}",
        )
