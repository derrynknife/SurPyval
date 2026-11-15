"""Size of ``gray_test`` on interval-censored rows (Gray's test by multiple
imputation) when the groups are inspected on different schedules.

Design (fixed before the first run): two groups of 100 units with the same
cumulative incidences -- cause ``a`` exponential with mean 12.5, cause
``b`` Weibull(20, 1.5). Group 0 is inspected every 1 time unit, group 1
every 3, each unit from its own offset (Uniform(0, gap), to 0.1); each
unit's follow-up ends at an independent Uniform(10, 25) time. A failure
before the end is recorded with its cause in the window between the
inspections around it (from 0 before the first); the others are right
censored at their last inspection. 300 replicates, 10 imputations each,
seed 20261115. The rejection rate at the 5% level must lie in [0.02,
0.085] (about 2.5 binomial standard errors around 0.05).
"""

import os
import warnings

import numpy as np

from surpyval import gray_test
from surpyval.tests.calibration._montecarlo import process_pool

N_GROUP = 100
REPS = 300
M = 10
SEED = 20261115
GAPS = (1.0, 3.0)


def _group(rng, gap):
    a = rng.exponential(12.5, N_GROUP)
    b = 20.0 * rng.weibull(1.5, N_GROUP)
    t = np.minimum(a, b)
    cause = np.where(a <= b, "a", "b").astype(object)
    offset = np.round(rng.uniform(0, gap, N_GROUP), 1)
    end = rng.uniform(10.0, 25.0, N_GROUP)
    # Inspections at offset + j * gap (j >= 0) up to the end.
    last = offset + gap * np.floor((end - offset) / gap)
    found = t <= last
    j = np.ceil((t - offset) / gap)
    hi = np.where(t <= offset, offset, offset + gap * j)
    lo = np.where(t <= offset, 0.0, hi - gap)
    hi = np.where(hi <= 0, gap, hi)
    x = np.column_stack(
        [np.where(found, lo, last), np.where(found, hi, np.inf)]
    )
    e = np.where(found, cause, None)
    return x, e


def _replicate(seed):
    rng = np.random.default_rng(seed)
    parts = [_group(rng, g) for g in GAPS]
    x = np.vstack([p[0] for p in parts])
    e = np.concatenate([p[1] for p in parts])
    group = np.repeat([0, 1], N_GROUP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gray_test(
            x, e, group, event="a", n_imputations=M, random_state=seed
        )
    return res.p_value


def test_size_with_different_inspection_schedules():
    seeds = np.random.SeedSequence(SEED).generate_state(REPS)
    with process_pool(max_workers=os.cpu_count()) as pool:
        p = np.array(list(pool.map(_replicate, [int(s) for s in seeds])))
    size = float(np.mean(p < 0.05))
    assert 0.02 <= size <= 0.085, f"size {size:.3f} over {REPS} replicates"
