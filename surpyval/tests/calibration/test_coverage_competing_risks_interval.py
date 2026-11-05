"""Coverage of the Wald bounds of ``ParametricCompetingRisks`` fitted to
failures found at periodic inspections (left- and interval-censored rows,
fitted jointly over the causes).

Data: n = 300 units, cause ``a`` Weibull(10, 2) and cause ``b``
Weibull(14, 1); each unit's follow-up ends at an independent Uniform(2,
30) time and it is inspected every 2 time units until then. A unit that
fails before its last inspection is recorded with its cause in the window
between the inspections around the failure (left censored at 2 for the
first); the others are right censored at their last inspection. 200
replicates, seed 20261107 (fixed before the first run).

The two-sided 95% bounds on each cause's cumulative incidence and on the
all-cause survival at t = 4, 8 and 15 are checked against the truth, with
the slack of the other coverage studies (0.01).
``scripts/studies/parametric_competing_risks_interval_coverage.py`` runs
the same design (and inspections every 5) over 500 replicates.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool
from surpyval.univariate.competing_risks import ParametricCompetingRisks

N = 300
REPS = 200
SEED = 20261107
GAP = 2.0
TIMES = np.array([4.0, 8.0, 15.0])
TRUE = {"a": (10.0, 2.0), "b": (14.0, 1.0)}


def _sample(rng):
    latent = np.column_stack(
        [alpha * rng.weibull(beta, size=N) for alpha, beta in TRUE.values()]
    )
    first = latent.min(axis=1)
    cause = np.where(latent[:, 0] <= latent[:, 1], "a", "b")
    last = GAP * np.floor(rng.uniform(2.0, 30.0, size=N) / GAP)
    found = first <= last
    hi = GAP * np.ceil(first / GAP)
    lo = hi - GAP
    x = np.column_stack(
        [
            np.where(found, np.where(lo > 0, lo, hi), last),
            np.where(found, hi, last),
        ]
    )
    c = np.where(found, np.where(lo > 0, 2, -1), 1)
    e = cause.astype(object)
    e[~found] = None
    return x, e, c


def _replicate(job):
    x, e, c = job
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model = ParametricCompetingRisks.fit(x, e, c=c, dist=sp.Weibull)
        return (
            model.cb(TIMES, "a"),
            model.cb(TIMES, "b"),
            model.cb(TIMES, on="sf"),
        )


def test_wald_coverage_with_inspection_data():
    rng = np.random.default_rng(SEED)
    jobs = [_sample(rng) for _ in range(REPS)]
    with process_pool(max_workers=os.cpu_count()) as pool:
        out = list(pool.map(_replicate, jobs))
    truth = ParametricCompetingRisks.from_fitted(
        {k: sp.Weibull.from_params(list(v)) for k, v in TRUE.items()}
    )
    targets = (truth.cif(TIMES, "a"), truth.cif(TIMES, "b"), truth.sf(TIMES))
    for i, (label, target) in enumerate(
        zip(("cif a", "cif b", "sf"), targets)
    ):
        band = np.array([o[i] for o in out])
        check_coverage(band[..., 0], band[..., 1], target, 0.95, label)
