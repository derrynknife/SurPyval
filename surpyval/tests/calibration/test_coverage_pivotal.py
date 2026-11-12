"""Coverage of the pivotal bounds of a probability-plotting (MPP) fit,
with the Wald bounds of the maximum-likelihood fit to the same samples
alongside.

The truth: a Weibull with scale 100 and shape 1.5. Samples of n = 5, 10
and 20, complete or censored after the ``r``-th failure (type II, r =
3, 6 and 12, the rest suspended at that failure), each fitted with
``Weibull.fit(x, c=c, how="MPP")`` (the defaults: Nelson-Aalen
positions, y on x). Checked over 1000 replicates each, at the 90%
two-sided level: ``param_cb`` on the shape and the scale,
``quantile_cb`` on the B10 life, ``cb`` on the survival at t = 100 (the
true scale, sf = 0.368), and the 90% lower bound on the B10 life,
all with ``method="pivotal"`` and ``n_sim = 999`` (so (n_sim + 1) a is a
whole number at both levels, and the bounds are exact over the
simulation too). For complete and type II samples the pivotal argument
is exact, so every cell should be at 0.90 up to the Monte Carlo error;
the suite's slack of 0.01 is kept.

The Wald bounds (``cb``, ``quantile_cb`` and ``param_cb`` of
``Weibull.fit(x, c=c)``) are printed for comparison, not checked: they
are asymptotic, and short at these sizes. The seed, replicates, sizes,
levels and slack were fixed before any run.
"""

import os
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

REPS = 1000
SEED = 13112026
ALPHA, BETA = 100.0, 1.5
N_SIM = 999
LEVEL = 0.1
T = 100.0
B10 = ALPHA * (-np.log(0.9)) ** (1.0 / BETA)
TRUTH = np.array([BETA, ALPHA, B10, np.exp(-((T / ALPHA) ** BETA))])
DESIGNS = [(5, None), (5, 3), (10, None), (10, 6), (20, None), (20, 12)]


def _sample(n, r, rep):
    rng = np.random.default_rng([SEED, n, r or 0, rep])
    x = np.sort(ALPHA * rng.weibull(BETA, n))
    c = np.zeros(n, dtype=int)
    if r is not None:
        x[r:] = x[r - 1]
        c[r:] = 1
    return x, c


def _bounds(model, method, **kw):
    out = [
        model.param_cb("beta", alpha_ci=LEVEL, method=method, **kw),
        model.param_cb("alpha", alpha_ci=LEVEL, method=method, **kw),
        model.quantile_cb(0.1, alpha_ci=LEVEL, method=method, **kw),
        model.cb(T, alpha_ci=LEVEL, method=method, **kw),
    ]
    lower = model.quantile_cb(
        0.1, alpha_ci=LEVEL, bound="lower", method=method, **kw
    )
    return np.array(out), float(np.ravel(lower)[0])


def _one(job):
    n, r, rep = job
    x, c = _sample(n, r, rep)
    nan = (np.full((4, 2), np.nan), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mpp = sp.Weibull.fit(x, c=c, how="MPP")
        pivotal = _bounds(mpp, "pivotal", n_sim=N_SIM, random_state=rep)
        try:
            wald = _bounds(sp.Weibull.fit(x, c=c), "wald")
        except ValueError:
            wald = nan
    return pivotal, wald


@pytest.mark.parametrize("n, r", DESIGNS)
def test_pivotal_bounds_cover(n, r):
    jobs = [(n, r, rep) for rep in range(REPS)]
    with process_pool(max_workers=min(os.cpu_count() or 1, 2)) as pool:
        out = list(pool.map(_one, jobs, chunksize=50))
    label = f"n = {n}, " + ("complete" if r is None else f"type II r = {r}")
    for k, name in enumerate(("pivotal", "Wald (MLE)")):
        two = np.array([o[k][0] for o in out])
        lower = np.array([o[k][1] for o in out])
        if name == "pivotal":
            check_coverage(
                two[:, :, 0],
                two[:, :, 1],
                TRUTH,
                1 - LEVEL,
                f"pivotal [beta, alpha, B10, sf(100)], {label}",
            )
            check_coverage(
                lower,
                np.full_like(lower, np.inf),
                B10,
                1 - LEVEL,
                f"pivotal lower bound on B10, {label}",
            )
        else:
            hit = (two[:, :, 0] <= TRUTH) & (TRUTH <= two[:, :, 1])
            print(
                f"Wald (MLE) [beta, alpha, B10, sf(100)], {label}: "
                f"{np.round(hit.mean(axis=0), 4)}; lower bound on B10: "
                f"{np.mean(lower <= B10):.4f} (not checked)"
            )
