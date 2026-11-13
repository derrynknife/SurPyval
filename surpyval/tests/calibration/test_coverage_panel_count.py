"""Coverage of the panel-count models' bootstrap bounds when the counts are
not Poisson.

The truth: each unit has a binary covariate ``z`` (probability 0.5) and a
gamma frailty ``f`` with mean 1 and variance 0.5, and its events are a
Poisson process with mean function ``f exp(0.5 z) Lambda0(t)``,
``Lambda0(t) = 0.2 t^1.5``. Given ``z`` the counts are a mixed Poisson
process (over-dispersed, with counts in different intervals correlated),
so the Poisson working model of both likelihoods is wrong; the mean model
``E[N(t) | z] = exp(0.5 z) Lambda0(t)`` holds. ``n = 150`` units, each
followed to time 8.

Two inspection designs, both independent of the counts:

* ``scheduled``: visits due at times 1, 2, ..., 8, each kept with
  probability 0.6 (at least one kept), a finite set of inspection times;
* ``irregular``: 1 to 6 visits (equally likely) at uniform times on (0.5,
  8), every unit at its own times.

Checked, with 95% percentile bootstrap bounds (``n_boot = 200``) from 300
replicates:

* ``PanelCountMCF`` (``how="likelihood"``, the default) at t = 2, 4 and 6,
  the truth the population mean ``Lambda0(t) (1 + e^0.5) / 2``, under both
  designs;
* ``PanelCountRegression`` (``how="pseudo"``, the default, and
  ``"likelihood"``), the coefficient ``beta = 0.5``, under the irregular
  design.

With irregular times the non-parametric estimate at a fixed time converges
at the cube-root rate, where the bootstrap is not known to be valid
(Kosorok, 2008); that cell is checked like the others, and a shortfall
there is that known limit, reported with its numbers. The seed,
replicates, ``n_boot`` and slack (0.01, the suite's rule) were fixed before
any run.
"""

import os
import warnings

import numpy as np
import pytest

from surpyval.recurrent import PanelCountMCF, PanelCountRegression
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 150
REPS = 300
N_BOOT = 200
SEED = 20261113
BETA = 0.5
TIMES = np.array([2.0, 4.0, 6.0])


def _lambda0(t):
    return 0.2 * np.asarray(t, dtype=float) ** 1.5


def _sample(r, design):
    rng = np.random.default_rng([SEED, r, design == "scheduled"])
    x, n, i, z = [], [], [], []
    for k in range(N):
        zk = int(rng.random() < 0.5)
        frailty = rng.gamma(2.0, 0.5)
        if design == "scheduled":
            keep = rng.random(8) < 0.6
            if not keep.any():
                keep[rng.integers(8)] = True
            visits = np.arange(1.0, 9.0)[keep]
        else:
            visits = np.sort(rng.uniform(0.5, 8.0, rng.integers(1, 7)))
        means = np.diff(np.r_[0.0, _lambda0(visits)])
        counts = rng.poisson(frailty * np.exp(BETA * zk) * means)
        x += list(visits)
        n += list(counts)
        i += [k] * len(visits)
        z += [zk] * len(visits)
    return np.array(x), np.array(n, float), np.array(i), np.array(z, float)


def _mcf(args):
    r, design = args
    x, n, i, _ = _sample(r, design)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelCountMCF.fit(x, n, i=i)
        return model.mcf_cb(TIMES, n_boot=N_BOOT, random_state=r)


def _beta(args):
    r, how = args
    x, n, i, z = _sample(r, "irregular")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelCountRegression.fit(x, z, n, i=i, how=how)
        return model.param_cb("coef_0", n_boot=N_BOOT, random_state=r)


def _run(fn, tag):
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        return np.array(list(pool.map(fn, [(r, tag) for r in range(REPS)])))


@pytest.mark.parametrize("design", ["scheduled", "irregular"])
def test_mcf_bootstrap_coverage(design):
    out = _run(_mcf, design)
    truth = _lambda0(TIMES) * (1 + np.exp(BETA)) / 2
    check_coverage(
        out[:, :, 0],
        out[:, :, 1],
        truth,
        0.95,
        f"PanelCountMCF ({design}) bootstrap, t = 2, 4, 6",
    )


@pytest.mark.parametrize("how", ["pseudo", "likelihood"])
def test_regression_bootstrap_coverage(how):
    out = _run(_beta, how)
    check_coverage(
        out[:, :1],
        out[:, 1:],
        BETA,
        0.95,
        f"PanelCountRegression ({how}) bootstrap beta",
    )
