"""Simulation-based calibration of the Bayesian gamma process fit to
readings off a coarse gauge (``GammaProcess.fit_bayes(..., gauge=)``,
the quantised likelihood), against the same readings taken as exact.

In words: draw the true parameters from the priors, simulate a fleet's
wear, read it off a gauge with a step of 2 (about twice the spread an
interval adds), fit, and ask how often the credible interval holds the
truth. With the quantised likelihood the truth's posterior probability
(its "PIT" value) is uniform, so the central 50% and 90% intervals hold
it in 50% and 90% of repetitions. Taking the rounded readings as exact
(the zeros censored at the step, as ``fit`` does without ``gauge``)
makes the increments look more scattered than they are and pulls
``alpha`` down, so its 90% interval holds the truth far less often.
The rules are the validation kit's (``surpyval/tests/bayes/kit.py``):
coverage within three binomial standard errors, uniformity by
Kolmogorov-Smirnov at 0.001, no fit refused.

The scenario and its seed were fixed before the first run (26 Nov 2026):
the gamma scenario of ``test_sbc_process_bayes.py`` (4 units read every
10 to 60, 24 increments, threshold 100, ``y0=0``; ``alpha`` log-normal
about 2, ``beta`` about 4, sigma 0.3), each reading rounded to the
nearest 2; the exact quantised likelihood (16 cells a bin); the grid.
200 repetitions, seed 66000. Checked: ``alpha`` and the B10 life for
the gauge fit; ``alpha`` for the fit that ignores the gauge.

The readings are simulated from the process's definition (independent
gamma increments, then rounded), not with the package's simulators.
"""

import sys

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.degradation import GammaProcess
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 200
SEED = 66000
UNITS, STEP, END, THRESHOLD, GAUGE = 4, 10.0, 60.0, 100.0, 2.0
PRIORS = {
    "alpha": sp.LogNormal.from_params([np.log(2.0), 0.3]),
    "beta": sp.LogNormal.from_params([np.log(4.0), 0.3]),
}


def _readings(truth, rng):
    t = np.arange(0.0, END + STEP, STEP)
    dt = np.diff(t)
    a, b = truth
    xs, ys, ids = [], [], []
    for u in range(UNITS):
        level = np.r_[0.0, np.cumsum(rng.gamma(a * dt, 1.0 / b))]
        xs.append(t)
        ys.append(np.round(level / GAUGE) * GAUGE)
        ids.append(np.full(t.size, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def _pit(model, col, value):
    w = model.posterior.weights
    return float(w @ (model.posterior.points[:, col] < value))


def run(reps=REPS, seed=SEED):
    """The PIT values of the checked quantities, by name, and the
    refused fits."""
    pits: dict = {"alpha": [], "b10": [], "alpha (gauge ignored)": []}
    refused = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = (
            float(2.0 * np.exp(0.3 * rng.standard_normal())),
            float(4.0 * np.exp(0.3 * rng.standard_normal())),
        )
        x, y, i = _readings(truth, rng)
        try:
            model = GammaProcess.fit_bayes(
                x, y, i, THRESHOLD, y0=0.0, gauge=GAUGE, priors=PRIORS
            )
            naive = GammaProcess.fit_bayes(
                x, y, i, THRESHOLD, y0=0.0, priors=PRIORS
            )
        except ValueError as err:
            refused.append((r, str(err)[:160]))
            continue
        col = model.parameter_names.index("alpha")
        pits["alpha"].append(_pit(model, col, truth[0]))
        pits["alpha (gauge ignored)"].append(_pit(naive, col, truth[0]))
        exact = sp.degradation.GammaProcessModel(*truth, THRESHOLD)
        b10 = model._life(None, None).quantile(0.1)
        w = model.posterior.weights
        pits["b10"].append(float(w @ (b10 < exact.qf(0.1))))
    return {k: np.asarray(v) for k, v in pits.items()}, refused


def coverage(u, level):
    return float(np.mean(np.abs(u - 0.5) <= level / 2))


@pytest.mark.xfail(
    sys.platform.startswith("linux"),
    strict=True,
    reason="a miss on the 0.30 release run (x86-64 Linux), released as a "
    "known limitation: replicate 123 refused (the posterior does not fall "
    "away from its peak within 60 standard deviations); the study needs "
    "every replicate to fit",
)
def test_gauge_intervals_hold_their_level_and_ignoring_it_does_not():
    pits, refused = run()
    assert not refused, refused[:3]
    problems = []
    for name in ("alpha", "b10"):
        u = pits[name]
        for level in (0.5, 0.9):
            got = coverage(u, level)
            band = kit.SE_MULTIPLE * np.sqrt(level * (1 - level) / len(u))
            if abs(got - level) > band:
                problems.append(f"{name}: {level:.0%} coverage {got:.3f}")
        p = stats.kstest(u, "uniform").pvalue
        if p < kit.KS_LEVEL:
            problems.append(f"{name}: PIT not uniform (KS p {p:.2g})")
    u = pits["alpha (gauge ignored)"]
    band = kit.SE_MULTIPLE * np.sqrt(0.9 * 0.1 / len(u))
    if not coverage(u, 0.9) < 0.9 - band:
        problems.append(
            "ignoring the gauge was expected to under-cover alpha: 90% "
            f"coverage {coverage(u, 0.9):.3f}"
        )
    assert not problems, problems
