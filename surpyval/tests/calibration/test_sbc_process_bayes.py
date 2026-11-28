"""Simulation-based calibration of the Bayesian Wiener and gamma process
fits (``WienerProcess.fit_bayes``, ``GammaProcess.fit_bayes``): the
coverage of their credible intervals on a parameter and on the B10 life.

In words: draw the true parameters from the priors, simulate a fleet's
readings from them, fit, and ask how often the credible interval holds
the truth. When the posterior is computed correctly the truth's
posterior probability (its "PIT" value) is uniform, so the central 50%
and 90% intervals hold it in 50% and 90% of repetitions. The rules are
the validation kit's (``surpyval/tests/bayes/kit.py``): coverage within
three binomial standard errors, uniformity by Kolmogorov-Smirnov at
0.001, no fit refused.

The scenarios and their seeds were fixed before the first run (26 Nov
2026): 4 units read every 10 to 60 (24 increments), threshold 100, each
unit starting at 0 (``y0=0``). Wiener: ``mu`` Normal(0.5, 0.1),
``sigma`` log-normal about 1 (sigma 0.3); checked ``mu`` and the B10
life. Gamma: ``alpha`` log-normal about 2 (sigma 0.3), ``beta``
log-normal about 4 (sigma 0.3); checked ``alpha`` and the B10 life. The
grid (exact). 200 repetitions each, seeds 64000 and 65000.

The readings are simulated from the processes' definitions (independent
normal or gamma increments), not with the package's simulators.
"""

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.degradation import GammaProcess, WienerProcess
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 200
SEEDS = {"wiener": 64000, "gamma": 65000}
UNITS, STEP, END, THRESHOLD = 4, 10.0, 60.0, 100.0


def _priors(kind):
    if kind == "wiener":
        return {
            "mu": sp.Normal.from_params([0.5, 0.1]),
            "sigma": sp.LogNormal.from_params([0.0, 0.3]),
        }
    return {
        "alpha": sp.LogNormal.from_params([np.log(2.0), 0.3]),
        "beta": sp.LogNormal.from_params([np.log(4.0), 0.3]),
    }


def _truth(kind, rng):
    if kind == "wiener":
        return (
            float(0.5 + 0.1 * rng.standard_normal()),
            float(np.exp(0.3 * rng.standard_normal())),
        )
    return (
        float(2.0 * np.exp(0.3 * rng.standard_normal())),
        float(4.0 * np.exp(0.3 * rng.standard_normal())),
    )


def _readings(kind, truth, rng):
    t = np.arange(0.0, END + STEP, STEP)
    dt = np.diff(t)
    a, b = truth
    xs, ys, ids = [], [], []
    for u in range(UNITS):
        if kind == "wiener":
            steps = rng.normal(a * dt, b * np.sqrt(dt))
        else:
            steps = rng.gamma(a * dt, 1.0 / b)
        xs.append(t)
        ys.append(np.r_[0.0, np.cumsum(steps)])
        ids.append(np.full(t.size, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def run(kind, reps=REPS, seed=None):
    """The PIT values of the checked quantities, by name."""
    seed = SEEDS[kind] if seed is None else seed
    fitter = WienerProcess if kind == "wiener" else GammaProcess
    name = "mu" if kind == "wiener" else "alpha"
    model_cls = (
        sp.degradation.WienerProcessModel
        if kind == "wiener"
        else sp.degradation.GammaProcessModel
    )
    priors = _priors(kind)
    pits: dict = {name: [], "b10": []}
    refused = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = _truth(kind, rng)
        x, y, i = _readings(kind, truth, rng)
        try:
            model = fitter.fit_bayes(
                x, y, i, threshold=THRESHOLD, y0=0.0, priors=priors
            )
        except ValueError as err:
            refused.append((r, str(err)[:160]))
            continue
        w = model.posterior.weights
        col = model.parameter_names.index(name)
        pits[name].append(
            float(w @ (model.posterior.points[:, col] < truth[0]))
        )
        exact = model_cls(*truth, THRESHOLD)
        b10 = model._life(None, None).quantile(0.1)
        pits["b10"].append(float(w @ (b10 < exact.qf(0.1))))
    return {k: np.asarray(v) for k, v in pits.items()}, refused


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_parameter_and_b10_intervals_hold_their_level(kind):
    pits, refused = run(kind)
    assert not refused, refused[:3]
    problems = []
    for name, u in pits.items():
        for level in (0.5, 0.9):
            got = float(np.mean(np.abs(u - 0.5) <= level / 2))
            band = kit.SE_MULTIPLE * np.sqrt(level * (1 - level) / len(u))
            if abs(got - level) > band:
                problems.append(f"{name}: {level:.0%} coverage {got:.3f}")
        p = stats.kstest(u, "uniform").pvalue
        if p < kit.KS_LEVEL:
            problems.append(f"{name}: PIT not uniform (KS p {p:.2g})")
    assert not problems, problems
