"""Simulation-based calibration of the Bayesian parametric competing-risks
fit (``ParametricCompetingRisks.fit_bayes``): the coverage of its
credible intervals on the cumulative incidence and the all-cause B-life.

In words: draw the true parameters of each cause from the priors,
simulate a test from them, fit, and ask how often the credible interval
holds the true incidence. When the posterior is computed correctly the
truth's posterior probability (its "PIT" value: the share of the joint
draws below it) is uniform, so the central 50% and 90% intervals hold it
in 50% and 90% of repetitions. The rules are the validation kit's
(``surpyval/tests/bayes/kit.py``): coverage within three binomial
standard errors, uniformity by Kolmogorov-Smirnov at 0.001, no fit
refused.

The scenario and its seed were fixed before the first run (26 Nov 2026):
50 units, two Weibull causes, every unit observed to 150 (censored there
if it has not failed); cause ``a``: ``alpha`` log-normal about 100
(sigma 0.3), ``beta`` log-normal about 2 (sigma 0.2); cause ``b``:
``alpha`` log-normal about 150 (sigma 0.3), ``beta`` log-normal about
1.2 (sigma 0.2). Checked: each cause's incidence at 100, and the
all-cause B10 life. The grid (exact). 300 repetitions, seed 63000.

The data are simulated from the model's definition (independent latent
failure times, the earliest observed), not with the package's
simulators.
"""

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.tests.bayes import kit
from surpyval.univariate.competing_risks import ParametricCompetingRisks

pytestmark = pytest.mark.calibration

REPS = 300
SEED = 63000
UNITS, END, AT = 50, 150.0, 100.0
CAUSES = {"a": (100.0, 2.0), "b": (150.0, 1.2)}


def _priors():
    return {
        k: {
            "alpha": sp.LogNormal.from_params([np.log(a), 0.3]),
            "beta": sp.LogNormal.from_params([np.log(b), 0.2]),
        }
        for k, (a, b) in CAUSES.items()
    }


def _truth(rng):
    return {
        k: (
            float(a * np.exp(0.3 * rng.standard_normal())),
            float(b * np.exp(0.2 * rng.standard_normal())),
        )
        for k, (a, b) in CAUSES.items()
    }


def _simulate(truth, rng):
    t = {k: a * rng.weibull(b, UNITS) for k, (a, b) in truth.items()}
    first = np.minimum(t["a"], t["b"])
    x = np.minimum(first, END)
    e = np.where(first > END, None, np.where(t["a"] < t["b"], "a", "b"))
    return x, e.astype(object)


def run(reps=REPS, seed=SEED):
    """The PIT values of the checked quantities, by name."""
    priors = _priors()
    pits: dict = {"cif_a": [], "cif_b": [], "b10": []}
    refused = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = _truth(rng)
        x, e = _simulate(truth, rng)
        try:
            model = ParametricCompetingRisks.fit_bayes(x, e, priors=priors)
        except ValueError as err:
            refused.append((r, str(err)[:160]))
            continue
        exact = ParametricCompetingRisks.from_fitted(
            {k: sp.Weibull.from_params(list(v)) for k, v in truth.items()}
        )
        for k in ("a", "b"):
            draws = model.cif_draws(AT, k)[:, 0]
            pits[f"cif_{k}"].append(float(np.mean(draws < exact.cif(AT, k))))
        pits["b10"].append(
            float(np.mean(model.quantile_draws(0.1) < exact.qf(0.1)))
        )
    return {k: np.asarray(v) for k, v in pits.items()}, refused


def test_incidence_and_b10_intervals_hold_their_level():
    pits, refused = run()
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
