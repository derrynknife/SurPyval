"""Simulation-based calibration of the Bayesian regressions (Talts et
al., 2018): a proportional hazards and an accelerated failure time model,
fitted by ``fit_bayes`` (SurPyval's No-U-Turn sampler).

The rules and seeds were fixed before the first run (DECISIONS.md, 21 Nov
2026), as the sampler's scenario of ``test_sbc_bayes.py`` has them:

- 500 replicates per scenario; replicate ``r`` draws the true parameters
  from the priors and simulates its data with seed ``46100 + r``, and
  fits with four chains of 250 draws after 500 warm-up iterations, seeded
  ``46600 + r``;
- every tenth draw of each chain is kept, the first 99 of those 100 used
  (a sampler's draws follow one another, and the ranks are uniform only
  for independent ones), so a rank is one of 100 values, in 20 bins of 5;
- each parameter's ranks pass a chi-square test of uniformity at the
  0.001 level, and its 90% interval (rank 5 to 94) covers the truth in a
  share within 0.03 of 0.90;
- a replicate whose fit refuses fails the scenario; one that fails its
  convergence checks is counted, printed and kept.

The design is fixed for every replicate: 40 units, the covariate evenly
spaced from -1 to 1, and Type I censoring at a time from the priors'
central model (every parameter at its prior median), so that it does not
depend on the truth.

Study to run at consolidation (``--run-calibration``).
"""

import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp

pytestmark = pytest.mark.calibration

REPS = 500
BINS = 20
CHAINS, DRAWS, WARMUP, THIN, KEEP = 4, 250, 500, 10, 99
N = 40
Z = np.linspace(-1.0, 1.0, N)[:, None]

SCENARIOS = {
    "WeibullPH": dict(
        fitter=sp.WeibullPH,
        priors={
            "alpha": sp.LogNormal.from_params([np.log(100.0), 0.3]),
            "beta": sp.LogNormal.from_params([np.log(1.5), 0.2]),
            "coef_0": sp.Normal.from_params([0.0, 0.5]),
        },
        # the central model's 70% quantile at Z = 0
        censor=lambda: 100.0 * (-np.log(0.3)) ** (1 / 1.5),
    ),
    "LogNormalAFT": dict(
        fitter=sp.LogNormalAFT,
        priors={
            "mu": sp.Normal.from_params([4.0, 0.3]),
            "sigma": sp.LogNormal.from_params([np.log(0.5), 0.2]),
            "coef_0": sp.Normal.from_params([0.0, 0.5]),
        },
        censor=lambda: float(np.exp(4.0 + 0.5 * stats.norm.ppf(0.7))),
    ),
}


def _truth(priors, names, rng):
    return np.array(
        [float(priors[k].random(1, random_state=rng)[0]) for k in names]
    )


def _ranks(name):
    spec = SCENARIOS[name]
    fitter, priors, censor = spec["fitter"], spec["priors"], spec["censor"]()
    # a model to draw from: any fit, given each replicate's parameters
    template = fitter.fit(np.linspace(10.0, 300.0, N), Z)
    names = list(template.parameter_names)
    ranks = {k: [] for k in names}
    not_converged = 0
    for r in range(REPS):
        rng = np.random.default_rng(46100 + r)
        theta = _truth(priors, names, rng)
        life, _ = template.with_params(theta).random(1, Z, random_state=rng)
        x = np.minimum(life, censor)
        c = (life > censor).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit_bayes(
                x,
                Z,
                c,
                priors=priors,
                chains=CHAINS,
                draws=DRAWS,
                warmup=WARMUP,
                random_state=46600 + r,
            )
        not_converged += int(not model.ok)
        points = model.posterior.points.reshape(CHAINS, DRAWS, -1)
        kept = points[:, ::THIN].reshape(-1, points.shape[2])[:KEEP]
        for k, p in enumerate(names):
            col = model.posterior.parameter_names.index(p)
            ranks[p].append(int(np.sum(kept[:, col] < theta[k])))
    print(f"{name}: {not_converged} of {REPS} fits failed their checks")
    return ranks


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_simulation_based_calibration_regression(name):
    failures = []
    for p, rk in _ranks(name).items():
        rk = np.asarray(rk)
        counts = np.bincount(rk * BINS // (KEEP + 1), minlength=BINS)
        expected = len(rk) / BINS
        stat = float(np.sum((counts - expected) ** 2 / expected))
        pval = float(stats.chi2.sf(stat, BINS - 1))
        cover = float(np.mean((rk >= 5) & (rk < KEEP + 1 - 5)))
        print(f"{name} {p}: chi-square p = {pval:.4f}, 90% coverage {cover}")
        if pval < 0.001:
            failures.append(f"{p}: ranks not uniform (p = {pval:.2e})")
        if abs(cover - 0.90) > 0.03:
            failures.append(f"{p}: 90% coverage {cover:.3f}")
    assert not failures, f"{name}: " + "; ".join(failures)
