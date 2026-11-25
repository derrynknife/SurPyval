"""Coverage of the Bayesian regressions' credible intervals (the Laplace
engine): a proportional hazards and an accelerated failure time model.

If the parameters are drawn from the priors and the data from them, an
exact posterior's 90% interval holds the truth in 90% of replicates; this
measures what the normal approximation costs.

The rules and seeds were fixed before the first run (25 Nov 2026):

- 300 replicates per scenario; replicate ``r`` draws the true parameters
  from the priors and simulates its data with seed ``47100 + r``;
- each parameter's 90% two-sided interval (5% to 95% points) holds the
  truth in 0.90 +- 0.05 of replicates (three standard errors at 300);
- a replicate whose fit refuses fails the scenario.

The design is fixed for every replicate: 40 units, the covariate evenly
spaced from -1 to 1, and Type I censoring at a time from the priors'
central model (every parameter at its prior median), so that it does not
depend on the truth.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp

pytestmark = pytest.mark.calibration

REPS = 300
LEVEL = 0.90
TOLERANCE = 0.05
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


def coverage(name):
    spec = SCENARIOS[name]
    fitter, priors, censor = spec["fitter"], spec["priors"], spec["censor"]()
    template = fitter.fit(np.linspace(10.0, 300.0, N), Z)
    names = list(template.parameter_names)
    hits = {k: [] for k in names}
    refused, ks = [], []
    for r in range(REPS):
        rng = np.random.default_rng(47100 + r)
        theta = _truth(priors, names, rng)
        life, _ = template.with_params(theta).random(1, Z, random_state=rng)
        x = np.minimum(life, censor)
        c = (life > censor).astype(int)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fitter.fit_bayes(x, Z, c, priors=priors)
        except ValueError as e:
            refused.append((r, str(e)[:120]))
            continue
        ks.append(model.diagnostics["pareto_k"])
        for k, p in enumerate(names):
            lo, hi = model.param_cri(p, alpha_ci=1 - LEVEL)
            hits[p].append(lo <= theta[k] <= hi)
    return {k: float(np.mean(v)) for k, v in hits.items()}, refused, ks


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_laplace_regression_coverage(name):
    cover, refused, ks = coverage(name)
    print(
        f"{name}: coverage "
        + ", ".join(f"{k} {v:.3f}" for k, v in cover.items())
        + f"; Pareto k median {np.median(ks):.2f}"
    )
    assert not refused, f"{name}: {len(refused)} refused: {refused[:3]}"
    bad = {k: v for k, v in cover.items() if abs(v - LEVEL) > TOLERANCE}
    assert not bad, f"{name}: 90% coverage {bad}"
