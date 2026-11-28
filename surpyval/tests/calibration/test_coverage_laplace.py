"""Coverage of the Laplace engine's credible intervals.

If the parameters are drawn from the prior and the data from them, an
exact posterior's 90% credible interval holds the truth in 90% of
replicates. The Laplace approximation is a normal in place of the exact
posterior, so this measures what the approximation costs.

The rules and seeds were fixed before the first run (25 Nov 2026):

- 400 replicates per scenario; replicate ``r`` draws the truth from the
  prior and simulates its data with seed ``9800 + r`` (the priors and
  the designs of ``test_sbc_bayes``: censoring times from the prior's
  central model, never from the truth);
- the 90% two-sided interval (5% to 95% points) of each parameter and of
  the B10 life must hold the truth in 0.90 +- 0.045 of replicates (three
  standard errors at 400 replicates);
- a replicate whose fit refuses fails the scenario.

Scenarios: Weibull with 20 complete lives (plain, and reweighted by
importance sampling), Weibull with 50 lives 30% censored, LogNormal with
20 lives 30% censored, and the hard case, about 3 failures among 40
units (reweighted).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration.test_sbc_bayes import (
    PRIORS,
    _centre,
    _complete,
    _draw_prior,
    _few_failures,
    _type_one,
)

pytestmark = pytest.mark.calibration

REPS = 400
LEVEL = 0.90
TOLERANCE = 0.045


def _censored(share):
    return lambda d, t, n, r, c: _type_one(d, t, n, share, r, c)


SCENARIOS = {
    "Weibull complete 20": (sp.Weibull, "weak", _complete, 20, False),
    "Weibull complete 20, reweighted": (
        sp.Weibull,
        "weak",
        _complete,
        20,
        True,
    ),
    "Weibull 30% censored 50": (sp.Weibull, "weak", _censored(0.3), 50, False),
    "LogNormal 30% censored 20": (
        sp.LogNormal,
        "lognormal",
        _censored(0.3),
        20,
        False,
    ),
    "Weibull 3 failures in 40, reweighted": (
        sp.Weibull,
        "weak",
        _few_failures,
        40,
        True,
    ),
}


def coverage(label):
    """Each quantity's coverage, the refusals and the Pareto ks."""
    dist, key, simulate, n, importance = SCENARIOS[label]
    priors = PRIORS[key]
    centre = _centre(dist, priors)
    names = list(dist.parameter_names)
    hits = {k: [] for k in names + ["B10"]}
    refused, ks = [], []
    lo_hi = [(1 - LEVEL) / 2, (1 + LEVEL) / 2]
    for r in range(REPS):
        rng = np.random.default_rng(9800 + r)
        theta = _draw_prior(dist, priors, rng)
        data = simulate(dist, theta, n, rng, centre)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = dist.fit_bayes(
                    **data,
                    priors=priors,
                    method="laplace",
                    importance=importance,
                )
        except ValueError as e:
            refused.append((r, str(e)[:120]))
            continue
        ks.append(model.diagnostics["pareto_k"])
        post = model.posterior
        for k, name in enumerate(names):
            lo, hi = post.quantile(name, lo_hi)
            hits[name].append(lo <= theta[k] <= hi)
        cols = post.parameter_names
        b10 = dist.qf(0.1, *[post.points[:, cols.index(p)] for p in names])
        lo, hi = post.wquantile(b10, lo_hi)
        hits["B10"].append(lo <= float(dist.qf(0.1, *theta)) <= hi)
    cover = {k: float(np.mean(v)) for k, v in hits.items()}
    return cover, refused, np.asarray(ks)


# The results that fail the rules, kept as strict expected failures with
# the numbers; the rules are not loosened.
_FOUND: dict = {}


def _scenarios():
    return [
        pytest.param(
            label,
            marks=(
                [pytest.mark.xfail(strict=True, reason=_FOUND[label])]
                if label in _FOUND
                else []
            ),
        )
        for label in sorted(SCENARIOS)
    ]


@pytest.mark.parametrize("label", _scenarios())
def test_laplace_coverage(label):
    cover, refused, ks = coverage(label)
    print(
        f"{label}: coverage "
        + ", ".join(f"{k} {v:.3f}" for k, v in cover.items())
        + f"; Pareto k median {np.median(ks):.2f}, above 0.7 in "
        f"{int(np.sum(ks > 0.7))} of {len(ks)}"
    )
    assert not refused, f"{label}: {len(refused)} refused: {refused[:3]}"
    bad = {k: v for k, v in cover.items() if abs(v - LEVEL) > TOLERANCE}
    assert not bad, f"{label}: 90% coverage {bad}"
