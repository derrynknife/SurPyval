"""Size of the Bayesian-fit checks of ``diagnose()``.

- **Posterior predictive checks**: Weibull data (scale 10, shape 2, 40
  units, randomly censored by a uniform on (0, 25), about a third
  censored) fitted with ``Weibull.fit_bayes`` and a weak prior on the
  shape; 200 posterior draws a replicate. A posterior predictive p-value
  is conservative under a correct model (it is not uniform but piles up
  near 0.5), so the check is one-sided: each discrepancy must reject a
  correct model at 5% no more often than 5% (slack 0.01). The rates are
  printed.
- **Prior-data conflict**: the true parameters drawn from the prior
  itself (a proper prior on both, so the data are a draw from the prior
  predictive distribution), 40 complete units. Then the conflict check's
  null holds exactly, and each row must reject at 5% about 5% of the
  time, both ways (slack 0.02: the check is a normal approximation).

Replicate ``r`` simulates with seed ``6288 + r`` and draws with seed
``7288 + r``; the seeds and rules were fixed before the first run.
"""

import warnings

import numpy as np

import surpyval as sp
from surpyval.diagnostics.bayes import (
    posterior_predictive,
    prior_data_conflict,
)
from surpyval.tests.calibration._montecarlo import check_rate

ALPHA = 0.05
N = 40
REPS = 300
DRAWS = 200


def _fit(x, c=None, priors=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Weibull.fit_bayes(x, c, priors=priors)


def test_posterior_predictive_size():
    priors = {"beta": sp.prior.around(2, between=(1, 4))}
    hits = {"AD": 0, "bins": 0}
    for r in range(REPS):
        rng = np.random.default_rng(6288 + r)
        t = 10 * rng.weibull(2, N)
        cens = rng.uniform(0, 25, N)
        model = _fit(np.minimum(t, cens), (cens < t).astype(int), priors)
        out = posterior_predictive(model, DRAWS, random_state=7288 + r)
        for key in hits:
            hits[key] += out["p"][key] <= ALPHA
    for key, value in hits.items():
        check_rate(
            value,
            REPS,
            ALPHA,
            f"posterior predictive {key} size",
            side="upper",
        )


def test_prior_data_conflict_size():
    priors = {
        "alpha": sp.prior.around(10, between=(5, 20)),
        "beta": sp.prior.around(2, between=(1.5, 3)),
    }
    bound = _fit(10 * np.random.default_rng(0).weibull(2, N), None, priors)
    bound = bound._bound()
    hits = {"alpha": 0, "beta": 0, "joint": 0}
    for r in range(REPS):
        rng = np.random.default_rng(6288 + r)
        truth = {}
        for name in ("alpha", "beta"):
            t = bound.transforms[bound.names.index(name)]
            phi = priors[name].draw_phi(1, t, rng)
            truth[name] = float(t.to_theta(np.atleast_1d(phi))[0])
        x = truth["alpha"] * rng.weibull(truth["beta"], N)
        out = prior_data_conflict(_fit(x, None, priors))
        assert out["reason"] is None, out["reason"]
        hits["alpha"] += out["p"]["alpha"] <= ALPHA
        hits["beta"] += out["p"]["beta"] <= ALPHA
        hits["joint"] += out["joint"][2] <= ALPHA
    for key, value in hits.items():
        check_rate(value, REPS, ALPHA, f"prior-data conflict {key} size", 0.02)
