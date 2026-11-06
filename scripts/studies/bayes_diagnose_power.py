"""Power of the Bayesian-fit checks of ``diagnose()`` against a wrong
distribution.

Each replicate draws 100 log-normal lifetimes (``mu = 2``, ``sigma = 1``;
a long right tail no Weibull follows), fits ``Weibull.fit_bayes`` with a
weak prior on the shape (``prior.around(2, between=(1, 4))``) and records
whether each check rejects at 5%:

- the posterior predictive checks (the realised Anderson-Darling
  discrepancy and the failures in time bins; 500 posterior draws);
- the goodness-of-fit tests of the maximum-likelihood fit that the report
  runs beside them (Anderson-Darling, parametric bootstrap, 200 refits).

The posterior predictive p-value is conservative by construction, so it
is expected to have less power than the bootstrap test; the study says by
how much.

    python scripts/studies/bayes_diagnose_power.py [reps]

The seeds are fixed (replicate ``r`` simulates with ``default_rng(5300 +
r)`` and resamples with seed ``5800 + r``), chosen before the results were
seen. 100 replicates by default; a few minutes on one core.
"""

import sys
import warnings

import numpy as np

import surpyval as sp
from surpyval.diagnostics.bayes import posterior_predictive
from surpyval.diagnostics.univariate import bootstrap

ALPHA = 0.05


def replicate(r: int) -> dict:
    rng = np.random.default_rng(5300 + r)
    x = np.exp(rng.normal(2, 1, 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit_bayes(
            x, priors={"beta": sp.prior.around(2, between=(1, 4))}
        )
        ppc = posterior_predictive(model, 500, random_state=5800 + r)
        gof = bootstrap(model.mle, 200, random_state=5800 + r)
    return {
        "ppc AD": ppc["p"]["AD"] <= ALPHA,
        "ppc bins": ppc["p"]["bins"] <= ALPHA,
        "ML fit AD (bootstrap)": gof["p"]["AD"] <= ALPHA,
    }


def main(reps: int) -> None:
    rows = [replicate(r) for r in range(reps)]
    for key in rows[0]:
        rate = float(np.mean([row[key] for row in rows]))
        se = np.sqrt(rate * (1 - rate) / reps)
        print(f"{key:24s} power {rate:.3f} (MC se {se:.3f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
