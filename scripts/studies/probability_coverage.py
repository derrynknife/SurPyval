"""Coverage of the 95% two-sided bounds on a probability ``p``.

For each true ``p`` in {0.01, 0.05, 0.3} and number of trials ``N`` in
{10, 50}, each replicate draws ``k ~ Binomial(N, p)``, fits
``Bernoulli.fit([1, 0], n=[k, N - k])`` and records whether each interval
contains ``p``:

- ``exact``: ``param_cb("p")``, Clopper-Pearson (the default);
- ``jeffreys``: ``param_cb("p", method="jeffreys")``;
- ``wald-cov``: ``p_hat +- 1.96 sqrt(covariance())``, the plain normal
  interval a caller builds from the covariance alone (``nan`` at
  ``k = 0`` or ``N``, counted as not covering);
- ``wald-logit``: ``param_cb("p", method="wald")``, the package's Wald
  bound on the logit scale (also undefined at the edges).

The Monte Carlo share is shown with its standard error, and beside it the
exact coverage, the sum of the binomial probabilities of the ``k`` whose
interval covers ``p`` (no simulation error), as a check.

    python scripts/studies/probability_coverage.py [reps]

The seed is fixed before the results were seen: cell ``i`` uses
``default_rng(20261105 + i)``. 4000 replicates by default; under a minute.
"""

import sys
import warnings

import numpy as np
from scipy.stats import binom, norm

import surpyval as sp

PS = (0.01, 0.05, 0.3)
NS = (10, 50)
METHODS = ("exact", "jeffreys", "wald-cov", "wald-logit")
Z = norm.ppf(0.975)


def intervals(k: int, n: int) -> dict:
    """Each method's two-sided 95% interval for k events in n trials."""
    model = sp.Bernoulli.fit([1, 0], n=[k, n - k])
    p_hat = k / n
    se = float(np.sqrt(model.covariance()[0, 0]))
    with warnings.catch_warnings():
        # The logit Wald bound warns (and is nan) at k = 0 and k = n.
        warnings.simplefilter("ignore")
        logit = model.param_cb("p", method="wald")
    return {
        "exact": model.param_cb("p"),
        "jeffreys": model.param_cb("p", method="jeffreys"),
        "wald-cov": np.array([p_hat - Z * se, p_hat + Z * se]),
        "wald-logit": logit,
    }


def covers(interval: np.ndarray, p: float) -> bool:
    # nan compares False: an undefined interval does not cover.
    return bool(interval[0] <= p <= interval[1])


def main(reps: int) -> None:
    print(f"{reps} replicates per cell; 95% two-sided; MC (se) | exact")
    header = f"{'p':>5} {'N':>3}  " + "  ".join(f"{m:>24}" for m in METHODS)
    print(header)
    cell = 0
    for p in PS:
        for n in NS:
            rng = np.random.default_rng(20261105 + cell)
            cell += 1
            table = {k: intervals(k, n) for k in range(n + 1)}
            ks = rng.binomial(n, p, size=reps)
            row = []
            for m in METHODS:
                hit = np.array([covers(table[k][m], p) for k in ks])
                share = hit.mean()
                se = np.sqrt(share * (1 - share) / reps)
                exact = sum(
                    binom.pmf(k, n, p)
                    for k in range(n + 1)
                    if covers(table[k][m], p)
                )
                row.append(f"{share:.3f} ({se:.3f}) | {exact:.3f}")
            print(f"{p:>5} {n:>3}  " + "  ".join(f"{c:>24}" for c in row))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
