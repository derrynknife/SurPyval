"""Coverage of the weighted Turnbull estimate's case-bootstrap bounds
(``Turnbull.fit(..., w=).bootstrap_cb``) under informative sampling.

R has no weighted Turnbull bounds to compare with, so the bounds are
checked by simulation. Fixed before the first run:

- the population: Weibull lives (alpha 8, beta 1.5), inspected every 2
  time units up to 16 (a unit alive at 16 is right censored there), so
  every row is an interval ``(2k, 2k + 2]`` or ``(16, inf)``;
- the sample: 600 candidate units per replicate, each kept with
  probability 0.15 if its life is below the median (6.27) and 0.45 if
  not (about 180 kept), weight ``w = 1 / probability``; so the short
  lives are under-sampled and the unweighted estimate is biased;
- the bounds: two-sided 95% percentile bounds, ``n_boot = 200``, of the
  survival at 4 and at 8 (inspection times, where the estimate is
  identified), against the Weibull's true survival there;
- 500 replicates, replicate ``r`` seeded ``26105 + r`` for the data and
  ``36105 + r`` for the bootstrap; coverage reported with its Monte Carlo
  standard error, ``sqrt(p (1 - p) / 500)``.

    python scripts/studies/weighted_turnbull/coverage.py [reps]

Prints the coverage at each time, the weighted and unweighted estimates'
mean bias, and writes summary.json beside this script. At most 4 worker
processes; about 10 minutes on four cores.
"""

import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, BETA = 8.0, 1.5
TIMES = np.array([4.0, 8.0])
TRUTH = np.exp(-((TIMES / ALPHA) ** BETA))
MEDIAN = ALPHA * np.log(2) ** (1 / BETA)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def replicate(r: int) -> dict:
    import surpyval as sp

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(26105 + r)
    t = ALPHA * rng.weibull(BETA, 600)
    prob = np.where(t < MEDIAN, 0.15, 0.45)
    kept = rng.random(600) < prob
    t, w = t[kept], 1.0 / prob[kept]
    lo = np.minimum(np.floor(t / 2) * 2, 16.0)
    hi = np.where(t > 16.0, np.inf, lo + 2)
    model = sp.Turnbull.fit(
        xl=lo, xr=hi, w=w, turnbull_estimator="Kaplan-Meier"
    )
    plain = sp.Turnbull.fit(xl=lo, xr=hi, turnbull_estimator="Kaplan-Meier")
    cb = model.bootstrap_cb(TIMES, n_boot=200, random_state=36105 + r)
    covered = (cb[:, 0] <= TRUTH) & (TRUTH <= cb[:, 1])
    return {
        "r": r,
        "covered": covered.tolist(),
        "est": model.sf(TIMES).tolist(),
        "plain": plain.sf(TIMES).tolist(),
        "width": (cb[:, 1] - cb[:, 0]).tolist(),
    }


def main() -> None:
    with Pool(4) as pool:
        results = pool.map(replicate, range(REPS))
    cov = np.array([res["covered"] for res in results], dtype=float)
    est = np.array([res["est"] for res in results])
    plain = np.array([res["plain"] for res in results])
    width = np.array([res["width"] for res in results])
    p = cov.mean(axis=0)
    summary = {
        "reps": REPS,
        "times": TIMES.tolist(),
        "truth": TRUTH.tolist(),
        "coverage": p.tolist(),
        "coverage_mcse": np.sqrt(p * (1 - p) / REPS).tolist(),
        "weighted_bias": (est.mean(axis=0) - TRUTH).tolist(),
        "unweighted_bias": (plain.mean(axis=0) - TRUTH).tolist(),
        "mean_width": width.mean(axis=0).tolist(),
    }
    for k, t0 in enumerate(TIMES):
        print(
            f"S({t0:g}) = {TRUTH[k]:.4f}: coverage {p[k]:.3f} "
            f"(MC se {summary['coverage_mcse'][k]:.3f}), weighted bias "
            f"{summary['weighted_bias'][k]:+.4f}, unweighted bias "
            f"{summary['unweighted_bias'][k]:+.4f}, mean width "
            f"{summary['mean_width'][k]:.3f}"
        )
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
