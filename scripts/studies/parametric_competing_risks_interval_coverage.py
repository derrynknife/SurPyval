"""Do the Wald bounds of ``ParametricCompetingRisks`` cover when the
failures are found at periodic inspections?

A unit inspected every ``gap`` time units is found failed at the first
inspection after its failure, so its failure time is known only to lie
between two inspections (interval censored, ``c = 2``; left censored,
``c = -1``, when it is the first). The model is then fitted jointly over
both causes (the likelihood no longer factorises), and its covariance is
the inverse of the joint Hessian, with terms between the causes. This
study checks by simulation that the delta-method bounds built on it
cover, and that the estimates are unbiased.

Design (fixed before any result was seen):

- cause ``a`` Weibull(alpha 10, beta 2), cause ``b`` Weibull(alpha 14,
  beta 1);
- each unit's follow-up ends at an independent Uniform(2, 30) time; it
  is inspected at ``gap``, ``2 gap``, ... up to the last inspection
  before then. A unit failing before its last inspection is recorded
  with its cause in the window between the inspections around its
  failure; any other unit is right censored at its last inspection;
- scenarios ``gap2`` (inspections every 2) and ``gap5`` (every 5), each
  with ``n`` = 300 units;
- ``ParametricCompetingRisks.fit(x, e, c=c, dist=Weibull)``; the 95%
  two-sided Wald bounds ``cb(t, "a")``, ``cb(t, "b")`` (the cumulative
  incidences, logit scale) and ``cb(t, on="sf")`` (all-cause survival,
  log cumulative hazard scale) at t = 4, 8 and 15, and the Wald
  intervals ``estimate +/- 1.96 standard error`` of the four parameters.

Replicate ``r`` of scenario ``gap2`` uses ``default_rng(20261106 + r)``,
of ``gap5`` ``default_rng(20261106 + 10**6 + r)``. A fit or bound that
fails or is ``nan`` is counted and left out of the coverage, with the
count reported. Coverage is reported with its Monte Carlo standard error
``sqrt(p (1 - p) / R)``; the bias of each parameter as the mean estimate
over the truth, minus 1.

    python scripts/studies/parametric_competing_risks_interval_coverage.py \\
        [reps] [processes]

Defaults: 500 replicates, 2 processes.

Results (5 Nov 2026, surpyval dev, 500 replicates, 2 processes, about 7
minutes): no fit or bound failed, every maximum was verified and no bound
was ``nan``. Coverage of the 95% bounds, MC standard error in brackets:

=========  =======  =============  =============  =============
scenario   bound    t = 4          t = 8          t = 15
=========  =======  =============  =============  =============
gap2       cif_a    0.964 (0.010)  0.948 (0.010)  0.958 (0.010)
gap2       cif_b    0.948 (0.010)  0.962 (0.010)  0.960 (0.010)
gap2       sf       0.956 (0.010)  0.952 (0.010)  0.942 (0.010)
gap5       cif_a    0.948 (0.010)  0.956 (0.010)  0.956 (0.010)
gap5       cif_b    0.922 (0.010)  0.934 (0.010)  0.938 (0.010)
gap5       sf       0.932 (0.010)  0.938 (0.010)  0.948 (0.010)
=========  =======  =============  =============  =============

The parameters' Wald intervals (alpha_a, beta_a, alpha_b, beta_b) cover
0.946, 0.960, 0.954, 0.940 (gap2) and 0.940, 0.952, 0.944, 0.958 (gap5);
the relative biases are -0.002, +0.011, +0.030, -0.004 (gap2) and -0.001,
+0.007, +0.014, +0.005 (gap5), the usual small-sample bias of a Weibull
scale and shape (cause ``b``'s scale has a standard error of about 20%).

Every gap2 cell is within two MC standard errors of 0.95. With
inspections every 5, cause ``b``'s incidence at t = 4 covers 0.922, 2.8
standard errors low, and its cells and the all-cause survival's are all
below 0.95 (0.922 to 0.948): t = 4 lies inside the first inspection
window, where nothing is seen of when the failures happened, and the
estimate there leans on the model's shape. Reported as found.
"""

import sys
import warnings
from multiprocessing import Pool
from typing import Any

import numpy as np
from scipy.stats import norm

import surpyval as sp
from surpyval.univariate.competing_risks import ParametricCompetingRisks

SEED = 20261106
TIMES = np.array([4.0, 8.0, 15.0])
Z95 = norm.ppf(0.975)
N = 300
TRUE = {"a": (10.0, 2.0), "b": (14.0, 1.0)}
GAPS = {"gap2": 2.0, "gap5": 5.0}
OFFSET = {"gap2": 0, "gap5": 10**6}


def truth() -> ParametricCompetingRisks:
    return ParametricCompetingRisks.from_fitted(
        {k: sp.Weibull.from_params(list(v)) for k, v in TRUE.items()}
    )


def simulate(gap: float, n: int, rng: np.random.Generator) -> tuple:
    """``(x, e, c)`` of ``n`` units inspected every ``gap``."""
    latent = np.column_stack(
        [alpha * rng.weibull(beta, size=n) for alpha, beta in TRUE.values()]
    )
    first = latent.min(axis=1)
    cause = np.where(latent[:, 0] <= latent[:, 1], "a", "b")
    end = rng.uniform(2.0, 30.0, size=n)
    last = gap * np.floor(end / gap)
    found = (first <= last) & (last > 0)
    hi = gap * np.ceil(first / gap)
    lo = hi - gap
    x = np.column_stack(
        [
            np.where(found, np.where(lo > 0, lo, hi), last),
            np.where(found, hi, last),
        ]
    )
    c = np.where(found, np.where(lo > 0, 2, -1), 1)
    e = cause.astype(object)
    e[~found] = None
    # A unit with no inspection before its follow-up ended tells nothing.
    keep = last > 0
    return x[keep], e[keep], c[keep]


def replicate(job: tuple) -> Any:
    name, r = job
    rng = np.random.default_rng(SEED + OFFSET[name] + r)
    x, e, c = simulate(GAPS[name], N, rng)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            model = ParametricCompetingRisks.fit(x, e, c=c, dist=sp.Weibull)
            est = model.params
            se = model.standard_errors()
            out = {
                "params": np.column_stack([est - Z95 * se, est + Z95 * se]),
                "est": est,
                "cif_a": model.cb(TIMES, "a"),
                "cif_b": model.cb(TIMES, "b"),
                "sf": model.cb(TIMES, on="sf"),
                "maximum": model.maximum,
            }
        except Exception as err:  # noqa: BLE001 - counted, reported
            return {"error": repr(err)}
    return out


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    true = truth()
    targets = {
        "cif_a": true.cif(TIMES, "a"),
        "cif_b": true.cif(TIMES, "b"),
        "sf": true.sf(TIMES),
    }
    theta = np.concatenate([np.array(v) for v in TRUE.values()])
    for name in GAPS:
        with Pool(procs) as pool:
            results = pool.map(replicate, [(name, r) for r in range(reps)])
        ok = [res for res in results if "error" not in res]
        failed = len(results) - len(ok)
        unverified = sum(res["maximum"] != "verified" for res in ok)
        print(
            f"\n{name}: {reps} replicates, {failed} failed, "
            f"{unverified} unverified maxima"
        )
        for key, target in [("params", theta), *targets.items()]:
            bounds = np.array([res[key] for res in ok])
            lower, upper = bounds[..., 0], bounds[..., 1]
            nan = np.isnan(lower) | np.isnan(upper)
            hit = (lower <= target) & (target <= upper) & ~nan
            cov = hit.sum(axis=0) / (~nan).sum(axis=0)
            se = np.sqrt(0.95 * 0.05 / (~nan).sum(axis=0))
            cells = ", ".join(f"{v:.3f} ({s:.3f})" for v, s in zip(cov, se))
            print(f"  {key:7s} coverage: {cells}; nan: {int(nan.sum())}")
        est = np.array([res["est"] for res in ok])
        bias = est.mean(axis=0) / theta - 1
        print("  relative bias of alpha_a, beta_a, alpha_b, beta_b:")
        print("  " + ", ".join(f"{b:+.4f}" for b in bias))


if __name__ == "__main__":
    main()
