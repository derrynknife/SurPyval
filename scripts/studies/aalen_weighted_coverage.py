"""Design-based coverage of the weighted Aalen model's robust bounds.

Each replicate draws a population of 1,500 units from an additive hazard
``h(t | z) = 0.1 + 0.05 z`` (``z`` uniform on 0 to 4), censored uniformly
on (0, 12), so that ``B_z(t) = 0.05 t``. A sample is then drawn with
outcome-dependent probabilities -- every unit that failed, a quarter of
those censored, and twice as often where ``z > 2`` (``pi = 0.25`` or
``0.5`` for the censored) -- and fitted with ``w = 1 / pi``. The study
records the share of 95% two-sided bounds on ``B_z(5)`` and ``B_0(5)``
that contain the truth (0.25 and 0.5), with their Monte Carlo standard
errors, for

- the weighted fit's robust bounds (what ``AalenAdditive.fit(w=)``
  gives);
- the weighted estimate with the variance the classical formula gives
  when the weights are read as counts (``n = w`` rounded to whole units),
  to show why weighted fits refuse it;
- the unweighted fit of the sample (biased by the design).

    python scripts/studies/aalen_weighted_coverage.py [reps]

The seeds are fixed (replicate ``r`` uses ``default_rng(20261105 + r)``),
chosen before the results were seen. 500 replicates by default; about ten
minutes on one core.
"""

import sys
import warnings

import numpy as np
from scipy.stats import norm

import surpyval as sp

T0 = 5.0
TRUTH = {"baseline": 0.1 * T0, "z": 0.05 * T0}


def replicate(r: int) -> dict:
    rng = np.random.default_rng(20261105 + r)
    N = 1500
    z = rng.uniform(0, 4, N)
    t = rng.exponential(1 / (0.1 + 0.05 * z))
    u = rng.uniform(0, 12, N)
    x = np.minimum(t, u)
    c = (t > u).astype(int)
    pi = np.where(c == 0, 1.0, np.where(z > 2, 0.5, 0.25))
    keep = rng.uniform(size=N) < pi
    x, c, z, w = x[keep], c[keep], z[keep, None], 1 / pi[keep]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "robust (w)": sp.AalenAdditive.fit(x, z, c=c, w=w),
            "classical (n = w)": sp.AalenAdditive.fit(
                x, z, c=c, n=np.round(w).astype(int)
            ),
            "unweighted": sp.AalenAdditive.fit(x, z, c=c),
        }
    for name, model in fits.items():
        band = model.cumulative_coefficients_cb([T0])[0]  # (terms, 2)
        out[name] = [
            band[0, 0] <= TRUTH["baseline"] <= band[0, 1],
            band[1, 0] <= TRUTH["z"] <= band[1, 1],
        ]
    return out


def main(reps: int) -> None:
    hits: dict[str, list] = {}
    for r in range(reps):
        for name, covered in replicate(r).items():
            hits.setdefault(name, []).append(covered)
    print(f"{reps} replicates; 95% bounds at t = {T0:g}")
    print(f"{'method':<20}{'B_0 cover':>12}{'B_z cover':>12}  (MC se)")
    for name, rows in hits.items():
        share = np.mean(rows, axis=0)
        se = np.sqrt(share * (1 - share) / reps)
        print(
            f"{name:<20}{share[0]:>12.3f}{share[1]:>12.3f}  "
            f"({se[0]:.3f}, {se[1]:.3f})"
        )
    lo, hi = 0.95 + np.array([-1, 1]) * norm.isf(0.025) * np.sqrt(
        0.95 * 0.05 / reps
    )
    print(
        f"(95% of Monte Carlo draws of a correct 95% bound: {lo:.3f}-{hi:.3f})"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
