"""Bias and coverage of the Aalen-Johansen cumulative incidence with
delayed entry (``CompetingRisks.fit(..., tl=)``).

Each replicate draws units with two causes of constant cause-specific
hazard, 0.10 (cause "a") and 0.05 (cause "b"), so that the cumulative
incidence of "a" is ``F_a(t) = (2 / 3) (1 - exp(-0.15 t))``. A third of
the units enter at 0 and the rest at a time uniform on (0, 4); a unit
that fails before its entry is never seen (left truncation), and a seen
unit is censored at its entry plus a time uniform on (0, 10). Units are
drawn until 200 are seen. The study records, at ``t = 2`` and ``t = 6``:

- the bias of ``cif`` with ``tl`` and without it (the fit that ignores
  the delayed entry, which counts too many at risk early on);
- the share of 95% two-sided bounds (``cb``, the default log(-log)
  bounds) that contain the truth, with Aalen's variance (the unweighted
  fit) and with the robust variance (the fit with ``w = 1``, as R's
  ``survfit(Surv(tl, x, factor(e)) ~ 1)``), with their Monte Carlo
  standard errors.

    python scripts/studies/aalen_johansen_entry_coverage.py [reps]

The seeds are fixed (replicate ``r`` uses ``default_rng(20261105 + r)``),
chosen before the results were seen. 1,000 replicates by default; about five
seconds on one core.

Results (1,000 replicates): bias with ``tl`` 0.0012 and 0.0002 (Monte
Carlo se 0.0011, 0.0013), ignoring the entry -0.074 and -0.059; coverage
of the 95% bounds 0.962 and 0.952 (Aalen) and 0.960 and 0.951 (robust),
Monte Carlo se 0.006-0.007.
"""

import sys
import warnings

import numpy as np

import surpyval as sp

TIMES = np.array([2.0, 6.0])
H_A, H_B = 0.10, 0.05
TRUTH = H_A / (H_A + H_B) * (1 - np.exp(-(H_A + H_B) * TIMES))
SEEN = 200


def replicate(r: int) -> dict:
    rng = np.random.default_rng(20261105 + r)
    xs: list[float] = []
    tls: list[float] = []
    es: list[str | None] = []
    while len(xs) < SEEN:
        entry = 0.0 if rng.uniform() < 1 / 3 else rng.uniform(0, 4)
        t = rng.exponential(1 / (H_A + H_B))
        if t <= entry:
            continue
        cause = "a" if rng.uniform() < H_A / (H_A + H_B) else "b"
        censor = entry + rng.uniform(0, 10)
        xs.append(min(t, censor))
        tls.append(entry)
        es.append(cause if t <= censor else None)
    x, tl = np.array(xs), np.array(tls)
    e = np.array(es, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "Aalen": sp.CompetingRisks.fit(x, e, tl=tl),
            "robust (w = 1)": sp.CompetingRisks.fit(
                x, e, tl=tl, w=np.ones(len(x))
            ),
        }
        naive = sp.CompetingRisks.fit(x, e)
    out = {
        "estimate": fits["Aalen"].cif(TIMES, "a"),
        "ignoring entry": naive.cif(TIMES, "a"),
    }
    for name, model in fits.items():
        cb = model.cb(TIMES, "a")
        out[name] = (cb[:, 0] <= TRUTH) & (TRUTH <= cb[:, 1])
    return out


def main(reps: int) -> None:
    results = [replicate(r) for r in range(reps)]
    print(f"{reps} replicates, {SEEN} units seen each; truth F_a = {TRUTH}")
    for name in ("estimate", "ignoring entry"):
        est = np.array([res[name] for res in results])
        bias = est.mean(axis=0) - TRUTH
        se = est.std(axis=0, ddof=1) / np.sqrt(reps)
        print(f"bias, {name}: {bias.round(4)} (MC se {se.round(4)})")
    for name in ("Aalen", "robust (w = 1)"):
        hit = np.array([res[name] for res in results], dtype=float)
        cover = hit.mean(axis=0)
        se = np.sqrt(cover * (1 - cover) / reps)
        print(f"coverage, {name}: {cover.round(3)} (MC se {se.round(3)})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
