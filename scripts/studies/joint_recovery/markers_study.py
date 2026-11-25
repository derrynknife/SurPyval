"""The recovery study of the joint model of two markers.

Design (fixed before the first run, 25 Nov 2026): 200 fleets (seeds 3000
to 3199) of 200 items from ``_multi_fleet.simulate`` (the test fleet:
vibration and temperature each a straight line with a start and a rate
of each item's own, the four correlated across the markers, read every
30 and 60 days, noise sd 0.5 and 2.0; Weibull PH, alpha 800 days and
beta 2 at vibration 4 and temperature 65, effects 0.8 and 0.08 per
unit), followed to 730 days. Each is fitted with ``JointPH(Weibull,
{"vibration": LinearPath, "temperature": LinearPath})`` (every parameter
varying, every correlation estimated; the full grid, 2,401 nodes an
item) and ``marker_reference={"vibration": 4, "temperature": 65}``, so
the truth is in the model's own parameterisation. Recorded: each
parameter's estimate and its 95% Wald interval from ``summary()`` (the
sds and the Weibull's on the log scale, the correlations on Fisher's),
and whether the maximum was verified. Reported: the bias (with its Monte
Carlo error) and the interval's coverage over the verified fits.

    python scripts/studies/joint_recovery/markers_study.py [reps]

Writes ``results_markers.jsonl`` (one fit a line; a rerun skips the seeds
there) and ``summary_markers.json`` here. ``PROCS=<n>`` sets the worker
processes (default 2).
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 3000
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PROCS = int(os.environ.get("PROCS", "2"))
RESULTS = os.path.join(HERE, "results_markers.jsonl")
SUMMARY = os.path.join(HERE, "summary_markers.json")


def truth() -> dict:
    from surpyval.tests.univariate.joint._multi_fleet import TRUTH

    names = ["vibration.a", "vibration.b", "temperature.a", "temperature.b"]
    out = {k: v for k, v in TRUTH.items() if k != "corr"}
    corr = np.asarray(TRUTH["corr"], dtype=float)
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"corr_{names[i]}_{names[j]}"] = float(corr[i, j])
    return out


def one(seed: int) -> dict:
    from surpyval import JointPH, Weibull
    from surpyval.degradation import LinearPath
    from surpyval.tests.univariate.joint._multi_fleet import simulate

    x, c, m, _ = simulate(n=200, seed=seed)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = JointPH(
                Weibull, {"vibration": LinearPath, "temperature": LinearPath}
            ).fit(
                x,
                c=c,
                markers=m,
                marker_reference={"vibration": 4.0, "temperature": 65.0},
            )
        except Exception as error:  # a failed fit is recorded as such
            return {"seed": seed, "error": repr(error)}
        table = model.summary()
    table = table.droplevel(0)
    return {
        "seed": seed,
        "seconds": time.time() - start,
        "maximum": model.maximum,
        "failures": float(model.n_events),
        "estimate": dict(zip(model.parameter_names, map(float, model.params))),
        "lower": {k: float(v) for k, v in table["coef lower 95%"].items()},
        "upper": {k: float(v) for k, v in table["coef upper 95%"].items()},
    }


def summarise(rows: list) -> dict:
    true = truth()
    good = [r for r in rows if r.get("maximum") == "verified"]
    out: dict = {
        "design": __doc__.split("\n\n")[1],
        "fits": len(rows),
        "verified": len(good),
        "failed": sum(1 for r in rows if "error" in r),
        "median_seconds": float(
            np.median([r["seconds"] for r in rows if "seconds" in r])
        ),
        "parameters": {},
    }
    for name, value in true.items():
        est = np.array([r["estimate"][name] for r in good])
        lo = np.array([r["lower"][name] for r in good])
        hi = np.array([r["upper"][name] for r in good])
        cover = (lo <= value) & (value <= hi)
        out["parameters"][name] = {
            "truth": value,
            "mean": float(est.mean()),
            "bias": float(est.mean() - value),
            "bias_mc_se": float(est.std(ddof=1) / np.sqrt(est.size)),
            "relative_bias": (
                float((est.mean() - value) / value) if value else None
            ),
            "coverage_95": float(cover.mean()),
            "coverage_mc_se": float(np.sqrt(0.95 * 0.05 / est.size)),
        }
    return out


def main() -> None:
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            for line in handle:
                row = json.loads(line)
                done[row["seed"]] = row
    seeds = [s for s in range(SEED0, SEED0 + REPS) if s not in done]
    with Pool(PROCS) as pool, open(RESULTS, "a") as handle:
        for row in pool.imap_unordered(one, seeds):
            done[row["seed"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(row["seed"], row.get("maximum"), row.get("seconds"))
    rows = [done[s] for s in sorted(done)]
    summary = summarise(rows)
    with open(SUMMARY, "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
