"""The study of a spline path whose bend varies between items: fleets
whose pumps each turn up at their own pace after a year, fitted with the
bend common to all (each pump its own level and rate) and with the bend
varying too.

    python scripts/studies/joint_recovery/shape_study.py [reps]

The design, fixed before the first run:

- 200 fleets (seeds 2000 to 2199) of 150 pumps, each read every 30 days
  for up to 730 days.
- Each pump's marker is the spline ``SplinePath(knots=[365],
  boundary=(0, 730))`` with level 3.0 (sd 0.5), rate 0 a day (sd 0.3 /
  365) and bend ``s1`` 3.0 (sd 1.0), independent between pumps, read
  with noise of sd 0.4.
- The hazard is Weibull proportional hazards (alpha 800 days, beta 2 at
  a marker of 4) with a marker effect of 0.9 per unit.
- Each fleet is fitted twice with ``marker_reference=4``:
  ``JointPH(Weibull, path, varying=("level", "rate"))`` (the bend
  common) and ``varying=("level", "rate", "s1")`` (the bend varying,
  with the default structure: level and rate a block, the bend on its
  own), whose parameters the truth is in.

Each fleet is appended to results_shape.jsonl as it finishes; a rerun
skips the seeds already there. The summary is written to
summary_shape.json and printed. PROCS=<n> sets the worker processes
(default 2).
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from knee_study import fitted, rows  # noqa: E402

SEED0 = 2000
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PROCS = int(os.environ.get("PROCS", "2"))
N = 150
KNOTS, BOUNDARY = [365.0], (0.0, 730.0)
MEAN = [3.0, 0.0, 3.0]
SD = [0.5, 0.3 / 365, 1.0]
NOISE = 0.4

TRUTH = {
    "level": 3.0,
    "rate": 0.0,
    "s1": 3.0,
    "sd_level": 0.5,
    "sd_rate": 0.3 / 365,
    "sd_s1": 1.0,
    "corr_level_rate": 0.0,
    "sd_noise": NOISE,
    "alpha": 800.0,
    "beta": 2.0,
    "marker_effect": 0.9,
}


def one(seed):
    from surpyval import JointPH, Weibull
    from surpyval.degradation import SplinePath
    from surpyval.tests.univariate.joint._fleet import simulate_path

    path = SplinePath(knots=KNOTS, boundary=BOUNDARY)
    data = simulate_path(path, MEAN, SD, NOISE, n=N, seed=seed)
    out = dict(seed=seed, ok=True)
    t0 = time.time()
    try:
        for name, varying in (
            ("common", ("level", "rate")),
            ("varying", ("level", "rate", "s1")),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = JointPH(Weibull, path, varying=varying).fit(
                    data, marker_reference=4.0
                )
            out[name] = fitted(model)
            out[name]["warnings"] = [str(w.message) for w in caught]
    except Exception as error:  # recorded, not hidden
        out = dict(seed=seed, ok=False, error=repr(error))
    out["time"] = time.time() - t0
    return out


def summarise(results):
    good = [r for r in results if r["ok"]]
    spread = [
        any("lower bound, 0" in w for w in r["varying"]["warnings"])
        for r in good
    ]
    out = dict(
        reps=len(results),
        fitted=len(good),
        failed=[r for r in results if not r["ok"]],
        fit_time_mean=float(np.mean([r["time"] for r in results])),
        varying_lower_aic=sum(
            r["varying"]["aic"] < r["common"]["aic"] for r in good
        ),
        bend_spread_at_zero=int(sum(spread)),
    )
    common_truth = {k: v for k, v in TRUTH.items() if k not in ("sd_s1",)}
    for name, truth in (("varying", TRUTH), ("common", common_truth)):
        fits = [r[name] for r in good]
        out[name] = dict(
            verified=sum(f["maximum"] == "verified" for f in fits),
            rows=rows(fits, truth),
        )
    return out


if __name__ == "__main__":
    log = os.path.join(HERE, "results_shape.jsonl")
    done = {}
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["seed"]] = r
    todo = [s for s in range(SEED0, SEED0 + REPS) if s not in done]
    start = time.time()
    with Pool(PROCS) as pool, open(log, "a") as f:
        for r in pool.imap_unordered(one, todo):
            done[r["seed"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
    summary = summarise([done[s] for s in range(SEED0, SEED0 + REPS)])
    summary["wall_time_this_run"] = time.time() - start
    summary["date"] = time.strftime("%Y-%m-%d")
    with open(os.path.join(HERE, "summary_shape.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
