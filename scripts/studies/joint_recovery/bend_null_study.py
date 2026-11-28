"""The null study of a varying spline bend: the shape study's fleets
(``shape_study.py``) with the bends alike, or nearly, to see how often
the two checks on a varying bend get that case wrong.

The warning this study tested (a bend's spread estimated at 0) was
dropped on 2026-11-26 after its reduced run: the fits are compared by
``aic()`` instead. The study still records the warning's rule
(``warned``: the varying fit's ``sd_s1`` moves no item's marker by 1%
of the noise sd at any reading age), computed here from the fit, so a
rerun gives the same counts as before.

    python scripts/studies/joint_recovery/bend_null_study.py [reps]

The design, fixed before the first run:

- The shape study's fleets: seeds 2000 to 2000 + reps - 1 (200 in the
  full design) of 150 pumps on ``SplinePath(knots=[365], boundary=(0,
  730))``, level 3.0 (sd 0.5), rate 0 a day (sd 0.3 / 365), read every
  30 days to 730 with noise sd 0.4, the hazard Weibull PH (800, 2 at a
  marker of 4), 0.9 per unit of the marker.
- The bend ``s1`` 3.0 with sd 0 (the bends alike), 0.1 and 0.3 (small
  spreads), each fleet drawn afresh from its seed for each sd.
- Each fleet is fitted with ``marker_reference=4`` with the bend common,
  ``varying=("level", "rate")``, and varying, ``("level", "rate",
  "s1")``. Recorded: whether the bends-without-spread rule is met, the
  two AICs, the varying fit's ``sd_s1`` and whether its maximum is
  verified.
- The readings rule: each fleet cut at 450 and at 540 days (readings
  and follow-up after the cut dropped, an item alive then censored) is
  checked by ``SplinePath.check_varying`` for ``s1`` varying: refused or
  not. (Only the check: a refused fit never starts.)

What is counted, and the rule decided before the run (from the plan):
at sd 0 the warning firing is right and its not firing a miss; at sd 0.1
and 0.3 its firing is a false alarm. The 1% threshold and the readings
rule are kept unless false alarms are above 5% or misses above 20%. (A
spread whose truth is 0 is estimated above 0 in about half of fleets,
the boundary case of a variance, so misses near 50% are what a correct
estimate gives; ``aic()`` is the guard, and at sd 0 it should prefer
the varying fit in about 8% of fleets: half the chance a chi-square on
one degree of freedom passes 2.)

Each (sd, seed) is appended to results_bend_null.jsonl as it finishes; a
rerun skips those already there. The summary is written to
summary_bend_null.json and printed. PROCS=<n> sets the worker processes
(default 2).
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEED0 = 2000
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PROCS = int(os.environ.get("PROCS", "2"))
N = 150
KNOTS, BOUNDARY = [365.0], (0.0, 730.0)
MEAN = [3.0, 0.0, 3.0]
BEND_SDS = (0.0, 0.1, 0.3)
CUTS = (450.0, 540.0)
NOISE = 0.4
#: the dropped warning's rule: a spread under 1% of the noise sd
NO_SPREAD = 0.01


def at_zero(path: Any, model: Any, data: Any) -> bool:
    """Whether the varying fit's bend ``s1`` has its spread at 0 by the
    dropped warning's rule (:data:`NO_SPREAD`)."""
    params = dict(zip(model.parameter_names, model.params))
    if "sd_s1" not in params:
        return False
    ages = np.array(
        [a for item in data.to_items() for a in (item.markers or {})],
        dtype=float,
    )
    basis = np.abs(np.asarray(path.basis(ages), dtype=float))
    j = list(path.parameter_names).index("s1")
    reach = abs(float(params["sd_s1"])) * float(np.max(basis[:, j]))
    return bool(reach < NO_SPREAD * float(params["sd_noise"]))


def cut(data: Any, at: float) -> Any:
    """The fleet as it was seen at age ``at``: readings after it
    dropped, an item alive then censored there."""
    from surpyval import ItemLifetime, SurpyvalData

    items = []
    for item in data.to_items():
        failed = item.observed is not None
        end = float(item.observed if failed else item.right_censored)
        markers = {a: v for a, v in item.markers.items() if a <= at}
        if failed and end <= at:
            outcome = dict(observed=end)
        else:
            outcome = dict(right_censored=min(end, at))
        items.append(ItemLifetime(item.id, **outcome, markers=markers))
    return SurpyvalData.from_items(items)


def one(job: Any) -> Any:
    sd_s1, seed = job
    from surpyval import JointPH, Weibull
    from surpyval.degradation import SplinePath
    from surpyval.tests.univariate.joint._fleet import simulate_path

    path = SplinePath(knots=KNOTS, boundary=BOUNDARY)
    sd = [0.5, 0.3 / 365, sd_s1]
    data = simulate_path(path, MEAN, sd, NOISE, n=N, seed=seed)
    out: dict = dict(sd_s1=sd_s1, seed=seed, ok=True)
    t0 = time.time()
    try:
        for at in CUTS:
            seen = cut(data, at)
            ages = [list(i.markers) for i in seen.to_items() if i.markers]
            try:
                path.check_varying(ages, ("level", "rate", "s1"))
                out[f"refused_{at:g}"] = False
            except ValueError:
                out[f"refused_{at:g}"] = True
        for name, varying in (
            ("common", ("level", "rate")),
            ("varying", ("level", "rate", "s1")),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = JointPH(Weibull, path, varying=varying).fit(
                    data, marker_reference=4.0
                )
            params = dict(zip(model.parameter_names, model.params))
            out[name] = dict(
                aic=float(model.aic()),
                neg_ll=float(model.neg_ll()),
                maximum=model.maximum,
                sd_s1=float(params.get("sd_s1", np.nan)),
                warned=at_zero(path, model, data),
            )
    except Exception as error:  # recorded, not hidden
        out = dict(sd_s1=sd_s1, seed=seed, ok=False, error=repr(error))
    out["time"] = time.time() - t0
    return out


def summarise(results: Any) -> Any:
    out: dict = dict(reps=REPS, cells={})
    for sd_s1 in BEND_SDS:
        cell = [r for r in results if r["sd_s1"] == sd_s1]
        good = [r for r in cell if r["ok"]]
        n = len(good)
        warned = sum(r["varying"]["warned"] for r in good)
        prefers = sum(r["varying"]["aic"] < r["common"]["aic"] for r in good)
        short = sum(
            r["varying"]["neg_ll"] > r["common"]["neg_ll"] + 1e-6 for r in good
        )
        out["cells"][f"{sd_s1:g}"] = dict(
            fleets=len(cell),
            fitted=n,
            failed=[r for r in cell if not r["ok"]],
            warned=warned,
            warned_share=warned / n if n else None,
            aic_prefers_varying=prefers,
            aic_prefers_varying_share=prefers / n if n else None,
            varying_short_of_common=short,
            varying_verified=sum(
                r["varying"]["maximum"] == "verified" for r in good
            ),
            sd_s1_median=(
                float(np.median([r["varying"]["sd_s1"] for r in good]))
                if n
                else None
            ),
            refused={
                f"{at:g}": sum(r[f"refused_{at:g}"] for r in good)
                for at in CUTS
            },
            fit_time_mean=(
                float(np.mean([r["time"] for r in cell])) if cell else None
            ),
        )
    return out


if __name__ == "__main__":
    log = os.path.join(HERE, "results_bend_null.jsonl")
    done = {}
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[(r["sd_s1"], r["seed"])] = r
    jobs = [
        (sd_s1, seed)
        for seed in range(SEED0, SEED0 + REPS)
        for sd_s1 in BEND_SDS
    ]
    todo = [j for j in jobs if j not in done]
    start = time.time()
    with Pool(PROCS) as pool, open(log, "a") as f:
        for r in pool.imap_unordered(one, todo):
            done[(r["sd_s1"], r["seed"])] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
    summary = summarise([done[j] for j in jobs])
    summary["wall_time_this_run"] = time.time() - start
    summary["date"] = time.strftime("%Y-%m-%d")
    with open(os.path.join(HERE, "summary_bend_null.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
