"""Summarise ``results.jsonl`` (``study.py``) into ``summary.json`` and
print a table: per design, method, level and parameter, the coverage of
the two-sided interval, its Monte Carlo standard error
``sqrt(p (1 - p) / reps)`` at the observed coverage, the target (on the
edge, ``lfp_p``'s is ``1 - alpha / 2``, as in the calibration study) and
the median interval width.

    python scripts/studies/lfp_zi_bootstrap/summary.py [first_reps]

``first_reps`` restricts to replicates ``r < first_reps`` (200: the
calibration study's data sets, where ``param200`` is its bounds).
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("param200", "case200", "casepct200")
LEVELS = ("0.90", "0.95")


def main() -> None:
    first = int(sys.argv[1]) if len(sys.argv) > 1 else None
    from surpyval.tests.calibration.test_coverage_extra_parameters import (
        BOOT_STUDIES,
    )

    rows = [
        json.loads(line) for line in open(os.path.join(HERE, "results.jsonl"))
    ]
    # Only the replicates every design finished (a run is stopped by the
    # clock): the largest R with r < R done in all four designs.
    done: dict = {}
    for r in rows:
        done.setdefault(r["r"], set()).add(r["name"])
    full = 0
    while len(done.get(full, ())) == len(BOOT_STUDIES):
        full += 1
    first = full if first is None else min(first, full)
    rows = [r for r in rows if r["r"] < first]
    summary = {}
    for name, (seed, dist, truth, _) in BOOT_STUDIES.items():
        mine = sorted(
            (r for r in rows if r["name"] == name), key=lambda r: r["r"]
        )
        if not mine:
            continue
        names = list(truth)
        values = np.array([truth[k] for k in names])
        reps = len(mine)
        block: dict = {
            "seed": seed,
            "distribution": dist,
            "truth": truth,
            "replicates": reps,
            "refits_failed": {
                "param": int(sum(r["param_failed"] for r in mine)),
                "case": int(sum(r["case_failed"] for r in mine)),
            },
            "on_edge_share": None,
            "coverage": {},
        }
        est = np.array([r["estimate"] for r in mine])
        if "lfp_p" in truth:
            block["on_edge_share"] = float(
                np.mean(est[:, names.index("lfp_p")] == 1.0)
            )
        if "f0" in truth:
            block["on_edge_share"] = float(
                np.mean(est[:, names.index("f0")] == 0.0)
            )
        edge = block["on_edge_share"]
        print(f"\n{name} (seed {seed}, {reps} replicates; on edge {edge})")
        print(
            f"{'method':<11}{'level':<6}" + "".join(f"{k:>22}" for k in names)
        )
        for level in LEVELS:
            nominal = float(level)
            for m in METHODS:
                b = np.array([r[f"{m}_{level}"] for r in mine], dtype=float)
                hit = (b[:, :, 0] <= values) & (values <= b[:, :, 1])
                cov = hit.mean(axis=0)
                width = np.nanmedian(b[:, :, 1] - b[:, :, 0], axis=0)
                cells = {}
                line = f"{m:<11}{level:<6}"
                for j, k in enumerate(names):
                    target = nominal
                    if name == "lfp_edge" and k == "lfp_p":
                        target = 1 - (1 - nominal) / 2
                    se = math.sqrt(cov[j] * (1 - cov[j]) / reps)
                    cells[k] = {
                        "coverage": round(float(cov[j]), 4),
                        "mc_se": round(se, 4),
                        "target": round(target, 4),
                        "median_width": round(float(width[j]), 4),
                    }
                    line += f"{cov[j]:>12.3f} +/- {se:.3f}"
                block["coverage"].setdefault(level, {})[m] = cells
                print(line)
        summary[name] = block
    tag = "" if first == full else f"_first{first}"
    with open(os.path.join(HERE, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
