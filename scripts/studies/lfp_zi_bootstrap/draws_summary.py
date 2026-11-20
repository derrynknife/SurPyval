"""Summarise ``draws_results.jsonl`` (``draws.py``) into
``draws_summary.json`` and print a table.

    python scripts/studies/lfp_zi_bootstrap/draws_summary.py

For each design, level, kind of bound (``param``, ``cb``, ``qf``) and
quantity: the coverage of the two-sided interval at ``n_boot`` 200 and
500 (a ``nan`` bound a miss, as in the calibration study), the target
(on the edge ``lfp_p``'s is ``1 - alpha / 2``), whether the calibration
rule (``Z_TOL`` Monte Carlo errors at the target plus slack 0.01) holds,
the replicates covered at one count and not the other (the paired
comparison) and the median widths. Then, per level and count, the mean
of every cell whose target is the nominal level (the calibration's pooled
report), and the coverage of the distribution's own parameters split by
the number of failures in the data set (below, or at and above, the
design's median).
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTS = (200, 500)
LEVELS = ("0.90", "0.95")
KINDS = ("param", "cb", "qf")
Z_TOL = 3.0
SLACK = 0.01


def _rows():
    with open(os.path.join(HERE, "draws_results.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    # Only the replicates every design finished: the largest R with every
    # r < R done in all four.
    done: dict = {}
    for r in rows:
        done.setdefault(r["r"], set()).add(r["name"])
    names = {r["name"] for r in rows}
    full = 0
    while done.get(full, set()) == names:
        full += 1
    return [r for r in rows if r["r"] < full], full


def _truths(name, truth, probs):
    from surpyval.tests.calibration.test_coverage_extra_parameters import (
        _boot_truth,
    )

    true = _boot_truth(name)
    p = np.array(probs)
    return {
        "param": (list(truth), np.array(list(truth.values()))),
        "cb": ([f"sf at q{q}" for q in probs], true.sf(true.qf(p))),
        "qf": ([f"qf({q})" for q in probs], true.qf(p)),
    }


def main() -> None:
    from surpyval.tests.calibration.test_coverage_extra_parameters import (
        BOOT_STUDIES,
    )

    rows, reps = _rows()
    summary: dict = {"replicates": reps, "designs": {}}
    pooled: dict = {}
    by_failures: dict = {}
    print(f"{reps} replicates per design")
    for name, (seed, dist, truth, probs) in BOOT_STUDIES.items():
        mine = sorted(
            (r for r in rows if r["name"] == name), key=lambda r: r["r"]
        )
        failures = np.array([r["failures"] for r in mine])
        est = np.array([r["estimate"] for r in mine])
        names = list(truth)
        extra = "lfp_p" if "lfp_p" in truth else "f0"
        edge = 1.0 if extra == "lfp_p" else 0.0
        block: dict = {
            "seed": seed,
            "distribution": dist,
            "truth": truth,
            "failures": {
                "median": float(np.median(failures)),
                "min": int(failures.min()),
                "max": int(failures.max()),
            },
            "on_edge_share": float(
                np.mean(est[:, names.index(extra)] == edge)
            ),
            "refits_failed_500": int(sum(r["failed_500"] for r in mine)),
            "nested_checked": [r["nested"] for r in mine if r["nested"]],
            "cells": {},
        }
        fewer = failures < np.median(failures)
        print(
            f"\n{name} (seed {seed}; failures median "
            f"{block['failures']['median']:.0f}, "
            f"{block['failures']['min']}-{block['failures']['max']}; "
            f"{extra} on its edge in {block['on_edge_share']:.3f})"
        )
        print(
            f"{'level':<6}{'bound':<20}{'n=200':>8}{'n=500':>8}{'target':>8}"
            f"{'only200':>9}{'only500':>9}{'width200':>10}{'width500':>10}"
        )
        for kind, (labels, values) in _truths(name, truth, probs).items():
            for level in LEVELS:
                nominal = float(level)
                hits = {}
                widths = {}
                for n in COUNTS:
                    b = np.array(
                        [r[f"{kind}_{level}_{n}"] for r in mine], dtype=float
                    )
                    hits[n] = (b[:, :, 0] <= values) & (values <= b[:, :, 1])
                    # A quantile beyond lfp_p has an infinite upper bound.
                    with np.errstate(all="ignore"):
                        widths[n] = np.nanmedian(
                            b[:, :, 1] - b[:, :, 0], axis=0
                        )
                for j, label in enumerate(labels):
                    on_edge = name == "lfp_edge" and label == "lfp_p"
                    target = 1 - (1 - nominal) / 2 if on_edge else nominal
                    tol = Z_TOL * math.sqrt(target * (1 - target) / reps)
                    tol += SLACK
                    cov = {n: float(hits[n][:, j].mean()) for n in COUNTS}
                    cell = {
                        "target": round(target, 4),
                        "tolerance": round(tol, 4),
                        "coverage": {str(n): cov[n] for n in COUNTS},
                        "passes": {
                            str(n): bool(abs(cov[n] - target) <= tol)
                            for n in COUNTS
                        },
                        "only_200": int(
                            np.sum(hits[200][:, j] & ~hits[500][:, j])
                        ),
                        "only_500": int(
                            np.sum(hits[500][:, j] & ~hits[200][:, j])
                        ),
                        "median_width": {
                            str(n): round(float(widths[n][j]), 5)
                            for n in COUNTS
                        },
                    }
                    block["cells"][f"{level} {kind} {label}"] = cell
                    flag = "" if all(cell["passes"].values()) else "  rule!"
                    print(
                        f"{level:<6}{kind + ' ' + label:<20}"
                        f"{cov[200]:>8.3f}{cov[500]:>8.3f}{target:>8.3f}"
                        f"{cell['only_200']:>9}{cell['only_500']:>9}"
                        f"{widths[200][j]:>10.4g}{widths[500][j]:>10.4g}"
                        + flag
                    )
                    if not on_edge:
                        for n in COUNTS:
                            for group in ("all", name):
                                pooled.setdefault((group, level, n), [])
                                pooled[(group, level, n)].append(cov[n])
                    if kind == "param" and label not in ("lfp_p", "f0"):
                        for n in COUNTS:
                            for part, mask in (
                                ("fewer", fewer),
                                ("more", ~fewer),
                            ):
                                key = (level, n, part)
                                got = by_failures.setdefault(key, [0, 0])
                                got[0] += int(hits[n][mask, j].sum())
                                got[1] += int(mask.sum())
        summary["designs"][name] = block
    print(
        "\nMean coverage over the cells whose target is the nominal level,"
        " every design and each"
    )
    summary["pooled"] = {}
    for (group, level, n), covs in sorted(
        pooled.items(), key=lambda kv: (kv[0][0] != "all",) + kv[0]
    ):
        mean = float(np.mean(covs))
        summary["pooled"][f"{group} {level} n_boot={n}"] = {
            "mean": round(mean, 4),
            "min": round(float(np.min(covs)), 4),
            "max": round(float(np.max(covs)), 4),
            "cells": len(covs),
        }
        print(
            f"  {group:<14}{level} n_boot={n}: {mean:.4f} "
            f"({np.min(covs):.3f}-{np.max(covs):.3f}, {len(covs)} cells)"
        )
    print(
        "\nDistribution parameters' coverage by the data set's failures"
        " (below the design's median, or at and above it)"
    )
    summary["by_failures"] = {}
    for (level, n, part), (hit, tot) in sorted(by_failures.items()):
        rate = hit / tot
        summary["by_failures"][f"{level} n_boot={n} {part}"] = {
            "coverage": round(rate, 4),
            "intervals": tot,
            "mc_se": round(math.sqrt(rate * (1 - rate) / tot), 4),
        }
        print(f"  {level} n_boot={n} {part:<6}{rate:.4f} of {tot}")
    with open(os.path.join(HERE, "draws_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
