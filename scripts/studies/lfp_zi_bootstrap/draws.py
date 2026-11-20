"""Does the number of bootstrap draws explain the short 90% intervals of
limited-failure and zero-inflated fits? The package's bootstrap bounds
(parametric bootstrap, BCa with the jackknife's acceleration) at
``n_boot = 200`` and ``n_boot = 500`` on the same data sets: those of the
calibration study
``surpyval/tests/calibration/test_coverage_extra_parameters.py::
test_bootstrap_coverage``.

    python scripts/studies/lfp_zi_bootstrap/draws.py [reps] [n_jobs]

Replicate ``r`` of a design is the ``r``-th data set the calibration
study draws from the design's seed (311-314), so replicates 0-199 are its
data exactly. Each fit gets the bounds the calibration study checks --
``param_cb`` on every parameter, ``cb`` (``sf``) and ``quantile_cb`` at
three true quantiles, two-sided 90% and 95% -- through the public calls
with ``random_state = 1000 * seed + r`` (the calibration's), at
``n_boot = 500`` and at ``n_boot = 200``.

The refits are drawn one resample after another from one stream, so the
first 200 of the 500 are the 200 refits the calibration study uses. The
study draws the 500 once and gives the bounds at 200 from their first
200 (put in the model's refit cache, where the public calls read them),
so the two counts differ only in the extra 300 draws: a paired
comparison. Where a refit of the 500 failed (left out, which would shift
the subset) the 200 are drawn afresh instead. On the first two replicates
of each design the subset is checked against a fresh draw of 200
(``nested`` in the results). The jackknife, and so the acceleration, is
the same at both counts.

Each replicate is appended to ``draws_results.jsonl`` beside this
script as it finishes; a rerun skips the replicates already there, so a
killed run resumes. ``draws_summary.py`` turns it into
``draws_summary.json``. Run from the repository root with the package
importable.
"""

import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "draws_results.jsonl")
COUNTS = (200, 500)
LEVELS = (0.05, 0.10)
CHECK_NESTED = 2


def designs():
    """The calibration study's designs ``{name: (seed, dist, truth,
    probs)}``, its data generator and its true model."""
    from surpyval.tests.calibration.test_coverage_extra_parameters import (
        BOOT_STUDIES,
        _boot_sample,
        _boot_truth,
    )

    return BOOT_STUDIES, _boot_sample, _boot_truth


def datasets(reps):
    """Every design's first ``reps`` data sets, drawn as the calibration
    study draws them, replicate by replicate across the designs so that a
    run cut short has the same number of each."""
    studies, sample, _ = designs()
    jobs = []
    for name, (seed, _, _, _) in studies.items():
        rng = np.random.default_rng(seed)
        for r in range(reps):
            x, c = sample(name, rng)
            jobs.append((name, r, x, c))
    jobs.sort(key=lambda j: (j[1], j[0]))
    return jobs


def _bounds(model, names, sf_at, p, n_boot, seed):
    """The public bootstrap bounds the calibration study checks."""
    boot = {"method": "bootstrap", "n_boot": n_boot, "random_state": seed}
    out = {}
    for a in LEVELS:
        lv = f"{1 - a:.2f}"
        params = [model.param_cb(k, alpha_ci=a, **boot) for k in names]
        out[f"param_{lv}"] = np.array(params).tolist()
        out[f"cb_{lv}"] = np.asarray(
            model.cb(sf_at, alpha_ci=a, **boot)
        ).tolist()
        out[f"qf_{lv}"] = np.asarray(
            model.quantile_cb(p, alpha_ci=a, **boot)
        ).tolist()
    return out


def replicate(name, r, x, c):
    import surpyval as sp
    from surpyval.univariate.parametric import _bootstrap as bs

    studies, _, truth_of = designs()
    seed, dist, truth, probs = studies[name]
    kind = "zi" if "f0" in truth else "lfp"
    p = np.array(probs)
    sf_at = truth_of(name).qf(p)
    names = list(truth)
    rs = 1000 * seed + r
    t0 = time.time()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model = getattr(sp, dist).fit(x, c=c, **{kind: True})
        big = bs.refits(model, COUNTS[1], rs)
        out = {
            "name": name,
            "r": r,
            "failures": int(np.sum(c == 0)),
            # Each parameter's estimate, lfp_p and f0 on their edge where
            # the data put them there (as the bounds read them).
            "estimate": [
                big.estimate[bs._point_index(model, k)] for k in names
            ],
            "failed_500": int(big.failed),
            "nested": None,
        }
        if big.failed == 0:
            small = dataclasses.replace(
                big,
                n_boot=COUNTS[0],
                models=big.models[: COUNTS[0]],
                values=big.values[: COUNTS[0]],
            )
            if r < CHECK_NESTED:
                fresh = bs._draw(
                    model,
                    bs._UnivariateDesign(model),
                    bs._point(model),
                    COUNTS[0],
                    np.random.default_rng(rs),
                )
                out["nested"] = bool(
                    np.array_equal(fresh.values, small.values)
                )
            model._bootstrap_refits[(COUNTS[0], rs)] = small
        for n_boot in COUNTS:
            for key, value in _bounds(
                model, names, sf_at, p, n_boot, rs
            ).items():
                out[f"{key}_{n_boot}"] = value
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                row = json.loads(line)
                done.add((row["name"], row["r"]))
    jobs = [j for j in datasets(reps) if (j[0], j[1]) not in done]
    print(f"{len(jobs)} replicates to run ({len(done)} done)", flush=True)
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(replicate)(*job) for job in jobs
    )
    with open(OUT, "a") as f:
        for k, row in enumerate(results, 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
