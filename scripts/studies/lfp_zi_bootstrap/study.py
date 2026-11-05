"""Coverage of the bootstrap bounds of limited-failure and zero-inflated
fits: the package's parametric (model) bootstrap with BCa intervals
against a case bootstrap (the units resampled with replacement), on the
designs and data seeds of the calibration study
``surpyval/tests/calibration/test_coverage_extra_parameters.py::
test_bootstrap_coverage``.

    python scripts/studies/lfp_zi_bootstrap/study.py [reps] [n_jobs]

Replicate ``r`` of a design is the ``r``-th data set the calibration
study draws from the design's seed (311-314), so replicates 0-199 are its
data exactly and further ones continue the same stream. For each data set
the fit is ``Weibull`` (or ``LogNormal``) with ``lfp=True`` or
``zi=True``, and two-sided 90% and 95% bounds on every parameter
(``lfp_p`` or ``f0`` among them) are computed five ways:

- ``param200``: the parametric bootstrap with BCa intervals as the
  package computes them (``param_cb(method="bootstrap")``) at ``n_boot =
  200`` and ``random_state = 1000 * seed + r``: the calibration study's
  bounds exactly, for the replicates it runs.
- ``case200``: a case bootstrap -- the 100 units drawn i.i.d. with
  replacement and the same model refitted, ``lfp_p`` and ``f0`` put on
  their edge where the data put them there, as the package does for its
  refits -- with BCa intervals, the acceleration the same jackknife's.
  Its seed is ``2000 * seed + r`` (fixed before the first run).
- ``casepct200``: the plain percentile interval of the 200 case refits.

The work plan was 1,000 replicates at ``n_boot = 500``; one replicate of
all four designs costs about 1,250 core-seconds at that size, so the
study runs at the calibration study's ``n_boot = 200`` for a fixed
wall-clock budget instead and reports the replicates it finished.

Each replicate is appended to ``results.jsonl`` beside this script as it
finishes; a rerun skips the replicates already there, so a killed run
resumes. ``summary.py`` turns it into ``summary.json``. Run from the
repository root with the package importable.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.jsonl")
N_BOOT = 200
LEVELS = (0.05, 0.10)


def designs():
    """The calibration study's designs: ``{name: (seed, dist, truth,
    probs)}``, and its data generator."""
    from surpyval.tests.calibration.test_coverage_extra_parameters import (
        BOOT_STUDIES,
        _boot_sample,
    )

    return BOOT_STUDIES, _boot_sample


def datasets(reps):
    """Every design's first ``reps`` data sets, drawn as the calibration
    study draws them."""
    studies, sample = designs()
    jobs = []
    for name, (seed, _, _, _) in studies.items():
        rng = np.random.default_rng(seed)
        for r in range(reps):
            x, c = sample(name, rng)
            jobs.append((name, r, x, c))
    # Replicate by replicate across the designs, so a run cut short has
    # the same number of each.
    jobs.sort(key=lambda j: (j[1], j[0]))
    return jobs


def replicate(name, r, x, c):
    import surpyval as sp
    from surpyval.univariate.parametric import _bootstrap as bs
    from surpyval.univariate.regression._bootstrap import (
        REFIT_ERRORS,
        bca_bounds,
    )

    studies, _ = designs()
    seed, dist, truth, _ = studies[name]
    kind = "zi" if "f0" in truth else "lfp"
    t0 = time.time()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model = getattr(sp, dist).fit(x, c=c, **{kind: True})
        names = list(truth)
        idx = [bs._point_index(model, k) for k in names]

        # The package's parametric bootstrap and its jackknife.
        fits = bs.refits(model, N_BOOT, 1000 * seed + r)
        jack = bs.jackknife(model, fits)
        a = bs.jackknife_acceleration(jack.values[:, idx], jack.weights)
        est = np.array([fits.estimate[i] for i in idx])
        pvals = fits.values[:, idx]

        # The case bootstrap.
        rng = np.random.default_rng(2000 * seed + r)
        fixed, init = bs._refit_options(model)
        cvals, case_failed = [], 0
        for _ in range(N_BOOT):
            k = rng.integers(0, x.size, size=x.size)
            xk, ck = x[k], c[k]
            try:
                refit = model.dist.fit(
                    xk, c=ck, fixed=fixed, init=init, **{kind: True}
                )
            except REFIT_ERRORS:
                case_failed += 1
                continue
            point = bs._at_edges(refit, bs._point(refit), xk, ck, {})
            cvals.append([point[i] for i in idx])
        cvals = np.array(cvals, dtype=float).reshape(-1, len(idx))

        out = {
            "name": name,
            "r": r,
            "estimate": est.tolist(),
            "a": a.tolist(),
            "param_failed": int(fits.failed),
            "case_failed": case_failed,
        }
        for alpha in LEVELS:
            lv = f"{1 - alpha:.2f}"
            out[f"param200_{lv}"] = bca_bounds(
                pvals, est, a, alpha, "two-sided"
            ).tolist()
            out[f"case200_{lv}"] = bca_bounds(
                cvals, est, a, alpha, "two-sided"
            ).tolist()
            out[f"casepct200_{lv}"] = np.quantile(
                cvals, [alpha / 2, 1 - alpha / 2], axis=0
            ).T.tolist()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
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
