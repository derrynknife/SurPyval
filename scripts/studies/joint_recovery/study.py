"""The joint model's recovery study (#891): 200 simulated fleets per path
model, each of 300 pumps read monthly for up to two years, the hazard
Weibull PH (alpha 800 days, beta 2 at a marker of 4) with a marker effect
of 0.9 per unit; fitted with marker_reference=4, so the truth is in the
model's own parameterisation.

    python scripts/studies/joint_recovery/study.py <path> [reps]

    path: linear, exp, quad, loglink

Each fit is appended to results_<path>.jsonl as it finishes; a rerun
skips the seeds already there (so a killed run resumes). At the end the
summary is written to summary_<path>.json and printed. At most 4 worker
processes.
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

#: The results go beside this script (run it from the repository root
#: with the package importable, e.g. pip install -e .).
HERE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 1000

import numpy as np  # noqa: E402

WHICH = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
#: A diagnostic only: GH=<points> refits with that many quadrature points
#: a dimension (the package's is 7), to separate quadrature error from
#: the estimator's own bias; its files are tagged _gh<points>.
GH = os.environ.get("GH")
TAG = WHICH if GH is None else f"{WHICH}_gh{GH}"
#: A diagnostic only: N=<pumps> simulates fleets of that size (the
#: study's is 300), to see whether a bias shrinks as 1/n; tagged _n<N>.
N = int(os.environ.get("N", "300"))
if N != 300:
    TAG += f"_n{N}"


def setup():
    """``(truth, simulation spec, fitter)`` of the path model."""
    from surpyval import JointPH, Weibull
    from surpyval.degradation import (
        ExponentialPath,
        LinearPath,
        LinkedPathModel,
        QuadraticPath,
    )

    if WHICH == "exp":
        truth = dict(
            a=3.0, b=0.0008, sd_a=0.4, sd_b=0.0003, sd_noise=0.3, corr_a_b=0.0
        )
        spec = (ExponentialPath, [3.0, 0.0008], [0.4, 0.0003], 0.3)
        fitter = JointPH(Weibull, ExponentialPath)
    elif WHICH == "loglink":
        path = LinkedPathModel(ExponentialPath, {"b": "log"})
        truth = {
            "a": 3.0,
            "log(b)": float(np.log(0.0008)),
            "sd_a": 0.4,
            "sd_log(b)": 0.3,
            "corr_a_log(b)": 0.0,
            "sd_noise": 0.3,
        }
        spec = (path, [3.0, np.log(0.0008)], [0.4, 0.3], 0.3)
        fitter = JointPH(Weibull, path)
    elif WHICH == "quad":
        truth = dict(
            a=3.0,
            b=0.5 / 365,
            c=1.0 / 365**2,
            sd_a=0.5,
            sd_b=0.3 / 365,
            sd_noise=0.4,
            corr_a_b=0.0,
        )
        spec = (
            QuadraticPath,
            [3.0, 0.5 / 365, 1.0 / 365**2],
            [0.5, 0.3 / 365, 0.0],
            0.4,
        )
        fitter = JointPH(Weibull, QuadraticPath, varying=("a", "b"))
    elif WHICH == "linear":
        truth = dict(
            a=3.0,
            b=1.0 / 365,
            sd_a=0.6,
            sd_b=0.5 / 365,
            sd_noise=0.7,
            corr_a_b=0.0,
        )
        spec = (LinearPath, [3.0, 1.0 / 365], [0.6, 0.5 / 365], 0.7)
        fitter = JointPH(Weibull, LinearPath)
    else:
        raise SystemExit(f"unknown path {WHICH!r}")
    truth.update(alpha=800.0, beta=2.0, marker_effect=0.9)
    return truth, spec, fitter


def one(seed):
    warnings.simplefilter("ignore")
    from surpyval.tests.univariate.joint._fleet import simulate_path

    if GH is not None:
        import surpyval.univariate.joint._likelihood as lk

        lk.GH_FIT = int(GH)

    truth, spec, fitter = setup()
    data = simulate_path(*spec, n=N, seed=seed)
    t0 = time.time()
    try:
        m = fitter.fit(data, marker_reference=4.0)
        out = dict(
            seed=seed,
            ok=True,
            maximum=m.maximum,
            names=list(m.parameter_names),
            params=np.asarray(m.params, float).tolist(),
            se=np.asarray(m.standard_errors(), float).tolist(),
            n_events=float(m.n_events),
        )
    except Exception as error:  # recorded, not hidden
        out = dict(seed=seed, ok=False, error=repr(error))
    out["time"] = time.time() - t0
    return out


def summarise(results, truth):
    fits = [r for r in results if r["ok"]]
    names = fits[0]["names"]
    P = np.array([r["params"] for r in fits])
    S = np.array([r["se"] for r in fits])
    R = len(fits)
    rows = []
    for j, name in enumerate(names):
        t = truth[name]
        est, se = P[:, j], S[:, j]
        finite = np.isfinite(se)
        err = est - t
        scale = abs(t) if t != 0 else 1.0
        bias = float(np.mean(err) / scale)
        bias_mcse = float(np.std(err, ddof=1) / np.sqrt(R) / scale)
        cover = np.abs(err[finite]) <= 1.959964 * se[finite]
        p = float(np.mean(cover))
        rows.append(
            dict(
                name=name,
                truth=t,
                relative=t != 0,
                mean=float(np.mean(est)),
                bias=bias,
                bias_mcse=bias_mcse,
                bias_z=bias / bias_mcse if bias_mcse > 0 else float("nan"),
                coverage=p,
                coverage_mcse=float(np.sqrt(p * (1 - p) / finite.sum())),
                sd_over_se=float(np.std(est, ddof=1) / np.mean(se[finite])),
                finite_se=int(finite.sum()),
            )
        )
    return dict(
        path=WHICH,
        reps=len(results),
        fitted=R,
        failed=[r for r in results if not r["ok"]],
        verified=sum(r.get("maximum") == "verified" for r in fits),
        maxima=sorted({str(r.get("maximum")) for r in fits}),
        fit_time_total=float(sum(r["time"] for r in results)),
        fit_time_mean=float(np.mean([r["time"] for r in results])),
        rows=rows,
    )


if __name__ == "__main__":
    log = os.path.join(HERE, f"results_{TAG}.jsonl")
    done = {}
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["seed"]] = r
    todo = [s for s in range(SEED0, SEED0 + REPS) if s not in done]
    start = time.time()
    with Pool(4) as pool, open(log, "a") as f:
        for r in pool.imap_unordered(one, todo):
            done[r["seed"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
    wall = time.time() - start
    truth, _, _ = setup()
    results = [done[s] for s in range(SEED0, SEED0 + REPS)]
    summary = summarise(results, truth)
    summary["wall_time_this_run"] = wall
    summary["date"] = time.strftime("%Y-%m-%d")
    with open(os.path.join(HERE, f"summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        TAG,
        "reps",
        summary["reps"],
        "fitted",
        summary["fitted"],
        "verified",
        summary["verified"],
        "maxima",
        summary["maxima"],
        "fit time total %.0f s" % summary["fit_time_total"],
        "wall %.0f s" % wall,
    )
    for row in summary["rows"]:
        flag = ""
        if not 0.92 <= row["coverage"] <= 0.98:
            flag += " COVERAGE"
        if abs(row["bias_z"]) > 2:
            flag += " BIAS"
        kind = "rel" if row["relative"] else "abs"
        print(
            f"{row['name']:15s} {kind} bias {row['bias']:+.4f} "
            f"({row['bias_mcse']:.4f})  coverage {row['coverage']:.3f} "
            f"({row['coverage_mcse']:.3f})  sd/se {row['sd_over_se']:.2f}"
            f"{flag}"
        )
