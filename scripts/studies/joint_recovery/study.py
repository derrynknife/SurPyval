"""The joint model's recovery study (#891): 200 simulated fleets per path
model, each of 300 pumps read monthly for up to two years, the hazard
Weibull PH (alpha 800 days, beta 2 at a marker of 4) with a marker effect
of 0.9 per unit; fitted with marker_reference=4, so the truth is in the
model's own parameterisation. The accelerated model's fleets (``aft_``
before the path's name) are the same pumps ageing at the rate exp(0.45
(marker - 4)) against a Weibull (800 days, 2) of effective age, fitted
with JointAFT.

    python scripts/studies/joint_recovery/study.py <path> [reps]

    path: linear, exp, quad, loglink, aft_linear, aft_exp, aft_quad,
    aft_loglink, cubic

``cubic`` is the study of four varying parameters: fleets of 100 items
whose marker is a cubic in the age (years) with every coefficient its
own, read every 0.1 years to failure or 3 years (the fleets of
``surpyval/tests/univariate/joint/test_joint_integration.py``:
``simulate(100, seed, q=4)``), fitted with marker_reference=3.6.

INTEGRATION=<name> fits with that integral (``fit(...,
integration=...)``; the default is the fit's own); tagged _<name>.
PROCESSES=<n> uses that many worker processes (4 unless given).

EVERY=<days> reads the pumps that often (the study's is 30); tagged
_every<days>.

HOW=two-stage fits the same fleets by the approximate two-stage estimate
(``fit(..., how="two-stage")``) in place of the joint likelihood; its
files are tagged _two_stage. Its marker part has no standard errors, so
those parameters' coverage is not recorded (null).

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
#: The accelerated survival part: its fleets and fitter.
AFT = WHICH.startswith("aft_")
PATH = WHICH[4:] if AFT else WHICH
#: The marker effect of the simulation (per unit of the marker).
EFFECT = 0.45 if AFT else 0.9
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
#: Days between readings (the study's is monthly, 30); tagged _every<days>.
EVERY = float(os.environ.get("EVERY", "30"))
if EVERY != 30:
    TAG += f"_every{EVERY:g}"
#: The way to fit: "MLE" (the study's), or "two-stage", the approximate
#: estimate it is compared with; tagged _two_stage.
HOW = os.environ.get("HOW", "MLE")
if HOW != "MLE":
    TAG += "_" + HOW.replace("-", "_")
#: How each item's integral is done (None: the fit's default); tagged.
INTEGRATION = os.environ.get("INTEGRATION")
if INTEGRATION is not None:
    TAG += f"_{INTEGRATION}"
PROCESSES = int(os.environ.get("PROCESSES", "4"))
#: The cubic study's fleet size and the marker the hazard is quoted at.
CUBIC_N, CUBIC_CENTRE = 100, 3.6


def cubic_setup():
    """``(truth, fitter)`` of the four-parameter study."""
    from surpyval import JointPH, Weibull
    from surpyval.tests.univariate.joint import test_joint_integration as t

    names = list(t.CUBIC.parameter_names)
    truth = dict(zip(names, t.MEANS[:4].tolist()))
    truth.update({f"sd_{n}": float(s) for n, s in zip(names, t.SDS[:4])})
    truth.update({f"corr_{a}_{b}": 0.0 for a in names for b in names})
    truth.update(
        sd_noise=t.NOISE, alpha=t.ALPHA, beta=t.BETA, marker_effect=t.EFFECT
    )
    return truth, JointPH(Weibull, t.CUBIC)


def setup():
    """``(truth, simulation spec, fitter)`` of the path model."""
    from surpyval import JointAFT, JointPH, Weibull
    from surpyval.degradation import (
        ExponentialPath,
        LinearPath,
        LinkedPathModel,
        QuadraticPath,
    )

    Joint = JointAFT if AFT else JointPH
    if PATH == "cubic":
        truth, fitter = cubic_setup()
        return truth, None, fitter
    if PATH == "exp":
        truth = dict(
            a=3.0, b=0.0008, sd_a=0.4, sd_b=0.0003, sd_noise=0.3, corr_a_b=0.0
        )
        spec = (ExponentialPath, [3.0, 0.0008], [0.4, 0.0003], 0.3)
        fitter = Joint(Weibull, ExponentialPath)
    elif PATH == "loglink":
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
        fitter = Joint(Weibull, path)
    elif PATH == "quad":
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
        fitter = Joint(Weibull, QuadraticPath, varying=("a", "b"))
    elif PATH == "linear":
        truth = dict(
            a=3.0,
            b=1.0 / 365,
            sd_a=0.6,
            sd_b=0.5 / 365,
            sd_noise=0.7,
            corr_a_b=0.0,
        )
        spec = (LinearPath, [3.0, 1.0 / 365], [0.6, 0.5 / 365], 0.7)
        fitter = Joint(Weibull, LinearPath)
    else:
        raise SystemExit(f"unknown path {WHICH!r}")
    truth.update(alpha=800.0, beta=2.0, marker_effect=EFFECT)
    return truth, spec, fitter


def one(seed):
    warnings.simplefilter("ignore")
    from surpyval.tests.univariate.joint._fleet import simulate_path

    if GH is not None:
        import surpyval.univariate.joint._likelihood as lk

        lk.GH_FIT = int(GH)

    truth, spec, fitter = setup()
    options = {} if INTEGRATION is None else {"integration": INTEGRATION}
    if PATH == "cubic":
        from surpyval.tests.univariate.joint import test_joint_integration

        arrays = test_joint_integration.simulate(CUBIC_N, seed, q=4)
        options.update(arrays, marker_reference=CUBIC_CENTRE)
        data = None
    else:
        data = simulate_path(
            *spec,
            n=N,
            seed=seed,
            every=EVERY,
            effect=EFFECT,
            survival="AFT" if AFT else "PH",
        )
        options.update(marker_reference=4.0)
    t0 = time.time()
    try:
        if data is None:
            m = fitter.fit(how=HOW, **options)
        else:
            m = fitter.fit(data, how=HOW, **options)
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
        if not finite.any():
            # no standard errors (a two-stage fit's marker part)
            rows.append(
                dict(
                    name=name,
                    truth=t,
                    relative=t != 0,
                    mean=float(np.mean(est)),
                    bias=bias,
                    bias_mcse=bias_mcse,
                    bias_z=bias / bias_mcse if bias_mcse > 0 else None,
                    coverage=None,
                    coverage_mcse=None,
                    sd_over_se=None,
                    finite_se=0,
                )
            )
            continue
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
        how=HOW,
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
    with Pool(PROCESSES) as pool, open(log, "a") as f:
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
        kind = "rel" if row["relative"] else "abs"
        line = (
            f"{row['name']:15s} {kind} bias {row['bias']:+.4f} "
            f"({row['bias_mcse']:.4f})"
        )
        if row["bias_z"] is not None and abs(row["bias_z"]) > 2:
            flag += " BIAS"
        if row["coverage"] is None:
            line += "  no standard errors"
        else:
            if not 0.92 <= row["coverage"] <= 0.98:
                flag += " COVERAGE"
            line += (
                f"  coverage {row['coverage']:.3f} "
                f"({row['coverage_mcse']:.3f})  sd/se "
                f"{row['sd_over_se']:.2f}"
            )
        print(line + flag)
