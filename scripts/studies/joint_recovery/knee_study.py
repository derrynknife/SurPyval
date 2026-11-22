"""The spline path's recovery study: fleets whose marker stays low and
then climbs (a knee), fitted with a spline path and with a straight line
on the same data.

    python scripts/studies/joint_recovery/knee_study.py [reps]

The design, fixed before the first run:

- 200 fleets (seeds 1000 to 1199) of ``N`` pumps, each read every 30
  days for up to 730 days.
- Each pump's marker is the spline ``SplinePath(knots=[365],
  boundary=(0, 730))`` with level 3.0 (sd 0.5 between pumps), rate 0 a
  day (sd 0.3 / 365) and the bend ``s1`` = 3.0 (common), read with noise
  of sd 0.4: about 3.0 at age 0, 3.4 at 365 days and 5.25 at 730.
- The hazard is Weibull proportional hazards (alpha 800 days, beta 2 at
  a marker of 4) with a marker effect of 0.9 per unit.
- Each fleet is fitted twice, with ``marker_reference=4``: ``JointPH(
  Weibull, SplinePath(knots=[365], boundary=(0, 730)), varying=("level",
  "rate"))``, whose parameters the truth is in, and ``JointPH(Weibull,
  LinearPath)``.
- Dynamic predictions: for the first 40 pumps of each fleet still
  running at 450 days, each model's prediction from the readings up to
  450 days: the smoothed marker at 450 against the pump's true marker
  there, and the chance of surviving to 540 days against the pump's true
  conditional survival (from its true path).

Each fleet is appended to results_knee.jsonl as it finishes; a rerun
skips the seeds already there. The summary is written to
summary_knee.json and printed. PROCS=<n> sets the worker processes
(default 2); N=<pumps> the fleet size (the study's is 300), tagged
_n<N>.
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 1000
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PROCS = int(os.environ.get("PROCS", "2"))
N = int(os.environ.get("N", "300"))
TAG = "knee" if N == 300 else f"knee_n{N}"

KNOTS, BOUNDARY = [365.0], (0.0, 730.0)
MEAN = np.array([3.0, 0.0, 3.0])
SD = np.array([0.5, 0.3 / 365, 0.0])
NOISE = 0.4
ALPHA, BETA, EFFECT, CENTRE = 800.0, 2.0, 0.9, 4.0
EVERY, FOLLOW = 30.0, 730.0
LANDMARK, HORIZON, PREDICTED = 450.0, 540.0, 40

TRUTH = {
    "level": 3.0,
    "rate": 0.0,
    "s1": 3.0,
    "sd_level": 0.5,
    "sd_rate": 0.3 / 365,
    "corr_level_rate": 0.0,
    "sd_noise": NOISE,
    "alpha": ALPHA,
    "beta": BETA,
    "marker_effect": EFFECT,
}


def spline():
    from surpyval.degradation import SplinePath

    return SplinePath(knots=KNOTS, boundary=BOUNDARY)


def simulate(seed):
    """``(data, thetas, lives)``: the fleet, each pump's true path
    parameters and its lifetime (drawn as
    ``surpyval.tests.univariate.joint._fleet.simulate_path`` draws)."""
    from surpyval import ItemLifetime, SurpyvalData

    path = spline()
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 20 * FOLLOW, 40001)
    items, thetas, lives = [], [], []
    for k in range(N):
        theta = MEAN + SD * rng.normal(size=MEAN.size)
        m = path.path(grid, *theta)
        h = (
            (BETA / ALPHA)
            * (grid / ALPHA) ** (BETA - 1)
            * np.exp(EFFECT * (m - CENTRE))
        )
        h = np.where(np.isfinite(h), h, 1e300)
        H = np.concatenate(
            [[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(grid))]
        )
        life = float(np.interp(rng.exponential(), H, grid))
        end = min(life, FOLLOW)
        ages = np.arange(0.0, end + 1e-9, EVERY)
        readings = path.path(ages, *theta) + rng.normal(0, NOISE, ages.size)
        outcome = "observed" if life <= FOLLOW else "right_censored"
        items.append(
            ItemLifetime(
                k,
                **{outcome: float(end)},
                markers=dict(zip(ages.tolist(), readings.tolist())),
            )
        )
        thetas.append(theta)
        lives.append(life)
    return SurpyvalData.from_items(items), items, thetas, lives


def true_survival(theta):
    """A pump's true chance of surviving from the landmark to the
    horizon."""
    grid = np.linspace(LANDMARK, HORIZON, 2001)
    m = spline().path(grid, *theta)
    h = (
        (BETA / ALPHA)
        * (grid / ALPHA) ** (BETA - 1)
        * np.exp(EFFECT * (m - CENTRE))
    )
    return float(np.exp(-np.sum((h[1:] + h[:-1]) / 2 * np.diff(grid))))


def fitted(model):
    return dict(
        maximum=model.maximum,
        names=list(model.parameter_names),
        params=np.asarray(model.params, float).tolist(),
        se=np.asarray(model.standard_errors(), float).tolist(),
        aic=float(model.aic()),
    )


def one(seed):
    warnings.simplefilter("ignore")
    from surpyval import JointPH, Weibull
    from surpyval.degradation import LinearPath

    data, items, thetas, lives = simulate(seed)
    fitters = {
        "spline": JointPH(Weibull, spline(), varying=("level", "rate")),
        "line": JointPH(Weibull, LinearPath),
    }
    out = dict(seed=seed, ok=True)
    t0 = time.time()
    try:
        models = {
            name: fitter.fit(data, marker_reference=CENTRE)
            for name, fitter in fitters.items()
        }
        running = [k for k in range(N) if lives[k] > LANDMARK][:PREDICTED]
        true_level = np.array(
            [float(spline().path(LANDMARK, *thetas[k])) for k in running]
        )
        true_sf = np.array([true_survival(thetas[k]) for k in running])
        for name, model in models.items():
            out[name] = fitted(model)
            level, sf = [], []
            for k in running:
                read = {
                    a: v for a, v in items[k].markers.items() if a <= LANDMARK
                }
                prediction = model.predict(markers=read, given=LANDMARK)
                level.append(float(prediction.marker(LANDMARK)))
                sf.append(float(prediction.sf(HORIZON)))
            out[name]["level_error"] = (np.array(level) - true_level).tolist()
            out[name]["sf_error"] = (np.array(sf) - true_sf).tolist()
        out["true_sf_mean"] = float(np.mean(true_sf))
    except Exception as error:  # recorded, not hidden
        out = dict(seed=seed, ok=False, error=repr(error))
    out["time"] = time.time() - t0
    return out


def rows(fits, truth):
    """Bias and coverage of each parameter with a truth."""
    names = fits[0]["names"]
    P = np.array([r["params"] for r in fits])
    S = np.array([r["se"] for r in fits])
    R = len(fits)
    out = []
    for j, name in enumerate(names):
        if name not in truth:
            continue
        t = truth[name]
        err = P[:, j] - t
        scale = abs(t) if t != 0 else 1.0
        finite = np.isfinite(S[:, j])
        cover = np.abs(err[finite]) <= 1.959964 * S[finite, j]
        p = float(np.mean(cover))
        mcse = float(np.std(err, ddof=1) / np.sqrt(R) / scale)
        out.append(
            dict(
                name=name,
                truth=t,
                relative=t != 0,
                mean=float(np.mean(P[:, j])),
                bias=float(np.mean(err) / scale),
                bias_mcse=mcse,
                bias_z=float(np.mean(err) / scale / mcse),
                coverage=p,
                coverage_mcse=float(np.sqrt(p * (1 - p) / finite.sum())),
                sd_over_se=float(
                    np.std(P[:, j], ddof=1) / np.mean(S[finite, j])
                ),
                finite_se=int(finite.sum()),
            )
        )
    return out


def errors(fits, key):
    """The mean and root-mean-square error over every prediction, and the
    Monte Carlo standard error of the mean from the fleets' means."""
    every = np.concatenate([np.asarray(r[key], float) for r in fits])
    means = np.array([np.mean(r[key]) for r in fits if len(r[key])])
    return dict(
        predictions=int(every.size),
        mean=float(np.mean(every)),
        mean_mcse=float(np.std(means, ddof=1) / np.sqrt(means.size)),
        rmse=float(np.sqrt(np.mean(every**2))),
    )


def summarise(results):
    good = [r for r in results if r["ok"]]
    out = dict(
        reps=len(results),
        fitted=len(good),
        failed=[r for r in results if not r["ok"]],
        fit_time_mean=float(np.mean([r["time"] for r in results])),
        true_sf_mean=float(np.mean([r["true_sf_mean"] for r in good])),
        spline_lower_aic=sum(
            r["spline"]["aic"] < r["line"]["aic"] for r in good
        ),
    )
    line_truth = {k: TRUTH[k] for k in ("alpha", "beta", "marker_effect")}
    for name, truth in (("spline", TRUTH), ("line", line_truth)):
        fits = [r[name] for r in good]
        out[name] = dict(
            verified=sum(f["maximum"] == "verified" for f in fits),
            rows=rows(fits, truth),
            level=errors(fits, "level_error"),
            survival=errors(fits, "sf_error"),
        )
    return out


if __name__ == "__main__":
    log = os.path.join(HERE, f"results_{TAG}.jsonl")
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
    with open(os.path.join(HERE, f"summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
