"""Do the Wald bounds of ``ParametricCompetingRisks.cb`` cover?

The delta-method bounds of a parametric competing-risks fit, on the
cumulative incidence of each cause (``on="cif"``, on the logit scale) and
on the all-cause survival and cumulative hazard (``on="sf"``, ``"Hf"``,
one bound on ``log H``), checked by simulation. Each replicate draws
``n`` units with a latent failure time per cause and an independent
uniform censoring time, fits the model, and records whether the 95%
two-sided bound at each time contains the true value.

Scenarios (fixed before any result was seen):

- ``weibull2``: cause ``a`` Weibull(alpha 10, beta 2), cause ``b``
  Weibull(alpha 14, beta 1), censoring Uniform(0, 30); ``n`` = 100 and
  300; ``ParametricCompetingRisks.fit(x, e, dist=Weibull)``.
- ``lfp``: cause ``a`` Weibull(10, 2) of which only half the units are
  susceptible (``lfp_p`` 0.5), cause ``b`` Weibull(20, 1.5), censoring
  Uniform(0, 60); ``n`` = 300; cause ``a`` fitted with
  ``Weibull.fit(..., lfp=True)`` and the two assembled with
  ``from_fitted`` (so the covariance carries ``lfp_p``).

Times 4, 8 and 15. Replicate ``r`` uses ``default_rng(20261105 + r)``
at ``n`` = 100 and ``default_rng(20261105 + r + 10**6)`` at ``n`` =
300. A fit or bound that fails is counted and left out of the coverage,
with the count reported. Coverage is reported with its Monte Carlo
standard error ``sqrt(p (1 - p) / R)``.

Two further checks, on the first 20 replicates of ``weibull2`` at
``n`` = 300:

- bootstrap vs delta: 200 case-resampling bootstrap refits (seed
  ``20261105 + 100000 + r``); the bootstrap SD of ``logit cif_a(8)`` and
  of ``log Hf(8)`` over the delta-method standard error on the same scale
  (read from the 95% bound: ``(g(U) - g(L)) / (2 z)``);
- ``with_params`` draws vs delta: 2000 parameter vectors from the normal
  with the fit's ``covariance()`` (seed ``20261105 + 200000 + r``), each
  turned into a model by ``with_params``; the SD of the same two
  quantities over the draws, over the delta SE. Draws outside the
  parameter space (a negative scale or shape, ``lfp_p`` above 1) are
  counted and dropped.

    python scripts/studies/parametric_competing_risks_cb_coverage.py \
        [reps] [processes]

Defaults: 1000 replicates, 4 processes.

Results (5 Nov 2026, surpyval dev, 1000 replicates, 4 processes, about
15 minutes): no fit or bound failed and no bound was ``nan``. Coverage of
the 95% bounds, MC standard error in brackets (``sf`` and ``Hf`` are one
bound, so they agree):

==============  =======  =============  =============  =============
scenario        bound    t = 4          t = 8          t = 15
==============  =======  =============  =============  =============
weibull2 n=100  cif_a    0.944 (0.007)  0.952 (0.007)  0.960 (0.006)
weibull2 n=100  cif_b    0.949 (0.007)  0.942 (0.007)  0.951 (0.007)
weibull2 n=100  sf, Hf   0.957 (0.006)  0.952 (0.007)  0.959 (0.006)
weibull2 n=300  cif_a    0.958 (0.006)  0.950 (0.007)  0.954 (0.007)
weibull2 n=300  cif_b    0.951 (0.007)  0.951 (0.007)  0.946 (0.007)
weibull2 n=300  sf, Hf   0.962 (0.006)  0.936 (0.008)  0.936 (0.008)
lfp n=300       cif_a    0.938 (0.008)  0.958 (0.006)  0.960 (0.006)
lfp n=300       cif_b    0.957 (0.006)  0.959 (0.006)  0.952 (0.007)
lfp n=300       sf, Hf   0.945 (0.007)  0.944 (0.007)  0.947 (0.007)
==============  =======  =============  =============  =============

Every value is within two MC standard errors of 0.95. The lowest are the
all-cause bound at n = 300, t = 8 and 15 (0.936, 1.8 SE below) and
``cif_a`` of the lfp scenario at t = 4 (0.938, 1.5 SE below); the
all-cause bound at n = 100 errs the other way (0.952 to 0.959).

Agreement (weibull2, n = 300, t = 8, 20 data sets; every bootstrap refit
and every draw kept): the bootstrap SD over the delta SE has median 1.017
(range 0.953 to 1.111) for ``logit cif_a`` and 1.015 (0.936 to 1.158)
for ``log Hf``; the ``with_params`` draws' SD over the delta SE has
median 1.020 (0.987 to 1.042) and 1.030 (1.005 to 1.070).
"""

import sys
import warnings
from multiprocessing import Pool
from typing import Any

import numpy as np
from scipy.special import logit
from scipy.stats import norm

import surpyval as sp
from surpyval.univariate.competing_risks import ParametricCompetingRisks

SEED = 20261105
TIMES = np.array([4.0, 8.0, 15.0])
Z95 = norm.ppf(0.975)

SCENARIOS: dict[str, dict[str, Any]] = {
    "weibull2": {
        "a": (10.0, 2.0, 1.0),
        "b": (14.0, 1.0, 1.0),
        "censor": 30.0,
        "lfp": False,
    },
    "lfp": {
        "a": (10.0, 2.0, 0.5),
        "b": (20.0, 1.5, 1.0),
        "censor": 60.0,
        "lfp": True,
    },
}


def truth(name: str) -> ParametricCompetingRisks:
    spec = SCENARIOS[name]
    models = {}
    for k in ("a", "b"):
        alpha, beta, p = spec[k]
        extra = {"lfp_p": p} if p < 1 else {}
        models[k] = sp.Weibull.from_params([alpha, beta], **extra)
    return ParametricCompetingRisks.from_fitted(models)


def simulate(name: str, n: int, rng: np.random.Generator) -> tuple:
    spec = SCENARIOS[name]
    latent = []
    for k in ("a", "b"):
        alpha, beta, p = spec[k]
        t = alpha * rng.weibull(beta, size=n)
        if p < 1:
            t = np.where(rng.uniform(size=n) < p, t, np.inf)
        latent.append(t)
    latent_arr = np.column_stack(latent)
    first = latent_arr.min(axis=1)
    cause = np.where(latent_arr[:, 0] <= latent_arr[:, 1], "a", "b")
    cens = rng.uniform(0, spec["censor"], size=n)
    x = np.minimum(first, cens)
    e = cause.astype(object)
    e[first > cens] = None
    return x, e


def fit(name: str, x: Any, e: Any) -> ParametricCompetingRisks:
    if not SCENARIOS[name]["lfp"]:
        return ParametricCompetingRisks.fit(x, e, dist=sp.Weibull)
    models = {}
    for k in ("a", "b"):
        c_k = np.where(e == k, 0, 1)
        models[k] = sp.Weibull.fit(x, c=c_k, lfp=(k == "a"))
    return ParametricCompetingRisks.from_fitted(models)


def quantities(model: ParametricCompetingRisks) -> dict[str, Any]:
    return {
        "cif_a": model.cif(TIMES, "a"),
        "cif_b": model.cif(TIMES, "b"),
        "sf": model.sf(TIMES),
        "Hf": model.Hf(TIMES),
    }


def bounds(model: ParametricCompetingRisks) -> dict[str, Any]:
    return {
        "cif_a": model.cb(TIMES, "a", on="cif"),
        "cif_b": model.cb(TIMES, "b", on="cif"),
        "sf": model.cb(TIMES, on="sf"),
        "Hf": model.cb(TIMES, on="Hf"),
    }


def replicate(args: tuple) -> Any:
    name, n, r = args
    rng = np.random.default_rng(SEED + r + 10**6 * (n != 100))
    x, e = simulate(name, n, rng)
    true = quantities(truth(name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = fit(name, x, e)
            cb = bounds(model)
        except Exception as err:  # noqa: BLE001 -- counted, reported
            return ("failed", type(err).__name__)
    out = {}
    for key, value in true.items():
        lo, hi = cb[key][:, 0], cb[key][:, 1]
        ok = np.isfinite(lo) & np.isfinite(hi)
        out[key] = (ok, ok & (lo <= value) & (value <= hi))
    return out


def coverage(name: str, n: int, reps: int, pool: Any) -> None:
    results = pool.map(replicate, [(name, n, r) for r in range(reps)])
    failed = [r for r in results if isinstance(r, tuple)]
    good = [r for r in results if not isinstance(r, tuple)]
    print(f"\n{name}, n = {n}: {len(failed)} of {reps} fits failed", end="")
    if failed:
        kinds = sorted({f[1] for f in failed})
        print(f" ({', '.join(kinds)})", end="")
    print()
    print(f"{'quantity':<8}" + "".join(f"{f't = {t:g}':>22}" for t in TIMES))
    for key in ("cif_a", "cif_b", "sf", "Hf"):
        ok = np.array([g[key][0] for g in good])
        hit = np.array([g[key][1] for g in good])
        cells = []
        for j in range(TIMES.size):
            m = ok[:, j].sum()
            p = hit[:, j].sum() / m if m else np.nan
            se = np.sqrt(p * (1 - p) / m) if m else np.nan
            nan = len(good) - m
            extra = f" [{nan} nan]" if nan else ""
            cells.append(f"{p:.3f} ({se:.3f}){extra}")
        print(f"{key:<8}" + "".join(f"{c:>22}" for c in cells))


def _transforms(model: Any, t: float) -> tuple[float, float]:
    return (
        float(logit(model.cif(t, "a"))),
        float(np.log(model.Hf(t))),
    )


def agreement(r: int) -> Any:
    name, n, t = "weibull2", 300, 8.0
    rng = np.random.default_rng(SEED + r + 10**6)
    x, e = simulate(name, n, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit(name, x, e)
        cb_cif = model.cb(t, "a", on="cif")
        cb_H = model.cb(t, on="Hf")
        se_delta = np.array(
            [
                (logit(cb_cif[1]) - logit(cb_cif[0])) / (2 * Z95),
                (np.log(cb_H[1]) - np.log(cb_H[0])) / (2 * Z95),
            ]
        )
        # Case-resampling bootstrap
        boot_rng = np.random.default_rng(SEED + 100000 + r)
        boot = []
        for _ in range(200):
            idx = boot_rng.integers(0, n, size=n)
            try:
                refit = fit(name, x[idx], e[idx])
                boot.append(_transforms(refit, t))
            except Exception:  # noqa: BLE001 -- counted by its absence
                continue
        se_boot = np.std(np.array(boot), axis=0, ddof=1)
        # with_params draws from the covariance
        draw_rng = np.random.default_rng(SEED + 200000 + r)
        draws = draw_rng.multivariate_normal(
            model._estimates(), model.covariance(), size=2000
        )
        spread, dropped = [], 0
        for d in draws:
            try:
                spread.append(_transforms(model.with_params(d), t))
            except ValueError:
                dropped += 1
        se_draws = np.std(np.array(spread), axis=0, ddof=1)
    return se_delta, se_boot, len(boot), se_draws, dropped


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(procs) as pool:
        coverage("weibull2", 100, reps, pool)
        coverage("weibull2", 300, reps, pool)
        coverage("lfp", 300, reps, pool)
        rows = pool.map(agreement, range(20))
    se_delta = np.array([r[0] for r in rows])
    se_boot = np.array([r[1] for r in rows])
    se_draws = np.array([r[3] for r in rows])
    print("\nweibull2, n = 300, t = 8, 20 data sets:")
    print(f"bootstrap refits kept: {min(r[2] for r in rows)} to 200")
    print(f"with_params draws dropped: {sum(r[4] for r in rows)} of 40000")
    for j, label in enumerate(("logit cif_a", "log Hf")):
        rb = se_boot[:, j] / se_delta[:, j]
        rd = se_draws[:, j] / se_delta[:, j]
        print(
            f"{label:<12} bootstrap SD / delta SE: median {np.median(rb):.3f}"
            f" (range {rb.min():.3f} to {rb.max():.3f}); draws SD / delta"
            f" SE: median {np.median(rd):.3f} (range {rd.min():.3f} to"
            f" {rd.max():.3f})"
        )


if __name__ == "__main__":
    main()
