"""Time common SurPyval fits by data size, for docs/Speed.rst.

Not run in CI. Run from the repository root::

    python scripts/bench_speed.py                  # everything (about an hour)
    python scripts/bench_speed.py weibull km cox   # some benchmarks
    python scripts/bench_speed.py --max-n 10000    # small sizes only
    python scripts/bench_speed.py --json out.json  # also save the numbers

Each benchmark simulates its data with a fixed seed, fits it and reports
the wall-clock time of the fit alone (simulation excluded): the best of
three repeats for fits under a second, one run otherwise (and one run,
the first, for a bound, since a model keeps what a bound worked out). A
benchmark stops growing once a fit takes longer than ``--budget`` seconds
(60 by default) and prints the size it stopped at, so the slow models say
how slow they are without running for hours.

The machine, Python, NumPy, SciPy and SurPyval versions are printed first.
Times depend on the machine and on what else it is running; read them as
orders of magnitude and as how each model scales with n.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

import surpyval as sp

SIZES = [1_000, 10_000, 100_000, 1_000_000]


def _machine() -> str:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        chip = ""
    return chip or platform.processor() or platform.machine()


def _versions() -> dict[str, str]:
    out = {
        "machine": _machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    for name in ("surpyval", "numpy", "scipy", "pandas", "autograd"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _time(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    if first > 1.0 or getattr(fn, "once", False):
        return first
    best = first
    for _ in range(2):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _weibull_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Weibull(100, 2) lives with about 30% right censored."""
    rng = np.random.default_rng(seed)
    life = 100 * rng.weibull(2.0, n)
    cens = 100 * rng.weibull(2.0, n) * 1.6
    x = np.minimum(life, cens)
    c = (cens < life).astype(int)
    return x, c


def _covariate_data(
    n: int, seed: int = 0, ties: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weibull PH lives with three covariates, about 30% censored."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.normal(size=n), rng.binomial(1, 0.5, n), rng.uniform(size=n)]
    )
    eta = Z @ np.array([0.5, -0.4, 0.3])
    life = 100 * rng.weibull(2.0, n) * np.exp(-eta / 2.0)
    cens = 100 * rng.weibull(2.0, n) * 1.6
    x = np.minimum(life, cens)
    if ties:
        x = np.ceil(x)
    c = (cens < life).astype(int)
    return x, c, Z


# --- the benchmarks: each takes n and returns a zero-argument fit -------


def bench_weibull(n: int) -> Callable[[], Any]:
    x, c = _weibull_data(n)
    return lambda: sp.Weibull.fit(x, c)


def bench_weibull_grouped(n: int) -> Callable[[], Any]:
    # the same data recorded to the nearest whole time unit, then grouped:
    # a few hundred distinct (x, c) rows with counts n, whatever n is
    x, c = _weibull_data(n)
    x = np.ceil(x)
    rows, counts = np.unique(
        np.column_stack([x, c]), axis=0, return_counts=True
    )
    return lambda: sp.Weibull.fit(rows[:, 0], rows[:, 1].astype(int), counts)


def bench_weibull_from_df(n: int) -> Callable[[], Any]:
    import pandas as pd

    x, c = _weibull_data(n)
    df = pd.DataFrame({"t": x, "c": c})
    return lambda: sp.Weibull.fit_from_df(df, x_col="t", c_col="c")


def bench_km(n: int) -> Callable[[], Any]:
    x, c = _weibull_data(n)
    return lambda: sp.KaplanMeier.fit(x, c)


def bench_cox(n: int) -> Callable[[], Any]:
    # whole-number times, so there are ties (Efron, the default)
    x, c, Z = _covariate_data(n, ties=True)
    return lambda: sp.CoxPH.fit(x, Z, c)


def bench_cox_no_ties(n: int) -> Callable[[], Any]:
    x, c, Z = _covariate_data(n, ties=False)
    return lambda: sp.CoxPH.fit(x, Z, c)


def bench_weibull_aft(n: int) -> Callable[[], Any]:
    x, c, Z = _covariate_data(n)
    return lambda: sp.WeibullAFT.fit(x, Z, c)


def _inspections(
    n: int, own_schedule: bool, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    life = 100 * rng.weibull(2.0, n)
    if own_schedule:
        # each unit inspected on its own schedule: many distinct endpoints
        offset = rng.uniform(0, 10, n)
    else:
        # every unit inspected at 10, 20, ..., 300
        offset = np.zeros(n)
    left = np.floor((life - offset) / 10) * 10 + offset
    left = np.maximum(left, 0.0)
    right = left + 10
    return left, right


def bench_turnbull_grid(n: int) -> Callable[[], Any]:
    xl, xr = _inspections(n, own_schedule=False)
    return lambda: sp.Turnbull.fit(xl=xl, xr=xr)


def bench_turnbull_own(n: int) -> Callable[[], Any]:
    xl, xr = _inspections(n, own_schedule=True)
    return lambda: sp.Turnbull.fit(xl=xl, xr=xr)


def bench_crow_amsaa(n: int) -> Callable[[], Any]:
    from surpyval.recurrent import CrowAMSAA

    # n failures over n / 10 systems, each watched to T = 1000
    rng = np.random.default_rng(0)
    systems = max(n // 10, 1)
    times: list[float] = []
    ids: list[int] = []
    cens: list[int] = []
    for k in range(systems):
        t = np.sort(1000 * rng.uniform(size=10) ** (1 / 1.5))
        times.extend(t)
        ids.extend([k] * 10)
        cens.extend([0] * 10)
        times.append(1000.0)
        ids.append(k)
        cens.append(1)
    x, i, c = np.array(times), np.array(ids), np.array(cens)
    return lambda: CrowAMSAA.fit(x=x, i=i, c=c)


def bench_rsf(n: int) -> Callable[[], Any]:
    from surpyval.ml import RandomSurvivalForest

    x, c, Z = _covariate_data(n)
    return lambda: RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=100, random_state=0
    )


def bench_rsf_np(n: int) -> Callable[[], Any]:
    from surpyval.ml import RandomSurvivalForest

    x, c, Z = _covariate_data(n)
    return lambda: RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=100, kind="non-parametric", random_state=0
    )


def bench_rsf_jobs(n: int) -> Callable[[], Any]:
    from surpyval.ml import RandomSurvivalForest

    x, c, Z = _covariate_data(n)
    return lambda: RandomSurvivalForest.fit(
        x=x, Z=Z, c=c, n_trees=100, random_state=0, n_jobs=4
    )


def bench_joint(n: int) -> Callable[[], Any]:
    from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH

    rng = np.random.default_rng(42)
    t = np.linspace(0, 3000, 3001)
    items = []
    for k in range(n):
        start = rng.normal(3.0, 0.6)
        rate = rng.normal(1.0, 0.5) / 365
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (start + rate * t - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(rng.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 73.0)  # about ten readings over two years
        readings = start + rate * ages + rng.normal(0, 0.7, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        items.append(
            ItemLifetime(
                f"P{k}",
                **{outcome: round(end, 1)},
                markers=dict(zip(ages, readings.round(2))),
            )
        )
    fleet = SurpyvalData.from_items(items)
    return lambda: WeibullJointPH.fit(fleet)


def _fitted_weibull(n: int) -> Any:
    x, c = _weibull_data(n)
    return sp.Weibull.fit(x, c)


def _once(fn: Callable[[], Any]) -> Callable[[], Any]:
    # a model keeps what a bound has worked out, so a second call on the
    # same model is not a fair time: time the first call only
    fn.once = True  # type: ignore[attr-defined]
    return fn


def bench_cb_wald(n: int) -> Callable[[], Any]:
    model = _fitted_weibull(n)
    return _once(lambda: model.cb(50.0))


def bench_cb_lr(n: int) -> Callable[[], Any]:
    model = _fitted_weibull(n)
    return _once(lambda: model.cb(50.0, method="lr"))


def bench_cb_bootstrap(n: int) -> Callable[[], Any]:
    model = _fitted_weibull(n)
    return _once(
        lambda: model.cb(50.0, method="bootstrap", n_boot=200, random_state=0)
    )


def _fitted_weibull_grouped(n: int) -> Any:
    x, c = _weibull_data(n)
    x = np.ceil(x)
    rows, counts = np.unique(
        np.column_stack([x, c]), axis=0, return_counts=True
    )
    return sp.Weibull.fit(rows[:, 0], rows[:, 1].astype(int), counts)


def bench_cb_lr_grouped(n: int) -> Callable[[], Any]:
    model = _fitted_weibull_grouped(n)
    return _once(lambda: model.cb(50.0, method="lr"))


def bench_cb_bootstrap_grouped(n: int) -> Callable[[], Any]:
    model = _fitted_weibull_grouped(n)
    return _once(
        lambda: model.cb(50.0, method="bootstrap", n_boot=200, random_state=0)
    )


def bench_cb_bootstrap_50(n: int) -> Callable[[], Any]:
    model = _fitted_weibull(n)
    return _once(
        lambda: model.cb(50.0, method="bootstrap", n_boot=50, random_state=0)
    )


BENCHMARKS: dict[
    str, tuple[str, Callable[[int], Callable[[], Any]], list[int]]
] = {
    "weibull": ("Weibull MLE, 30% right censored", bench_weibull, SIZES),
    "weibull_grouped": (
        "Weibull MLE, same data grouped with counts n",
        bench_weibull_grouped,
        SIZES,
    ),
    "weibull_df": (
        "Weibull MLE via fit_from_df",
        bench_weibull_from_df,
        SIZES,
    ),
    "km": ("Kaplan-Meier", bench_km, SIZES),
    "cox": ("CoxPH, 3 covariates, Efron ties", bench_cox, SIZES),
    "cox_no_ties": ("CoxPH, 3 covariates, no ties", bench_cox_no_ties, SIZES),
    "aft": ("Weibull AFT, 3 covariates", bench_weibull_aft, SIZES),
    "turnbull": (
        "Turnbull, common inspection grid",
        bench_turnbull_grid,
        SIZES,
    ),
    "turnbull_own": (
        "Turnbull, each unit on its own schedule",
        bench_turnbull_own,
        SIZES,
    ),
    "crow": (
        "Crow-AMSAA, n events over n/10 systems",
        bench_crow_amsaa,
        SIZES,
    ),
    "rsf": (
        "RandomSurvivalForest, 100 trees, Weibull leaves",
        bench_rsf,
        SIZES,
    ),
    "rsf_np": (
        "RandomSurvivalForest, 100 trees, non-parametric leaves",
        bench_rsf_np,
        SIZES,
    ),
    "rsf_jobs4": (
        "RandomSurvivalForest, 100 trees, Weibull leaves, n_jobs=4",
        bench_rsf_jobs,
        SIZES,
    ),
    "joint": (
        "WeibullJointPH, ~10 readings per item (n = items)",
        bench_joint,
        [100, 300, 1_000, 3_000, 10_000],
    ),
    "cb_wald": ("Weibull cb, Wald", bench_cb_wald, SIZES),
    "cb_lr": ("Weibull cb, likelihood ratio", bench_cb_lr, SIZES),
    "cb_boot": (
        "Weibull cb, bootstrap, 200 samples",
        bench_cb_bootstrap,
        SIZES,
    ),
    "cb_lr_grouped": (
        "Weibull cb, likelihood ratio, grouped data",
        bench_cb_lr_grouped,
        SIZES,
    ),
    "cb_boot_grouped": (
        "Weibull cb, bootstrap, 200 samples, grouped data",
        bench_cb_bootstrap_grouped,
        SIZES,
    ),
    "cb_boot50": (
        "Weibull cb, bootstrap, 50 samples",
        bench_cb_bootstrap_50,
        SIZES,
    ),
}


def _fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="benchmarks to run")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--json", default=None, help="write results here")
    args = parser.parse_args(argv)

    names = args.names or list(BENCHMARKS)
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        parser.error(
            f"unknown benchmark(s) {unknown}; choose from {list(BENCHMARKS)}"
        )

    info = _versions()
    for key, value in info.items():
        print(f"{key:>9}: {value}")
    print()

    results: dict[str, Any] = {"versions": info, "results": {}}
    for name in names:
        label, factory, sizes = BENCHMARKS[name]
        print(f"{name}: {label}", flush=True)
        rows = []
        for n in sizes:
            if args.max_n is not None and n > args.max_n:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = factory(n)
                    seconds = _time(fit)
                except Exception as exc:  # report, keep going
                    print(f"  n={n:>9,}  failed: {type(exc).__name__}: {exc}")
                    rows.append(
                        {"n": n, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    break
            print(f"  n={n:>9,}  {_fmt(seconds)}", flush=True)
            rows.append({"n": n, "seconds": seconds})
            if seconds > args.budget:
                print(f"  stopped: over the {args.budget:.0f} s budget")
                break
        results["results"][name] = {"label": label, "rows": rows}

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
