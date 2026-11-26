"""Extremely randomised survival forest against the searched forest:
out-of-bag score and fit time.

The design, fixed before the first run (27 Nov 2026):

- data: ``load_rossi_static`` (432 rows, 7 covariates; ``arrest`` is the
  event) and ``load_lung`` (complete cases of the 228 rows; ``status`` 2
  is a death, 1 censored);
- forests: ``RandomSurvivalForest.fit`` with ``kind="weibull"``,
  ``n_trees=100``, every other option at its default, one core, seeds
  0 to 9, once with ``splitrule="searched"`` and once with ``"random"``;
- score: ``oob_log_likelihood()``, the mean per observation (higher is
  better);
- noise: the standard deviation of the searched forest's score over the
  10 seeds. The random forest passes if its mean score is within two of
  these standard deviations of the searched forest's mean;
- time: the median wall-clock seconds of one ``fit`` over the 10 seeds,
  reported, with no pass or fail.

Run::

    python scripts/studies/extra_trees_oob.py
"""

import contextlib
import io
import statistics
import time
import warnings
from typing import Any

import numpy as np

import surpyval as sp
from surpyval.ml.forest import RandomSurvivalForest

SEEDS = range(10)
N_TREES = 100


def rossi() -> tuple[Any, Any, Any]:
    df = sp.datasets.load_rossi_static()
    Z = df[["fin", "age", "race", "wexp", "mar", "paro", "prio"]].to_numpy(
        float
    )
    return df["week"].to_numpy(float), Z, 1 - df["arrest"].to_numpy(int)


def lung() -> tuple[Any, Any, Any]:
    df = sp.datasets.load_lung().drop(columns=["inst"]).dropna()
    cols = ["age", "sex", "ph.ecog", "ph.karno", "pat.karno", "meal.cal"]
    cols += ["wt.loss"]
    return (
        df["time"].to_numpy(float),
        df[cols].to_numpy(float),
        (df["status"].to_numpy(int) == 1).astype(int),
    )


def run(data: Any, splitrule: str) -> tuple[Any, Any]:
    x, Z, c = data
    scores, times = [], []
    for seed in SEEDS:
        start = time.perf_counter()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            forest = RandomSurvivalForest.fit(
                x,
                Z,
                c=c,
                n_trees=N_TREES,
                kind="weibull",
                random_state=seed,
                splitrule=splitrule,
            )
            times.append(time.perf_counter() - start)
            scores.append(forest.oob_log_likelihood())
    return np.array(scores), np.array(times)


def main() -> None:
    for name, data in (("rossi", rossi()), ("lung", lung())):
        s_scores, s_times = run(data, "searched")
        r_scores, r_times = run(data, "random")
        sd = s_scores.std(ddof=1)
        gap = r_scores.mean() - s_scores.mean()
        print(
            f"{name}: n = {len(data[0])}\n"
            f"  searched: mean {s_scores.mean():.4f}, sd over seeds "
            f"{sd:.4f}, fit {statistics.median(s_times):.1f} s\n"
            f"  random:   mean {r_scores.mean():.4f}, sd over seeds "
            f"{r_scores.std(ddof=1):.4f}, fit "
            f"{statistics.median(r_times):.1f} s\n"
            f"  gap (random - searched) {gap:+.4f} = {gap / sd:+.2f} "
            f"searched sds: {'within' if abs(gap) <= 2 * sd else 'OUTSIDE'} "
            f"the band; fit time ratio "
            f"{statistics.median(r_times) / statistics.median(s_times):.2f}"
        )


if __name__ == "__main__":
    main()
