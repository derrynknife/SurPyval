"""Would a Wald standard error for an offset ``gamma`` be trustworthy?

Why an offset fit's ``covariance()`` holds ``gamma`` at its estimate
rather than giving it a row. Each replicate draws ``n`` failure times
``50 + 100 W`` with ``W`` a Weibull of shape ``beta``, fits
``Weibull.fit(x, offset=True)``, and takes the inverse of numdifftools'
Hessian of the negative log-likelihood in ``(gamma, alpha, beta)`` at the
fit. The study reports, per shape:

- the sampling SD of ``gamma_hat`` over the replicates (with its Monte
  Carlo standard error, ``SD / sqrt(2 (R - 1))``);
- the mean and median of the Hessian standard error of ``gamma``;
- how often that Hessian was not usable (non-finite, or not positive
  definite);
- the coverage of the 95% Wald interval ``gamma_hat +- 1.96 SE`` for the
  true 50 (Monte Carlo SE ``sqrt(0.95 * 0.05 / R)``);
- the median gap between the first failure and ``gamma_hat``.

Regular asymptotics hold for the shape above 2 (Smith, 1985); between 1
and 2 the estimate exists but is not asymptotically normal at the usual
rate; below 1 the likelihood is unbounded at the first failure.

    python scripts/studies/offset_gamma_wald.py [reps] [n] [shapes]

Defaults: 1000 replicates of n = 200 at shapes 3.5, 2.5, 1.5 and 0.8,
about a minute on one core. The seed (``default_rng(20261105)`` per
shape) was fixed before the results were seen.

Results (5 Nov 2026, surpyval dev, 1000 replicates, n = 200):

====  ==============  ==========  =========  ========  =============
beta  SD(gamma_hat)   mean SE     median SE  unusable  Wald coverage
====  ==============  ==========  =========  ========  =============
3.5   12.56 (0.28)    12.23       10.76      0/1000    0.889 (0.007)
2.5   6.66 (0.15)     6.51        6.00       0/1000    0.873 (0.007)
1.5   1.93 (0.04)     1.60        1.47       0/1000    0.762 (0.007)
0.8   0.174 (0.004)   --          --         1000/1000 0 (0.007)
====  ==============  ==========  =========  ========  =============

Even where the theory is regular (shape 3.5) the 95% Wald interval covers
the true offset only 89% of the time at n = 200, below 1.5 it covers 76%,
and below 1 there is no Hessian at all (the fit sits on the first
failure). So ``covariance()`` gives ``gamma`` no row, and
``param_cb("gamma")`` offers the likelihood-ratio and bootstrap bounds
instead of a Wald one.
"""

import sys
import warnings
from typing import Any

import numpy as np
from numdifftools import Hessian  # type: ignore

import surpyval as sp

GAMMA = 50.0


def gamma_se(model: Any) -> float:
    """The Hessian standard error of gamma, or nan where unusable."""

    def nll(theta: Any) -> Any:
        return model.dist._neg_ll_func(
            model.surv_data, theta[1], theta[2], theta[0], 0.0, 1.0
        )

    theta = np.r_[model.gamma, model.params]
    step = 1e-4 * np.maximum(np.abs(theta), 1)
    hess = Hessian(nll, step=step)(theta)
    if not np.all(np.isfinite(hess)):
        return np.nan
    if np.min(np.linalg.eigvalsh(hess)) <= 0:
        return np.nan
    return float(np.sqrt(np.linalg.inv(hess)[0, 0]))


def study(beta: float, reps: int, n: int) -> None:
    rng = np.random.default_rng(20261105)
    est, ses, gaps, covered = [], [], [], 0
    for _ in range(reps):
        x = GAMMA + 100 * rng.weibull(beta, n)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            model = sp.Weibull.fit(x, offset=True)
            se = gamma_se(model)
        est.append(model.gamma)
        gaps.append(x.min() - model.gamma)
        if np.isfinite(se):
            ses.append(se)
            covered += abs(model.gamma - GAMMA) < 1.959964 * se
    est_arr = np.array(est)
    sd = est_arr.std(ddof=1)
    se_arr = np.array(ses)
    mean_se = se_arr.mean() if se_arr.size else np.nan
    median_se = np.median(se_arr) if se_arr.size else np.nan
    print(
        f"beta={beta} n={n} R={reps}: SD(gamma_hat)={sd:.3f} "
        f"(MC se {sd / np.sqrt(2 * (reps - 1)):.3f}); "
        f"Hessian SE mean={mean_se:.3f} median={median_se:.3f}; "
        f"unusable {reps - se_arr.size}/{reps}; "
        f"Wald 95% coverage={covered / reps:.3f} "
        f"(MC se {np.sqrt(0.95 * 0.05 / reps):.3f}); "
        f"median(min x - gamma_hat)={np.median(gaps):.3g}"
    )


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    shapes = sys.argv[3] if len(sys.argv) > 3 else "3.5,2.5,1.5,0.8"
    for beta in (float(b) for b in shapes.split(",")):
        study(beta, reps, n)
