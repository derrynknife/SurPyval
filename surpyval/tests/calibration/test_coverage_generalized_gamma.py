"""Coverage of the GeneralizedGamma's confidence bounds.

Data: n = 200 from the generalised gamma with mu = 2, sigma = 0.5 and
Q = 0.7 (between the lognormal, Q = 0, and the Weibull, Q = 1), right
censored by an independent uniform censoring time on (0, 25), 28%
censored. At each replicate the Wald bounds on the three parameters, on
``sf`` at the true 10%, 50% and 90% quantiles (two-sided and one-sided
lower), on the B10 life and on the mean, and the likelihood-ratio bounds
on the parameters, are checked against the truth.

Two-sided and one-sided bounds must cover 95% of the time, within the
usual slack of 0.01 plus three Monte Carlo standard errors
(``_montecarlo``). n = 200 rather than the 100 of the two-parameter
families, for a third (shape) parameter. On 25 Nov 2026 the Wald study
gave 0.937 to 0.953 over its eleven cells (1000 replicates), and the
likelihood-ratio study 0.957 to 0.967 on the three parameters (300).
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

PARAMS = [2.0, 0.5, 0.7]
C_MAX = 25.0
N = 200
SEED = 109


def _replicate(seed, method):
    dist = sp.GeneralizedGamma
    rng = np.random.default_rng(seed)
    true = dist.from_params(PARAMS)
    t_eval = true.qf(np.array([0.1, 0.5, 0.9]))
    t = true.qf(rng.uniform(size=N))
    cens = rng.uniform(0, C_MAX, N)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    out = {"censored": c.mean()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = dist.fit(x, c=c)
        out["params"] = np.array(
            [
                model.param_cb(name, method=method)
                for name in dist.parameter_names
            ]
        )
        if method == "wald":
            band = model.cb(t_eval, on="sf")
            out["sf"] = band
            out["sf_lower"] = model.cb(t_eval, on="sf", bound="lower")
            out["b10"] = np.asarray(model.quantile_cb(0.1))
            out["mean"] = np.asarray(model.mean_cb())
    return out


def _study(method, reps):
    seeds = np.random.SeedSequence(SEED + (0 if method == "wald" else 1000))
    children = [int(s.generate_state(1)[0]) for s in seeds.spawn(reps)]
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        rows = list(pool.map(_replicate, children, [method] * reps))
    true = sp.GeneralizedGamma.from_params(PARAMS)
    t_eval = true.qf(np.array([0.1, 0.5, 0.9]))
    print(
        f"GeneralizedGamma ({method}): "
        f"{np.mean([r['censored'] for r in rows]):.0%} censored"
    )
    p = np.array([r["params"] for r in rows])
    label = f"GeneralizedGamma {method}"
    check_coverage(p[:, :, 0], p[:, :, 1], np.array(PARAMS), 0.95, label)
    if method == "wald":
        sf = np.array([r["sf"] for r in rows])
        check_coverage(
            sf[:, :, 0], sf[:, :, 1], true.sf(t_eval), 0.95, label + " sf"
        )
        lower = np.array([r["sf_lower"] for r in rows])
        check_coverage(
            lower, np.inf, true.sf(t_eval), 0.95, label + " sf lower"
        )
        b10 = np.array([r["b10"] for r in rows])
        check_coverage(b10[:, 0], b10[:, 1], t_eval[0], 0.95, label + " B10")
        mean = np.array([r["mean"] for r in rows])
        check_coverage(
            mean[:, 0], mean[:, 1], true.mean(), 0.95, label + " mean"
        )


def test_wald_coverage():
    _study("wald", reps=1000)


def test_likelihood_ratio_coverage():
    _study("lr", reps=300)
