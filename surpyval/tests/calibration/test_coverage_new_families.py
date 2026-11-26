"""Coverage of the confidence bounds of the Gompertz, the InverseGaussian
and the GeneralizedF, as ``test_coverage_generalized_gamma.py`` checks
the GeneralizedGamma's.

Data: n = 200 from each family at the parameters below, right censored
by an independent uniform censoring time on (0, ``c_max``). At each
replicate the Wald bounds on the parameters, on ``sf`` at the true 10%,
50% and 90% quantiles (two-sided and one-sided lower), on the B10 life
and on the mean, and the likelihood-ratio bounds on the parameters, are
checked against the truth.

Two-sided and one-sided bounds must cover 95% of the time, within the
usual slack of 0.01 plus three Monte Carlo standard errors
(``_montecarlo``): 1000 Wald and 300 likelihood-ratio replicates each.

- Gompertz, shape 0.1 and rate 0.01 (a hazard that grows tenfold every
  23 time units; median 20.7), censored on (0, 40);
- InverseGaussian, mean 10 and shape 20 (a hazard that rises and then
  levels off), censored on (0, 30);
- GeneralizedF, mu = 2, sigma = 0.5, Q = 0.5 and P = 1 (well inside the
  family, away from the generalised gamma edge P = 0), censored on
  (0, 25).
"""

import os
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

N = 200
STUDIES = {
    "Gompertz": ([0.1, 0.01], 40.0, 211),
    "InverseGaussian": ([10.0, 20.0], 30.0, 223),
    "GeneralizedF": ([2.0, 0.5, 0.5, 1.0], 25.0, 227),
}


def _replicate(seed, name, method):
    dist = getattr(sp, name)
    params, c_max, _ = STUDIES[name]
    rng = np.random.default_rng(seed)
    true = dist.from_params(params)
    t_eval = true.qf(np.array([0.1, 0.5, 0.9]))
    t = true.qf(rng.uniform(size=N))
    cens = rng.uniform(0, c_max, N)
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
            out["sf"] = model.cb(t_eval, on="sf")
            out["sf_lower"] = model.cb(t_eval, on="sf", bound="lower")
            out["b10"] = np.asarray(model.quantile_cb(0.1))
            out["mean"] = np.asarray(model.mean_cb())
    return out


def _study(name, method, reps):
    params, _, seed = STUDIES[name]
    seeds = np.random.SeedSequence(seed + (0 if method == "wald" else 1000))
    children = [int(s.generate_state(1)[0]) for s in seeds.spawn(reps)]
    workers = min(os.cpu_count() or 1, 4)
    with process_pool(max_workers=workers) as pool:
        rows = list(
            pool.map(_replicate, children, [name] * reps, [method] * reps)
        )
    true = getattr(sp, name).from_params(params)
    t_eval = true.qf(np.array([0.1, 0.5, 0.9]))
    print(
        f"{name} ({method}): "
        f"{np.mean([r['censored'] for r in rows]):.0%} censored"
    )
    p = np.array([r["params"] for r in rows])
    label = f"{name} {method}"
    check_coverage(p[:, :, 0], p[:, :, 1], np.array(params), 0.95, label)
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


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_wald_coverage(name):
    _study(name, "wald", reps=1000)


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_likelihood_ratio_coverage(name):
    _study(name, "lr", reps=300)
