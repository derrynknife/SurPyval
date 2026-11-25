"""Generate the frozen PyMC references for the Crow-AMSAA Bayesian fit.

Run once, in an environment with PyMC (SurPyval does not depend on it),
from surpyval/tests/bayes/references (it writes there):

    python ../../../../scripts/reference/make_crow_amsaa.py

NUTS, 4 chains of 10,000 draws after 2,000 tuning steps, target
acceptance 0.95, seeds below. The likelihood is written here from the
power-law process's definition, not taken from SurPyval, so the
reference is independent: for systems observed from 0 to ``T_q`` with
failures at ``t``,

    log L = N log(beta) - N beta log(alpha) + (beta - 1) sum(log t)
            - sum_q (T_q / alpha)^beta.

The priors are normal on log(alpha) and log(beta).
"""

import json
import sys

import arviz as az  # type: ignore[import-not-found]
import numpy as np
import pymc as pm  # type: ignore[import-not-found]

CASES = {
    # one prototype, time-terminated at 400
    "crow_amsaa_one_system": dict(
        times=[15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390],
        ends=[400.0],
        log_alpha=(np.log(40.0), 1.0),
        log_beta=(np.log(0.7), 0.35),
        seed=20261125,
    ),
    # three systems with unequal ends, a looser prior
    "crow_amsaa_three_systems": dict(
        times=[30, 110, 240, 420, 55, 190, 380, 610, 820, 95, 300],
        ends=[500.0, 900.0, 350.0],
        log_alpha=(np.log(100.0), 2.0),
        log_beta=(0.0, 0.5),
        seed=20261126,
    ),
}


def run(name, case):
    t = np.asarray(case["times"], dtype=float)
    ends = np.asarray(case["ends"], dtype=float)
    with pm.Model():
        la = pm.Normal("log_alpha", *case["log_alpha"])
        lb = pm.Normal("log_beta", *case["log_beta"])
        beta = pm.math.exp(lb)
        ll = (
            len(t) * lb
            - len(t) * beta * la
            + (beta - 1) * np.log(t).sum()
            - pm.math.sum(pm.math.exp(beta * (np.log(ends) - la)))
        )
        pm.Potential("ll", ll)
        idata = pm.sample(
            draws=10_000,
            tune=2_000,
            chains=4,
            cores=2,
            target_accept=0.95,
            random_seed=case["seed"],
            progressbar=False,
        )
    summary = az.summary(idata, kind="diagnostics")
    out = {"parameters": {}}
    probs = [0.025, 0.5, 0.975]
    for var, label in (("log_alpha", "alpha"), ("log_beta", "beta")):
        draws = np.exp(idata.posterior[var].values)
        q = np.quantile(draws.reshape(-1), probs)
        mcse = [
            float(az.mcse(draws, method="quantile", prob=p)) for p in probs
        ]
        out["parameters"][label] = {"q": q.tolist(), "mcse": mcse}
    out["sampler"] = {
        "pymc": pm.__version__,
        "draws": 40_000,
        "seed": case["seed"],
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }
    with open(f"{name}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(name, json.dumps(out["sampler"]))


if __name__ == "__main__":
    for name, case in CASES.items():
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        run(name, case)
