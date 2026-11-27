"""Generate the frozen PyMC reference for the ARA (imperfect repair)
Bayesian fit.

Run once, in an environment with PyMC (SurPyval does not depend on it),
from surpyval/tests/bayes/references (it writes there):

    python ../../../../scripts/reference/make_renewal_ara.py

NUTS, 4 chains of 10,000 draws after 2,000 tuning steps, target
acceptance 0.95, the seed below. The likelihood is written here from the
model's definition, not taken from SurPyval, so the reference is
independent. ARA with memory 1 (Kijima I) repairs a fraction ``rho`` of
the age, so the gap after a system's failure at ``T`` starts at the
virtual age ``v = (1 - rho) T`` (0 for its first gap). With a Weibull
life, each gap ``g`` contributes ``log f(v + g) - log S(v)`` when it ends
in a failure and ``log S(v + g) - log S(v)`` when it ends the
observation. The priors: ``rho`` uniform on (0, 1), ``alpha`` and
``beta`` log-normal.

The data are three systems observed to 30, repaired as good as new
(``rho = 1``): the maximum-likelihood ``rho`` is on its end, 1, and the
posterior piles against it.
"""

import json

import arviz as az  # type: ignore[import-not-found]
import numpy as np
import pymc as pm  # type: ignore[import-not-found]

#: (time, system, censored), as ``ARA.fit_bayes(x, i, c=c, m=1)`` takes
#: them: ``test_renewal_bayes.AT_END``.
X = [
    2.421352952858048,
    18.2811529910686,
    22.97976834043246,
    30.0,
    8.189901651319435,
    15.251349417317478,
    28.463734308798713,
    30.0,
    1.5505813512199516,
    11.43426925018319,
    15.14080680045863,
    23.74264955050325,
    30.0,
]
I = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
C = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
PRIORS = {"alpha": (np.log(10.0), 0.5), "beta": (np.log(2.0), 0.3)}
SEED = 20261127


def layout() -> tuple:
    x, i, c = (np.asarray(v) for v in (X, I, C))
    previous = np.zeros(x.size)
    for k in range(1, x.size):
        if i[k] == i[k - 1]:
            previous[k] = x[k - 1]
    return previous, x - previous, c


def main() -> None:
    previous, gap, c = layout()
    failed = c == 0
    with pm.Model():
        rho = pm.Uniform("rho", 0.0, 1.0)
        alpha = pm.LogNormal("alpha", *PRIORS["alpha"])
        beta = pm.LogNormal("beta", *PRIORS["beta"])
        v = (1.0 - rho) * previous
        end = v + gap
        H_end = (end / alpha) ** beta
        # (a first gap starts at age 0, where H is 0: left out, as the
        # gradient of (0 / alpha) ** beta in beta is 0 * inf)
        later = np.flatnonzero(previous > 0)
        H_start = (v[later] / alpha) ** beta
        log_h_end = pm.math.log(beta / alpha) + (beta - 1) * pm.math.log(
            end / alpha
        )
        ll = pm.math.sum(
            np.where(failed, 1.0, 0.0) * log_h_end - H_end
        ) + pm.math.sum(H_start)
        pm.Potential("ll", ll)
        idata = pm.sample(
            draws=10_000,
            tune=2_000,
            chains=4,
            cores=2,
            target_accept=0.95,
            random_seed=SEED,
            progressbar=False,
        )
    summary = az.summary(idata, kind="diagnostics")
    out: dict = {"data": {"x": X, "i": I, "c": C}, "parameters": {}}
    probs = [0.025, 0.5, 0.975]
    for name in ("rho", "alpha", "beta"):
        draws = idata.posterior[name].values
        q = np.quantile(draws.reshape(-1), probs)
        mcse = [
            float(az.mcse(draws, method="quantile", prob=p)) for p in probs
        ]
        out["parameters"][name] = {"q": q.tolist(), "mcse": mcse}
    rho = idata.posterior["rho"].values.reshape(-1)
    out["perfect_within_0.05"] = float(np.mean(rho >= 0.95))
    out["sampler"] = {
        "pymc": pm.__version__,
        "draws": 40_000,
        "seed": SEED,
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }
    with open("renewal_ara_pymc.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["sampler"]), out["perfect_within_0.05"])


if __name__ == "__main__":
    main()
