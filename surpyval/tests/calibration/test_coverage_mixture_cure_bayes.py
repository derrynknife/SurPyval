"""Coverage of the mixture-cure regression's Bayesian fit
(``MixtureCure(Weibull).fit_bayes``), by simulation-based calibration
(the validation kit's template, ``surpyval/tests/bayes/kit.py``).

In words: the true parameters are drawn from the priors, a study is
simulated from them, and each credible interval should hold the truth
in its share of repetitions. The fit is the Laplace approximation, so it
is held to the stated band of the Laplace regressions (90% intervals
within 0.90 +- 0.05, 50% within 0.50 +- 0.09, no fit refused) in place
of the exact engines' rule, with no uniformity test. The scenario and
its seed were fixed before the first run (27 Nov 2026):

- ``weibull-logit``: 200 units, a two-level covariate (half at each
  level), followed to 40 (four times the latency's scale, so follow-up
  is long enough for a plateau); ``alpha`` log-normal about 10 (sigma
  0.2), ``beta`` log-normal about 2 (sigma 0.15), ``lfp_p`` Beta(6, 3)
  (two-thirds of units at the reference level fail eventually), the
  coefficients normal about 0 (sd 0.3 on the latency's scale, 0.7 on the
  incidence's logit). 300 repetitions, seed 72100.

The data are simulated from the model's definition (a unit fails at all
with probability ``expit(logit(lfp_p) + b_p z)``, at a Weibull life with
scale ``alpha exp(-b_a z)``), not with the package's simulator.
"""

import numpy as np
import pytest
from scipy.special import expit, logit

import surpyval as sp
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 300
UNITS, END = 200, 40.0
LAPLACE = dict(tolerance={0.5: 0.09, 0.9: 0.05}, uniform=False)


def _weibull_logit():
    priors = {
        "alpha": sp.LogNormal.from_params([np.log(10.0), 0.2]),
        "beta": sp.LogNormal.from_params([np.log(2.0), 0.15]),
        "lfp_p": sp.Beta.from_params([6.0, 3.0]),
        "alpha_coef_0": sp.Normal.from_params([0.0, 0.3]),
        "lfp_p_coef_0": sp.Normal.from_params([0.0, 0.7]),
    }
    Z = np.repeat([0.0, 1.0], UNITS // 2)[:, None]

    def truth(rng):
        return {
            "alpha": float(np.exp(np.log(10.0) + 0.2 * rng.standard_normal())),
            "beta": float(np.exp(np.log(2.0) + 0.15 * rng.standard_normal())),
            "lfp_p": float(rng.beta(6.0, 3.0)),
            "alpha_coef_0": float(0.3 * rng.standard_normal()),
            "lfp_p_coef_0": float(0.7 * rng.standard_normal()),
        }

    def simulate(t, rng):
        z = Z[:, 0]
        fails = rng.uniform(size=UNITS) < expit(
            logit(t["lfp_p"]) + t["lfp_p_coef_0"] * z
        )
        scale = t["alpha"] * np.exp(-t["alpha_coef_0"] * z)
        life = scale * (-np.log(rng.uniform(size=UNITS))) ** (1 / t["beta"])
        failed = fails & (life <= END)
        return np.where(failed, life, END), Z, np.where(failed, 0, 1)

    def fit(data):
        x, Zd, c = data
        return sp.MixtureCure(sp.Weibull).fit_bayes(x, Zd, c, priors=priors)

    return truth, simulate, fit, list(priors), 72100, dict(LAPLACE)


SCENARIOS = {"weibull-logit": _weibull_logit}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_coverage(name):
    truth, simulate, fit, names, seed, rules = SCENARIOS[name]()
    reps = rules.pop("reps", REPS)
    result = kit.sbc(truth, simulate, fit, names, reps, seed)
    print(name, kit.describe_sbc(result))
    kit.check_sbc(result, **rules)
