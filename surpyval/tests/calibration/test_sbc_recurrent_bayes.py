"""Simulation-based calibration of the recurrent-event processes'
Bayesian fits (``CrowAMSAA.fit_bayes`` and ``HPP.fit_bayes``), by the
validation kit's template (``surpyval/tests/bayes/kit.py``).

In words: the true parameters are drawn from the priors, a test is
simulated from them, and the fit's posterior should then put the truth
below its ``p`` point in a share ``p`` of repetitions. The rules are the
kit's (50% and 90% coverage within three standard errors, the truth's
posterior probability uniform by Kolmogorov-Smirnov at 0.001, no fit
refused), and each scenario and its seed were fixed before the first run
(25 Nov 2026):

- ``crow-amsaa``: two systems, each observed from 0 to 1,000;
  ``alpha`` log-normal about 80 (sigma 0.4), ``beta`` log-normal about 1
  (sigma 0.25): about a dozen failures a system. The grid (exact). 300
  repetitions, seed 61200.
- ``hpp``: one system observed from 0 to 500; ``lambda`` log-normal
  about 0.01 (sigma 0.5): about five failures, and sometimes none, which
  the fit takes on its prior. The grid (the prior is not conjugate). 300
  repetitions, seed 61500.
- ``hpp-gamma``: the same design with ``lambda`` Gamma(2, 200): the
  closed form. 300 repetitions, seed 61800.

The data are simulated from the processes' definitions, not with the
package's simulators.
"""

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.recurrent import HPP, CrowAMSAA
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 300


def _crow_amsaa():
    priors = {
        "alpha": sp.LogNormal.from_params([np.log(80.0), 0.4]),
        "beta": sp.LogNormal.from_params([0.0, 0.25]),
    }
    end, systems = 1000.0, 2

    def truth(rng):
        return {
            "alpha": float(np.exp(np.log(80.0) + 0.4 * rng.standard_normal())),
            "beta": float(np.exp(0.25 * rng.standard_normal())),
        }

    def simulate(t, rng):
        xs, ids, cs = [], [], []
        total = (end / t["alpha"]) ** t["beta"]
        for k in range(systems):
            m = rng.poisson(total)
            u = np.sort(total * rng.uniform(size=m))
            xs += [*(t["alpha"] * u ** (1 / t["beta"])), end]
            ids += [k] * (m + 1)
            cs += [0] * m + [1]
        return xs, ids, cs

    def fit(data):
        return CrowAMSAA.fit_bayes(*data, priors=priors)

    return truth, simulate, fit, ["alpha", "beta"], 61200


def _hpp(prior, draw, seed):
    end = 500.0

    def truth(rng):
        return {"lambda": float(draw(rng))}

    def simulate(t, rng):
        m = rng.poisson(t["lambda"] * end)
        return [*np.sort(end * rng.uniform(size=m)), end], [0] * m + [1]

    def fit(data):
        return HPP.fit_bayes(data[0], c=data[1], priors={"lambda": prior})

    return truth, simulate, fit, ["lambda"], seed


SCENARIOS = {
    "crow-amsaa": _crow_amsaa,
    "hpp": lambda: _hpp(
        sp.LogNormal.from_params([np.log(0.01), 0.5]),
        lambda rng: np.exp(np.log(0.01) + 0.5 * rng.standard_normal()),
        61500,
    ),
    "hpp-gamma": lambda: _hpp(
        sp.Gamma.from_params([2.0, 200.0]),
        lambda rng: stats.gamma(2.0, scale=1 / 200.0).rvs(random_state=rng),
        61800,
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_sbc(name):
    truth, simulate, fit, names, seed = SCENARIOS[name]()
    result = kit.sbc(truth, simulate, fit, names, REPS, seed)
    print(name, kit.describe_sbc(result))
    kit.check_sbc(result)
