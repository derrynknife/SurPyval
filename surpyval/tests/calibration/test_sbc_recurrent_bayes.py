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

- ``cox-lewis``: two systems observed from 0 to 100; ``alpha`` normal
  about -3 (sd 0.3), ``beta`` normal about 0.01 (sd 0.005). The grid.
  300 repetitions, seed 62700. **This scenario misses its rule, and is
  kept as it was fixed**: ``alpha``'s 90% interval held the truth in
  0.847 of repetitions, against a lower limit of 0.848 (three standard
  errors below 0.90); its 50% coverage was 0.490, ``beta``'s 0.540 and
  0.920, and both parameters' uniformity tests passed (p 0.49 and
  0.86). Investigated, with no fault found: no fit refused, no truth
  fell outside a posterior's range, and an independent run of 1,000
  repetitions of the same scenario (``cox-lewis-1000``, seed 900000,
  chosen after seeing the first result and before its own run) gave
  0.911 and 0.513 for ``alpha``, 0.908 and 0.499 for ``beta``, inside
  the rule (which at 1,000 repetitions is 0.90 +- 0.028). The first run
  is marked as an expected failure with these numbers, not re-seeded.
- ``pi-hpp``: ten systems observed from 0 to 300, a two-level covariate
  (five systems at each level); ``lambda`` log-normal about 0.02 (sigma
  0.4), the coefficient normal about 0 (sd 0.5). The grid. 300
  repetitions, seed 62400.
- ``pi-crow-amsaa``: twenty systems observed from 0 to 300 with two
  covariates (a two-level one, and one evenly spaced from -1 to 1);
  ``alpha`` log-normal about 60 (sigma 0.3), ``beta`` log-normal about 1
  (sigma 0.2), the coefficients normal about 0 (sd 0.5). Four free
  parameters, so the Laplace approximation: an approximate engine, held
  to a stated band in place of the exact engines' rule, 0.90 +- 0.05
  and 0.50 +- 0.09 (the Laplace regressions' band), with no uniformity
  test. 300 repetitions, seed 62100.

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

    return truth, simulate, fit, ["alpha", "beta"], 61200, {}


def _hpp(prior, draw, seed):
    end = 500.0

    def truth(rng):
        return {"lambda": float(draw(rng))}

    def simulate(t, rng):
        m = rng.poisson(t["lambda"] * end)
        return [*np.sort(end * rng.uniform(size=m)), end], [0] * m + [1]

    def fit(data):
        return HPP.fit_bayes(data[0], c=data[1], priors={"lambda": prior})

    return truth, simulate, fit, ["lambda"], seed, {}


def _cox_lewis():
    from surpyval.recurrent import CoxLewis

    priors = {
        "alpha": sp.Normal.from_params([-3.0, 0.3]),
        "beta": sp.Normal.from_params([0.01, 0.005]),
    }
    end, systems = 100.0, 2

    def truth(rng):
        return {
            "alpha": float(-3.0 + 0.3 * rng.standard_normal()),
            "beta": float(0.01 + 0.005 * rng.standard_normal()),
        }

    def simulate(t, rng):
        # cif(x) = exp(alpha) (exp(beta x) - 1) / beta
        a, b = t["alpha"], t["beta"]
        total = np.exp(a) * np.expm1(b * end) / b
        xs, ids, cs = [], [], []
        for k in range(systems):
            m = rng.poisson(total)
            u = np.sort(total * rng.uniform(size=m))
            xs += [*(np.log1p(u * b / np.exp(a)) / b), end]
            ids += [k] * (m + 1)
            cs += [0] * m + [1]
        return xs, ids, cs

    def fit(data):
        return CoxLewis.fit_bayes(*data, priors=priors)

    return truth, simulate, fit, ["alpha", "beta"], 62700, {}


def _cox_lewis_again():
    truth, simulate, fit, names, _, rules = _cox_lewis()
    return truth, simulate, fit, names, 900000, {**rules, "reps": 1000}


def _pi(baseline):
    from surpyval.recurrent import (
        ProportionalIntensityHPP,
        ProportionalIntensityNHPP,
    )

    end = 300.0
    if baseline == "hpp":
        design = np.repeat([0.0, 1.0], 5)[:, None]
        priors = {
            "lambda": sp.LogNormal.from_params([np.log(0.02), 0.4]),
            "coef_0": sp.Normal.from_params([0.0, 0.5]),
        }
        seed, rules = 62400, {}

        def truth(rng):
            return {
                "lambda": float(
                    np.exp(np.log(0.02) + 0.4 * rng.standard_normal())
                ),
                "coef_0": float(0.5 * rng.standard_normal()),
            }

        def fit(data):
            return ProportionalIntensityHPP.fit_bayes(*data, priors=priors)

    else:
        design = np.column_stack(
            [np.tile([0.0, 1.0], 10), np.linspace(-1.0, 1.0, 20)]
        )
        priors = {
            "alpha": sp.LogNormal.from_params([np.log(60.0), 0.3]),
            "beta": sp.LogNormal.from_params([0.0, 0.2]),
            "coef_0": sp.Normal.from_params([0.0, 0.5]),
            "coef_1": sp.Normal.from_params([0.0, 0.5]),
        }
        seed = 62100
        rules = dict(tolerance={0.5: 0.09, 0.9: 0.05}, uniform=False)

        def truth(rng):
            return {
                "alpha": float(
                    np.exp(np.log(60.0) + 0.3 * rng.standard_normal())
                ),
                "beta": float(np.exp(0.2 * rng.standard_normal())),
                "coef_0": float(0.5 * rng.standard_normal()),
                "coef_1": float(0.5 * rng.standard_normal()),
            }

        def fit(data):
            return ProportionalIntensityNHPP.fit_bayes(
                *data, baseline=CrowAMSAA, priors=priors
            )

    names = list(priors)

    def simulate(t, rng):
        coef = np.array([t[k] for k in names if k.startswith("coef")])
        xs, Zs, ids, cs = [], [], [], []
        for k, z in enumerate(design):
            if baseline == "hpp":
                m = rng.poisson(t["lambda"] * np.exp(z @ coef) * end)
                times = np.sort(end * rng.uniform(size=m))
            else:
                base = (end / t["alpha"]) ** t["beta"]
                m = rng.poisson(base * np.exp(z @ coef))
                u = np.sort(base * rng.uniform(size=m))
                times = t["alpha"] * u ** (1 / t["beta"])
            xs += [*times, end]
            Zs += [list(z)] * (m + 1)
            ids += [k] * (m + 1)
            cs += [0] * m + [1]
        return xs, Zs, ids, cs

    return truth, simulate, fit, names, seed, rules


SCENARIOS = {
    "cox-lewis": _cox_lewis,
    "cox-lewis-1000": _cox_lewis_again,
    "pi-hpp": lambda: _pi("hpp"),
    "pi-crow-amsaa": lambda: _pi("crow-amsaa"),
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


#: Scenarios that miss their rule as fixed (see the module's
#: description): reported, not re-seeded.
MISSES = {
    "cox-lewis": "alpha's 90% coverage 0.847 against a limit of 0.848; "
    "1,000 further repetitions (cox-lewis-1000) are inside the rule",
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=(
                [pytest.mark.xfail(reason=MISSES[name], strict=True)]
                if name in MISSES
                else []
            ),
        )
        for name in SCENARIOS
    ],
)
def test_sbc(name):
    truth, simulate, fit, names, seed, rules = SCENARIOS[name]()
    rules = dict(rules)
    reps = rules.pop("reps", REPS)
    result = kit.sbc(truth, simulate, fit, names, reps, seed)
    print(name, kit.describe_sbc(result))
    kit.check_sbc(result, **rules)
