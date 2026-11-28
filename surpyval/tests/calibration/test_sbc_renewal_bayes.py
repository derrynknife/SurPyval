"""Simulation-based calibration of the renewal and imperfect-repair
processes' Bayesian fits (``GeneralizedRenewal.fit_bayes``,
``ARA.fit_bayes``, ``ARI.fit_bayes``), by the validation kit's template
(``surpyval/tests/bayes/kit.py``).

In words: the true parameters are drawn from the priors, the systems'
failures are simulated from them, and the fit's posterior should then
put the truth below its ``p`` point in a share ``p`` of repetitions. The
rules are the kit's (50% and 90% coverage within three standard errors,
the truth's posterior probability uniform by Kolmogorov-Smirnov at
0.001, no fit refused). Each scenario and its seed were fixed before the
first run (27 Nov 2026). Truths at an end of the repair parameter's
range are covered by priors that pile their mass there (a calibration
draws its truths from the fit's own prior, which is continuous):

- ``grp-near-perfect``: Kijima I, four systems each observed from 0 to
  40; ``q`` exponential with mean 0.1 (a tenth of the truths within
  0.01 of perfect repair, two-fifths within 0.05), ``alpha`` log-normal
  about 10 (sigma 0.2), ``beta`` log-normal about 2.5 (sigma 0.15):
  about five failures a system. 300 repetitions, seed 70100.
- ``ara-near-perfect``: ARA memory 2, the same design and life; ``rho``
  Beta(5, 1) (a twentieth of the truths within 0.01 of perfect repair,
  ``rho = 1``, more than a fifth within 0.05). 300 repetitions, seed
  70400.
- ``grp-ii``: Kijima II, the same design and life; ``q`` uniform on
  (0, 1). 300 repetitions, seed 71000.
- ``ari``: ARI memory 1 on a power-law baseline, four systems observed
  from 0 to 40; ``rho`` uniform on (0, 1), ``alpha`` log-normal about 5
  (sigma 0.2), ``beta`` log-normal about 1.8 (sigma 0.1). 300
  repetitions, seed 70700.

The data are simulated from the processes' definitions (each gap drawn
from the life conditioned on the virtual age it starts at, or from the
reduced intensity by inverting its compensator), not with the package's
simulators.
"""

import sys

import numpy as np
import pytest
from scipy import optimize

import surpyval as sp
from surpyval.recurrent import ARA, ARI, CrowAMSAA, GeneralizedRenewal
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 300
END, SYSTEMS = 40.0, 4
LIFE = {
    "alpha": (np.log(10.0), 0.2),
    "beta": (np.log(2.5), 0.15),
}


def _lognormal_truths(rng, spec):
    return {
        k: float(np.exp(mu + sigma * rng.standard_normal()))
        for k, (mu, sigma) in spec.items()
    }


def _lognormal_priors(spec):
    return {
        k: sp.LogNormal.from_params([mu, s]) for k, (mu, s) in spec.items()
    }


def _virtual_age_systems(t, rng, age_after):
    """Each system's failure times: each gap from the Weibull life
    conditioned on surviving to the virtual age it starts at, and the
    age after a failure from ``age_after(history, gaps)``."""
    a, b = t["alpha"], t["beta"]
    xs, ids, cs = [], [], []
    for k in range(SYSTEMS):
        now, v, history, gaps = 0.0, 0.0, [], []
        while True:
            e = -np.log(rng.uniform())
            gap = a * ((v / a) ** b + e) ** (1 / b) - v
            if now + gap > END:
                break
            now += gap
            history.append(now)
            gaps.append(gap)
            v = age_after(history, gaps)
        xs += [*history, END]
        ids += [k] * (len(history) + 1)
        cs += [0] * len(history) + [1]
    return xs, ids, cs


def _grp(kijima, q_prior, q_truth, seed):
    priors = {"q": q_prior, **_lognormal_priors(LIFE)}

    def truth(rng):
        return {"q": float(q_truth(rng)), **_lognormal_truths(rng, LIFE)}

    def simulate(t, rng):
        q = t["q"]

        def age_after(history, gaps):
            if kijima == "i":
                # v_n = v_{n-1} + q x_n = q T_n
                return q * history[-1]
            v = 0.0
            for g in gaps:
                v = q * (v + g)
            return v

        return _virtual_age_systems(t, rng, age_after)

    def fit(data):
        return GeneralizedRenewal.fit_bayes(
            *data, kijima=kijima, priors=priors
        )

    return truth, simulate, fit, list(priors), seed, {}


def _ara():
    priors = {
        "rho": sp.Beta.from_params([5.0, 1.0]),
        **_lognormal_priors(LIFE),
    }

    def truth(rng):
        return {
            "rho": float(rng.beta(5.0, 1.0)),
            **_lognormal_truths(rng, LIFE),
        }

    def simulate(t, rng):
        rho = t["rho"]

        def age_after(history, gaps):
            # memory 2: v = T_n - rho (T_n + (1 - rho) T_{n-1})
            recent = history[::-1][:2]
            return history[-1] - rho * sum(
                (1 - rho) ** j * x for j, x in enumerate(recent)
            )

        return _virtual_age_systems(t, rng, age_after)

    def fit(data):
        return ARA.fit_bayes(*data, m=2, priors=priors)

    return truth, simulate, fit, list(priors), 70400, {}


def _ari():
    base = {"alpha": (np.log(5.0), 0.2), "beta": (np.log(1.8), 0.1)}
    priors = {"rho": sp.prior.uniform(0, 1), **_lognormal_priors(base)}

    def truth(rng):
        return {"rho": float(rng.uniform()), **_lognormal_truths(rng, base)}

    def simulate(t, rng):
        rho, a, b = t["rho"], t["alpha"], t["beta"]

        def cif(x):
            return (x / a) ** b

        def iif(x):
            return b / a * (x / a) ** (b - 1)

        xs, ids, cs = [], [], []
        for k in range(SYSTEMS):
            now, reduction, history = 0.0, 0.0, []
            while True:
                e = -np.log(rng.uniform())

                # memory 1: the intensity after a failure at T_n is
                # lambda_0(t) - rho lambda_0(T_n)
                def g(x, now=now, reduction=reduction, e=e):
                    return cif(now + x) - cif(now) - reduction * x - e

                hi = 1.0
                while g(hi) < 0 and now + hi <= END:
                    hi *= 2.0
                if g(hi) < 0:
                    break
                gap = optimize.brentq(g, 0.0, hi, xtol=1e-12)
                if now + gap > END:
                    break
                now += gap
                history.append(now)
                reduction = rho * iif(now)
            xs += [*history, END]
            ids += [k] * (len(history) + 1)
            cs += [0] * len(history) + [1]
        return xs, ids, cs

    def fit(data):
        return ARI.fit_bayes(*data, m=1, baseline=CrowAMSAA, priors=priors)

    return truth, simulate, fit, list(priors), 70700, {}


SCENARIOS = {
    "grp-near-perfect": lambda: _grp(
        "i",
        sp.Exponential.from_params([10.0]),
        lambda rng: rng.exponential(0.1),
        70100,
    ),
    "ara-near-perfect": _ara,
    "grp-ii": lambda: _grp(
        "ii", sp.prior.uniform(0, 1), lambda rng: rng.uniform(), 71000
    ),
    "ari": _ari,
}


LINUX = sys.platform.startswith("linux")

#: Misses on the 0.30 release run (x86-64 Linux), released as known
#: limitations (the maintainer's decision, 28 Nov 2026), to be fixed.
_RELEASE_MISSES = {
    "grp-near-perfect": "1 of 300 fits refused on x86-64 Linux (repetition "
    "168: the grid's two offset summaries differ by 1.25% of the 95% "
    "interval's width); the rule is none",
}


@pytest.mark.parametrize(
    "name",
    [
        (
            pytest.param(
                name,
                marks=pytest.mark.xfail(
                    LINUX, strict=True, reason=_RELEASE_MISSES[name]
                ),
            )
            if name in _RELEASE_MISSES
            else name
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
