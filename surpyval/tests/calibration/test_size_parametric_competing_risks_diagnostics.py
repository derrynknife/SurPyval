"""Size and power of the parametric competing-risks checks of
``diagnose()``.

Two causes, ``a`` a Weibull (scale 10, shape 2) and ``b``, uniform
censoring on (0, 25), n = 100, 49 bootstrap refits, judged at 10% (with 49
refits a p-value is at most 0.1 for exactly 5 of 50 ranks):

- **Size**: ``b`` a LogNormal (2.5, 0.6), each cause fitted with its true
  distribution. The Anderson-Darling test of each cause's Cox-Snell
  residuals (the univariate family's bootstrap on the cause's model) must
  not reject more than 10% (slack 0.01; the univariate test is somewhat
  conservative with censoring), and the incidence test (the largest gap
  to the Aalen-Johansen estimate) must reject 10% (slack 0.01).
- **Power, by cause**: ``b`` is 4 plus a Weibull (scale 6, shape 1), no
  failures before 4, and both causes are fitted with a 2-parameter
  Weibull. ``b``'s goodness of fit and incidence must reject at least half
  the time; ``a``'s goodness of fit, whose model is right, must still not
  reject more than 10% (slack 0.01): the wrong cause is the one flagged.
"""

import warnings

import numpy as np

import surpyval as sp
from surpyval.diagnostics.competing_risks import incidence_bootstrap
from surpyval.diagnostics.univariate import bootstrap
from surpyval.tests.calibration._montecarlo import check_rate
from surpyval.univariate.competing_risks import ParametricCompetingRisks

ALPHA = 0.1
N = 100
N_BOOT = 49
REPS = 200


def _data(rng, wrong):
    t_a = 10 * rng.weibull(2.0, N)
    if wrong:
        t_b = 4 + 6 * rng.weibull(1.0, N)
    else:
        t_b = rng.lognormal(2.5, 0.6, N)
    cens = rng.uniform(0, 25, N)
    first = np.minimum(t_a, t_b)
    e = np.where(cens < first, None, np.where(t_a < t_b, "a", "b"))
    return np.minimum(first, cens), e.astype(object)


def _p_values(rng, wrong):
    x, e = _data(rng, wrong)
    dist = sp.Weibull if wrong else {"a": sp.Weibull, "b": sp.LogNormal}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricCompetingRisks.fit(x, e, dist=dist)
        seeds = rng.integers(0, 2**63 - 1, size=3)
        gof = {
            k: bootstrap(model.models[k], N_BOOT, int(s))["p"]["AD"]
            for k, s in zip(("a", "b"), seeds)
        }
        incidence = incidence_bootstrap(model, N_BOOT, int(seeds[2]))["p"]
    return gof, incidence


def _rates(seed, wrong):
    rng = np.random.default_rng(seed)
    gof = {"a": 0, "b": 0}
    incidence = {"a": 0, "b": 0}
    for _ in range(REPS):
        g, i = _p_values(rng, wrong)
        for k in ("a", "b"):
            gof[k] += g[k] <= ALPHA
            incidence[k] += i[k] <= ALPHA
    return gof, incidence


def test_parametric_competing_risks_checks_size():
    gof, incidence = _rates(289, wrong=False)
    for k in ("a", "b"):
        check_rate(gof[k], REPS, ALPHA, f"cause {k} AD size", side="upper")
        check_rate(incidence[k], REPS, ALPHA, f"cause {k} incidence size")


def test_parametric_competing_risks_flags_the_wrong_cause():
    gof, incidence = _rates(290, wrong=True)
    check_rate(gof["b"], REPS, 0.5, "cause b AD power", side="lower")
    check_rate(
        incidence["b"], REPS, 0.5, "cause b incidence power", side="lower"
    )
    check_rate(gof["a"], REPS, ALPHA, "cause a AD (right model)", side="upper")
    print(
        "cause a incidence rejected: {:.3f} (not judged: it depends on "
        "b's survival too)".format(incidence["a"] / REPS)
    )
