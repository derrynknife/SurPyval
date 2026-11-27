"""Recovery and coverage of the inverse Gaussian and positive stable shared
frailties: ``Frailty(Weibull, family=...)``, ``CoxFrailty.fit(...,
family=...)`` and ``CrowAMSAA.fit(..., frailty=...)``, and the size of the
likelihood-ratio test of no frailty with its 50:50 boundary mixture.

The truths, simulated from the definitions (not with the package):

* grouped lifetimes: ``G = 100`` groups of 4, a Weibull baseline (scale
  10, shape 1.5), a binary covariate with conditional coefficient 0.5,
  censoring at 30; the frailty inverse Gaussian of mean 1 and variance
  ``theta = 0.5``, or positive stable of index ``1 - theta`` with
  ``theta = 0.3`` (drawn by Kanter's representation);
* recurrent events: ``N = 150`` units under a power-law intensity (scale
  20, shape 1.3), each watched to a uniform (60, 120) age, with the same
  frailties;
* the null: the grouped design with no frailty, for the size of the test
  of ``theta = 0`` (twice the gain in log-likelihood over the fit without
  frailty, against half a point mass at 0 and half a chi-square(1)).

Checked: the bias of every parameter (with its reported standard error
against its spread) and the coverage of its Wald interval
(``param_cb``), and the test's size at 5%. The seeds, replicates, sizes and
tolerances (the suite's: 3 Monte Carlo standard errors plus 0.01 for rates
and coverages, 0.2 of a standard deviation for the bias) were fixed
before any run. ``LAPLACE_FRAILTY_REPS`` runs every study with fewer
replicates (a quick look; the tolerances follow the replicates).
"""

import os
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from surpyval import CoxFrailty, Frailty, Weibull, WeibullPH
from surpyval.recurrent import CrowAMSAA
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    check_rate,
    process_pool,
)

SEED = 77123
REPS = int(os.environ.get("LAPLACE_FRAILTY_REPS", 200))
REPS_STABLE_RECURRENT = int(os.environ.get("LAPLACE_FRAILTY_REPS", 100))
REPS_NULL = int(os.environ.get("LAPLACE_FRAILTY_REPS", 300))
G, PER = 100, 4
SCALE, SHAPE, BETA, CENSOR = 10.0, 1.5, 0.5, 30.0
N_UNITS, R_SCALE, R_SHAPE = 150, 20.0, 1.3
THETA = {"inverse_gaussian": 0.5, "positive_stable": 0.3}
WORKERS = min(os.cpu_count() or 1, 4)


def _frailty(rng, family, theta, size):
    if theta == 0:
        return np.ones(size)
    if family == "inverse_gaussian":
        return rng.wald(1.0, 1.0 / theta, size)
    alpha = 1.0 - theta
    U = rng.uniform(0, np.pi, size)
    E = rng.exponential(1, size)
    A = (
        (np.sin(alpha * U) / np.sin(U)) ** (1 / (1 - alpha))
        * np.sin((1 - alpha) * U)
        / np.sin(alpha * U)
    )
    return (A / E) ** ((1 - alpha) / alpha)


def _grouped(r, family, theta):
    rng = np.random.default_rng([SEED, r, len(family), int(theta * 10)])
    g = np.repeat(np.arange(G), PER)
    u = _frailty(rng, family, theta, G)
    z = rng.binomial(1, 0.5, g.size).astype(float)
    t = SCALE * (rng.exponential(size=g.size) / (u[g] * np.exp(BETA * z))) ** (
        1 / SHAPE
    )
    return np.minimum(t, CENSOR), z[:, None], (t > CENSOR).astype(int), g


def _recurrent(r, family, theta):
    rng = np.random.default_rng([SEED, r, len(family), 7])
    u = _frailty(rng, family, theta, N_UNITS)
    end = rng.uniform(60.0, 120.0, N_UNITS)
    x, i, c = [], [], []
    for k in range(N_UNITS):
        top = (end[k] / R_SCALE) ** R_SHAPE
        n = rng.poisson(u[k] * top)
        times = R_SCALE * np.sort(rng.uniform(0.0, top, n)) ** (1 / R_SHAPE)
        x += [*times, end[k]]
        i += [k] * (n + 1)
        c += [0] * n + [1]
    return np.array(x), np.array(i), np.array(c)


def _wald(model):
    return np.array([model.param_cb(n) for n in model.parameter_names])


def _parametric(args):
    r, family = args
    x, Z, c, g = _grouped(r, family, THETA[family])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Frailty(Weibull, family=family).fit(x, Z=Z, c=c, groups=g)
        return model.params, model.standard_errors(), _wald(model)


def _cox(args):
    r, family = args
    x, Z, c, g = _grouped(r, family, THETA[family])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family=family)
        return model.params, model.standard_errors(), _wald(model)


def _recurrent_fit(args):
    r, family = args
    x, i, c = _recurrent(r, family, THETA[family])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(x, i, c, frailty=family)
        return model.params, model.standard_errors(), _wald(model)


def _null(args):
    r, family = args
    x, Z, c, g = _grouped(r, family, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Frailty(Weibull, family=family).fit(x, Z=Z, c=c, groups=g)
        ph = WeibullPH.fit(x, Z=Z, c=c)
        cox = CoxFrailty.fit(x, Z=Z, c=c, groups=g, family=family)
    out = []
    for gain in (
        ph.neg_ll() - model._neg_ll,
        cox.log_likelihood - cox.log_likelihood_no_frailty,
    ):
        statistic = max(0.0, 2.0 * gain)
        out.append(0.5 * chi2.sf(statistic, 1) if statistic > 1e-9 else 1.0)
    return out, model.theta == 0, cox.theta == 0


def _study(job, family, reps, truth, label):
    with process_pool(max_workers=WORKERS) as pool:
        out = list(pool.map(job, [(r, family) for r in range(reps)]))
    params = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    wald = np.array([o[2] for o in out])
    check_bias(params, truth, label, standard_errors=se)
    check_coverage(wald[:, :, 0], wald[:, :, 1], truth, 0.95, label + " Wald")


def test_parametric_inverse_gaussian():
    truth = np.array([SCALE, SHAPE, BETA, THETA["inverse_gaussian"]])
    _study(_parametric, "inverse_gaussian", REPS, truth, "WeibullFrailty[IG]")


def test_parametric_positive_stable():
    truth = np.array([SCALE, SHAPE, BETA, THETA["positive_stable"]])
    _study(_parametric, "positive_stable", REPS, truth, "WeibullFrailty[PS]")


def test_cox_inverse_gaussian():
    truth = np.array([BETA, THETA["inverse_gaussian"]])
    _study(_cox, "inverse_gaussian", REPS, truth, "CoxFrailty[IG]")


def test_cox_positive_stable():
    truth = np.array([BETA, THETA["positive_stable"]])
    _study(_cox, "positive_stable", REPS, truth, "CoxFrailty[PS]")


def test_recurrent_inverse_gaussian():
    truth = np.array([R_SCALE, R_SHAPE, THETA["inverse_gaussian"]])
    _study(_recurrent_fit, "inverse_gaussian", REPS, truth, "CrowAMSAA[IG]")


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="the positive stable group integral takes at most 1000 events "
    "in a group (its recursion is O(D^2)), and this design's heavy-tailed "
    "frailty gives a unit more than that in about 81% of simulated fleets "
    "(32 of the first 40 replicates; the largest 636,079 events). No "
    "positive stable truth keeps every replicate under the cap, and "
    "keeping only the fleets under it would bias the study: it waits for "
    "a form of the integral whose cost does not grow with the count.",
)
def test_recurrent_positive_stable():
    truth = np.array([R_SCALE, R_SHAPE, THETA["positive_stable"]])
    _study(
        _recurrent_fit,
        "positive_stable",
        REPS_STABLE_RECURRENT,
        truth,
        "CrowAMSAA[PS]",
    )


def test_size_of_the_test_of_no_frailty():
    for family in ("inverse_gaussian", "positive_stable"):
        with process_pool(max_workers=WORKERS) as pool:
            out = list(
                pool.map(_null, [(r, family) for r in range(REPS_NULL)])
            )
        p = np.array([o[0] for o in out])
        for k, label in enumerate(("WeibullFrailty", "CoxFrailty")):
            check_rate(
                int(np.sum(p[:, k] < 0.05)),
                REPS_NULL,
                0.05,
                f"{label}[{family}] size of the test of theta = 0 at 5%",
            )
        print(
            "{}: theta on its boundary in {:.2f} (parametric) and {:.2f} "
            "(Cox) of the null fits (about half expected; not "
            "asserted)".format(
                family,
                np.mean([o[1] for o in out]),
                np.mean([o[2] for o in out]),
            )
        )
