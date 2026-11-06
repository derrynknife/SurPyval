"""Simulation-based calibration of the Bayesian fits (Talts et al., 2018).

If a Bayesian fit is right, then drawing the parameters from the prior,
simulating data from them and fitting gives a posterior in which the true
parameters are as likely to fall at any rank among the posterior's draws:
the ranks are uniform. A fit that is too sure of itself piles the ranks at
the ends; one biased one way piles them at one end.

The rules and seeds were fixed before the first run (DECISIONS.md,
2026-10-29):

- 500 replicates per scenario; replicate ``r`` draws the true parameters
  from the prior and simulates its data with seed ``8100 + r``, and takes
  999 posterior draws with seed ``8600 + r``;
- the rank of the truth among the draws, for each parameter and for the
  B10 life, in 20 bins, passes a chi-square test of uniformity at the
  0.001 level (about 30 tests in all: a Bonferroni note, not a correction);
- the 95% interval's coverage (the truth between the 2.5% and 97.5% points
  of the draws) is within 0.02 of 0.95;
- a replicate whose fit refuses fails the scenario.

The censoring, inspection and truncation times are fixed for each
scenario, from the prior's central model (each parameter at its prior
median). They must not depend on the true parameters of a replicate: a
design that did would tell the fit something about the truth that the
likelihood leaves out, and the ranks could not be uniform. (The study's
first runs set them from the truth, and failed for that reason; see
DECISIONS.md.)

Scenarios: Weibull with 20 complete values (weakly informative and
informative priors), with 30% and 80% Type I censoring, inspections
(interval censored), left truncation, and about 3 failures among 40
units; LogNormal, Gamma and LogLogistic with 30% Type I censoring; and the
Exponential with a Gamma prior (the closed form) with 30% censoring.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.bayes.prior import NormalOnScale

pytestmark = pytest.mark.calibration

REPS = 500
DRAWS = 999
BINS = 20


def _log(loc, scale):
    return NormalOnScale(loc=np.log(loc), scale=scale, kind="log")


def _normal(loc, scale):
    return NormalOnScale(loc=loc, scale=scale, kind="identity")


PRIORS = {
    "weak": {"alpha": _log(100.0, 1.0), "beta": _log(1.5, 0.5)},
    "informative": {"alpha": _log(100.0, 0.3), "beta": _log(1.5, 0.15)},
    "lognormal": {"mu": _normal(4.5, 0.5), "sigma": _log(0.8, 0.4)},
    "gamma": {"alpha": _log(2.0, 0.4), "beta": _log(0.02, 0.5)},
    "loglogistic": {"alpha": _log(100.0, 0.5), "beta": _log(3.0, 0.3)},
}


def _draw_prior(dist, priors, rng):
    """True parameters from the prior (each normal on its own scale)."""
    out = []
    for name in dist.parameter_names:
        p = priors[name]
        phi = rng.normal(p.loc, p.scale)
        out.append(float(np.exp(phi)) if p.kind == "log" else float(phi))
    return out


def _centre(dist, priors):
    """The prior's central model: each parameter at its prior median."""
    out = []
    for name in dist.parameter_names:
        p = priors[name]
        out.append(float(np.exp(p.loc)) if p.kind == "log" else p.loc)
    return out


def _type_one(dist, theta, n, censored, rng, centre):
    """Censored at the time the prior's central model has ``censored``
    of its units still running (fixed for the scenario: a design that
    depended on the true parameters would tell the fit about them)."""
    t = dist.random(n, *theta, random_state=rng)
    cut = float(dist.qf(1 - censored, *centre))
    return dict(x=np.minimum(t, cut), c=(t > cut).astype(int))


def _complete(dist, theta, n, rng, centre):
    return dict(x=dist.random(n, *theta, random_state=rng))


def _inspections(dist, theta, n, rng, centre):
    """Inspected every third of the prior's central median, 12 times:
    failed in an interval (the first, before the first inspection, left
    censored), or still running at the last."""
    t = dist.random(n, *theta, random_state=rng)
    step = float(dist.qf(0.5, *centre)) / 3
    k = np.floor(t / step)
    last = 12
    xl = np.where(k >= last, last * step, k * step)
    xr = np.where(k >= last, last * step, (k + 1) * step)
    c = np.where(k >= last, 1, np.where(k == 0, -1, 2))
    x = np.where((c == 2)[:, None], np.c_[xl, xr], np.c_[xr, xr])
    x = np.where((c == 1)[:, None], np.c_[xl, xl], x)
    return dict(x=x, c=c)


def _truncated(dist, theta, n, rng, centre):
    """Entered at the prior's central 20% point: only units still running
    then are seen.

    The draw is made on the survival scale: a unit seen after ``tl`` has
    ``S(x) = S(tl) (1 - U)``, so its cumulative hazard is
    ``H(tl) - log(1 - U)`` and ``x = alpha H**(1 / beta)`` (Weibull, the
    only distribution this scenario uses). The same uniforms as the
    inverse-cdf draw ``qf(U(F(tl), 1))``, which it equals in exact
    arithmetic; that draw gave ``inf`` when ``F(tl)`` rounded to 1 (a
    truth whose life is far shorter than the entry age), while this one
    stays finite."""
    if dist is not sp.Weibull:
        raise NotImplementedError("the truncated draw is written for Weibull")
    tl = float(dist.qf(0.2, *centre))
    alpha, beta = theta
    u = rng.uniform(0.0, 1.0, n)
    H = (tl / alpha) ** beta - np.log1p(-u)
    return dict(x=alpha * H ** (1 / beta), tl=np.full(n, tl))


def _few_failures(dist, theta, n, rng, centre):
    """40 units on test until the prior's central 7.5% point: about 3
    failures."""
    return _type_one(dist, theta, n, 1 - 3 / 40, rng, centre)


SCENARIOS = {
    "Weibull complete, weak": (sp.Weibull, "weak", _complete, 20),
    "Weibull complete, informative": (
        sp.Weibull,
        "informative",
        _complete,
        20,
    ),
    "Weibull 30% censored": (
        sp.Weibull,
        "weak",
        lambda d, t, n, r, c: _type_one(d, t, n, 0.3, r, c),
        20,
    ),
    "Weibull 80% censored": (
        sp.Weibull,
        "weak",
        lambda d, t, n, r, c: _type_one(d, t, n, 0.8, r, c),
        30,
    ),
    "Weibull inspections": (sp.Weibull, "weak", _inspections, 20),
    "Weibull left truncated": (sp.Weibull, "weak", _truncated, 20),
    "Weibull 3 failures in 40": (sp.Weibull, "weak", _few_failures, 40),
    "LogNormal 30% censored": (
        sp.LogNormal,
        "lognormal",
        lambda d, t, n, r, c: _type_one(d, t, n, 0.3, r, c),
        20,
    ),
    "Gamma 30% censored": (
        sp.Gamma,
        "gamma",
        lambda d, t, n, r, c: _type_one(d, t, n, 0.3, r, c),
        20,
    ),
    "LogLogistic 30% censored": (
        sp.LogLogistic,
        "loglogistic",
        lambda d, t, n, r, c: _type_one(d, t, n, 0.3, r, c),
        20,
    ),
}

EXPONENTIAL_PRIOR = (3.0, 300.0)


def _ranks(dist, priors_key, simulate, n):
    names = list(dist.parameter_names) + ["B10"]
    ranks = {k: [] for k in names}
    refused = []
    for r in range(REPS):
        rng = np.random.default_rng(8100 + r)
        if dist is sp.Exponential:
            a, b = EXPONENTIAL_PRIOR
            theta = [float(rng.gamma(a, 1 / b))]
            priors = {"failure_rate": sp.Gamma.from_params([a, b])}
            centre = [a / b]
        else:
            priors = PRIORS[priors_key]
            theta = _draw_prior(dist, priors, rng)
            centre = _centre(dist, priors)
        data = simulate(dist, theta, n, rng, centre)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = dist.fit_bayes(**data, priors=priors)
        except ValueError as e:
            refused.append((r, str(e)[:120]))
            continue
        draws = model.posterior.draws(DRAWS, random_state=8600 + r)
        cols = model.posterior.parameter_names
        for k, name in enumerate(dist.parameter_names):
            d = draws[:, cols.index(name)]
            ranks[name].append(int(np.sum(d < theta[k])))
        b10 = dist.qf(
            0.1, *[draws[:, cols.index(p)] for p in dist.parameter_names]
        )
        ranks["B10"].append(int(np.sum(b10 < dist.qf(0.1, *theta))))
    return ranks, refused


def _check(label, ranks, refused):
    assert not refused, f"{label}: {len(refused)} fits refused: {refused[:3]}"
    failures = []
    for name, rk in ranks.items():
        rk = np.asarray(rk)
        counts = np.bincount(rk * BINS // (DRAWS + 1), minlength=BINS)
        expected = len(rk) / BINS
        stat = float(np.sum((counts - expected) ** 2 / expected))
        p = float(stats.chi2.sf(stat, BINS - 1))
        cover = float(np.mean((rk >= 25) & (rk < DRAWS + 1 - 25)))
        print(
            f"{label} {name}: chi-square p = {p:.4f}, "
            f"95% coverage {cover:.3f}"
        )
        if p < 0.001:
            failures.append(f"{name}: ranks not uniform (p = {p:.2e})")
        if abs(cover - 0.95) > 0.02:
            failures.append(f"{name}: 95% coverage {cover:.3f}")
    assert not failures, f"{label}: " + "; ".join(failures)


# The results that fail the rules, kept as strict expected failures with
# the numbers (DECISIONS.md, 2026-10-29); the rules are not loosened.
_FOUND = {
    "Weibull complete, informative": (
        "B10's 95% coverage 0.978 (rule: within 0.02 of 0.95; chi-square p "
        "= 0.0094 passes). The parameters' are 0.950 and 0.956. On 1,000 "
        "fresh replicates (seeds 9100 + r and 9600 + r, an investigation, "
        "not the study) B10's coverage was 0.954: taken as this seed set's "
        "excursion"
    ),
    "Weibull left truncated": (
        "With the draw on the survival scale (no replicate refused; before, "
        "r = 299's simulated time was infinite), the ranks are uniform "
        "(chi-square p = 0.43 alpha, 0.55 beta, 0.21 B10) but the 95% "
        "coverage is 0.972 for alpha and 0.970 for B10 (rule: within 0.02 "
        "of 0.95; beta's is 0.962). On 1,000 fresh replicates (seeds "
        "9100 + r and 9600 + r, an investigation, not the study) it was "
        "0.958, 0.937 and 0.933 with p = 0.94, 0.43 and 0.74: taken as this "
        "seed set's excursion"
    ),
}


def _scenarios():
    return [
        pytest.param(
            label,
            marks=(
                [pytest.mark.xfail(strict=True, reason=_FOUND[label])]
                if label in _FOUND
                else []
            ),
        )
        for label in sorted(SCENARIOS)
    ]


@pytest.mark.parametrize("label", _scenarios())
def test_simulation_based_calibration(label):
    dist, key, simulate, n = SCENARIOS[label]
    ranks, refused = _ranks(dist, key, simulate, n)
    _check(label, ranks, refused)


def test_simulation_based_calibration_closed_form():
    ranks, refused = _ranks(
        sp.Exponential,
        None,
        lambda d, t, n, r, c: _type_one(d, t, n, 0.3, r, c),
        20,
    )
    _check("Exponential (Gamma prior) 30% censored", ranks, refused)
