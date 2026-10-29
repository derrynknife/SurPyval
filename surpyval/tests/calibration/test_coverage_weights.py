"""Coverage of the robust intervals of fits with sampling weights ``w``.

Three studies, their seeds and pass rules fixed before the first run (see
DECISIONS.md, 2026-10-29):

- **Informative sampling.** A population of 1,000 units, lifetimes
  Weibull(10, 1.5), right censored at a uniform time on (0, 30). Long-lived
  units are sampled more: a unit whose lifetime is above the population
  median (8.73) is kept with probability 0.8, one below it with 0.2, and
  each kept unit has the weight 1 / (its probability). The unweighted fits
  are biased towards long lives; the weighted ones are not. Checked: the
  coverage of the robust 95% and 90% intervals of the Kaplan-Meier,
  Nelson-Aalen and Fleming-Harrington ``sf`` at the population quartiles,
  of the Weibull ``alpha`` and ``beta`` (``param_cb``), the Weibull ``sf``
  at the quartiles and its median (``quantile_cb``).
- **Weight scale.** The same samples with every weight multiplied by 0.1
  and by 10 give the same intervals (to 1e-8), so the same coverage: the
  property the classical variances lack (a "95%" Greenwood interval
  covered about a third of the time once the weights were multiplied by
  10).
- **Inverse probability of treatment weights.** A confounder ``X ~ N(0,
  1)``, treatment with probability ``expit(0.8 X)``, lifetimes Weibull PH
  with log hazard ratios 0.5 (treatment) and 0.7 (``X``). The weights
  ``1 / P(treatment received | X)`` are estimated by a logistic
  regression in each replicate, and the weighted Kaplan-Meier of the
  treated estimates their marginal survival ``E_X S(t | treated, X)``.
  The sandwich ignores that the weights were estimated, which makes it
  conservative: the check is one-sided (coverage at least nominal less
  the tolerance) and the coverage is reported.

Tolerance: ``Z_TOL`` Monte Carlo standard errors plus a slack of 0.02 (the
weighted fits' intervals are asymptotic in the effective sample size,
about 350 here).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, check_rate

TRUE = sp.Weibull.from_params([10.0, 1.5])
N_POP = 1000
C_MAX = 30.0
SLACK = 0.02
REPS = 1000
QUARTILES = TRUE.qf(np.array([0.25, 0.5, 0.75]))
MEDIAN = float(TRUE.qf(0.5))


def _informative_sample(rng, scale=1.0):
    t = TRUE.qf(rng.uniform(size=N_POP))
    cens = rng.uniform(0, C_MAX, N_POP)
    prob = np.where(t > MEDIAN, 0.8, 0.2)
    keep = rng.uniform(size=N_POP) < prob
    x = np.minimum(t, cens)[keep]
    c = (cens < t).astype(int)[keep]
    return x, c, scale / prob[keep]


@pytest.fixture(scope="module")
def informative():
    """Every replicate's bounds, at each weight scale."""
    rng = np.random.default_rng(26101)
    out = {}
    names = ("KaplanMeier", "NelsonAalen", "FlemingHarrington")
    keys = [f"{nm}{a}" for nm in names for a in (95, 90)]
    keys += ["alpha", "beta", "alpha90", "beta90", "wsf", "wsf90", "median"]
    for key in keys:
        out[key] = ([], [])
    out["scale_same"] = []
    out["unweighted_alpha"] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(REPS):
            x, c, w = _informative_sample(rng)
            for nm in names:
                model = getattr(sp, nm).fit(x, c, w=w)
                for a, alpha_ci in ((95, 0.05), (90, 0.10)):
                    b = model.cb(QUARTILES, alpha_ci=alpha_ci)
                    out[f"{nm}{a}"][0].append(b[:, 0])
                    out[f"{nm}{a}"][1].append(b[:, 1])
            wb = sp.Weibull.fit(x, c, w=w)
            for name in ("alpha", "beta"):
                for a, alpha_ci in (("", 0.05), ("90", 0.10)):
                    b = wb.param_cb(name, alpha_ci=alpha_ci)
                    out[name + a][0].append(b[0])
                    out[name + a][1].append(b[1])
            for a, alpha_ci in (("", 0.05), ("90", 0.10)):
                b = wb.cb(QUARTILES, alpha_ci=alpha_ci)
                out["wsf" + a][0].append(b[:, 0])
                out["wsf" + a][1].append(b[:, 1])
            q = wb.quantile_cb(0.5)
            out["median"][0].append(q[0])
            out["median"][1].append(q[1])
            # The weight scale: the same intervals at w / 10 and 10 w
            km = sp.KaplanMeier.fit(x, c, w=w).cb(QUARTILES)
            same = True
            for s in (0.1, 10.0):
                km_s = sp.KaplanMeier.fit(x, c, w=w * s).cb(QUARTILES)
                wb_s = sp.Weibull.fit(x, c, w=w * s)
                same &= np.allclose(km_s, km, rtol=1e-8, atol=0)
                same &= np.allclose(
                    wb_s.cb(QUARTILES),
                    wb.cb(QUARTILES),
                    rtol=1e-6,
                    atol=0,
                )
            out["scale_same"].append(same)
            out["unweighted_alpha"].append(sp.Weibull.fit(x, c).params[0])
    return out


@pytest.mark.parametrize(
    "name", ["KaplanMeier", "NelsonAalen", "FlemingHarrington"]
)
@pytest.mark.parametrize("level", [95, 90])
def test_nonparametric_coverage(informative, name, level):
    lo, hi = informative[f"{name}{level}"]
    check_coverage(
        np.array(lo),
        np.array(hi),
        TRUE.sf(QUARTILES),
        level / 100,
        f"weighted {name} cb(sf) {level}%",
        slack=SLACK,
    )


@pytest.mark.parametrize("level", [95, 90])
def test_weibull_coverage(informative, level):
    suffix = "" if level == 95 else "90"
    for k, name in enumerate(("alpha", "beta")):
        lo, hi = informative[name + suffix]
        check_coverage(
            np.array(lo),
            np.array(hi),
            TRUE.params[k],
            level / 100,
            f"weighted Weibull param_cb({name!r}) {level}%",
            slack=SLACK,
        )
    lo, hi = informative["wsf" + suffix]
    check_coverage(
        np.array(lo),
        np.array(hi),
        TRUE.sf(QUARTILES),
        level / 100,
        f"weighted Weibull cb(sf) {level}%",
        slack=SLACK,
    )


def test_weibull_median_coverage(informative):
    lo, hi = informative["median"]
    check_coverage(
        np.array(lo),
        np.array(hi),
        MEDIAN,
        0.95,
        "weighted Weibull quantile_cb(0.5)",
        slack=SLACK,
    )


def test_weight_scale_changes_nothing(informative):
    assert all(informative["scale_same"])


def test_unweighted_fit_is_biased(informative):
    # The reason for the weights: without them the long-lived units the
    # design oversampled pull the scale up (reported, and checked to be
    # well away from the truth, so the study does test something).
    mean_alpha = float(np.mean(informative["unweighted_alpha"]))
    print(f"unweighted Weibull alpha: mean {mean_alpha:.3f} (truth 10)")
    assert mean_alpha > 11.0


def _logistic_probability(X, a, iters=25):
    """P(a = 1 | X) from a logistic regression of ``a`` on ``X`` (with an
    intercept), fitted by Newton's method."""
    D = np.column_stack([np.ones_like(X), X])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-D @ b))
        H = D.T @ (D * (p * (1 - p))[:, None])
        b = b + np.linalg.solve(H, D.T @ (a - p))
    return 1 / (1 + np.exp(-D @ b))


def test_iptw_coverage():
    rng = np.random.default_rng(26102)
    n, reps = 600, 1000
    t_eval = np.array([4.0, 8.0, 14.0])
    # The truth: the treated survival averaged over the confounder, by
    # Gauss-Hermite quadrature (exact to far below the Monte Carlo error).
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()
    hazard_ratio = np.exp(0.5 + 0.7 * nodes)
    truth = np.array(
        [np.sum(weights * TRUE.sf(t) ** hazard_ratio) for t in t_eval]
    )
    lo, hi = np.empty((reps, 3)), np.empty((reps, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(reps):
            X = rng.normal(size=n)
            a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.8 * X))).astype(
                float
            )
            u = rng.uniform(size=n)
            t = TRUE.qf(1 - u ** (1 / np.exp(0.5 * a + 0.7 * X)))
            cens = rng.uniform(0, C_MAX, n)
            x, c = np.minimum(t, cens), (cens < t).astype(int)
            p = _logistic_probability(X, a)
            treated = a == 1
            model = sp.KaplanMeier.fit(
                x[treated], c[treated], w=1 / p[treated]
            )
            b = model.cb(t_eval)
            lo[r], hi[r] = b[:, 0], b[:, 1]
    hits = (lo <= truth) & (truth <= hi)
    for k in range(3):
        check_rate(
            hits[:, k].sum(),
            reps,
            0.95,
            f"IPTW KaplanMeier cb(sf) at t={t_eval[k]:g} (one-sided)",
            slack=SLACK,
            side="lower",
        )
