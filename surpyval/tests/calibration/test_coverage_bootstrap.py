"""Coverage of the parametric bootstrap bounds of plain models
(``method="bootstrap"`` on ``param_cb``, ``cb`` and ``quantile_cb``).

The limited-failure and zero-inflated studies
(``test_coverage_extra_parameters``) found their bootstrap intervals a
little narrow at 100 units. This study measures the same intervals on
the plain models, with no extra parameter, so that a shortfall can be
traced to the bootstrap itself or to the extra parameter.

Data: n = 30, 100 and 300 units from a Weibull (scale 10, shape 2) or a
LogNormal (``mu`` 2, ``sigma`` 0.5), right censored by an independent
uniform time on (0, 30) or (0, 25) (about 30% censored), as in
``test_coverage_parametric``. Each replicate is fitted by maximum
likelihood and bounded with ``n_boot = 200`` (the default) and
``random_state = 1000 * seed + r``; the refits are drawn once and shared
by every bound of the replicate. At 90% and 95% the two-sided bounds on
every parameter, on ``sf`` at the true 10%, 50% and 90% quantiles, and
on those three quantiles (``quantile_cb``) are checked against the
truth. A plain model's bootstrap interval is the bias-corrected
percentile interval of the refits (see
``surpyval/univariate/parametric/_bootstrap.py``).

200 replicates a design (seeds 331-336, fixed with the replicate counts
before the first run; see ``DECISIONS.md``), slack 0.01: the tolerance is
+/- 0.074 at 90% and +/- 0.056 at 95%. At 200 replicates a cell's Monte
Carlo standard error is about two points, so a shortfall of a point or
two shows in the pooled mean over the cells (printed at the end of the
run by ``surpyval/tests/conftest.py``) rather than in any one cell.

A replicate takes 5 to 12 seconds on one core (200 refits), so each
design runs its replicates in turn and the designs run side by side
(pytest-xdist); in CI each family is a group of its own. On the first run
(12 November 2026, an M-series Mac at two workers, shared with other
work) each design took 2,190 to 2,430 s.

What the first run found, and the fix: the intervals were about a point
short of nominal at every sample size, not only the smallest (85 of the
96 cells below nominal), because each bound was read from the refits
with the ``(n - 1) p`` rule, which puts every tail level nearer the
middle (at 200 refits a 95% interval spanned about 3.0-97.0% of the
refits' distribution). The bounds now use the ``(n + 1) p``-th refit
(``bca_bounds``). The same seeds give the same refits, so only the
bounds moved. The mean coverage of each design's eight cells at 90% /
95%, before and after:

=========  =====  ==============  ==============
Family     Units  before          after
=========  =====  ==============  ==============
Weibull    30     0.876 / 0.923   0.884 / 0.937
Weibull    100    0.876 / 0.935   0.886 / 0.946
Weibull    300    0.876 / 0.940   0.888 / 0.949
LogNormal  30     0.859 / 0.913   0.877 / 0.924
LogNormal  100    0.883 / 0.934   0.896 / 0.949*
LogNormal  300    0.879 / 0.936   0.889 / 0.944
=========  =====  ==============  ==============

(After: the designs re-run replicate by replicate with the same seeds,
12 November 2026; * LogNormal at 100 units over its first 160
replicates only, every cell inside its tolerance.) At 200 replicates
a design's mean is uncertain by about a point. Weibull at 30 units
now passes every cell; LogNormal at 30 units still has one outside its
tolerance (``SHORTFALL``). The
likelihood-ratio bounds of the same families hold their level
(``test_coverage_parametric``).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage

REPS = 200
N_BOOT = 200
LEVELS = (0.10, 0.05)
PROBS = np.array([0.1, 0.5, 0.9])

# (distribution, true parameters, censoring upper limit)
FAMILIES = {
    "Weibull": (sp.Weibull, [10.0, 2.0], 30.0),
    "LogNormal": (sp.LogNormal, [2.0, 0.5], 25.0),
}
# (family, n): seed
SEEDS = {
    ("Weibull", 30): 331,
    ("Weibull", 100): 332,
    ("Weibull", 300): 333,
    ("LogNormal", 30): 334,
    ("LogNormal", 100): 335,
    ("LogNormal", 300): 336,
}


def _sample(family, n, rng):
    dist, params, c_max = FAMILIES[family]
    true = dist.from_params(params)
    t = true.qf(rng.uniform(size=n))
    cens = rng.uniform(0, c_max, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _study(family, n):
    dist, params, _ = FAMILIES[family]
    seed = SEEDS[(family, n)]
    rng = np.random.default_rng(seed)
    true = dist.from_params(params)
    q_true = true.qf(PROBS)
    k = len(params)
    shape = (len(LEVELS), REPS)
    p_cb = np.empty(shape + (k, 2))
    s_cb = np.empty(shape + (len(PROBS), 2))
    q_cb = np.empty(shape + (len(PROBS), 2))
    censored = 0.0
    for r in range(REPS):
        x, c = _sample(family, n, rng)
        censored += c.mean() / REPS
        boot = {
            "method": "bootstrap",
            "n_boot": N_BOOT,
            "random_state": 1000 * seed + r,
        }
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            model = dist.fit(x, c=c)
            for i, a in enumerate(LEVELS):
                p_cb[i, r] = [
                    model.param_cb(name, alpha_ci=a, **boot)
                    for name in dist.parameter_names
                ]
                s_cb[i, r] = model.cb(q_true, on="sf", alpha_ci=a, **boot)
                q_cb[i, r] = model.quantile_cb(PROBS, alpha_ci=a, **boot)
    print(f"{family} n={n}: {censored:.0%} censored")
    return p_cb, s_cb, q_cb


# The designs that put a cell outside its tolerance on the run after the
# bootstrap's percentiles were corrected (12 November 2026, the seeds
# above): known shortfalls, not to be met by a change of seed, replicate
# count or tolerance. Strict, so a fix shows as a pass. (Weibull at 30
# units was one on the first run, at 0.890 on the 95% quantile_cb of the
# 90% quantile; with the correction that cell covers 0.915 and every
# cell passes.)
SHORTFALL = {
    ("LogNormal", 30): (
        "known shortfall at 30 units: the 95% quantile_cb on the 10% "
        "quantile covered 0.890 (tolerance 0.894 to 1); the design's 16 "
        "cells averaged 0.877 at 90% and 0.924 at 95%, 15 of them below "
        "nominal"
    ),
}


@pytest.mark.parametrize(
    "family, n",
    [
        pytest.param(
            family,
            n,
            marks=(
                [pytest.mark.xfail(strict=True, reason=SHORTFALL[(family, n)])]
                if (family, n) in SHORTFALL
                else []
            ),
        )
        for family, n in SEEDS
    ],
)
def test_bootstrap_coverage(family, n):
    dist, params, _ = FAMILIES[family]
    true = dist.from_params(params)
    q_true = true.qf(PROBS)
    p_cb, s_cb, q_cb = _study(family, n)
    # Every cell is checked (and printed) before any failure is raised,
    # so a run's log holds the whole table.
    failures = []
    for i, a in enumerate(LEVELS):
        label = f"{family} n={n} bootstrap {1 - a:.0%}"
        for bounds, truth, what in [
            (p_cb[i], np.asarray(params), "param_cb"),
            (s_cb[i], true.sf(q_true), "cb(sf)"),
            (q_cb[i], q_true, "quantile_cb"),
        ]:
            try:
                check_coverage(
                    bounds[..., 0],
                    bounds[..., 1],
                    truth,
                    1 - a,
                    label + " " + what,
                )
            except AssertionError as error:
                failures.append(str(error))
    assert not failures, "\n".join(failures)
