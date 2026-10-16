"""Size of the competing-risks proportional-hazards tests of
``diagnose()`` (#874).

- **Cause-specific**: R's ``cox.zph`` score test (``zph_score_test``) on
  each cause's own Cox model of ``CompetingRisksProportionalHazards``,
  with two causes whose hazards are proportional in a binary and a
  normal covariate, 30% censoring, n = 200: the km and rank tests of each
  cause and covariate, and GLOBAL.
- **Fine-Gray**: the weighted score test (``fine_gray_zph_test``) of a
  ``FineGray`` fit to data from a Fine-Gray model (Fine and Gray 1999's
  simulation), n = 200, on the cif and rank scales.

Each test must reject a true null at 5% (slack 0.01). Delayed entry is
not studied: the competing-risks models take no ``tl`` yet (#915).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.diagnostics.competing_risks import fine_gray_zph_test
from surpyval.tests.calibration._montecarlo import check_rate
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    zph_score_test,
)

ALPHA = 0.05
N = 200
REPS = 2000


def _covariates(rng):
    return np.column_stack([rng.binomial(1, 0.5, N), rng.normal(size=N)])


def test_cause_specific_ph_test_size():
    rng = np.random.default_rng(874)
    rejected = {}
    for _ in range(REPS):
        Z = _covariates(rng)
        t_a = rng.exponential(1 / np.exp(Z @ [0.7, 0.3]))
        t_b = 1.5 * rng.weibull(1.5, N) / np.exp(Z @ [-0.4, 0.5]) ** (1 / 1.5)
        cens = rng.uniform(0, 3, N)
        first = np.minimum(t_a, t_b)
        e = np.where(cens < first, None, np.where(t_a < t_b, "a", "b"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.CompetingRisksProportionalHazards.fit(
                np.minimum(first, cens), Z, e.astype(object)
            )
        for cause, cox in model.models.items():
            for transform in ("km", "rank"):
                res = zph_score_test(cox, transform)
                p = [r["p_value"] for r in res["per_covariate"]]
                p.append(res["global"]["p_value"])
                for j, value in enumerate(p):
                    key = (cause, transform, j)
                    rejected[key] = rejected.get(key, 0) + (value <= ALPHA)
    for (cause, transform, j), hits in sorted(rejected.items()):
        name = ["z", "w", "GLOBAL"][j]
        check_rate(
            hits,
            REPS,
            ALPHA,
            "cause-specific {} {} {} size".format(cause, transform, name),
        )


def _fine_gray_data(rng, lp, p=0.6):
    r = np.exp(lp)
    P = 1 - (1 - p) ** r
    is_a = rng.uniform(size=lp.size) < P
    u = rng.uniform(size=lp.size)
    t_a = -np.log(1 - (1 - (1 - u * P) ** (1 / r)) / p)
    t = np.where(is_a, t_a, rng.exponential(size=lp.size))
    return t, np.where(is_a, "a", "b")


@pytest.mark.parametrize("transform", ["cif", "rank"])
def test_fine_gray_ph_test_size(transform):
    rng = np.random.default_rng(875 + (transform == "rank"))
    rejected = np.zeros(3)
    for _ in range(REPS):
        Z = _covariates(rng)
        t, cause = _fine_gray_data(rng, Z @ [0.7, 0.3])
        cens = rng.uniform(0, 4, N)
        e = np.where(cens < t, None, cause).astype(object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.FineGray.fit(np.minimum(t, cens), Z, e, event="a")
        res = fine_gray_zph_test(model, transform)
        p = [r["p_value"] for r in res["per_covariate"]]
        p.append(res["global"]["p_value"])
        rejected += np.asarray(p) <= ALPHA
    for name, hits in zip(["z", "w", "GLOBAL"], rejected):
        check_rate(
            hits, REPS, ALPHA, "Fine-Gray {} {} size".format(transform, name)
        )
