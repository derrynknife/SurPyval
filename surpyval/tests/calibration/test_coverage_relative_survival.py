"""Coverage of the relative survival bounds on simulated cohorts.

The truth: 500 units a replicate, aged uniformly from 50 to 85, each with
an excess hazard and a population hazard from a rate table whose rates
rise 9% a year of age (5e-5 exp(0.09 a), in annual bands); each unit's
time is the first of its excess and its population lifetimes, censored
at a fixed horizon. Checked at 90% (alpha_ci=0.1), two-sided (3 Monte
Carlo standard errors + 0.02):

- Pohar Perme's net survival at 5 years (log bounds), with an excess
  hazard of 0.1 exp(0.05 (age - 65)) a year (rising with age, where
  Ederer II is biased), followed for 10 years; the truth is the mean of
  each unit's own net survival at 5;
- the excess hazard model's coefficient (Wald), with an excess hazard of
  0.15 exp(-0.7 z) a year, z a fair coin, followed for 8 years.

300 replicates; the seed, sizes and tolerance were fixed before the
recorded run.

Result on 19 Nov 2026: Pohar Perme 0.923, the coefficient 0.907.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_rate

SEED = 20261119
SIZE = 500
REPS = 300
NOMINAL = 0.9


def _table():
    ages = np.arange(0, 121)
    return sp.RateTable(5e-5 * np.exp(0.09 * ages), ages=ages)


def _population(table, age, u):
    """Population lifetimes: the expected survival of a table by age
    alone inverted at u, through the cumulative hazard by attained age."""
    edges = np.r_[table.ages, 1e3]
    C = np.r_[0.0, np.cumsum(table.rates[0, :, 0] * np.diff(edges))]
    return np.interp(np.interp(age, edges, C) - np.log(u), C, edges) - age


@pytest.mark.calibration
def test_relative_survival_coverage():
    table = _table()
    rng = np.random.default_rng(SEED)
    net, coef = [], []
    for _ in range(REPS):
        age = rng.uniform(50, 85, SIZE)
        lam = 0.1 * np.exp(0.05 * (age - 65))
        z = rng.binomial(1, 0.5, SIZE)
        t = np.minimum(
            np.minimum(
                rng.exponential(1 / lam),
                _population(table, age, rng.uniform(size=SIZE)),
            ),
            10.0,
        )
        truth = np.mean(np.exp(-lam * 5.0))
        model = sp.RelativeSurvival.fit(
            t, t == 10.0, rate_table=table, age=age
        )
        lo, hi = model.cb(5.0, alpha_ci=0.1)
        net.append(lo < truth < hi)
        t2 = np.minimum(
            np.minimum(
                rng.exponential(1 / (0.15 * np.exp(-0.7 * z))),
                _population(table, age, rng.uniform(size=SIZE)),
            ),
            8.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.ExcessHazard.fit(
                t2, z, c=t2 == 8.0, rate_table=table, age=age
            )
        lo, hi = fit.param_cb("coef_0", alpha_ci=0.1)
        coef.append(lo < -0.7 < hi)
    check_rate(sum(net), REPS, NOMINAL, "Pohar Perme sf(5)", slack=0.02)
    check_rate(sum(coef), REPS, NOMINAL, "excess hazard coef", slack=0.02)
