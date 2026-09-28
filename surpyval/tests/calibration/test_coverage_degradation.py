"""Coverage of the degradation bands.

- **Two-stage ``cb``** (``method="analytic"``): 30 units with linear paths
  crossing a threshold of 100 at a Weibull(50, 3) time, measured six times
  out to t = 30 with noise sd 2. The pseudo failure times are then close to
  the true ones, which is the regime in which the pseudo-failure-time
  method estimates the true life distribution at all: with much noisier or
  shorter paths the pseudo times are more spread out than the true ones
  (errors in variables), and no band on the fitted model can cover the
  truth -- that is the method's bias, not its band's, so it is not tested
  here. Checked: the two-sided 95% band and the one-sided lower bound at
  t = 35, 50 and 60 (reliabilities 0.71, 0.37, 0.18). The band that was
  really 90% (each side given the full ``alpha``) covered 0.90 here.
- **``predict_rul`` interval** for a new unit: 40 training units with random
  intercept and slope, a new unit from the same population seen three
  times; the 95% interval for its failure time must contain the true
  crossing time 95% of the time. With the population estimated from 40
  units the prior is nearly the true one, and a Bayesian interval under the
  true prior has exactly its nominal coverage averaged over the population.

Slack 0.01 for the band; 0.015 for ``predict_rul``, whose prior is a
plug-in estimate.
"""

import warnings

import numpy as np

import surpyval as sp
from surpyval.degradation import DegradationAnalysis
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    check_rate,
)


def test_two_stage_band_coverage():
    rng = np.random.default_rng(701)
    alpha, beta, thr = 50.0, 3.0, 100.0
    units, t_meas = 30, np.linspace(5.0, 30.0, 6)
    t_eval = np.array([35.0, 50.0, 60.0])
    truth = sp.Weibull.from_params([alpha, beta]).sf(t_eval)
    reps = 1500
    lo, hi, lo1, lo90, hi90 = (np.empty((reps, 3)) for _ in range(5))
    for r in range(reps):
        T = alpha * rng.weibull(beta, units)
        x = np.tile(t_meas, units)
        i = np.repeat(np.arange(units), t_meas.size)
        y = thr / np.repeat(T, t_meas.size) * x
        y = y + rng.normal(0.0, 2.0, x.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DegradationAnalysis.fit(
                x, y, i, threshold=thr, path="linear", distribution=sp.Weibull
            )
            b = model.cb(t_eval, on="sf")
            lo[r], hi[r] = b[:, 0], b[:, 1]
            lo1[r] = model.cb(t_eval, on="sf", bound="lower")
            b90 = model.cb(t_eval, on="sf", alpha_ci=0.1)
            lo90[r], hi90[r] = b90[:, 0], b90[:, 1]
    check_coverage(lo, hi, truth, 0.95, "degradation cb(sf) two-sided")
    check_coverage(lo1, np.inf, truth, 0.95, "degradation cb(sf) lower")
    # Design check: the old error made the "95%" band the 90% one, which
    # this study tells apart from a 95% band at every time.
    cov90 = ((lo90 <= truth) & (truth <= hi90)).mean(axis=0)
    print("90% band coverage:", cov90.round(4))
    assert np.all(cov90 < 0.95 - 3 * np.sqrt(0.95 * 0.05 / reps) - 0.01)


def test_predict_rul_interval_coverage():
    rng = np.random.default_rng(702)
    thr, noise = 450.0, 3.0
    t_train = np.arange(100.0, 1100.0, 100.0)
    t_new = np.array([100.0, 200.0, 300.0])
    units = 40
    reps = 600
    hits = 0
    for r in range(reps):
        a = rng.normal(10.0, 3.0, units + 1)
        b = rng.normal(0.3, 0.05, units + 1)
        x = np.tile(t_train, units)
        i = np.repeat(np.arange(units), t_train.size)
        y = a[i] + b[i] * x + rng.normal(0.0, noise, x.size)
        y_new = a[-1] + b[-1] * t_new + rng.normal(0.0, noise, t_new.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DegradationAnalysis.fit(x, y, i, threshold=thr)
            pred = model.predict_rul(
                t_new, y_new, n_samples=4000, random_state=r
            )
        lo, hi = pred.failure_time_interval
        hits += lo <= (thr - a[-1]) / b[-1] <= hi
    check_rate(
        hits, reps, 0.95, "predict_rul failure_time_interval", slack=0.015
    )
