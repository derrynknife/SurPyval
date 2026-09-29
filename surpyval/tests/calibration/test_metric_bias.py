"""Bias of the Brier score and time-dependent AUC with tied times (#365).

Event and censoring times are both discrete (whole units), so events,
censorings and the scoring horizons tie often -- the setting in which the
censoring weights were wrong before #365 (Brier biased by -0.029 in a
simulation like this one). Covariate ``z`` takes three levels, so the risk
scores tie too.

The predictor is the true model, whose Brier score and AUC have closed
forms: with ``S_z`` the true survival of level ``z`` at the horizon and
``p_z`` its probability,

- ``BS(t) = sum_z p_z S_z (1 - S_z)`` (for the true predictor the squared
  error given ``z`` is the Bernoulli variance), and
- ``AUC(t) = sum_{z, w} p_z F_z p_w S_w [I(r_z > r_w) + I(r_z = r_w) / 2]
  / (sum_z p_z F_z)(sum_w p_w S_w)`` with risk ``r_z = F_z = 1 - S_z``.

The mean estimate over the replicates must match within 3 Monte Carlo
standard errors plus a slack of 0.002 (Brier) and 0.003 (AUC) for the
O(1/n) bias of a ratio of estimated weights at n = 300.

Run against the code before #365, this study fails: its Brier scores were
off by +0.004, -0.005 and -0.011 at the three horizons (tolerance about
0.0025). Its AUC was off by -0.0015 to -0.0025, which is inside the AUC
tolerance: the design resolves the Brier error, not that one.
"""

import math

import numpy as np

from surpyval.metrics import auc_td, brier_score
from surpyval.tests.calibration._montecarlo import Z_TOL

LEVELS = np.array([0.0, 1.0, 2.0])
RATE = 0.15 * np.exp(0.5 * LEVELS)  # event hazard per level
CENSOR_RATE = 0.08
HORIZONS = np.array([2.0, 4.0, 6.0])
N = 300


def _true_values():
    S = np.exp(-np.outer(RATE, HORIZONS))  # (level, horizon)
    F = 1 - S
    p = np.full(LEVELS.size, 1 / LEVELS.size)
    bs = (p[:, None] * S * F).sum(axis=0)
    auc = np.empty(HORIZONS.size)
    for k in range(HORIZONS.size):
        r = F[:, k]
        num = 0.0
        for a in range(LEVELS.size):
            for b in range(LEVELS.size):
                win = float(r[a] > r[b]) + 0.5 * float(r[a] == r[b])
                num += p[a] * F[a, k] * p[b] * S[b, k] * win
        auc[k] = num / ((p * F[:, k]).sum() * (p * S[:, k]).sum())
    return bs, auc


def _check_mean(estimates, truth, label, slack):
    est = np.asarray(estimates)
    mean = est.mean(axis=0)
    se = est.std(axis=0, ddof=1) / math.sqrt(est.shape[0])
    bad = []
    for k, t in enumerate(HORIZONS):
        tol = Z_TOL * se[k] + slack
        line = (
            "{} at t={:g}: mean {:.4f} (true {:.4f}, tolerance +/- "
            "{:.4f})".format(label, t, mean[k], truth[k], tol)
        )
        print(line)
        if abs(mean[k] - truth[k]) > tol:
            bad.append(line)
    assert not bad, "\n".join(bad)


def test_brier_and_auc_unbiased_with_ties():
    rng = np.random.default_rng(601)
    bs_true, auc_true = _true_values()
    reps = 5000
    bs = np.empty((reps, HORIZONS.size))
    auc = np.empty((reps, HORIZONS.size))
    ties = 0.0
    for r in range(reps):
        z = rng.integers(0, LEVELS.size, N)
        t = np.ceil(rng.exponential(1 / RATE[z]))
        cens = np.ceil(rng.exponential(1 / CENSOR_RATE, N))
        # An event and a censoring at the same whole time: the event is
        # seen (it happened first within the unit), as the metrics assume.
        x = np.minimum(t, cens)
        c = (cens < t).astype(int)
        ties += np.isin(x[c == 0], x[c == 1]).mean() / reps
        # The true model's prediction: P(T > t) = exp(-rate t) at a whole t.
        surv = np.exp(-np.outer(RATE[z], HORIZONS))
        bs[r] = brier_score(x, c, surv, HORIZONS)[1]
        auc[r] = auc_td(x, c, 1 - surv, HORIZONS)[1]
    print("events tied with a censoring: {:.0%}".format(ties))
    _check_mean(bs, bs_true, "Brier score", 0.002)
    _check_mean(auc, auc_true, "AUC", 0.003)
