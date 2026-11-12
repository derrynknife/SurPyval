"""The IDI and NRI of a marker that carries no information
(``surpyval.metrics.idi_nri``).

The truth: a covariate ``z1 ~ N(0, 1)`` with hazard ``0.1 exp(0.7 z1)``
(exponential lives), censoring exponential with rate 0.05, the horizon
``t = 5``; the added marker ``z2 ~ N(0, 1)`` is independent of
everything. ``N = 300`` units per data set; 95% perturbation bounds from
400 resamples. The seed, replicates, sizes and slack (0.01, the suite's
rule) were fixed before any run.

1. **Fixed models, scored on new data.** The old risk is the true one,
   ``1 - exp(-0.5 exp(0.7 z1))``; the new one adds the noise with a fixed
   coefficient, ``1 - exp(-0.5 exp(0.7 z1 + 0.5 z2))``: a model made
   worse by a useless marker. The true IDI, continuous NRI and
   categorical NRI (cut-offs 0.2 and 0.4) are those of 2,000,000
   uncensored units. Checked over 1000 replicates: the bounds cover these
   truths at 95%. Reported, not checked: the truths themselves, and how
   often the bounds exclude 0. The continuous NRI's truth is exactly 0
   (``z2`` moves a risk up or down with probability one half whatever the
   outcome); the IDI's and the categorical NRI's need not be.

2. **Models fitted to the same data** (the common use): Cox models of
   ``z1`` and of ``z1, z2`` fitted to each data set and scored on it.
   The bounds hold the predicted risks fixed, as ``idi_nri`` documents,
   so they leave out the variation of the fits. Reported only, over 400
   replicates: how often each bound covers 0, the mean estimates, and the
   rejection rate of the likelihood-ratio test of ``z2`` for comparison.
"""

import numpy as np
from scipy.stats import chi2

import surpyval as sp
from surpyval.metrics import idi_nri, survival_probability
from surpyval.tests.calibration._montecarlo import check_coverage

SEED = 20261113
N = 300
T = 5.0
H0 = 0.1 * T
B1, B2 = 0.7, 0.5
CENSOR_RATE = 0.05
CUTS = [0.2, 0.4]
RESAMPLES = 400
MEASURES = ("idi", "nri", "nri_categorical")


def _risk(lp):
    return -np.expm1(-H0 * np.exp(lp))


def _units(rng, n):
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    life = rng.exponential(1 / (0.1 * np.exp(B1 * z1)))
    return z1, z2, life


def _true_values():
    rng = np.random.default_rng([SEED, 0])
    z1, z2, life = _units(rng, 2_000_000)
    old, new = _risk(B1 * z1), _risk(B1 * z1 + B2 * z2)
    res = idi_nri(
        life,
        np.zeros(life.size, int),
        old,
        new,
        T,
        cutoffs=CUTS,
        bounds="none",
    )
    est = res.estimates.set_index("measure")["estimate"]
    return np.array([est[m] for m in MEASURES])


def _sample(rng):
    z1, z2, life = _units(rng, N)
    cens = rng.exponential(1 / CENSOR_RATE, N)
    return z1, z2, np.minimum(life, cens), (cens < life).astype(int)


def _bounds(res):
    e = res.estimates.set_index("measure")
    return e.loc[list(MEASURES), "lower"], e.loc[list(MEASURES), "upper"]


def test_noise_marker_fixed_models_cover_the_truth():
    truth = _true_values()
    print("truth (IDI, NRI, categorical NRI):", np.round(truth, 5))
    reps = 1000
    lower = np.empty((reps, 3))
    upper = np.empty((reps, 3))
    est = np.empty((reps, 3))
    for r in range(reps):
        rng = np.random.default_rng([SEED, 1, r])
        z1, z2, x, c = _sample(rng)
        res = idi_nri(
            x,
            c,
            _risk(B1 * z1),
            _risk(B1 * z1 + B2 * z2),
            T,
            cutoffs=CUTS,
            n_resamples=RESAMPLES,
            random_state=[SEED, 2, r],
        )
        lower[r], upper[r] = _bounds(res)
        est[r] = [res.idi, res.nri, res.nri_categorical]
    print("mean estimates:", np.round(est.mean(axis=0), 5))
    excl = ((lower > 0) | (upper < 0)).mean(axis=0)
    print("share of bounds excluding 0:", np.round(excl, 4))
    print(
        "share excluding 0 above it:",
        np.round((lower > 0).mean(axis=0), 4),
    )
    check_coverage(lower, upper, truth, 0.95, "fixed models, noise marker")


def test_noise_marker_fitted_to_the_same_data_report():
    reps = 400
    lower = np.empty((reps, 3))
    upper = np.empty((reps, 3))
    est = np.empty((reps, 3))
    lr_reject = 0
    for r in range(reps):
        rng = np.random.default_rng([SEED, 3, r])
        z1, z2, x, c = _sample(rng)
        Z_old = z1[:, None]
        Z_new = np.column_stack([z1, z2])
        m_old = sp.CoxPH.fit(x=x, Z=Z_old, c=c)
        m_new = sp.CoxPH.fit(x=x, Z=Z_new, c=c)
        lr = 2 * (m_new.log_likelihood - m_old.log_likelihood)
        lr_reject += int(chi2.sf(lr, 1) < 0.05)
        old = 1 - survival_probability(m_old, Z_old, [T])[:, 0]
        new = 1 - survival_probability(m_new, Z_new, [T])[:, 0]
        res = idi_nri(
            x,
            c,
            old,
            new,
            T,
            cutoffs=CUTS,
            n_resamples=RESAMPLES,
            random_state=[SEED, 4, r],
        )
        lower[r], upper[r] = _bounds(res)
        est[r] = [res.idi, res.nri, res.nri_categorical]
    cover = ((lower <= 0) & (upper >= 0)).mean(axis=0)
    print("same data: coverage of 0 (IDI, NRI, categorical NRI):", cover)
    print("same data: share above 0:", (lower > 0).mean(axis=0))
    print("same data: mean estimates:", np.round(est.mean(axis=0), 5))
    print("same data: LR test of z2 rejects:", lr_reject / reps)
    assert np.isfinite(est).all()
