"""Distributional regression checked against R.

The data are 80 units (written out below): a Weibull whose scale depends
on ``z1`` and ``z2`` and whose shape depends on ``z1``, with observed,
right, left and interval censored times (``L``, ``R``: ``-inf`` for a
left censored and ``inf`` for a right censored one) and, for some units,
a left truncation time ``TL``.

**gamlss 5.5.5 with gamlss.cens 5.0.7** (R 4.4.3), every censoring at
once, ``mu ~ z1 + z2`` and ``sigma ~ z1``:

.. code-block:: r

    library(gamlss); library(gamlss.cens); library(survival)
    y <- Surv(L, R, type = "interval2")     # NA for -inf / inf
    gen.cens(WEI, type = "interval")
    fit <- gamlss(y ~ z1 + z2, sigma.formula = ~ z1, family = WEIic(),
                  control = gamlss.control(c.crit = 1e-10, n.cyc = 2000),
                  method = mixed(20, 200))
    logLik(fit); coef(fit, "mu"); coef(fit, "sigma")

with the families ``WEI`` (SurPyval's Weibull: ``mu`` the scale, ``sigma``
the shape, both on log links), ``LOGNO`` (LogNormal), ``NO`` (Normal),
``LO`` (Logistic) and ``RG`` (the reverse, largest-extreme Gumbel,
SurPyval's GumbelLEV), each ``mu`` on its own link and ``log(sigma)``.
gamlss's intercepts are on the link scale. (gamlss's ``GU``, the
smallest-extreme Gumbel, failed to converge on these data, so SurPyval's
Gumbel is not checked here.)

**flexsurv 2.3.2**, for left truncation (which gamlss takes only as one
truncation point for every unit), on the observed and right censored
units:

.. code-block:: r

    flexsurvreg(Surv(tl, time, event) ~ z1 + z2, anc = list(shape = ~ z1),
                dist = "weibull")   # and dist = "gamma"

flexsurv's gamma has ``rate`` as its location (SurPyval's Gamma ``beta``)
and ``shape`` (``alpha``) as the ancillary parameter.
"""

import numpy as np
import pytest

import surpyval as sp

INF = np.inf

L = [
    8.0,
    2.39,
    4.32,
    1.37,
    -INF,
    3.02,
    10.2,
    1.15,
    7.2,
    2.0,
    4.0,
    5.2,
    6.25,
    1.59,
    2.0,
    7.96,
    12.97,
    4.72,
    3.95,
    1.43,
    -INF,
    4.0,
    4.0,
    1.45,
    11.76,
    4.11,
    1.26,
    6.54,
    0.39,
    -INF,
    5.27,
    0.09,
    2.0,
    1.86,
    7.16,
    5.71,
    5.07,
    1.68,
    2.24,
    5.26,
    23.59,
    2.0,
    5.45,
    14.88,
    13.38,
    11.08,
    5.3,
    7.36,
    16.42,
    -INF,
    4.15,
    0.39,
    9.41,
    2.23,
    3.86,
    4.56,
    7.79,
    4.0,
    2.07,
    3.87,
    4.81,
    4.0,
    0.94,
    1.38,
    14.0,
    -INF,
    3.37,
    3.73,
    -INF,
    16.1,
    3.73,
    4.31,
    -INF,
    0.79,
    2.45,
    5.64,
    13.26,
    3.7,
    14.0,
    6.83,
]

R = [
    10.0,
    2.39,
    4.32,
    INF,
    2.0,
    3.02,
    INF,
    1.15,
    INF,
    4.0,
    6.0,
    INF,
    6.25,
    INF,
    4.0,
    7.96,
    12.97,
    4.72,
    INF,
    1.43,
    26.41,
    6.0,
    6.0,
    1.45,
    11.76,
    4.11,
    INF,
    INF,
    INF,
    4.53,
    5.27,
    INF,
    4.0,
    INF,
    7.16,
    INF,
    INF,
    INF,
    2.24,
    5.26,
    INF,
    4.0,
    5.45,
    14.88,
    INF,
    11.08,
    5.3,
    7.36,
    INF,
    2.0,
    4.15,
    0.39,
    INF,
    INF,
    3.86,
    4.56,
    INF,
    6.0,
    INF,
    3.87,
    4.81,
    6.0,
    0.94,
    INF,
    16.0,
    2.0,
    3.37,
    3.73,
    5.3,
    INF,
    INF,
    INF,
    3.5,
    0.79,
    INF,
    5.64,
    13.26,
    INF,
    16.0,
    INF,
]

Z1 = [
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
]

Z2 = [
    -2.51,
    0.02,
    0.09,
    -0.68,
    -0.44,
    -0.47,
    -0.58,
    -0.4,
    0.12,
    -0.6,
    0.02,
    0.19,
    -1.02,
    1.44,
    0.45,
    0.43,
    0.32,
    0.65,
    1.56,
    -0.43,
    -0.97,
    1.13,
    1.4,
    -0.8,
    1.68,
    -0.96,
    -0.06,
    -0.15,
    0.64,
    -0.14,
    -0.08,
    2.69,
    -0.86,
    -1.0,
    1.04,
    -1.41,
    0.04,
    2.29,
    -0.42,
    1.13,
    1.56,
    -1.53,
    0.65,
    1.38,
    -0.93,
    -1.11,
    0.81,
    0.94,
    1.27,
    -0.7,
    -0.15,
    -1.28,
    2.27,
    1.24,
    -0.74,
    0.31,
    0.91,
    -0.53,
    0.7,
    -0.96,
    -1.8,
    -0.3,
    -1.38,
    -1.76,
    0.18,
    0.54,
    1.01,
    -2.58,
    -0.87,
    0.76,
    0.68,
    -1.01,
    0.39,
    1.25,
    2.64,
    -0.57,
    0.1,
    0.67,
    0.42,
    0.61,
]

TL = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.91,
    3.06,
    0.0,
    0.0,
    0.0,
    0.0,
    1.56,
    0.0,
    0.0,
    0.0,
    0.0,
    3.89,
    0.0,
    1.18,
    0.43,
    0.0,
    1.2,
    1.2,
    0.44,
    0.0,
    1.23,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.15,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    4.01,
    0.0,
    0.0,
    0.0,
    4.93,
    0.0,
    0.0,
    0.0,
    2.82,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.16,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.01,
    0.0,
    0.0,
    4.83,
    0.0,
    1.29,
    0.0,
    0.24,
    0.0,
    0.0,
    0.0,
    1.11,
    0.0,
    0.0,
]

# gamlss: log-likelihood, mu (intercept on its link, z1, z2), sigma
# (log intercept, z1)
GAMLSS = {
    "Weibull": (
        -131.4126892138,
        [2.3029360574, -0.5040353570, 0.2995645811],
        [0.1113531468, 0.5850304823],
    ),
    "LogNormal": (
        -131.4590035401,
        [1.8741641046, -0.3369194373, 0.3213354293],
        [0.1636872808, -0.7315179107],
    ),
    "Normal": (
        -142.5792681878,
        [8.7120937696, -3.2941510921, 1.7386353026],
        [1.8871406825, -0.7951582886],
    ),
    "Logistic": (
        -142.3746549488,
        [8.2329567718, -3.0170094884, 1.5416873503],
        [1.3806079556, -0.9042201304],
    ),
    "GumbelLEV": (
        -137.5304479015,
        [5.7364794951, -1.5753212094, 1.3281301908],
        [1.6876906569, -0.8594126218],
    ),
}
# flexsurv, Weibull: loglik; res.t est (log shape, log scale, z1, z2,
# shape(z1)); sqrt(diag(cov)) on that scale.
FLEXSURV_WEIBULL = (
    -94.6395804117,
    [0.0141142434, 2.2633886679, -0.4201369997, 0.2618104977, 1.0999224030],
    [0.2106721809, 0.2264416502, 0.2428411137, 0.0657499466, 0.3082196974],
)
# flexsurv, gamma: loglik; est (log shape, log rate, rate z1, rate z2,
# shape(z1)).
FLEXSURV_GAMMA = (
    -95.7910465489,
    [0.0600108342, -2.1933473961, 2.0127141502, -0.2902803217, 1.5185622671],
)


def _censored():
    lo, hi = np.asarray(L), np.asarray(R)
    c = np.where(
        np.isinf(hi), 1, np.where(np.isinf(lo), -1, np.where(lo == hi, 0, 2))
    )
    x = np.c_[np.where(np.isinf(lo), hi, lo), np.where(np.isinf(hi), lo, hi)]
    return x, c, np.c_[Z1, Z2].astype(float)


@pytest.mark.parametrize("dist", sorted(GAMLSS))
def test_every_censoring_matches_gamlss(dist):
    x, c, Z = _censored()
    fitter = sp.Distributional(getattr(sp, dist))
    first, second = fitter.parameter_names
    model = fitter.fit(x, Z, c=c, covariates={first: [0, 1], second: [0]})
    assert model.maximum == "verified"
    ll, mu, sigma = GAMLSS[dist]
    np.testing.assert_allclose(-model.neg_ll(), ll, atol=1e-8)
    p = np.asarray(model.params, dtype=float)
    intercept = np.log(p[0]) if model.links[0] == "log" else p[0]
    np.testing.assert_allclose([intercept, p[2], p[3]], mu, rtol=1e-4)
    np.testing.assert_allclose([np.log(p[1]), p[4]], sigma, rtol=1e-4)


def _truncated():
    lo, hi = np.asarray(L), np.asarray(R)
    keep = np.isinf(hi) | (lo == hi)
    time = np.where(np.isinf(hi), lo, hi)[keep]
    c = np.isinf(hi)[keep].astype(int)
    Z = np.c_[Z1, Z2].astype(float)[keep]
    return time, c, Z, np.asarray(TL)[keep]


def test_left_truncation_matches_flexsurv_weibull():
    x, c, Z, tl = _truncated()
    assert (tl > 0).sum() > 10
    model = sp.WeibullDistributional.fit(
        x, Z, c=c, tl=tl, covariates={"alpha": [0, 1], "beta": [0]}
    )
    assert model.maximum == "verified"
    ll, est, se = FLEXSURV_WEIBULL
    np.testing.assert_allclose(-model.neg_ll(), ll, atol=1e-8)
    p = np.asarray(model.params, dtype=float)
    ours = [np.log(p[1]), np.log(p[0]), p[2], p[3], p[4]]
    np.testing.assert_allclose(ours, est, rtol=1e-4)
    s = model.standard_errors()
    ours_se = [s[1] / p[1], s[0] / p[0], s[2], s[3], s[4]]
    np.testing.assert_allclose(ours_se, se, rtol=1e-3)


def test_left_truncation_matches_flexsurv_gamma():
    x, c, Z, tl = _truncated()
    model = sp.GammaDistributional.fit(
        x, Z, c=c, tl=tl, covariates={"alpha": [0], "beta": [0, 1]}
    )
    assert model.maximum == "verified"
    ll, est = FLEXSURV_GAMMA
    np.testing.assert_allclose(-model.neg_ll(), ll, atol=1e-8)
    p = np.asarray(model.params, dtype=float)
    # alpha (shape), beta (rate), alpha_z1, beta_z1, beta_z2
    ours = [np.log(p[0]), np.log(p[1]), p[3], p[4], p[2]]
    np.testing.assert_allclose(ours, est, rtol=1e-4)
