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

**The sign.** gamlss's and flexsurv's coefficients take the link's sign:
a positive coefficient raises its parameter, ``g(theta(z)) = g(theta) +
b'z``, so on a scale or a location it means a *longer* life. SurPyval's
take its own sign (that of its AFT models), ``g(theta(z)) = g(theta) +
s b'z`` with ``s = -1`` for a parameter whose increase lengthens life (a
scale, a location, a Gamma's shape) and ``s = +1`` otherwise
(``model.coefficient_signs``). Each of our coefficients is multiplied by
its parameter's ``s`` before it is compared with R's.
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
    s1, s2 = model.coefficient_signs.values()
    intercept = np.log(p[0]) if model.links[0] == "log" else p[0]
    # R's sign: each of our coefficients times its parameter's sign
    np.testing.assert_allclose(
        [intercept, s1 * p[2], s1 * p[3]], mu, rtol=1e-4
    )
    np.testing.assert_allclose([np.log(p[1]), s2 * p[4]], sigma, rtol=1e-4)


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
    # R's sign: the scale's coefficients (s = -1) negated
    assert model.coefficient_signs == {"alpha": -1.0, "beta": 1.0}
    ours = [np.log(p[1]), np.log(p[0]), -p[2], -p[3], p[4]]
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
    # alpha (shape), beta (rate), alpha_z1, beta_z1, beta_z2; R's sign:
    # the shape's coefficient (s = -1, a larger Gamma shape is a longer
    # life) negated, the rate's (s = +1) as it is
    assert model.coefficient_signs == {"alpha": -1.0, "beta": 1.0}
    ours = [np.log(p[0]), np.log(p[1]), p[3], p[4], -p[2]]
    np.testing.assert_allclose(ours, est, rtol=1e-4)


# -- the likelihood-ratio tests --------------------------------------------
#
# The fitted model's log-likelihood is gamlss's and flexsurv's (above). The
# null models are R's too: ``gamlss(y ~ 1, sigma.formula = ~ 1)`` (every
# coefficient at 0), ``gamlss(y ~ z1 + z2, sigma.formula = ~ 1)`` (the
# shape's coefficient at 0) and the same two fits in flexsurv without
# ``anc``. Printed by scripts/reference/reference_r_distributional.R
# (R 4.6.2, gamlss 5.5.5, gamlss.cens 5.0.7, flexsurv 2.3.2).

GAMLSS_WEIBULL_NULL = -142.7428355194
GAMLSS_WEIBULL_BETA_NULL = -134.7672711689
FLEXSURV_WEIBULL_NULL = -105.6082665438
FLEXSURV_WEIBULL_BETA_NULL = -100.3461226943


def test_lr_tests_every_censoring_match_gamlss_nulls():
    x, c, Z = _censored()
    model = sp.WeibullDistributional.fit(
        x, Z, c=c, covariates={"alpha": [0, 1], "beta": [0]}
    )
    global_test = model.likelihood_ratio_test()
    np.testing.assert_allclose(
        global_test["log_likelihood"], GAMLSS["Weibull"][0], atol=1e-8
    )
    np.testing.assert_allclose(
        global_test["log_likelihood_null"], GAMLSS_WEIBULL_NULL, atol=1e-6
    )
    assert global_test["df"] == 3
    shape_test = model.likelihood_ratio_test(parameter="beta")
    np.testing.assert_allclose(
        shape_test["log_likelihood_null"], GAMLSS_WEIBULL_BETA_NULL, atol=1e-6
    )
    assert shape_test["df"] == 1


def test_lr_tests_left_truncation_match_flexsurv_nulls():
    x, c, Z, tl = _truncated()
    model = sp.WeibullDistributional.fit(
        x, Z, c=c, tl=tl, covariates={"alpha": [0, 1], "beta": [0]}
    )
    global_test = model.likelihood_ratio_test()
    np.testing.assert_allclose(
        global_test["log_likelihood"], FLEXSURV_WEIBULL[0], atol=1e-8
    )
    np.testing.assert_allclose(
        global_test["log_likelihood_null"], FLEXSURV_WEIBULL_NULL, atol=1e-6
    )
    shape_test = model.likelihood_ratio_test(parameter="beta")
    np.testing.assert_allclose(
        shape_test["log_likelihood_null"],
        FLEXSURV_WEIBULL_BETA_NULL,
        atol=1e-6,
    )
    assert (global_test["df"], shape_test["df"]) == (3, 1)


# -- a published case: the PET film accelerated life test ------------------
#
# Mazucheli, Louzada-Neto and Achcar (2003), "Lifetime models with
# nonconstant shape parameters", REVSTAT 1(1), 25-39, section 4.1: the
# breakdown times (hours) of PET film in SF6 gas insulated transformers at
# 5, 7, 10 and 15 kV (Hirose 1993), 44 units, three censored at 5 kV,
# with the covariate x = log(v - 4.76). Model B puts a log-linear model on
# both the Weibull scale and shape; model A on the scale only.
#
# The data are R's ``boot::hirose`` (boot 1.3.32). The paper's Table 1
# prints 9 of the 10 times at 10 kV (23.90 is missing) though its text
# counts 10; its estimates are those of the 44 units. Its text writes the
# covariate as -log(v - 4.76), but its estimates are on +log(v - 4.76)
# (alpha1 < 0: life falls as the voltage rises), the sign used here. The
# paper's alpha0 and beta0 are the logs of SurPyval's alpha and beta, so
# their standard errors are SurPyval's divided by alpha and beta (the
# delta method).
#
# The same fits in flexsurv 2.3.2 (R 4.6.2; printed by
# scripts/reference/reference_r_distributional.R) give the unrounded
# values. They agree with every printed digit but one: the Wald statistic
# for beta1 = 0 is 16.18952 in flexsurv (and SurPyval), which rounds to
# 16.1895, where the paper prints 16.1896 (its Hessian came from SAS's NLP
# procedure; the log-likelihoods, estimates and standard errors agree to
# all four printed places).

PET_VOLTS = [5] * 10 + [7] * 15 + [10] * 10 + [15] * 9
PET_TIMES = [
    *[7131, 8482, 8559, 8762, 9026, 9034, 9104, 9104.25, 9104.25, 9104.25],
    *[50.25, 87.75, 87.76, 87.77, 92.90, 92.91, 95.96, 108.3, 108.3],
    *[117.9, 123.9, 124.3, 129.7, 135.6, 135.6],
    *[15.17, 19.87, 20.18, 21.50, 21.88, 22.23, 23.02, 23.90, 28.17, 29.70],
    *[2.40, 2.42, 3.17, 3.75, 4.65, 4.95, 6.23, 6.68, 7.30],
]
PET_CENSORED = [0] * 7 + [1] * 3 + [0] * 34
# Table 3: (alpha0, alpha1, beta0, beta1) and their standard deviations
PET_MODEL_B = (
    [6.3285, -1.9529, 2.2311, -0.4636],
    [0.0213, 0.0156, 0.1776, 0.1152],
)
PET_MODEL_A = ([6.3480, -1.9629, 1.6080], [0.0399, 0.0265, 0.1281])
PET_LOGLIK = {"A": -179.9849, "B": -173.2728}
PET_LR, PET_WALD = 13.4240, 16.1896
# flexsurv: log-likelihoods; model B's (beta0, alpha0, alpha1, beta1) and
# standard errors on that scale; the Wald statistic.
FLEXSURV_PET_LL = {"A": -179.9848607049, "B": -173.2728398405}
FLEXSURV_PET_B = (
    [2.2311269868, 6.3285364232, -1.9529264080, -0.4636296810],
    [0.1776084164, 0.0212945355, 0.0155804390, 0.1152270039],
)
FLEXSURV_PET_WALD = 16.1895181501


def _pet(beta_covariates):
    x = np.log(np.asarray(PET_VOLTS, dtype=float) - 4.76)
    return sp.WeibullDistributional.fit(
        np.asarray(PET_TIMES),
        x,
        c=PET_CENSORED,
        covariates={"alpha": [0], "beta": beta_covariates},
    )


def _printed(value, digits=4):
    """``value`` rounded as the paper prints it."""
    return np.round(np.asarray(value, dtype=float), digits)


def test_pet_film_matches_the_published_fits():
    assert len(PET_TIMES) == 44 and sum(PET_CENSORED) == 3
    b = _pet([0])
    assert b.maximum == "verified"
    assert b.parameter_names == [
        "alpha",
        "beta",
        "alpha_coef_0",
        "beta_coef_0",
    ]
    p, se = np.asarray(b.params, dtype=float), b.standard_errors()
    s = b.coefficient_signs["alpha"]  # the paper's sign is the link's
    ours = [np.log(p[0]), s * p[2], np.log(p[1]), p[3]]
    ours_se = [se[0] / p[0], se[2], se[1] / p[1], se[3]]
    np.testing.assert_array_equal(_printed(ours), PET_MODEL_B[0])
    np.testing.assert_array_equal(_printed(ours_se), PET_MODEL_B[1])
    np.testing.assert_array_equal(_printed(-b.neg_ll()), PET_LOGLIK["B"])

    a = _pet([])
    assert a.maximum == "verified"
    p, se = np.asarray(a.params, dtype=float), a.standard_errors()
    s = a.coefficient_signs["alpha"]  # the paper's sign is the link's
    np.testing.assert_array_equal(
        _printed([np.log(p[0]), s * p[2], np.log(p[1])]), PET_MODEL_A[0]
    )
    np.testing.assert_array_equal(
        _printed([se[0] / p[0], se[2], se[1] / p[1]]), PET_MODEL_A[1]
    )
    np.testing.assert_array_equal(_printed(-a.neg_ll()), PET_LOGLIK["A"])


def test_pet_film_shape_tests_match_the_paper():
    b = _pet([0])
    test = b.likelihood_ratio_test(parameter="beta")
    assert test["df"] == 1
    np.testing.assert_array_equal(
        _printed(test["log_likelihood_null"]), PET_LOGLIK["A"]
    )
    np.testing.assert_array_equal(_printed(test["statistic"]), PET_LR)
    wald = (b.params[3] / b.standard_errors()[3]) ** 2
    # One unit in the paper's last place (see above): flexsurv's value
    np.testing.assert_allclose(wald, PET_WALD, atol=1.5e-4)
    np.testing.assert_allclose(wald, FLEXSURV_PET_WALD, rtol=1e-5)


def test_pet_film_matches_flexsurv():
    b, a = _pet([0]), _pet([])
    np.testing.assert_allclose(-b.neg_ll(), FLEXSURV_PET_LL["B"], atol=1e-8)
    np.testing.assert_allclose(-a.neg_ll(), FLEXSURV_PET_LL["A"], atol=1e-8)
    p, se = np.asarray(b.params, dtype=float), b.standard_errors()
    s = b.coefficient_signs["alpha"]  # flexsurv's sign is the link's
    ours = [np.log(p[1]), np.log(p[0]), s * p[2], p[3]]
    ours_se = [se[1] / p[1], se[0] / p[0], se[2], se[3]]
    np.testing.assert_allclose(ours, FLEXSURV_PET_B[0], rtol=1e-6)
    np.testing.assert_allclose(ours_se, FLEXSURV_PET_B[1], rtol=1e-5)
