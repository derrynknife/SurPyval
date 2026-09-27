"""SurPyval's parametric fits against stored results (#379).

* R ``survival::survreg``: Weibull, log-normal, log-logistic and
  exponential AFT regression and intercept-only fits, on right-censored
  data (ovarian, lung) and on interval, left and right censored data;
* R ``fitdistrplus::fitdistcens``: gamma, normal and logistic (and a
  Weibull / log-normal cross-check) on censored data;
* lifelines: the same families with left truncation, which neither R
  package fits, and on the interval data.

survreg writes ``log T = X b + scale W``. For SurPyval's Weibull and
log-logistic that is ``alpha = exp(b0)``, ``beta = 1 / scale``; for the
log-normal ``mu = b0``, ``sigma = scale``; for the exponential the rate is
``exp(-b0)``. SurPyval's AFT multiplies time by ``exp(beta' Z)``, so its
coefficients are ``-b``.

Tolerances. Every program here maximises the same likelihood numerically,
so what must agree is the maximum: the log-likelihoods to ``1e-5`` (they
agree to ~1e-7 in practice). The parameters then agree to the optimisers'
stopping rules: ``rtol=5e-4`` against survreg (Newton-Raphson; the
largest difference seen is 8e-5) and ``rtol=2e-3`` against fitdistrplus
and lifelines, whose general-purpose optimisers stop on flatter parts of
the surface (the gamma on lung: fitdistrplus's log-likelihood is 3e-5
*below* SurPyval's, its shape 5e-4 away). Standard errors come from a
numerical Hessian in SurPyval and survreg's analytic information:
``rtol=5e-3`` (largest seen 1.1e-3).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

LOGLIK = dict(rtol=0, atol=1e-5)

LogLogisticAFT = sp.AFT(sp.LogLogistic)

AFT_FITTERS = {
    "weibull": sp.WeibullAFT,
    "lognormal": sp.LogNormalAFT,
    "loglogistic": LogLogisticAFT,
    "exponential": sp.ExponentialAFT,
}
DISTRIBUTIONS = {
    "weibull": sp.Weibull,
    "lognormal": sp.LogNormal,
    "loglogistic": sp.LogLogistic,
    "exponential": sp.Exponential,
}


def _interval_xc():
    """The interval fixture in SurPyval's x / c form."""
    d = fixture("interval")
    x, c = [], []
    for left, right in zip(d["left"], d["right"]):
        if np.isnan(right):
            x.append([left, left])
            c.append(1)
        elif left == 0:
            x.append([right, right])
            c.append(-1)
        elif left == right:
            x.append([left, left])
            c.append(0)
        else:
            x.append([left, right])
            c.append(2)
    return np.array(x), np.array(c), d["z"][:, None]


def _data(name):
    if name == "ovarian":
        d = fixture("ovarian")
        return (
            d["futime"],
            1 - d["fustat"],
            np.column_stack([d["ecog_ps"], d["rx"]]),
        )
    if name == "lung":
        d = fixture("lung")
        return d["time"], d["c"], np.column_stack([d["age"], d["sex"]])
    x, c, Z = _interval_xc()
    return x, c, Z


def _survreg_to_surpyval(dist, coef, scale):
    coef = np.atleast_1d(coef)
    if dist in ("weibull", "loglogistic"):
        head = [np.exp(coef[0]), 1 / scale]
    elif dist == "lognormal":
        head = [coef[0], scale]
    else:
        head = [np.exp(-coef[0])]
    return np.concatenate([head, -coef[1:]])


@pytest.mark.parametrize("data", ["ovarian", "lung", "interval"])
@pytest.mark.parametrize("dist", sorted(AFT_FITTERS))
def test_aft_matches_survreg(dist, data):
    x, c, Z = _data(data)
    ref = values("r_survival", "survreg_{}_{}".format(data, dist))
    model = AFT_FITTERS[dist].fit(x, Z, c=c)
    expected = _survreg_to_surpyval(dist, ref["coef"], ref["scale"])
    assert_allclose(model.params, expected, rtol=5e-4)
    assert_allclose(-model.neg_ll(), ref["loglik"], **LOGLIK)
    # The covariate coefficients are -b in both parameterisations, so
    # their standard errors are directly comparable.
    k = Z.shape[1]
    n_coef = np.atleast_1d(ref["coef"]).size
    ref_se = np.sqrt(np.diag(np.atleast_2d(ref["var"])))[n_coef - k : n_coef]
    assert_allclose(model.standard_errors()[-k:], ref_se, rtol=5e-3)


@pytest.mark.parametrize("data", ["ovarian", "lung", "interval"])
@pytest.mark.parametrize(
    "dist, fitter",
    [("weibull", sp.WeibullPH), ("exponential", sp.ExponentialPH)],
)
def test_weibull_ph_is_survreg_reparameterised(dist, fitter, data):
    # The Weibull (and exponential) families are both AFT and PH, so
    # survreg's fit is also the PH fit: the hazard ratio coefficients are
    # -b / scale and the likelihood is the same.
    x, c, Z = _data(data)
    ref = values("r_survival", "survreg_{}_{}".format(data, dist))
    model = fitter.fit(x, Z, c=c)
    k = Z.shape[1]
    coef = np.atleast_1d(ref["coef"])
    assert_allclose(model.params[-k:], -coef[1:] / ref["scale"], rtol=5e-4)
    assert_allclose(-model.neg_ll(), ref["loglik"], **LOGLIK)


@pytest.mark.parametrize("data", ["lung", "interval"])
@pytest.mark.parametrize("dist", sorted(DISTRIBUTIONS))
def test_distribution_matches_intercept_only_survreg(dist, data):
    x, c, _ = _data(data)
    ref = values("r_survival", "survreg_{}_null_{}".format(data, dist))
    model = DISTRIBUTIONS[dist].fit(x=x, c=c)
    expected = _survreg_to_surpyval(dist, ref["coef"], ref["scale"])
    assert_allclose(model.params, expected, rtol=5e-4)
    assert_allclose(-model.neg_ll(), ref["loglik"], **LOGLIK)


def _check_mle(model, ref_params, ref_loglik):
    loglik = -model.neg_ll()
    # SurPyval must reach the reference's maximum (or a higher point).
    assert loglik >= ref_loglik - 1e-6
    assert_allclose(loglik, ref_loglik, rtol=0, atol=1e-4)
    assert_allclose(model.params, ref_params, rtol=2e-3)


# fitdistrplus's names and parameterisations mapped to SurPyval's:
# R's gamma (shape, rate) is SurPyval's Gamma (alpha, beta); R's weibull
# is (shape, scale), the reverse of SurPyval's (alpha scale, beta shape).
FITDISTRPLUS = {
    "interval_gamma": (sp.Gamma, lambda e: e),
    "lung_gamma": (sp.Gamma, lambda e: e),
    "interval_weibull": (sp.Weibull, lambda e: e[::-1]),
    "interval_lnorm": (sp.LogNormal, lambda e: e),
    "interval_norm": (sp.Normal, lambda e: e),
    "interval_logis": (sp.Logistic, lambda e: e),
}


@pytest.mark.parametrize("ref_id", sorted(FITDISTRPLUS))
def test_distribution_matches_fitdistcens(ref_id):
    dist, convert = FITDISTRPLUS[ref_id]
    x, c, _ = _data(ref_id.split("_")[0])
    ref = values("r_fitdistrplus", "fitdistcens_" + ref_id)
    model = dist.fit(x=x, c=c)
    _check_mle(model, convert(ref["estimate"]), ref["loglik"])


def _lifelines_to_surpyval(dist, params):
    params = np.atleast_1d(params)
    if dist == "exponential":
        # lifelines' lambda_ is the scale, SurPyval's the rate.
        return 1 / params
    return params


@pytest.mark.parametrize("dist", sorted(DISTRIBUTIONS))
def test_left_truncated_distribution_matches_lifelines(dist):
    d = fixture("left_truncation")
    ref = values("py_lifelines", "{}_left_truncation".format(dist))
    model = DISTRIBUTIONS[dist].fit(x=d["x"], c=d["c"], tl=d["tl"])
    _check_mle(
        model, _lifelines_to_surpyval(dist, ref["params"]), ref["loglik"]
    )


@pytest.mark.parametrize("dist", sorted(DISTRIBUTIONS))
def test_interval_censored_distribution_matches_lifelines(dist):
    x, c, _ = _interval_xc()
    ref = values("py_lifelines", "{}_interval".format(dist))
    model = DISTRIBUTIONS[dist].fit(x=x, c=c)
    _check_mle(
        model, _lifelines_to_surpyval(dist, ref["params"]), ref["loglik"]
    )


def test_left_truncated_weibull_aft_matches_lifelines():
    d = fixture("left_truncation")
    ref = values("py_lifelines", "weibull_aft_left_truncation")
    t = np.column_stack([d["tl"], np.full(d["x"].size, np.inf)])
    model = sp.WeibullAFT.fit(d["x"], d["z"][:, None], c=d["c"], t=t)
    expected = [
        np.exp(ref["lambda_intercept"]),
        np.exp(ref["rho_intercept"]),
        -ref["lambda_z"],
    ]
    _check_mle(model, expected, ref["loglik"])
