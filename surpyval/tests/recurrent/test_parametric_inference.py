"""Likelihood inference (AIC/BIC/SE) for the parametric and regression
recurrent models -- the behaviour added when these fitters started setting
``_neg_ll``/``_mle``/``_n_obs`` and inheriting ``LikelihoodInferenceMixin``."""

import numpy as np
import pytest

from surpyval import Exponential
from surpyval.recurrent import (
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)


def _intensity_events():
    np.random.seed(1)
    return Exponential.random(40, 1e-2).cumsum()


def _assert_information_criteria(model, dist):
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert np.isclose(model.aic, 2 * k - 2 * ll)
    assert np.isclose(model.bic, k * np.log(n) - 2 * ll)
    assert model.parameter_names == list(dist.param_names)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_parametric_intensity_information_criteria(dist):
    # Every MLE-fitted intensity model now exposes a likelihood and the
    # standard information criteria derived from it.
    model = dist.fit(_intensity_events())
    _assert_information_criteria(model, dist)


def test_cox_lewis_information_criteria():
    # Cox-Lewis needs a log-linear intensity over a bounded window (its
    # exponential CIF overflows on very large cumulative times), so it gets a
    # tailored dataset rather than the shared one.
    rng = np.random.default_rng(0)
    alpha, beta, T = 0.0, 0.3, 20.0
    lam_max = np.exp(alpha + beta * T)
    cand = np.sort(rng.uniform(0, T, rng.poisson(lam_max * T)))
    keep = rng.uniform(0, 1, cand.size) < np.exp(alpha + beta * cand) / lam_max
    model = CoxLewis.fit(cand[keep], tl=0.0, tr=T)
    _assert_information_criteria(model, CoxLewis)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_parametric_intensity_standard_errors(dist):
    # Standard errors come from the observed information and are finite and
    # positive for these well-identified fits.
    x = _intensity_events()
    model = dist.fit(x)
    se = model.standard_errors()
    assert se.shape == (model._mle.size,)
    assert np.all(np.isfinite(se)) and np.all(se > 0)


def test_mse_fit_has_no_likelihood():
    # The MSE fit minimises a sum of squares, not a likelihood, so inference
    # must raise rather than report a meaningless AIC.
    x = _intensity_events()
    model = CrowAMSAA.fit(x, how="MSE")
    for attr in ("log_likelihood", "aic", "bic"):
        with pytest.raises(ValueError, match="fitted from data"):
            getattr(model, attr)


def test_from_params_has_no_likelihood():
    # A model built directly from parameters carries no data/likelihood.
    model = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        model.aic


def _regression_data():
    x = [9, 14, 18, 20, 7, 12, 16, 19, 20, 5, 9, 13, 16, 18, 20]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    c = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    Z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    return x, i, c, Z


def test_nhpp_regression_information_criteria():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert np.isclose(model.aic, 2 * k - 2 * ll)
    assert np.isclose(model.bic, k * np.log(n) - 2 * ll)
    # Base-rate parameters first, then one coefficient per covariate column.
    assert model.parameter_names == ["alpha", "beta", "beta_0"]
    assert np.all(np.isfinite(model.standard_errors()))


def test_hpp_regression_information_criteria():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    ll = model.log_likelihood
    # ``_mle`` is in natural (rate) space; ``_neg_ll`` must agree there.
    assert np.isclose(ll, -float(model._neg_ll(model._mle)))
    assert model.parameter_names == ["lambda", "beta_0"]
    assert np.isfinite(model.aic) and np.isfinite(model.bic)
    assert np.all(np.isfinite(model.standard_errors()))
