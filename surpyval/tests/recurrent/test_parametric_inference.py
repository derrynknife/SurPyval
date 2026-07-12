"""Likelihood inference (AIC/BIC/SE/confidence bounds) for the parametric,
regression and renewal recurrent models -- the behaviour provided by
``LikelihoodInferenceMixin`` from ``_neg_ll``/``_mle``/``_n_obs``."""

import matplotlib
import numpy as np
import pytest
from scipy.stats import norm

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from surpyval import Exponential, Weibull  # noqa: E402
from surpyval.recurrent import (  # noqa: E402
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedRenewal,
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


def test_hpp_param_cb_matches_analytic():
    # For an HPP with n events observed to the last event, the observed
    # information gives se = lambda / sqrt(n), so the log-Wald bounds are
    # exactly lambda * exp(+/- z / sqrt(n)).
    x = _intensity_events()
    model = HPP.fit(x)
    lam, n = model.params[0], len(x)
    z = norm.ppf(0.975)
    expected = lam * np.exp(np.array([-1.0, 1.0]) * z / np.sqrt(n))
    assert np.allclose(model.param_cb("lambda"), expected, rtol=1e-4)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_param_cb_brackets_mle_and_respects_support(dist):
    x = _intensity_events()
    model = dist.fit(x)
    for name, (lo, hi), p_hat in zip(
        model.parameter_names, model._parameter_bounds(), model._mle
    ):
        lower, upper = model.param_cb(name)
        assert lower < p_hat < upper
        if lo is not None:
            assert lower > lo
        if hi is not None:
            assert upper < hi
        # One-sided bounds are single values on the matching side of the MLE,
        # and less extreme than the two-sided ones at the same alpha_ci.
        (lower_1s,) = model.param_cb(name, bound="lower")
        (upper_1s,) = model.param_cb(name, bound="upper")
        assert lower < lower_1s < p_hat < upper_1s < upper


def test_param_cb_unknown_name_raises():
    model = CrowAMSAA.fit(_intensity_events())
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.param_cb("nope")


def test_param_cb_requires_likelihood():
    model = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        model.param_cb("alpha")


def test_hpp_cif_cb_matches_analytic():
    # cif = lambda * x, so the delta-method se is x * se(lambda) and the
    # log-transformed band is cif * exp(+/- z * se(lambda) / lambda) -- the
    # relative width is constant in x.
    x = _intensity_events()
    model = HPP.fit(x)
    lam, n = model.params[0], len(x)
    z = norm.ppf(0.975)
    t = np.array([100.0, 500.0])
    expected = (lam * t)[:, None] * np.exp(
        np.array([-1.0, 1.0]) * z / np.sqrt(n)
    )
    assert np.allclose(model.cif_cb(t), expected, rtol=1e-4)


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA, Duane])
def test_cif_cb_brackets_cif(dist):
    x = _intensity_events()
    model = dist.fit(x)
    t = np.linspace(0.0, x.max(), 25)
    cb = model.cif_cb(t)
    cif = model.cif(t)
    assert cb.shape == (t.size, 2)
    # The band starts as a point at the origin (cif(0) == 0) and brackets
    # the fitted curve everywhere else.
    assert np.all(cb[0] == 0.0)
    assert np.all(cb[1:, 0] < cif[1:])
    assert np.all(cb[1:, 1] > cif[1:])
    assert np.all(cb >= 0.0)
    # One-sided bounds match the corresponding side's shape.
    assert model.cif_cb(t, bound="lower").shape == t.shape
    assert model.cif_cb(t, bound="upper").shape == t.shape


def test_cif_cb_requires_likelihood():
    model = CrowAMSAA.fit(_intensity_events(), how="MSE")
    with pytest.raises(ValueError, match="fitted from data"):
        model.cif_cb([1.0, 2.0])


def test_plot_confidence_band():
    model = CrowAMSAA.fit(_intensity_events())
    # The band is drawn as a fill_between collection for MLE fits...
    ax = model.plot()
    assert len(ax.collections) == 1
    plt.close("all")
    # ...can be turned off...
    ax = model.plot(plot_bounds=False)
    assert len(ax.collections) == 0
    plt.close("all")
    # ...and is skipped (not an error) for MSE fits with no likelihood.
    mse_model = CrowAMSAA.fit(_intensity_events(), how="MSE")
    ax = mse_model.plot()
    assert len(ax.collections) == 0
    plt.close("all")


def test_regression_param_cb():
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    # Positive base-rate parameter: log-Wald bounds stay positive.
    lower, upper = model.param_cb("alpha")
    assert 0 < lower < model.params[0] < upper
    # Unbounded coefficient: plain Wald bounds are symmetric about the MLE.
    cb = model.param_cb("beta_0")
    assert np.isclose(cb.mean(), model.coeffs[0])
    assert cb[0] < model.coeffs[0] < cb[1]


def test_regression_cif_cb_brackets_cif():
    x, i, c, Z = _regression_data()
    Z_0 = np.array([0.5])
    t = np.array([5.0, 10.0, 20.0])
    for model in (
        ProportionalIntensityHPP.fit(x, Z, i=i, c=c),
        ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA),
    ):
        cb = model.cif_cb(t, Z_0)
        cif = model.cif(t, Z_0)
        assert cb.shape == (t.size, 2)
        assert np.all(cb[:, 0] < cif) and np.all(cif < cb[:, 1])
        assert np.all(cb > 0)
        ax = model.plot()
        assert len(ax.collections) == 1
        plt.close("all")


def test_pi_hpp_model_functions_delegate_to_constant_baseline():
    # The PI-HPP fitted model's dist used to be a bare namespace with only a
    # name, so cif/iif/inv_cif (and anything built on them) raised.
    x, i, c, Z = _regression_data()
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    Z_0 = np.array([0.5])
    lam = model.params[0]
    phi = np.exp(model.coeffs @ Z_0)
    assert np.allclose(
        model.cif([2.0, 4.0], Z_0), lam * phi * np.array([2.0, 4.0])
    )
    assert np.allclose(model.iif([2.0, 4.0], Z_0), lam * phi)
    assert np.allclose(model.inv_cif(model.cif([3.0], Z_0), Z_0), [3.0])
    assert model.dist.name == "Constant"


def test_renewal_param_cb():
    # The renewal models share the same mixin; the restoration parameter's
    # bounds flow through so its confidence bounds respect the support.
    true = GeneralizedRenewal.fit_from_parameters([10, 2.5], 0.3, dist=Weibull)
    data = true.count_terminated_simulation_data(10, items=6, seed=3)
    model = GeneralizedRenewal.fit_from_recurrent_data(data)
    assert model.parameter_names == ["q", "alpha", "beta"]
    assert model._parameter_bounds() == [(0, None), (0, None), (0, None)]
    lower, upper = model.param_cb("q")
    assert 0 < lower < model.q < upper
    lower, upper = model.param_cb("alpha")
    assert 0 < lower < model.model.params[0] < upper
