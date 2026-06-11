import numpy as np
import pytest
from numdifftools import Hessian

from surpyval import Weibull
from surpyval.utils.surpyval_data import SurpyvalData


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


def test_hess_inv_matches_observed_information():
    # At the optimum, the delta-method covariance (computed from the
    # Hessian in the transformed optimisation space) must agree with
    # the inverse observed information in the bounded parameter space.
    x = Weibull.random(5000, 10, 3)
    model = Weibull.fit(x)

    data = SurpyvalData(x)

    def neg_ll(params):
        return Weibull._neg_ll_func(data, *params, 0, 0, 1)

    reference = np.linalg.inv(Hessian(neg_ll)(model.params))
    assert np.allclose(model.hess_inv, reference, rtol=1e-4)


def test_param_cb_brackets_true_parameters():
    x = Weibull.random(5000, 10, 3)
    model = Weibull.fit(x)

    alpha_lower, alpha_upper = model.param_cb("alpha")
    beta_lower, beta_upper = model.param_cb("beta")
    assert alpha_lower < 10 < alpha_upper
    assert beta_lower < 3 < beta_upper

    # Two-sided sf bounds must bracket the point estimate.
    t = np.array([5.0, 8.0, 12.0])
    bounds = model.cb(t, on="sf")
    sf = model.sf(t)
    assert (bounds.min(axis=1) <= sf).all()
    assert (bounds.max(axis=1) >= sf).all()


def test_hess_inv_offset_zi_lfp_models():
    # The covariance block returned for offset / zero-inflated / lfp
    # models covers the distribution's parameters only, with gamma, f0,
    # and p marginalised out, so confidence bounds remain available.
    x = Weibull.random(5000, 10, 3) + 5
    model = Weibull.fit(x, offset=True)
    assert model.hess_inv.shape == (2, 2)
    assert np.isfinite(model.hess_inv).all()
    lower, upper = model.param_cb("alpha")
    assert lower < model.params[0] < upper

    x = Weibull.random(5000, 10, 3)
    x[:500] = 0
    model = Weibull.fit(x, zi=True)
    assert model.hess_inv.shape == (2, 2)
    assert np.isfinite(model.hess_inv).all()

    x = Weibull.random(5000, 10, 3)
    never = np.random.uniform(size=5000) < 0.3
    observed = np.where(never, 30.0, np.minimum(x, 30.0))
    c = np.where(never | (x > 30), 1, 0)
    model = Weibull.fit(observed, c, lfp=True)
    assert model.hess_inv.shape == (2, 2)
    assert np.isfinite(model.hess_inv).all()


def test_hess_inv_fixed_parameter_has_zero_variance():
    x = Weibull.random(5000, 10, 3)
    model = Weibull.fit(x, fixed={"beta": 3.0})

    beta_idx = Weibull.param_map["beta"]
    alpha_idx = Weibull.param_map["alpha"]
    assert np.allclose(model.hess_inv[beta_idx, :], 0)
    assert np.allclose(model.hess_inv[:, beta_idx], 0)
    assert model.hess_inv[alpha_idx, alpha_idx] > 0

    lower, upper = model.param_cb("alpha")
    assert lower < 10 < upper
