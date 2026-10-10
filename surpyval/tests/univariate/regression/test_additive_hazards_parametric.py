"""Parametric additive hazards regression (AH factory + WeibullAH etc.).

The model is h(x|Z) = h_0(x; theta) + beta'Z, fit by plain maximum
likelihood. The additive hazard is not constrained positive: the fit finds
the best positive-hazard solution and raises if the optimum genuinely
requires a negative hazard.
"""

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import (
    AH,
    Exponential,
    ExponentialAH,
    ExponentialPH,
    Weibull,
    WeibullAH,
    WeibullPH,
)
from surpyval.tests._helpers import weibull_binary_covariate_data
from surpyval.univariate.regression.additive_hazards import (
    additive_hazards_fitter,
)


def _exp_ah_data(N, seed, lam=0.5, beta=(0.30, -0.15), tau=6.0):
    # For an Exponential baseline the additive hazard lam + beta'Z is
    # constant in time, so survival is Exp(lam + beta'Z). Recovering
    # (lam, beta) validates the estimator against a known truth.
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta)
    Z = rng.uniform(-1, 1, size=(N, beta.size))
    rate = lam + Z @ beta
    Z, rate = Z[rate > 0], rate[rate > 0]
    x = rng.exponential(1.0 / rate)
    c = (x > tau).astype(int)
    x = np.minimum(x, tau)
    return x, c, Z


def test_factory_and_prebuilt_names():
    assert AH(Weibull).name == "WeibullAH"
    assert AH(Exponential).name == "ExponentialAH"
    assert WeibullAH.name == "WeibullAH"


def test_exponential_ah_recovers_parameters():
    x, c, Z = _exp_ah_data(20000, 0)
    model = ExponentialAH.fit(x=x, Z=Z, c=c)
    # params are [baseline rate, coef_0, coef_1].
    assert np.isclose(model.params[0], 0.5, atol=0.03)
    assert np.allclose(model.params[1:], [0.30, -0.15], atol=0.03)


def test_hazard_is_additive():
    x, c, Z = _exp_ah_data(5000, 1)
    model = ExponentialAH.fit(x=x, Z=Z, c=c)
    Z0 = np.array([[0.2, -0.1]])
    rate, beta = model.params[0], model.params[1:]
    # For the exponential baseline the hazard is constant: rate + beta'Z0.
    expected = rate + Z0 @ beta
    assert np.allclose(model.hf([1.0, 2.0, 3.0], Z0), expected)


def test_cumulative_hazard_additive_linear_term():
    # H(x|Z) - H(x|0) = x * beta'Z (the additive covariate contribution
    # integrates linearly in time).
    x, c, Z = _exp_ah_data(5000, 2)
    model = WeibullAH.fit(x=x, Z=Z, c=c)
    t = np.array([1.0, 2.0, 4.0])
    Z1 = np.array([[0.3, -0.2]])
    Z0 = np.array([[0.0, 0.0]])
    beta = model.params[-2:]
    diff = model.Hf(t, Z1) - model.Hf(t, Z0)
    assert np.allclose(diff, t * (Z1 @ beta), atol=1e-6)


def test_prediction_identities():
    x, c, Z = _exp_ah_data(4000, 3)
    model = ExponentialAH.fit(x=x, Z=Z, c=c)
    t = np.array([0.5, 1.0, 2.0])
    Z0 = np.array([[0.1, -0.1]])
    assert np.allclose(model.ff(t, Z0), 1 - model.sf(t, Z0))
    assert np.allclose(model.df(t, Z0), model.hf(t, Z0) * model.sf(t, Z0))


def test_baseline_reduces_to_distribution_when_Z_zero():
    # With Z = 0 the additive term vanishes and the survival must equal the
    # plain baseline distribution's survival at the fitted parameters.
    x, c, Z = _exp_ah_data(8000, 4)
    model = ExponentialAH.fit(x=x, Z=Z, c=c)
    t = np.array([0.5, 1.0, 2.0, 3.0])
    Z0 = np.array([[0.0, 0.0]])
    assert np.allclose(
        model.sf(t, Z0), Exponential.sf(t, model.params[0]), atol=1e-9
    )


def test_weibull_ah_on_exponential_data_has_unit_shape():
    # A Weibull baseline fit to exponentially-distributed inter-event times
    # should recover a shape parameter near 1 and the same covariate betas.
    x, c, Z = _exp_ah_data(20000, 5)
    model = WeibullAH.fit(x=x, Z=Z, c=c)
    # Weibull params are (alpha, beta_shape); shape ~ 1 for exponential data.
    assert np.isclose(model.params[1], 1.0, atol=0.1)
    assert np.allclose(model.params[-2:], [0.30, -0.15], atol=0.04)


def test_information_criteria_available():
    x, c, Z = _exp_ah_data(4000, 6)
    model = ExponentialAH.fit(x=x, Z=Z, c=c)
    assert np.isfinite(model.aic())
    assert np.isfinite(model.bic())
    assert "Additive Hazard" in repr(model)


def test_fit_from_df_retains_feature_names():
    x, c, Z = _exp_ah_data(3000, 7)
    df = pd.DataFrame({"t": x, "c": c, "age": Z[:, 0], "dose": Z[:, 1]})
    model = ExponentialAH.fit_from_df(
        df, x_col="t", Z_cols=["age", "dose"], c_col="c"
    )
    assert model.feature_names == ["age", "dose"]
    pred = model.sf([1.0, 2.0], df[["age", "dose"]].iloc[[0]])
    assert pred.shape == (2,)


def test_fit_fails_when_hazard_forced_negative():
    # Forcing a large negative coefficient makes h_0(x) + beta'Z < 0 at
    # observed times, so no finite-likelihood fit exists and the fit raises
    # with a message pointing at proportional hazards.
    rng = np.random.default_rng(8)
    Z = rng.uniform(0, 1, size=(500, 1))
    x = rng.exponential(2.0, size=500)
    with pytest.raises(ValueError, match="positive"):
        ExponentialAH.fit(x=x, Z=Z, fixed={"coef_0": -5.0})
    # The same data fits fine without the pathological constraint.
    model = ExponentialAH.fit(x=x, Z=Z)
    assert np.all(model.hf(x, Z) > 0)


def test_counts_equivalent_to_repeated_rows():
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    c = np.array([0, 0, 1, 0, 0])
    Z = np.array([[0.4], [0.7], [0.7], [-0.2], [0.9]])
    m_rep = ExponentialAH.fit(
        x=np.repeat(x, 2), Z=np.repeat(Z, 2, axis=0), c=np.repeat(c, 2)
    )
    m_cnt = ExponentialAH.fit(x=x, Z=Z, c=c, n=np.full(5, 2))
    assert np.allclose(m_rep.params, m_cnt.params, atol=1e-3)


def test_agrees_with_ph_on_direction_of_effect():
    # AH and PH are different scales, but a covariate that raises the hazard
    # should have the same sign of effect under both.
    x, c, Z = _exp_ah_data(20000, 9)
    ah = ExponentialAH.fit(x=x, Z=Z, c=c)
    ph = ExponentialPH.fit(x=x, Z=Z, c=c)
    assert np.all(np.sign(ah.params[-2:]) == np.sign(ph.params[-2:]))


# ---------------------------------------------------------------------------
# Parametric additive hazards fits take the exact gradient first
# (#515): BFGS on the likelihood's autograd gradient from the
# start, with a budget, instead of Nelder-Mead then TNC on
# finite differences.
# ---------------------------------------------------------------------------


def _positive_data():
    rng = np.random.default_rng(2)
    n = 400
    Z = rng.uniform(0, 1, (n, 2))
    x = rng.exponential(1 / (0.05 + Z @ [0.05, 0.02]))
    c = (x > 15).astype(int)
    return dict(x=np.minimum(x, 15), Z=Z, c=c)


def _doc_data():
    rng = np.random.RandomState(1)
    Z = rng.binomial(1, 0.5, 100).reshape(-1, 1)
    x = sp.Weibull.qf(rng.uniform(size=100), 10, 2) * np.exp(-0.5 * Z[:, 0])
    return dict(x=x, Z=Z, c=np.zeros(100))


# The negative log-likelihood the Nelder-Mead-then-TNC search reached
# (before #515).
_OLD_NEG_LL = {
    ("positive", "WeibullAH"): 974.2367457938165,
    ("positive", "ExponentialAH"): 974.3385172069866,
    ("positive", "GammaAH"): 974.2940760703287,
    ("positive", "LogNormalAH"): 977.2681312431007,
}


@pytest.mark.parametrize("case,name", sorted(_OLD_NEG_LL))
def test_reaches_at_least_the_old_searchs_optimum(case, name):
    model = getattr(sp, name).fit(**_positive_data())
    assert model._neg_ll <= _OLD_NEG_LL[(case, name)] + 1e-6


@pytest.mark.parametrize("name", ["WeibullAH", "ExponentialAH", "GammaAH"])
def test_no_nelder_mead_when_the_gradient_search_converges(name):
    methods = []
    real = additive_hazards_fitter.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real(*args, **kwargs)

    with mock.patch.object(additive_hazards_fitter, "minimize", recording):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = getattr(sp, name).fit(**_doc_data())
    assert np.isfinite(model._neg_ll)
    assert "Nelder-Mead" not in methods, methods


def test_the_gradient_search_has_a_budget():
    # Line searches that chase a coefficient can take thousands of
    # gradients (a no-event level's ran away before #828 kept the fit
    # inside its support); past the budget the fit falls back to its
    # derivative-free search, and reaches the same fit.
    x, Z = weibull_binary_covariate_data()
    full = WeibullAH.fit(x, Z)
    calls = []
    real = additive_hazards_fitter.AdditiveHazardsFitter._gradient_first

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out[1])
        return out

    fitter = additive_hazards_fitter.AdditiveHazardsFitter
    with mock.patch.object(fitter, "GRADIENT_FIRST_BUDGET", 3):
        with mock.patch.object(fitter, "_gradient_first", staticmethod(spy)):
            fell_back = WeibullAH.fit(x, Z)
    assert calls == [False]
    np.testing.assert_allclose(fell_back.params, full.params, rtol=1e-4)


# ---------------------------------------------------------------------------
# ``random()`` shapes; the positivity boundary warns.
# ---------------------------------------------------------------------------


def test_ah_random_matches_ph_shapes():
    x, Z = weibull_binary_covariate_data()
    ah = WeibullAH.fit(x, Z)
    ph = WeibullPH.fit(x, Z)
    for Zq in ([[0.0], [1.0]], [[1.0]], [0.0]):
        ah_x, ah_Z = ah.random(3, Zq)
        ph_x, ph_Z = ph.random(3, Zq)
        assert ah_x.shape == ph_x.shape
        assert ah_Z.shape == ph_Z.shape
        assert ah_Z.ndim == 2
    np.random.seed(0)
    draws, rows = ah.random(4, [[0.0], [1.0]])
    assert np.array_equal(rows[:, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    assert np.all(draws > 0)


def test_additive_hazards_warns_on_positivity_boundary():
    # A strongly protective group with one failure: its hazard at that
    # failure is pressed towards 0 (the barrier of log h), and that one
    # failure carries most of the information about beta.
    rng = np.random.default_rng(0)
    x = np.r_[rng.exponential(10, 60), 1.0, np.full(30, 50.0)]
    Z = np.r_[np.zeros(60), np.ones(31)].reshape(-1, 1)
    c = np.r_[np.zeros(61), np.ones(30)]
    with pytest.warns(UserWarning, match="positivity boundary"):
        model = sp.ExponentialAH.fit(x, Z, c=c)
    assert model.maximum == "unverified"


def test_additive_hazards_warns_on_the_support_boundary():
    # A strongly protective covariate on a baseline whose hazard starts at
    # 0: unconstrained, the fit put sf above 1 at an early observation of
    # the protective rows. Kept inside its support (#828) it ends where H
    # reaches 0 there, and says so.
    np.random.seed(0)
    Z = np.random.binomial(1, 0.5, 300).reshape(-1, 1).astype(float)
    x = Weibull.random(300, 10, 2) * np.exp(1.5 * Z[:, 0])
    with pytest.warns(UserWarning, match="boundary of the model's support"):
        model = WeibullAH.fit(x, Z)
    assert model.maximum == "unverified"
    H = np.ravel(model.model.Hf(x, Z, *model.params))
    assert H.min() >= 0 and H.min() < 1e-6


def test_additive_hazards_interior_fit_does_not_warn():
    x, Z = weibull_binary_covariate_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WeibullAH.fit(x, Z)
