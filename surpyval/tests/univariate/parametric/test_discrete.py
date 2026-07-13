"""Discrete lifetime distributions on the positive integers {1, 2, 3, ...}:
Geometric, DiscreteWeibull (Nakagawa-Osaki Type I) and NegativeBinomial.

The closed forms are checked against scipy where an equivalent exists, the
internal survival/CDF/pmf/hazard identities are checked directly, and the
MLE path is checked to recover known parameters under censoring, truncation
and zero-inflation.
"""

import numpy as np
import pytest
from scipy.stats import geom, nbinom

from surpyval import DiscreteWeibull, Geometric, NegativeBinomial

INTS = np.array([1, 2, 3, 4, 5, 6], dtype=float)


# --- internal consistency shared by every discrete distribution -----------


DISTS = [
    (Geometric, (0.3,)),
    (DiscreteWeibull, (0.7, 1.6)),
    (DiscreteWeibull, (0.5, 0.7)),
    (NegativeBinomial, (2.5, 0.4)),
]


@pytest.mark.parametrize("dist,params", DISTS)
def test_pmf_sums_to_one(dist, params):
    assert np.isclose(np.sum(dist.df(np.arange(1, 5000.0), *params)), 1.0)


@pytest.mark.parametrize("dist,params", DISTS)
def test_survival_cdf_pmf_hazard_identities(dist, params):
    k = INTS
    # sf = 1 - ff
    assert np.allclose(dist.sf(k, *params), 1 - dist.ff(k, *params))
    # pmf = F(k) - F(k-1)
    assert np.allclose(
        dist.df(k, *params), dist.ff(k, *params) - dist.ff(k - 1, *params)
    )
    # discrete hazard = pmf(k) / P(T >= k) = pmf(k) / sf(k-1)
    assert np.allclose(
        dist.hf(k, *params), dist.df(k, *params) / dist.sf(k - 1, *params)
    )
    # cumulative hazard = -log S
    assert np.allclose(dist.Hf(k, *params), -np.log(dist.sf(k, *params)))
    # log_df / log_sf match the logs of df / sf
    assert np.allclose(dist.log_df(k, *params), np.log(dist.df(k, *params)))
    assert np.allclose(dist.log_sf(k, *params), np.log(dist.sf(k, *params)))


@pytest.mark.parametrize("dist,params", DISTS)
def test_qf_inverts_cdf(dist, params):
    # qf(u) is the smallest integer k with F(k) >= u.
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    k = dist.qf(u, *params)
    assert np.all(k >= 1)
    assert np.all(dist.ff(k, *params) >= u - 1e-9)
    assert np.all(dist.ff(k - 1, *params) < u + 1e-9)


# --- checks against scipy where an equivalent exists ----------------------


def test_geometric_matches_scipy():
    p = 0.3
    assert np.allclose(Geometric.df(INTS, p), geom.pmf(INTS, p))
    assert np.allclose(Geometric.sf(INTS, p), geom.sf(INTS, p))
    assert np.allclose(Geometric.ff(INTS, p), geom.cdf(INTS, p))
    assert np.allclose(Geometric.hf(INTS, p), p)  # constant hazard
    assert np.isclose(Geometric.mean(p), 1 / p)


def test_negative_binomial_matches_scipy():
    r, p = 2.5, 0.4
    # T = Y + 1 with Y ~ nbinom(r, p).
    assert np.allclose(
        NegativeBinomial.df(INTS, r, p), nbinom.pmf(INTS - 1, r, p)
    )
    assert np.allclose(
        NegativeBinomial.ff(INTS, r, p), nbinom.cdf(INTS - 1, r, p)
    )
    assert np.isclose(NegativeBinomial.mean(r, p), 1 + nbinom.mean(r, p))


# --- special-case reductions ----------------------------------------------


def test_discrete_weibull_reduces_to_geometric():
    # beta = 1 gives R(k) = q^k, i.e. Geometric with p = 1 - q.
    q = 0.65
    assert np.allclose(
        DiscreteWeibull.df(INTS, q, 1.0), Geometric.df(INTS, 1 - q)
    )


def test_negative_binomial_reduces_to_geometric():
    # r = 1 gives the Geometric with per-trial probability p.
    p = 0.35
    assert np.allclose(
        NegativeBinomial.df(INTS, 1.0, p), Geometric.df(INTS, p)
    )


# --- MLE recovery ---------------------------------------------------------


def test_geometric_mle_recovers_parameter():
    np.random.seed(1)
    x = Geometric.random(6000, 0.25)
    model = Geometric.fit(x)
    assert np.isclose(model.params[0], 0.25, atol=0.02)


def test_discrete_weibull_mle_recovers_parameters():
    np.random.seed(2)
    x = DiscreteWeibull.random(8000, 0.7, 1.6)
    model = DiscreteWeibull.fit(x)
    assert np.allclose(model.params, [0.7, 1.6], atol=0.06)


def test_negative_binomial_mle_recovers_parameters():
    np.random.seed(3)
    x = NegativeBinomial.random(8000, 2.5, 0.4)
    model = NegativeBinomial.fit(x)
    assert np.allclose(model.params, [2.5, 0.4], rtol=0.1)


def test_mle_with_right_censoring():
    np.random.seed(4)
    x = DiscreteWeibull.random(6000, 0.8, 2.0)
    c = np.zeros_like(x)
    c[x > 6] = 1
    x[x > 6] = 6
    model = DiscreteWeibull.fit(x, c=c)
    assert np.allclose(model.params, [0.8, 2.0], atol=0.08)


def test_mle_with_left_truncation():
    np.random.seed(5)
    x = Geometric.random(5000, 0.3)
    keep = x > 2
    model = Geometric.fit(x[keep], tl=2)
    assert np.isclose(model.params[0], 0.3, atol=0.03)


def test_mle_with_interval_censoring():
    # Every unit known only to fail within (xl, xr].
    model = Geometric.fit(xl=[2, 3, 4, 5, 2, 3], xr=[4, 5, 6, 7, 5, 6])
    assert 0.0 < model.params[0] < 1.0


def test_zero_inflation_recovers_structural_zeros():
    np.random.seed(6)
    x = NegativeBinomial.random(5000, 2.0, 0.5)
    x = np.concatenate([x, np.zeros(1250)])  # ~20% structural zeros
    model = NegativeBinomial.fit(x, zi=True)
    assert np.isclose(model.f0, 1250 / x.size, atol=0.03)
    assert np.allclose(model.params, [2.0, 0.5], rtol=0.2)


# --- model object ---------------------------------------------------------


@pytest.mark.parametrize("dist,params", DISTS)
def test_from_params_and_prediction(dist, params):
    model = dist.from_params(list(params))
    assert np.all(model.sf(INTS) >= 0) and np.all(model.sf(INTS) <= 1)
    assert np.allclose(model.ff(INTS), 1 - model.sf(INTS))
    assert dist.name in repr(model)
    # aic is finite for a fitted-parameter model with data
    fitted = dist.fit(dist.random(500, *params))
    assert np.isfinite(fitted.aic())


@pytest.mark.parametrize("dist,params", DISTS)
def test_fitted_model_methods_use_uncorrupted_parameters(dist, params):
    # Distributions whose parameter is named "p" (Geometric,
    # NegativeBinomial) must not overwrite the reserved limited-failure
    # proportion ``model.p`` when the fitted parameters are exposed by
    # name, or the survival functions silently compute against the wrong
    # mixing. Fit (not from_params, which took a different path) and check
    # the model's sf/mean match the distribution evaluated at the fitted
    # parameters directly.
    x = dist.random(3000, *params)
    model = dist.fit(x)
    assert model.p == 1.0  # no limited-failure component was requested
    k = np.array([1.0, 2.0, 3.0, 5.0])
    assert np.allclose(model.sf(k), dist.sf(k, *model.params))
    assert np.isclose(model.mean(), dist.mean(*model.params))


def test_supports_mpp_is_false():
    # Probability plotting is not defined for these discrete lifetimes, so
    # the MPP method must be rejected rather than silently misbehave.
    for dist in (Geometric, DiscreteWeibull, NegativeBinomial):
        assert dist.supports_mpp is False
    with pytest.raises(ValueError):
        Geometric.fit(Geometric.random(200, 0.3), how="MPP")
