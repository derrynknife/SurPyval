"""Discrete lifetime distributions on the positive integers {1, 2, 3, ...}:
Geometric, DiscreteWeibull (Nakagawa-Osaki Type I) and NegativeBinomial.

The closed forms are checked against scipy where an equivalent exists, the
internal survival/CDF/pmf/hazard identities are checked directly, and the
MLE path is checked to recover known parameters under censoring, truncation
and zero-inflation.
"""

import numpy as np
import pytest
from scipy.stats import geom, nbinom, poisson

from surpyval import (
    Bernoulli,
    BetaGeometric,
    Binomial,
    DiscreteWeibull,
    Discretize,
    Gamma,
    Geometric,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    Weibull,
)

INTS = np.array([1, 2, 3, 4, 5, 6], dtype=float)

# A continuous distribution discretized onto {1, 2, 3, ...}.
DiscretizedWeibull = Discretize(Weibull)


# --- internal consistency shared by every discrete distribution -----------


DISTS = [
    (Geometric, (0.3,)),
    (DiscreteWeibull, (0.7, 1.6)),
    (DiscreteWeibull, (0.5, 0.7)),
    (NegativeBinomial, (2.5, 0.4)),
    (BetaGeometric, (2.5, 3.0)),
    (DiscretizedWeibull, (10.0, 2.0)),
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


def test_discrete_distributions_are_discrete_fitters():
    # The discrete/continuous distinction is structural: every integer-
    # support distribution is a DiscreteParametricFitter with the
    # ``discrete`` trait, and the continuous catalogue is not.
    from surpyval import Bernoulli, Binomial, FixedEventProbability
    from surpyval.univariate.parametric import DiscreteParametricFitter

    for dist in (
        Geometric,
        Poisson,
        DiscreteWeibull,
        NegativeBinomial,
        Binomial,
        Bernoulli,
        FixedEventProbability,
        BetaGeometric,
        DiscretizedWeibull,
    ):
        assert isinstance(dist, DiscreteParametricFitter), dist.name
        assert dist.discrete is True
        assert dist.supports_mpp is False
    for dist in (Weibull, Gamma, LogNormal, Normal):
        assert not isinstance(dist, DiscreteParametricFitter)
        assert dist.discrete is False


def test_mps_rejected_for_discrete():
    # Maximum product of spacings is defined by increments of a continuous
    # CDF; repeated integers make the spacings degenerate, so it must be
    # rejected with a clear error rather than fit nonsense.
    x = Geometric.random(200, 0.3)
    with pytest.raises(ValueError, match="MPS"):
        Geometric.fit(x, how="MPS")
    with pytest.raises(ValueError, match="MPS"):
        Poisson.fit(Poisson.random(200, 3.0), how="MPS")
    # MLE remains the standard path.
    assert 0.0 < Geometric.fit(x).params[0] < 1.0


# --- Tier 2: Poisson, Beta-Geometric, and the Discretize wrapper ----------


def test_poisson_matches_scipy():
    mu = 3.5
    k = np.arange(0, 12, dtype=float)
    assert np.allclose(Poisson.df(k, mu), poisson.pmf(k, mu))
    assert np.allclose(Poisson.sf(k, mu), poisson.sf(k, mu))
    assert np.allclose(Poisson.ff(k, mu), poisson.cdf(k, mu))
    assert np.isclose(Poisson.mean(mu), mu)


def test_poisson_pmf_sums_to_one_from_zero():
    # Poisson lives on {0, 1, 2, ...}, so the mass at 0 must be counted.
    assert np.isclose(np.sum(Poisson.df(np.arange(0, 60.0), 4.0)), 1.0)


def test_poisson_mle_recovers_parameter():
    x = poisson.rvs(4.0, size=6000, random_state=1).astype(float)
    model = Poisson.fit(x)
    assert np.isclose(model.params[0], 4.0, rtol=0.05)


def test_beta_geometric_hazard_decreases():
    # The frailty signature: mixing p over a Beta makes the marginal hazard
    # fall with time (a single Geometric has constant hazard).
    k = np.arange(1, 30, dtype=float)
    h = BetaGeometric.hf(k, 2.5, 3.0)
    assert np.all(np.diff(h) < 1e-9)


def test_beta_geometric_mean_matches_truncated_sum():
    a, b = 2.5, 3.0  # a > 1 so the mean is finite
    k = np.arange(1, 200000, dtype=float)
    mean_sum = np.sum(k * BetaGeometric.df(k, a, b))
    assert np.isclose(BetaGeometric.mean(a, b), mean_sum, rtol=1e-3)
    assert np.isinf(BetaGeometric.mean(0.5, 3.0))  # a <= 1 diverges


def test_beta_geometric_mle_recovers_parameters():
    np.random.seed(7)
    x = BetaGeometric.random(8000, 2.0, 4.0)
    model = BetaGeometric.fit(x)
    assert np.allclose(model.params, [2.0, 4.0], rtol=0.2)


def test_discretize_equals_continuous_survival_and_binned_mass():
    w = Weibull.from_params([10.0, 2.0])
    k = np.arange(1, 30, dtype=float)
    # discrete survival at an integer is the continuous survival there
    assert np.allclose(DiscretizedWeibull.sf(k, 10.0, 2.0), w.sf(k))
    # the pmf is the continuous probability of the bin (k-1, k]
    assert np.allclose(
        DiscretizedWeibull.df(k, 10.0, 2.0), w.ff(k) - w.ff(k - 1.0)
    )
    assert np.isclose(
        np.sum(DiscretizedWeibull.df(np.arange(1, 500.0), 10.0, 2.0)), 1.0
    )


@pytest.mark.parametrize(
    "dist,params",
    [(Weibull, (12.0, 1.8)), (Gamma, (3.0, 2.0)), (LogNormal, (2.0, 0.4))],
)
def test_discretize_mle_recovers_parameters(dist, params):
    np.random.seed(8)
    discrete = Discretize(dist)
    x = np.ceil(dist.random(6000, *params)).astype(float)
    model = discrete.fit(x)
    assert np.allclose(model.params, params, rtol=0.1)


def test_discretize_rejects_negative_support():
    # Discretize is for non-negative lifetimes; the Normal spans the reals.
    with pytest.raises(ValueError, match="non-negative"):
        Discretize(Normal)


def test_discretize_name_is_distinct_from_discrete_weibull():
    assert DiscretizedWeibull.name == "Discretize(Weibull)"
    assert DiscretizedWeibull.name != DiscreteWeibull.name


def test_log_df_uses_the_discrete_mass_identity():
    # P(T = k) = h(k) R(k - 1). ParametricFitter.log_df encodes the
    # continuous f = h R(x) instead, which puts R(k) where R(k - 1)
    # belongs. Binomial inherited it and returned the mass scaled by
    # R(k)/R(k - 1) -- 0.88 of the truth at k=1, falling to 0.15 by k=7.
    x = np.arange(1, 8, dtype=float)
    cases = [
        (Binomial, (10, 0.3)),
        (Poisson, (3.0,)),
        (Geometric, (0.3,)),
        (NegativeBinomial, (4.0, 0.4)),
        (DiscreteWeibull, (0.6, 1.2)),
        (BetaGeometric, (2.0, 3.0)),
    ]
    for dist, params in cases:
        df = np.asarray(dist.df(x, *params), dtype=float)
        log_df = np.asarray(dist.log_df(x, *params), dtype=float)
        np.testing.assert_allclose(
            np.exp(log_df), df, rtol=1e-9, err_msg=f"{dist.name}"
        )


def test_binomial_log_df_matches_scipy():
    # Binomial is the one that reached the inherited implementation, so
    # it is the one worth pinning against an independent source.
    from scipy.stats import binom

    x = np.arange(0, 11, dtype=float)
    np.testing.assert_allclose(
        np.asarray(Binomial.log_df(x, 10, 0.3), dtype=float),
        binom.logpmf(x, 10, 0.3),
        rtol=1e-9,
    )


def test_discrete_hazard_is_a_probability():
    # The discrete hazard is P(T = k)/P(T >= k), which is a probability
    # and cannot exceed one. The continuous form P(T = k)/P(T > k) can,
    # which is how a mixed-up convention shows itself.
    x = np.arange(1, 12, dtype=float)
    for dist, params in [
        (Poisson, (3.0,)),
        (Geometric, (0.3,)),
        (DiscreteWeibull, (0.6, 1.2)),
        (NegativeBinomial, (4.0, 0.4)),
        (BetaGeometric, (2.0, 3.0)),
        (Binomial, (10, 0.3)),
    ]:
        hf = np.asarray(dist.hf(x, *params), dtype=float)
        finite = hf[np.isfinite(hf)]
        assert np.all(finite >= 0.0), dist.name
        assert np.all(
            finite <= 1.0 + 1e-12
        ), f"{dist.name}: max {finite.max()}"


def test_cumulative_hazard_accumulates_the_discrete_way():
    # For a discrete distribution R(k) = prod(1 - h), so the cumulative
    # hazard is -sum log(1 - h), not the sum of the hazards. Hf and hf
    # must agree through that relation.
    x = np.arange(1, 9, dtype=float)
    for dist, params in [
        (Poisson, (3.0,)),
        (Geometric, (0.3,)),
        (DiscreteWeibull, (0.6, 1.2)),
        (Binomial, (10, 0.3)),
    ]:
        hf = np.asarray(dist.hf(x, *params), dtype=float)
        Hf = np.asarray(dist.Hf(x, *params), dtype=float)
        seed = -np.log(float(np.asarray(dist.sf(0.0, *params))))
        np.testing.assert_allclose(
            np.cumsum(-np.log1p(-hf)) + seed,
            Hf,
            rtol=1e-7,
            err_msg=f"{dist.name}",
        )


# ---------------------------------------------------------------------------
# Bernoulli: a true coin flip since 0.19.1
# ---------------------------------------------------------------------------

P_BERN = 0.3


def test_bernoulli_functions_at_the_two_outcomes():
    x = np.array([0, 1])
    np.testing.assert_allclose(Bernoulli.sf(x, P_BERN), [1.0, P_BERN])
    np.testing.assert_allclose(Bernoulli.ff(x, P_BERN), [0.0, 1 - P_BERN])
    np.testing.assert_allclose(Bernoulli.df(x, P_BERN), [1 - P_BERN, P_BERN])
    np.testing.assert_allclose(Bernoulli.hf(x, P_BERN), [1 - P_BERN, 1.0])
    np.testing.assert_allclose(Bernoulli.Hf(x, P_BERN), [0.0, -np.log(P_BERN)])


def test_bernoulli_rejects_anything_but_zero_and_one():
    # x is the outcome of the flip, not a time. Before 0.19.1 every x
    # returned the same number, so nothing marked 37.5 as meaningless.
    for bad in (0.5, 2, -1, 37.5):
        for method in (
            Bernoulli.sf,
            Bernoulli.ff,
            Bernoulli.df,
            Bernoulli.hf,
            Bernoulli.Hf,
        ):
            with pytest.raises(ValueError, match="x = 0 and x = 1 only"):
                method(bad, P_BERN)


def test_bernoulli_internal_identities():
    x = np.array([0, 1])
    sf = np.asarray(Bernoulli.sf(x, P_BERN), dtype=float)
    ff = np.asarray(Bernoulli.ff(x, P_BERN), dtype=float)
    df = np.asarray(Bernoulli.df(x, P_BERN), dtype=float)
    hf = np.asarray(Bernoulli.hf(x, P_BERN), dtype=float)
    Hf = np.asarray(Bernoulli.Hf(x, P_BERN), dtype=float)
    log_df = np.asarray(Bernoulli.log_df(x, P_BERN), dtype=float)

    np.testing.assert_allclose(sf + ff, 1.0)
    np.testing.assert_allclose(df.sum(), 1.0)
    np.testing.assert_allclose(hf, df / sf)
    np.testing.assert_allclose(Hf, -np.log(sf))
    np.testing.assert_allclose(np.exp(log_df), df)
    # E[X] from the mass equals the parameter, and equals mean/moment.
    np.testing.assert_allclose((x * df).sum(), P_BERN)
    np.testing.assert_allclose(Bernoulli.mean(P_BERN), P_BERN)
    np.testing.assert_allclose(Bernoulli.moment(1, P_BERN), P_BERN)
    # X is 0 or 1 so X**m == X, and every moment is p.
    for m in (1, 2, 5):
        np.testing.assert_allclose(Bernoulli.moment(m, P_BERN), P_BERN)


def test_bernoulli_log_df_needs_its_own_relation():
    # DiscreteParametricFitter.log_df is f(k) = h(k) R(k - 1), which
    # assumes R(k) = P(X > k). Bernoulli's R is P(X >= x), so the
    # at-risk set at x is R(x) itself. Inheriting the discrete relation
    # would give df(1) = 1 instead of p.
    x = np.array([0, 1])
    np.testing.assert_allclose(
        np.exp(np.asarray(Bernoulli.log_df(x, P_BERN), dtype=float)),
        np.asarray(Bernoulli.df(x, P_BERN), dtype=float),
    )
    # At x = 1 the discrete relation would read h(1) * R(0) = 1 * 1 = 1,
    # where the mass is p. (R(-1) is not even askable here, which is the
    # other half of why that relation does not transfer.)
    h1 = float(np.ravel(Bernoulli.hf(1, P_BERN))[0])
    R0 = float(np.ravel(Bernoulli.sf(0, P_BERN))[0])
    assert np.isclose(h1 * R0, 1.0)
    assert not np.isclose(h1 * R0, P_BERN)
    with pytest.raises(ValueError):
        Bernoulli.sf(-1, P_BERN)


def test_bernoulli_fit_recovers_the_fraction_of_ones():
    data = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    model = Bernoulli.fit(data)
    np.testing.assert_allclose(model.params, [data.mean()])


def test_fixed_event_probability_is_unchanged_and_separate():
    # The flat model Bernoulli used to be. It is now its own class, so
    # making Bernoulli a real Bernoulli did not take it away.
    from surpyval import FixedEventProbability

    assert type(FixedEventProbability) is not type(Bernoulli)
    for x in (0.0, 1.0, 37.5, -12.0):
        np.testing.assert_allclose(FixedEventProbability.ff(x, P_BERN), P_BERN)
        np.testing.assert_allclose(
            FixedEventProbability.sf(x, P_BERN), 1 - P_BERN
        )
    # Its F is constant, so it still has no density, hazard rate, quantile
    # or mean: the mass is an atom rather than a density, and there is no
    # time axis to invert or average over.
    for absent in ("df", "hf", "qf", "mean"):
        assert not any(
            absent in k.__dict__ for k in type(FixedEventProbability).__mro__
        ), absent
    # ``Hf`` is the exception, and is present. -ln R(x) is a perfectly good
    # constant, exactly as for ExactEventTime, whose Hf exists while its hf
    # does not. Its absence was not a design decision but an omission: the
    # base class writes log_sf and log_ff in terms of Hf, so both raised
    # AttributeError instead of returning the constants they should.
    np.testing.assert_allclose(
        FixedEventProbability.Hf(np.array([1.0, 9.0]), P_BERN),
        -np.log(1 - P_BERN),
    )
    np.testing.assert_allclose(
        FixedEventProbability.log_sf(np.array([1.0, 9.0]), P_BERN),
        np.log(1 - P_BERN),
    )
    np.testing.assert_allclose(
        FixedEventProbability.log_ff(np.array([1.0, 9.0]), P_BERN),
        np.log(P_BERN),
    )


def test_both_models_round_trip_under_their_own_names():
    import surpyval
    from surpyval import FixedEventProbability

    for dist in (Bernoulli, FixedEventProbability):
        model = dist.from_params(P_BERN)
        restored = surpyval.from_dict(model.to_dict())
        assert restored.dist.name == dist.name
        np.testing.assert_allclose(restored.params, model.params)


def test_bernoulli_qf_is_the_standard_quantile():
    from scipy.stats import binom

    # On (0, 1) it is exactly binom.ppf(u, 1, p) -- the smallest outcome
    # k with P(X <= k) >= u.
    u = np.array([0.01, 0.1, 0.5, 0.699, 0.7, 0.701, 0.9, 0.99])
    np.testing.assert_allclose(
        np.asarray(Bernoulli.qf(u, P_BERN), dtype=float),
        binom.ppf(u, 1, P_BERN),
    )
    np.testing.assert_allclose(
        np.asarray(Bernoulli.qf(u, P_BERN), dtype=float),
        np.asarray(Binomial.qf(u, 1, P_BERN), dtype=float),
    )
    # It steps at 1 - p, not at p.
    assert float(np.ravel(Bernoulli.qf(1 - P_BERN, P_BERN))[0]) == 0.0
    assert float(np.ravel(Bernoulli.qf(1 - P_BERN + 1e-9, P_BERN))[0]) == 1.0
    # At u = 0 scipy answers -1, one below the support; this answers 0.
    assert float(np.ravel(Bernoulli.qf(0.0, P_BERN))[0]) == 0.0


def test_bernoulli_qf_drives_inverse_transform_sampling():
    # The property that makes qf worth having: qf(U) reproduces the
    # distribution, which is what ParametricFitter.random does.
    rng = np.random.default_rng(0)
    U = rng.uniform(size=200_000)
    draws = np.asarray(Bernoulli.qf(U, P_BERN), dtype=float)
    assert set(np.unique(draws)) <= {0.0, 1.0}
    assert np.isclose(draws.mean(), P_BERN, atol=0.005)


def test_bernoulli_qf_does_not_invert_this_ff_and_says_so():
    # Documented consequence of R(x) = P(X >= x): the failure function is
    # P(X < x), which never exceeds 1 - p on {0, 1}, so the usual
    # discrete check ff(qf(u)) >= u cannot hold once u passes 1 - p. The
    # other discrete distributions, whose R(k) is P(X > k), are fine.
    u = 0.9  # above 1 - p = 0.7
    k = float(np.ravel(Bernoulli.qf(u, P_BERN))[0])
    assert k == 1.0
    assert float(np.ravel(Bernoulli.ff(k, P_BERN))[0]) < u
    # Whereas for a distribution using the package's R(k) = P(X > k):
    k_pois = float(np.ravel(Poisson.qf(u, 3.0))[0])
    assert float(np.ravel(Poisson.ff(k_pois, 3.0))[0]) >= u - 1e-9


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_bernoulli_qf_at_the_degenerate_ends(p):
    u = np.array([0.01, 0.5, 0.99])
    expected = 0.0 if p == 0.0 else 1.0
    np.testing.assert_allclose(
        np.asarray(Bernoulli.qf(u, p), dtype=float), expected
    )


# ---------------------------------------------------------------------------
# Behaviour below the support
# ---------------------------------------------------------------------------

# (distribution, params, first mass point). Every one of these is defined on
# consecutive integers from the third entry upwards; below it there is no
# mass, so the pmf is zero and the survival is one. The closed forms are
# algebraic and do not know that -- left alone they returned a positive
# "probability" (Geometric 0.43 at k = 0), a survival above one
# (BetaGeometric 2.0 at k = -1), a complex number (DiscreteWeibull) or a
# NaN (Poisson, NegativeBinomial).
_DISCRETE_SUPPORTS = [
    (Geometric, (0.3,), 1),
    (DiscreteWeibull, (0.6, 1.4), 1),
    (BetaGeometric, (2.0, 3.0), 1),
    (NegativeBinomial, (3.0, 0.4), 1),
    (Poisson, (2.5,), 0),
    (Binomial, (5.0, 0.3), 0),
]


@pytest.mark.parametrize("dist, params, first", _DISCRETE_SUPPORTS)
def test_below_the_support_is_real_and_finite(dist, params, first):
    below = np.arange(-4.0, float(first))
    for method in ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_df"):
        value = np.asarray(getattr(dist, method)(below, *params))
        assert np.isrealobj(value), f"{dist.name}.{method} returned complex"
        assert not np.isnan(
            np.asarray(value, dtype=float)
        ).any(), f"{dist.name}.{method} returned NaN below its support"


@pytest.mark.parametrize("dist, params, first", _DISCRETE_SUPPORTS)
def test_no_mass_below_the_support(dist, params, first):
    below = np.arange(-4.0, float(first))
    as_float = lambda v: np.asarray(v, dtype=float)  # noqa: E731
    np.testing.assert_allclose(as_float(dist.df(below, *params)), 0.0)
    np.testing.assert_allclose(as_float(dist.hf(below, *params)), 0.0)
    np.testing.assert_allclose(as_float(dist.sf(below, *params)), 1.0)
    np.testing.assert_allclose(as_float(dist.ff(below, *params)), 0.0)
    np.testing.assert_allclose(as_float(dist.Hf(below, *params)), 0.0)


@pytest.mark.parametrize("dist, params, first", _DISCRETE_SUPPORTS)
def test_the_pmf_sums_to_one_over_the_support(dist, params, first):
    # Summed from below the first mass point, so any spurious mass there
    # would push the total past one. BetaGeometric's tail decays as k^-a,
    # hence the looser tolerance rather than a longer sum.
    k = np.arange(-4.0, 4000.0)
    total = float(np.sum(np.asarray(dist.df(k, *params), dtype=float)))
    assert total == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("dist, params, first", _DISCRETE_SUPPORTS)
def test_the_quantile_inverts_the_cdf_on_the_support(dist, params, first):
    # u = F(k) is formed by cancellation, so it lands a few ulp off the
    # value the quantile is looking for. Geometric, DiscreteWeibull and
    # BetaGeometric all answered k + 1 for a u that came straight out of
    # their own ff.
    last = 5 if dist is Binomial else 11
    k = np.arange(float(first), float(last) + 1.0)
    inverted = np.asarray(dist.qf(dist.ff(k, *params), *params), dtype=float)
    np.testing.assert_array_equal(inverted, k)


def test_beta_geometric_moment_diverges_when_the_tail_is_too_heavy():
    # R(k) decays as k^-a, so E[T^m] exists only for a > m. A truncated
    # sum cannot see that: it reported about 25 for a second moment that
    # is infinite.
    assert BetaGeometric.moment(2, 2.0, 3.0) == np.inf
    assert BetaGeometric.mean(1.0, 3.0) == np.inf
    assert np.isfinite(BetaGeometric.moment(2, 4.0, 3.0))


def test_beta_geometric_mean_agrees_with_its_first_moment():
    for a, b in [(2.0, 3.0), (4.0, 1.5), (3.0, 7.0)]:
        assert BetaGeometric.moment(1, a, b) == pytest.approx(
            BetaGeometric.mean(a, b)
        )
