"""
Tests for the Binomial distribution.

The Binomial is a discrete count distribution (number of events in a fixed
number of pass/fail trials) and, like the Bernoulli, sits outside the
gradient-based MLE machinery. These tests check the closed-form identities,
the closed-form fit, the reduction to the Bernoulli at ``n = 1``, input
validation and serialisation.
"""

import numpy as np
import pytest
from scipy.stats import binom

from surpyval import Bernoulli, Binomial, Parametric

N, P = 5, 0.3


def test_from_params_repr_and_params():
    model = Binomial.from_params([N, P])
    assert np.allclose(model.params, [N, P])
    assert "Binomial" in repr(model)


def test_pmf_cdf_sf_match_scipy():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    assert np.allclose(model.df(k), binom.pmf(k, N, P))
    assert np.allclose(model.ff(k), binom.cdf(k, N, P))
    assert np.allclose(model.sf(k), binom.sf(k, N, P))


def test_sf_plus_ff_is_one():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    assert np.allclose(model.sf(k) + model.ff(k), 1.0)


def test_pmf_sums_to_one():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.df(np.arange(0, N + 1)).sum(), 1.0)


def test_mean_var_moment_entropy():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.mean(), N * P)
    assert np.isclose(model.var(), N * P * (1 - P))
    assert np.isclose(model.moment(2), binom.moment(2, N, P))
    assert np.isclose(model.entropy(), binom.entropy(N, P))


def test_hazard_and_cumulative_hazard():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N)
    expected_hf = binom.pmf(k, N, P) / (binom.sf(k, N, P) + binom.pmf(k, N, P))
    assert np.allclose(model.hf(k), expected_hf)
    assert np.allclose(model.Hf(k), -np.log(binom.sf(k, N, P)))


def test_qf_roundtrip():
    model = Binomial.from_params([N, P])
    k = np.arange(0, N + 1)
    # The quantile of the cdf returns a value no smaller than k
    assert np.all(model.qf(binom.cdf(k, N, P)) >= k)


def test_conditional_survival():
    model = Binomial.from_params([N, P])
    assert np.isclose(model.cs(1, 2), model.sf(3) / model.sf(2))


def test_random_moments():
    model = Binomial.from_params([N, P])
    np.random.seed(1)
    samples = model.random(100_000)
    assert np.isclose(samples.mean(), N * P, atol=0.05)
    assert np.isclose(samples.var(), N * P * (1 - P), atol=0.05)
    assert samples.min() >= 0
    assert samples.max() <= N


def test_fit_closed_form():
    # 2 + 3 + 1 + 4 = 10 events out of 4 * 5 = 20 trials -> p = 0.5
    model = Binomial.fit([2, 3, 1, 4], n_trials=5)
    assert np.allclose(model.params, [5, 0.5])
    assert isinstance(model, Parametric)


def test_fit_with_counts():
    # 0 events once, 5 events once -> 5 / (2 * 5) = 0.5
    model = Binomial.fit([0, 5], n_trials=5, n=[1, 1])
    assert np.isclose(model.params[1], 0.5)


def test_reduces_to_bernoulli_at_n_one():
    # At n = 1 the event probabilities match the Bernoulli: P(K=1) = p and
    # P(K=0) = 1 - p. Note surpyval's Bernoulli is a degenerate
    # "fixed event probability" model whose survival is the constant
    # probability of *no* event (1 - p), so it lines up with the binomial's
    # P(K = 0) = ff(0), not its sf(0).
    binomial = Binomial.from_params([1, P])
    bernoulli = Bernoulli.from_params(P)
    assert np.isclose(binomial.df(1), P)
    assert np.isclose(binomial.df(0), 1 - P)
    assert np.isclose(binomial.ff(0), bernoulli.sf(0))
    assert np.isclose(binomial.sf(0), bernoulli.ff(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": [1, 6], "n_trials": 5},  # out of range
        {"x": [1, 2], "n_trials": 5, "c": [0, 1]},  # censoring unsupported
        {"x": [1.5, 2], "n_trials": 5},  # non-integer counts
    ],
)
def test_fit_input_validation(kwargs):
    with pytest.raises(ValueError):
        Binomial.fit(**kwargs)


@pytest.mark.parametrize(
    "params",
    [
        [5.5, 0.3],  # non-integer n
        [0, 0.3],  # n must be positive
        [5, 1.3],  # p out of bounds
        [5],  # wrong number of params
    ],
)
def test_from_params_validation(params):
    with pytest.raises(ValueError):
        Binomial.from_params(params)


def test_to_dict_roundtrip():
    model = Binomial.from_params([N, P])
    restored = Parametric.from_dict(model.to_dict())
    assert np.allclose(restored.params, [N, P])
    assert np.isclose(restored.mean(), N * P)
