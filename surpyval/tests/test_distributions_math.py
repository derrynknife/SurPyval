"""
Mathematical correctness tests for parametric distributions.

Checks closed-form identities independent of fitting:
  1. sf(x) + ff(x) == 1  at several quantile-derived x values
  2. ff(qf(p)) ≈ p       qf/ff roundtrip across the probability range
  3. qf(0.5)             equals the known closed-form median
  4. random() samples    have mean and variance matching analytical values
"""

import math

import numpy as np
import pytest

from surpyval import (
    Beta,
    ExpoWeibull,
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Weibull,
)

# (distribution, params)
DIST_PARAMS = [
    (Gumbel, (-1.0, 2.0)),
    (GumbelLEV, (3.0, 1.5)),
    (Normal, (5.0, 2.0)),
    (Weibull, (10.0, 2.0)),
    (LogNormal, (1.0, 0.5)),
    (Logistic, (4.0, 1.0)),
    (LogLogistic, (5.0, 2.0)),
    (Beta, (2.0, 5.0)),
    (ExpoWeibull, (3.0, 1.5, 0.8)),
    (Gamma, (3.0, 2.0)),
]

DIST_PARAM_IDS = [d.name for d, _ in DIST_PARAMS]

# Probabilities used for the qf/ff roundtrip test
ROUNDTRIP_PROBS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

DIST_PARAMS_ROUNDTRIP = list(DIST_PARAMS)


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. sf(x) + ff(x) == 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist, params", DIST_PARAMS, ids=DIST_PARAM_IDS)
def test_sf_plus_ff_equals_one(dist, params):
    """sf and ff must be complementary at every point in the support."""
    x_vals = np.array(
        [dist.qf(p, *params) for p in [0.05, 0.25, 0.5, 0.75, 0.95]]
    )
    total = dist.sf(x_vals, *params) + dist.ff(x_vals, *params)
    max_err = np.max(np.abs(total - 1))
    assert np.allclose(
        total, 1.0, atol=1e-12
    ), f"{dist.name}: sf + ff deviates from 1 — max error {max_err}"


# ---------------------------------------------------------------------------
# 2. ff(qf(p)) ≈ p  (qf is the inverse of ff)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist, params", DIST_PARAMS_ROUNDTRIP, ids=DIST_PARAM_IDS
)
def test_qf_ff_roundtrip(dist, params):
    """Applying ff after qf must recover the original probability."""
    p = np.array(ROUNDTRIP_PROBS)
    recovered = dist.ff(dist.qf(p, *params), *params)
    max_err = np.max(np.abs(recovered - p))
    assert np.allclose(
        recovered, p, atol=1e-9
    ), f"{dist.name}: ff(qf(p)) roundtrip failed — max error {max_err}"


# ---------------------------------------------------------------------------
# 3. qf(0.5) matches the closed-form median
# ---------------------------------------------------------------------------

KNOWN_MEDIANS = [
    # (distribution, params, expected_median, description)
    (Normal, (5.0, 2.0), 5.0, "mu"),
    (Logistic, (4.0, 1.0), 4.0, "mu"),
    (LogLogistic, (7.0, 3.0), 7.0, "alpha"),
    (LogNormal, (1.5, 0.5), math.exp(1.5), "exp(mu)"),
    (Weibull, (10.0, 1.0), 10.0 * math.log(2), "alpha * ln(2)"),
    (Beta, (1.0, 1.0), 0.5, "uniform midpoint"),
]

MEDIAN_IDS = [f"{d.name}({desc})" for d, _, _, desc in KNOWN_MEDIANS]


@pytest.mark.parametrize(
    "dist, params, expected, desc", KNOWN_MEDIANS, ids=MEDIAN_IDS
)
def test_qf_median(dist, params, expected, desc):
    """qf(0.5) must equal the analytically known median."""
    computed = dist.qf(0.5, *params)
    assert math.isclose(
        computed, expected, rel_tol=1e-9
    ), f"{dist.name}: qf(0.5) = {computed}, expected {expected} ({desc})"


# ---------------------------------------------------------------------------
# 4. random() samples have the expected mean and variance
# ---------------------------------------------------------------------------

# (distribution, params, theoretical_mean, theoretical_variance)
RANDOM_STATS = [
    # Normal(mu, sigma): mean=mu, var=sigma^2
    (Normal, (5.0, 2.0), 5.0, 4.0),
    # Weibull(alpha, 1) = Exponential: mean=alpha, var=alpha^2
    (Weibull, (10.0, 1.0), 10.0, 100.0),
    # Beta(1,1) = Uniform(0,1): mean=0.5, var=1/12
    (Beta, (1.0, 1.0), 0.5, 1.0 / 12.0),
    # Gamma(alpha, beta) rate-parameterised: mean=alpha/beta, var=alpha/beta^2
    (Gamma, (4.0, 2.0), 2.0, 1.0),
]

RANDOM_STATS_IDS = [d.name for d, *_ in RANDOM_STATS]

N_SAMPLES = 500_000
# Tolerance: 3 standard errors of the estimator at N_SAMPLES
MEAN_REL_TOL = 0.01  # 1% of the true mean
VAR_REL_TOL = 0.02  # 2% of the true variance


@pytest.mark.parametrize(
    "dist, params, true_mean, true_var", RANDOM_STATS, ids=RANDOM_STATS_IDS
)
def test_random_sample_statistics(dist, params, true_mean, true_var):
    """random() samples must match analytical mean and variance."""
    samples = dist.random(N_SAMPLES, *params)
    sample_mean = samples.mean()
    sample_var = samples.var()

    assert math.isclose(
        sample_mean, true_mean, rel_tol=MEAN_REL_TOL
    ), f"{dist.name}: sample mean {sample_mean:.4f} vs expected {true_mean}"
    assert math.isclose(
        sample_var, true_var, rel_tol=VAR_REL_TOL
    ), f"{dist.name}: sample var {sample_var:.4f} vs expected {true_var}"
