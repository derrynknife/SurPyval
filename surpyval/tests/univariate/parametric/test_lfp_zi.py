import numpy as np
import pytest

from surpyval import (
    Beta,
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Weibull,
)


def test_zi():
    np.random.seed(42)
    for dist in [Beta, Weibull, LogNormal, LogLogistic, Gamma]:
        for zeros in [1, 10, 100, 1000, 10000]:
            x = dist.random(100, 10, 2)
            x = np.concatenate((x, np.zeros(zeros)))
            model = dist.fit(x, zi=True)
            assert model.res.success
            # The zero-inflation estimate must match the actual
            # proportion of zeros
            f0_true = zeros / (100 + zeros)
            assert abs(model.f0 - f0_true) < 0.05


def test_lfp():
    np.random.seed(42)
    for dist in [
        Beta,
        Gamma,
        Gumbel,
        GumbelLEV,
        Logistic,
        LogLogistic,
        LogNormal,
        Normal,
        Weibull,
    ]:
        for censored in [1, 10, 100, 1000, 10000]:
            x = dist.random(100, 10, 2)
            c = np.concatenate((np.zeros_like(x), np.ones(censored)))
            x = np.concatenate((x, x.max() * np.ones(censored)))
            model = dist.fit(x, c=c, lfp=True)
            assert model.res.success
            # All failures are observed before the censor time, so the
            # max proportion estimate must match the failing fraction
            p_true = 100 / (100 + censored)
            assert abs(model.p - p_true) < 0.1


def test_lfp_zi():
    np.random.seed(42)
    for dist in [Gamma, Weibull, LogNormal, LogLogistic]:
        for zi_lfp_values in [1, 10, 100]:
            for num_samples in [100, 1000, 10000]:
                x = dist.random(num_samples, 10, 2)
                c = np.concatenate(
                    (
                        np.zeros_like(x),
                        np.zeros(zi_lfp_values),
                        np.ones(zi_lfp_values),
                    )
                )
                x = np.concatenate(
                    (
                        x,
                        np.zeros(zi_lfp_values),
                        x.max() * np.ones(zi_lfp_values) + 1,
                    )
                )
                model = dist.fit(x, c=c, zi=True, lfp=True)
                if not model.res.success:
                    raise ValueError(model, model.res)
                total = num_samples + 2 * zi_lfp_values
                f0_true = zi_lfp_values / total
                p_true = (num_samples + zi_lfp_values) / total
                assert abs(model.f0 - f0_true) < 0.05
                assert abs(model.p - p_true) < 0.1


def test_offset_lfp():
    np.random.seed(1)
    n = 2000
    x = Weibull.random(n, 10, 2) + 10
    c = np.zeros(n)
    never = np.random.uniform(size=n) > 0.7
    x[never] = x.max() + 1
    c[never] = 1

    model = Weibull.fit(x, c=c, lfp=True, offset=True)
    assert model.res.success
    assert abs(model.gamma - 10) < 1
    assert abs(model.p - 0.7) < 0.05


def test_offset_zi():
    # The offset bound and initial guess must come from the nonzero
    # observations; the zeros belong to the zero-inflation mass
    np.random.seed(2)
    x = np.concatenate([Weibull.random(1000, 10, 2) + 10, np.zeros(100)])

    model = Weibull.fit(x, zi=True, offset=True)
    assert model.res.success
    assert abs(model.gamma - 10) < 1
    assert abs(model.f0 - 100 / 1100) < 0.02


# --- quantile function for the mixture (LFP / zero-inflation / offset) -----


def test_qf_inverts_ff_for_lfp():
    # Below the cure ceiling p, the quantile inverts the failure function.
    model = Weibull.from_params([10.0, 2.0], p=0.6)
    u = np.array([0.05, 0.2, 0.4, 0.59])
    q = model.qf(u)
    assert np.all(np.isfinite(q))
    assert np.allclose(model.ff(q), u)


def test_qf_infinite_above_cure_fraction():
    # A cure fraction 1 - p never fails, so any quantile at or above p is
    # infinite -- and the median of a majority-cured population is infinite.
    model = Weibull.from_params([10.0, 2.0], p=0.6)
    assert np.isinf(model.qf(0.6))
    assert np.isinf(model.qf(0.85))
    cured = Weibull.from_params([10.0, 2.0], p=0.4)
    assert np.isinf(cured.qf(0.5))


def test_qf_inverts_ff_for_zero_inflation():
    # The zero-inflation mass f0 sits at the offset (here 0), and above it
    # the quantile inverts the failure function.
    model = LogNormal.from_params([2.0, 0.4], f0=0.2)
    assert model.qf(0.1) == 0.0
    assert model.qf(0.2) == 0.0
    u = np.array([0.3, 0.5, 0.9])
    assert np.allclose(model.ff(model.qf(u)), u)


def test_qf_respects_offset_with_cure_and_inflation():
    # gamma + f0 + p all together: mass below f0 lands on the offset, the
    # interior inverts ff, and u >= p is infinite.
    model = Weibull.from_params([10.0, 2.0], gamma=5.0, p=0.7, f0=0.1)
    assert model.qf(0.05) == 5.0
    u = np.array([0.2, 0.4, 0.6])
    q = model.qf(u)
    assert np.all(q > 5.0)
    assert np.allclose(model.ff(q), u)
    assert np.isinf(model.qf(0.7))


def test_qf_scalar_and_array_shape():
    model = Weibull.from_params([10.0, 2.0], p=0.8)
    assert np.ndim(model.qf(0.3)) == 0
    out = model.qf([0.1, 0.3, 0.5])
    assert out.shape == (3,)


def test_qf_matches_plain_distribution_without_mixture():
    # With no offset, cure or inflation the quantile is exactly the base
    # distribution's, so ordinary models are unchanged.
    model = Weibull.from_params([10.0, 3.0])
    assert np.isclose(model.qf(0.2), Weibull.qf(0.2, 10.0, 3.0))


# --- moment for the mixture (LFP / zero-inflation / offset) ----------------


def test_moment_matches_plain_distribution_without_mixture():
    model = Weibull.from_params([10.0, 3.0])
    assert np.isclose(model.moment(2), Weibull.moment(2, 10.0, 3.0))
    assert np.isclose(model.moment(3), Weibull.moment(3, 10.0, 3.0))


def test_moment_one_equals_mean_across_mixtures():
    # moment(1) must be consistent with mean() for every configuration.
    for model in (
        Weibull.from_params([10.0, 2.0]),
        Weibull.from_params([10.0, 2.0], gamma=5.0),
        Weibull.from_params([10.0, 2.0], p=0.6),
        Weibull.from_params([10.0, 2.0], gamma=5.0, p=0.7),
    ):
        assert np.isclose(model.moment(1), model.mean())


def test_offset_moment_includes_offset():
    # Regression: the offset previously dropped out of moment. E[(gamma+X)^2]
    # is strictly greater than E[X^2].
    base = Weibull.from_params([10.0, 2.0])
    offset = Weibull.from_params([10.0, 2.0], gamma=5.0)
    assert offset.moment(2) > base.moment(2)
    # exact binomial value: E[(g+X)^2] = g^2 + 2 g E[X] + E[X^2]
    g = 5.0
    expected = g**2 + 2 * g * base.moment(1) + base.moment(2)
    assert np.isclose(offset.moment(2), expected)


def test_lfp_moment_is_finite_and_defective():
    # With a cure fraction the defective moment is finite and equals the base
    # moment scaled by the failing proportion p (no offset).
    p = 0.6
    model = Weibull.from_params([10.0, 2.0], p=p)
    assert np.isclose(model.moment(2), p * Weibull.moment(2, 10.0, 2.0))


def test_defective_moment_matches_monte_carlo():
    # cured units contribute nothing; offset shifts the failures.
    g, p, params = 6.0, 0.7, (10.0, 2.0)
    model = Weibull.from_params(list(params), gamma=g, p=p)
    rng = np.random.default_rng(0)
    n = 2_000_000
    fail = rng.uniform(size=n) < p
    t = np.where(fail, g + Weibull.random(n, *params), 0.0)
    assert np.isclose(model.moment(1), t.mean(), rtol=0.02)
    assert np.isclose(model.moment(2), (t**2).mean(), rtol=0.02)


# --- entropy for the mixture ----------------------------------------------


def test_entropy_is_offset_invariant():
    # Differential entropy is translation-invariant, so an offset model has
    # the same entropy as the un-offset one.
    base = Weibull.from_params([10.0, 2.0])
    offset = Weibull.from_params([10.0, 2.0], gamma=5.0)
    assert np.isclose(offset.entropy(), base.entropy())


def test_entropy_raises_with_a_probability_atom():
    # A cure fraction (mass at infinity) or zero-inflation (mass at the
    # offset) leaves no single differential entropy.
    with pytest.raises(ValueError, match="probability atom"):
        Weibull.from_params([10.0, 2.0], p=0.6).entropy()
    with pytest.raises(ValueError, match="probability atom"):
        LogNormal.from_params([2.0, 0.4], f0=0.2).entropy()
