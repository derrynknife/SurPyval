import numpy as np

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
