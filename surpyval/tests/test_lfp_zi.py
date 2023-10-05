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
    for dist in [Beta, Weibull, LogNormal, LogLogistic, Gamma]:
        for zeros in [1, 10, 100, 1000, 10000]:
            x = dist.random(100, 10, 2)
            x = np.concatenate((x, np.zeros(zeros)))
            model = dist.fit(x, zi=True)
            assert model.res.success


def test_lfp():
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


def test_lfp_zi():
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
