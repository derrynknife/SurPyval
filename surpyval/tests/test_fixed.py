import numpy as np

from surpyval import (
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Weibull,
)


def test_fixed():
    for dist in [
        Gamma,
        Gumbel,
        GumbelLEV,
        Weibull,
        LogNormal,
        Logistic,
        LogLogistic,
        Normal,
    ]:
        for method in ["MLE", "MSE", "MPS"]:
            for param in dist.param_names:
                x = dist.random(100, 10, 2)
                fixed_value = np.random.randint(2, 10)
                model = dist.fit(x, fixed={param: fixed_value}, how=method)
                if not model.params[dist.param_map[param]] == fixed_value:
                    raise ValueError(model.params, fixed_value)
