import numpy as np
import pytest

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
        for method in ["MLE", "MSE", "MPS", "MOM"]:
            for param in dist.param_names:
                x = dist.random(100, 10, 2)
                fixed_value = np.random.randint(2, 10)
                model = dist.fit(x, fixed={param: fixed_value}, how=method)
                if not model.params[dist.param_map[param]] == fixed_value:
                    raise ValueError(model.params, fixed_value)


def test_fixed_gamma():
    np.random.seed(1)
    x = Weibull.random(1000, 10, 2) + 10
    for method in ["MLE", "MSE", "MPS"]:
        model = Weibull.fit(x, offset=True, how=method, fixed={"gamma": 10.0})
        assert model.gamma == 10.0
        assert np.allclose(model.params, [10, 2], rtol=0.1)


def test_fixed_all_params():
    np.random.seed(2)
    x = Weibull.random(100, 10, 2)
    model = Weibull.fit(x, fixed={"alpha": 10.0, "beta": 2.0})
    assert np.allclose(model.params, [10.0, 2.0])
    # Nothing is estimated, so nothing carries variance
    assert np.all(model.hess_inv == 0)
    assert np.all(model.cov_matrix == 0)


def test_mpp_fixed_raises():
    # MPP cannot honour fixed parameters; it used to silently ignore
    # them and return a fully estimated model
    x = Weibull.random(100, 10, 2)
    with pytest.raises(ValueError, match="MPP"):
        Weibull.fit(x, how="MPP", fixed={"beta": 2.0})
