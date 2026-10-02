import numpy as np
import pytest

import surpyval as surv
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
            for param in dist.parameter_names:
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


def test_fixed_with_free_only_init():
    # An initial guess covering only the free parameters is merged with
    # the fixed values. This used to work only when the fixed parameters
    # came last in the parameter vector (silent zip truncation), and
    # crashed once the transforms became strict.
    x = [87.0, 100.0]
    c = [0, 1]
    n = [1, 9]
    model = Weibull.fit(x, c, n, fixed={"beta": 1.3776}, init=[100.0])
    assert model.params[1] == 1.3776

    np.random.seed(3)
    x = Weibull.random(100, 10, 2)
    model = Weibull.fit(x, fixed={"alpha": 10.0}, init=[2.0])
    assert model.params[0] == 10.0
    # And a full-length init still works
    model = Weibull.fit(x, fixed={"alpha": 10.0}, init=[10.0, 2.0])
    assert model.params[0] == 10.0


def test_mpp_fixed_raises():
    # MPP cannot honour fixed parameters; it used to silently ignore
    # them and return a fully estimated model
    x = Weibull.random(100, 10, 2)
    with pytest.raises(ValueError, match="MPP"):
        Weibull.fit(x, how="MPP", fixed={"beta": 2.0})


# ---------------------------------------------------------------------------
# ``fixed`` and ``init`` are validated.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(fixed={"shape": 1}),
            r"Unknown parameter\(s\) \['shape'\] in `fixed`",
        ),
        (dict(fixed={"p": 0.5}), "needs lfp=True"),
        (dict(fixed={"gamma": 0.5}), "needs offset=True"),
        (dict(fixed={"alpha": -1}), "Cannot fix alpha"),
        (dict(lfp=True, fixed={"p": 1.5}), "Cannot fix p"),
        (dict(offset=True, fixed={"gamma": 1.5}), "Cannot fix gamma"),
        (dict(init=[1.0]), "`init` has 1 value"),
        (dict(init=[-1.0, 2.0]), "Bad `init`: alpha"),
    ],
)
def test_fixed_and_init_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        W.fit([1.0, 2, 3, 4, 5], **kwargs)


def test_init_for_the_free_parameters_only_still_works():
    model = W.fit([1.0, 2, 3, 4, 5], fixed={"alpha": 3.0}, init=[2.0])
    assert model.params[0] == 3.0
