import numpy as np
import pytest

from surpyval.degradation import (
    PATH_MODELS,
    ExponentialPath,
    LinearPath,
    LloydLipowPath,
    LogarithmicPath,
    PathModel,
    PowerPath,
    get_path_model,
)

MODELS_AND_PARAMS = [
    (LinearPath, (2.0, 0.5)),
    (LinearPath, (10.0, -0.3)),
    (ExponentialPath, (2.0, 0.05)),
    (ExponentialPath, (5.0, -0.1)),
    (PowerPath, (2.0, 0.8)),
    (LogarithmicPath, (1.0, 2.0)),
    (LloydLipowPath, (10.0, 5.0)),
]


@pytest.mark.parametrize("model,params", MODELS_AND_PARAMS)
def test_fit_recovers_exact_parameters(model, params):
    x = np.linspace(1, 10, 20)
    y = model.path(x, *params)
    fitted = model.fit(x, y)
    assert np.allclose(fitted, params, rtol=1e-5)


@pytest.mark.parametrize("model,params", MODELS_AND_PARAMS)
def test_inv_path_round_trip(model, params):
    x = np.linspace(1, 10, 20)
    y = model.path(x, *params)
    # a level strictly inside the observed range of the path
    level = 0.25 * y.min() + 0.75 * y.max()
    t = model.inv_path(level, *params)
    assert np.isfinite(t)
    assert t > 0
    assert np.allclose(model.path(t, *params), level)


@pytest.mark.parametrize("model,params", MODELS_AND_PARAMS)
def test_analytic_jacobian_matches_finite_differences(model, params):
    x = np.linspace(1, 10, 20)
    analytic = model.jacobian(x, *params)
    # invoke the base class' finite-difference implementation directly
    numeric = PathModel.jacobian(model, x, *params)
    assert analytic.shape == (len(x), len(model.param_names))
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_fit_with_noise_is_close():
    rng = np.random.default_rng(42)
    x = np.linspace(1, 10, 50)
    y = ExponentialPath.path(x, 2.0, 0.2) + rng.normal(0, 0.05, 50)
    fitted = ExponentialPath.fit(x, y)
    assert np.allclose(fitted, [2.0, 0.2], rtol=0.05)


def test_unreachable_levels_are_not_positive_finite():
    # increasing linear path never drops below its intercept
    t = LinearPath.inv_path(1.0, 2.0, 0.5)
    assert not (np.isfinite(t) and t > 0)
    # decaying exponential path never rises above its start
    t = ExponentialPath.inv_path(10.0, 5.0, -0.1)
    assert not (np.isfinite(t) and t > 0)
    # constant path never moves at all
    t = LinearPath.inv_path(5.0, 2.0, 0.0)
    assert not (np.isfinite(t) and t > 0)
    # Lloyd-Lipow path never exceeds its asymptote a
    t = LloydLipowPath.inv_path(11.0, 10.0, 5.0)
    assert not (np.isfinite(t) and t > 0)


@pytest.mark.parametrize(
    "model,x,y",
    [
        (ExponentialPath, [1, 2, 3], [1.0, -1.0, 2.0]),
        (PowerPath, [0, 1, 2], [1.0, 2.0, 3.0]),
        (PowerPath, [1, 2, 3], [1.0, 0.0, 3.0]),
        (LogarithmicPath, [-1, 1, 2], [1.0, 2.0, 3.0]),
        (LloydLipowPath, [0, 1, 2], [1.0, 2.0, 3.0]),
    ],
)
def test_domain_validation(model, x, y):
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_get_path_model():
    assert get_path_model("linear") is LinearPath
    assert get_path_model("Lloyd-Lipow") is LloydLipowPath
    assert get_path_model(PowerPath) is PowerPath
    with pytest.raises(ValueError):
        get_path_model("not-a-model")
    with pytest.raises(ValueError):
        get_path_model(1)


def test_registry_is_complete():
    for name, model in PATH_MODELS.items():
        assert isinstance(model, PathModel)
        assert get_path_model(name) is model
