import numpy as np
import pytest

from surpyval.degradation import (
    PATH_MODELS,
    ExponentialPath,
    GompertzPath,
    LinearPath,
    LloydLipowPath,
    LogarithmicPath,
    MichaelisMentenPath,
    OffsetExponentialPath,
    PathModel,
    PowerPath,
    QuadraticPath,
    get_path_model,
)
from surpyval.degradation.path_models import path_model_key

MODELS_AND_PARAMS = [
    (LinearPath, (2.0, 0.5)),
    (LinearPath, (10.0, -0.3)),
    (QuadraticPath, (1.0, 0.3, 0.01)),
    (QuadraticPath, (20.0, -0.5, -0.02)),
    (ExponentialPath, (2.0, 0.05)),
    (ExponentialPath, (5.0, -0.1)),
    (OffsetExponentialPath, (10.0, -8.0, -0.2)),
    (PowerPath, (2.0, 0.8)),
    (LogarithmicPath, (1.0, 2.0)),
    (LloydLipowPath, (10.0, 5.0)),
    (GompertzPath, (10.0, 3.0, 0.3)),
    (MichaelisMentenPath, (10.0, 5.0)),
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
    assert analytic.shape == (len(x), len(model.parameter_names))
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
        (GompertzPath, [1, 2, 3], [1.0, -1.0, 2.0]),
        (MichaelisMentenPath, [0, 1, 2], [1.0, 2.0, 3.0]),
        (MichaelisMentenPath, [1, 2, 3], [1.0, 0.0, 3.0]),
    ],
)
def test_domain_validation(model, x, y):
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_quadratic_inv_path_takes_first_crossing():
    # downward parabola y = x - 0.01 x^2 crosses 16 at t=20 and t=80
    assert np.isclose(QuadraticPath.inv_path(16.0, 0.0, 1.0, -0.01), 20.0)
    # the vertex peaks at 25, so 30 is never reached
    t = QuadraticPath.inv_path(30.0, 0.0, 1.0, -0.01)
    assert not (np.isfinite(t) and t > 0)
    # degenerate c = 0 falls back to the linear crossing
    assert np.isclose(QuadraticPath.inv_path(6.0, 2.0, 0.5, 0.0), 8.0)


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


# ---------------------------------------------------------------------------
# Path-model lookup by display name.
# ---------------------------------------------------------------------------


def test_get_path_model_accepts_display_names():
    for key, model in PATH_MODELS.items():
        assert get_path_model(model.name) is model
        assert get_path_model(model.name.upper()) is model
        assert path_model_key(model) == key
    with pytest.raises(ValueError):
        get_path_model("Offset  Exponential")


# ---------------------------------------------------------------------------
# Quadratic ``inv_path`` is stable for near-zero curvature.
# ---------------------------------------------------------------------------


def test_quadratic_inv_path_near_zero_curvature() -> None:
    x = np.arange(1.0, 11.0)
    params = QuadraticPath.fit(x, 2 + 3 * x)
    assert QuadraticPath.inv_path(50.0, *params) == pytest.approx(16.0)
    for c in (1e-17, -1e-17, 1e-10):
        assert QuadraticPath.inv_path(100.0, 0.0, 1.0, c) == pytest.approx(
            100.0, rel=1e-6
        )
    # genuine roots are unchanged
    assert QuadraticPath.inv_path(100.0, 0.0, -1.0, 1.0) == pytest.approx(
        (1 + np.sqrt(401)) / 2
    )


# #621: the offset exponential start that bends against nearly straight
# measurements runs to the straight-line limit (b -> inf, c -> 0); it is
# stopped there, and the answer is the full searches'.


def _straight_units(n_units=20):
    rng = np.random.default_rng(0)
    t = np.arange(1, 9, dtype=float)
    return t, [
        rng.lognormal(0, 0.3) * t + rng.normal(0, 0.1, t.size)
        for _ in range(n_units)
    ]


def _full_searches(t, y):
    # Every start searched to the end, the better kept (the fit before)
    from scipy.optimize import curve_fit

    best, best_rss = None, np.inf
    for p0 in OffsetExponentialPath._initial_guess(t, y):
        try:
            params, _ = curve_fit(
                OffsetExponentialPath.path, t, y, p0=p0, maxfev=10_000
            )
        except RuntimeError:
            continue
        rss = float(np.sum((y - OffsetExponentialPath.path(t, *params)) ** 2))
        if rss < best_rss:
            best, best_rss = params, rss
    return best


def test_621_offset_exponential_stops_the_search_running_to_the_line(
    monkeypatch,
):
    t, units = _straight_units()
    expected = [_full_searches(t, y) for y in units]
    calls = [0]
    path = OffsetExponentialPath.path

    def counted(*args):
        calls[0] += 1
        return path(*args)

    monkeypatch.setattr(OffsetExponentialPath, "path", counted)
    for y, want in zip(units, expected):
        np.testing.assert_array_equal(OffsetExponentialPath.fit(t, y), want)
    # 2,500 evaluations a unit for the start running to the line, and
    # 200 to 700 for the other, were 64,000 for these 20 units (now 26,000)
    assert calls[0] < 30_000
