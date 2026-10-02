"""A REML between-unit covariance on the boundary warns.

``population_method="reml"`` could return a singular between-unit
covariance of the path parameters -- an intercept-slope correlation of
0.999998 on six units whose slopes hardly vary -- in silence, while the
moments method warned on the same data that its estimate had to be
clipped. REML now warns when its estimate is on the boundary of the
positive semi-definite cone: when the covariance with its smallest
eigenvalue removed fits at least as well (``population._on_boundary``).
"""

import warnings

import numpy as np
import pytest

from surpyval.degradation import DegradationAnalysis

T = np.arange(0.0, 650.0, 50.0)


def _units(seed, n=6):
    # Linear paths, 13 readings per unit; slopes vary little between
    # units (sd 0.002 against a measurement noise of 1).
    rng = np.random.default_rng(seed)
    a = rng.normal(10.0, 1.5, n)
    b = rng.normal(0.1, 0.002, n)
    x = np.tile(T, n)
    i = np.repeat(np.arange(n), T.size)
    y = np.repeat(a, T.size) + np.repeat(b, T.size) * x
    return x, y + rng.normal(0, 1, x.size), i


def _fit(x, y, i, method):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = DegradationAnalysis.fit(
            x, y, i, threshold=100.0, population_method=method
        )
    return model, caught


def _correlation(cov):
    return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_reml_on_the_boundary_warns_once(scale):
    x, y, i = _units(1)
    model, caught = _fit(x * scale, y, i, "reml")
    assert _correlation(model.path_param_cov) == pytest.approx(1, abs=1e-5)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith(
        "The REML estimate of the between-unit covariance of the path "
        "parameters (path_param_cov) is singular, on the boundary"
    )
    assert "correlation of +-1" in message
    assert "unreliable" in message and "More units" in message
    assert caught[0].filename == __file__


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_reml_inside_the_cone_is_quiet(scale):
    x, y, i = _units(0)
    model, caught = _fit(x * scale, y, i, "reml")
    assert abs(_correlation(model.path_param_cov)) < 0.5
    assert caught == []


def test_moments_warning_is_unchanged_in_substance():
    x, y, i = _units(1)
    model, caught = _fit(x, y, i, "moments")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith(
        "The noise-corrected between-unit covariance of the path "
        "parameters was not positive semi-definite"
    )
    assert "clipped to zero" in message
    assert "correlation of +-1" in message
    assert "population_method='reml'" in message
    assert "may also land on the boundary" in message
    assert caught[0].filename == __file__
    _, caught = _fit(*_units(0), "moments")
    assert caught == []


# ---------------------------------------------------------------------------
# A REML clock fit on noise-free data is refused early.
# ---------------------------------------------------------------------------


def test_reml_clock_on_noise_free_data_refused_early() -> None:
    levels = np.array([0.0, 0.5, 1.0])
    times = np.arange(1.0, 31.0)
    zrow = np.select([times <= 10, times <= 20], levels[:2], levels[2])
    af = np.exp(2.0 * zrow)
    rng = np.random.default_rng(0)
    xs, ys, ids, Zs = [], [], [], []
    for u in range(6):
        a, b = rng.normal([1.0, 0.2], [0.2, 0.03])
        tau = np.cumsum(np.diff(np.r_[0.0, times]) * af)
        xs.append(times)
        ys.append(a + b * tau)
        ids.append(np.full(times.size, u))
        Zs.append(zrow)
    x, y, i, Z = map(np.concatenate, (xs, ys, ids, Zs))
    for noise in (0.0, 1e-8):
        yy = y + noise * rng.normal(size=y.size)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError, match="population_method='moments'"
            ):
                DegradationAnalysis.fit(
                    x,
                    yy,
                    i,
                    threshold=20.0,
                    Z=Z,
                    acceleration="clock",
                    stress_ref=[0.0],
                    population_method="reml",
                )
    # moments recovers the clock exactly
    model = DegradationAnalysis.fit(
        x, y, i, threshold=20.0, Z=Z, acceleration="clock", stress_ref=[0.0]
    )
    assert model.gamma is not None
    assert model.gamma[0] == pytest.approx(2.0)
