"""Monotone (PCHIP) smooth interpolation, seedable random(), and
to_dict/from_dict/JSON serialization for NonParametric models."""

import json

import numpy as np
import pytest

import surpyval

# --- interp='cubic' must stay a valid survival function (PCHIP) -----------


def _overshooting_data():
    # A sharp drop followed by a long flat tail makes an ordinary cubic
    # spline overshoot below zero.
    return np.array([1.0, 2.0, 3.0, 3.2, 3.4, 8.0, 12.0, 20.0])


def test_cubic_interpolation_is_monotone_and_bounded():
    model = surpyval.KaplanMeier.fit(_overshooting_data())
    grid = np.linspace(model.x.min(), model.x.max(), 500)
    sf = model.sf(grid, interp="cubic")
    finite = sf[np.isfinite(sf)]
    # Non-increasing and inside [0, 1] -- a plain cubic spline breaks both.
    assert np.all(np.diff(finite) <= 1e-12)
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_cubic_interpolation_passes_through_estimates():
    # A shape-preserving interpolant must still interpolate: at the
    # observed times it returns the fitted survival exactly.
    model = surpyval.KaplanMeier.fit(_overshooting_data())
    assert np.allclose(model.sf(model.x, interp="cubic"), model.R, atol=1e-9)


def test_cubic_interpolation_does_not_poison_hazard():
    # A negative interpolated survival used to make Hf = -log(sf) NaN in
    # the interior. With a monotone interpolant Hf is instead a valid,
    # non-decreasing cumulative hazard (with +inf only where the survival
    # estimate legitimately reaches 0 at the last uncensored point).
    model = surpyval.KaplanMeier.fit(_overshooting_data())
    grid = np.linspace(model.x.min(), model.x.max(), 200)
    Hf = model.Hf(grid, interp="cubic")
    assert not np.any(np.isnan(Hf))
    finite = Hf[np.isfinite(Hf)]
    assert np.all(np.diff(finite) >= -1e-9)


def test_cubic_interpolation_with_turnbull_zero_width_bounds():
    # Turnbull's ``x`` carries duplicated (zero-width) bounds at exact
    # times; PCHIP requires strictly increasing abscissae, so the fix must
    # collapse them rather than raise.
    x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10.0]])
    model = surpyval.Turnbull.fit(x, turnbull_estimator="Kaplan-Meier")
    grid = np.linspace(1.0, 10.0, 200)
    sf = model.sf(grid, interp="cubic")
    finite = sf[np.isfinite(sf)]
    assert np.all(np.diff(finite) <= 1e-9)
    assert finite.min() >= -1e-9 and finite.max() <= 1 + 1e-9


# --- random() is seedable --------------------------------------------------


def test_random_is_reproducible_with_seed():
    model = surpyval.KaplanMeier.fit(_overshooting_data())
    a = model.random(500, random_state=42)
    b = model.random(500, random_state=42)
    c = model.random(500, random_state=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_draws_from_observed_values():
    model = surpyval.KaplanMeier.fit(_overshooting_data())
    draws = model.random(1000, random_state=1)
    assert set(np.unique(draws)).issubset(set(model.x))
    assert draws.size == 1000


# --- serialization ---------------------------------------------------------


def _models():
    x = _overshooting_data()
    c = (np.arange(x.size) % 3 == 0).astype(int)
    interval = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10.0]])
    return {
        "kaplan_meier": surpyval.KaplanMeier.fit(x, c=c),
        "nelson_aalen": surpyval.NelsonAalen.fit(x),
        "fleming_harrington": surpyval.FlemingHarrington.fit(x, c=c),
        "turnbull": surpyval.Turnbull.fit(
            interval, turnbull_estimator="Kaplan-Meier"
        ),
    }


@pytest.mark.parametrize("name", list(_models()))
def test_to_dict_from_dict_round_trip(name):
    model = _models()[name]
    restored = surpyval.NonParametric.from_dict(model.to_dict(with_data=True))
    grid = np.linspace(1.5, 9.0, 40)
    assert np.allclose(model.sf(grid), restored.sf(grid), equal_nan=True)
    assert np.allclose(model.cb(grid), restored.cb(grid), equal_nan=True)
    assert np.allclose(model.R, restored.R, equal_nan=True)
    assert restored.model == model.model


def test_to_dict_is_json_serializable_and_round_trips(tmp_path):
    model = surpyval.KaplanMeier.fit(
        _overshooting_data(), c=np.array([0, 0, 1, 0, 0, 1, 0, 0])
    )
    # The plain dict must be JSON-encodable...
    encoded = json.dumps(model.to_dict())
    assert isinstance(encoded, str)
    # ...and the file round-trip must reproduce the survival estimate.
    path = tmp_path / "model.json"
    model.to_json(path)
    restored = surpyval.NonParametric.from_json(path)
    grid = np.linspace(1.5, 9.0, 40)
    assert np.allclose(model.sf(grid), restored.sf(grid), equal_nan=True)


def test_serialized_turnbull_keeps_estimator_and_bootstrap():
    model = surpyval.Turnbull.fit(
        np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10.0]]),
        turnbull_estimator="Kaplan-Meier",
    )
    restored = surpyval.NonParametric.from_dict(model.to_dict(with_data=True))
    assert restored.data["estimator"] == "Kaplan-Meier"
    # The stored data lets the restored model still bootstrap.
    cb = restored.bootstrap_cb([2.0, 4.0, 6.0], B=20, random_state=0)
    assert cb.shape == (3, 2)


def test_from_dict_rejects_wrong_parameterization():
    with pytest.raises(ValueError, match="non-parametric"):
        surpyval.NonParametric.from_dict({"parameterization": "parametric"})


def test_fit_from_ecdf_serialization_preserves_missing_variance():
    # No r/d/greenwood -> confidence bounds must stay unavailable after a
    # round trip.
    model = surpyval.NonParametric.fit_from_ecdf(
        np.array([1.0, 2.0, 3.0]), np.array([0.9, 0.6, 0.3])
    )
    restored = surpyval.NonParametric.from_dict(model.to_dict())
    assert restored.greenwood is None
    assert np.allclose(model.sf([1.5, 2.5]), restored.sf([1.5, 2.5]))
    with pytest.raises(ValueError, match="no variance estimate"):
        restored.cb([1.5])
