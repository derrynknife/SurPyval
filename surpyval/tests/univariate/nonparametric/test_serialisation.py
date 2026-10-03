"""Serialisation of the non-parametric models: ``to_dict`` / ``from_dict`` /
JSON round trips, keeping the Turnbull estimator and bootstrap settings.
"""

import json

import numpy as np
import pytest

import surpyval
from surpyval.tests._helpers import sharp_drop_long_tail_data

# --- serialization ---------------------------------------------------------


def _models():
    x = sharp_drop_long_tail_data()
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
        sharp_drop_long_tail_data(), c=np.array([0, 0, 1, 0, 0, 1, 0, 0])
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
    cb = restored.bootstrap_cb([2.0, 4.0, 6.0], n_boot=20, random_state=0)
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
