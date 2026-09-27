"""Serialisation round trip (Conventions, "Saving and Loading Models").

``to_dict`` gives strict JSON (``json.dumps(..., allow_nan=False)``
succeeds: a non-finite value is written as ``null`` and recorded under
``"non_finite"``), stamped with the oldest ``"schema"`` that reads it;
``surpyval.from_dict`` restores a model of the same class that predicts
exactly what the fitted one did, for every function the registry calls.
"""

import json

import numpy as np
import pytest

import surpyval
from surpyval.serialisation import NON_FINITE_KEY, SCHEMA_VERSION
from surpyval.tests.conformance.registry import (
    cases_for,
    fitted,
    predictions,
)


def _has_key(obj, key):
    if isinstance(obj, dict):
        return key in obj or any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(v, key) for v in obj)
    return False


@pytest.mark.parametrize("case", cases_for("serialise"))
def test_strict_json_round_trip(case):
    model = fitted(case)
    d = model.to_dict()
    text = json.dumps(d, allow_nan=False)
    schema = d["schema"]
    assert isinstance(schema, int) and 1 <= schema <= SCHEMA_VERSION
    if _has_key(d, NON_FINITE_KEY):
        # A reader of an older schema would take the nulls for missing
        # entries, so such a document must be stamped with the newest.
        assert schema == SCHEMA_VERSION

    restored = surpyval.from_dict(json.loads(text))
    expected = model if isinstance(model, type) else type(model)
    got = restored if isinstance(restored, type) else type(restored)
    assert got is expected

    ref = predictions(case, model)
    new = predictions(case, restored)
    for key in ref:
        np.testing.assert_array_equal(new[key], ref[key], err_msg=key)
