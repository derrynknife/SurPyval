"""Serialisation round trip (Conventions, "Saving and Loading Models").

``to_dict`` gives strict JSON (``json.dumps(..., allow_nan=False)``
succeeds: a non-finite value is written as ``null`` and recorded under
``"non_finite"``), stamped with the oldest ``"schema"`` that reads it;
``surpyval.from_dict`` restores a model of the same class that predicts
exactly what the fitted one did, for every function the registry calls.
"""

import pytest

from surpyval.tests.conformance.checks import check_round_trip
from surpyval.tests.conformance.registry import cases_for, fitted


@pytest.mark.parametrize("case", cases_for("serialise"))
def test_strict_json_round_trip(case):
    check_round_trip(case, fitted(case))
