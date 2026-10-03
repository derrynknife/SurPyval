"""Serialisation round trips (Conventions, "Saving and Loading Models").

``to_dict`` gives strict JSON (``json.dumps(..., allow_nan=False)``
succeeds: a non-finite value is written as ``null`` and recorded under
``"non_finite"``), stamped with the oldest ``"schema"`` that reads it;
``surpyval.from_dict`` restores a model of the same class that predicts
exactly what the fitted one did, for every function the registry calls.

A fitted model also survives ``pickle``: ``multiprocessing``,
``concurrent.futures`` process pools, joblib, Dask and Ray all move
objects between processes that way, and so do the packages built on
surpyval (RePyability's ``n_jobs``, derrynknife/RePyability#181). A
closure kept on the model, or a class whose module name is rebound to
its singleton instance, breaks it (#573).
"""

import pickle

import numpy as np
import pytest

from surpyval.tests.conformance.checks import check_round_trip
from surpyval.tests.conformance.registry import (
    cases_for,
    fitted,
    predictions,
)


@pytest.mark.parametrize("case", cases_for("serialise"))
def test_strict_json_round_trip(case):
    check_round_trip(case, fitted(case))


@pytest.mark.parametrize("case", cases_for("pickle"))
def test_pickle_round_trip(case):
    model = fitted(case)
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is type(model)
    ref = predictions(case, model)
    new = predictions(case, restored)
    for key in ref:
        np.testing.assert_array_equal(new[key], ref[key], err_msg=key)
