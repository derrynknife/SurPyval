"""Cause labels of mixed types and tuples: their order, and a
round trip through JSON in every competing-risks model.
"""

import json

import numpy as np
import pytest

import surpyval
from surpyval import Exponential
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)
from surpyval.univariate.competing_risks.labels import ordered_labels


def test_ordered_labels_helper():
    assert ordered_labels([2, None, 1, 10]) == [1, 2, 10]
    assert ordered_labels(["b", 1, None, "a"]) == [1, "a", "b"]


X8 = [1, 2, 3, 4, 5, 6, 7, 8]
Z8 = [[0], [1], [0], [1], [1], [0], [0], [1]]
LABELS = {
    "mixed": [1, "b", None, 1, "b", 1, "b", 1],
    "tuple": [("a", 1), ("b", 2), None, ("a", 1)] * 2,
}


def _round_trip(model):
    return surpyval.from_dict(json.loads(json.dumps(model.to_dict())))


@pytest.mark.parametrize("name", LABELS)
def test_nonparametric_labels(name):
    e = LABELS[name]
    model = CompetingRisks.fit(X8, e)
    restored = _round_trip(model)
    assert list(restored.event_idx_map) == list(model.event_idx_map)
    for k in model.event_idx_map:
        np.testing.assert_allclose(restored.cif(X8, k), model.cif(X8, k))


@pytest.mark.parametrize("name", LABELS)
def test_parametric_labels(name):
    e = LABELS[name]
    model = ParametricCompetingRisks.fit(X8, e, dist=Exponential)
    restored = _round_trip(model)
    assert restored.causes == model.causes
    for k in model.causes:
        np.testing.assert_allclose(restored.cif(X8, k), model.cif(X8, k))


@pytest.mark.parametrize("name", LABELS)
@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_labels(name, how):
    e = LABELS[name]
    model = CompetingRisksProportionalHazards.fit(X8, Z8, e, model=how)
    restored = _round_trip(model)
    for k in model.event_idx_map:
        np.testing.assert_allclose(
            restored.cif(X8, [1], k), model.cif(X8, [1], k)
        )


def test_fine_gray_tuple_cause_round_trips():
    e = LABELS["tuple"]
    model = FineGray.fit(X8, Z8, e, event=("a", 1))
    restored = _round_trip(model)
    assert restored.cause == ("a", 1)
    np.testing.assert_allclose(restored.cif(X8, [1]), model.cif(X8, [1]))
