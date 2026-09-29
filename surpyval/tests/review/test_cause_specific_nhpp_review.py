"""Targeted review of
``recurrent/competing_risks/parametric/cause_specific_nhpp.py`` (#399).

Each test pins a bug found by reading the module adversarially (#440,
fixed): the recurrent cause-specific models now handle cause labels
through ``univariate/competing_risks/labels.py`` (``ordered_labels``,
``label_mask``, ``label_from_native``), as the univariate ones do.
"""

import json

import pytest

import surpyval as surv
from surpyval.recurrent import HPP, CauseSpecificMCF, CauseSpecificNHPP

X = [2, 5, 7, 10, 3, 4, 8, 12]
I = [1, 1, 1, 1, 2, 2, 2, 2]
C = [0, 0, 0, 1, 0, 0, 0, 1]


def _fit(kind, e):
    if kind == "NHPP":
        return CauseSpecificNHPP.fit(X, i=I, c=C, e=e, dist=HPP)
    return CauseSpecificMCF.fit(X, i=I, c=C, e=e)


def _cif(model, x, cause):
    if isinstance(model, CauseSpecificNHPP):
        return model.cif(x, cause)
    return model.mcf(x, cause)


@pytest.mark.parametrize("kind", ["NHPP", "MCF"])
def test_tuple_cause_labels_round_trip(kind):
    # #440: from_dict raised TypeError "unhashable type: 'list'" (JSON
    # writes the tuple as a list).
    seal, motor = ("s", 1), ("m", 2)
    e = [seal, motor, seal, None, seal, seal, motor, None]
    model = _fit(kind, e)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert _cif(restored, [10.0], seal) == pytest.approx(
        _cif(model, [10.0], seal)
    )


@pytest.mark.parametrize("kind", ["NHPP", "MCF"])
def test_mixed_cause_labels_fit(kind):
    # #440: the fit raised a bare TypeError from sorting 's' and 2.
    e = ["s", 2, "s", None, "s", "s", 2, None]
    model = _fit(kind, e)
    assert set(model.event_types) == {"s", 2}


@pytest.mark.parametrize("kind", ["NHPP", "MCF"])
def test_tuple_cause_labels_on_every_row(kind):
    # #440: with a tuple mark on every row (no None row to stop it),
    # np.array split the marks into a (rows, 2) array and the fit raised
    # TypeError "unhashable type: 'numpy.ndarray'".
    seal, motor = ("s", 1), ("m", 2)
    e = [seal, motor, seal, seal, seal, seal, motor, motor]
    model = _fit(kind, e)
    assert model.event_types == [motor, seal]
    # The same fit as with string marks.
    plain = _fit(kind, ["s" if v == seal else "m" for v in e])
    assert _cif(model, [10.0], seal) == pytest.approx(_cif(plain, [10.0], "s"))
