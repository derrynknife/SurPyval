"""Targeted review of
``recurrent/competing_risks/parametric/cause_specific_nhpp.py`` (#399).

Each test pins a bug found by reading the module adversarially. They are
strict expected failures until the bug is fixed. The univariate
competing-risks models handle both label cases through
``univariate/competing_risks/labels.py`` (``ordered_labels``,
``label_from_native``); the recurrent ones do not use it.
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


@pytest.mark.xfail(
    strict=True,
    reason="#440: a recurrent cause-specific model (NHPP or MCF) with tuple "
    "cause labels cannot be read back: from_dict raises TypeError "
    "'unhashable type: list' (the univariate models restore the tuple)",
)
@pytest.mark.parametrize("kind", ["NHPP", "MCF"])
def test_tuple_cause_labels_round_trip(kind):
    seal, motor = ("s", 1), ("m", 2)
    e = [seal, motor, seal, None, seal, seal, motor, None]
    model = _fit(kind, e)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert _cif(restored, [10.0], seal) == pytest.approx(
        _cif(model, [10.0], seal)
    )


@pytest.mark.xfail(
    strict=True,
    reason="#440: a recurrent cause-specific fit (NHPP or MCF) with mixed "
    "cause labels ('s' and 2) raises a bare TypeError from sorting them "
    "(the univariate models order them with ordered_labels)",
)
@pytest.mark.parametrize("kind", ["NHPP", "MCF"])
def test_mixed_cause_labels_fit(kind):
    e = ["s", 2, "s", None, "s", "s", 2, None]
    model = _fit(kind, e)
    assert set(model.event_types) == {"s", 2}
