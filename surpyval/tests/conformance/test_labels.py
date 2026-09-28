"""Cause labels (principle 20, and Conventions, "Competing Risks").

A cause (or event-type mark) may be any hashable value, not only a
string: a tuple such as ``("seal", 1)``, or a mixture of strings and
integers. For every registered model fitted with marks ``e``, relabelling
the causes -- as tuples, and as a mixture of ``str`` and ``int`` -- must
fit, give each cause the predictions it had under its original label,
and survive a strict-JSON ``to_dict`` / ``from_dict`` round trip with
identical predictions (which needs the tuple, written as a JSON list,
turned back into a tuple).

A case with marks but no per-cause functions (Fine-Gray) models one cause
of interest, ``"a"``; it is refitted with that cause relabelled.
"""

import functools
from dataclasses import replace

import numpy as np
import pytest

from surpyval.tests.conformance.checks import check_round_trip
from surpyval.tests.conformance.leaks import quiet
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    CASES,
    call,
    fitted,
)

# Each fixture's causes are "a" and "b".
RELABEL = {
    "tuple": {"a": ("a", 1), "b": ("b", 2)},
    "mixed": {"a": "a", "b": 2},
}

# (case, labels) -> reason, for a failure not fixed yet.
KNOWN_FAILURES: dict[tuple[str, str], str] = {}


def _params():
    out = []
    for case in CASES:
        if "e" not in case.rows:
            continue
        for labels in RELABEL:
            marks = []
            reason = KNOWN_FAILURES.get((case.name, labels))
            if reason is not None:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            out.append(
                pytest.param(
                    case.name, labels, id=f"{case.name}-{labels}", marks=marks
                )
            )
    return out


def _relabelled(name, labels):
    """The case with its events renamed, and its fixture relabelled."""
    case = CASE_BY_NAME[name]
    mapping = RELABEL[labels]
    data = case.data()
    # One element per row: ``np.array`` would split tuples into columns.
    e = np.empty(len(data["e"]), dtype=object)
    for k, v in enumerate(data["e"]):
        e[k] = None if v is None else mapping[v]
    data = {**data, "e": e}
    if not case.events:
        # One cause of interest (Fine-Gray), "a" in the fixture.
        data["event"] = mapping["a"]
    events = tuple(mapping[v] for v in case.events)
    return replace(case, events=events), data


@functools.lru_cache(maxsize=None)
def _fit(name, labels):
    case, data = _relabelled(name, labels)
    with quiet():
        return case, case.fit(data)


def _per_cause(case, model, mapping=None):
    """Every function at the case's query, keyed by function and the
    fixture's original cause name."""
    out = {}
    for fname in case.functions:
        out[fname] = np.asarray(call(case, model, fname, case.x), float)
    for fname in case.event_functions:
        for e in RELABEL["tuple"]:
            label = e if mapping is None else mapping[e]
            out[f"{fname}[{e}]"] = np.asarray(
                call(case, model, fname, case.x, event=label), float
            )
    return out


@pytest.mark.parametrize("name, labels", _params())
def test_labels_fit_and_predict(name, labels):
    """Relabelled causes predict what the original labels did."""
    case, model = _fit(name, labels)
    original = CASE_BY_NAME[name]
    got = _per_cause(case, model, RELABEL[labels])
    ref = _per_cause(original, fitted(original))
    assert got.keys() == ref.keys()
    for key in ref:
        np.testing.assert_allclose(
            got[key],
            ref[key],
            rtol=case.rtol,
            atol=case.rtol * 1e-2,
            err_msg=key,
        )


@pytest.mark.parametrize("name, labels", _params())
def test_labels_round_trip(name, labels):
    """Relabelled causes survive a strict-JSON round trip."""
    case, model = _fit(name, labels)
    check_round_trip(case, model)
