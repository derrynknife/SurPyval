"""Behaviour outside the data (principle 11, #379).

A parametric model is defined everywhere by its formula. An estimate with
no formula for its shape -- a step estimate, a semi-parametric baseline --
has nothing to say outside the observed times, so what it gives there is a
convention, documented and the same for all of the model's functions:

- before the first time it is at its start: ``sf`` 1, and ``ff``, ``Hf``,
  a cumulative incidence ``cif`` and a mean cumulative function ``mcf`` 0
  (at time 0 for an estimate whose hazard acts from time 0,
  ``STARTS_AT_ZERO``);
- after the last time it either holds its value at the last time (the
  single-event estimates, ``"hold"``) or is NaN (the recurrent mean
  cumulative functions, ``"nan"``; see "Recurrent Event Analysis").

``RULES`` gives each data-bounded case its convention. A case that behaves
as data-bounded -- its curve changes within the data but is exactly flat,
or NaN, from the last time on -- but is not listed fails
``test_data_bounded_cases_are_listed``, so a new step estimator is not
skipped silently.
"""

import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    BIVARIATE,
    CASES,
    call,
    calls,
    cases_for,
    fitted,
)

RULES = {
    "KaplanMeier": "hold",
    "NelsonAalen": "hold",
    "FlemingHarrington": "hold",
    "Turnbull": "hold",
    "CoxPH": "hold",
    "CoxPH[strata]": "hold",
    "AdditiveHazards": "hold",
    "BuckleyJames": "hold",
    "SurvivalTree[non-parametric]": "hold",
    "CompetingRisks[Nelson-Aalen]": "hold",
    "CompetingRisks[Kaplan-Meier]": "hold",
    "CompetingRisksProportionalHazards[Cox]": "hold",
    "CompetingRisksProportionalHazards[Fine-Gray]": "hold",
    "FineGray": "hold",
    "NonParametricCounting": "nan",
    "CauseSpecificMCF": "nan",
}
START = {"sf": 1.0, "ff": 0.0, "Hf": 0.0, "cif": 0.0, "mcf": 0.0}
# Estimates whose hazard acts from time 0, not only at the event times: an
# additive hazards model's covariate effect is a constant added hazard, so
# before the first observed time it is already the fitted model, and it
# starts at time 0.
STARTS_AT_ZERO = frozenset({"AdditiveHazards"})


def _span(case, data):
    """The first and last finite times in the fixture."""
    times = [np.asarray(data[k], float).ravel() for k in case.times]
    t = np.concatenate(times)
    t = t[np.isfinite(t)]
    return t.min(), t.max()


def _values(case, model, fname, x, event):
    z = None if case.Z is None else case.Z[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(call(case, model, fname, x, z, event=event), float)


def _curves(case):
    """(label, function, cause) for each cumulative function of the case."""
    return [
        (fname if e is None else f"{fname}[{e}]", fname, e)
        for fname, e in calls(case)
        if fname in START
    ]


@pytest.mark.parametrize(
    "case", cases_for("outside_data", where=lambda c: c.name in RULES)
)
def test_outside_the_data(case):
    model = fitted(case)
    first, last = _span(case, case.data())
    before = np.array([first / 10 if first > 0 else first - 1.0])
    if case.name in STARTS_AT_ZERO:
        before = np.zeros(1)
    after = last * np.array([1.0, 2.0, 10.0, 100.0])
    if not case.continuous:
        before, after = np.floor(before), np.ceil(after)
    for label, fname, event in _curves(case):
        start = _values(case, model, fname, before, event)
        np.testing.assert_allclose(
            start, START[fname], atol=1e-12, err_msg=f"{label} before"
        )
        v = _values(case, model, fname, after, event)
        # inf is a value here (Hf once a Kaplan-Meier estimate reaches 0).
        assert not np.isnan(v[0]), f"{label} at the last time: {v[0]}"
        if RULES[case.name] == "hold":
            assert np.all(v[1:] == v[0]), f"{label} after the data: {v}"
        else:
            assert np.all(np.isnan(v[1:])), f"{label} after the data: {v}"


def _looks_data_bounded(case):
    """Whether the curve behaves as an estimate that stops at the data:
    NaN far past the last time, or exactly flat from the last time on,
    strictly inside (0, 1), while it changed within the data. A
    parametric curve that levels off (a limited failure population, a
    cause's share of the failures) only approaches its limit, so it still
    moves between the last time and far past it."""
    try:
        model = fitted(case)
        first, last = _span(case, case.data())
    except Exception:
        return False
    if not last > 0:
        return False
    x = np.array([first, last, 10 * last, 100 * last, 1000 * last])
    for label, fname, event in _curves(case):
        try:
            v = _values(case, model, fname, x, event)
        except ValueError:  # outside the support, refused (Bernoulli)
            continue
        if np.all(np.isnan(v[2:])):
            return True
        if fname == "Hf":
            continue  # its ceiling is a saturation of sf, below
        # A curve that has reached 0 or 1 (a bounded support, a
        # saturated distribution) is flat for another reason.
        inside = 1e-12 < v[1] < 1 - 1e-12 if fname != "mcf" else v[1] > 0
        if inside and np.all(v[2:] == v[1]) and v[0] != v[1]:
            return True
    return False


def test_data_bounded_cases_are_listed():
    unlisted = [
        case.name
        for case in CASES
        if case.interface != BIVARIATE
        and case.name not in RULES
        and _looks_data_bounded(case)
    ]
    assert not unlisted, (
        "these behave as data-bounded estimates: give each its rule in "
        f"RULES: {unlisted}"
    )
