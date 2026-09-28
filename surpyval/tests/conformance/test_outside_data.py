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

A non-parametric estimate can also be given an explicit support with
``set_support(lower, upper)``: every function is then at its start from
``lower`` to its first time, carries its value at the last time to
``upper``, and is NaN outside ``[lower, upper]``, and so are its
confidence bounds. Every case in ``RULES`` has it, except the
semi-parametric ones in ``WITHOUT_SET_SUPPORT``.
"""

import copy
import warnings
from dataclasses import replace

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
# The data-bounded cases without ``set_support``, each with the reason.
_SEMI = "semi-parametric: the explicit support is for the non-parametric "
_SEMI += "estimates only"
WITHOUT_SET_SUPPORT = {
    "CoxPH": _SEMI,
    "CoxPH[strata]": _SEMI,
    "AdditiveHazards": _SEMI,
    "BuckleyJames": _SEMI,
    "CompetingRisksProportionalHazards[Cox]": _SEMI,
    "CompetingRisksProportionalHazards[Fine-Gray]": _SEMI,
    "FineGray": _SEMI,
    "SurvivalTree[non-parametric]": "a tree of Kaplan-Meier leaves "
    "(surpyval.beta), not a single estimate; not given set_support yet",
}


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


# ---------------------------------------------------------------------------
# set_support: an explicit support
# ---------------------------------------------------------------------------
def _with_bounds(case):
    return case.name in RULES and case.name not in WITHOUT_SET_SUPPORT


def _support(case):
    """Bounds a little outside the fixture (and its time 0, where a
    recurrent MCF's observation begins)."""
    first, last = _span(case, case.data())
    margin = 0.1 * (last - first)
    return min(first, 0.0) - margin, last + margin, last


def _plain(case, model, fname, x, event, **kw):
    """``fname`` at ``x`` with keyword arguments, no warning silenced (the
    leak check applies)."""
    c = replace(case, call_kwargs={**case.call_kwargs, **kw})
    return np.asarray(call(c, model, fname, x, event=event), float)


def _bound_calls(case):
    """(label, method, kwargs, function, cause) for each confidence bound
    of the case evaluated at query times. A band is left out: it is
    defined only between the first and last events, and is NaN outside
    them with or without bounds (it is checked for that below)."""
    out = []
    for spec in case.bounds:
        if spec.kind != "function" or spec.point == "qf" or spec.slow:
            continue
        if spec.method == "band":
            continue
        names = spec.on or (spec.point,)
        causes = case.events if spec.per_cause else (None,)
        for fname in names:
            for event in causes:
                kw = dict(spec.kwargs)
                if spec.on:
                    kw["on"] = fname
                label = f"{spec.name}[{fname}]"
                if event is not None:
                    label += f"[{event}]"
                out.append((label, spec.method, kw, fname, event))
    return out


def _bound(model, method, x, event, kw):
    args = [x] if event is None else [x, event]
    return np.asarray(getattr(model, method)(*args, **kw), float)


@pytest.mark.parametrize("case", cases_for("outside_data", where=_with_bounds))
def test_set_support(case):
    fit = fitted(case)
    lower, upper, last = _support(case)
    # A copy: the fitted model is cached and shared by every test.
    model = copy.deepcopy(fit)
    assert model.set_support(lower, upper) is model
    assert fit.support is None, "set_support changed the cached model"
    x = np.array([lower - 1.0, lower, last, upper, upper + 1.0])
    x = np.append(x, [-np.inf, np.inf, np.nan])
    interps = case.interp or ("step",)
    for fname, event in calls(case):
        if fname == "qf":
            continue
        label = fname if event is None else f"{fname}[{event}]"
        for interp in interps:
            kw = {} if interp == "step" else {"interp": interp}
            where = f"{label}(interp={interp!r})"
            v = _plain(case, model, fname, x, event, **kw)
            assert np.isnan(v[[0, 4, 5, 6, 7]]).all(), f"{where}: {v}"
            start = START.get(fname, 0.0)
            assert v[1] == start and not np.signbit(v[1]), f"{where}: {v}"
            same = np.array_equal(v[3], v[2], equal_nan=True)
            assert same, f"{where} after the last time: {v}"
    for label, method, kw, fname, event in _bound_calls(case):
        b = _bound(model, method, x, event, kw)
        assert np.isnan(b[[0, 4, 5, 6, 7]]).all(), f"{label}: {b}"
        start = START.get(fname, 0.0)
        assert np.all(b[1] == start), f"{label} at the start: {b}"
        # (A plain normal bound on Hf can be NaN at the last time.)
        same = np.array_equal(b[3], b[2], equal_nan=True)
        assert same, f"{label} after the last time: {b}"
    if hasattr(model, "band"):
        assert np.isnan(model.band(x[[0, 1, 3, 4]])).all()

    # Infinite bounds take infinite queries.
    model.set_support(-np.inf, np.inf)
    for fname, event in calls(case):
        if fname in START:
            v = _plain(case, model, fname, [-np.inf, last, np.inf], event)
            assert v[0] == START[fname], f"{fname} at -inf: {v}"
            same = np.array_equal(v[2], v[1], equal_nan=True)
            assert same, f"{fname} at inf: {v}"


def test_non_parametric_estimates_have_set_support():
    """So a new non-parametric estimator gets an explicit support too."""
    missing = [
        name
        for name in RULES
        if name not in WITHOUT_SET_SUPPORT
        and not hasattr(fitted(_case(name)), "set_support")
    ]
    assert not missing, f"no set_support (principle 11): {missing}"
    stale = [
        name
        for name in WITHOUT_SET_SUPPORT
        if name not in RULES or hasattr(fitted(_case(name)), "set_support")
    ]
    assert not stale, f"remove from WITHOUT_SET_SUPPORT: {stale}"


def _case(name):
    return next(case for case in CASES if case.name == name)


@pytest.mark.parametrize("case", cases_for("outside_data", where=_with_bounds))
def test_bounds_are_nan_outside_the_data_without_a_support(case):
    """Without a support every confidence bound of the case -- the
    pointwise ones and the bootstrap alike -- is NaN below the first and
    above the last time, where the estimate says nothing (principle 11).
    ``bootstrap_cb`` used to carry its step convention there instead (1,
    and the last bounds; #452). (A missing time is ``test_missing``'s.)"""
    model = fitted(case)
    assert model.support is None
    lower, upper, _ = _support(case)
    x = np.array([lower, upper])
    for label, method, kw, _, event in _bound_calls(case):
        b = _bound(model, method, x, event, kw)
        assert np.isnan(b).all(), f"{label}: {b}"
