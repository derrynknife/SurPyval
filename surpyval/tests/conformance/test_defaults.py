"""The same default everywhere (principle 15, #379).

Which default is statistically best is a matter of judgement, informed by
the calibration studies; that the choice, once made, is the same however a
model is fitted can be checked. #387 was this bug: ``CoxPH.fit`` defaulted
to Breslow ties while ``fit_from_df`` and ``fit_tvc`` defaulted to Efron,
so the same tied data gave different models by different routes.

- Every entry point of a fitter (``fit``, ``fit_from_df``, ``fit_tvc`` and
  the rest) gives an argument they share the same default.
- The Cox tie-handling default is the same in every model built on Cox
  fits.

Arguments that share a name across *different* fitters mostly mean
different things there (``how``, ``dist``, ``method``), so only named
concepts are compared across fitters.
"""

import importlib
import inspect

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import CASES


def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


FITTERS = sorted({f for case in CASES for f in case.fitters})


def _entry_points(fitter):
    """name -> signature of each public ``fit*`` method of ``fitter``."""
    out = {}
    for name in dir(fitter):
        if not name.startswith("fit"):
            continue
        method = getattr(fitter, name)
        if not callable(method):
            continue
        try:
            out[name] = inspect.signature(method)
        except (TypeError, ValueError):
            continue
    return out


def _same(a, b):
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    try:
        return bool(np.all(a == b))
    except Exception:
        return repr(a) == repr(b)


@pytest.mark.parametrize("dotted", FITTERS)
def test_entry_points_share_defaults(dotted):
    defaults = {}  # argument -> {entry point: default}
    for method, sig in _entry_points(_resolve(dotted)).items():
        for p in sig.parameters.values():
            if p.default is not inspect.Parameter.empty:
                defaults.setdefault(p.name, {})[method] = p.default
    differing = {
        arg: by
        for arg, by in defaults.items()
        if not all(_same(v, next(iter(by.values()))) for v in by.values())
    }
    assert not differing, f"{dotted}: entry points disagree: {differing}"


def _default(func, arg):
    return inspect.signature(func).parameters[arg].default


def test_cox_tie_method_default_is_the_same_everywhere():
    cr_ph = sp.univariate.competing_risks.CompetingRisksProportionalHazards
    found = {
        "CoxPH.fit": _default(sp.CoxPH.fit, "method"),
        "CoxPH.fit_from_df": _default(sp.CoxPH.fit_from_df, "method"),
        "CoxPH.fit_tvc": _default(sp.CoxPH.fit_tvc, "method"),
        "CompetingRisksProportionalHazards.fit": _default(
            cr_ph.fit, "tie_method"
        ),
    }
    assert set(found.values()) == {"efron"}, found
