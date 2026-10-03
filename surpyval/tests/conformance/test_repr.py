"""Every registered fitter prints as itself (#614).

A fitter printed as ``<surpyval...Weibull_ object at 0x7f...>``: in a
notebook, ``WeibullAFT`` and ``CoxPH`` said nothing of what they are. Each
fitter registered in the conformance registry (a case's ``fitters``) now
prints one line that starts with its public name and says what it fits
(``FitterRepr``, ``surpyval/utils/fitter_repr.py``)::

    WeibullAFT: accelerated failure time fitter (Weibull baseline)

A registered class (a model class whose ``fit`` is a class method) prints
as the class, and is left out; a factory (``AcceleratedLife``,
``Frailty``, ``Discretize``) is checked through the fitter it returns.
"""

import importlib
import inspect
import re

import pytest

import surpyval as sp
from surpyval.life_models import Power
from surpyval.tests.conformance.registry import CASES

#: A fitter each factory makes, by the factory's registered path.
FACTORIES = {
    "surpyval.AcceleratedLife": (
        lambda: sp.AcceleratedLife(sp.Weibull, Power),
        "WeibullPowerAL",
    ),
    "surpyval.Frailty": (lambda: sp.Frailty(sp.Weibull), "WeibullFrailty"),
    "surpyval.Discretize": (
        lambda: sp.Discretize(sp.Weibull),
        "Discretize(Weibull)",
    ),
}

#: The repr Python gives an object that has none of its own.
DEFAULT_REPR = re.compile(r"<.* at 0x[0-9a-fA-F]+>")


def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _fitters():
    out = []
    for dotted in sorted({p for case in CASES for p in case.fitters}):
        obj = _resolve(dotted)
        if dotted in FACTORIES:
            make, name = FACTORIES[dotted]
            out.append(pytest.param(make, name, id=dotted))
        elif not (inspect.isclass(obj) or inspect.isfunction(obj)):
            name = dotted.rpartition(".")[2]
            out.append(pytest.param(lambda obj=obj: obj, name, id=dotted))
    return out


def test_every_factory_is_listed():
    factories = {
        dotted
        for case in CASES
        for dotted in case.fitters
        if inspect.isfunction(_resolve(dotted))
    }
    assert factories == set(FACTORIES)


@pytest.mark.parametrize("make, name", _fitters())
def test_614_a_fitter_prints_its_name_and_kind(make, name):
    text = repr(make())
    assert not DEFAULT_REPR.search(text), text
    assert "\n" not in text, text
    assert text.startswith(name), (name, text)
