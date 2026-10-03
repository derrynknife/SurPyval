"""Helpers for defining fitter singletons.

SurPyval distributions and processes are *configured values*, not distinct
types: ``Weibull`` and ``Exponential`` (or ``HPP`` and ``CrowAMSAA``) share all
of their fitting logic and differ only in the data they carry -- parameter
names, bounds, support. The natural way to express that is a single instance of
a fitter class rather than a class that is only ever instantiated once.

``singleton_fitter`` removes the ``Foo_`` shadow-class plus module-level
``Foo = Foo_(...)`` boilerplate that this otherwise requires.
"""

import importlib
import sys
from typing import Any, TypeVar

T = TypeVar("T")


def singleton_fitter(cls: type[T]) -> T:
    """Bind the decorated class' name to a single configured instance.

    Use on a fitter whose ``__init__`` takes no required arguments. The name is
    rebound to one instance, so ``fit`` and friends are ordinary instance
    methods and callers write ``HPP.fit(x)`` against the instance::

        @singleton_fitter
        class HPP(CountingProcess):
            def __init__(self):
                ...

        HPP.fit(x)        # HPP is the instance; fit is an instance method

    The underlying class remains reachable via ``type(instance)`` and, for
    import-by-name introspection, is also kept in its defining module under
    the conventional ``<Name>_`` alias (e.g. ``HPP_``) -- the same private
    name the explicit ``Foo_`` + ``Foo = Foo_()`` pattern used before. The
    class's ``__name__`` stays ``<Name>``, so Sphinx's ``autoclass`` renders
    the alias only as "alias of"; the API pages document such a fitter
    with ``autodata`` on the instance and ``automethod`` for its methods.

    The instance pickles as a reference to its module-level name, so a
    pickled model that holds it unpickles holding the same instance, and
    another instance of the class (a fitted model, for a fitter whose
    ``fit`` returns one) pickles through the ``<Name>_`` alias (#573).
    pickle finds a class by its ``__qualname__``, which is the name of
    the instance, so the default would fail.
    """
    module = sys.modules.get(cls.__module__)
    if module is not None:
        setattr(module, cls.__name__ + "_", cls)
    setattr(cls, "__reduce_ex__", _reduce_singleton)
    return cls()


def _reduce_singleton(self: object, protocol: int) -> Any:
    """``__reduce_ex__`` of a ``singleton_fitter`` class.

    The singleton is saved as its module-level name; another instance of
    the class as a new instance of the ``<Name>_`` alias with this one's
    state; an instance of a subclass as usual."""
    cls = type(self)
    module = sys.modules.get(cls.__module__)
    if getattr(module, cls.__name__, None) is self:
        return cls.__name__
    reduced = object.__reduce_ex__(self, protocol)
    alias = cls.__name__ + "_"
    if getattr(module, alias, None) is not cls:
        return reduced
    return (_new_instance, (cls.__module__, alias), *reduced[2:])


def _new_instance(module: str, name: str) -> Any:
    """A bare instance of the class ``name`` of ``module`` (unpickling a
    ``singleton_fitter`` class's other instances)."""
    cls = getattr(importlib.import_module(module), name)
    return cls.__new__(cls)
