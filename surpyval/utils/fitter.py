"""Helpers for defining fitter singletons.

SurPyval distributions and processes are *configured values*, not distinct
types: ``Weibull`` and ``Exponential`` (or ``HPP`` and ``CrowAMSAA``) share all
of their fitting logic and differ only in the data they carry -- parameter
names, bounds, support. The natural way to express that is a single instance of
a fitter class rather than a class that is only ever instantiated once.

``singleton_fitter`` removes the ``Foo_`` shadow-class plus module-level
``Foo = Foo_(...)`` boilerplate that this otherwise requires.
"""


def singleton_fitter(cls):
    """Bind the decorated class' name to a single configured instance.

    Use on a fitter whose ``__init__`` takes no required arguments. The name is
    rebound to one instance, so ``fit`` and friends are ordinary instance
    methods and callers write ``HPP.fit(x)`` against the instance::

        @singleton_fitter
        class HPP(CountingProcess):
            def __init__(self):
                ...

        HPP.fit(x)        # HPP is the instance; fit is an instance method
        type(HPP)         # the underlying class, if it is ever needed

    The underlying class remains reachable via ``type(instance)``.
    """
    return cls()
