"""Callables a fitted model keeps, in a form that pickles (#573).

A fit often builds its objective as a closure over the data (the
negative log-likelihood of a recurrence model, the partial likelihood of
a Cox model) and keeps it on the fitted model, which needs it again for
profile-likelihood bounds, standard errors or a likelihood-ratio test.
pickle cannot save a closure, so such a model could not be sent to a
worker process (``multiprocessing``, ``joblib``, ``concurrent.futures``)
or cached with ``pickle`` / ``joblib.dump``.

:class:`Rebuilt` keeps what the closure is built from instead -- the
function that builds it and that function's arguments -- and builds it
again where it is unpickled. The arguments are the same values, so the
closure is the same function, and the unpickled model predicts and
infers exactly as the original did.
"""

from __future__ import annotations

from typing import Any, Callable


class Rebuilt:
    """A callable built by ``build(*args)``, which pickles as ``build``
    and ``args``.

    Parameters
    ----------
    build : callable
        A function, or a method of a picklable object, that returns the
        callable (or, with ``item``, a sequence holding it).
    args : tuple
        The arguments ``build`` is called with; they must pickle.
    item : int, optional
        Where ``build`` returns several callables (a likelihood and its
        derivatives, say), the position of this one.
    built : callable, optional
        The callable already built from ``build(*args)``, to save
        building it again; otherwise it is built at the first call.

    Examples
    --------
    ``functools.partial`` stands in for a function that builds a
    closure:

    >>> import functools, operator, pickle
    >>> f = Rebuilt(functools.partial, (operator.mul, 3.0))
    >>> f(2.0)
    6.0
    >>> pickle.loads(pickle.dumps(f))(2.0)
    6.0
    """

    def __init__(
        self,
        build: Callable[..., Any],
        args: tuple = (),
        item: int | None = None,
        built: Callable[..., Any] | None = None,
    ) -> None:
        self.build = build
        self.args = tuple(args)
        self.item = item
        self._built = built

    def _function(self) -> Callable[..., Any]:
        if self._built is None:
            built = self.build(*self.args)
            self._built = built if self.item is None else built[self.item]
        return self._built

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._function()(*args, **kwargs)

    def __reduce__(self) -> tuple:
        return (type(self), (self.build, self.args, self.item))
