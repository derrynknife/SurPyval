"""
Calls in an argument order that has changed.

``DegradationModel.cb`` and the process models' ``random`` took the
stress ``Z`` last; it now comes straight after the query, as everywhere
else (Design Principles, principle 21). :func:`old_order` keeps a call
written in the old order working until
:data:`~surpyval.utils.deprecation.REMOVED_IN`, with a
``DeprecationWarning`` saying to pass the arguments by keyword.
"""

import functools
import inspect
import numbers
import warnings
from typing import Any, Callable, TypeVar

import numpy as np

from surpyval.utils.deprecation import REMOVED_IN

F = TypeVar("F", bound=Callable[..., Any])


def old_order(
    names: tuple[str, ...],
    is_old: Callable[[Any, tuple, dict], bool],
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorate a method so that a call in its old argument order still works.

    Parameters
    ----------
    names : tuple of str
        The (current) names of the arguments after the first one, in the
        order they used to be taken by position.
    is_old : callable
        ``is_old(self, rest, kwargs)`` says whether a call whose positional
        arguments after the first are ``rest`` (never empty) was written in
        the old order.
    stacklevel : int, optional
        As for ``warnings.warn``: 2 when this is the outermost decorator,
        3 directly under
        :func:`~surpyval.utils.deprecation.renamed_arguments`, so that the
        warning points at the caller's line.

    Returns
    -------
    callable
        A decorator. An old-order call has its positional arguments after
        the first passed by name instead, with a ``DeprecationWarning``.
    """

    def decorate(func: F) -> F:
        name = func.__qualname__
        new = list(inspect.signature(func).parameters)[1:]
        message = (
            "{}: the argument order is now ({}); calls by position in the "
            "old order ({}) are deprecated and will stop working in v{}. "
            "Pass the arguments after {} by keyword.".format(
                name,
                ", ".join(new),
                ", ".join(new[:1] + list(names)),
                REMOVED_IN,
                new[0],
            )
        )

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            rest = args[1:]
            if not rest or not is_old(self, rest, kwargs):
                return func(self, *args, **kwargs)
            if len(rest) > len(names):
                raise TypeError(
                    "{}() takes at most {} positional arguments ({} given)"
                    "".format(name, len(names) + 1, len(args))
                )
            for key, value in zip(names, rest):
                if key in kwargs:
                    raise TypeError(
                        "{}() got multiple values for argument '{}'".format(
                            name, key
                        )
                    )
                kwargs[key] = value
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            return func(self, args[0], **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def cb_is_old(model: Any, rest: tuple, kwargs: dict) -> bool:
    """``cb(x, on, ...)``: a string second argument is the old ``on``
    (``Z`` is never a string)."""
    return isinstance(rest[0], str)


_SEEDS = (np.random.Generator, np.random.BitGenerator, np.random.SeedSequence)


def random_is_old(model: Any, rest: tuple, kwargs: dict) -> bool:
    """
    ``random(size, random_state, Z)``, the old order, rather than
    ``random(size, Z, random_state)``.

    Two positional arguments after ``size`` keep the old meaning (an int
    seed and a scalar stress cannot be told apart). A lone one is the old
    ``random_state`` when it can only be a seed: a numpy generator, an
    int for a model fitted without stress (which refuses ``Z``), or any
    value when ``Z`` is also passed by name; otherwise it is ``Z``.
    """
    if len(rest) > 1 or "Z" in kwargs:
        return True
    value = rest[0]
    if isinstance(value, _SEEDS):
        return True
    return not model.is_accelerated and isinstance(value, numbers.Integral)
