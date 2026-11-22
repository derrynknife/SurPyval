"""Methods that read like values, and say so when used as one.

``model.standard_errors`` beside ``model.params`` reads as an array, and
``np.round(model.standard_errors, 4)`` then failed with numpy's "unsupported
operand type(s) for *: 'method' and 'float'", which does not say what to do.
A method decorated with :func:`value_method` is called exactly as before;
used as a value (in arithmetic, ``np.asarray``, ``float``, indexing,
``len``, iteration) it raises a ``TypeError`` naming the call to make.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class _BoundValueMethod:
    """A method bound to its instance, refusing use as a value."""

    __slots__ = ("_function", "_instance", "__wrapped__")

    def __init__(self, function: Callable[..., Any], instance: Any) -> None:
        self._function = function
        self._instance = instance
        self.__wrapped__ = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._function(self._instance, *args, **kwargs)

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        return self._function.__doc__

    @property
    def __name__(self) -> str:
        return self._function.__name__

    def __repr__(self) -> str:
        owner = type(self._instance).__name__
        return f"<bound method {owner}.{self._function.__name__}>"

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        name = self._function.__name__
        raise TypeError(
            f"{name} is a method, not a value: call it, {name}(), to get "
            "its values (e.g. np.round(model."
            f"{name}(), 4))."
        )

    __array__ = _refuse
    __float__ = _refuse
    __int__ = _refuse
    __index__ = _refuse
    __len__ = _refuse
    __iter__ = _refuse
    __getitem__ = _refuse
    __neg__ = _refuse
    __pos__ = _refuse
    __abs__ = _refuse
    __round__ = _refuse
    __lt__ = _refuse
    __le__ = _refuse
    __gt__ = _refuse
    __ge__ = _refuse
    for _op in (
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "pow",
        "matmul",
        "mod",
    ):
        locals()[f"__{_op}__"] = _refuse
        locals()[f"__r{_op}__"] = _refuse
    del _op


class _ValueMethod:
    """The descriptor :func:`value_method` puts on the class."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function
        self.__doc__ = function.__doc__
        self.__wrapped__ = function

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            # On the class: the plain function, for help(), Sphinx and
            # inspect.signature.
            return self._function
        return _BoundValueMethod(self._function, instance)


def value_method(function: F) -> F:
    """
    Decorate a method whose result users take for an attribute, such as
    ``standard_errors`` or ``phi``: it is called as before, and used as a
    value it raises a ``TypeError`` that says to call it.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils._value_method import value_method
    >>> class Fit:
    ...     @value_method
    ...     def standard_errors(self):
    ...         return np.array([0.12345, 0.5])
    >>> np.round(Fit().standard_errors(), 2)
    array([0.12, 0.5 ])
    >>> np.round(Fit().standard_errors, 2)
    Traceback (most recent call last):
        ...
    TypeError: standard_errors is a method, not a value: call it, \
standard_errors(), to get its values (e.g. np.round(model.standard_errors(), \
4)).
    """
    return cast(F, _ValueMethod(function))
