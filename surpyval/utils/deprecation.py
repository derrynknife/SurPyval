"""
Renamed names: accept the old name for one release, with a warning.

When a public name changes so that the same thing has the same name
everywhere (Design Principles, principle 21), the old one keeps working
until :data:`REMOVED_IN`, with a ``DeprecationWarning`` that names the new
one and points at the caller's line. This module holds the three shapes
such a rename takes:

- :func:`renamed_arguments`, an argument of a function or method;
- :class:`RenamedAttribute`, an attribute or property of a class;
- :class:`CallableList`, a method that became a property returning a list
  (``model.parameter_names()`` -> ``model.parameter_names``).
"""

import functools
import warnings
from typing import Any, Callable, TypeVar

__all__ = [
    "REMOVED_IN",
    "CallableList",
    "RenamedAttribute",
    "renamed_arguments",
    "renamed_class_attribute",
]

#: The release in which the old names stop being accepted.
REMOVED_IN = "0.23.0"

F = TypeVar("F", bound=Callable[..., Any])


def _message(where: str, old: str, new: str) -> str:
    return (
        f"{where}: '{old}' is deprecated and will be removed in "
        f"v{REMOVED_IN}; use '{new}'."
    )


def renamed_arguments(**renames: str) -> Callable[[F], F]:
    """
    Decorate a function so that it accepts its arguments' old names.

    Parameters
    ----------
    **renames : str
        ``old="new"``, one per renamed argument.

    Returns
    -------
    callable
        A decorator. The decorated function takes the old names as keyword
        arguments, warns with a ``DeprecationWarning`` pointing at the
        caller's line, and calls the function with the new name. Passing
        both names raises a ``ValueError``.

    Notes
    -----
    Put it outermost (above any other decorator that adds a frame) so the
    warning points at the caller.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import renamed_arguments
    >>> @renamed_arguments(names="labels")
    ... def describe(labels=()):
    ...     return list(labels)
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     describe(names=["a"])
    ['a']
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    describe: 'names' is deprecated and will be removed in v0.23.0;
    use 'labels'.
    """

    def decorate(func: F) -> F:
        where = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise ValueError(
                        "{}: pass '{}' only; '{}' is its deprecated "
                        "old name.".format(where, new, old)
                    )
                warnings.warn(
                    _message(where, old, new), DeprecationWarning, stacklevel=2
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


class RenamedAttribute:
    """
    A class attribute's old name: reading or setting it warns and uses the
    new one.

    Declare it on the class under the old name,
    ``param_names = RenamedAttribute("parameter_names")``. It works on
    instances and on the class itself (``Weibull.param_names`` for a
    singleton, ``PowerPath.param_names`` for a class attribute).

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import RenamedAttribute
    >>> class Model:
    ...     parameter_names = ["a", "b"]
    ...     param_names = RenamedAttribute("parameter_names")
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().param_names
    ['a', 'b']
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.param_names is deprecated and will be removed in v0.23.0;
    use 'parameter_names'.
    """

    def __init__(self, new: str) -> None:
        self.new = new
        self.old = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.old = name

    def _warn(self, owner: type) -> None:
        warnings.warn(
            "{}.{} is deprecated and will be removed in v{}; use "
            "'{}'.".format(owner.__name__, self.old, REMOVED_IN, self.new),
            DeprecationWarning,
            stacklevel=3,
        )

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        owner = type(obj) if owner is None else owner
        self._warn(owner)
        return getattr(owner if obj is None else obj, self.new)

    def __set__(self, obj: Any, value: Any) -> None:
        self._warn(type(obj))
        setattr(obj, self.new, value)


def renamed_class_attribute(cls: type, old: str, new: str) -> None:
    """
    Accept a subclass that still defines a class attribute by its old name.

    Call it from the base class's ``__init_subclass__``: a subclass body
    that sets ``old`` (a user's own path model or copula written against the
    old name) warns once, at class creation, and the value is moved to
    ``new``, so the package, which reads only ``new``, sees it.
    """
    value = cls.__dict__.get(old)
    if value is None or isinstance(value, RenamedAttribute):
        return
    warnings.warn(
        "{}: the class attribute '{}' is deprecated and will be removed in "
        "v{}; define '{}'.".format(cls.__qualname__, old, REMOVED_IN, new),
        DeprecationWarning,
        stacklevel=3,
    )
    if new not in cls.__dict__:
        setattr(cls, new, value)
    delattr(cls, old)


class CallableList(list):
    """
    A list that can still be called, for a method that became a property.

    ``model.parameter_names`` was a method, and is now a property; it
    returns one of these, which is a plain ``list`` in every other respect
    (equality, ``len``, iteration, indexing, ``json.dumps``, pandas and
    numpy), and whose call, the old spelling, warns and returns itself.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import CallableList
    >>> names = CallableList(["alpha", "beta"], "WeibullPH.parameter_names")
    >>> names == ["alpha", "beta"]
    True
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     names()
    ['alpha', 'beta']
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    WeibullPH.parameter_names is now a property: 'parameter_names()' is
    deprecated and will be removed in v0.23.0; use 'parameter_names'.
    """

    def __init__(self, items: Any = (), where: str = "") -> None:
        super().__init__(items)
        self._where = where

    def __call__(self, *args: Any, **kwargs: Any) -> "CallableList":
        # pandas calls a callable key with the frame (``df.loc[names]``,
        # ``df[names]``): that is not the old spelling, so no warning.
        if args or kwargs:
            return self
        attr = self._where.rpartition(".")[2]
        warnings.warn(
            "{} is now a property: '{}()' is deprecated and will be removed "
            "in v{}; use '{}'.".format(self._where, attr, REMOVED_IN, attr),
            DeprecationWarning,
            stacklevel=2,
        )
        return self
