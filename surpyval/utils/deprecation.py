"""
Renamed arguments: accept the old name for one release, with a warning.

When an argument is renamed so that the same option has the same name
everywhere (Design Principles, principle 21), the old name keeps working
until :data:`REMOVED_IN`, with a ``DeprecationWarning`` naming the new
one. :func:`renamed_arguments` does this for a function or method.
"""

import functools
import warnings
from typing import Any, Callable, TypeVar

__all__ = ["REMOVED_IN", "renamed_arguments"]

#: The release in which the old names stop being accepted.
REMOVED_IN = "0.22.0"

F = TypeVar("F", bound=Callable[..., Any])


def renamed_arguments(**renames: Any) -> Callable[[F], F]:
    """
    Decorate a function so that it accepts its arguments' old names.

    Parameters
    ----------
    **renames : str or (str, callable)
        ``old="new"`` for a plain rename, or ``old=("new", convert)`` when
        the value changes meaning too: ``convert(old_value)`` gives the new
        argument's value (``confidence=("alpha_ci", lambda c: 1 - c)``).

    Returns
    -------
    callable
        A decorator. The decorated function takes the old names as keyword
        arguments, warns with a ``DeprecationWarning`` pointing at the
        caller's line, and calls the function with the new name. Passing
        both names raises a ``ValueError``.

    Notes
    -----
    Put it outermost (above any other decorator that adds a frame, such as
    :func:`surpyval.utils.shapes.keeps_query_shape`) so the warning points
    at the caller; under ``@classmethod`` or ``@staticmethod`` is fine, as
    they add no frame.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import renamed_arguments
    >>> @renamed_arguments(
    ...     B="n_boot", confidence=("alpha_ci", lambda c: round(1 - c, 12))
    ... )
    ... def bootstrap(n_boot=1000, alpha_ci=0.05):
    ...     return n_boot, alpha_ci
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     bootstrap(B=10, confidence=0.9)
    (10, 0.1)
    >>> print(caught[0].message)
    bootstrap: 'B' is deprecated and will be removed in v0.22.0; use 'n_boot'.
    >>> print(caught[1].message)  # doctest: +NORMALIZE_WHITESPACE
    bootstrap: 'confidence' is deprecated and will be removed in v0.22.0;
    use 'alpha_ci' (alpha_ci=0.1).
    """
    specs = {
        old: spec if isinstance(spec, tuple) else (spec, None)
        for old, spec in renames.items()
    }

    def decorate(func: F) -> F:
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, (new, convert) in specs.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise ValueError(
                        "{}: pass '{}' only; '{}' is its deprecated "
                        "old name.".format(name, new, old)
                    )
                value = kwargs.pop(old)
                message = (
                    "{}: '{}' is deprecated and will be removed in v{}; "
                    "use '{}'.".format(name, old, REMOVED_IN, new)
                )
                if convert is not None:
                    value = convert(value)
                    message = message[:-1] + " ({}={!r}).".format(new, value)
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                kwargs[new] = value
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
