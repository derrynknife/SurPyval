"""
Names removed from SurPyval, and what replaced them (#653).

A renamed name is deprecated for a release, with a ``DeprecationWarning``
(:mod:`surpyval.utils.deprecation`), and then removed. For a release or
two after its removal, using the old name raises the error Python would
raise anyway -- an ``AttributeError``, a ``TypeError`` or an
``ImportError`` -- with the replacement and the release named, so that
upgrading code finds the new name in the traceback rather than in the
changelog:

- :class:`RemovedName`, a removed attribute of a class;
- :func:`removed_arguments`, a function's or method's keyword arguments
  (``NHPPProportionalIntensity.fit(dist=)``);
- :data:`REMOVED_TOP_LEVEL`, the package's own old names (read by
  ``surpyval.__getattr__``), with :func:`from_import` so that
  ``from surpyval import RecurrentEventData`` keeps the message.

An entry is dropped once the old name has been gone for a release or two:
the hints for the names removed in v0.23 and v0.24 were retired in v0.27,
and ``surpyval.utils.recurrent_event_data`` (a module that only raised on
import, removed in v0.26) in v0.30.
"""

from __future__ import annotations

import dis
import functools
import sys
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "CONCORDANCE_ESTIMATE_REMOVED",
    "REMOVED_TOP_LEVEL",
    "RemovedName",
    "from_import",
    "raise_removed",
    "removed_arguments",
    "removed_message",
]

F = TypeVar("F", bound=Callable[..., Any])

#: The package's top-level names that were removed: ``{old: (what to do
#: instead, release it was removed in)}``.
REMOVED_TOP_LEVEL: dict[str, tuple[str, str]] = {
    "RecurrentEventData": (
        "it became surpyval.EventHistoryData, itself deprecated in v0.30: "
        "use surpyval.SurpyvalData with each row's unit i",
        "0.26",
    ),
    "concordance_estimate": (
        "use surpyval.metrics.concordance_index(x, c, risk, weighting=..., "
        "tau=..., alpha_ci=...), which gives the standard error and interval",
        "0.30",
    ),
}

#: The advice for ``concordance_estimate``, removed in v0.30 from
#: ``surpyval``, ``surpyval.metrics`` and the regression models.
CONCORDANCE_ESTIMATE_REMOVED = REMOVED_TOP_LEVEL["concordance_estimate"]


def removed_message(prefix: str, entry: tuple[str, str]) -> str:
    """``"<prefix>: it was removed in v<release>; <what to do instead>"``
    for an ``entry`` ``(what to do instead, release)``."""
    instead, release = entry
    return f"{prefix}: it was removed in v{release}; {instead}"


class RemovedName:
    """
    A removed attribute: reading it raises an ``AttributeError`` that names
    its replacement.

    A non-data descriptor, so an instance attribute or a subclass's own
    attribute of the same name wins over it, and ``hasattr`` is ``False``.
    """

    def __init__(self, instead: str, removed_in: str) -> None:
        self.entry = (instead, removed_in)
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        cls = type(obj) if obj is not None else owner
        what = (
            f"type object {cls.__name__!r}"
            if obj is None and cls is not None
            else f"{type(obj).__name__!r} object"
        )
        # No ``name=``/``obj=``: Python would add its own "Did you mean"
        # after the replacement this message names.
        raise AttributeError(
            removed_message(
                f"{what} has no attribute {self.name!r}", self.entry
            )
        )


def removed_arguments(
    removed_in: str, verb: str = "use", **renames: str
) -> Callable[[F], F]:
    """
    Decorate a function so that an argument's removed name raises a
    ``TypeError`` naming the replacement.

    Parameters
    ----------
    removed_in : str
        The release the old names were removed in.
    verb : str, optional
        What the message says to do with the replacement: ``"use"``.
    **renames : str
        ``old="replacement"``, one per removed argument. The replacement is
        quoted in the message as given: ``dist="'baseline'"``.

    Returns
    -------
    callable
        A decorator. The decorated function raises Python's ``TypeError``
        for an unexpected keyword argument, with the replacement named,
        when it is given an old name; otherwise it is called as it is.

    Examples
    --------
    >>> from surpyval.utils.removed_names import removed_arguments
    >>> @removed_arguments("0.25", dist="'baseline'")
    ... def fit(x, baseline=None):
    ...     return x
    >>> fit(1, dist=2)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    TypeError: fit() got an unexpected keyword argument 'dist': it was
    removed in v0.25; use 'baseline'
    """

    def decorate(func: F) -> F:
        where = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                for old in sorted(renames.keys() & kwargs.keys()):
                    raise TypeError(
                        removed_message(
                            f"{where}() got an unexpected keyword argument "
                            f"{old!r}",
                            (f"{verb} {renames[old]}", removed_in),
                        )
                    )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


try:
    _IMPORT_FROM = dis.opmap["IMPORT_FROM"]
except KeyError:  # pragma: no cover - every CPython has it
    _IMPORT_FROM = -1


def from_import(depth: int = 1) -> bool:
    """
    Whether the frame ``depth`` levels above the caller is running a
    ``from module import name`` statement.

    A module's ``__getattr__`` asks this to raise an ``ImportError`` with
    its message for ``from surpyval import Power``: Python replaces an
    ``AttributeError`` raised there with a generic "cannot import name",
    while ``surpyval.Power`` (and ``hasattr``) must still see an
    ``AttributeError``. Where the frame cannot be read, ``False``.
    """
    try:
        frame = sys._getframe(depth + 1)
        return frame.f_code.co_code[frame.f_lasti] == _IMPORT_FROM
    except (AttributeError, IndexError, ValueError):  # pragma: no cover
        return False


def raise_removed(module: str, name: str, entry: tuple[str, str]) -> None:
    """
    Raise the error for ``module.name``, a removed name of a subpackage,
    from that subpackage's ``__getattr__``: an ``AttributeError`` naming
    what to do instead and the release of ``entry`` (``(what to do
    instead, release)``), or an ``ImportError`` with the same message for
    ``from module import name``.

    Examples
    --------
    >>> from surpyval.utils.removed_names import raise_removed
    >>> raise_removed("surpyval.utils", "group_xcnt",
    ...               ("use SurpyvalData", "0.30"))
    Traceback (most recent call last):
    ...
    AttributeError: module 'surpyval.utils' has no attribute 'group_xcnt': it was removed in v0.30; use SurpyvalData
    """  # noqa: E501
    message = removed_message(
        f"module {module!r} has no attribute {name!r}", entry
    )
    if from_import(2):
        raise ImportError(
            removed_message(f"cannot import name {name!r}", entry),
            name=module,
        )
    raise AttributeError(message)
