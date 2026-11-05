"""
Renamed names: accept the old name for one release, with a warning.

When a public name changes so that the same thing has the same name
everywhere (Design Principles, principle 21), the old one keeps working
until :data:`REMOVED_IN`, with a ``DeprecationWarning`` that names the new
one and points at the caller's line: a name renamed in v0.26 is removed in
v0.27. This module holds the shapes such a rename takes (each with the
rename it was written for):

- :func:`renamed_arguments`, an argument of a function or method;
- :class:`RenamedAttribute`, an attribute or property of a class;
- :func:`renamed_name`, a class or other name in a module's namespace
  (``RecurrentEventData`` -> ``EventHistoryData``, v0.25).

A name deprecated in v0.27 is accepted until :data:`REMOVED_IN_NEXT`.

A few deprecations are given a longer window, chosen by the maintainer:
they are listed, each with the release that removes it, in
:data:`SCHEDULED_REMOVALS`, and warn through :func:`scheduled_removal`
(``CoxPH.check_ph``, deprecated in v0.26 and removed in v0.28, whose test
``model.diagnose()`` now runs).

The top-level data-format functions (``surpyval.fs_to_xcnt`` and the
rest), whose work :class:`~surpyval.SurpyvalData` now does, are deprecated
by date instead (:data:`DATED_DATA_FUNCTIONS`): accepted, with a warning,
until the first release on or after :data:`DATA_FUNCTIONS_REMOVED_AFTER`.
"""

import datetime
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "DATA_FUNCTIONS_DEPRECATED_ON",
    "DATA_FUNCTIONS_REMOVED_AFTER",
    "DATED_DATA_FUNCTIONS",
    "dated_data_function",
    "REMOVED_IN",
    "REMOVED_IN_NEXT",
    "SCHEDULED_REMOVALS",
    "scheduled_removal",
    "RenamedAttribute",
    "renamed_arguments",
    "renamed_name",
]

#: The release in which the names deprecated in v0.26 stop being
#: accepted. (Those deprecated in v0.25 were removed in v0.26.)
REMOVED_IN = "0.27"

#: The release in which the names deprecated in v0.27 stop being accepted.
REMOVED_IN_NEXT = "0.28"

#: The deprecations with a longer window than one release, by name, each
#: with the release that removes it. ``test_removed_arguments`` fails once
#: the package's version reaches an entry still listed here.
SCHEDULED_REMOVALS: dict[str, str] = {
    # Deprecated in v0.26: diagnose() runs the same test (DECISIONS.md).
    "check_ph": "0.28",
}

#: The day the top-level data-format functions were deprecated.
DATA_FUNCTIONS_DEPRECATED_ON = datetime.date(2026, 10, 12)

#: Six months on: the first release on or after it removes them.
DATA_FUNCTIONS_REMOVED_AFTER = datetime.date(2027, 4, 12)

#: Each deprecated top-level data function, with what replaces it.
DATED_DATA_FUNCTIONS: dict[str, str] = {
    "fs_to_xcnt": (
        "SurpyvalData.from_observations(observed=f, right_censored=s)"
        ".to_xcnt()"
    ),
    "fsl_to_xcnt": (
        "SurpyvalData.from_observations(observed=f, right_censored=s, "
        "left_censored=l).to_xcnt()"
    ),
    "fsli_to_xcnt": (
        "SurpyvalData.from_observations(observed=f, right_censored=s, "
        "left_censored=l, interval_censored=i).to_xcnt()"
    ),
    "fsli_handler": (
        "SurpyvalData.from_observations(observed=f, right_censored=s, "
        "left_censored=l, interval_censored=i), which validates them"
    ),
    "fs_to_xrd": (
        "SurpyvalData.from_observations(observed=f, right_censored=s)"
        ".to_xrd()"
    ),
    "xcn_to_fs": "SurpyvalData(x, c, n).to_observations()",
    "xcnt_handler": "SurpyvalData(x, c, n, t, ...).to_xcnt()",
    "xcnt_to_xrd": "SurpyvalData(x, c, n, t).to_xrd()",
    "xrd_handler": "SurpyvalData.from_xrd(x, r, d), which validates them",
    "xrd_to_xcnt": "SurpyvalData.from_xrd(x, r, d).to_xcnt()",
}


def dated_data_function(name: str) -> Any:
    """``surpyval.<name>``, one of :data:`DATED_DATA_FUNCTIONS`, with a
    FutureWarning that gives the dates and the replacement. A
    FutureWarning, not a DeprecationWarning: Python shows it wherever the
    name is used, while a DeprecationWarning is hidden outside
    ``__main__``, and these are removed on a date, not at a release. Called
    by the package ``__getattr__``, so the warning points at the code
    that used the name. The release-based scan of
    ``test_removed_arguments`` leaves these alone: the date decides when
    they go (``test_dated_removal_is_not_overdue``)."""
    from importlib import import_module

    warnings.warn(
        f"surpyval.{name} is deprecated since "
        f"{DATA_FUNCTIONS_DEPRECATED_ON.isoformat()} and will be removed in "
        "the first release on or after "
        f"{DATA_FUNCTIONS_REMOVED_AFTER.isoformat()}; use "
        f"{DATED_DATA_FUNCTIONS[name]}.",
        FutureWarning,
        stacklevel=3,
    )
    return getattr(import_module("surpyval.utils.data_formats"), name)


def scheduled_removal(name: str, advice: str, stacklevel: int = 3) -> None:
    """Warn, with a DeprecationWarning pointing at the caller's caller,
    that ``name`` (one of :data:`SCHEDULED_REMOVALS`) is deprecated and
    will be removed in its release; ``advice`` says what to use instead.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     scheduled_removal("check_ph", "use model.diagnose()", 2)
    >>> print(caught[0].message)
    check_ph is deprecated and will be removed in v0.28; use model.diagnose()
    """
    warnings.warn(
        f"{name} is deprecated and will be removed in "
        f"v{SCHEDULED_REMOVALS[name]}; {advice}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


F = TypeVar("F", bound=Callable[..., Any])


def _message(
    where: str, old: str, new: str, removed_in: str = REMOVED_IN
) -> str:
    return (
        f"{where}: '{old}' is deprecated and will be removed in "
        f"v{removed_in}; use '{new}'."
    )


def renamed_name(
    where: str,
    old: str,
    new: str,
    value: Any,
    removed_in: str = REMOVED_IN,
) -> Any:
    """``value``, reached by its old name ``old`` in the namespace ``where``
    (a module's ``__getattr__``), with a DeprecationWarning naming ``new``
    and pointing at the line that used the old name. ``value`` is the
    renamed object itself, so ``isinstance`` checks against the old name
    keep working until it is removed."""
    warnings.warn(
        _message(where, old, new, removed_in),
        DeprecationWarning,
        stacklevel=3,
    )
    return value


def renamed_arguments(
    removed_in: str = REMOVED_IN, **renames: str
) -> Callable[[F], F]:
    """
    Decorate a function so that it accepts its arguments' old names.

    Parameters
    ----------
    removed_in : str, optional
        The release in which the old names stop being accepted:
        :data:`REMOVED_IN` (the default) for the names renamed in v0.26,
        :data:`REMOVED_IN_NEXT` for those renamed in v0.27.
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
    describe: 'names' is deprecated and will be removed in v0.27;
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
                    _message(where, old, new, removed_in),
                    DeprecationWarning,
                    stacklevel=2,
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
    ``loglik = RenamedAttribute("log_likelihood")``. It works on
    instances and on the class itself (for a singleton fitter, or a class
    attribute). A name deprecated in v0.27 passes
    ``removed_in=REMOVED_IN_NEXT``.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import RenamedAttribute
    >>> class Model:
    ...     log_likelihood = -12.5
    ...     loglik = RenamedAttribute("log_likelihood")
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().loglik
    -12.5
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.loglik is deprecated and will be removed in v0.27;
    use 'log_likelihood'.
    """

    def __init__(self, new: str, removed_in: str = REMOVED_IN) -> None:
        self.new = new
        self.old = ""
        self.removed_in = removed_in

    def __set_name__(self, owner: type, name: str) -> None:
        self.old = name

    def _warn(self, owner: type) -> None:
        warnings.warn(
            "{}.{} is deprecated and will be removed in v{}; use "
            "'{}'.".format(
                owner.__name__, self.old, self.removed_in, self.new
            ),
            DeprecationWarning,
            stacklevel=3,
        )

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        owner = type(obj) if owner is None else owner
        if obj is None and self.new not in dir(owner):
            # The class of an instance attribute (``ParametricFitter``
            # itself): nothing to read, and introspection (``help``,
            # Sphinx) should not warn.
            return self
        self._warn(owner)
        return getattr(owner if obj is None else obj, self.new)

    def __set__(self, obj: Any, value: Any) -> None:
        self._warn(type(obj))
        setattr(obj, self.new, value)
