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
  (``RecurrentEventData`` -> ``EventHistoryData``, v0.25);
- :func:`positional_option`, an argument that, given by position, is
  silently read as something other than what that place holds in the
  package's other methods of the name (``Parametric.mean(5)`` takes the
  5 as ``defective``, where ``KaplanMeier`` models' ``mean(5)`` takes it
  as ``tau``): a ``FutureWarning`` that names the keyword to use;
- :func:`ambiguous_item_ids`, the recurrent fits' item ids ``i``, which
  stand where the other fits take the censoring flags ``c``: a
  ``FutureWarning`` for ``CrowAMSAA.fit(x, c)``, the one call that reads
  flags as ids.

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
So is ``surpyval.beta.ml``, the old name of ``surpyval.ml`` (the survival
tree and random survival forest, promoted in v0.27): it imports them, with
a warning from :func:`beta_ml_alias`, until the first release on or after
:data:`BETA_ML_REMOVED_AFTER`, the same day, so one removal clears both.
"""

import datetime
import functools
import inspect
import os
import sys
import warnings
from collections.abc import Callable
from types import FrameType
from typing import Any, TypeVar

import numpy as np

__all__ = [
    "DATA_FUNCTIONS_DEPRECATED_ON",
    "DATA_FUNCTIONS_REMOVED_AFTER",
    "DATED_DATA_FUNCTIONS",
    "dated_data_function",
    "BETA_ML_DEPRECATED_ON",
    "BETA_ML_REMOVED_AFTER",
    "beta_ml_alias",
    "REMOVED_IN",
    "REMOVED_IN_NEXT",
    "SCHEDULED_REMOVALS",
    "scheduled_removal",
    "RenamedAttribute",
    "renamed_arguments",
    "renamed_name",
    "positional_option",
    "ambiguous_item_ids",
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


#: The day the survival tree and random survival forest were promoted
#: from ``surpyval.beta.ml`` to ``surpyval.ml`` (DECISIONS.md, 2026-11-03).
BETA_ML_DEPRECATED_ON = datetime.date(2026, 11, 4)

#: The first release on or after it drops ``surpyval.beta.ml``: the day
#: the top-level data-format functions go, so one removal clears both.
BETA_ML_REMOVED_AFTER = datetime.date(2027, 4, 12)

_IMPORTLIB_INIT = os.path.join("importlib", "__init__.py")


def _import_stacklevel(*files: str) -> int:
    """The ``stacklevel`` that points a warning raised while a module is
    imported at the ``import`` that asked for it: past this module, the
    modules in ``files`` (the one being imported) and
    ``importlib.import_module``. The import machinery's own frames
    (``importlib._bootstrap``) are not counted: ``warnings.warn`` skips
    those itself. Python 3.12's ``skip_file_prefixes`` does this; 3.11 is
    supported."""
    skip = (__file__,) + files
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if name in skip or name.endswith(_IMPORTLIB_INIT):
            level += 1
        elif not ("importlib" in name and "_bootstrap" in name):
            break
        frame = frame.f_back
    return level


def beta_ml_alias(module_file: str) -> None:
    """Warn, from the body of ``surpyval.beta.ml`` (``module_file`` its
    ``__file__``), that it is the old name of ``surpyval.ml``: a
    FutureWarning, as :func:`dated_data_function`'s, pointing at the
    import statement that loaded it rather than at the import machinery.
    Once per interpreter, since a module body runs once."""
    warnings.warn(
        "surpyval.beta.ml is deprecated since "
        f"{BETA_ML_DEPRECATED_ON.isoformat()} and will be removed in the "
        f"first release on or after {BETA_ML_REMOVED_AFTER.isoformat()}; "
        "it is now surpyval.ml (from surpyval.ml import SurvivalTree, "
        "RandomSurvivalForest).",
        FutureWarning,
        stacklevel=_import_stacklevel(module_file),
    )


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


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")


def _owner_name(owner: Any, method: str) -> str:
    """How a user names what ``method`` was called on: a fitter by its
    name (``CrowAMSAA``, ``Weibull``), a model by its class."""
    if method.startswith("fit"):
        # The fitter's own name where it reads as one (``Weibull``), not a
        # display name ("Crow-AMSAA"): as its ``repr`` names it.
        repr_name = getattr(owner, "_repr_name", None)
        if callable(repr_name) and not isinstance(owner, type):
            return str(repr_name())
        name = getattr(owner, "name", None)
        if (
            isinstance(name, str)
            and name
            and " " not in name
            and "-" not in name
        ):
            return name
    cls = owner if isinstance(owner, type) else type(owner)
    return cls.__name__.rstrip("_")


def _position(func: Callable[..., Any], name: str) -> tuple[int, bool, str]:
    """Where ``name`` sits among ``func``'s positional arguments: its
    index in ``args`` (``self`` or ``cls`` counted), whether ``func`` is a
    method, and the place a caller sees ("second")."""
    names = [
        p.name
        for p in inspect.signature(func).parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if name not in names:
        raise ValueError(
            f"{func.__qualname__} has no positional argument {name!r}"
        )
    index = names.index(name)
    bound = names[0] in ("self", "cls")
    place = index - 1 if bound else index
    ordinal = _ORDINALS[place] if place < len(_ORDINALS) else f"{place + 1}th"
    return index, bound, ordinal


def positional_option(
    name: str,
    taken_as: str,
    elsewhere: str,
    advice: str,
) -> Callable[[F], F]:
    """
    Decorate a method whose argument ``name`` is read, when it is given by
    position, as something other than what the same position holds in the
    package's other methods of that name (or in the ``x, c, n, t`` order
    of the data), so that a value meant for one is silently taken as the
    other. Passing ``name``, or any argument after it, by position then
    warns with a ``FutureWarning`` (shown wherever the call is made, since
    the risk is a wrong number, not an error) that says what the value was
    taken as and how to pass it by name. The call and its result are
    unchanged, and the argument stays positional: making it keyword-only
    would be a breaking change, which waits for 2.0 (DECISIONS.md).

    Parameters
    ----------
    name : str
        The argument, a parameter of the decorated function that can be
        given by position.
    taken_as : str
        What ``name`` is, in a few words: "the item each row belongs to".
    elsewhere : str
        What a user from another family would have meant at that place:
        "the censoring flags 'c', as in fit(x, c, n, t)".
    advice : str
        How to pass it by name: "i=... (or c=... for censoring flags)".

    Returns
    -------
    callable
        A decorator. Put it outermost (above any decorator that adds a
        frame), so that the warning points at the caller's line.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import positional_option
    >>> class Model:
    ...     @positional_option(
    ...         "defective", "whether to include the never-failing part",
    ...         "the time 'tau' to restrict the mean to", "defective=...",
    ...     )
    ...     def mean(self, defective=False, tau=None):
    ...         return 1.0
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().mean(5)
    1.0
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.mean: the first positional argument was taken as 'defective'
    (whether to include the never-failing part), not as the time 'tau' to
    restrict the mean to. The call is ambiguous: pass it by name,
    defective=...; the result is unchanged.
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().mean(defective=True)
    1.0
    >>> caught
    []
    """

    def decorate(func: F) -> F:
        index, bound, ordinal = _position(func, name)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) > index:
                where = (
                    _owner_name(args[0], func.__name__) + "." if bound else ""
                ) + func.__name__
                warnings.warn(
                    f"{where}: the {ordinal} positional argument was taken "
                    f"as '{name}' ({taken_as}), not as {elsewhere}. "
                    f"The call is ambiguous: pass it by name, {advice}; "
                    "the result is unchanged.",
                    FutureWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def ambiguous_item_ids(
    elsewhere: str,
    column: bool = False,
) -> Callable[[F], F]:
    """
    Decorate a recurrent-event ``fit`` (or, with ``column=True``, its
    ``fit_from_df``) whose item ids ``i`` stand where the other fits take
    the censoring flags ``c``, to catch the call that silently reads
    censoring flags as item ids: ``CrowAMSAA.fit(x, c)``.

    The ids stay a positional data argument (``fit(x, i, c)`` is the
    recurrent data, as ``fit(x, c, n, t)`` is the univariate). The warning,
    a ``FutureWarning``, fires only for the call that cannot be told
    apart: ``i`` given by position, ``c`` not given at all, and the values
    of ``i`` (or of its column) 0s and 1s, as censoring flags are. (Ids
    all 0 or all 1, one item, are left alone: read as flags they would
    give the same fit, or mark every row censored.)
    The call and its result are unchanged; refusing it would be a
    breaking change, which waits for 2.0 (DECISIONS.md).

    Parameters
    ----------
    elsewhere : str
        The call whose order the user may have meant:
        "fit(x, c, n, t)".
    column : bool, optional
        ``True`` for a ``fit_from_df``, whose ``i_col`` names a column of
        its ``df``.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import ambiguous_item_ids
    >>> class Fitter:
    ...     @ambiguous_item_ids("fit(x, c, n, t)")
    ...     def fit(self, x, i=None, c=None):
    ...         return len(set(i or [1]))
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Fitter().fit([1, 2, 3], [0, 0, 1])
    2
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Fitter.fit: the second positional argument was taken as 'i', the item
    each event belongs to, but no 'c' was given and its values are 0s and
    1s, as censoring flags are (the 'c' of fit(x, c, n, t)). Read as item
    ids, every row is an event and none is censored. Name it to say which
    it is: c=... for censoring flags, or i=... for item ids. The call is
    ambiguous; the result is unchanged.
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Fitter().fit([1, 2, 3], [1, 1, 2])  # ids, not flags
    ...     Fitter().fit([1, 2, 3], [1, 1, 1])  # one item
    ...     Fitter().fit([1, 2, 3], i=[0, 0, 1])  # named
    ...     Fitter().fit([1, 2, 3], [0, 0, 1], [0, 0, 1])  # c given
    2
    1
    2
    2
    >>> caught
    []
    """
    i_name, c_name = ("i_col", "c_col") if column else ("i", "c")
    what = (
        "the column of item ids"
        if column
        else ("the item each event belongs to")
    )

    def decorate(func: F) -> F:
        index, bound, ordinal = _position(func, i_name)
        c_index = _position(func, c_name)[0]
        df_index = _position(func, "df")[0] if column else -1

        def flags_like(args: tuple) -> bool:
            value = args[index]
            if column:
                try:
                    value = args[df_index][value]
                except Exception:
                    return False
            if value is None or isinstance(value, str):
                return False
            try:
                values = set(np.unique(np.asarray(value, dtype=float)))
            except (TypeError, ValueError):
                return False
            return values == {0.0, 1.0}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if (
                index < len(args) <= c_index
                and kwargs.get(c_name) is None
                and flags_like(args)
            ):
                where = (
                    _owner_name(args[0], func.__name__) + "." if bound else ""
                ) + func.__name__
                warnings.warn(
                    f"{where}: the {ordinal} positional argument was taken "
                    f"as '{i_name}', {what}, but no '{c_name}' was given "
                    "and its values are 0s and 1s, as censoring flags "
                    f"are (the '{c_name}' of {elsewhere}). Read as item "
                    "ids, every row is an event and none is censored. Name "
                    f"it to say which it is: {c_name}=... for censoring "
                    f"flags, or {i_name}=... for item ids. The call is "
                    "ambiguous; the result is unchanged.",
                    FutureWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
