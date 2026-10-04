"""
Names removed from SurPyval, and what replaced them (#653).

A renamed name is deprecated for a release, with a ``DeprecationWarning``
(:mod:`surpyval.utils.deprecation`), and then removed. For a release or
two after its removal, using the old name raises the error Python would
raise anyway -- an ``AttributeError``, a ``TypeError`` or an
``ImportError`` -- with the replacement and the release named, so that
upgrading code finds the new name in the traceback rather than in the
changelog:

- :data:`REMOVED_ATTRIBUTES`, attributes of every fitter and model
  (``param_names``, ``se``, ``cov_matrix``, ...), and
  :data:`MIXTURE_EM_ATTRIBUTES`, the public names of ``MixtureModel``'s EM
  steps, read through :class:`RemovedNames`;
- :func:`removed_arguments`, a function's or method's keyword arguments
  (``cs(X=)``, ``ARI.fit(dist=)``, ``from_params(p=)``, ...), and
  :func:`column_arguments`, a ``fit_from_df`` column given by the ``fit``
  argument's name (``x=`` for ``x_col=``);
- :data:`REMOVED_TOP_LEVEL`, the package's own old names (the life models
  and the numeric constants, read by ``surpyval.__getattr__``), with
  :func:`from_import` so that ``from surpyval import Power`` keeps the
  message;
- :func:`removed_parameter_note`, a parameter's old name in ``fixed`` and
  ``param_cb`` (``p`` for ``lfp_p``, ``beta_j`` for a coefficient);
- ``surpyval.utils.score``, a module that raises on import.

An entry is dropped once the old name has been gone for a release or two.
"""

from __future__ import annotations

import dis
import functools
import re
import sys
from typing import Any, Callable, TypeVar

__all__ = [
    "MIXTURE_EM_ATTRIBUTES",
    "REMOVED_ATTRIBUTES",
    "REMOVED_TOP_LEVEL",
    "RemovedName",
    "RemovedNames",
    "column_arguments",
    "from_import",
    "removed_arguments",
    "removed_attributes",
    "removed_message",
    "removed_parameter_note",
]

F = TypeVar("F", bound=Callable[..., Any])

#: Attributes removed from every fitter and model: ``{old: (what to do
#: instead, release it was removed in)}``.
REMOVED_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "param_names": ("use 'parameter_names'", "0.23"),
    "cov_matrix": ("use 'covariance()'", "0.24"),
    "cov": ("use 'covariance()'", "0.24"),
    "se": ("use 'standard_errors()'", "0.24"),
    "loglike": ("use 'neg_ll()', the negative log-likelihood", "0.24"),
    "loglik": ("use 'log_likelihood'", "0.24"),
    "loglik_no_frailty": ("use 'log_likelihood_no_frailty'", "0.24"),
}

#: ``MixtureModel``'s EM steps, which have no public names since v0.24.
MIXTURE_EM_ATTRIBUTES: dict[str, tuple[str, str]] = dict.fromkeys(
    (
        "EM",
        "Q",
        "expectation",
        "maximisation",
        "likelihood",
        "initialise_params",
    ),
    ("the EM steps are internal to fit()", "0.24"),
)

#: The package's top-level names that were removed: ``{old: (what to do
#: instead, release it was removed in)}``.
REMOVED_TOP_LEVEL: dict[str, tuple[str, str]] = {
    **{
        name: (
            f"it is surpyval.life_models.{name} (from surpyval import "
            "life_models)",
            "0.23",
        )
        for name in (
            "DualExponential",
            "DualPower",
            "Eyring",
            "InverseExponential",
            "InverseEyring",
            "InversePower",
            "LifeModel",
            "Linear",
            "Power",
            "PowerExponential",
        )
    },
    "ExponentialLifeModel": (
        "it is surpyval.life_models.Exponential (from surpyval import "
        "life_models; at the top level 'Exponential' is the distribution)",
        "0.23",
    ),
    "NUM": ("use numpy.float64", "0.24"),
    "TINIEST": ("use numpy.finfo(float).tiny", "0.24"),
    "EPS": ("use numpy.sqrt(numpy.finfo(float).eps)", "0.24"),
}


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


def _is_removed(cls: type, name: str) -> bool:
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return isinstance(klass.__dict__[name], RemovedName)
    return False


class RemovedNames:
    """
    The removed attributes of :data:`REMOVED_ATTRIBUTES`, on the base
    classes of every fitter and model: reading one raises an
    ``AttributeError`` naming its replacement and the release it was
    removed in. They are left out of ``dir()``.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> try:
    ...     Weibull.param_names
    ... except AttributeError as error:
    ...     print(error)  # doctest: +NORMALIZE_WHITESPACE
    'Weibull_' object has no attribute 'param_names': it was removed in
    v0.23; use 'parameter_names'
    >>> hasattr(Weibull, "param_names"), "param_names" in dir(Weibull)
    (False, False)
    """

    def __dir__(self) -> list[str]:
        own = getattr(self, "__dict__", {})
        return [
            name
            for name in super().__dir__()
            if name in own or not _is_removed(type(self), name)
        ]


for _old, _entry in REMOVED_ATTRIBUTES.items():
    _descriptor = RemovedName(*_entry)
    _descriptor.__set_name__(RemovedNames, _old)
    setattr(RemovedNames, _old, _descriptor)


def removed_attributes(
    entries: dict[str, tuple[str, str]],
) -> dict[str, RemovedName]:
    """``{old: RemovedName}`` for a class body (``vars().update(...)``
    does not work there; assign each, or pass to ``setattr``)."""
    out = {}
    for old, entry in entries.items():
        descriptor = RemovedName(*entry)
        descriptor.name = old
        out[old] = descriptor
    return out


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
        quoted in the message as given: ``X="'given'"``.

    Returns
    -------
    callable
        A decorator. The decorated function raises Python's ``TypeError``
        for an unexpected keyword argument, with the replacement named,
        when it is given an old name; otherwise it is called as it is.

    Examples
    --------
    >>> from surpyval.utils.removed_names import removed_arguments
    >>> @removed_arguments("0.23", X="'given'")
    ... def cs(x, given):
    ...     return x + given
    >>> cs(1, X=2)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    TypeError: cs() got an unexpected keyword argument 'X': it was removed
    in v0.23; use 'given'
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


def column_arguments(*names: str) -> Callable[[F], F]:
    """
    Decorate a ``fit_from_df`` so that a column passed by the ``fit``
    argument's name (``x="time"``) raises a ``TypeError`` naming the
    ``_col`` argument (``x_col``), as the univariate fitters' ``x=``,
    removed in v0.23, does: every ``fit_from_df`` names its columns so.

    Examples
    --------
    >>> from surpyval.utils.removed_names import column_arguments
    >>> @column_arguments("x")
    ... def fit_from_df(df, x_col):
    ...     return df[x_col]
    >>> fit_from_df({"t": 1}, x="t")  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    TypeError: fit_from_df() got an unexpected keyword argument 'x'; name
    the column with 'x_col'
    """

    def decorate(func: F) -> F:
        where = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                for old in sorted(kwargs.keys() & set(names)):
                    raise TypeError(
                        f"{where}() got an unexpected keyword argument "
                        f"{old!r}; name the column with '{old}_col'"
                    )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


_BETA_J = re.compile(r"beta_(\d+)$")


def removed_parameter_note(name: Any, valid: Any = ()) -> str:
    """
    A note for an unknown parameter's message when ``name`` is a removed
    parameter name, else ``""``: ``p`` for the limited-failure proportion
    ``lfp_p`` and ``beta_j`` for a regression coefficient (both removed in
    v0.24).

    Examples
    --------
    >>> from surpyval.utils.removed_names import removed_parameter_note
    >>> print(removed_parameter_note("beta_1", ["alpha", "beta", "temp"]))
    ... # doctest: +NORMALIZE_WHITESPACE
     ('beta_1' was removed in v0.24: a coefficient is named by its
     covariate's column, else coef_1)
    >>> removed_parameter_note("alpha", ["beta"])
    ''
    """
    if not isinstance(name, str):
        return ""
    valid = list(valid)
    if name == "p" and "lfp_p" in valid:
        return (
            " ('p' was removed in v0.24: the limited-failure proportion is "
            "'lfp_p')"
        )
    match = _BETA_J.match(name)
    if match and name not in valid:
        return (
            f" ({name!r} was removed in v0.24: a coefficient is named by its "
            f"covariate's column, else coef_{match.group(1)})"
        )
    return ""


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
