"""
Renamed names: accept the old name for one release, with a warning.

When a public name changes so that the same thing has the same name
everywhere (Design Principles, principle 21), the old one keeps working
until :data:`REMOVED_IN`, with a ``DeprecationWarning`` that names the new
one and points at the caller's line: a name renamed in v0.24 is removed in
v0.25. This module holds the shapes such a rename takes (each with the
v0.23 rename it was written for, removed in v0.24):

- :func:`renamed_arguments`, an argument of a function or method;
- :class:`RenamedAttribute`, an attribute or property of a class;
- :class:`CallableFloat`, a method that became a property returning a
  number (``MixtureModel.log_likelihood``);
- :class:`MethodFloat`, a property returning a number that became a
  method (``CrowAMSAA.aic`` -> ``CrowAMSAA.aic()``);
- :class:`ArrayMethod` (with :class:`MethodArray`), the same for an array
  attribute (``FrailtyModel.covariance`` -> ``FrailtyModel.covariance()``);
- :class:`RenamedToMethod`, an attribute whose value a method of another
  name now gives (``Parametric.cov_matrix`` -> ``covariance()``);
- :class:`MadePrivate`, the public name of an internal method or attribute
  (``MixtureModel.EM`` -> ``MixtureModel._em_iteration``).

A name deprecated in v0.25 is accepted until :data:`REMOVED_IN_NEXT`.
"""

import functools
import warnings
from typing import Any, Callable, TypeVar

import numpy as np

__all__ = [
    "REMOVED_IN",
    "REMOVED_IN_NEXT",
    "ArrayMethod",
    "CallableFloat",
    "MethodArray",
    "MadePrivate",
    "MethodFloat",
    "RenamedAttribute",
    "RenamedToMethod",
    "renamed_arguments",
]

#: The release in which the names deprecated in v0.24 stop being
#: accepted.
REMOVED_IN = "0.25"

#: The release in which the names deprecated in v0.25 stop being accepted.
REMOVED_IN_NEXT = "0.26"

F = TypeVar("F", bound=Callable[..., Any])


def _message(
    where: str, old: str, new: str, removed_in: str = REMOVED_IN
) -> str:
    return (
        f"{where}: '{old}' is deprecated and will be removed in "
        f"v{removed_in}; use '{new}'."
    )


def renamed_arguments(
    removed_in: str = REMOVED_IN, **renames: str
) -> Callable[[F], F]:
    """
    Decorate a function so that it accepts its arguments' old names.

    Parameters
    ----------
    removed_in : str, optional
        The release in which the old names stop being accepted:
        :data:`REMOVED_IN` (the default) for the names renamed in v0.24,
        :data:`REMOVED_IN_NEXT` for those renamed in v0.25.
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
    describe: 'names' is deprecated and will be removed in v0.25;
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
    attribute). A name deprecated in v0.25 passes
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
    Model.loglik is deprecated and will be removed in v0.25;
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


class CallableFloat(float):
    """
    A number that can still be called, for a method that became a
    property.

    ``MixtureModel.log_likelihood(params)`` was one component's
    log-likelihood at ``params``, and ``log_likelihood`` became the fitted
    log-likelihood in v0.23, a property as on every other model. For one
    release it returned one of these, which is a plain ``float`` in every
    other respect, and whose call, the old spelling, warns and calls
    ``old`` (or, with no ``old`` or no arguments, returns the number).

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import CallableFloat
    >>> ll = CallableFloat(-12.5, "Model.log_likelihood")
    >>> ll + 1
    -11.5
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     ll()
    -12.5
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.log_likelihood is now a property: 'log_likelihood()' is
    deprecated and will be removed in v0.25; use 'log_likelihood'.
    """

    _where: str
    _old: "Callable[..., Any] | None"
    _note: str

    def __new__(
        cls,
        value: float,
        where: str = "",
        old: "Callable[..., Any] | None" = None,
        note: str = "",
    ) -> "CallableFloat":
        out = super().__new__(cls, value)
        out._where, out._old, out._note = where, old, note
        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        attr = self._where.rpartition(".")[2]
        warnings.warn(
            "{} is now a property{}: '{}()' is deprecated and will be "
            "removed in v{}; use '{}'.".format(
                self._where, self._note, attr, REMOVED_IN, attr
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        if self._old is not None and (args or kwargs):
            return self._old(*args, **kwargs)
        return float(self)

    def __reduce__(self) -> Any:
        # Pickled (and copied) as the plain number it is
        return (float, (float(self),))


class MethodFloat(float):
    """
    The number of a property that became a method, for the property's
    spelling.

    ``CrowAMSAA.aic`` was a property, and ``aic()`` became a method in
    v0.23, as on every other model. For one release the property returned
    one of these: calling it, the new spelling, returns the plain
    ``float``; using it as a number without the call -- arithmetic,
    comparison, ``round``, ``float``, formatting or printing -- gives that
    number with a ``DeprecationWarning`` pointing at the caller's line.
    (What numpy reads directly, without calling any of these, is the
    number as it is, with no warning.)

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import MethodFloat
    >>> aic = MethodFloat(10.25, "Model.aic")
    >>> aic()
    10.25
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     aic + 1
    11.25
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.aic is now a method: 'aic' without the call is deprecated and
    will be removed in v0.25; use 'aic()'.
    """

    _where: str

    def __new__(cls, value: float, where: str = "") -> "MethodFloat":
        out = super().__new__(cls, value)
        out._where = where
        return out

    def __call__(self) -> float:
        return float.__float__(self)

    def _warn(self) -> None:
        attr = self._where.rpartition(".")[2]
        warnings.warn(
            "{} is now a method: '{}' without the call is deprecated and "
            "will be removed in v{}; use '{}()'.".format(
                self._where, attr, REMOVED_IN, attr
            ),
            DeprecationWarning,
            stacklevel=3,
        )

    def __reduce__(self) -> Any:
        return (float, (float.__float__(self),))


def _warning_operator(name: str) -> Callable[..., Any]:
    base = getattr(float, name)

    def operator(self: MethodFloat, *args: Any) -> Any:
        out = base(self, *args)
        # An operand that is not a number (``obj in (type, object)``, as
        # ``inspect.signature`` asks) is not a use of the number.
        if out is not NotImplemented:
            self._warn()
        return out

    operator.__name__ = name
    return operator


# Every use of the number as a number (``__hash__`` is kept, unwarned, as
# ``__eq__`` is replaced).
for _name in (
    "__abs__",
    "__add__",
    "__bool__",
    "__divmod__",
    "__eq__",
    "__float__",
    "__floordiv__",
    "__format__",
    "__ge__",
    "__gt__",
    "__int__",
    "__le__",
    "__lt__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__pos__",
    "__pow__",
    "__radd__",
    "__rdivmod__",
    "__repr__",
    "__rfloordiv__",
    "__rmod__",
    "__rmul__",
    "__round__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__str__",
    "__sub__",
    "__truediv__",
    "__trunc__",
):
    setattr(MethodFloat, _name, _warning_operator(_name))
MethodFloat.__hash__ = float.__hash__  # type: ignore[method-assign]


def _method_warning(where: str, stacklevel: int) -> None:
    attr = where.rpartition(".")[2]
    warnings.warn(
        "{} is now a method: '{}' without the call is deprecated and will "
        "be removed in v{}; use '{}()'.".format(where, attr, REMOVED_IN, attr),
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def _plain(value: Any) -> Any:
    # A MethodArray as the plain array it is, anything else as it is.
    if isinstance(value, MethodArray):
        return value.view(np.ndarray)
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    return value


class MethodArray(np.ndarray):
    """
    The array of an attribute that became a method of the same name.

    ``FrailtyModel.covariance`` was an array, and ``covariance()`` became
    a method in v0.23, as on every other model (#605). For one release the
    attribute's name gave one of these (see :class:`ArrayMethod`): calling
    it, the new spelling, returns the plain array; using it as an array
    without the call -- indexing, arithmetic, a numpy function or printing
    -- gives the same result with a ``DeprecationWarning`` pointing at the
    caller's line. (What numpy reads directly, such as ``shape``, gives it
    with no warning.)

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from surpyval.utils.deprecation import MethodArray
    >>> cov = MethodArray(np.eye(2), "Model.covariance")
    >>> cov()
    array([[1., 0.],
           [0., 1.]])
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     np.diag(cov)
    array([1., 1.])
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.covariance is now a method: 'covariance' without the call is
    deprecated and will be removed in v0.25; use 'covariance()'.
    """

    _where: str

    def __new__(cls, value: Any, where: str = "") -> "MethodArray":
        out = np.asarray(value).view(cls)
        out._where = where
        return out

    def __array_finalize__(self, obj: Any) -> None:
        self._where = getattr(obj, "_where", "")

    def __call__(self) -> np.ndarray:
        return self.view(np.ndarray)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        _method_warning(self._where, 2)
        if "out" in kwargs:
            kwargs["out"] = _plain(kwargs["out"])
        return getattr(ufunc, method)(*_plain(inputs), **kwargs)

    def __array_function__(
        self, func: Any, types: Any, args: Any, kwargs: Any
    ) -> Any:
        _method_warning(self._where, 2)
        return func(*_plain(args), **{k: _plain(v) for k, v in kwargs.items()})

    def __getitem__(self, key: Any) -> Any:
        _method_warning(self._where, 2)
        return self.view(np.ndarray)[key]

    def __iter__(self) -> Any:
        _method_warning(self._where, 2)
        return iter(self.view(np.ndarray))

    def __repr__(self) -> str:
        _method_warning(self._where, 2)
        return repr(self.view(np.ndarray))

    def __str__(self) -> str:
        _method_warning(self._where, 2)
        return str(self.view(np.ndarray))

    def __reduce__(self) -> Any:
        # Pickled (and copied) as the plain array it is
        return self.view(np.ndarray).__reduce__()


class _Unavailable:
    """What :class:`ArrayMethod` gives when the model holds no array: its
    call, the new spelling, raises the model's own error."""

    def __init__(self, where: str, error: Callable[[], Exception]) -> None:
        self._where, self._error = where, error

    def __call__(self) -> Any:
        raise self._error()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<{}: not available>".format(self._where)


class ArrayMethod:
    """
    An array attribute that became a method of the same name.

    Declare it on the class under that name, with the attribute the array
    is now stored in and the error the call raises when there is none:
    ``covariance = ArrayMethod("_covariance", no_covariance_error)``.
    ``model.covariance()`` gives the array; ``model.covariance`` without
    the call, the old spelling, still works as the array (a
    :class:`MethodArray`) with a ``DeprecationWarning`` until
    :data:`REMOVED_IN`. Setting the old attribute sets the stored
    array, with the warning.
    """

    def __init__(self, stored: str, error: Callable[[], Exception]) -> None:
        self.stored = stored
        self.error = error
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, owner: "type | None" = None) -> Any:
        if obj is None:
            return self
        where = "{}.{}".format(type(obj).__name__, self.name)
        value = getattr(obj, self.stored, None)
        if value is None:
            return _Unavailable(where, self.error)
        return MethodArray(value, where)

    def __set__(self, obj: Any, value: Any) -> None:
        _method_warning("{}.{}".format(type(obj).__name__, self.name), 2)
        setattr(obj, self.stored, value)


class RenamedToMethod:
    """
    An attribute whose value a method of another name now gives.

    ``Parametric.cov_matrix`` and ``FineGrayModel.cov`` were the
    covariance, which ``covariance()`` gives on every model since v0.23
    (#605).
    Declare the old name on the class, ``cov_matrix =
    RenamedToMethod("covariance", "_covariance")``: reading it warns and
    returns ``covariance()`` (``None`` where that raises a ``ValueError``,
    as a model without one had); setting it warns and sets the stored
    attribute.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import RenamedToMethod
    >>> class Model:
    ...     _covariance = [[1.0]]
    ...     cov = RenamedToMethod("covariance", "_covariance")
    ...     def covariance(self):
    ...         return self._covariance
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().cov
    [[1.0]]
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.cov is deprecated and will be removed in v0.25; use
    'covariance()'.
    """

    def __init__(self, new: str, stored: str) -> None:
        self.new = new
        self.stored = stored
        self.old = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.old = name

    def _warn(self, owner: type) -> None:
        warnings.warn(
            "{}.{} is deprecated and will be removed in v{}; use "
            "'{}()'.".format(owner.__name__, self.old, REMOVED_IN, self.new),
            DeprecationWarning,
            stacklevel=3,
        )

    def __get__(self, obj: Any, owner: "type | None" = None) -> Any:
        if obj is None:
            return self
        self._warn(type(obj))
        try:
            return getattr(obj, self.new)()
        except ValueError:
            return None

    def __set__(self, obj: Any, value: Any) -> None:
        self._warn(type(obj))
        setattr(obj, self.stored, value)


class MadePrivate(RenamedAttribute):
    """
    The public name of an internal method or attribute, now private.

    Declare it on the class under the public name, ``EM =
    MadePrivate("_em_iteration")``: reading it warns that it is internal
    and will be removed in :data:`REMOVED_IN`, and gives the private
    one. There is no public replacement to name.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.deprecation import MadePrivate
    >>> class Model:
    ...     def _step(self):
    ...         return 1
    ...     step = MadePrivate("_step")
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Model().step()
    1
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    Model.step is internal to the fit; its public name is deprecated and
    will be removed in v0.25.
    """

    def __init__(self, private: str) -> None:
        super().__init__(private)

    def _warn(self, owner: type) -> None:
        warnings.warn(
            "{}.{} is internal to the fit; its public name is deprecated "
            "and will be removed in v{}.".format(
                owner.__name__, self.old, self.removed_in
            ),
            DeprecationWarning,
            stacklevel=3,
        )
