"""
A two-sided confidence bound that says which column is which.

A two-sided bound is an array with ``[lower, upper]`` on its last axis
(see :doc:`Conventions`). :class:`ConfidenceBounds` is that same array,
so indexing, ``np.asarray``, arithmetic and ``tolist`` work as before,
with the two columns also given by name (``.lower``, ``.upper``) and a
printout that labels them::

    >>> import numpy as np
    >>> from surpyval.utils.bounds import ConfidenceBounds
    >>> b = ConfidenceBounds.wrap(np.array([2.70, 4.97]), alpha_ci=0.05)
    >>> b
    ConfidenceBounds(lower=2.7, upper=4.97, level=95%)
    >>> b.lower, b.upper
    (np.float64(2.7), np.float64(4.97))
    >>> b[1]
    np.float64(4.97)

Anything computed from it (``b * 2``, ``b[:, 0]``, ``b.mean()``) is a
plain array again: only the bound itself carries the labels.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

F = TypeVar("F", bound=Callable[..., Any])


class ConfidenceBounds(np.ndarray):
    """
    A two-sided confidence bound: ``[lower, upper]`` on the last axis.

    An ``ndarray`` subclass, so code that reads ``bounds[..., 0]`` and
    ``bounds[..., 1]`` is unchanged.

    Attributes
    ----------
    lower : numpy array or float
        The lower bound at each query (``self[..., 0]``).
    upper : numpy array or float
        The upper bound at each query (``self[..., 1]``).
    alpha_ci : float or None
        The total tail probability of the interval, when known.
    """

    alpha_ci: float | None

    @classmethod
    def wrap(
        cls, values: npt.ArrayLike, alpha_ci: float | None = None
    ) -> ConfidenceBounds:
        """``values`` (last axis ``[lower, upper]``) as labelled bounds."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0 or arr.shape[-1] != 2:
            raise ValueError(
                "A two-sided bound has [lower, upper] on its last axis; "
                f"got shape {arr.shape}."
            )
        out = arr.view(cls)
        out.alpha_ci = alpha_ci
        return out

    def __array_finalize__(self, obj: Any) -> None:
        self.alpha_ci = getattr(obj, "alpha_ci", None)

    def __array_wrap__(
        self,
        obj: Any,
        context: Any = None,
        return_scalar: bool = False,
    ) -> Any:
        # Results of ufuncs and reductions are plain arrays (or scalars):
        # a sum or a scaled bound is not a [lower, upper] pair.
        arr = np.asarray(obj).view(np.ndarray)
        if return_scalar:
            return arr[()]
        return arr

    def __getitem__(self, key: Any) -> Any:
        out = super().__getitem__(key)
        # A slice that is still a whole [lower, upper] pair (some of the
        # queries) keeps the labels; a column or a single value does not.
        if isinstance(out, ConfidenceBounds) and not self._is_pair(out):
            return out.view(np.ndarray)
        return out

    @staticmethod
    def _is_pair(arr: np.ndarray) -> bool:
        return arr.ndim > 0 and arr.shape[-1] == 2

    def _column(self, k: int) -> Any:
        col = np.asarray(self).view(np.ndarray)[..., k]
        return col[()] if col.ndim == 0 else col

    @property
    def lower(self) -> Any:
        """The lower bound at each query."""
        return self._column(0)

    @property
    def upper(self) -> Any:
        """The upper bound at each query."""
        return self._column(1)

    def __reduce__(self) -> Any:
        # Pickle as the plain array plus the level.
        return (
            _rebuild,
            (np.asarray(self).view(np.ndarray).copy(), self.alpha_ci),
        )

    def __repr__(self) -> str:
        if not self._is_pair(self):
            return repr(self.view(np.ndarray))
        lower, upper = self.lower, self.upper
        level = (
            ""
            if self.alpha_ci is None
            else f", level={_percent(1 - self.alpha_ci)}"
        )
        return (
            f"ConfidenceBounds(lower={_show(lower)}, "
            f"upper={_show(upper)}{level})"
        )


def _rebuild(values: np.ndarray, alpha_ci: float | None) -> ConfidenceBounds:
    return ConfidenceBounds.wrap(values, alpha_ci)


def _show(v: Any) -> str:
    # numpy's own printing (8 significant digits), for one value too.
    return np.array2string(np.asarray(v, dtype=float), separator=", ")


def _percent(p: float) -> str:
    return f"{100 * p:.10g}%"


def labels_two_sided(method: F) -> F:
    """
    Decorate a ``*_cb`` method so that its two-sided result is a
    :class:`ConfidenceBounds`. The method takes ``bound`` (default
    ``"two-sided"``) and ``alpha_ci``; a one-sided result is returned as
    it is.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = method(*args, **kwargs)
        try:
            bound_args = signature.bind(*args, **kwargs)
        except TypeError:
            return out
        bound_args.apply_defaults()
        if bound_args.arguments.get("bound") != "two-sided":
            return out
        return ConfidenceBounds.wrap(out, bound_args.arguments.get("alpha_ci"))

    return wrapper  # type: ignore[return-value]
