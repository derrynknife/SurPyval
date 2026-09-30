"""Shape in, shape out: the query-shape rule of every model function.

A function evaluated at query points -- times ``x``, or probabilities
for ``qf`` -- returns a result of the query's shape (Design Principles,
principle 7):

- a scalar query (a Python number or a 0-d array) gives a numpy scalar;
- a 1-D query (list, tuple or array) gives a 1-D array of its length,
  and a 2-D query an array of its 2-D shape;
- an empty query gives an empty array of its shape.

A confidence bound adds its own trailing axis: a two-sided bound has
shape ``query_shape + (2,)`` (``[lower, upper]`` on the last axis), a
one-sided bound ``query_shape``.

Most functions are written for a flat 1-D array of points. Rather than
each handling the other shapes, :func:`flatten_query` hands them the
flat array and a function restoring the query's shape to the result,
and :func:`keeps_query_shape` does the same for a method whose first
argument is the query.
"""

import functools
import inspect
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

F = TypeVar("F", bound=Callable[..., Any])


def flatten_query(
    x: npt.ArrayLike, point_ndim: int = 0
) -> "tuple[npt.NDArray, Callable[..., Any]]":
    """
    The query as a flat array of points, and a function giving a result
    computed on that array the query's shape.

    Parameters
    ----------
    x : array_like
        The query: a scalar, or an array of any shape.
    point_ndim : int, optional
        The number of trailing axes that make up one point: 0 (the
        default) for times or probabilities, 1 for a copula, whose
        points are ``(x1, x2)`` pairs, so an ``(m, 2)`` query is ``m``
        points and a ``(2,)`` query is one.

    Returns
    -------
    flat : ndarray
        The points as a float array of shape ``(n,)`` (``(n, 2)`` for a
        copula's pairs).
    restore : callable
        ``restore(result, axis=0)`` takes a result with one entry per
        point on its axis ``axis`` (the first by default; any other axes
        are its own, such as the ``[lower, upper]`` of a two-sided bound,
        or the covariate rows of a survival tree's grid, whose points are
        on the last axis) and gives it the query's shape in place of that
        axis: a numpy scalar for a scalar query. A result whose axis is
        not one per point (a regression model evaluated at one time for
        several covariate rows, say) is returned unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.shapes import flatten_query
    >>> flat, restore = flatten_query([[1, 2], [3, 4]])
    >>> flat
    array([1., 2., 3., 4.])
    >>> restore(flat ** 2)
    array([[ 1.,  4.],
           [ 9., 16.]])
    >>> flat, restore = flatten_query(3)
    >>> restore(flat * 2)
    np.float64(6.0)
    >>> restore(np.array([[0.1, 0.9]]))  # a two-sided bound at one point
    array([0.1, 0.9])
    """
    # A duration would be read in its storage ticks (#480)
    from surpyval.utils import refuse_time_values

    refuse_time_values(x, "x")
    arr = np.asarray(x, dtype=float)
    if point_ndim:
        shape = arr.shape[: arr.ndim - point_ndim]
        flat = arr.reshape((-1,) + arr.shape[arr.ndim - point_ndim :])
    else:
        shape = arr.shape
        flat = arr.reshape(-1)
    n = flat.shape[0]

    def restore(result: Any, axis: int = 0) -> Any:
        out = np.asarray(result)
        if out.ndim == 0:
            # One value for every point (a constant): the query's shape.
            out = np.broadcast_to(out, shape).copy()
        else:
            ax = axis % out.ndim
            if out.shape[ax] != n:
                return result
            out = out.reshape(out.shape[:ax] + shape + out.shape[ax + 1 :])
        return out[()] if out.ndim == 0 else out

    return flat, restore


def keeps_query_shape(
    method: "F | None" = None, *, point_ndim: int = 0
) -> Any:
    """
    Decorate a method whose first argument is the query so that it sees
    the query flat (see :func:`flatten_query`) and its result has the
    query's shape. The query may be passed by position or by name.
    ``point_ndim`` is as for :func:`flatten_query`: use
    ``@keeps_query_shape(point_ndim=1)`` for a copula's ``(x1, x2)``
    points.

    The wrapper is one more frame between a warning raised in the method
    and the caller: a warning meant for the caller's line needs a
    ``stacklevel`` one higher than without it (3 raised in the method
    itself).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.shapes import keeps_query_shape
    >>> class Model:
    ...     @keeps_query_shape
    ...     def sf(self, x):
    ...         return np.exp(-np.atleast_1d(x))
    >>> Model().sf(0.0)
    np.float64(1.0)
    >>> Model().sf(x=[[0.0], [0.0]])
    array([[1.],
           [1.]])
    """

    def decorate(method: F) -> F:
        name = list(inspect.signature(method).parameters)[1]

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                x, args = args[0], args[1:]
            elif name in kwargs:
                x = kwargs.pop(name)
            else:
                # No query: the method's own default, or its own error.
                return method(self, *args, **kwargs)
            if x is None:
                # A query left to its default (the fitted times, say).
                return method(self, x, *args, **kwargs)
            flat, restore = flatten_query(x, point_ndim)
            return restore(method(self, flat, *args, **kwargs))

        return wrapper  # type: ignore[return-value]

    return decorate if method is None else decorate(method)
