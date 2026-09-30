"""Conditional survival, ``sf(x, given=g)`` (#514).

The probability of surviving to ``x`` for a unit known to have survived to
``g``:

.. math::
    S(x \\mid T > g) = \\frac{S(x)}{S(g)} \\quad (x > g),

and 1 for ``x <= g``, the same meaning as the regression models'
``sf_tvc(..., given=)``. Its complement is the conditional failure
probability, :math:`(F(x) - F(g)) / S(g)`, computed from ``F`` so that it
keeps its precision where ``F`` is small. Where :math:`S(g) = 0` (the unit
cannot have survived to ``g``) or either time is missing the value is
``nan``.

``x`` is the unit's age, as for the unconditional functions; ``cs(x, X)``
takes the further time instead: ``cs(x, X) = sf(X + x, given=X)``.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def broadcast_given(
    x: npt.ArrayLike, given: npt.ArrayLike
) -> "tuple[npt.NDArray, npt.NDArray]":
    """
    ``x`` and ``given`` as float arrays of their common shape.

    Raises a ``ValueError`` naming ``given`` when the shapes do not
    broadcast (principle 2).

    Examples
    --------
    >>> from surpyval.utils.conditional import broadcast_given
    >>> x, g = broadcast_given([[1, 2], [3, 4]], [0, 1])
    >>> g
    array([[0., 1.],
           [0., 1.]])
    """
    from surpyval.utils import refuse_time_values

    refuse_time_values(x, "x")
    refuse_time_values(given, "given")
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(given, dtype=float)
    try:
        xb, gb = np.broadcast_arrays(x_arr, g_arr)
    except ValueError:
        raise ValueError(
            "given must be a scalar or an array that broadcasts against x; "
            f"got given of shape {g_arr.shape} for x of shape {x_arr.shape}"
        ) from None
    return xb, gb


def _undefined(x: Any, g: Any, s_g: Any) -> Any:
    return np.isnan(x) | np.isnan(g) | ~(s_g > 0)


def conditional_sf(
    sf: Callable[[Any], Any], x: npt.ArrayLike, given: npt.ArrayLike
) -> Any:
    """
    :math:`S(x \\mid T > g)` from the unconditional survival function
    ``sf``, with the shape of ``x`` and ``given`` broadcast together.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.utils.conditional import conditional_sf
    >>> model = Weibull.from_params([10, 3])
    >>> conditional_sf(model.sf, [5, 12], 10).round(4)
    array([1.    , 0.4829])
    """
    x_b, g_b = broadcast_given(x, given)
    s_x = np.asarray(sf(x_b), dtype=float)
    s_g = np.asarray(sf(g_b), dtype=float)
    with np.errstate(all="ignore"):
        out = s_x / s_g
    out = np.where(x_b <= g_b, 1.0, out)
    out = np.where(_undefined(x_b, g_b, s_g), np.nan, out)
    return out[()] if out.ndim == 0 else out


def conditional_ff(
    ff: Callable[[Any], Any],
    sf: Callable[[Any], Any],
    x: npt.ArrayLike,
    given: npt.ArrayLike,
) -> Any:
    """
    :math:`1 - S(x \\mid T > g) = (F(x) - F(g)) / S(g)` from the
    unconditional ``ff`` and ``sf``.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.utils.conditional import conditional_ff
    >>> model = Weibull.from_params([10, 3])
    >>> conditional_ff(model.ff, model.sf, [5, 12], 10).round(4)
    array([0.    , 0.5171])
    """
    x_b, g_b = broadcast_given(x, given)
    f_x = np.asarray(ff(x_b), dtype=float)
    f_g = np.asarray(ff(g_b), dtype=float)
    s_g = np.asarray(sf(g_b), dtype=float)
    with np.errstate(all="ignore"):
        out = (f_x - f_g) / s_g
    out = np.where(x_b <= g_b, 0.0, out)
    out = np.where(_undefined(x_b, g_b, s_g), np.nan, out)
    return out[()] if out.ndim == 0 else out
