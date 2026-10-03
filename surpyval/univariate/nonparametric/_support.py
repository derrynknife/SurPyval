"""The support and interpolation helpers of the non-parametric estimates.

Shared by :class:`~surpyval.univariate.nonparametric.nonparametric.\
NonParametric` and the other step-function estimates (the competing-risks
CIFs and the mean cumulative functions): checking and restoring an
explicit support (``set_support``), evaluating an estimate on it, and
interpolating an estimate and its confidence bounds between its times.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def check_support(
    lower: Any, upper: Any, first: float, last: float, what: tuple[str, str]
) -> tuple[float, float]:
    """The ``(lower, upper)`` of ``set_support`` as floats, refused unless
    it is an interval containing the estimate's own range ``[first,
    last]``; ``what`` names that range in the messages."""
    bounds = {}
    for name, value in (("lower", lower), ("upper", upper)):
        try:
            bounds[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "'{}' must be a number; got {!r}.".format(name, value)
            ) from None
        if np.isnan(bounds[name]):
            raise ValueError(
                "'{}' must be a number, not NaN; pass -inf or inf for "
                "no bound on that side.".format(name)
            )
    lo, hi = bounds["lower"], bounds["upper"]
    if not lo < hi:
        raise ValueError(
            "'lower' must be below 'upper'; got lower={} and "
            "upper={}.".format(lo, hi)
        )
    if lo > first:
        raise ValueError(
            "'lower' ({}) is above {} ({}); the bounds must contain it, so "
            "pass a 'lower' of at most {}.".format(lo, what[0], first, first)
        )
    if hi < last:
        raise ValueError(
            "'upper' ({}) is below {} ({}); the bounds must contain it, so "
            "pass an 'upper' of at least {}.".format(hi, what[1], last, last)
        )
    return lo, hi


def support_from_dict(model_dict: dict) -> "tuple[Any, Any] | None":
    """The ``"support"`` a ``to_dict`` wrote (see ``set_support``), as a
    ``(lower, upper)`` pair for the model's ``set_support`` to check, or
    ``None`` when it has none."""
    value = model_dict.get("support")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "The serialised 'support' must be a [lower, upper] pair; "
            "got {!r}.".format(value)
        )
    return value[0], value[1]


def on_support(
    support: tuple[float, float],
    first: float,
    last: float,
    x: npt.ArrayLike,
    f: Callable[[npt.NDArray], npt.ArrayLike],
    start: float,
) -> npt.NDArray:
    """``f`` evaluated with an explicit support (``set_support``).

    ``f`` is the estimate's function as it is without one, and is only
    ever evaluated within ``[first, last]``, the estimate's own range: a
    query in ``[lower, first)`` gets ``start`` (the value before the first
    time), one in ``(last, upper]`` the value at ``last``, carried, and
    one outside ``[lower, upper]`` (or missing) NaN. A result with a
    column per query (a two-sided bound) keeps its columns.
    """
    xf = np.atleast_1d(np.asarray(x, dtype=float))
    lower, upper = support
    inside = (xf >= lower) & (xf <= upper)
    q = np.clip(xf[inside], first, last)
    # Evaluated at one point when nothing is inside, so that the result
    # has the right trailing shape and ``f`` still checks its arguments.
    values = np.asarray(f(q if q.size else np.array([first])), dtype=float)
    out = np.full(xf.shape + values.shape[1:], np.nan)
    out[inside] = values[: q.size]
    out[inside & (xf < first)] = start
    return out


def interp_function(
    x: npt.ArrayLike, y: npt.ArrayLike, kind: str
) -> Callable[[npt.ArrayLike], npt.NDArray]:
    # Collapse any duplicated ``x`` (the zero-width Turnbull bounds at
    # exactly observed times) to the last value there, which is where the
    # step function has settled. PCHIP requires strictly increasing
    # abscissae; ``interp1d`` accepts repeats, but then joins the value
    # before the drop at one exact time to the value after the drop at the
    # previous one, so a Turnbull ``interp='linear'`` curve kept its steps
    # at the exact times instead of interpolating across them as the
    # Kaplan-Meier's does.
    from scipy.interpolate import PchipInterpolator, interp1d

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.append(np.diff(x) > 0, True)
    x, y = x[keep], y[keep]
    if kind == "cubic":
        # A plain cubic spline can overshoot and produce a non-monotone
        # (even out-of-[0, 1]) survival curve, which then propagates into
        # ``Hf``, ``hf`` and the interpolated confidence bounds. PCHIP is a
        # shape-preserving piecewise-cubic Hermite interpolant, so it stays
        # monotone wherever the data are monotone.
        pchip = PchipInterpolator(x, y, extrapolate=False)
        y_arr = np.asarray(y, dtype=float)
        lo, hi = float(np.min(y_arr)), float(np.max(y_arr))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return lambda q: pchip(np.asarray(q, dtype=float))
        # PCHIP never leaves the range of its knots, but its round-off
        # can: evaluated at the last knot of a Kaplan-Meier that ends at
        # 0 it gave sf = -2.3e-17, and so Hf = NaN with a raw "invalid
        # value in log" warning, instead of 0 and inf.
        return lambda q: np.clip(pchip(np.asarray(q, dtype=float)), lo, hi)
    return interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)


def interp_bound(
    xs: npt.NDArray,
    R: npt.NDArray,
    bound: npt.NDArray,
    defined: npt.NDArray,
    x: npt.NDArray,
    kind: str,
    in_unit: bool,
) -> npt.NDArray:
    """A confidence bound of the estimate ``R``, both known at the times
    ``xs``, interpolated to ``x`` with ``kind``.

    Up to the last time it is defined, the bound is the interpolated
    estimate plus its interpolated distance from the estimate, the latter
    over the times where the bound is defined. Where the variance is
    undefined (from the time a Kaplan-Meier reaches zero) the bound is a
    fill (see ``R_cb``), not an estimate, and interpolating the bound
    itself let the fill feed the curve: a PCHIP's slope at a knot depends
    on the next one, and the fill at the last time bent the cubic bounds
    before it, so that they did not close onto ``sf`` as ``alpha_ci -> 1``
    (0.1873 against 0.1948, #417). The distance is 0 where the estimate
    has no variance and is interpolated without leaving the range of its
    knots, so the bounds close onto ``sf`` and hold it between them. For
    the linear, nearest and previous kinds it is the bound interpolated,
    as before. Past the last defined time the fill is interpolated as
    before. ``in_unit`` keeps the result within [0, 1] (the exponential
    bounds).
    """
    everywhere = interp_function(xs, bound, kind=kind)(x)
    if not defined.any():
        return everywhere
    last = int(np.flatnonzero(defined)[-1])
    keep = defined[: last + 1]
    try:
        distance = interp_function(
            xs[: last + 1][keep],
            (bound - R)[: last + 1][keep],
            kind=kind,
        )(x)
    except ValueError:
        # Too few defined times for this ``kind`` (PCHIP needs two).
        return everywhere
    inside = interp_function(xs, R, kind=kind)(x) + distance
    if in_unit:
        inside = np.clip(inside, 0.0, 1.0)
    return np.where(x <= xs[last], inside, everywhere)
