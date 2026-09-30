r"""
Continuously varying covariate paths for time-varying-covariate
*evaluation* (#172).

A :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
describes a covariate that is constant between change-points, and along it
each family's cumulative hazard is an exact sum over the segments. A
:class:`CovariatePath` describes a covariate that changes *continuously*:
a ramp, a ramp-hold-step test profile, a thermal cycle, or a densely
sampled measured path. Along it the cumulative hazard is an integral,

.. math::
    H(t) = \int_0^t h\bigl(u \mid Z(u)\bigr)\, du ,

which ``sf_tvc`` / ``Hf_tvc`` evaluate by adaptive Gauss-Kronrod
quadrature (for accelerated failure time, the accelerated age
:math:`\psi(t) = \int_0^t \phi(Z(u))\,du` is integrated instead, and the
baseline evaluated once at it). Cox needs no quadrature: its baseline is
a step function, so only the covariate at the baseline's jump times
matters.

The quadrature integrates, on each panel, only the *difference* between
the hazard along the path and the hazard with the covariate frozen at the
panel's midpoint, and adds the exact frozen-covariate increment
``Hf(b, z) - Hf(a, z)``. On a flat stretch the difference is exactly
zero, so a flat path gives the step schedule's sum: the two are one
method.

The path is an *external* covariate, known in advance (a planned load, a
test profile, ambient conditions): the survival along it is a probability
only when the path does not depend on the item's own failure process.
Fitting a model on continuous paths is not provided; measured covariates
are fitted as steps with ``fit_tvc``.
"""

import math
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

__all__ = ["CovariatePath"]


# Gauss-Kronrod 7/15 on [-1, 1]: the Kronrod nodes and weights, and the
# weights of the embedded 7-point Gauss rule on the same nodes (zero on the
# Kronrod-only nodes), from QUADPACK's qk15.
_XGK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.0,
    ]
)
_WGK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_WG = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)
_NODES = np.concatenate([-_XGK[:-1], _XGK[::-1]])
_W_KRONROD = np.concatenate([_WGK[:-1], _WGK[::-1]])
# The Gauss nodes are the odd-indexed Kronrod nodes (and 0).
_W_GAUSS = np.zeros(15)
_W_GAUSS[[1, 3, 5]] = _WG[:3]
_W_GAUSS[[13, 11, 9]] = _WG[:3]
_W_GAUSS[7] = _WG[3]

#: A path whose knots and query times need more panels than this raises a
#: ``ValueError`` (it repeats too often for the horizon asked); refinement
#: stops before passing it, and warns.
_PANEL_CAP = 10**6
#: Rounds of bisection before the accuracy target is declared missed.
#: Each halves the panels near an undeclared jump, so 50 rounds close in
#: on one to rounding error.
_MAX_ROUNDS = 50
#: Geometric panels ``e_1 2^{-k}`` towards 0, for hazards singular there
#: (a Weibull shape below 1, say).
_GRADING = 40
#: The rounding floor: a panel whose error estimate is below this many
#: machine epsilons of its integral of ``|h|`` cannot be improved.
_ROUNDING = 50 * np.finfo(float).eps


def _as_rows(values: Any, name: str) -> npt.NDArray:
    """``values`` as a finite ``(m, p)`` float array, or a ValueError
    naming the argument."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'{}' must be numeric: {}".format(name, exc)
        ) from exc
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            "'{}' must be one value per time, shape (m,) or (m, p); got "
            "shape {}".format(name, arr.shape)
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "'{}' has a missing or infinite value; the covariate path of "
            "one unit must be complete".format(name)
        )
    return arr


def _as_period(period: "float | None") -> "float | None":
    if period is None:
        return None
    try:
        value = float(period)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'period' must be a positive number, got {!r}".format(period)
        ) from exc
    if not (np.isfinite(value) and value > 0):
        raise ValueError(
            "'period' must be a positive number, got {!r}".format(period)
        )
    return value


class CovariatePath:
    r"""
    A continuously varying covariate path ``Z(t)`` for time-varying-covariate
    evaluation: ``model.sf_tvc(x, path)`` and ``model.Hf_tvc(x, path)``.

    Build one with a class method rather than the raw constructor:

    * :meth:`from_points` -- straight lines between ``(time, value)``
      points, a repeated time being a jump (ramps, ramp-hold-step test
      profiles, sampled measurements);
    * :meth:`from_callable` -- a vectorised function of time (a thermal
      cycle, a fitted trend).

    Either can repeat with a ``period``. Calling the path, ``path(t)``,
    returns its values, one row per time, shape ``(n, p)``; at a jump the
    value *at* the jump time is the one before it (the left value), as for
    a covariate recorded in ``(start, stop]`` rows.

    Where :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
    describes a piecewise-constant path, whose survival is an exact sum
    over the segments, a ``CovariatePath`` is integrated by adaptive
    quadrature to a relative accuracy of about ``1e-10`` on the cumulative
    hazard (exactly, with no quadrature, for Cox). The path is measured from
    time ``0``: before its first time the first value applies, after its
    last the last value.

    The path must be *external* -- known in advance and not driven by the
    unit's own failure process (a planned load, a test profile, ambient
    conditions) -- for the survival along it to be a probability. Models are
    still fitted on step paths (``fit_tvc``); a ``CovariatePath`` only
    evaluates a fitted model.

    Examples
    --------
    A stress ramped from 0 to 1 over 50 hours, then stepped down to 0.6:

    >>> import numpy as np
    >>> from surpyval import CovariatePath
    >>> ramp = CovariatePath.from_points([0, 50, 50], [0.0, 1.0, 0.6])
    >>> ramp
    CovariatePath(points, 2 knot(s), p=1)
    >>> ramp([25, 50, 80]).ravel()
    array([0.5, 1. , 0.6])

    The survival of a fitted proportional hazards model along the ramp
    lies between those at the two ends of the stress range:

    >>> from surpyval import WeibullPH
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.uniform(0, 1, (200, 1))
    >>> x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
    >>> model = WeibullPH.fit(x, Z)
    >>> model.sf_tvc([40, 80], ramp).round(4)
    array([0.8079, 0.4059])
    >>> model.sf(np.array([40, 80]), [0.0]).round(4)
    array([0.8492, 0.5122])
    """

    def __init__(
        self,
        evaluate: Callable[[npt.NDArray, bool], npt.NDArray],
        p: int,
        knots: npt.NDArray,
        period: "float | None",
        kind: str,
    ) -> None:
        # Private: build a path with from_points or from_callable.
        self._evaluate = evaluate
        self._p = int(p)
        self._knots = np.asarray(knots, dtype=float)
        self.period = period
        self._kind = kind

    @property
    def p(self) -> int:
        """Number of covariates (columns of the path's values)."""
        return self._p

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        times: npt.ArrayLike,
        values: npt.ArrayLike,
        period: "float | None" = None,
    ) -> "CovariatePath":
        """
        A piecewise-linear path through ``(time, value)`` points.

        Between two points the covariate changes linearly; a time given
        twice is a jump from the first value to the second. Before the
        first point the first value applies, and after the last point the
        last value. With ``period`` the pattern repeats: the path at ``t``
        is the pattern at ``t mod period``, so ``times`` must lie in
        ``[0, period]`` (a sawtooth or a thermal cycle).

        Parameters
        ----------
        times : array_like, shape (m,)
            The times of the points, non-decreasing; a time may appear at
            most twice (the two sides of a jump).
        values : array_like, shape (m,) or (m, p)
            The covariate row at each point.
        period : float, optional
            The repetition period, if the pattern repeats.

        Examples
        --------
        A load ramped up over 10 hours, held, then dropped at 30:

        >>> from surpyval import CovariatePath
        >>> path = CovariatePath.from_points([0, 10, 30, 30],
        ...                                  [0.0, 2.0, 2.0, 0.5])
        >>> path([5, 20, 30, 40]).ravel()
        array([1. , 2. , 2. , 0.5])

        A sawtooth, rising from 0 to 1 every 24 hours:

        >>> saw = CovariatePath.from_points([0, 24], [0.0, 1.0], period=24)
        >>> saw([6, 30, 54]).ravel()
        array([0.25, 0.25, 0.25])
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        if t.ndim != 1:
            raise ValueError(
                "'times' must be one-dimensional; got shape {}".format(t.shape)
            )
        if t.size == 0:
            raise ValueError("'times' must hold at least one point")
        if not np.isfinite(t).all():
            raise ValueError(
                "'times' has a missing or infinite value; the covariate "
                "path of one unit must be complete"
            )
        v = _as_rows(values, "values")
        if v.shape[0] != t.shape[0]:
            raise ValueError(
                "'times' and 'values' must have the same length ({} vs "
                "{})".format(t.shape[0], v.shape[0])
            )
        step = np.diff(t)
        if np.any(step < 0):
            raise ValueError("'times' must be non-decreasing")
        if np.any((step[1:] == 0) & (step[:-1] == 0)):
            raise ValueError(
                "a time appears more than twice in 'times'; give a jump as "
                "the same time twice, once with the value before it and "
                "once with the value after"
            )
        period = _as_period(period)
        if period is not None and (t[0] < 0 or t[-1] > period):
            raise ValueError(
                "with a period, 'times' describe one repetition and must "
                "lie in [0, period] = [0, {:g}]".format(period)
            )
        m = t.shape[0]

        def evaluate(u: npt.NDArray, left: bool) -> npt.NDArray:
            # Linear interpolation that returns the left (or right) value
            # at a jump: searchsorted on the chosen side finds the two
            # points around u, which are the two sides of the jump when u
            # is one.
            j = np.searchsorted(t, u, side="left" if left else "right")
            lo = np.clip(j - 1, 0, m - 1)
            hi = np.clip(j, 0, m - 1)
            tl, th = t[lo], t[hi]
            width = th - tl
            with np.errstate(invalid="ignore", divide="ignore"):
                w = np.where(width > 0, (u - tl) / np.where(width > 0, width, 1.0), 0.0)
            w = np.clip(w, 0.0, 1.0)[:, None]
            vl, vh = v[lo], v[hi]
            # Exactly the point's value at a point, and exactly the flat
            # value on a flat stretch.
            return np.where(w == 1.0, vh, vl + w * (vh - vl))

        return cls(evaluate, v.shape[1], np.unique(t), period, "points")

    @classmethod
    def from_callable(
        cls,
        func: Callable[[npt.NDArray], Any],
        p: int = 1,
        breakpoints: "npt.ArrayLike | None" = None,
        period: "float | None" = None,
    ) -> "CovariatePath":
        """
        A path given by a vectorised function of time.

        ``func`` takes a one-dimensional array of ``n`` times and returns
        the covariate at each, shape ``(n,)`` (for one covariate) or
        ``(n, p)``; a scalar is taken as constant. Its values must be
        finite. The quadrature finds kinks and jumps by refining around
        them, but list any you know in ``breakpoints`` so the panels line
        up with them: it is faster and certain. With ``period`` the path at
        ``t`` is ``func(t mod period)``, so ``func`` need only describe one
        period, and ``breakpoints`` lie in ``[0, period]``.

        Parameters
        ----------
        func : callable
            ``func(t) -> values`` for an array ``t``.
        p : int, optional
            Number of covariates. Default 1.
        breakpoints : array_like, optional
            Times at which ``func`` has a kink or a jump.
        period : float, optional
            The repetition period, if the path repeats.

        Examples
        --------
        A daily thermal cycle between 0 and 1:

        >>> import numpy as np
        >>> from surpyval import CovariatePath
        >>> cycle = CovariatePath.from_callable(
        ...     lambda t: 0.5 - 0.5 * np.cos(2 * np.pi * t / 24), period=24)
        >>> cycle([0, 12, 36]).ravel().round(6)
        array([0., 1., 1.])

        A stress that climbs until 100 hours and is then held:

        >>> climb = CovariatePath.from_callable(
        ...     lambda t: np.sqrt(np.minimum(t, 100.0)) / 10,
        ...     breakpoints=[100.0])
        >>> climb([25, 100, 400]).ravel()
        array([0.5, 1. , 1. ])
        """
        if not callable(func):
            raise ValueError(
                "'func' must be a callable taking an array of times; got "
                "{!r}".format(type(func).__name__)
            )
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise ValueError("'p' must be a positive integer, got {!r}".format(p))
        if p < 1:
            raise ValueError("'p' must be a positive integer, got {!r}".format(p))
        p = int(p)
        period = _as_period(period)
        if breakpoints is None:
            knots = np.empty(0)
        else:
            knots = np.atleast_1d(np.asarray(breakpoints, dtype=float))
            if knots.ndim != 1 or not np.isfinite(knots).all():
                raise ValueError(
                    "'breakpoints' must be a one-dimensional list of finite "
                    "times"
                )
            if period is not None and (
                np.any(knots < 0) or np.any(knots > period)
            ):
                raise ValueError(
                    "with a period, 'breakpoints' lie within one period, in "
                    "[0, period] = [0, {:g}]".format(period)
                )
            knots = np.unique(knots)

        def evaluate(u: npt.NDArray, left: bool) -> npt.NDArray:
            with np.errstate(all="ignore"):
                out = np.asarray(func(u), dtype=float)
            n = u.shape[0]
            if out.ndim == 0:
                out = np.full((n, p), float(out))
            elif out.shape == (n,) and p == 1:
                out = out.reshape(n, 1)
            elif out.shape != (n, p):
                raise ValueError(
                    "'func' must return one value per time, shape ({n},) "
                    "or ({n}, {p}) for {n} times and p={p}; it returned "
                    "shape {shape}".format(n=n, p=p, shape=out.shape)
                )
            bad = ~np.isfinite(out).all(axis=1)
            if bad.any():
                raise ValueError(
                    "'func' returned a missing or infinite value at t = "
                    "{:g}; the covariate path must be finite wherever it "
                    "is evaluated".format(float(u[np.argmax(bad)]))
                )
            return out

        return cls(evaluate, p, knots, period, "callable")

    # -- evaluation ---------------------------------------------------------

    def _values(self, t: npt.ArrayLike, left: bool = True) -> npt.NDArray:
        """The path at the times ``t``, ``(n, p)``: the value before a jump
        at the jump time (``left``), or the value after it."""
        u = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        if self.period is not None:
            r = np.mod(u, self.period)
            if left:
                # The end of one repetition, not the start of the next,
                # for the value before a jump at a multiple of the period.
                r = np.where(r == 0, self.period, r)
            u = r
        return self._evaluate(u, left)

    def __call__(self, t: npt.ArrayLike) -> npt.NDArray:
        """
        The covariate at the times ``t``: one row per time, shape
        ``(n, p)``. A missing (``NaN``) time gives a row of ``NaN``.

        Examples
        --------
        >>> from surpyval import CovariatePath
        >>> CovariatePath.from_points([0, 10], [[0.0, 1.0], [1.0, 3.0]])([5])
        array([[0.5, 2. ]])
        """
        u = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        out = np.full((u.shape[0], self.p), np.nan)
        known = ~np.isnan(u)
        if known.any():
            out[known] = self._values(u[known])
        return out

    def _n_breakpoints(self, t_max: float) -> int:
        """How many breakpoints :meth:`breakpoints` would return up to
        ``t_max``, without building them."""
        if self.period is None:
            return int(np.sum((self._knots > 0) & (self._knots < t_max)))
        per_period = np.unique(
            np.mod(np.concatenate([[0.0], self._knots]), self.period)
        ).size
        return math.ceil(t_max / self.period) * per_period

    def breakpoints(self, t_max: float) -> npt.NDArray:
        """
        The path's knots (for :meth:`from_points`) or declared breakpoints
        (for :meth:`from_callable`) in ``(0, t_max)``, repeated every period
        for a periodic path, with the period boundaries.

        Examples
        --------
        >>> from surpyval import CovariatePath
        >>> CovariatePath.from_points([0, 5], [0.0, 1.0], period=8).breakpoints(20)
        array([ 5.,  8., 13., 16.])
        """
        t_max = float(t_max)
        if self.period is None:
            k = self._knots
        else:
            pattern = np.unique(np.concatenate([[0.0], self._knots]))
            starts = self.period * np.arange(
                math.ceil(t_max / self.period) + 1
            )
            k = np.add.outer(starts, pattern).ravel()
        return np.unique(k[(k > 0) & (k < t_max)])

    def __repr__(self) -> str:
        count = "{} {}(s)".format(
            self._knots.size,
            "knot" if self._kind == "points" else "breakpoint",
        )
        period = (
            ", period={:g}".format(self.period) if self.period is not None else ""
        )
        return "CovariatePath({}, {}, p={}{})".format(
            self._kind, count, self.p, period
        )


# -- the quadrature engine ---------------------------------------------------
#
# Families supply ``panel_terms(a, b)``: for panels [a_i, b_i] it returns the
# exact frozen-covariate increment on each, the correction integrand at the
# 15 Kronrod nodes of each, the |integrand| there (for the rounding floor)
# and an optional per-panel flag. The engine knows nothing about families.


def _evaluate_panels(
    panel_terms: Callable, a: npt.NDArray, b: npt.NDArray
) -> "tuple[npt.NDArray, ...]":
    exact, g, scale, flag = panel_terms(a, b)
    half = 0.5 * (b - a)
    kronrod = half * (g @ _W_KRONROD)
    gauss = half * (g @ _W_GAUSS)
    err = np.abs(kronrod - gauss)
    noise = _ROUNDING * half * (scale @ _W_KRONROD)
    return exact + kronrod, err, noise, flag


def path_mesh(path: CovariatePath, points: npt.NDArray) -> npt.NDArray:
    """
    The starting panel edges for integrating ``path`` up to the largest of
    the positive ``points``: 0, the path's breakpoints below it, the points
    themselves (query times, and the conditioning age) and geometric edges
    towards 0.
    """
    t_max = float(np.max(points))
    count = path._n_breakpoints(t_max) + points.size + _GRADING
    if count > _PANEL_CAP:
        raise ValueError(
            "evaluating this CovariatePath up to t = {:g} needs about {:,} "
            "quadrature panels, more than the limit of {:,}{}. Ask for "
            "fewer or earlier times, or describe a path that changes this "
            "often by its average over a period.".format(
                t_max,
                count,
                _PANEL_CAP,
                (
                    " (the path repeats every {:g})".format(path.period)
                    if path.period is not None
                    else ""
                ),
            )
        )
    edges = np.unique(
        np.concatenate([[0.0], path.breakpoints(t_max), points])
    )
    grading = edges[1] * 0.5 ** np.arange(1, _GRADING + 1)
    return np.unique(np.concatenate([edges, grading]))


def _missed(
    value: npt.NDArray,
    err: npt.NDArray,
    noise: npt.NDArray,
    rtol: float,
) -> npt.NDArray:
    """Per panel end: whether the error estimate accumulated from 0
    exceeds ``rtol`` of the integral of ``|h|`` from 0 (and the rounding
    floor accumulated with it)."""
    with np.errstate(invalid="ignore", over="ignore"):
        allowed = np.maximum(
            rtol * np.cumsum(np.abs(value)), np.cumsum(noise)
        )
        return np.cumsum(err) > allowed


def integrate_panels(
    panel_terms: Callable,
    edges: npt.NDArray,
    rtol: float,
    max_rounds: "int | None" = None,
) -> "dict[str, Any]":
    """
    Integrate over the panels between ``edges``, refining until the error
    estimate accumulated from 0 is within ``rtol`` of the integral at every
    panel end.

    Each round bisects every panel whose error estimate ``|K15 - G7|``
    exceeds ``rtol`` times the larger of its own increment and its share of
    the cumulative integral (and the rounding floor), among the panels
    before the last panel end where the accumulated target is missed; if
    none does, every panel there above the rounding floor. The share is
    ``cumulative * width / end``, so the panels graded towards 0 get
    their proportion. Refinement stops after ``max_rounds`` rounds
    (default ``_MAX_ROUNDS``) or before it would pass ``_PANEL_CAP``
    panels; ``limit`` then says which, for the warning.

    Returns the final ``edges`` and, per panel, the integral ``value``, the
    error estimate ``err``, the rounding floor ``noise`` and the ``flag``
    from ``panel_terms``.
    """
    rounds = _MAX_ROUNDS if max_rounds is None else max_rounds
    a, b = edges[:-1], edges[1:]
    value, err, noise, flag = _evaluate_panels(panel_terms, a, b)
    limit = "rounds"
    for _ in range(rounds):
        missed = _missed(value, err, noise, rtol)
        if not missed.any():
            break
        # The panels up to the last end that misses the target.
        before = np.arange(a.size) <= np.flatnonzero(missed)[-1]
        with np.errstate(invalid="ignore", over="ignore"):
            share = np.cumsum(np.abs(value)) * (b - a) / b
            allowed = rtol * np.maximum(np.abs(value), share)
            bad = before & (err > np.maximum(allowed, noise))
            if not bad.any():
                bad = before & (err > noise)
        if not bad.any():
            break
        if a.size + int(bad.sum()) > _PANEL_CAP:
            limit = "panels"
            break
        mid = 0.5 * (a[bad] + b[bad])
        new_a = np.concatenate([a[bad], mid])
        new_b = np.concatenate([mid, b[bad]])
        n_value, n_err, n_noise, n_flag = _evaluate_panels(
            panel_terms, new_a, new_b
        )
        keep = ~bad
        order = np.argsort(np.concatenate([a[keep], new_a]), kind="stable")
        a = np.concatenate([a[keep], new_a])[order]
        b = np.concatenate([b[keep], new_b])[order]
        value = np.concatenate([value[keep], n_value])[order]
        err = np.concatenate([err[keep], n_err])[order]
        noise = np.concatenate([noise[keep], n_noise])[order]
        flag = np.concatenate([flag[keep], n_flag])[order]
    return {
        "edges": np.concatenate([a, b[-1:]]),
        "value": value,
        "err": err,
        "noise": noise,
        "flag": flag,
        "limit": limit,
    }


def sum_between(
    edges: npt.NDArray,
    per_panel: npt.NDArray,
    origin: float,
    x: npt.NDArray,
    signed: bool = True,
) -> npt.NDArray:
    """
    ``sum`` of ``per_panel`` over the panels between ``origin`` and each
    ``x`` (both must be edges), accumulated outward from ``origin`` so that
    nothing is subtracted: negative (when ``signed``) for ``x < origin``.
    """
    per_panel = np.asarray(per_panel, dtype=float)
    io = int(np.searchsorted(edges, origin))
    ix = np.searchsorted(edges, x)
    forward = np.concatenate([[0.0], np.cumsum(per_panel[io:])])
    backward = np.concatenate([[0.0], np.cumsum(per_panel[:io][::-1])])
    after = ix >= io
    out = np.where(
        after,
        forward[np.clip(ix - io, 0, forward.size - 1)],
        backward[np.clip(io - ix, 0, backward.size - 1)],
    )
    if signed:
        out = np.where(after, out, -out)
    return out


def missed_target(
    res: "dict[str, Any]",
    origin: float,
    reach: npt.NDArray,
    rtol: float,
    missing: npt.NDArray,
) -> "tuple[int, int, float, str] | None":
    """
    ``(missed, total, worst, limit)``: how many of the (non-missing) query times,
    whose values sum the panels from ``origin`` to ``reach``, have an error
    estimate above ``rtol`` of the integral of ``|h|`` from 0 (and the
    rounding floor), of how many, and the worst estimated relative error;
    ``None`` when none missed. The last entry says which refinement
    limit was reached.
    """
    edges = res["edges"]
    est = sum_between(edges, res["err"], origin, reach, signed=False)
    floor = sum_between(edges, res["noise"], origin, reach, signed=False)
    size = sum_between(
        edges, np.abs(res["value"]), 0.0, np.maximum(reach, origin), False
    )
    with np.errstate(invalid="ignore", over="ignore"):
        missed = (est > np.maximum(rtol * size, floor)) & ~missing
    if not missed.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = est[missed] / size[missed]
    worst = float(np.nanmax(rel)) if np.isfinite(rel).any() else np.inf
    return int(missed.sum()), int((~missing).sum()), worst, res["limit"]


def warn_missed_target(
    missed: int,
    total: int,
    worst: float,
    limit: str,
    rtol: float,
    stacklevel: int,
) -> None:
    """One warning for the query times whose quadrature missed ``rtol``."""
    import warnings

    stopped = (
        "{} rounds of refinement".format(_MAX_ROUNDS)
        if limit == "rounds"
        else "reaching the limit of {:,} panels".format(_PANEL_CAP)
    )
    warnings.warn(
        "The quadrature along the CovariatePath missed its {:.0e} "
        "relative accuracy target on the cumulative hazard at {} of the "
        "{} query time(s) (worst estimated relative error {:.2g}) after "
        "{}. The values are returned as computed. If the path has kinks "
        "or jumps, pass their times as CovariatePath.from_callable(..., "
        "breakpoints=[...]) or build it with CovariatePath.from_points; a "
        "path that oscillates without limit cannot be integrated to the "
        "target.".format(rtol, missed, total, worst, stopped),
        RuntimeWarning,
        stacklevel=stacklevel,
    )
