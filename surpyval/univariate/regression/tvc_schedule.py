r"""
Piecewise-constant (step) covariate schedules for time-varying-covariate
*evaluation*.

Where :mod:`...tvc_fit` reshapes time-varying-covariate data to *fit* a model,
this module describes a covariate path so an already-fitted model can be
*evaluated* along it: given a fitted proportional- or additive-hazards model
and a step schedule ``Z(t)``, ``model.sf_tvc`` returns the survival ``S(t)``
that results from the covariate following that schedule.

The covariate is constrained to be **piecewise-constant** (a "step" function):
flat on each segment, jumping at change-points. This is not a stylistic choice
-- each family's cumulative-hazard form is exact *only* when ``Z`` is constant
on each segment (the cumulative hazard is then additive over the segments). A
continuously-varying ``Z`` would silently give a wrong answer, so this class
owns the step-valued guarantee two ways:

* the **structural** constructors (:meth:`StepSchedule.from_changepoints`,
  :meth:`~StepSchedule.from_intervals`, :meth:`~StepSchedule.cyclic`) can only
  ever describe a step function -- there is no way to express a continuous
  ramp; and
* the **expression** constructor (:meth:`~StepSchedule.from_expression`)
  accepts a string in ``t`` but first proves, statically from the syntax tree,
  that ``t`` can only reach the value through a *quantizer* (``floor``,
  ``ceil``, ``//`` or a comparison) -- anything continuous is rejected before
  it is ever evaluated.

Whatever the construction, the one thing the family math consumes is
:meth:`~StepSchedule.segments`, which materialises the schedule up to a horizon
into concrete ``(start, end, Z)`` triples.
"""

import ast
import math

import numpy as np

__all__ = ["StepSchedule", "StepValuedError"]


# Functions that turn a continuous input into a piecewise-constant output. Once
# ``t`` passes through one of these, later arithmetic keeps the result stepped
# (any pure function of a step function is still a step function).
_QUANTIZERS = {"floor", "ceil", "round", "trunc"}

# Names an expression may reference besides the free variable ``t``.
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

# Callables an expression may use. Kept to quantizers plus a few pure helpers
# that cannot smuggle in continuous variation on their own.
_FUNCTIONS = {
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "trunc": math.trunc,
    "abs": abs,
    "min": min,
    "max": max,
}


class StepValuedError(ValueError):
    """
    Raised when an expression covariate schedule is not provably step-valued.

    The message points at the sub-expression through which ``t`` reaches the
    output continuously (e.g. a bare ``t`` or ``sin(t)``), so the offending
    term can be quantized (``floor(t / dt)``) or replaced.
    """


def _varies_continuously(node):
    """
    Return ``True`` if the AST ``node`` lets the free variable ``t`` influence
    its value *continuously* (rather than only through a quantizer/comparison).

    Sound -- never returns ``False`` for a genuinely continuous function -- and
    conservative: a few things that algebraically reduce to a constant (e.g.
    ``t - t``) are still reported as continuous. Decidable purely from syntax;
    no sampling.
    """
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        # ``t`` reaching here unquantized is the continuous case; named
        # constants (pi, e, ...) are fine.
        return node.id == "t"
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        # A boolean is two-valued -> stepped, regardless of its operands.
        return False
    if isinstance(node, ast.IfExp):
        # test only selects a branch (it is a comparison/boolean); the value
        # is continuous iff a selectable branch is.
        return _varies_continuously(node.body) or _varies_continuously(
            node.orelse
        )
    if isinstance(node, ast.UnaryOp):
        return _varies_continuously(node.operand)
    if isinstance(node, ast.BinOp):
        # ``t // k`` is a step; every other binary op is continuous in an
        # operand that is itself continuous in ``t``.
        if isinstance(node.op, ast.FloorDiv):
            return False
        return _varies_continuously(node.left) or _varies_continuously(
            node.right
        )
    if isinstance(node, ast.Call):
        fname = getattr(node.func, "id", None)
        if fname in _QUANTIZERS:
            return False
        # Non-quantizing call (e.g. abs, min, max, or -- once rejected -- a
        # trig function): continuous iff any argument is.
        return any(_varies_continuously(a) for a in node.args)
    # Anything unrecognised (attribute access, comprehensions, ...) is treated
    # as continuous; it is rejected here and would also fail the safe eval.
    return True


def _safe_eval(node, t):
    """
    Evaluate a validated expression AST at a single time ``t``.

    Only the node types the step-guard understands are supported, so this can
    never execute arbitrary code -- unknown syntax raises. It is a plain
    interpreter, not ``eval``.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, t)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise StepValuedError(
                "only numeric constants are allowed in a schedule expression"
            )
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "t":
            return t
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise StepValuedError(
            "unknown name {!r} in schedule expression (allowed: t, "
            "{})".format(node.id, ", ".join(sorted(_CONSTANTS)))
        )
    if isinstance(node, ast.UnaryOp):
        val = _safe_eval(node.operand, t)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return not val
        raise StepValuedError("unsupported unary operator in expression")
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, t)
        right = _safe_eval(node.right, t)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left**right
        raise StepValuedError("unsupported binary operator in expression")
    if isinstance(node, ast.BoolOp):
        vals = [_safe_eval(v, t) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left, t)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval(comparator, t)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            else:
                raise StepValuedError("unsupported comparison in expression")
            result = result and ok
            left = right
        return result
    if isinstance(node, ast.IfExp):
        return (
            _safe_eval(node.body, t)
            if _safe_eval(node.test, t)
            else _safe_eval(node.orelse, t)
        )
    if isinstance(node, ast.Call):
        fname = getattr(node.func, "id", None)
        if fname not in _FUNCTIONS:
            raise StepValuedError(
                "unknown function {!r} in schedule expression (allowed: "
                "{})".format(fname, ", ".join(sorted(_FUNCTIONS)))
            )
        args = [_safe_eval(a, t) for a in node.args]
        return float(_FUNCTIONS[fname](*args))
    raise StepValuedError(
        "unsupported syntax in schedule expression: {}".format(
            type(node).__name__
        )
    )


def _coalesce(times, values):
    """
    Collapse a fine grid of ``(time, value-row)`` samples into the minimal set
    of step segments, merging consecutive samples that share a covariate row.

    ``times`` is the strictly-increasing left edge of each sample; ``values``
    is ``(len(times), p)``. Returns ``(edges, Z)`` where ``edges`` has one more
    entry than ``Z`` has rows (the trailing edge is the horizon).
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = [0]
    for i in range(1, values.shape[0]):
        if not np.array_equal(values[i], values[i - 1]):
            keep.append(i)
    starts = times[keep]
    Z = values[keep]
    return starts, Z


class StepSchedule:
    r"""
    A piecewise-constant covariate path ``Z(t)`` for time-varying-covariate
    evaluation.

    The canonical form is a set of ``edges`` (segment boundaries) and a ``Z``
    matrix with one covariate row per segment: segment ``i`` covers
    ``[edges[i], edges[i + 1])`` and carries covariate ``Z[i]``. Beyond the
    last edge the final segment's covariate is held constant (or, when
    ``period`` is set, the pattern repeats). ``Z`` is always two-dimensional,
    so a schedule is multivariate for free (a single covariate is ``p = 1``).

    Construct one with a class method rather than the raw constructor:

    * :meth:`constant` -- a fixed covariate (reduces to the ordinary ``sf``);
    * :meth:`from_changepoints` -- ``(time, value)`` pairs, the value taking
      effect at each time;
    * :meth:`from_intervals` -- explicit ``(xl, xr]`` interval rows, matching
      the ``fit_tvc`` start-stop layout;
    * :meth:`cyclic` -- a within-period pattern repeated to a horizon;
    * :meth:`from_expression` -- a step-valued expression string in ``t``.

    All the family math needs is :meth:`segments`.
    """

    def __init__(self, edges, Z, period=None):
        edges = np.asarray(edges, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if edges.ndim != 1:
            raise ValueError("edges must be one-dimensional")
        if edges.shape[0] != Z.shape[0] + 1:
            raise ValueError(
                "edges must have exactly one more entry than Z has rows "
                "(got {} edges for {} segments)".format(
                    edges.shape[0], Z.shape[0]
                )
            )
        if Z.shape[0] == 0:
            raise ValueError("a schedule needs at least one segment")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be strictly increasing")
        if not np.isfinite(edges[:-1]).all():
            raise ValueError("every edge except the last must be finite")
        if not np.isfinite(Z).all():
            raise ValueError("Z must contain only finite values")
        if period is not None:
            period = float(period)
            if not (period > 0):
                raise ValueError("period must be positive")
            if not np.isfinite(edges[-1]):
                raise ValueError(
                    "a cyclic schedule needs a finite final edge (the pattern "
                    "length); use cyclic() to build one"
                )
            if edges[-1] - edges[0] > period + 1e-12:
                raise ValueError("the pattern spans longer than one period")
        self.edges = edges
        self.Z = Z
        self.period = period

    @property
    def p(self):
        """Number of covariates (columns of ``Z``)."""
        return self.Z.shape[1]

    # -- structural constructors ------------------------------------------

    @classmethod
    def constant(cls, Z):
        """
        A constant covariate ``Z`` over all time. Evaluating a model on it is
        identical to ``model.sf(x, Z)``.
        """
        Z = np.asarray(Z, dtype=float).reshape(1, -1)
        return cls(np.array([0.0, np.inf]), Z)

    @classmethod
    def from_changepoints(cls, times, values):
        """
        Build a schedule from ``(time, value)`` change-points.

        ``values[i]`` is the covariate row in effect from ``times[i]`` until
        the next change-point; the last value is held to the horizon.
        ``times`` must be strictly increasing; ``times[0]`` is the path's
        start (usually ``0``).

        Parameters
        ----------
        times : array_like, shape (m,)
            The instants at which the covariate takes each value.
        values : array_like, shape (m,) or (m, p)
            The covariate row active from each time.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if times.shape[0] != values.shape[0]:
            raise ValueError(
                "times and values must have the same length ({} vs {})".format(
                    times.shape[0], values.shape[0]
                )
            )
        edges = np.concatenate([times, [np.inf]])
        return cls(edges, values)

    @classmethod
    def from_intervals(cls, xl, xr, Z):
        """
        Build a schedule from explicit ``(xl, xr]`` interval rows -- the same
        layout ``fit_tvc`` consumes.

        The intervals must be contiguous (each ``xr`` equals the next ``xl``);
        the covariate beyond the final ``xr`` is held constant. Rows may be
        given in any order and are sorted by ``xl``.
        """
        xl = np.atleast_1d(np.asarray(xl, dtype=float))
        xr = np.atleast_1d(np.asarray(xr, dtype=float))
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if not (xl.shape[0] == xr.shape[0] == Z.shape[0]):
            raise ValueError("xl, xr and Z must have the same number of rows")
        order = np.argsort(xl)
        xl, xr, Z = xl[order], xr[order], Z[order]
        if np.any(xl >= xr):
            raise ValueError("every interval must have xl < xr")
        if np.any(np.abs(xl[1:] - xr[:-1]) > 1e-9):
            raise ValueError(
                "intervals must be contiguous (each xr must equal the next "
                "xl); a step schedule cannot have gaps or overlaps"
            )
        edges = np.concatenate([xl, [xr[-1]]])
        return cls(edges, Z)

    @classmethod
    def cyclic(cls, pattern_times, pattern_values, period):
        """
        A within-period covariate pattern repeated indefinitely.

        ``pattern_times`` are the change-points *within* one period (starting
        at ``0`` and all ``< period``); ``pattern_values`` the covariate on
        each. The pattern repeats every ``period`` and is materialised up to
        whatever horizon :meth:`segments` is asked for -- for duty cycles.

        Parameters
        ----------
        pattern_times : array_like, shape (m,)
            Change-points within one period, ``0 <= t < period``, strictly
            increasing (the first should be ``0``).
        pattern_values : array_like, shape (m,) or (m, p)
            The covariate row active from each within-period change-point.
        period : float
            The repetition period.
        """
        pattern_times = np.atleast_1d(np.asarray(pattern_times, dtype=float))
        pattern_values = np.asarray(pattern_values, dtype=float)
        if pattern_values.ndim == 1:
            pattern_values = pattern_values.reshape(-1, 1)
        period = float(period)
        if pattern_times.shape[0] != pattern_values.shape[0]:
            raise ValueError(
                "pattern_times and pattern_values must have the same length"
            )
        if pattern_times[0] != 0.0:
            raise ValueError("the first pattern time must be 0")
        if np.any(pattern_times < 0) or np.any(pattern_times >= period):
            raise ValueError("pattern times must satisfy 0 <= t < period")
        edges = np.concatenate([pattern_times, [period]])
        return cls(edges, pattern_values, period=period)

    # -- expression constructor -------------------------------------------

    @classmethod
    def from_expression(cls, expr, horizon, resolution=1.0, t0=0.0):
        r"""
        Build a schedule from a step-valued expression string (or list of
        strings) in the free variable ``t``.

        The expression is parsed and *proved* piecewise-constant before it is
        ever evaluated: ``t`` may reach the value only through a quantizer
        (``floor``, ``ceil``, ``//``) or a comparison. A continuously-varying
        expression (``0.3 + 1e-4 * t``, ``sin(t)``) raises
        :class:`StepValuedError`.

        The (guaranteed stepped) expression is then materialised by sampling on
        a grid of spacing ``resolution`` over ``[t0, horizon]`` and coalescing
        runs of equal value into segments, so ``resolution`` must be no
        coarser than the narrowest step (e.g. ``1`` hour for an
        ``8``-on/``16``-off duty cycle).

        Parameters
        ----------
        expr : str or list of str
            A scalar covariate (``p = 1``) or, for a multivariate covariate,
            one expression per covariate.
        horizon : float
            Materialise the schedule out to this time.
        resolution : float, optional
            Grid spacing for sampling. Default ``1.0``.
        t0 : float, optional
            Start of the path. Default ``0.0``.

        Examples
        --------
        >>> StepSchedule.from_expression("0.9 if t % 24 < 8 else 0.3", 96)
        >>> StepSchedule.from_expression("0.3 * 2 ** floor(t / 1000)", 5000)
        """
        exprs = [expr] if isinstance(expr, str) else list(expr)
        if len(exprs) == 0:
            raise ValueError("at least one expression is required")
        horizon = float(horizon)
        resolution = float(resolution)
        if not (resolution > 0):
            raise ValueError("resolution must be positive")
        if not (horizon > t0):
            raise ValueError("horizon must be greater than t0")

        trees = []
        for e in exprs:
            try:
                tree = ast.parse(e, mode="eval")
            except SyntaxError as exc:
                raise StepValuedError(
                    "could not parse schedule expression {!r}: {}".format(
                        e, exc
                    )
                )
            if _varies_continuously(tree.body):
                raise StepValuedError(
                    "expression {!r} is not step-valued: t reaches the "
                    "value continuously. Quantize it (e.g. floor(t / dt)) "
                    "or use a comparison so the covariate stays "
                    "piecewise-constant.".format(e)
                )
            trees.append(tree)

        n = int(math.floor((horizon - t0) / resolution)) + 1
        grid = t0 + resolution * np.arange(n)
        # Ensure the horizon itself is represented as a left edge so the final
        # step is not truncated early.
        if grid[-1] < horizon:
            grid = np.append(grid, horizon)
        values = np.empty((grid.shape[0], len(trees)), dtype=float)
        for j, tree in enumerate(trees):
            for i, tv in enumerate(grid):
                values[i, j] = float(_safe_eval(tree, float(tv)))

        starts, Z = _coalesce(grid, values)
        edges = np.concatenate([starts, [np.inf]])
        return cls(edges, Z)

    # -- consumption ------------------------------------------------------

    def segments(self, t_max):
        """
        Materialise the schedule up to ``t_max`` into concrete step segments.

        Returns ``(starts, ends, Z)`` with one row per segment, contiguous and
        covering ``[edges[0], t_max]``. The final segment (or, for a cyclic
        schedule, every repetition) is clipped so ``ends[-1] == t_max``. This
        is the sole interface the family cumulative-hazard math consumes.

        Parameters
        ----------
        t_max : float
            The horizon to materialise to (typically the largest query time).
        """
        t_max = float(t_max)
        if t_max <= self.edges[0]:
            raise ValueError(
                "t_max ({}) must be greater than the schedule start "
                "({})".format(t_max, self.edges[0])
            )

        if self.period is None:
            edges = self.edges.copy()
            # Held-constant tail: replace an infinite (or short) final edge
            # with the horizon.
            edges[-1] = t_max if not np.isfinite(edges[-1]) else edges[-1]
            Z = self.Z
        else:
            # Repeat the within-period pattern until it covers t_max.
            pattern_starts = self.edges[:-1]
            base = self.edges[0]
            reps = int(math.ceil((t_max - base) / self.period))
            starts = []
            Zrows = []
            for k in range(reps):
                offset = k * self.period
                for s, z in zip(pattern_starts, self.Z):
                    starts.append(s + offset)
                    Zrows.append(z)
            starts = np.asarray(starts, dtype=float)
            edges = np.concatenate([starts, [starts[0] + reps * self.period]])
            Z = np.asarray(Zrows, dtype=float)

        # Clip to [start, t_max]: drop segments that begin at or after t_max
        # and trim the final edge.
        within = edges[:-1] < t_max
        starts = edges[:-1][within]
        Z = Z[within]
        ends = np.concatenate([starts[1:], [t_max]])
        ends = np.minimum(ends, t_max)
        return starts, ends, Z

    def __repr__(self):
        kind = "cyclic" if self.period is not None else "step"
        return "StepSchedule({}, {} segment(s), p={})".format(
            kind, self.Z.shape[0], self.p
        )


# -- shared helpers for the model ``sf_tvc`` / ``Hf_tvc`` methods ----------
#
# These give every regression family (parametric PH/AH and the semi-parametric
# Cox) the same time-varying-covariate calling convention: pass either a
# ready-made StepSchedule, or ``(xl, Z)`` arrays (segment start times plus one
# covariate row per segment).


def as_step_schedule(Z, xl=None):
    """
    Coerce a model ``sf_tvc`` covariate argument into a :class:`StepSchedule`.

    ``Z`` is either a ready-made schedule (then ``xl`` must be ``None``), or an
    array of per-segment covariate rows whose segment start times are ``xl``.
    """
    if isinstance(Z, StepSchedule):
        if xl is not None:
            raise ValueError(
                "xl must not be given when Z is already a StepSchedule"
            )
        return Z
    if xl is None:
        raise ValueError(
            "for the array form pass xl (the segment start times) alongside "
            "Z (one covariate row per segment); or pass a StepSchedule"
        )
    return StepSchedule.from_changepoints(xl, Z)


def segments_from_origin(schedule, t_max):
    """
    Materialise ``schedule`` to ``t_max``, holding the first segment back to
    the time origin so cumulative hazard is measured from ``0`` (unconditional
    survival). Returns ``(starts, ends, Z)``.
    """
    starts, ends, Z = schedule.segments(t_max)
    if starts[0] > 0:
        starts = starts.copy()
        starts[0] = 0.0
    return starts, ends, Z
