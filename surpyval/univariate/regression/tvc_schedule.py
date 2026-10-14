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

A covariate that does vary continuously is described by a
:class:`~surpyval.univariate.regression.tvc_path.CovariatePath` instead, along
which ``sf_tvc`` integrates the hazard (#172); the type of the path chooses the
method.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

__all__ = ["StepSchedule", "StepValuedError"]


# Functions that turn a continuous input into a piecewise-constant output. Once
# ``t`` passes through one of these, later arithmetic keeps the result stepped
# (any pure function of a step function is still a step function).
_QUANTIZERS: "set[str]" = {"floor", "ceil", "round", "trunc"}

# Names an expression may reference besides the free variable ``t``.
_CONSTANTS: "dict[str, float]" = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Callables an expression may use. Kept to quantizers plus a few pure helpers
# that cannot smuggle in continuous variation on their own.
_FUNCTIONS: "dict[str, Callable]" = {
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

    Examples
    --------
    >>> from surpyval import StepSchedule, StepValuedError
    >>> try:
    ...     StepSchedule.from_expression("2 * t", 10)
    ... except StepValuedError as error:
    ...     print(error)
    expression '2 * t' is not step-valued: t reaches the value
    continuously. Quantize it (e.g. floor(t / dt)) or use a comparison so
    the covariate stays piecewise-constant.

    Quantized, the same growth is accepted:

    >>> StepSchedule.from_expression("2 * floor(t / 5)", 10)
    StepSchedule.from_changepoints([0.0, 5.0, 10.0], [[0.0], [2.0], [4.0]])
    """


def _varies_continuously(node: ast.AST) -> bool:
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
    if isinstance(node, ast.Compare):
        # A boolean is two-valued -> stepped, regardless of its operands.
        return False
    if isinstance(node, ast.BoolOp):
        # ``and`` / ``or`` return an operand, as in Python. ``and`` returns
        # an operand before the last only when it is falsy (0), so only its
        # last operand can carry ``t`` through; ``or`` returns any truthy
        # operand as it is.
        if isinstance(node.op, ast.And):
            return _varies_continuously(node.values[-1])
        return any(_varies_continuously(v) for v in node.values)
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
        return any(_varies_continuously(a) for a in node.args) or any(
            _varies_continuously(k.value) for k in node.keywords
        )
    # Anything unrecognised (attribute access, comprehensions, ...) is treated
    # as continuous; it is rejected here and would also fail the safe eval.
    return True


def _whole(value: "float | bool") -> "float | int":
    """``value`` as an ``int`` when it is a whole number, else unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


_UNARY_OPS: "dict[type, Callable[[Any], Any]]" = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_BINARY_OPS: "dict[type, Callable[[Any, Any], Any]]" = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_COMPARISONS: "dict[type, Callable[[Any, Any], Any]]" = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _eval_constant(node: ast.Constant) -> float:
    """A numeric literal, as a float (booleans and strings are refused)."""
    if isinstance(node.value, bool) or not isinstance(
        node.value, (int, float)
    ):
        raise StepValuedError(
            "only numeric constants are allowed in a schedule expression"
        )
    return float(node.value)


def _eval_name(node: ast.Name, t: float) -> float:
    """The free variable ``t`` or one of the named constants."""
    if node.id == "t":
        return t
    if node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise StepValuedError(
        "unknown name {!r} in schedule expression (allowed: t, "
        "{})".format(node.id, ", ".join(sorted(_CONSTANTS)))
    )


def _eval_unary(node: ast.UnaryOp, t: float) -> "float | bool":
    val = _safe_eval(node.operand, t)
    op = _UNARY_OPS.get(type(node.op))
    if op is None:
        raise StepValuedError("unsupported unary operator in expression")
    return op(val)


def _eval_binary(node: ast.BinOp, t: float) -> "float | bool":
    left = _safe_eval(node.left, t)
    right = _safe_eval(node.right, t)
    op = _BINARY_OPS.get(type(node.op))
    if op is None:
        raise StepValuedError("unsupported binary operator in expression")
    return op(left, right)


def _eval_bool_op(node: ast.BoolOp, t: float) -> "float | bool":
    """``and`` / ``or``, short-circuiting as Python does.

    ``and`` returns the first falsy operand, ``or`` the first truthy one,
    else either returns the last; the operands after the one returned are
    not evaluated.
    """
    stop_on = isinstance(node.op, ast.Or)
    for operand in node.values[:-1]:
        val = _safe_eval(operand, t)
        if bool(val) == stop_on:
            return val
    return _safe_eval(node.values[-1], t)


def _eval_compare(node: ast.Compare, t: float) -> "float | bool":
    """A (possibly chained) comparison; every comparator is evaluated."""
    left = _safe_eval(node.left, t)
    result = True
    for cmp_op, comparator in zip(node.ops, node.comparators):
        right = _safe_eval(comparator, t)
        compare = _COMPARISONS.get(type(cmp_op))
        if compare is None:
            raise StepValuedError("unsupported comparison in expression")
        ok = compare(left, right)
        result = result and ok
        left = right
    return result


def _call_arguments(
    node: ast.Call, fname: str, t: float
) -> "tuple[list, dict]":
    """The evaluated positional and keyword arguments of a call."""
    args = [_safe_eval(a, t) for a in node.args]
    kwargs = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(
                "'**' arguments are not supported in a schedule "
                "expression (in the call to {})".format(fname)
            )
        kwargs[keyword.arg] = _safe_eval(keyword.value, t)
    if fname == "round":
        # Every constant is read as a float, but round's ndigits must
        # be an int: pass a whole number as one.
        if len(args) > 1:
            args[1] = _whole(args[1])
        if "ndigits" in kwargs:
            kwargs["ndigits"] = _whole(kwargs["ndigits"])
    return args, kwargs


def _eval_call(node: ast.Call, t: float) -> float:
    """A call to one of the allowed functions."""
    fname = getattr(node.func, "id", None)
    if fname not in _FUNCTIONS:
        raise StepValuedError(
            "unknown function {!r} in schedule expression (allowed: "
            "{})".format(fname, ", ".join(sorted(_FUNCTIONS)))
        )
    args, kwargs = _call_arguments(node, fname, t)
    try:
        return float(_FUNCTIONS[fname](*args, **kwargs))
    except TypeError as exc:
        raise ValueError(
            "cannot evaluate {}() in the schedule expression with "
            "{} positional argument(s){}: {}".format(
                fname,
                len(args),
                (
                    " and keyword(s) " + ", ".join(sorted(kwargs))
                    if kwargs
                    else ""
                ),
                exc,
            )
        ) from exc


def _safe_eval(node: ast.AST, t: float) -> "float | bool":
    """
    Evaluate a validated expression AST at a single time ``t``.

    Only the node types the step-guard understands are supported, so this can
    never execute arbitrary code -- unknown syntax raises. It is a plain
    interpreter, not ``eval``.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, t)
    if isinstance(node, ast.Constant):
        return _eval_constant(node)
    if isinstance(node, ast.Name):
        return _eval_name(node, t)
    if isinstance(node, ast.UnaryOp):
        return _eval_unary(node, t)
    if isinstance(node, ast.BinOp):
        return _eval_binary(node, t)
    if isinstance(node, ast.BoolOp):
        return _eval_bool_op(node, t)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, t)
    if isinstance(node, ast.IfExp):
        return (
            _safe_eval(node.body, t)
            if _safe_eval(node.test, t)
            else _safe_eval(node.orelse, t)
        )
    if isinstance(node, ast.Call):
        return _eval_call(node, t)
    raise StepValuedError(
        "unsupported syntax in schedule expression: {}".format(
            type(node).__name__
        )
    )


def _coalesce(
    times: npt.NDArray, values: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
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

    Examples
    --------
    A covariate that switches from 0 to 1 at time 50, used to predict
    survival from a fitted proportional hazards model:

    >>> import numpy as np
    >>> from surpyval import StepSchedule, WeibullPH
    >>> schedule = StepSchedule.from_changepoints([0, 50], [0.0, 1.0])
    >>> schedule.edges
    array([ 0., 50., inf])
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.binomial(1, 0.5, (100, 1)).astype(float)
    >>> x = 100 * rng.weibull(2, 100) * np.exp(-0.25 * Z[:, 0])
    >>> model = WeibullPH.fit(x, Z)
    >>> model.sf_tvc([40, 80, 120], schedule).round(4)
    array([0.9015, 0.4598, 0.1128])

    Before time 50 that is the survival at ``Z = 0``; after it, the
    hazard of ``Z = 1`` applies:

    >>> model.sf(np.array([40, 80, 120]), [0.0]).round(4)
    array([0.9015, 0.6008, 0.2746])
    """

    def __init__(
        self,
        edges: npt.ArrayLike,
        Z: npt.ArrayLike,
        period: "float | None" = None,
    ) -> None:
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
        if not np.isfinite(edges[1:-1]).all() or np.isnan(edges[0]):
            raise ValueError(
                "every edge except the first and last must be finite (the "
                "first may be -inf: a value held since -inf)"
            )
        if not np.isfinite(Z).all():
            raise ValueError("Z must contain only finite values")
        if period is not None:
            period = float(period)
            if not (period > 0):
                raise ValueError("period must be positive")
            if not np.isfinite(edges[[0, -1]]).all():
                raise ValueError(
                    "a cyclic schedule needs finite first and last edges "
                    "(the pattern); use cyclic() to build one"
                )
            if edges[-1] - edges[0] > period + 1e-12:
                raise ValueError("the pattern spans longer than one period")
        self.edges = edges
        self.Z = Z
        self.period = period

    @property
    def p(self) -> int:
        """Number of covariates (columns of ``Z``)."""
        return self.Z.shape[1]

    @property
    def start(self) -> float:
        """The time the schedule starts, its first edge (``-inf`` for one
        held since ``-inf``): it gives no value before it."""
        return float(self.edges[0])

    # -- structural constructors ------------------------------------------

    @classmethod
    def constant(cls, Z: npt.ArrayLike) -> "StepSchedule":
        """
        A constant covariate ``Z`` over all time, from ``-inf`` (#952).
        Evaluating a model on it is identical to ``model.sf(x, Z)``.
        """
        Z = np.asarray(Z, dtype=float).reshape(1, -1)
        return cls(np.array([-np.inf, np.inf]), Z)

    @classmethod
    def from_changepoints(
        cls, times: npt.ArrayLike, values: npt.ArrayLike
    ) -> "StepSchedule":
        """
        Build a schedule from ``(time, value)`` change-points.

        ``values[i]`` is the covariate row in effect from ``times[i]`` until
        the next change-point; the last value is held to the horizon.
        ``times`` must be strictly increasing; ``times[0]`` is the path's
        start, :attr:`start`, and may be ``-inf`` (the first value held
        since ``-inf``). A path must cover every time a model's answer
        depends on, and one starting later is refused, as nothing says
        what the covariate was before it: from ``0`` for a baseline that
        starts at ``0`` (the part before ``0`` is ignored); from ``-inf``
        for a proportional hazards or proportional odds model on a
        baseline over the whole real line (``Normal``, ``Gumbel``,
        ``Logistic``), and from ``0``, and any earlier time asked about,
        for accelerated failure time and additive hazards, which measure
        the covariate's effect from ``0`` (#952). On a record it must start
        at or before the start of the item's observation.

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
    def from_intervals(
        cls, xl: npt.ArrayLike, xr: npt.ArrayLike, Z: npt.ArrayLike
    ) -> "StepSchedule":
        """
        Build a schedule from explicit ``(xl, xr]`` interval rows -- the same
        layout ``fit_tvc`` consumes.

        The intervals must be contiguous (each ``xr`` equals the next ``xl``);
        the covariate beyond the final ``xr`` is held constant. Rows may be
        given in any order and are sorted by ``xl``.
        """
        # Fresh names after coercion: assigning back to the ArrayLike
        # parameters keeps their declared union under some mypy releases,
        # which then rejects the arithmetic below.
        xl_a = np.atleast_1d(np.asarray(xl, dtype=float))
        xr_a = np.atleast_1d(np.asarray(xr, dtype=float))
        Z_a = np.asarray(Z, dtype=float)
        if Z_a.ndim == 1:
            Z_a = Z_a.reshape(-1, 1)
        if not (xl_a.shape[0] == xr_a.shape[0] == Z_a.shape[0]):
            raise ValueError("xl, xr and Z must have the same number of rows")
        order = np.argsort(xl_a)
        xl_a, xr_a, Z_a = xl_a[order], xr_a[order], Z_a[order]
        if np.any(xl_a >= xr_a):
            raise ValueError("every interval must have xl < xr")
        if np.any(np.abs(xl_a[1:] - xr_a[:-1]) > 1e-9):
            raise ValueError(
                "intervals must be contiguous (each xr must equal the next "
                "xl); a step schedule cannot have gaps or overlaps"
            )
        edges = np.concatenate([xl_a, [xr_a[-1]]])
        return cls(edges, Z_a)

    @classmethod
    def cyclic(
        cls,
        pattern_times: npt.ArrayLike,
        pattern_values: npt.ArrayLike,
        period: float,
    ) -> "StepSchedule":
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
    def from_expression(
        cls,
        expr: "str | list[str]",
        horizon: float,
        resolution: float = 1.0,
        t0: float = 0.0,
    ) -> "StepSchedule":
        r"""
        Build a schedule from a step-valued expression string (or list of
        strings) in the free variable ``t``.

        The expression is parsed and *proved* piecewise-constant before it is
        ever evaluated: ``t`` may reach the value only through a quantizer
        (``floor``, ``ceil``, ``//``) or a comparison. A continuously-varying
        expression (``0.3 + 1e-4 * t``, ``sin(t)``) raises
        :class:`StepValuedError`. The allowed syntax means what it does in
        Python: ``and`` / ``or`` return an operand, and keyword arguments
        (``round(t / 10, ndigits=1)``) are passed on; one a function cannot
        take raises ``ValueError``.

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
        >>> duty = StepSchedule.from_expression(
        ...     "0.9 if t % 24 < 8 else 0.3", 96
        ... )
        >>> duty.edges[:4], duty.Z[:4].ravel()
        (array([ 0.,  8., 24., 32.]), array([0.9, 0.3, 0.9, 0.3]))
        >>> StepSchedule.from_expression("0.3 * 2 ** floor(t / 1000)", 1000)
        StepSchedule.from_changepoints([0.0, 1000.0], [[0.3], [0.6]])
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

    def segments(
        self, t_max: float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
            start_list = []
            Zrows = []
            for k in range(reps):
                offset = k * self.period
                for s, z in zip(pattern_starts, self.Z):
                    start_list.append(s + offset)
                    Zrows.append(z)
            starts = np.asarray(start_list, dtype=float)
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

    def __eq__(self, other: object) -> bool:
        """Two schedules are equal when they have the same edges, values
        and period (#930)."""
        if not isinstance(other, StepSchedule):
            return NotImplemented
        return (
            self.period == other.period
            and np.array_equal(self.edges, other.edges)
            and np.array_equal(self.Z, other.Z)
        )

    # Equal by contents, which can change: not hashable (a record hashes
    # its own, fixed, copy by contents).
    __hash__ = None  # type: ignore[assignment]

    #: A schedule of more segments than this shows in short.
    _REPR_SEGMENTS = 20

    def __repr__(self) -> str:
        """The call that builds the schedule (#930); a long schedule in
        short."""
        edges, Z = [_number(e) for e in self.edges], self.Z.tolist()
        n = self.Z.shape[0]
        if n > self._REPR_SEGMENTS:
            kind = "cyclic" if self.period is not None else "step"
            return "StepSchedule({}, {} segments, p={})".format(
                kind, n, self.p
            )
        if self.period is not None:
            if edges[0] == 0 and edges[-1] == self.period:
                return "StepSchedule.cyclic({}, {}, period={!r})".format(
                    edges[:-1], Z, self.period
                )
            return "StepSchedule({}, {}, period={!r})".format(
                edges, Z, self.period
            )
        if np.isfinite(self.edges[-1]):
            return "StepSchedule.from_intervals({}, {}, {})".format(
                _listed(edges[:-1]), _listed(edges[1:]), Z
            )
        if n == 1 and np.isneginf(self.edges[0]):
            return "StepSchedule.constant({})".format(Z[0])
        return "StepSchedule.from_changepoints({}, {})".format(
            _listed(edges[:-1]), Z
        )


class _Inf(float):
    """``-inf`` that shows as code that makes it."""

    def __repr__(self) -> str:
        return "-np.inf"


def _number(edge: float) -> float:
    # -inf in a repr as -np.inf, which pastes back.
    return _Inf(edge) if np.isneginf(edge) else float(edge)


def _listed(values: list) -> str:
    return "[" + ", ".join(repr(v) for v in values) + "]"


# -- shared helpers for the model ``sf_tvc`` / ``Hf_tvc`` methods ----------
#
# These give every regression family (parametric PH/AH and the semi-parametric
# Cox) the same time-varying-covariate calling convention: pass either a
# ready-made StepSchedule, or ``(xl, Z)`` arrays (segment start times plus one
# covariate row per segment).


def as_step_schedule(
    Z: "StepSchedule | npt.ArrayLike", xl: "npt.ArrayLike | None" = None
) -> StepSchedule:
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


def as_covariate_path(
    Z: Any,
    xl: "npt.ArrayLike | None" = None,
    names: "list | None" = None,
) -> Any:
    """
    Coerce a model ``sf_tvc`` covariate argument into a :class:`StepSchedule`
    or a :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`.

    A ``CovariatePath`` is passed through (then ``xl`` must be ``None``);
    an item record (:class:`~surpyval.utils.item_lifetime.ItemLifetime` or
    :class:`~surpyval.utils.item_history.ItemHistory`) gives its own
    covariates, the fixed ones followed by the time-varying ones, as a
    model fitted on such items takes them (named covariates in the order
    of the model's ``names``, its ``feature_names``, where both have
    names); anything else is as for
    :func:`as_step_schedule`. The type decides the method: a schedule is
    summed exactly, a path is integrated.
    """
    from .tvc_path import CovariatePath

    if _is_item(Z):
        if xl is not None:
            raise ValueError(
                "xl must not be given when Z is an item: the item gives its "
                "own covariates over time"
            )
        return check_from_origin(item_covariate_path(Z, names), Z.id)
    if isinstance(Z, CovariatePath):
        if xl is not None:
            raise ValueError(
                "xl must not be given when Z is a CovariatePath (it gives "
                "the segment start times of the array form only)"
            )
        return check_from_origin(Z)
    return check_from_origin(as_step_schedule(Z, xl))


def check_from_origin(path: Any, item: Any = None) -> Any:
    """
    ``path`` (a :class:`StepSchedule` or a
    :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`),
    refused if it starts after the time origin, ``0``: a prediction is
    measured from ``0``, and nothing says what the covariate was before
    the path starts (#930; it used to be taken as the path's first value).
    ``item`` is the id of the item it is the covariates of.
    """
    start = path.start
    if start <= 0:
        return path
    who = (
        "The covariates"
        if item is None
        else f"Item {item!r}: its time-varying covariates"
    )
    example = (
        f"StepSchedule.from_changepoints([0, {start:g}, ...], ...)"
        if isinstance(path, StepSchedule)
        else f"CovariatePath.from_points([0, {start:g}, ...], ...)"
    )
    raise ValueError(
        f"{who} start at {start:g}, after the time origin (0): a "
        "prediction is measured from 0, so it needs the covariates from 0 "
        f"on. Give their values from 0, e.g. {example}."
    )


def segments_from_origin(
    schedule: StepSchedule, t_max: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Materialise ``schedule`` on ``[0, t_max]``, the time origin being where
    cumulative hazard (or accelerated age) is measured from, so the result
    is the unconditional survival. Returns ``(starts, ends, Z)`` with
    ``starts[0] == 0``.

    The schedule must start at or before ``0`` (:func:`check_from_origin`);
    the part of it before ``0`` is dropped, the segment in force at ``0``
    then starting there. For ``t_max <= 0`` the result is that one
    segment, ``[0, 0]``.
    """
    check_from_origin(schedule)
    return segments_from(schedule, 0.0, t_max)


def segments_from(
    schedule: StepSchedule, lo: float, t_max: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Materialise ``schedule`` on ``[lo, t_max]`` (``lo`` may be ``-inf``),
    the schedule starting at or before ``lo``: the part of it before
    ``lo`` is dropped, the segment in force at ``lo`` then starting there.
    For ``t_max <= lo`` the result is that one segment, ``[lo, lo]``.
    """
    t_max = float(t_max)
    end = max(t_max, lo)
    # Any horizon past lo shows the value in force at lo.
    starts, ends, Z = schedule.segments(end if end > lo else lo + 1.0)
    # Drop what ends at or before lo and start the rest at lo.
    keep = ends > lo
    starts, ends, Z = starts[keep].copy(), ends[keep], Z[keep]
    starts[0] = lo
    if end == lo:
        return starts[:1], np.array([lo]), Z[:1]
    return starts, ends, Z


def _in_model_order(item: Any, names: "list | None") -> Any:
    """``item``'s fixed covariates, in the order of the model's ``names``
    where both are named (the fixed columns come first in a model's
    names)."""
    have = getattr(item, "covariate_names", None)
    if item.covariates is None or have is None or names is None:
        return item.covariates
    wanted = list(names)[: len(have)]
    if set(wanted) != set(have):
        raise ValueError(
            f"Item {item.id!r} has covariates {list(have)!r}; the model was "
            f"fitted on {wanted!r}."
        )
    from surpyval.utils.item_history import covariate_row_in

    return covariate_row_in(item, wanted)


def _is_item(Z: Any) -> bool:
    from surpyval.utils.item_history import ItemHistory
    from surpyval.utils.item_lifetime import ItemLifetime

    return isinstance(Z, (ItemLifetime, ItemHistory))


def item_covariate_path(item: Any, names: "list | None" = None) -> Any:
    """
    An item record's covariates over time, in the column order of a model
    fitted on such items (``fit(SurpyvalData.from_items(...))``): its fixed
    covariates, then its time-varying ones. A constant
    :class:`StepSchedule` for fixed covariates alone; the item's schedule
    with the fixed columns added for a ``StepSchedule``; and for a
    ``CovariatePath``, a path with the fixed columns in front. Where the
    item's covariates are named and ``names`` (a model's
    ``feature_names``) are given, its fixed covariates are put in the
    order of ``names``, whatever order the item has them in.

    Examples
    --------
    >>> from surpyval import ItemLifetime, StepSchedule
    >>> from surpyval.univariate.regression.tvc_schedule import (
    ...     item_covariate_path,
    ... )
    >>> stress = StepSchedule.from_changepoints([0, 400], [[0.0], [1.0]])
    >>> item = ItemLifetime(
    ...     "P7", right_censored=900, covariates=[2.0],
    ...     time_varying_covariates=stress,
    ... )
    >>> item_covariate_path(item).Z
    array([[2., 0.],
           [2., 1.]])
    """
    from .tvc_path import CovariatePath

    fixed = _in_model_order(item, names)
    varying = item.time_varying_covariates
    if fixed is None and varying is None:
        raise ValueError(
            f"Item {item.id!r} has no covariates to predict with."
        )
    if varying is None:
        return StepSchedule.constant(fixed)
    if fixed is None:
        return varying
    fixed = np.asarray(fixed, dtype=float)
    if isinstance(varying, StepSchedule):
        return StepSchedule(
            varying.edges,
            np.hstack([np.tile(fixed, (len(varying.Z), 1)), varying.Z]),
            varying.period,
        )
    if isinstance(varying, CovariatePath):

        def combined(t: npt.NDArray) -> npt.NDArray:
            values = varying(np.atleast_1d(t))
            return np.hstack([np.tile(fixed, (len(values), 1)), values])

        knots = getattr(varying, "_knots", None)
        path = CovariatePath.from_callable(
            combined,
            p=len(fixed) + varying.p,
            breakpoints=None if knots is None or len(knots) == 0 else knots,
            period=varying.period,
        )
        # It starts where the item's path does.
        path._start = varying.start
        return path
    raise TypeError(
        f"Item {item.id!r}'s time-varying covariates are a "
        f"{type(varying).__name__}, not a StepSchedule or a CovariatePath."
    )
