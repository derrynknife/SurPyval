"""SurPyval's data formats: their handlers and the conversions between them.

The ``xcnt`` (times, censoring flags, counts, truncation), ``xrd`` (times,
numbers at risk, deaths) and ``fsli`` (failures, suspensions, left- and
interval-censored values) formats are defined in the Conventions page of
the documentation. The handlers validate data in one format; the
converters turn one format into another. Also the competing-risks event
helpers (``is_missing_event``, ``missing_events`` and
``resolve_cr_censoring``).
"""

from __future__ import annotations

import datetime as _dt
import warnings
from numbers import Number
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import caller_stacklevel


def validate_float_array(
    arr: "npt.ArrayLike | None", name: str
) -> npt.NDArray:
    """Convert input to float array with better error handling."""
    if arr is None:
        return np.array([], dtype=np.float64)
    try:
        return np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(
            f"'{name}' must be convertible to an array of float values"
        )


def group_xcnt(
    x: npt.NDArray, c: npt.NDArray, n: npt.NDArray, t: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Collapse identical ``(x, c, t)`` rows, summing their counts.

    Takes and returns the four ``xcnt`` arrays (``t`` of shape
    ``(k, 2)``); each group is represented by its first row, in order of
    first appearance. Rows containing NaN are never merged.

    It sorts and counts with ``np.bincount``: a walk over every
    observation in Python (a triple-nested ``defaultdict``) is O(N) but
    with a very large constant -- roughly 13 microseconds per observation,
    94% of a 50,000-point Normal fit and two seconds at 100,000 points.

    The group *order* is x-major, which is subtler than it looks: outer
    by first appearance of ``x``, then of ``(x, c)`` within it, then of
    the full key. ``xcnt_sort`` runs immediately after this and re-sorts
    on ``c``, ``t.min(axis=1)`` and ``x`` -- but it is a *stable* sort, so
    rows tying on all three of those keep whatever order arrived. Rows
    sharing an ``x`` and ``c`` with different ``tr`` but an equal
    ``t.min()`` are exactly such a tie, so a plain sorted ``np.unique``
    would silently reorder them; the lexsort below keeps the nesting.

    When nothing needs grouping the inputs are returned as they are,
    rather than copied. Callers inside the package sort immediately
    afterwards, which copies, so nothing aliases in practice.
    """
    # Continuous data has nothing to group: every row is already its own
    # group, and since the ordering is x-major and each x occurs once,
    # that order *is* the input order. So the answer is the input,
    # untouched. Establishing this costs one sort of a single column,
    # against a sort of the full key -- and it is the common case,
    # since only tied (rounded, discrete, or heavily weighted) data
    # groups at all. Distinct values in the first column are enough:
    # they make whole rows distinct whatever c and t hold. Empty input
    # trivially satisfies this and is likewise handed straight back.
    leading = x if x.ndim == 1 else x[:, 0]
    if np.unique(leading).size == leading.size:
        return x, c, n, t

    x_columns = x.reshape(-1, 1) if x.ndim == 1 else x
    width = x_columns.shape[1]
    # The key's columns, x first, then c, then the truncation bounds.
    columns = [x_columns[:, j] for j in range(width)] + [c, t[:, 0], t[:, 1]]
    # One sort on the full key. The x and (x, c) groups are prefixes of
    # it, so they are runs of the same order: three lexsorts (seven sort
    # passes) would be 36% of a tied Weibull fit at 1e5 (#515). A column
    # holding one value throughout (no truncation, all observed) cannot
    # change the order and is left out; one with a NaN is never constant,
    # since NaN != NaN. The sort need not be stable, as the first row of
    # each group is found as the minimum over the group, so with a single
    # varying column numpy's (much faster) default argsort does.
    varying = [j for j, col in enumerate(columns) if (col != col[0]).any()]
    if len(varying) == 1:
        order = np.argsort(columns[varying[0]])
    elif varying:
        order = np.lexsort([columns[j] for j in reversed(varying)])
    else:
        order = np.arange(len(x))

    # Where each level's run starts in sorted order: a change in x starts
    # an x run, a change in c as well an (x, c) run, and a change in t as
    # well a full-key run. Rows containing NaN never compare equal, so
    # they stay in runs of their own, as a dictionary keyed on the rows
    # would keep them.
    changed = [np.zeros(len(x) - 1, dtype=bool) for _ in range(3)]
    for j in varying:
        level = 0 if j < width else 1 if j == width else 2
        ordered = columns[j][order]
        changed[level] |= ordered[1:] != ordered[:-1]
    new_x = np.concatenate([[True], changed[0]])
    new_xc = new_x | np.concatenate([[False], changed[1]])
    new_full = new_xc | np.concatenate([[False], changed[2]])

    def first_rows(starts: npt.NDArray) -> npt.NDArray:
        # The earliest original row of each run, per full-key run.
        first = np.minimum.reduceat(order, np.flatnonzero(starts))
        return first[np.cumsum(starts)[new_full] - 1]

    # For each full group (in sorted order): its earliest row, and those
    # of its (x, c) and x groups.
    first_full = np.minimum.reduceat(order, np.flatnonzero(new_full))
    first_xc = first_rows(new_xc)
    first_x = first_rows(new_x)
    group = np.empty(len(order), dtype=np.intp)
    group[order] = np.cumsum(new_full) - 1

    # One representative row per group: the row it first appeared at.
    # np.lexsort takes its *last* key as primary, so this orders by first
    # appearance of x, then of (x, c), then of the whole key.
    order = np.lexsort((first_full, first_xc, first_x))
    representative = first_full[order]

    totals = np.bincount(group, weights=n, minlength=first_full.size)[order]
    # ``bincount`` always returns float64; the counts are integers going
    # in and callers rely on that (an integer ``n`` must stay integer).
    totals = totals.astype(n.dtype)

    return x[representative], c[representative], totals, t[representative]


def xcnt_sort(
    x: npt.NDArray, c: npt.NDArray, n: npt.NDArray, t: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Sort ``xcnt`` arrays by ``x`` (the interval midpoint for 2-D ``x``),
    breaking ties by the lower truncation bound and then by the censoring
    flag (so at a tied time left-censored rows come first, then observed,
    then right-censored, then interval-censored).

    Returns
    -------
    x, c, n, t : arrays
        The same arrays, reordered together.
    """
    # One stable sort on the three keys (the last is the primary): the
    # order three stable sorts, by c, then t, then x, would give.
    t_key = t if t.ndim == 1 else t.min(axis=1)
    x_key = x if x.ndim == 1 else x.mean(axis=1)
    idx = np.lexsort((c, t_key, x_key))
    return x[idx], c[idx], n[idx], t[idx]


def fsli_handler(
    f: "npt.ArrayLike | None" = None,
    s: "npt.ArrayLike | None" = None,
    l: "npt.ArrayLike | None" = None,
    i: "npt.ArrayLike | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Validate data in the ``fsli`` format: separate lists of failures,
    suspensions (right censored), left censored values and intervals.
    Any combination may be given, but at least one must hold data. Each
    is returned as a float array.

    Parameters
    ----------
    f: array-like, optional (default: None)
        array of values for which the failure/death was observed
    s: array-like, optional (default: None)
        array of right censored observation values
    l: array-like, optional (default: None)
        array of left censored observation values
    i: array-like, optional (default: None)
        array of ``[lower, upper]`` pairs, one per interval censored
        observation, with ``lower < upper``

    Raises
    ------
    ValueError
        If no data is given, if ``f``, ``s`` or ``l`` is not
        one-dimensional, if ``i`` is not of shape ``(k, 2)``, if any value
        is NaN, or if an interval's lower value is not below its upper
        value.

    Returns
    -------
    f: array
        array of values for which the failure/death was observed that have
        been checked for correctness
    s: array
        array of right censored observation values that have been checked
        for correctness
    l: array
        array of left censored observation values that have been checked
        for correctness
    i: array
        array of interval censored data that have been checked for correctness


    Examples
    --------

    >>> from surpyval import fsli_handler
    >>> f = [1, 2, 3, 4, 5, 6]
    >>> s = [1, 2, 3]
    >>> l = [4, 5, 6]
    >>> i = [[1, 2], [3, 4]]
    >>> fsli_handler(f, s, l, i)
    (array([1., 2., 3., 4., 5., 6.]),
    array([1., 2., 3.]),
    array([4., 5., 6.]),
    array([[1., 2.],
            [3., 4.]]))
    """
    if (f is None) and (s is None) and (l is None) and (i is None):
        raise ValueError("Must enter some data!")

    f = validate_float_array(f, "f")
    s = validate_float_array(s, "s")
    l = validate_float_array(l, "l")
    i = validate_float_array(i, "i")

    if (len(f) == 0) and (len(s) == 0) and (len(l) == 0) and (len(i) == 0):
        raise ValueError("Must enter some data!")

    if f.ndim != 1:
        raise ValueError("'f' array must be one-dimensional")
    if s.ndim != 1:
        raise ValueError("'s' array must be one-dimensional")
    if l.ndim != 1:
        raise ValueError("'l' array must be one-dimensional")

    if (i.ndim != 2) and (i.size != 0):
        raise ValueError(
            "'i' array must be two-dimensional: one [lower, upper] pair"
            " per interval, of shape (k, 2)"
        )

    if len(i) > 0:
        if i.shape[1] != 2:
            raise ValueError("'i' array must be of shape (k, 2)")

    # NaN compares false with everything, so it would slip past every
    # check below and into the fitted data (xcnt_handler refuses it in
    # 'x').
    for name, arr in (("f", f), ("s", s), ("l", l), ("i", i)):
        if np.isnan(arr).any():
            raise ValueError(f"'{name}' cannot contain NaN values")

    if i.size != 0:
        if (i[:, 0] >= i[:, 1]).any():
            raise ValueError(
                "Lower interval must not be greater than or equal to the"
                " upper interval"
            )

    return f, s, l, i


def _whole_number_array(values: npt.ArrayLike, name: str) -> npt.NDArray:
    """``values`` as an integer array, refusing anything not a whole number.

    An integer-valued float array such as ``[5.0, 4.0]`` is accepted: that
    is how counts arrive from a DataFrame column with a missing value, or
    after any arithmetic, and ``astype(int, casting="safe")`` would refuse
    it. Fractions, NaN and infinities are refused, as are booleans' string
    cousins -- anything numpy cannot read as a number.
    """
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(f"'{name}' must be an array of integers.")
    if not np.isfinite(arr).all() or (arr != np.floor(arr)).any():
        raise ValueError(f"'{name}' must be an array of integers.")
    return arr.astype(int)


def xrd_handler(
    x: npt.ArrayLike, r: npt.ArrayLike, d: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Takes a combination of 'x', 'r', and 'd' arrays and ensures that the data
    is feasible: the arrays are one-dimensional and the same length, ``x``
    has no NaN and no repeated time, ``r`` and ``d`` hold whole numbers
    (integer-valued floats such as ``[5.0, 4.0]`` are accepted), every
    ``r`` is at least one, no ``d`` is negative and no ``d`` exceeds its
    ``r``.

    xrd data lists each distinct time once, with the number at risk and
    the number of deaths *at that time*. Each ``(x, r, d)`` row is
    therefore self-contained, so rows given out of order are sorted by
    ``x`` (carrying their ``r`` and ``d`` with them) rather than refused.
    A repeated time is refused: there is no unambiguous way to merge two
    rows that each claim to be the risk set at that time.

    Does not check that ``r`` decreases, as it can grow when there is
    left truncation (late entry).

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    r: array
        array of at risk items at each value of x
    d: array
        array of failures / deaths at each value of x

    Returns
    ----------

    x: array
        array of values of variable for which observations were made,
        in increasing order.
    r: array
        array of at risk items at each value of x
    d: array
        array of failures / deaths at each value of x

    Raises
    ------
    ValueError
        If any of the rules above is broken.

    Examples
    --------

    >>> from surpyval import xrd_handler
    >>> x = [1, 2, 3, 4, 5]
    >>> r = [5, 4, 3, 2, 1]
    >>> d = [1, 1, 1, 1, 1]
    >>> x, r, d = xrd_handler(x, r, d)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> r
    array([5, 4, 3, 2, 1])
    >>> d
    array([1, 1, 1, 1, 1])

    Rows out of order are sorted together:

    >>> xrd_handler([3, 1, 2], [2, 5, 4], [1, 1, 1])
    (array([1., 2., 3.]), array([5, 4, 2]), array([1, 1, 1]))
    """

    try:
        x = np.array(x, dtype=np.float64)
    except Exception:
        raise ValueError(
            "'x' must be an array of scalar numbers with real values."
        )

    r = _whole_number_array(r, "r")
    d = _whole_number_array(d, "d")

    if x.ndim != 1:
        raise ValueError("'x' must be a one dimensional array")

    if x.shape != r.shape:
        raise ValueError("'x' array not the same length as 'r' array")
    if x.shape != d.shape:
        raise ValueError("'x' array not the same length as 'd' array")

    if x.size == 0:
        raise ValueError("'x' is empty: xrd data needs at least one time")

    if np.isnan(x).any():
        raise ValueError("'x' cannot contain NaN values")

    if (d < 0).any():
        raise ValueError("'d' array cannot have any negative values")

    if (r <= 0).any():
        raise ValueError(
            "'r' at risk item counts must be positive: every listed time"
            " needs at least one item at risk"
        )

    if (d > r).any():
        raise ValueError(
            "cannot have more deaths/failures than there are items at risk"
        )

    # Every estimator walks the rows in order, multiplying (or summing)
    # one step per row, so unsorted rows would give a silently wrong
    # curve (and xrd_to_xcnt wrong rows). The rows are independent
    # triples, so sorting them together is exact.
    order = np.argsort(x, kind="stable")
    x, r, d = x[order], r[order], d[order]
    if (np.diff(x) == 0).any():
        raise ValueError(
            "'x' has repeated times ({}): xrd data lists each distinct"
            " time once, with all the deaths at that time in one row".format(
                np.unique(x[1:][np.diff(x) == 0])
            )
        )

    return x, r, d


def _time_kind(value: Any) -> "str | None":
    """``"m"`` if ``value`` holds durations (``timedelta64``, pandas
    ``Timedelta``, ``datetime.timedelta``), ``"M"`` if it holds dates or
    times (``datetime64``, pandas ``Timestamp``, ``datetime``), else
    ``None``. Looks into lists and tuples (interval pairs included)."""
    if isinstance(value, (np.timedelta64, _dt.timedelta)):
        # pandas Timedelta is a datetime.timedelta
        return "m"
    if isinstance(value, (np.datetime64, _dt.date)):
        # pandas Timestamp and datetime.datetime are datetime.date
        return "M"
    if isinstance(value, (list, tuple)):
        # numpy infers a list's dtype in C; only a list of objects (pandas
        # Timedelta, say) or a ragged one (pairs among scalars) is looked
        # into element by element
        try:
            value = np.asarray(value)
        except (ValueError, TypeError):
            for v in value:
                kind = _time_kind(v)
                if kind is not None:
                    return kind
            return None
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind in ("m", "M"):
        return str(kind)
    if kind == "O":
        for v in np.asarray(value).flat:
            kind = _time_kind(v)
            if kind is not None:
                return kind
    return None


def refuse_time_values(value: Any, name: str) -> None:
    """Refuse durations or dates given where SurPyval needs numbers (#480).

    A ``timedelta64`` array converts to a float array silently, in its
    storage ticks: seconds for ``timedelta64[s]``, nanoseconds for
    ``timedelta64[ns]``, whichever pandas happened to pick, so a fit to
    six durations in days came back with a scale of 5.0e5 (seconds) or,
    after multiplying the input by 1.37, 6.9e14 (nanoseconds), without a
    word. SurPyval has no unit of time -- every model is in the units of
    the numbers it is given -- so the unit is the caller's to choose:
    this raises a ``ValueError`` naming the argument and saying how to
    convert, rather than choosing one.

    Parameters
    ----------
    value : any
        The argument as given.
    name : str
        Its name, for the message.

    Raises
    ------
    ValueError
        If ``value`` holds durations or dates.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils import refuse_time_values
    >>> refuse_time_values([1.0, 2.0], "x")
    >>> days = np.array([1, 2], dtype="timedelta64[D]")
    >>> refuse_time_values(days, "x")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: 'x' holds durations (timedelta64 or pandas Timedelta)...
    """
    kind = _time_kind(value)
    if kind == "m":
        raise ValueError(
            f"'{name}' holds durations (timedelta64 or pandas Timedelta), "
            "which would be read in their storage ticks -- seconds or "
            "nanoseconds, depending on the dtype. SurPyval works in the "
            "units of the numbers it is given, so convert them to numbers "
            f"in the unit you want first: e.g. {name} / pd.Timedelta(days=1) "
            f"(or {name} / np.timedelta64(1, 'D')) for days, or "
            f"{name}.dt.total_seconds() for a pandas Series in seconds"
        )
    if kind == "M":
        raise ValueError(
            f"'{name}' holds dates or times (datetime64 or pandas "
            "Timestamp); SurPyval needs durations as numbers. Subtract "
            "each unit's start and convert to the unit you want: e.g. "
            f"({name} - start) / pd.Timedelta(days=1) for days"
        )


def coerce_xcnt_x(x: npt.ArrayLike) -> npt.NDArray:
    """
    Coerce the ``x`` variable of xcnt-format data into a float numpy array.

    Accepts a scalar (a single observation), a 1D array of event values,
    or a 2D array / list-of-pairs of ``[left, right]`` interval bounds. In
    a list (or tuple), any element that is itself a list, tuple or array
    is an interval row, and the scalar elements are rows with equal ends.
    Validates dimensionality, the absence of NaNs and the interval
    ordering (``left <= right``). Shared by the univariate
    (``xcnt_handler``) and recurrent (``handle_xicn``) handlers.
    Durations and dates are refused (see ``refuse_time_values``).
    """
    refuse_time_values(x, "x")
    if isinstance(x, (list, tuple)) and any(
        isinstance(v, (list, tuple, np.ndarray)) and np.ndim(v) > 0 for v in x
    ):
        # A ragged mix of scalars and pairs. Tuples and arrays count as
        # pairs as lists do: ``[1, (2, 3), 4]`` would otherwise reach
        # np.array and fail with numpy's "inhomogeneous shape" error.
        x_ndarray = np.empty(shape=(len(x), 2))
        for idx, val in enumerate(x):
            try:
                val_arr = np.atleast_1d(np.asarray(val, dtype=float))
            except (ValueError, TypeError):
                raise ValueError(
                    "Each element of 'x' must be a number or a [left, right]"
                    " pair of numbers"
                )
            if val_arr.ndim != 1 or len(val_arr) not in (1, 2):
                raise ValueError(
                    "Each element of 'x' must be either scalar or"
                    " array-like of no more than length 2"
                )
            x_ndarray[idx, :] = val_arr
        x = x_ndarray
    else:
        # Always a copy: the handlers rewrite interval endpoints in place
        # (an infinite endpoint becomes a one-sided censoring), which must
        # not reach the caller's array, or refitting it would give a
        # different answer. ``atleast_1d``: a scalar is one observation.
        try:
            x = np.atleast_1d(np.array(x, dtype=float))
        except (ValueError, TypeError):
            raise ValueError(
                "Variable 'x' must be numbers, or [left, right] pairs of"
                " numbers"
            )

    if x.ndim == 2 and x.shape[1] == 1:
        # A single column, e.g. ``df[["t"]].to_numpy()``: one observation
        # per row, as sklearn reads a column vector ``y`` (#485).
        x = x[:, 0]
    if x.ndim > 2:
        raise ValueError("Variable 'x' array must be one or two dimensional")
    # Before the ordering check, which NaN would fail with a misleading
    # "left intervals must be less than ..." message.
    if np.isnan(x).any():
        raise ValueError("Variable 'x' cannot contain NaN values")
    if x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError(
                "Dimension 1 must be equal to 2, try transposing data, or do"
                " you have a 1d array in a 2d array?"
            )
        if not (x[:, 0] <= x[:, 1]).all():
            raise ValueError(
                "All left intervals must be less than or equal to right"
                " intervals"
            )
    return x


def format_truncation(
    t: "npt.ArrayLike | None",
    tl: "npt.ArrayLike | Number | None",
    tr: "npt.ArrayLike | Number | None",
    n_rows: int,
) -> npt.NDArray:
    """
    Build the ``(n_rows, 2)`` truncation array from either a ``t`` matrix or
    separate ``tl``/``tr`` bounds (scalars, including 0-d arrays, broadcast
    to all rows). The default window is the whole real line
    ``[-inf, inf]``. NaN bounds are refused. Shared by ``xcnt_handler`` and
    ``handle_xicn``.
    """
    for value, name in ((t, "t"), (tl, "tl"), (tr, "tr")):
        refuse_time_values(value, name)
    if t is not None and ((tl is not None) or (tr is not None)):
        raise ValueError(
            "Cannot use 't' with 'tl' or 'tr'. Use either 't' or any"
            " combination of 'tl' and 'tr'"
        )

    if (t is None) and (tl is None) and (tr is None):
        tl_arr = np.ones(n_rows) * -np.inf
        tr_arr = np.ones(n_rows) * np.inf
        return np.vstack([tl_arr, tr_arr]).T

    if (tl is not None) or (tr is not None):
        tl_arr = _truncation_bound(tl, "tl", -np.inf, n_rows)
        tr_arr = _truncation_bound(tr, "tr", np.inf, n_rows)

        if tl_arr.ndim > 1 or tr_arr.ndim > 1:
            raise ValueError(
                "Truncation arrays must be one dimensional, did you mean to"
                " use 't'"
            )
        if tl_arr.shape[0] != n_rows or tr_arr.shape[0] != n_rows:
            raise ValueError("'tl' and 'tr' must be the same length as 'x'")
        t = np.vstack([tl_arr, tr_arr]).T
    else:
        try:
            t = np.array(t, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("Truncation bounds 't' must be numbers")
        if t.ndim != 2:
            raise ValueError("Truncation ndarray must be 2 dimensional")
        if t.shape[0] != n_rows:
            raise ValueError(
                "Truncation ndarray must be same shape as variable array"
            )
        if t.shape[1] != 2:
            raise ValueError(
                "Truncation array must have shape (n, 2) with left and right"
                " bounds"
            )

    # NaN compares false with everything, so a NaN bound would pass every
    # validity check and then mean something different to each fitter:
    # "no truncation" to the parametric likelihood, a divide-by-zero in
    # Kaplan-Meier's risk sets, a third answer from Turnbull. A missing
    # bound is spelled -inf (left) or inf (right).
    if np.isnan(t).any():
        raise ValueError(
            "Truncation bounds must not contain NaN: use -inf for no left"
            " truncation and inf for no right truncation"
        )
    return t


def _truncation_bound(
    bound: "npt.ArrayLike | Number | None",
    name: str,
    default: float,
    n_rows: int,
) -> npt.NDArray:
    """One truncation bound as a float array, broadcasting a scalar.

    ``np.ndim`` rather than ``np.isscalar``: a 0-d array such as
    ``np.array(0.5)`` is not a "scalar" to numpy, and is broadcast like
    one rather than kept 0-d (which fails the length check with "tuple
    index out of range").
    """
    if bound is None:
        return np.full(n_rows, default)
    try:
        arr = np.array(bound, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"Truncation bound '{name}' must be numbers")
    if arr.ndim == 0:
        return np.full(n_rows, float(arr))
    return arr


def _check_truncation_bounds(
    x: npt.NDArray, c: npt.NDArray, t: npt.NDArray
) -> None:
    """Refuse a row that cannot have been seen inside its truncation window.

    The window is ``(tl, tr]`` and a row's event time ``X`` must be able
    to fall in it. A single value (observed, or censored on one side) is
    held to the same rules whether ``x`` has one column or two:

    * ``tl < x``: at ``x == tl`` the observation window has zero length
      (#260); a left censored ``X <= tl`` is likewise impossible.
    * ``x <= tr``, and for a right censored row ``x < tr``: censored at
      ``tr`` means ``tr < X <= tr``, an empty set. The likelihood for it
      is ``log 0``, and the fit would fail with a stream of warnings.

    An interval row ``[xl, xr]`` means ``xl < X <= xr``, so it may start
    at ``tl`` (``tl <= xl``) and must end by ``tr`` (``xr <= tr``).
    """
    lo = x if x.ndim == 1 else x[:, 0]
    hi = x if x.ndim == 1 else x[:, 1]
    point = lo == hi
    has_tl = np.isfinite(t[:, 0])
    has_tr = np.isfinite(t[:, 1])

    if (has_tl & point & (t[:, 0] >= lo)).any():
        # Strictly less: under the (entry, exit] risk-interval convention
        # a value at exactly its own left-truncation time has a
        # zero-length observation window — contradictory data that would
        # silently distort the Turnbull estimate (#260).
        raise ValueError(
            "All left truncated values must be strictly less than the"
            + " respective observed values: a value at its own left"
            + " truncation time has a zero-length observation window."
        )
    if (has_tl & ~point & (t[:, 0] > lo)).any():
        # Inspection data meets this often, and the tempting repair --
        # moving xl up to tl -- biases the fit, so the message says what
        # tl is for such a row (#576).
        raise ValueError(
            "All left truncated values must be less than the respective"
            + " observed values: an interval cannot start below its own"
            + " left truncation time. In inspection data a failure found"
            + " at the first inspection after the records begin starts at"
            + " its last good inspection, before the records do: its left"
            + " truncation time is that inspection (so tl <= xl), not the"
            + " date the records begin. Do not move xl up to tl, which"
            + " biases the fit."
        )
    if (has_tr & (hi > t[:, 1])).any():
        raise ValueError(
            "All right truncated values must be greater than the"
            + " respective observed values"
        )
    if (has_tr & point & (c == 1) & (lo >= t[:, 1])).any():
        raise ValueError(
            "A right censored value must be strictly less than its right"
            + " truncation time: censored at tr, the event would have to"
            + " lie after tr and at or before it."
        )


def xcnt_handler(
    x: "npt.ArrayLike | None" = None,
    c: "npt.ArrayLike | None" = None,
    n: "npt.ArrayLike | None" = None,
    t: "npt.ArrayLike | None" = None,
    xl: "npt.ArrayLike | None" = None,
    xr: "npt.ArrayLike | None" = None,
    tl: "npt.ArrayLike | Number | None" = None,
    tr: "npt.ArrayLike | Number | None" = None,
    group_and_sort: bool = True,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Main handler that ensures any input to a surpyval fitter meets the
    requirements to be used in one of the parametric or nonparametric fitters.

    It converts the inputs to numpy arrays and checks them: ``x`` is not
    empty and has no NaN (a scalar is one observation); ``c`` holds only
    -1, 0 and 1 (and 2 for a two-column ``x``); ``n`` holds positive
    whole numbers; the truncation bounds have no NaN and the window of
    each row has its left bound below its right bound. For two-column
    ``x``, a row with equal values is not an interval, a row with
    different values must be flagged 2 (when ``c`` is given), and an
    infinite end turns the row into one-sided censoring: ``[v, inf]``
    becomes right censored at ``v`` and ``[-inf, v]`` left censored at
    ``v``. If no interval is left, ``x`` is returned with one column.

    Each row must then fit its truncation window ``(tl, tr]``: a single
    value lies strictly above ``tl`` and at or below ``tr`` -- strictly
    below ``tr`` if it is right censored -- whether ``x`` has one column
    or two; an interval ``[xl, xr]`` needs ``tl <= xl`` and
    ``xr <= tr``. Identical rows are then merged (their counts summed)
    and the rows sorted with :func:`xcnt_sort`.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array, optional (default: None)
        array of censoring values (-1, 0, 1, 2) corresponding to x
    n: array, optional (default: None)
        array of count of observations at each x and with censoring c
    t: array, optional (default: None)
        array of values with shape (?, 2) with the left and right value of
        truncation
    xl: array or scalar, optional (default: None)
        array of the values of the left interval of interval censored data.
        Cannot be used with 'x' parameter, must be used with the 'xr'
        parameter
    xr: array or scalar, optional (default: None)
        array of the values of the right interval of interval censored data.
        Cannot be used with 'x' parameter, must be used with the 'xl'
        parameter
    tl: array or scalar, optional (default: None)
        array of values of the left value of truncation. If scalar, all values
        will be treated as left truncated by that value
        cannot be used with 't' parameter but can be used with the 'tr'
        parameter
    tr: array or scalar, optional (default: None)
        array of values of the right value of truncation. If scalar, all
        values will be treated as right truncated by that value
        cannot be used with 't' parameter but can be used with the 'tl'
        parameter
    group_and_sort: bool, optional (default: True)
        whether to group and sort the data. If False, the data will be returned
        in the order it was entered. This is useful for when validating
        survival data for which you also have covariates.

    Returns
    ----------

    x: array
        sorted array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corresponding to output array
        x. If c was None, every row is observed (0), except that the rows of
        a two-column x with different values are interval censored (2).
    n: array
        array of count of observations at output array x and with censoring c.
        If n was None, count array assumed to be all one observation.
    t: array
        array of truncation values of observations at output array x and with
        censoring c.

    Raises
    ------
    ValueError
        If the inputs break any of the rules above: for example ``x`` and
        ``xl``/``xr`` both given, empty ``x``, arrays of different
        lengths, a NaN in ``x`` or in a truncation bound, an unknown
        censoring flag, a count that is not a positive whole number, a
        value at or below its own left truncation, or a right censored
        value at its own right truncation.

    Examples
    --------

    >>> from surpyval import xcnt_handler
    >>> x = [1, 2, 3, 4, 5]
    >>> c = [0, 0, 1, 1, 1]
    >>> n = [1, 1, 1, 1, 1]
    >>> t = [[0, 6], [0, 6], [0, 6], [0, 6], [0, 6]]
    >>> xcnt_handler(x, c, n, t)
    (array([1., 2., 3., 4., 5.]),
    array([0, 0, 1, 1, 1]),
    array([1, 1, 1, 1, 1]),
    array([[0., 6.],
            [0., 6.],
            [0., 6.],
            [0., 6.],
            [0., 6.]]))
    >>> xcnt_handler(x, c, n, tl=0, tr=6)
    (array([1., 2., 3., 4., 5.]),
    array([0, 0, 1, 1, 1]),
    array([1, 1, 1, 1, 1]),
    array([[0., 6.],
            [0., 6.],
            [0., 6.],
            [0., 6.],
            [0., 6.]]))
    >>> xl = [1, 2, 3, 4, 5]
    >>> xr = [2, 3, 4, 5, 6]
    >>> xcnt_handler(xl=xl, xr=xr)
    (array([[1., 2.],
            [2., 3.],
            [3., 4.],
            [4., 5.],
            [5., 6.]]),
    array([2, 2, 2, 2, 2]),
    array([1, 1, 1, 1, 1]),
    array([[-inf,  inf],
            [-inf,  inf],
            [-inf,  inf],
            [-inf,  inf],
            [-inf,  inf]]))
    """

    x, c, n, t, xl, xr, tl, tr = (
        _numeric_list_as_array(v) for v in (x, c, n, t, xl, xr, tl, tr)
    )
    x = _xcnt_x(x, xl, xr)
    c = _xcnt_censoring(c, x)
    n = _xcnt_counts(n, x)

    t = format_truncation(t, tl, tr, x.shape[0])

    if (t[:, 1] <= t[:, 0]).any():
        raise ValueError(
            "All left truncated values must be less than right truncated"
            + " values"
        )

    if x.ndim == 2:
        x, c = _one_sided_rows(x, c)

    _check_truncation_bounds(x, c, t)

    x = x.astype(float)
    c = c.astype(int)
    n = n.astype(int)
    t = t.astype(float)

    # Right censoring with a finite right truncation is not contradictory
    # data (#195): the unit was detected, so its event is at or before
    # ``tr``, and it was censored, so the event is after ``x``. Together
    # that is simply ``x < X <= tr`` -- the ordinary setup of a
    # flux-limited or reporting-delay sample, e.g. a detector that
    # registers an event but cannot resolve where in the remaining window
    # it fell. The likelihood gives such a row the conditional
    # ``F(tr) - F(x)``, not the unconditional ``S(x)`` (which includes the
    # ``X > tr`` region the truncation excludes and has no maximum, #310),
    # so the fit is well posed and no warning is given.

    if group_and_sort:
        x, c, n, t = group_xcnt(x, c, n, t)
        x, c, n, t = xcnt_sort(x, c, n, t)

    return x, c, n, t


def _numeric_list_as_array(value: Any) -> Any:
    """A flat list or tuple of numbers as an array, anything else as given.

    The checks after this convert a list several times over: once each to
    look for durations and dates (``refuse_time_values``), for missing
    values and to read it as floats, about 3 ms per 100,000 numbers
    every time. An array answers the first two from its dtype, so
    converting once here saves most of that. Only a list numpy reads as
    one-dimensional numbers is converted: a list of pairs, a ragged one,
    or one holding objects (dates, pandas ``NA``, numbers too large for
    an integer dtype) reaches the checks as given, which read it as they
    always have.
    """
    if not isinstance(value, (list, tuple)):
        return value
    try:
        arr = np.asarray(value)
    except (ValueError, TypeError):
        return value
    if arr.ndim == 1 and arr.dtype.kind in "biuf":
        return arr
    return value


def _xcnt_x(
    x: "npt.ArrayLike | None",
    xl: "npt.ArrayLike | None",
    xr: "npt.ArrayLike | None",
) -> npt.NDArray:
    """``x`` as a checked, non-empty array, built from ``xl``/``xr``."""
    if (x is None) and (xl is None) and (xr is None):
        raise ValueError(
            "Must enter some data! Use either 'x' or both 'xl and 'xr'"
        )

    if (x is not None) and ((xl is not None) or (xr is not None)):
        raise ValueError("Must use either 'x' or both 'xl and 'xr'")

    if (x is None) and ((xl is None) or (xr is None)):
        raise ValueError("Must use either 'x' or both 'xl and 'xr'")

    if x is None:
        refuse_time_values(xl, "xl")
        refuse_time_values(xr, "xr")
        try:
            xl = np.array(xl, dtype=np.float64)
            xr = np.array(xr, dtype=np.float64)
        except Exception:
            raise ValueError(
                "'xl' and 'xr' must be an array of scalar numbers with real"
                + " values."
            )
        try:
            x = np.vstack([xl, xr]).T
        except Exception:
            raise ValueError("'xl' and 'xr' must be the same length")

    x = coerce_xcnt_x(x)

    # An empty sample is refused here: otherwise it builds a model
    # (Kaplan-Meier) that fails on first use, or fails inside a reduction
    # with "zero-size array to reduction operation" (parametric).
    if x.shape[0] == 0:
        raise ValueError("'x' is empty: at least one observation is needed")
    return x


def _check_interval_flags(c: npt.NDArray, x: npt.NDArray) -> None:
    """The flags of a two-column ``x``: 2 exactly on the interval rows."""
    if np.any(c[x[:, 0] == x[:, 1]] == 2):
        raise ValueError(
            "Censor flag indicates interval censored but only has one"
            + " failure time"
        )

    if np.any(c[x[:, 0] != x[:, 1]] != 2):
        mask1 = x[:, 0] != x[:, 1]
        mask2 = c != 2
        m = mask1 & mask2
        raise ValueError(
            "Censor flag indicates not interval censored but has"
            + " interval window.\nx:\n"
            + f"{x[m, :]}\n"
            + "censor flags:\n"
            + f"{c[m]}"
        )

    if np.any((c == 2) & (x[:, 0] == x[:, 1])):
        raise ValueError(
            "Censor flag provided, but case where interval flagged as"
            + " non interval censoring"
        )

    if np.any((c != 0) & (c != 1) & (c != -1) & (c != 2)):
        raise ValueError("Censoring value must only be one of -1, 0, 1, or 2")


def _has_missing(values: Any) -> bool:
    """Whether ``values`` holds a missing value: NaN, ``None``, or
    pandas' ``NA`` / ``NaT`` (a nullable ``Int64`` column gives ``NA``),
    told apart without importing pandas."""
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        return bool(np.isnan(arr).any())
    if arr.dtype.kind != "O":
        return False
    return any(
        v is None
        or (isinstance(v, float) and v != v)
        or type(v).__name__ in ("NAType", "NaTType")
        for v in arr.ravel()
    )


def _xcnt_censoring(c: "npt.ArrayLike | None", x: npt.NDArray) -> npt.NDArray:
    """The censoring flags, checked against ``x``, or the default.

    The default is observed (0), and interval censored (2) for the rows
    of a two-column ``x`` with different ends.
    """
    if c is None:
        default = np.zeros(x.shape[0])
        if x.ndim != 1:
            default[x[:, 0] != x[:, 1]] = 2
        return default

    c_arr = np.atleast_1d(np.array(c))
    # Said as it is said of ``x``: a missing flag used to be reported as
    # an invalid censoring value (#576).
    if _has_missing(c_arr):
        raise ValueError("Variable 'c' cannot contain NaN values")
    if c_arr.ndim == 2 and c_arr.shape[1] == 1:
        # A single column, as for ``x`` (#485)
        c_arr = c_arr[:, 0]
    if c_arr.ndim != 1:
        raise ValueError("Censoring flag array must be one dimensional")

    if c_arr.shape[0] != x.shape[0]:
        raise ValueError("'c' must be the same length as 'x'")

    if x.ndim == 2:
        _check_interval_flags(c_arr, x)
    elif np.any((c_arr != 0) & (c_arr != 1) & (c_arr != -1)):
        raise ValueError(
            "Censoring value must only be one of -1, 0, 1 for single"
            + " dimension input (0 failure, 1 right-censored, -1"
            + " left-censored). For interval censoring use c=2 with x"
            + " given as [left, right] pairs."
        )
    return c_arr


def _xcnt_counts(n: "npt.ArrayLike | None", x: npt.NDArray) -> npt.NDArray:
    """The counts as positive whole numbers, or one per row."""
    if n is None:
        # Do check here for groupby and binning
        return np.ones(x.shape[0])
    # A numeric array is checked as it is: as objects, one by one, the
    # check took a third of a Weibull fit to a million rows (#552).
    as_given = n if isinstance(n, np.ndarray) else np.array(n, dtype=object)
    if _has_missing(as_given):
        # As for ``x`` and ``c``; it read "must contain integer values"
        raise ValueError("Variable 'n' cannot contain NaN values")
    try:
        n_arr = np.atleast_1d(np.array(n, dtype=float))
    except (ValueError, TypeError):
        raise ValueError("Count array 'n' must contain integer values")
    if n_arr.ndim == 2 and n_arr.shape[1] == 1:
        n_arr = n_arr[:, 0]
    if n_arr.ndim != 1:
        raise ValueError("Count array must be one dimensional")
    if n_arr.shape[0] != x.shape[0]:
        raise ValueError("'n' must be the same length as 'x'")
    # isfinite as well: floor(inf) == inf, so an infinite count would pass
    # the whole-number test and become garbage in the integer cast.
    if not (np.isfinite(n_arr) & np.equal(n_arr, np.floor(n_arr))).all():
        raise ValueError("Count array 'n' must contain integer values")
    if not (n_arr > 0).all():
        raise ValueError("count array can't be 0 or less")
    return n_arr


def _one_sided_rows(
    x: npt.NDArray, c: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """A two-column ``x`` with its infinite ends made one-sided censoring.

    ``[v, inf]`` becomes right censored at ``v`` and ``[-inf, v]`` left
    censored at ``v`` (``x`` and ``c`` are changed in place). This runs
    before the truncation checks, so ``[5, inf]`` with ``tr=10`` is
    checked as the equivalent one-column row (5, right censored), not by
    its infinite end.
    """
    if np.isinf(x).all(axis=1).any():
        raise ValueError(
            "Interval censored entry has no info: in range (-inf, inf)"
        )
    # Convert interval censored from (v, to inf) to
    # a right censored point
    mask = np.isinf(x[:, 1])
    x[mask, 1] = x[mask, 0]
    c[mask] = 1

    # Convert interval censored from (-inf to v) to
    # a left censored point
    mask = np.isinf(x[:, 0])
    x[mask, 0] = x[mask, 1]
    c[mask] = -1

    # With no interval left the second column carries nothing, so hand
    # back the one-column form every fitter accepts: Kaplan-Meier,
    # Nelson-Aalen, the probability-plot and moment fitters,
    # xcnt_to_xrd and several regression fitters only take a 1-D x, and
    # ``xl``/``xr`` data with no real intervals (e.g. from a DataFrame
    # with separate left and right columns) is common.
    if (x[:, 0] == x[:, 1]).all():
        x = x[:, 0].copy()
    return x, c


def xcn_to_fs(
    x: npt.ArrayLike,
    c: "npt.ArrayLike | None" = None,
    n: "npt.ArrayLike | None" = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Convert observed and right-censored ``xcn`` data to the ``fs`` format:
    one array of failure times and one of suspension (right-censored)
    times, each time repeated by its count.

    Parameters
    ----------
    x : array like
        The times.
    c : array like, optional
        Censoring flags: 0 observed, 1 right-censored. Other values are
        dropped. Defaults to all observed.
    n : array like, optional
        The count at each time, a whole number. Defaults to 1.

    Returns
    -------
    f, s : arrays
        The failure times and the suspension times.

    Notes
    -----
    A two-column ``x`` (as :func:`xcnt_handler` returns for interval
    data) is accepted: its interval rows are dropped like any other
    non-0/1 flag, and every other row must have equal ends.

    Raises
    ------
    ValueError
        If ``c`` or ``n`` is not the same length as ``x``, a count is not
        a non-negative whole number, or an observed or right censored row
        of a two-column ``x`` has different ends.

    Examples
    --------
    >>> from surpyval import xcn_to_fs
    >>> xcn_to_fs([1, 2, 5], [0, 1, 0], [2, 1, 1])
    (array([1, 1, 5]), array([2]))
    """
    # Validated because the conversion is silent otherwise: a count of
    # 1.7 would be truncated to one item, a length mismatch would surface
    # as an IndexError, and a two-column x would crash inside np.repeat.
    x = np.atleast_1d(np.array(x))
    if x.ndim == 2 and x.shape[1] == 2:
        interval = x[:, 0] != x[:, 1]
        if c is None:
            c = np.where(interval, 2, 0)
        c = np.atleast_1d(np.array(c))
        if c.shape != interval.shape:
            raise ValueError("'c' must be the same length as 'x'")
        if (interval & ((c == 0) | (c == 1))).any():
            raise ValueError(
                "An observed or right censored row of a two-column 'x' must"
                " have equal ends"
            )
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(
            "'x' must be one-dimensional, or two columns of [left, right]"
        )
    if c is None:
        c = np.zeros(x.shape, dtype=int)
    c = np.atleast_1d(np.array(c))
    if c.shape != x.shape:
        raise ValueError("'c' must be the same length as 'x'")

    if n is None:
        n = np.ones(x.shape, dtype=int)
    try:
        n_float = np.atleast_1d(np.array(n, dtype=float))
    except (ValueError, TypeError):
        raise ValueError("Counts 'n' must be non-negative whole numbers")
    if n_float.shape != x.shape:
        raise ValueError("'n' must be the same length as 'x'")
    if not (
        np.isfinite(n_float) & (n_float == np.floor(n_float)) & (n_float >= 0)
    ).all():
        raise ValueError("Counts 'n' must be non-negative whole numbers")
    n = n_float.astype(int)

    f = np.repeat(x[c == 0], n[c == 0])
    s = np.repeat(x[c == 1], n[c == 1])
    return f, s


def _entered_before(
    tl: npt.NDArray, x: npt.NDArray, n: npt.NDArray
) -> npt.NDArray:
    """Weighted count of observations that entered strictly before each ``x``.

    ``e[j] = sum_i n_i * 1[tl_i < x_j]`` -- the ``(entry, exit]``
    risk-interval convention (#260): a subject entering exactly at an event
    time is not at risk for that event.

    Written directly this is
    ``((tl[:, None] < x[None, :]) * n[:, None]).sum(0)``, which materialises
    an ``N x K`` matrix and so costs quadratic time *and* memory: at 20,000
    observations that is a 3.2 GB intermediate taking ~15 s, and past ~50,000
    it raises ``MemoryError`` outright. Two cheap branches give identical
    counts:

    * **nothing is left truncated** -- the default and by far the common
      case. Every observation has entered before every event time, so the
      whole matrix is ``True`` and ``e`` collapses to the constant
      ``n.sum()``, with no matrix built to recompute a scalar.
    * **otherwise** -- sort the entry times once and read off the cumulative
      weight below each ``x`` with ``searchsorted``. ``side="left"`` counts
      entries *strictly* less than ``x``, matching the ``<`` above exactly,
      so the ``(entry, exit]`` convention is preserved unchanged.

    Counts are integers (``xcnt_handler`` rejects non-integer ``n``), so the
    cumulative sum is exact and equals the summation it replaces bit for
    bit. ``np.sort`` orders ``nan`` last and ``searchsorted`` treats it as
    larger than every value, which reproduces ``nan < x == False`` -- the
    behaviour of the form this replaces.
    """
    if np.isneginf(tl).all():
        # No left truncation: everything is at risk from the outset.
        return np.full(x.shape, n.sum(), dtype=n.dtype)

    order = np.argsort(tl, kind="stable")
    tl_sorted = tl[order]
    cumulative = np.concatenate(
        [np.zeros(1, dtype=n.dtype), np.cumsum(n[order])]
    )
    return cumulative[np.searchsorted(tl_sorted, x, side="left")]


def xcnt_to_xrd(
    x: npt.ArrayLike,
    c: "npt.ArrayLike | None" = None,
    n: "npt.ArrayLike | None" = None,
    t: "npt.ArrayLike | None" = None,
    **kwargs: Any,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Converts the xcnt format to the xrd format: the distinct times, the
    number at risk at each and the number of deaths at each. The data is
    validated with :func:`xcnt_handler` first. Only observed and right
    censored rows without right truncation can be converted; left
    truncation is allowed and sets when each item enters the risk set,
    under the (entry, exit] convention.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array, optional (default: None)
        array of censoring values (0 or 1) corresponding to x. If None, an
        array of 0s is created corresponding to each x.
    n: array, optional (default: None)
        array of count of observations at each x and with censoring c. If None,
        an array of ones is created.
    t: array, optional (default: None)
        array of shape (?, 2) of truncation bounds; the right bounds must
        be infinite.
    kwargs: keywords for truncation, ``tl`` and ``tr``, used in place of
        ``t`` as in :func:`xcnt_handler`

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    r: array
        array of count of units/people at risk at time x (including if it had
        an event at 'x').
    d: array
        array of the count of failures/deaths at each time x.

    Raises
    ------
    ValueError
        If any row is left (-1) or interval (2) censored, or right
        truncated.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> c = np.array([0, 1, 1, 0, 0])
    >>> n = np.array([1, 1, 1, 1, 1])
    >>> x, r, d = xcnt_to_xrd(x, c, n)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> r
    array([5, 4, 3, 2, 1])
    >>> d
    array([1, 0, 0, 1, 1])
    >>> # Using left truncated data: under the (entry, exit] convention a
    >>> # subject entering exactly at an event time is not at risk for it.
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> tl = np.array([0, 1, 2, 3, 4])
    >>> x, r, d = xcnt_to_xrd(x, tl=tl)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> r
    array([1, 1, 1, 1, 1])
    >>> d
    array([1, 1, 1, 1, 1])
    """
    return _handled_xcnt_to_xrd(*xcnt_handler(x, c, n, t, **kwargs))


def _handled_xcnt_to_xrd(
    x: npt.NDArray, c: npt.NDArray, n: npt.NDArray, t: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """:func:`xcnt_to_xrd` of data that ``xcnt_handler`` has already
    validated, grouped and sorted (which it leaves as they are): a fit
    that has handled its data need not pay for it again, a third of a
    Kaplan-Meier fit."""
    if np.isfinite(t[:, 1]).any():
        raise ValueError("xrd format can't be used right truncated data")

    # No warning for a shared entry time: validation guarantees tl < x for
    # every observation, so a common entry time never alters the (entry,
    # exit] risk sets -- a warning here would fire on perfectly ordinary
    # inputs like tl=0 everywhere, e.g. from check_ph on a model fit with
    # a constant entry column (#282).

    if ((c != 1) & (c != 0)).any():
        raise ValueError(
            "xrd format can't be used with left (c=-1) or interval (c=2)"
            + " censoring"
        )

    tl = t[:, 0]
    x, idx = np.unique(x, return_inverse=True)
    # d is the number of deaths (events) at each x
    d = np.bincount(idx, weights=n * (1 - c))
    # do is drop outs - i.e right censored
    do = np.bincount(idx, weights=n * c)
    # e is the number of items that have entered observation *before* each
    # x: the standard (entry, exit] risk-interval convention (R survival /
    # lifelines) — a subject entering exactly at an event time is not at
    # risk for that event. The previous inclusive comparison put surpyval's
    # KM on the nonstandard side and made it disagree with Turnbull's NPMLE
    # at exact entry/event ties (#260).
    e = _entered_before(tl, x, n)
    # r is the number of people at risk at each x
    r = e + d - d.cumsum() + do - do.cumsum()
    # change to correct data types
    r = r.astype(int)
    d = d.astype(int)
    x = x.astype(float)
    return x, r, d


def xrd_to_xcnt(
    x: npt.ArrayLike, r: npt.ArrayLike, d: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Converts the xrd format to the xcnt format. Each death becomes an
    observed row, and the items that leave the risk set without dying
    between ``x[j]`` and ``x[j + 1]``, ``r[j] - d[j] - r[j + 1]``, become
    right censored rows at ``x[j]`` (after the last time, ``r - d``).
    The input is validated with :func:`xrd_handler`, so the times must be
    distinct; rows out of order are sorted first. The result has no
    truncation and no left or interval censoring.

    Note: left truncation cannot be recovered from the xrd format because
    the at-risk count `r` collapses per-subject truncation times into a
    single scalar. Use xcnt format directly when left truncation is present.

    Parameters
    ----------

    x: array
        array of values of variable for which observations were made.
    r: array
        array of at risk items at each value of x
    d: array
        array of failures / deaths at each value of x

    Returns
    -------
    x: array
        array of values of variable for which observations were made.
    c: array
        array of censoring values (0 or 1) corresponding to x
    n: array
        array of count of observations at each x and with censoring c
    t: array
        array of values with shape (?, 2) with the left and right value of
        truncation (all ``[-inf, inf]``)

    Raises
    ------
    ValueError
        If :func:`xrd_handler` refuses the data, or if the risk set grows
        from one time to the next (late entry), which the xcnt output
        cannot represent.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> r = np.array([5, 4, 3, 2, 1])
    >>> d = np.array([1, 0, 0, 1, 1])
    >>> x, c, n, t = xrd_to_xcnt(x, r, d)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> c
    array([0, 1, 1, 0, 0])
    >>> n
    array([1, 1, 1, 1, 1])
    >>> t
    array([[-inf,  inf],
           [-inf,  inf],
           [-inf,  inf],
           [-inf,  inf],
           [-inf,  inf]])
    """
    # Validated (and sorted) like every other xrd input: the growth check
    # and the drop-out arithmetic below assume distinct, increasing times,
    # and would return the wrong rows for unsorted ones.
    x, r, d = xrd_handler(x, r, d)
    n_f = np.copy(d)
    x_f = np.copy(x)
    mask = n_f != 0
    n_f = n_f[mask]
    x_f = x_f[mask]

    # A risk set that grows (after accounting for that step's events) means
    # late entry / left truncation, which the xcnt output cannot represent,
    # so it is refused rather than turned into a different study (#281).
    r_arr = np.asarray(r)
    d_arr = np.asarray(d)
    if (np.diff(r_arr) + d_arr[:-1] > 0).any():
        raise ValueError(
            "The risk set increases between observation times (late entry /"
            " left truncation); truncation cannot be represented in xcnt"
            " output, so this data cannot be converted with xrd_to_xcnt."
        )

    x = np.asarray(x)
    delta = np.abs(np.diff(np.hstack([r, [0]])))

    sus = delta - d
    x_s = x[sus > 0]
    n_s = sus[sus > 0]

    x_f = np.repeat(np.asarray(x_f), np.asarray(n_f, dtype=int))
    x_s = np.repeat(x_s, np.asarray(n_s, dtype=int))

    return fs_to_xcnt(x_f, x_s)


def fsli_to_xcnt(
    f: "npt.ArrayLike | None" = None,
    s: "npt.ArrayLike | None" = None,
    l: "npt.ArrayLike | None" = None,
    i: "npt.ArrayLike | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Converts the fsli format to the xcnt format, so that the data can be
    passed to one of the parametric or nonparametric fitters. The inputs
    are validated with :func:`fsli_handler`. Repeated values are counted
    in ``n``. When there are intervals, ``x`` is returned with two
    columns, the other rows repeating their value.

    Parameters
    ----------
    f: array
        array of values for which the failure/death was observed
    s: array
        array of right censored observation values
    l: array
        array of left censored observation values
    i: array
        array of ``[lower, upper]`` pairs of interval censored data

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corresponding to output array
        x.
    n: array
        array of count of observations at output array x and with censoring
        c.
    t: ndarray
        ndarray of truncation values of observations at output array x and with
        censoring c.

    Examples
    --------

    >>> from surpyval import fsli_to_xcnt
    >>> f = [1, 4, 5]
    >>> s = [2, 3]
    >>> l = []
    >>> i = []
    >>> x, c, n, t = fsli_to_xcnt(f, s, l, i)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> c
    array([0, 1, 1, 0, 0])
    >>> n
    array([1, 1, 1, 1, 1])
    >>> t
    array([[-inf,  inf],
           [-inf,  inf],
           [-inf,  inf],
           [-inf,  inf],
           [-inf,  inf]])
    """

    f, s, l, i = fsli_handler(f, s, l, i)
    x, c, n, t = fsl_to_xcnt(f, s, l)

    if i.size == 0:
        return x, c, n, t
    else:
        x_i, n_i = np.unique(i, axis=0, return_counts=True)
        c_i = np.ones(x_i.shape[0]) * 2

        x_two = np.vstack([x, x]).T
        x = np.concatenate([x_two, x_i]).astype(float)
        c = np.hstack([c, c_i]).astype(int)
        n = np.hstack([n, n_i]).astype(int)
        t = np.vstack([np.ones_like(c) * -np.inf, np.ones_like(c) * np.inf]).T

        x, c, n, t = xcnt_sort(x, c, n, t)

        return x, c, n, t


def fsl_to_xcnt(
    f: "npt.ArrayLike | None" = None,
    s: "npt.ArrayLike | None" = None,
    l: "npt.ArrayLike | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Convert failure (``f``), suspension (right-censored, ``s``) and
    left-censored (``l``) times to the ``xcnt`` format, counting repeated
    times.

    Parameters
    ----------
    f : array like, optional
        Observed failure times.
    s : array like, optional
        Right-censored (suspension) times.
    l : array like, optional
        Left-censored times.

    Returns
    -------
    x, c, n, t : arrays
        The distinct times, censoring flags, counts and (untruncated)
        truncation bounds, sorted.

    Examples
    --------
    >>> from surpyval import fsl_to_xcnt
    >>> x, c, n, t = fsl_to_xcnt([4, 6], [8], [2])
    >>> x, c, n
    (array([2, 4, 6, 8]), array([-1,  0,  0,  1]), array([1, 1, 1, 1]))
    """
    if f is None:
        f = []
    if s is None:
        s = []
    if l is None:
        l = []

    # np.unique keeps NaN, so it would become a row of the output.
    named: list[tuple[str, npt.ArrayLike]] = [("f", f), ("s", s), ("l", l)]
    for name, values in named:
        if np.isnan(np.asarray(values, dtype=float)).any():
            raise ValueError(f"'{name}' cannot contain NaN values")

    x_f, n_f = np.unique(f, return_counts=True)
    c_f = np.zeros_like(x_f)

    x_s, n_s = np.unique(s, return_counts=True)
    c_s = np.ones_like(x_s)

    x_l, n_l = np.unique(l, return_counts=True)
    c_l = -np.ones_like(x_l)

    x = np.hstack([x_f, x_s, x_l])
    c = np.hstack([c_f, c_s, c_l]).astype(int)
    n = np.hstack([n_f, n_s, n_l]).astype(int)
    t = np.vstack([np.ones_like(x) * -np.inf, np.ones_like(x) * np.inf]).T

    x, c, n, t = xcnt_sort(x, c, n, t)

    return x, c, n, t


def fs_to_xcnt(
    f: "npt.ArrayLike | None" = None, s: "npt.ArrayLike | None" = None
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Convert failure (``f``) and suspension (right-censored, ``s``) times to
    the ``xcnt`` format, counting repeated times; see :func:`fsl_to_xcnt`.

    Parameters
    ----------
    f : array like, optional
        Observed failure times.
    s : array like, optional
        Right-censored (suspension) times.

    Returns
    -------
    x, c, n, t : arrays
        The distinct times, censoring flags (0 or 1), counts and
        (untruncated) truncation bounds, sorted.

    Examples
    --------
    >>> from surpyval import fs_to_xcnt
    >>> x, c, n, t = fs_to_xcnt([1, 3, 3, 7], [5, 9])
    >>> x
    array([1., 3., 5., 7., 9.])
    >>> c, n
    (array([0, 0, 1, 0, 1]), array([1, 2, 1, 1, 1]))
    """
    return fsl_to_xcnt(f, s, None)


def _get_idx(
    x_target: npt.NDArray, x: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Function to get the indices for a given vector of x values
    """
    x = np.atleast_1d(x)
    idx = np.argsort(x)
    rev = np.argsort(idx)
    x = x[idx]
    idx = np.searchsorted(x_target, x, side="right") - 1
    return idx, rev


def is_missing_event(value: Any) -> bool:
    """Whether a competing-risks event value marks *no* attributed cause -- a
    censored observation. That is Python ``None`` or any missing value
    (``NaN``, pandas ``NA``)."""
    if value is None:
        return True
    from pandas import isna

    try:
        return bool(isna(value))
    except (TypeError, ValueError):
        return False


def missing_events(values: npt.NDArray) -> npt.NDArray:
    """:func:`is_missing_event` of each element of a 1-D object array.

    A per-element loop over ``is_missing_event`` is a noticeable part of
    a competing-risks fit at 1e5 rows (#515). ``pandas.isna`` of an object
    array checks each element as ``isna`` checks a scalar, and so agrees
    with ``is_missing_event`` for every label type -- except an object
    that is array-like without being a numpy scalar, which the scalar
    ``isna`` converts to an array first. Such labels take the loop.
    """
    types = {type(v) for v in values}
    if values.ndim != 1 or any(
        hasattr(t, "__array__") and not issubclass(t, np.generic)
        for t in types
    ):
        return np.array([is_missing_event(v) for v in values], dtype=bool)
    from pandas import isna

    return np.asarray(isna(values), dtype=bool)


def resolve_cr_censoring(
    e: npt.ArrayLike, c: "npt.ArrayLike | None"
) -> tuple[npt.NDArray, npt.NDArray]:
    """Canonicalise a competing-risks event vector and its censoring flag.

    A *missing* event value (``None``, ``NaN`` or pandas ``NA``) marks a
    right-censored observation with no attributed cause; such values are
    canonicalised to ``None``. When ``c`` is not supplied it is derived from
    the events -- a missing event is censored (``c = 1``), an event present is
    observed (``c = 0``) -- so competing-risks data may be given as ``(x, e)``
    alone, without a separate censoring array.

    Returns the canonicalised event array (object dtype, ``None`` for censored)
    and the censoring flag (unchanged if supplied, otherwise derived).

    lifelines, scikit-survival and R's ``cmprsk`` code competing-risks data
    as one integer column with 0 for a censored row. Such data read here
    as a cause called 0 and no censoring at all, and every incidence is
    wrong, silently (#486). So where ``c`` is not given, no label is
    missing and the labels are numbers including 0, this warns, saying how
    to convert the data; passing ``c`` says which rows are censored and
    silences it.
    """
    if isinstance(e, (list, tuple)):
        # One element per row, whatever it is: ``np.asarray`` would split a
        # tuple cause label (``("a", 1)``) into a column of its own.
        values = list(e)
        e = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            e[i] = v
    else:
        e = np.asarray(e, dtype=object)
    missing = missing_events(e)
    e = e.copy()
    e[missing] = None
    if c is None:
        if _zero_coded(e, missing):
            warnings.warn(
                "Cause label 0 is taken as a cause, and no row is censored "
                "(a censored row has no cause: None or NaN). lifelines, "
                "scikit-survival and R's cmprsk code a censored row as 0; "
                "if 0 means censored here, pass e=np.where(np.asarray(e) "
                "== 0, None, e). If 0 is a cause and no row is censored, pass "
                "c=np.zeros(len(e)) to say so, which silences this "
                "warning.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        c = np.where(missing, 1, 0)
    return e, np.asarray(c)


def _zero_coded(e: npt.NDArray, missing: npt.NDArray) -> bool:
    """Whether the cause labels ``e`` look like the 0-for-censored coding
    of other packages (#486): no label missing, every label a number (not
    a bool), and 0 among them."""
    if missing.any() or e.size == 0:
        return False
    numbers = [
        v
        for v in e
        if isinstance(v, (Number, np.number))
        and not isinstance(v, (bool, np.bool_))
    ]
    return len(numbers) == e.size and any(v == 0 for v in numbers)


def fs_to_xrd(
    f: npt.ArrayLike, s: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Converts the fs format to the xrd format.

    Parameters
    ----------
    f: array
        array of values for which the failure/death was observed
    s: array
        array of right censored observation values

    Returns
    -------

    x: array
        sorted array of values of variable for which observations were made.
    r: array
        array of count of units/people at risk at time x (including if it had
        an event at 'x').
    d: array
        array of the count of failures/deaths at each time x.

    Examples
    --------

    >>> from surpyval import fs_to_xrd
    >>> f = [1, 4, 5]
    >>> s = [2, 3]
    >>> x, r, d = fs_to_xrd(f, s)
    >>> x
    array([1., 2., 3., 4., 5.])
    >>> r
    array([5, 4, 3, 2, 1])
    >>> d
    array([1, 0, 0, 1, 1])
    """
    x, c, n, _ = fs_to_xcnt(f, s)
    return xcnt_to_xrd(x, c, n)
