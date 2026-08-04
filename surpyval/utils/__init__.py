import warnings
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from formulaic import Formula
from pandas import Series, isna

COX_PH_METHODS = ["breslow", "efron", "exact", "kalbfleisch-prentice", "kp"]
FG_BASELINE_OPTIONS = ["Nelson-Aalen", "Kaplan-Meier"]


def _round_vals(x):
    not_different = True
    i = 1
    while not_different:
        x_ticks = np.array(round_sig(x, i))
        not_different = (np.diff(x_ticks) == 0).any()
        i += 1
    return x_ticks


def round_sig(points, sig=2):
    # Used to round to sig significant figures.
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, int(i)))
    return output


def _check_x_not_empty(func):
    # Decorator to check that x is not empty
    def wrap(obj, x, *args, **kwargs):
        x = np.array(x)
        if x.size == 0:
            return 0
        # Make sure we are using a numpy array (of 1D)
        x = np.atleast_1d(x)
        result = func(obj, x, *args, **kwargs)
        return result

    return wrap


def check_no_censoring(c):
    return any(c != 0)


def no_left_or_int(c):
    return any((c == -1) | (c == 2))


def validate_float_array(arr, name):
    """Convert input to float array with better error handling."""
    if arr is None:
        return np.array([], dtype=np.float64)
    try:
        return np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(
            f"'{name}' must be convertible to an array of float values"
        )


def _group_ids(key):
    """Group id per row, plus the row at which each group first appears.

    ``np.lexsort`` rather than ``np.unique(axis=0)``: the latter takes a
    structured-void view of the array, which is several times slower.
    Because lexsort is stable, the first row of each run in sorted order
    is also the group's earliest row in the original ordering, which is
    what the caller needs. Rows containing ``nan`` never compare equal
    and so stay in their own groups -- matching the dictionary keying
    this replaced, where a ``nan`` key never matched another.
    """
    order = np.lexsort(key.T[::-1])
    ordered = key[order]
    starts = np.empty(len(ordered), dtype=bool)
    if len(ordered) > 0:
        starts[0] = True
    if len(ordered) > 1:
        starts[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    group = np.empty(len(ordered), dtype=np.intp)
    group[order] = np.cumsum(starts) - 1
    return group, order[starts]


def group_xcnt(x, c, n, t):
    """Collapse identical ``(x, c, t)`` rows, summing their counts.

    This used to walk every observation in Python, accumulating into a
    triple-nested ``defaultdict``. That is O(N) but with a very large
    constant -- roughly 13 microseconds per observation -- which made it
    the dominant cost of fitting: 94% of a 50,000-point Normal fit, and
    two seconds at 100,000 points. It is now done by sorting and
    ``np.bincount``.

    The group *order* is preserved exactly, which is subtler than it
    looks. The nested dictionary iterated x-major: outer by first
    appearance of ``x``, then of ``(x, c)`` within it, then of the full
    key. ``xcnt_sort`` runs immediately after this and re-sorts on
    ``c``, ``t.min(axis=1)`` and ``x`` -- but it is a *stable* sort, so
    rows tying on all three of those keep whatever order arrived. Rows
    sharing an ``x`` and ``c`` with different ``tr`` but an equal
    ``t.min()`` are exactly such a tie, so a plain sorted ``np.unique``
    would silently reorder them. The lexsort below reproduces the
    original nesting instead.

    When nothing needs grouping the inputs are returned as they are,
    rather than copied. Callers inside the package sort immediately
    afterwards, which copies, so nothing aliases in practice.
    """
    # Continuous data has nothing to group: every row is already its own
    # group, and since the ordering is x-major and each x occurs once,
    # that order *is* the input order. So the answer is the input,
    # untouched. Establishing this costs one sort of a single column,
    # against three sorts of the full key -- and it is the common case,
    # since only tied (rounded, discrete, or heavily weighted) data
    # groups at all. Distinct values in the first column are enough:
    # they make whole rows distinct whatever c and t hold. Empty input
    # trivially satisfies this and is likewise handed straight back.
    leading = x if x.ndim == 1 else x[:, 0]
    if np.unique(leading).size == leading.size:
        return x, c, n, t

    x_columns = x.reshape(-1, 1) if x.ndim == 1 else x
    group, first_full = _group_ids(np.column_stack([x_columns, c, t]))
    by_x, first_x = _group_ids(x_columns)
    by_xc, first_xc = _group_ids(np.column_stack([x_columns, c]))

    # One representative row per group: the row it first appeared at.
    representative = first_full
    # np.lexsort takes its *last* key as primary, so this orders by first
    # appearance of x, then of (x, c), then of the whole key.
    order = np.lexsort(
        (
            first_full,
            first_xc[by_xc[representative]],
            first_x[by_x[representative]],
        )
    )
    representative = representative[order]

    totals = np.bincount(group, weights=n, minlength=first_full.size)[order]
    # ``bincount`` always returns float64; the counts were integers going
    # in and callers rely on that (an integer ``n`` must stay integer).
    totals = totals.astype(n.dtype)

    return x[representative], c[representative], totals, t[representative]


def xcnt_sort(x, c, n, t):
    idx_c = np.argsort(c, kind="stable")
    x = x[idx_c]
    c = c[idx_c]
    n = n[idx_c]
    t = t[idx_c]

    if t.ndim == 1:
        idx = np.argsort(t, kind="stable")
    else:
        idx = np.argsort(t.min(axis=1), kind="stable")
    x = x[idx]
    c = c[idx]
    n = n[idx]
    t = t[idx]

    if x.ndim == 1:
        idx = np.argsort(x, kind="stable")
    else:
        idx = np.argsort(x.mean(axis=1), kind="stable")

    x = x[idx]
    c = c[idx]
    n = n[idx]
    t = t[idx]

    return x, c, n, t


def fsli_handler(f=None, s=None, l=None, i=None):
    """
    Takes in the fsli format and ensures that the data is correctly defined.
    Takes an assorted combination of f, s, l, and i and returns them in the
    correct format as numpy arrays.

    Parameters
    ----------
    f: array-like, optional (default: None)
        array of values for which the failure/death was observed
    s: array-like, optional (default: None)
        array of right censored observation values
    l: array-like, optional (default: None)
        array of left censored observation values
    i: array-like, optional (default: None)
        array of length 2 arrays interval censored data

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
        raise ValueError("'i' array must be one-dimensional")

    if len(i) > 0:
        if i.shape[1] != 2:
            raise ValueError("'i' array must be of shape (?, 2)")

    if i.size != 0:
        if (i[:, 0] >= i[:, 1]).any():
            raise ValueError(
                "Lower interval must not be greater than or equal to the \
                upper interval"
            )

    return f, s, l, i


def xrd_handler(x, r, d):
    """
    Takes a combination of 'x', 'r', and 'd' arrays and ensures that the data
    is feasible.

    Does not check for the case where r is always decreasing as this is
    possible in some cases, i.e. when there is left truncation, a.k.a late
    entry.

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
        array of values of variable for which observations were made.
    r: array
        array of at risk items at each value of x
    d: array
        array of failures / deaths at each value of x

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
    """

    try:
        x = np.array(x, dtype=np.float64)
    except Exception:
        raise ValueError(
            "'x' must be an array of scalar numbers with real values."
        )

    try:
        r = np.array(r).astype(int, casting="safe")
    except Exception:
        raise ValueError("'r' must be an array of integers.")

    try:
        d = np.array(d).astype(int, casting="safe")
    except Exception:
        raise ValueError("'d' must be an array of integers.")

    if x.ndim != 1:
        raise ValueError("'x' must be a one dimensional array")

    if x.shape != r.shape:
        raise ValueError("'x' array not the same length as 'r' array")
    if x.shape != d.shape:
        raise ValueError("'x' array not the same length as 'd' array")

    if (d < 0).any():
        raise ValueError("'d' array cannot have any negative values")

    if (r <= 0).any():
        raise ValueError(
            "'r' at risk item count array cannot have any negative values"
        )

    if (d > r).any():
        raise ValueError(
            "cannot have more deaths/failures than there are items at risk"
        )

    return x, r, d


def coerce_xcnt_x(x) -> npt.NDArray:
    """
    Coerce the ``x`` variable of xcnt-format data into a numpy array.

    Accepts a 1D array of event values, or a 2D array / list-of-pairs of
    ``[left, right]`` interval bounds. Validates dimensionality, the interval
    ordering (``left <= right``) and the absence of NaNs. Shared by the
    univariate (``xcnt_handler``) and recurrent (``handle_xicn``) handlers.
    """
    if isinstance(x, list):
        if any(isinstance(v, list) for v in x):
            x_ndarray = np.empty(shape=(len(x), 2))
            for idx, val in enumerate(x):
                val_arr = np.atleast_1d(val)
                if len(val_arr) > 2:
                    raise ValueError(
                        "Each element of 'x' must be either scalar or"
                        " array-like of no more than length 2"
                    )
                x_ndarray[idx, :] = val_arr
            x = x_ndarray
        else:
            x = np.array(x)
    elif isinstance(x, Series):
        x = np.array(x)
    else:
        x = np.asarray(x)

    if x.ndim > 2:
        raise ValueError("Variable 'x' array must be one or two dimensional")
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
    if np.isnan(x).any():
        raise ValueError("Variable 'x' cannot contain NaN values")
    return x


def format_truncation(t, tl, tr, n_rows) -> npt.NDArray:
    """
    Build the ``(n_rows, 2)`` truncation array from either a ``t`` matrix or
    separate ``tl``/``tr`` bounds (scalars broadcast to all rows). The default
    window is the whole real line ``[-inf, inf]``. Shared by ``xcnt_handler``
    and ``handle_xicn``.
    """
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
        if tl is None:
            tl_arr = np.ones(n_rows) * -np.inf
        elif np.isscalar(tl):
            tl_arr = np.ones(n_rows) * float(tl)  # type: ignore[arg-type]
        else:
            tl_arr = np.array(tl, dtype=float)

        if tr is None:
            tr_arr = np.ones(n_rows) * np.inf
        elif np.isscalar(tr):
            tr_arr = np.ones(n_rows) * float(tr)  # type: ignore[arg-type]
        else:
            tr_arr = np.array(tr, dtype=float)

        if tl_arr.ndim > 1 or tr_arr.ndim > 1:
            raise ValueError(
                "Truncation arrays must be one dimensional, did you mean to"
                " use 't'"
            )
        if tl_arr.shape[0] != n_rows or tr_arr.shape[0] != n_rows:
            raise ValueError(
                "Truncation array must be same length as variable array"
            )
        return np.vstack([tl_arr, tr_arr]).T

    t = np.array(t, dtype=float)
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
    return t


def xcnt_handler(
    x=None,
    c=None,
    n=None,
    t=None,
    xl=None,
    xr=None,
    tl=None,
    tr=None,
    group_and_sort=True,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Main handler that ensures any input to a surpyval fitter meets the
    requirements to be used in one of the parametric or nonparametric fitters.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array, optional (default: None)
        array of censoring values (-1, 0, 1, 2) corrseponding to x
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
        array of censoring values (-1, 0, 1, 2) corrseponding to output array
        x. If c was None, defaults to creating array of zeros the length of x.
    n: array
        array of count of observations at output array x and with censoring c.
        If n was None, count array assumed to be all one observation.
    t: array
        array of truncation values of observations at output array x and with
        censoring c.

    Examples
    --------

    >>> from surpyval import xcnt_handler
    >>> x = [1, 2, 3, 4, 5]
    >>> c = [0, 0, 1, 1, 1]
    >>> n = [1, 1, 1, 1, 1]
    >>> t = [[0, 5], [0, 5], [0, 5], [0, 5], [0, 5]]
    >>> xcnt_handler(x, c, n, t)
    (array([1., 2., 3., 4., 5.]),
    array([0, 0, 1, 1, 1]),
    array([1, 1, 1, 1, 1]),
    array([[0., 5.],
            [0., 5.],
            [0., 5.],
            [0., 5.],
            [0., 5.]]))
    >>> xcnt_handler(x, c, n, tl=0, tr=5)
    (array([1., 2., 3., 4., 5.]),
    array([0, 0, 1, 1, 1]),
    array([1, 1, 1, 1, 1]),
    array([[0., 5.],
            [0., 5.],
            [0., 5.],
            [0., 5.],
            [0., 5.]]))
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

    # logic for 'variables'
    if (x is None) and (xl is None) and (xr is None):
        raise ValueError(
            "Must enter some data! Use either 'x' or both 'xl and 'xr'"
        )

    if (x is not None) and ((xl is not None) or (xr is not None)):
        raise ValueError("Must use either 'x' or both 'xl and 'xr'")

    if (x is None) and ((xl is None) or (xr is None)):
        raise ValueError("Must use either 'x' or both 'xl and 'xr'")

    if x is None:
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

    # logic for censoring flag
    if c is not None:
        c = np.array(c)
        if c.ndim != 1:
            raise ValueError("Censoring flag array must be one dimensional")

        if c.shape[0] != x.shape[0]:
            raise ValueError(
                "censoring flag array must be same length as variable array"
            )

        if x.ndim == 2:
            if any(c[x[:, 0] == x[:, 1]] == 2):
                raise ValueError(
                    "Censor flag indicates interval censored but only has one"
                    + " failure time"
                )

            if any(c[x[:, 0] != x[:, 1]] != 2):
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

            if any((c == 2) & (x[:, 0] == x[:, 1])):
                raise ValueError(
                    "Censor flag provided, but case where interval flagged as"
                    + " non interval censoring"
                )

            if any((c != 0) & (c != 1) & (c != -1) & (c != 2)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1, or 2"
                )

        else:
            if any((c != 0) & (c != 1) & (c != -1)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1 for single"
                    + " dimension input"
                )

    else:
        c = np.zeros(x.shape[0])
        if x.ndim != 1:
            c[x[:, 0] != x[:, 1]] = 2

    if n is not None:
        n = np.array(n)
        if n.ndim != 1:
            raise ValueError("Count array must be one dimensional")
        if n.shape[0] != x.shape[0]:
            raise ValueError(
                "count array must be same length as variable array."
            )
        if not np.equal(n, np.floor(n)).all():
            raise ValueError("Count array 'n' must contain integer values")
        if not (n > 0).all():
            raise ValueError("count array can't be 0 or less")
    else:
        # Do check here for groupby and binning
        n = np.ones(x.shape[0])

    t = format_truncation(t, tl, tr, x.shape[0])

    if (t[:, 1] <= t[:, 0]).any():
        raise ValueError(
            "All left truncated values must be less than right truncated"
            + " values"
        )
    if x.ndim == 2:
        if ((t[:, 0] > x[:, 0]) & (np.isfinite(t[:, 0]))).any():
            raise ValueError(
                "All left truncated values must be less than the respective"
                + " observed values"
            )
        elif ((t[:, 1] < x[:, 1]) & (np.isfinite(t[:, 1]))).any():
            raise ValueError(
                "All right truncated values must be greater than the"
                + " respective observed values"
            )
    else:
        # Strictly less: under the (entry, exit] risk-interval convention a
        # value at exactly its own left-truncation time has a zero-length
        # observation window — contradictory data that previously slipped
        # through and silently distorted the Turnbull estimate (#260).
        if ((t[:, 0] >= x) & np.isfinite(t[:, 0])).any():
            raise ValueError(
                "All left truncated values must be strictly less than the"
                + " respective observed values: a value at its own left"
                + " truncation time has a zero-length observation window."
            )
        elif (t[:, 1] < x).any():
            raise ValueError(
                "All right truncated values must be greater than the"
                + " respective observed values"
            )

    if x.ndim == 2:
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

    x = x.astype(float)
    c = c.astype(int)
    n = n.astype(int)
    t = t.astype(float)

    # Right censoring with a finite right truncation used to warn here
    # as contradictory data (#195). It is not: the unit was detected, so
    # its event is at or before ``tr``, and it was censored, so the
    # event is after ``x``. Together that is simply ``x < X <= tr`` --
    # the ordinary setup of a flux-limited or reporting-delay sample,
    # e.g. a detector that registers an event but cannot resolve where
    # in the remaining window it fell.
    #
    # The unbounded likelihoods that motivated the warning were not
    # caused by the data. Right-censored rows were being given the
    # unconditional ``S(x)``, which includes the ``X > tr`` region the
    # truncation excludes, and that is what had no maximum (#310). With
    # the conditional ``F(tr) - F(x)`` the fit is well posed and
    # consistent, so the warning has been withdrawn.

    if group_and_sort:
        x, c, n, t = group_xcnt(x, c, n, t)
        x, c, n, t = xcnt_sort(x, c, n, t)

    return x, c, n, t


def xcn_to_fs(x, c=None, n=None):
    x = np.array(x)
    if c is None:
        c = np.zeros_like(x)
    if n is None:
        n = np.ones_like(x).astype(int)

    c = np.array(c)
    n = np.array(n).astype(int)

    f = np.repeat(x[c == 0], n[c == 0])
    s = np.repeat(x[c == 1], n[c == 1])
    return f, s


def _entered_before(tl, x, n):
    """Weighted count of observations that entered strictly before each ``x``.

    ``e[j] = sum_i n_i * 1[tl_i < x_j]`` -- the ``(entry, exit]``
    risk-interval convention (#260): a subject entering exactly at an event
    time is not at risk for that event.

    Written directly this is
    ``((tl[:, None] < x[None, :]) * n[:, None]).sum(0)``, which materialises
    an ``N x K`` matrix and so costs quadratic time *and* memory: at 20,000
    observations that is a 3.2 GB intermediate taking ~15 s, and past ~50,000
    it raised ``MemoryError`` outright. Two cheap branches give identical
    counts:

    * **nothing is left truncated** -- the default and by far the common
      case. Every observation has entered before every event time, so the
      whole matrix is ``True`` and ``e`` collapses to the constant
      ``n.sum()``. The matrix was being built to recompute a scalar.
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


def xcnt_to_xrd(x, c=None, n=None, t=None, **kwargs):
    """
    Converts the xcn format to the xrd format.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array, optional (default: None)
        array of censoring values (-1, 0, 1, 2) corrseponding to x. If None, an
        array of 0s is created corresponding to each x.
    n: array, optional (default: None)
        array of count of observations at each x and with censoring c. If None,
        an array of ones is created.
    kwargs: keywords for truncation can be either 't' or a combo of 'tl' and
    'tr'

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    r: array
        array of count of units/people at risk at time x (including if it had
        an event at 'x').
    d: array
        array of the count of failures/deaths at each time x.

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
    x, c, n, t = xcnt_handler(x, c, n, t, **kwargs)

    if np.isfinite(t[:, 1]).any():
        raise ValueError("xrd format can't be used right truncated data")

    # No warning for a shared entry time: validation guarantees tl < x for
    # every observation, so a common entry time never alters the (entry,
    # exit] risk sets -- the old "Ignoring left truncated values" warning
    # fired on perfectly ordinary inputs like tl=0 everywhere, e.g. from
    # check_ph on a model fit with a constant entry column (#282).

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


def xrd_to_xcnt(x, r, d):
    """
    Converts the xrd format to the xcn format. Assumes that there is no
    right truncation or left censoring.

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
        array of censoring values (-1, 0, 1, 2) corrseponding to x
    n: array
        array of count of observations at each x and with censoring c
    t: array
        array of values with shape (?, 2) with the left and right value of
        truncation

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
    n_f = np.copy(d)
    x_f = np.copy(x)
    mask = n_f != 0
    n_f = n_f[mask]
    x_f = x_f[mask]

    # A risk set that grows (after accounting for that step's events) means
    # late entry / left truncation, which the xcnt output cannot represent;
    # the old np.abs() silently converted such data into a different study
    # instead of raising (#281).
    r_arr = np.asarray(r)
    d_arr = np.asarray(d)
    if (np.diff(r_arr) + d_arr[:-1] > 0).any():
        raise ValueError(
            "The risk set increases between observation times (late entry /"
            " left truncation); truncation cannot be represented in xcnt"
            " output, so this data cannot be converted with xrd_to_xcnt."
        )

    delta = np.abs(np.diff(np.hstack([r, [0]])))

    sus = delta - d
    x_s = x[sus > 0]
    n_s = sus[sus > 0]

    x_f = np.repeat(x_f, n_f)
    x_s = np.repeat(x_s, n_s)

    return fs_to_xcnt(x_f, x_s)


def fsli_to_xcnt(f=None, s=None, l=None, i=None):
    """
    Converts the fsli format to the xcn format. This ensures is so that the
    data can be passed to one of the parametric or nonparametric fitters.

    Parameters
    ----------
    f: array
        array of values for which the failure/death was observed
    s: array
        array of right censored observation values
    l: array
        array of left censored observation values
    i: array
        array of length 2 arrays interval censored data

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corrseponding to output array
        x.
    n: array
        array of count of observations at to output array x and with censoring
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


def fsl_to_xcnt(f=None, s=None, l=None):
    if f is None:
        f = []
    if s is None:
        s = []
    if l is None:
        l = []

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


def fs_to_xcnt(f=None, s=None):
    return fsl_to_xcnt(f, s, None)


def _get_idx(x_target, x):
    """
    Function to get the indices for a given vector of x values
    """
    x = np.atleast_1d(x)
    idx = np.argsort(x)
    rev = np.argsort(idx)
    x = x[idx]
    idx = np.searchsorted(x_target, x, side="right") - 1
    return idx, rev


def check_left_or_int_cens(c):
    if (-1 in c) or (2 in c):
        raise ValueError(
            "Left or interval censoring not implemented with Competing Risks"
        )


def check_Z_and_x(Z, x):
    if x.shape[0] != Z.shape[0]:
        raise ValueError("Z must have len(x) number of rows")


def check_e_and_x(e, x):
    if e.shape != x.shape:
        raise ValueError(
            "Event vector, e, and duration vector, x, must have same shape"
        )


def check_c_and_e(c, e):
    if any(e_i is not None for e_i in e[c == 1]) or any(
        e_i is None for e_i in e[c != 1]
    ):
        raise ValueError(
            "A missing event type (None / NaN) is allowed only for a "
            "censored observation (c = 1), and every censored observation "
            "must have one."
        )


def is_missing_event(value):
    """Whether a competing-risks event value marks *no* attributed cause -- a
    censored observation. That is Python ``None`` or any missing value
    (``NaN``, pandas ``NA``)."""
    if value is None:
        return True
    try:
        return bool(isna(value))
    except (TypeError, ValueError):
        return False


def resolve_cr_censoring(e, c):
    """Canonicalise a competing-risks event vector and its censoring flag.

    A *missing* event value (``None``, ``NaN`` or pandas ``NA``) marks a
    right-censored observation with no attributed cause; such values are
    canonicalised to ``None``. When ``c`` is not supplied it is derived from
    the events -- a missing event is censored (``c = 1``), an event present is
    observed (``c = 0``) -- so competing-risks data may be given as ``(x, e)``
    alone, without a separate censoring array.

    Returns the canonicalised event array (object dtype, ``None`` for censored)
    and the censoring flag (unchanged if supplied, otherwise derived).
    """
    e = np.asarray(e, dtype=object)
    missing = np.array([is_missing_event(v) for v in e], dtype=bool)
    e = np.array([None if m else v for v, m in zip(e, missing)], dtype=object)
    if c is None:
        c = np.where(missing, 1, 0)
    return e, c


def wrangle_and_check_form_and_Z_cols(Z_cols, formula, df):
    if (Z_cols is None) and (formula is None):
        raise ValueError("'Z_cols' or 'formula' cannot both be None")

    if (Z_cols is not None) and (formula is not None):
        raise ValueError(
            "Either 'Z_cols' or 'formula' must be provided; not both"
        )

    if Z_cols is not None:
        if isinstance(Z_cols, str):
            Z_cols = [Z_cols]
        unknown = [x for x in Z_cols if x not in df.columns]
        if len(unknown) > 0:
            raise ValueError("{} not in dataframe columns".format(unknown))
        Z = df[Z_cols].values.astype(float)
        mask = ~df[Z_cols].isna().any(axis=1).values
        Z = Z[mask]
        form = None
        feature_names = list(Z_cols)
        model_spec = None
    else:
        # Materialise with the implicit intercept so categoricals get
        # reference-level coding, then drop the intercept column — the
        # baseline hazard plays that role, and a full one-hot is collinear
        # with it (#252). An explicit "0 + ..." formula opts out.
        form = Formula(formula)
        model_matrix = form.get_model_matrix(df, na_action="ignore")
        model_spec = model_matrix.model_spec
        if "Intercept" in model_matrix.columns:
            model_matrix = model_matrix.drop(columns=["Intercept"])
        feature_names = list(model_matrix.columns)
        Z = model_matrix.values.astype(float)
        mask = ~np.any(np.isnan(Z), axis=1)

    return Z, mask, form, feature_names, model_spec


def wrangle_Z(Z):
    Z = np.array(Z)

    if Z.ndim == 1:
        Z = np.atleast_2d(Z).T
    elif Z.ndim == 2:
        pass
    else:
        raise ValueError("Covariate matrix must be two dimensional")

    mask = ~np.any(np.isnan(Z), axis=1)

    return Z[mask], mask


def validate_cr_df_inputs(df, x_col, e_col, c_col=None, n_col=None):
    x = df[x_col].values
    e = df[e_col].values

    if c_col:
        c = df[c_col].values
    else:
        c = None

    if n_col:
        n = df[n_col].values
    else:
        n = None
    return x, c, n, e


def validate_cr_inputs(x, c, n, e, method):
    # Validates the inputs prior to be used by the CoxPH model.
    # Use existing surpyval validator. But don't group and sort
    # so as to put it out of order of the event array, e.
    # A missing event (None / NaN) marks a censored observation; if c is not
    # given it is derived from the events.
    e, c = resolve_cr_censoring(e, c)
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    e = np.array(e)
    x, c, n = (np.array(a).astype(float) for a in [x, c, n])

    # Check same shape
    check_e_and_x(e, x)

    # Not implemented, yet.
    check_left_or_int_cens(c)

    # Ensure all cases where c is 0, e is not None and
    # where c is 1 e is None
    check_c_and_e(c, e)

    # Two baselines
    # TODO: Add fleming-harrington
    if method not in FG_BASELINE_OPTIONS:
        raise ValueError("Unrecognised baseline method")

    return x, c, n, e


def validate_event(mapping, event):
    if event is not None and event not in mapping:
        raise ValueError("Event type not in model")


def validate_cif_event(event):
    if event is None:
        raise ValueError("CIF needs event type, not None")


def _check_an_ids_tl_and_x(id, tl, x):
    # This function checks that, for a given item, id, the history
    # of the item is complete. That is, the timeline provided in the
    # steps from tl[0] > x[0] == tl[1] > x[1] == ... tl[-1] > x[-1]
    # This ensures there are no overlaps nor are there any gaps.
    if len(tl) != len(x):
        raise ValueError(
            "Item {id} has differing lengths of the tl and x vectors".format(
                id=id
            )
        )
    n = len(x)
    if n != len(np.unique(tl)):
        raise ValueError("Item {id} has repeated tl values".format(id=id))

    # Check finish times are unqiue
    if n != len(np.unique(x)):
        raise ValueError("Item {id} has repeated x values".format(id=id))
    # Check all starts are less than stops
    if np.any(tl >= x):
        raise ValueError("tl has some values greater than or equal to x")

    # check that there is one start time and one finish time and
    # no other double ups or gaps
    x, n = np.unique([tl, x], return_counts=True)

    if n[0] != 1:
        raise ValueError("Multiple start times for item {id}".format(id=id))
    if n[-1] != 1:
        raise ValueError("Multiple end times for item {id}".format(id=id))

    if np.any(n[1:-1] != 2):
        raise ValueError(
            "Missing or doubled-up time windows for item {id}".format(id=id)
        )


def validate_tv_coxph(id, tl, x, Z, c, n):
    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)

    if id is None:
        warnings.warn("No id provided, model fitted by coherence not checked")
    else:
        id = np.array(id)
        for i in id:
            tl_i = tl[id == i]
            x_i = x[id == i]
            _check_an_ids_tl_and_x(i, tl_i, x_i)

    # One validation pass is enough: mask the already-validated arrays
    # (including the truncation bounds — the old second xcnt_handler call
    # used the unmasked tl and would raise a confusing length error
    # whenever wrangle_Z actually dropped NaN rows).
    Z, mask = wrangle_Z(Z)
    x, c, n, t = (arr[mask] for arr in (x, c, n, t))
    x, c, n, Z = (arr.astype(float) for arr in [x, c, n, Z])

    check_Z_and_x(Z, x)

    return t[:, 0], x, Z, c, n


def validate_tv_coxph_df_inputs(
    df, id_col, tl_col, x_col, Z_cols, c_col, n_col, formula
):
    # TODO: Create count of dropped rows

    if x_col is None:
        raise ValueError("'x_col' not provided")
    elif x_col not in df.columns:
        raise ValueError("'{}' not in dataframe's columns".format(x_col))

    if tl_col is None:
        raise ValueError("'tl_col' not given")
    elif tl_col not in df.columns:
        raise ValueError("'{}' not in dataframe's columns".format(tl_col))

    if id_col is None:
        warnings.warn(
            "'id_cols' is None, fit carried out without checking coherence of \
                each id's timeline."
        )
        id = None
    elif id_col not in df.columns:
        raise ValueError("'{}' not in dataframe's columns".format(id_col))
    else:
        # Check to ensure each item (by id) has coherent data.
        # That is, make sure that there are no gaps or overlaps
        # in the timline of each item.
        id = df[id_col].values
        for i, s in df.groupby(id_col):
            tl = s[tl_col].values
            x = s[x_col].values
            _check_an_ids_tl_and_x(i, tl, x)

    Z, mask, form, _, _ = wrangle_and_check_form_and_Z_cols(
        Z_cols, formula, df
    )

    x = df.loc[mask, x_col].values
    tl = df.loc[mask, tl_col].values

    if c_col is None:
        c = None
    else:
        c = df.loc[mask, c_col].values

    if n_col is None:
        n = None
    else:
        n = df.loc[mask, n_col].values

    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)

    return t[:, 0], x, Z, id, c, n, form


def validate_coxph_df_inputs(df, x_col, c_col, n_col, Z_cols, formula):
    # TODO: Return the count of dropped rows?

    Z, mask, form, feature_names, model_spec = (
        wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)
    )

    x = df.loc[mask, x_col].values

    if c_col is None:
        c = None
    else:
        c = df.loc[mask, c_col].values

    if n_col is None:
        n = None
    else:
        n = df.loc[mask, n_col].values

    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    return x, c, n, Z, form, feature_names, model_spec


def validate_coxph(
    x, c, n, Z, tl, method
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    if method not in COX_PH_METHODS:
        raise ValueError("Method must be in {}".format(COX_PH_METHODS))

    # The Cox partial likelihood accommodates left-truncation (delayed entry)
    # by adjusting the risk sets, but has no way to incorporate right or
    # interval truncation (that needs a reverse-time / retro-hazard model). A
    # 2-D ``tl`` is the natural way a user would try to pass a [tl, tr] pair,
    # so reject it with a clear, Cox-specific message rather than the generic
    # truncation error (which points at a ``t`` argument CoxPH does not have).
    if tl is not None and np.ndim(tl) == 2:
        raise ValueError(
            "CoxPH supports left-truncation (delayed entry) only, supplied "
            "as a one-dimensional `tl`. Right or interval truncation is not "
            "available for the Cox partial likelihood; use a parametric "
            "proportional-hazards fitter (e.g. WeibullPH) with t=[tl, tr] "
            "for right/interval-truncated data."
        )

    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)

    tl = t[:, 0]

    x, c, n, Z, tl = (np.array(a).astype(float) for a in [x, c, n, Z, tl])

    Z, mask = wrangle_Z(Z)
    x, c, n, tl = (arr[mask] for arr in (x, c, n, tl))
    x, c, n, tl, Z = (arr.astype(float) for arr in [x, c, n, tl, Z])

    check_Z_and_x(Z, x)

    return x, c, n, tl, Z


def validate_fine_gray_inputs(x, Z, e, c, n):
    # A missing event (None / NaN) marks a censored observation; if c is not
    # given it is derived from the events.
    e, c = resolve_cr_censoring(e, c)
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    e = np.array(e)
    Z, mask = wrangle_Z(Z)
    x, c, n, e = (arr[mask] for arr in (x, c, n, e))

    # Set all dtypes to float. Very poor results otherwise.
    x, c, n, Z = (arr.astype(float) for arr in [x, c, n, Z])

    check_e_and_x(e, x)
    check_Z_and_x(Z, x)
    check_c_and_e(c, e)
    check_left_or_int_cens(c)

    return x, Z, e, c, n


def fs_to_xrd(f, s):
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
