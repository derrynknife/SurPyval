import sys
import warnings
from collections import defaultdict

import numpy as np
from formulaic import Formula
from pandas import Series

COX_PH_METHODS = ["breslow", "efron"]
FG_BASELINE_OPTIONS = ["Nelson-Aalen", "Kaplan-Meier"]


def init_from_bounds(dist):
    out = []
    for low, high in dist.bounds:
        if (low is None) & (high is None):
            out.append(0)
        elif high is None:
            out.append(low + 1.0)
        elif low is None:
            out.append(high - 1.0)
        else:
            out.append((high + low) / 2.0)

    return out


def check_no_censoring(c):
    return any(c != 0)


def no_left_or_int(c):
    return any((c == -1) | (c == 2))


def surv_tolist(x):
    """
    supplement to the .tolist() function where there is mixed scalars and
    arrays.

    Survival data can be confusing sometimes. I'm probably not helping by
    trying
    to make it fit with one input.
    """
    if x.ndim == 2:
        return [v[0] if v[0] == v[1] else v.tolist() for v in x]
    else:
        return x.tolist()


def check_and_convert_expected_float_array(a, name):
    if a is None:
        a = np.array([])
    else:
        try:
            a = np.array(a).astype(float, casting="safe")
        except Exception:
            raise ValueError(
                f"'{name}' must be an array of scalar numbers \
                with real values."
            )
    return a


def group_xcn(x, c, n):
    """
    Ensures that all unique combinations of x and c are grouped. This should
    reduce the chance of having arrays that are too long for the MLE fit.
    """

    # TODO: Change to numpy index
    grouped = defaultdict(lambda: defaultdict(int))
    if x.ndim == 2:
        for vx, vc, vn in zip(x, c, n):
            grouped[tuple(vx)][vc] += vn
    else:
        for vx, vc, vn in zip(x, c, n):
            grouped[vx][vc] += vn

    x_out = []
    c_out = []
    n_out = []

    for k, v in grouped.items():
        for cv, nv in v.items():
            x_out.append(k)
            c_out.append(cv)
            n_out.append(nv)
    return np.array(x_out), np.array(c_out), np.array(n_out)


def group_xcnt(x, c, n, t):
    """
    Ensures that all unique combinations of x and c are grouped. This should
    reduce the chance of having arrays that are too long for the MLE fit.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if x.ndim == 2:
        for vx, vc, vn, vt in zip(x, c, n, t):
            grouped[tuple(vx)][vc][tuple(vt)] += vn
    else:
        for vx, vc, vn, vt in zip(x, c, n, t):
            grouped[vx][vc][tuple(vt)] += vn

    x_out = []
    c_out = []
    n_out = []
    t_out = []

    for xv, level2 in grouped.items():
        for cv, level3 in level2.items():
            for tv, nv in level3.items():
                x_out.append(xv)
                c_out.append(cv)
                n_out.append(nv)
                t_out.append(tv)
    return np.array(x_out), np.array(c_out), np.array(n_out), np.array(t_out)


def xcn_sort(x, c, n):
    """
    Sorts the x, c, and n arrays so that x is increasing, and where there are
    ties they are ordered as left censored, failure, right censored, then
    intervally censored.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corrseponding to x
    n: array
        array of count of observations at each x and with censoring c

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    c: array
        sorted array of censoring values (-1, 0, 1, 2) corrseponding to
        rearranged x
    n: array
        array of count of observations at each rearranged x and with censoring
        c
    """

    idx_c = np.argsort(c, kind="stable")
    x = x[idx_c]
    c = c[idx_c]
    n = n[idx_c]

    if x.ndim == 1:
        idx = np.argsort(x, kind="stable")
    else:
        idx = np.argsort(x[:, 0], kind="stable")

    x = x[idx]
    c = c[idx]
    n = n[idx]
    return x, c, n


def xcn_handler(x, c=None, n=None):
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

    Returns
    ----------

    x: array
        sorted array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corrseponding to output array
        x. If c was None, defaults to creating array of zeros the length of x.
    n: array
        array of count of observations at to output array x and with censoring
        c. If n was None, count array assumed to be all one observation.
    """
    if type(x) == list:
        if any([list == type(v) for v in x]):
            x_ndarray = np.empty(shape=(len(x), 2))
            for idx, val in enumerate(x):
                x_ndarray[idx, :] = np.array(val)
            x = x_ndarray
        else:
            x = np.array(x).astype(float)
    elif type(x) == Series:
        x = np.array(x)

    if x.ndim > 2:
        raise ValueError("Variable (x) array must be one or two dimensional")

    if x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError(
                "Dimension 1 must be equalt to 2, try transposing data, or do \
                you have a 1d array in a 2d array?"
            )

        if not (x[:, 0] <= x[:, 1]).all():
            raise ValueError(
                "All left intervals must be less than right intervals"
            )

    if c is not None:
        c = np.array(c)
        if c.ndim != 1:
            raise ValueError("censoring array must be one dimensional")

        if c.shape[0] != x.shape[0]:
            raise ValueError(
                "censoring array must be same length as variable array"
            )

        if x.ndim == 2:
            if any(c[x[:, 0] == x[:, 1]] == 2):
                raise ValueError(
                    "Censor flag indicates interval censored but only has one \
                    failure time"
                )

            if not all(c[x[:, 0] != x[:, 1]] == 2):
                raise ValueError(
                    "Censor flag provided, but case where interval flagged as \
                    non interval censoring"
                )

            if any((c != 0) & (c != 1) & (c != -1) & (c != 2)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1, or 2"
                )

        else:
            if any((c != 0) & (c != 1) & (c != -1)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1 for single \
                    dimension input"
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
        if not (n > 0).all():
            raise ValueError("count array can't be 0")
    else:
        # Do check here for groupby and binning
        n = np.ones(x.shape[0])

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

    n = n.astype(int)
    c = c.astype(int)

    x, c, n = group_xcn(x, c, n)
    x, c, n = xcn_sort(x, c, n)

    return x, c, n


def xcnt_sort(x, c, n, t):
    """
    Sorts the x, c, and n arrays so that x is increasing, and where there are
    ties they are ordered as left censored, failure, right censored, then
    intervally censored.

    Parameters
    ----------
    x: array
        array of values of variable for which observations were made.
    c: array
        array of censoring values (-1, 0, 1, 2) corrseponding to x
    n: array
        array of count of observations at each x and with censoring c
    t: array, optional (default: None)
        array of values with shape (?, 2) with the left and right value of
        truncation

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    c: array
        sorted array of censoring values (-1, 0, 1, 2) corrseponding to
        rearranged x
    n: array
        array of count of observations at each rearranged x and with censoring
        c
    t: array, optional (default: None)
        array of values with shape (?, 2) with the left and right value of
        truncation
    """

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
        array of interval censored data

    Returns
    ----------
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
    """
    f = check_and_convert_expected_float_array(f, "f")
    s = check_and_convert_expected_float_array(s, "s")
    l = check_and_convert_expected_float_array(l, "l")
    i = check_and_convert_expected_float_array(i, "i")

    if f.ndim != 1:
        raise ValueError("'f' array must be one-dimensional")
    if s.ndim != 1:
        raise ValueError("'s' array must be one-dimensional")
    if l.ndim != 1:
        raise ValueError("'l' array must be one-dimensional")

    if (i.ndim != 2) & (i.size != 0):
        raise ValueError("'i' array must be one-dimensional")

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
    """

    try:
        x = np.array(x).astype(float)
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
):
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
    """

    if (x is None) & (xl is None) & (xr is None):
        raise ValueError(
            "Must enter some data! Use either 'x' of both 'xl and 'xr'"
        )

    if (x is not None) & ((xl is not None) | (xr is not None)):
        raise ValueError("Must use either 'x' of both 'xl and 'xr'")

    if (x is None) & ((xl is None) | (xr is None)):
        raise ValueError("Must use either 'x' of both 'xl and 'xr'")

    if x is None:
        xl = np.array(xl).astype(float)
        xr = np.array(xr).astype(float)
        x = np.vstack([xl, xr]).T

    if type(x) == list:
        if any([type(v) == list for v in x]):
            x_ndarray = np.empty(shape=(len(x), 2))
            for idx, val in enumerate(x):
                x_ndarray[idx, :] = np.array(val)
            x = x_ndarray
        else:
            x = np.array(x)
    elif type(x) == Series:
        x = np.array(x)

    if x.ndim > 2:
        raise ValueError("Variable (x) array must be one or two dimensional")

    if x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError(
                "Dimension 1 must be equalt to 2, try transposing data, or do \
                you have a 1d array in a 2d array?"
            )

        if not (x[:, 0] <= x[:, 1]).all():
            raise ValueError(
                "All left intervals must be less than (or equal to) right \
                intervals"
            )

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
                    "Censor flag indicates interval censored but only has one \
                        failure time"
                )

            if any((c == 2) & (x[:, 0] == x[:, 1])):
                raise ValueError(
                    "Censor flag provided, but case where interval flagged as \
                        non interval censoring"
                )

            if any((c != 0) & (c != 1) & (c != -1) & (c != 2)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1, or 2"
                )

        else:
            if any((c != 0) & (c != 1) & (c != -1)):
                raise ValueError(
                    "Censoring value must only be one of -1, 0, 1 for single \
                        dimension input"
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
        if not (n > 0).all():
            raise ValueError("count array can't be 0 or less")
    else:
        # Do check here for groupby and binning
        n = np.ones(x.shape[0])

    if t is not None and ((tl is not None) or (tr is not None)):
        raise ValueError(
            "Cannot use 't' with 'tl' or 'tr'. Use either 't' or any \
                combination of 'tl' and 'tr'"
        )

    elif (t is None) & (tl is None) & (tr is None):
        tl = np.ones(x.shape[0]) * -np.inf
        tr = np.ones(x.shape[0]) * np.inf
        t = np.vstack([tl, tr]).T
    elif (tl is not None) or (tr is not None):
        if tl is None:
            tl = np.ones(x.shape[0]) * -np.inf
        elif np.isscalar(tl):
            tl = np.ones(x.shape[0]) * tl
        else:
            tl = np.array(tl)

        if tr is None:
            tr = np.ones(x.shape[0]) * np.inf
        elif np.isscalar(tr):
            tr = np.ones(x.shape[0]) * tr
        else:
            tr = np.array(tr)

        if tl.ndim > 1:
            raise ValueError(
                "Left truncation array must be one dimensional, did you mean \
                    to use 't'"
            )
        if tr.ndim > 1:
            raise ValueError(
                "Left truncation array must be one dimensional, did you mean \
                    to use 't'"
            )
        if tl.shape[0] != x.shape[0]:
            raise ValueError(
                "Left truncation array must be same length as variable array"
            )
        if tr.shape[0] != x.shape[0]:
            raise ValueError(
                "Right truncation array must be same length as variable array"
            )
        if tl.shape != tr.shape:
            raise ValueError(
                "Left truncation array and right truncation array must be the \
                    same length"
            )
        t = np.vstack([tl, tr]).T

    else:
        t = np.array(t)
        if t.ndim != 2:
            raise ValueError("Truncation ndarray must be 2 dimensional")
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                "Truncation ndarray must be same shape as variable array"
            )

    if (t[:, 1] <= t[:, 0]).any():
        raise ValueError(
            "All left truncated values must be less than right truncated \
                values"
        )
    if x.ndim == 2:
        if ((t[:, 0] > x[:, 0]) & (np.isfinite(t[:, 0]))).any():
            raise ValueError(
                "All left truncated values must be less than the respective \
                    observed values"
            )
        elif ((t[:, 1] < x[:, 1]) & (np.isfinite(t[:, 1]))).any():
            raise ValueError(
                "All right truncated values must be greater than the \
                    respective observed values"
            )
    else:
        if (t[:, 0] > x).any():
            raise ValueError(
                "All left truncated values must be less than the respective \
                    observed values"
            )
        elif (t[:, 1] < x).any():
            raise ValueError(
                "All right truncated values must be greater than the \
                    respective observed values"
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

    if group_and_sort:
        x, c, n, t = group_xcnt(x, c, n, t)
        x, c, n, t = xcnt_sort(x, c, n, t)

    return x, c, n, t


def xcn_to_fsl(x, c=None, n=None):
    x = np.array(x)
    if c is None:
        c = np.zeros_like(x)
    if n is None:
        n = np.ones_like(x).astype(int)

    c = np.array(c)
    n = np.array(n).astype(int)

    f = np.repeat(x[c == 0], n[c == 0])
    s = np.repeat(x[c == 1], n[c == 1])
    l = np.repeat(x[c == -1], n[c == -1])
    return f, s, l


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
        array of count of units/people at risk at time x.
    d: array
        array of the count of failures/deaths at each time x.
    """
    x, c, n, t = xcnt_handler(x, c, n, t, **kwargs)

    if np.isfinite(t[:, 1]).any():
        raise ValueError("xrd format can't be used right truncated data")

    if (t[:, 0] == t[0, 0]).all() & np.isfinite(t[0, 0]):
        print(
            "Ignoring left truncated values as all observations truncated at \
            same value",
            file=sys.stderr,
        )

    if ((c != 1) & (c != 0)).any():
        raise ValueError(
            "xrd format can't be used with left (c=-1) or interval (c=2) \
            censoring"
        )

    x = np.repeat(x, n)
    c = np.repeat(c, n)
    t = np.repeat(t[:, 0], n)
    n = np.ones_like(x).astype(int)

    x, idx = np.unique(x, return_inverse=True)

    d = np.bincount(idx, weights=1 - c)
    le = (t.reshape(-1, 1) <= x).sum(axis=0)
    # do is drop outs
    do = np.bincount(idx, weights=c)
    r = le + d - d.cumsum() + do - do.cumsum()
    r = r.astype(int)
    d = d.astype(int)
    return x, r, d


def xcn_to_xrd(x, c=None, n=None):
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

    Returns
    ----------
    x: array
        sorted array of values of variable for which observations were made.
    r: array
        array of count of units/people at risk at time x.
    d: array
        array of the count of failures/deaths at each time x.
    """
    x, c, n = xcn_handler(x, c, n)

    if ((c != 1) & (c != 0)).any():
        raise ValueError(
            "xrd format can't handle left (c=-1) or interval (c=2) censoring"
        )

    x = np.repeat(x, n)
    c = np.repeat(c, n)
    n = np.ones_like(x).astype(int)

    x, idx = np.unique(x, return_inverse=True)

    d = np.bincount(idx, weights=1 - c)
    # do is drop outs
    do = np.bincount(idx, weights=c)
    r = n.sum() + d - d.cumsum() + do - do.cumsum()
    r = r.astype(int)
    d = d.astype(int)
    return x, r, d


def xrd_to_xcn(x, r, d):
    """
    Exact inverse of the xcn_to_xrd function.
    ONLY USE IF YOU HAVE NO TRUNCATION
    """
    n_f = np.copy(d)
    x_f = np.copy(x)
    mask = n_f != 0
    n_f = n_f[mask]
    x_f = x_f[mask]

    delta = np.abs(np.diff(np.hstack([r, [0]])))

    sus = delta - d
    x_s = x[sus > 0]
    n_s = sus[sus > 0]

    x_f = np.repeat(x_f, n_f)
    x_s = np.repeat(x_s, n_s)

    return fs_to_xcn(x_f, x_s)


def fsli_to_xcn(f=None, s=None, l=None, i=None):
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
        array of interval censored data

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
    """

    f, s, l, i = fsli_handler(f, s, l, i)
    x, c, n = fsl_to_xcn(f, s, l)

    if i.size == 0:
        return x, c, n
    else:
        x_i, n_i = np.unique(i, axis=0, return_counts=True)
        c_i = np.ones(x_i.shape[0]) * 2

        x_two = np.vstack([x, x]).T
        x = np.concatenate([x_two, x_i]).astype(float)
        c = np.hstack([c, c_i]).astype(int)
        n = np.hstack([n, n_i]).astype(int)

        x, c, n = xcn_sort(x, c, n)

        return x, c, n


def fsl_to_xcn(f=[], s=[], l=[]):
    x_f, n_f = np.unique(f, return_counts=True)
    c_f = np.zeros_like(x_f)

    x_s, n_s = np.unique(s, return_counts=True)
    c_s = np.ones_like(x_s)

    x_l, n_l = np.unique(l, return_counts=True)
    c_l = -np.ones_like(x_l)

    x = np.hstack([x_f, x_s, x_l])
    c = np.hstack([c_f, c_s, c_l]).astype(int)
    n = np.hstack([n_f, n_s, n_l]).astype(int)

    x, c, n = xcn_sort(x, c, n)

    return x, c, n


def fs_to_xcn(f=[], s=[]):
    return fsl_to_xcn(f, s, [])


def _scale(ll, n, scale):
    if scale:
        # Sometimes the ll can get too big. If we normalise it
        # against the number of observations it is smaller
        # This allows for better performance of the optimizer.
        return ll.sum() / n.sum()
    else:
        return ll.sum()


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
    if any(~(e[c == 1] is None)) or any(~(e[c != 1] is not None)):
        raise ValueError(
            "None can only be used as event type for censored observation"
        )


def wrangle_and_check_form_and_Z_cols(Z_cols, formula, df):
    if (Z_cols is None) & (formula is None):
        raise ValueError("'Z_cols' or 'formula' cannot both be None")

    if (Z_cols is not None) & (formula is not None):
        raise ValueError(
            "Either 'Z_cols' or 'formula' must be provided; not both"
        )

    if Z_cols is not None:
        unknown = [x for x in Z_cols if x not in df.columns]
        if len(unknown) > 0:
            raise ValueError("{} not in dataframe columns".format(unknown))
        Z = df[Z_cols].values.astype(float)
        mask = ~df[Z_cols].isna().any(axis=1).values
        Z = Z[mask]
        form = None
    else:
        form = Formula("0 + " + formula)
        Z = form.get_model_matrix(df, na_action="ignore").values.astype(float)
        mask = ~np.any(np.isnan(Z), axis=1)

    return Z, mask, form


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
    """
    Validates the inputs prior to be used by the CoxPH model.
    """
    # Use existing surpyval validator. But don't group and sort
    # so as to put it out of order of the event array, e.
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    e = np.array(e)
    (x, c, n) = (np.array(a).astype(float) for a in [x, c, n])

    # Check same shape
    check_e_and_x(e, x)

    # Not implemented, yet.
    if (-1 in c) or (2 in c):
        raise ValueError(
            "Left or interval censoring not implemented with Competing Risks"
        )

    # Ensure all cases where c is 0, e is not None and
    # where c is 1 e is None
    if any(~(e[c == 1] is None)) or any(~(e[c != 1] is not None)):
        raise ValueError(
            "None can only be used as event type for censored observation"
        )

    # Two baselines
    # TODO: Add fh
    if method not in ["Nelson-Aalen", "Kaplan-Meier"]:
        raise ValueError("Unrecognised baseline method")

    return x, c, n, e


def validate_event(mapping, event):
    if event is not None and event not in mapping:
        raise ValueError("Event type not in model")


def validate_cif_event(event):
    if event is None:
        raise ValueError("CIF needs event type, not None")


def check_an_ids_tl_and_x(id, tl, x):
    """
    This function checks that, for a given item, id, the history
    of the item is complete. That is, the timeline provided in the
    steps from tl[0] > x[0] == tl[1] > x[1] == ... tl[-1] > x[-1]
    This ensures there are no overlaps nor are there any gaps.
    # TODO: Consider allowing gaps in timeline, but not overlaps!
    """
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
            check_an_ids_tl_and_x(i, tl_i, x_i)

    Z, mask = wrangle_Z(Z)
    x, c, n = (arr[mask] for arr in (x, c, n))
    (x, c, n, Z) = (arr.astype(float) for arr in [x, c, n, Z])

    check_Z_and_x(Z, x)
    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)

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
            check_an_ids_tl_and_x(i, tl, x)

    Z, mask, form = wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)

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

    Z, mask, form = wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)

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

    return x, c, n, Z, form


def validate_coxph(x, c, n, Z, tl, method):
    if method not in COX_PH_METHODS:
        raise ValueError("Method must be in {}".format(COX_PH_METHODS))

    x, c, n, t = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)

    tl = t[:, 0]

    (x, c, n, Z, tl) = (np.array(a).astype(float) for a in [x, c, n, Z, tl])

    Z, mask = wrangle_Z(Z)
    x, c, n, tl = (arr[mask] for arr in (x, c, n, tl))
    (x, c, n, tl, Z) = (arr.astype(float) for arr in [x, c, n, tl, Z])

    check_Z_and_x(Z, x)

    return x, c, n, tl, Z


def validate_fine_gray_inputs(x, Z, e, c, n):

    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    e = np.array(e)
    Z, mask = wrangle_Z(Z)
    x, c, n, e = (arr[mask] for arr in (x, c, n, e))

    # Set all dtypes to float. Very poor results otherwise.
    (x, c, n, Z) = (arr.astype(float) for arr in [x, c, n, Z])

    check_e_and_x(e, x)
    check_Z_and_x(Z, x)
    check_c_and_e(c, e)
    check_left_or_int_cens(c)

    return x, Z, e, c, n


def tl_tr_to_t(tl=None, tr=None):
    if tl is not None:
        tl = np.array(tl)
    if tr is not None:
        tr = np.array(tr)

    if (tl is None) & (tr is None):
        raise ValueError("One input must not be None")
    elif (tl is not None) and (tr is not None):
        if tl.shape != tr.shape:
            raise ValueError(
                "Left array must be the same shape as right array"
            )
    elif tl is None:
        tl = np.ones_like(tr) * -np.inf
    elif tr is None:
        tr = np.ones_like(tl) * np.inf

    if (tr < tl).any():
        raise ValueError(
            "All left truncated values must be less than right truncated \
                values"
        )

    t = np.vstack([tl, tr]).T
    return t


def fs_to_xrd(f, s):
    """
    Chain of the fs_to_xrd and xrd_to_xcn functions.
    """
    x, c, n = fs_to_xcn(f, s)
    return xcn_to_xrd(x, c, n)


def fsl_to_xrd(f, s, l):
    """
    Chain of the fsl_to_xrd and xrd_to_xcn functions.
    """
    x, c, n = fsl_to_xcn(f, s, l)
    return xcn_to_xrd(x, c, n)


def round_sig(points, sig=2):
    """
    Used to round to sig significant figures.
    """
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, int(i)))
    return output
