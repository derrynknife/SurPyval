import numpy as np

from .recurrent_event_data import RecurrentEventData


def handle_xicn(x, i=None, c=None, n=None, Z=None, as_recurrent_data=True):
    if type(x) == list:
        if any([type(v) == list for v in x]):
            x_ndarray = np.empty(shape=(len(x), 2))
            for idx, val in enumerate(x):
                x_ndarray[idx, :] = np.array(val)
            x = x_ndarray
            if (x[0, :] > x[1, :]).any():
                raise ValueError("x values must be monotonically increasing")
        else:
            x = np.array(x)

    if i is None:
        i = np.ones(x.shape[0])
    else:
        i = np.array(i)

    if n is None:
        n = np.ones(x.shape[0])
    else:
        n = np.array(n)

    if c is None:
        c = np.zeros(x.shape[0])
    else:
        c = np.array(c)

    if Z is not None:
        if isinstance(Z, dict):
            Z = np.array([Z[ii] for ii in i])
        else:
            Z = np.array(Z, ndmin=2)
    # TODO: Z as a dict where the keys are the item numbers and the arrays
    # are the covariates for each i at all times (x)

    if x.shape[0] != i.shape[0]:
        raise ValueError("x and i must have the same length")
    if x.shape[0] != c.shape[0]:
        raise ValueError("x and c must have the same length")
    if x.shape[0] != n.shape[0]:
        raise ValueError("x and n must have the same length")

    if Z is not None:
        if x.shape[0] != Z.shape[0]:
            raise ValueError("x and Z must have the same length")

    if np.any((n > 1) & ((c == 0) | (c == 1))):
        raise ValueError(
            "Counts greater than 1 must be intervally or left censored"
        )

    # Check that if censored, it is the highest value
    for ii in set(i):
        ci = c[i == ii]
        if len(ci[ci == 1]) > 1:
            raise ValueError(f"Item {ii} is right censored more than once.")
        if len(ci[ci == -1]) > 1:
            raise ValueError(f"Item {ii} is left censored more than once.")

    # sort by item and x
    if x.ndim == 2:
        # Order 2D by the midpoint
        sort_order = np.lexsort((x.mean(axis=1), i))
    else:
        sort_order = np.lexsort((x, i))

    x, i, c, n = x[sort_order], i[sort_order], c[sort_order], n[sort_order]

    if Z is not None:
        Z = Z[sort_order]

    # Check that the x values for each item are monotonically increasing
    for ii in set(i):
        xi = x[i == ii]
        ci = c[i == ii]
        if xi.ndim == 2:
            for first, second in zip(xi[:-1], xi[1:]):
                if first[1] > second[0]:
                    raise ValueError(f"Item {ii} has overlapping intervals")
        else:
            if np.any(np.diff(xi) < 0):
                raise ValueError(f"Item {ii} has non-monotonic x values")

    if as_recurrent_data:
        data = RecurrentEventData(x, i, c, n)
        data.Z = Z
        return data
    else:
        return x, i, c, n
