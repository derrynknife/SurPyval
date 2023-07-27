import numpy as np
import numpy_indexed as npi

from .recurrent_event_data import RecurrentEventData


def handle_xicn(x, i=None, c=None, n=None, as_recurrent_data=True):
    x = np.array(x)

    if i is None:
        i = np.ones_like(x)
    else:
        i = np.array(i)

    if n is None:
        n = np.ones_like(x)
    else:
        n = np.array(n)

    if c is None:
        c = np.zeros_like(x)
    else:
        c = np.array(c)

    if npi.group_by(i).sum((c == 1).astype(int))[1].max() > 1:
        raise ValueError("An item is censored more than once.")

    # Check that if censored, it is the highest value
    for ii in set(i):
        ci = c[i == ii]
        xi = x[i == ii]
        if 1 in ci:
            if np.any(xi[ci == 1] <= xi[ci != 1]):
                raise ValueError(
                    f"Item {ii} has censored value lower than an event value"
                )

    # sort by item and x
    sort_order = np.lexsort((x, i))
    x, i, c, n = x[sort_order], i[sort_order], c[sort_order], n[sort_order]

    # Check that the x values for each item are monotonically increasing
    for ii in set(i):
        xi = x[i == ii]
        if np.any(np.diff(xi) < 0):
            raise ValueError(f"Item {ii} has non-monotonic x values")

    if as_recurrent_data:
        return RecurrentEventData(x, i, c, n)
    else:
        return x, i, c, n
