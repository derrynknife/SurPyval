import numpy as np
import numpy_indexed as npi


def handle_xicn(x, i=None, c=None, n=None):
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

    return x, i, c, n


def xicn_to_xrd(x, i, c, n):
    # This function assumes that it has been handled by the above
    # handler.
    gb_x = npi.group_by(x)
    r = npi.group_by(i).max(x)[1]
    r = np.array([1 if xi in r else 0 for xi in x])
    r = gb_x.sum(r)[1]
    r = r[::-1].cumsum()[::-1]
    x, d = gb_x.sum(n * (c == 0).astype(int))

    return x, r, d
