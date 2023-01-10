import numpy as np

from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

from .fleming_harrington import fleming_harrington as fh
from .kaplan_meier import kaplan_meier as km
from .nelson_aalen import nelson_aalen as na


def turnbull(x, c, n, t, estimator="Fleming-Harrington"):
    bounds = np.unique(np.concatenate([np.unique(x), np.unique(t)]))
    bounds = np.sort(np.concatenate([bounds, np.unique(x[c == 0])]))
    N = n.sum()

    if x.ndim == 1:
        x_new = np.empty(shape=(x.shape[0], 2))
        x_new[:, 0] = x
        x_new[:, 1] = x
        x = x_new

    # Unpack x array
    xl = x[:, 0]
    xr = x[:, 1]

    # Unpack t array
    tl = t[:, 0]
    tr = t[:, 1]

    # If there are left and right censored observations,
    # convert them to interval censored observations
    xl[c == -1] = -np.inf
    xr[c == 1] = np.inf

    # Find the count of intervals (M) and unique observation windows (N)
    M = bounds.size
    N = xl.size

    alpha = np.zeros(shape=(N, M))
    beta = np.ones(shape=(N, M))

    for i in range(0, N):
        x1, x2 = xl[i], xr[i]
        t1, t2 = tl[i], tr[i]
        if x1 == x2:
            # Find first index of repeated value
            idx = np.where(bounds == x1)[0][0]
            alpha[i, idx] = n[i]
        elif x2 == np.inf:
            alpha[i, :] = ((bounds > x1) & (bounds <= x2)).astype(int) * n[i]
            if x1 in x[c == 0]:
                idx = np.where(bounds == x1)[0][0]
                alpha[i, idx] = 0
        else:
            alpha[i, :] = ((bounds >= x1) & (bounds < x2)).astype(int) * n[i]
            if x1 in x[c == 0]:
                idx = np.where(bounds == x1)[0][0]
                alpha[i, idx] = 0

        beta[i, :] = ((bounds >= t1) & (bounds < t2)).astype(int)

    beta[:, M - 1] = 1
    n = n.reshape(-1, 1)
    d = np.zeros(M)
    p = np.ones(M) / M

    iters = 0
    p_prev = np.zeros_like(p)

    if estimator == "Kaplan-Meier":
        func = km
    elif estimator == "Nelson-Aalen":
        func = na
    else:
        func = fh

    old_err_state = np.seterr(all="ignore")

    while (iters < 1000) & (
        not np.allclose(p, p_prev, rtol=1e-30, atol=1e-30)
    ):
        p_prev = p
        iters += 1
        ap = alpha * p
        # Observed deaths (left, right, or intervally 'observed')
        mu = alpha * ap / ap.sum(axis=1, keepdims=True)
        # Expected additional failures due to truncation
        nu = (
            n
            * (1 - beta)
            * (1 - beta)
            * p
            / (beta * p).sum(axis=1, keepdims=True)
        )

        # Deaths/Failures
        d = (nu + mu).sum(axis=0)
        # M total observed and unobserved failures.
        M = (nu + mu).sum()
        # Risk set, i.e the number of items at risk at immediately before x
        r = M - d.cumsum() + d
        # Find the survival function values (R) using the deaths and risk set
        R = func(r, d)
        # Calculate the probability mass in each interval
        p = np.abs(np.diff(np.hstack([[1], R])))

        # The 'official' way to do it which is equivalent to using KM
        # p = (nu + mu).sum(axis=0)/(nu + mu).sum()

    out = {}
    out["x"] = bounds[1:-1]
    out["r"] = r[1:-1]
    out["d"] = d[1:-1]
    out["R"] = R[0:-2]
    out["R_upper"] = R[0:-2]
    out["R_lower"] = R[1:-1]
    out["alpha"] = alpha
    out["bounds"] = bounds

    np.seterr(**old_err_state)

    return out


class Turnbull_(NonParametricFitter):
    r"""
    Turnbull estimator class. Returns a `NonParametric` object from method
    :code:`fit()`. Calculates the Non-Parametric estimate of the survival
    function using the Turnbull NPMLE.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Turnbull
    >>> x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
    >>> model = Turnbull.fit(x)
    >>> model.R
    array([1.        , 1.        , 0.63472351, 0.29479882, 0.2631432 ,
           0.2631432 , 0.2631432 , 0.09680497])
    """

    def __init__(self):
        self.how = "Turnbull"


Turnbull = Turnbull_()
