import numpy as np
from scipy.sparse import dok_matrix

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)

from .fleming_harrington import fleming_harrington as fh
from .kaplan_meier import kaplan_meier as km
from .nelson_aalen import nelson_aalen as na


def turnbull(x, c, n, t, estimator="Fleming-Harrington"):
    any_truncated = np.isfinite(t).any()
    # Find all unique bounding points
    bounds = np.unique(np.concatenate([np.unique(x), np.unique(t)]))
    # Add the times at which there was an observation again since
    # the failure occurs in a 0 bound e.g. in the [1, 1] "interval".
    bounds = np.sort(np.concatenate([bounds, np.unique(x[c == 0])]))
    # Total items observed
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

    alpha = dok_matrix((N, M), dtype="int32")
    beta = dok_matrix((N, M), dtype="int32")

    for i in range(0, N):
        x1, x2 = xl[i], xr[i]
        t1, t2 = tl[i], tr[i]
        if x1 == x2:
            # Find first index of repeated value
            idx = np.searchsorted(bounds, x1)
            alpha[i, idx] = n[i]
        elif x2 == np.inf:
            alpha[i, :] = ((bounds > x1) & (bounds <= x2)).astype(int) * n[i]
            if x1 in x[c == 0]:
                idx = np.searchsorted(bounds, x1)
                alpha[i, idx] = 0
        else:
            alpha[i, :] = ((bounds >= x1) & (bounds < x2)).astype(int) * n[i]
            if x1 in x[c == 0]:
                idx = np.searchsorted(bounds, x1)
                alpha[i, idx] = 0
        # Find the indices of the bounds that are outside the interval
        if any_truncated:
            where_truncated = np.multiply((bounds < t1), (bounds >= t2))
            # Assign a value of 1 in the sparse matrix for the intervals where
            # the observation window is truncated
            for j in np.where(where_truncated == x1)[0]:
                beta[i, j] = 1

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
    expected = dok_matrix(alpha.shape, dtype="float64")

    while (iters < 1000) & (
        not np.allclose(p, p_prev, rtol=1e-30, atol=1e-30)
    ):
        p_prev = p
        iters += 1
        # TODO: Change this so that it does row iterations on sparse matrices
        # Row wise should, in the majority of cases, be more memory efficient
        denominator = np.zeros(N)

        for (i, j), v in zip(alpha.keys(), alpha.values()):
            denominator[i] += v * p[j]
            expected[i, j] = v**2 * p[j]

        d_observed = np.array(
            expected.multiply(1 / denominator[:, np.newaxis]).sum(0)
        ).ravel()

        d_ghosts = np.zeros(M)
        if any_truncated:
            beta_denominator = np.zeros(N)
            for (i, j), v in zip(beta.keys(), beta.values()):
                beta_denominator[i] += (1 - v) * p[j]
            beta_denominator = np.where(
                beta_denominator == 0, 1, beta_denominator
            )

            d_ghosts = np.array(
                beta.power(2)
                .multiply(n * p)
                .multiply(1 / beta_denominator[:, np.newaxis])
                .sum(0)
            ).ravel()

        # Deaths/Failures/Events
        d = d_ghosts + d_observed
        # total observed and unobserved failures.
        total_events = (d_ghosts + d_observed).sum()
        # Risk set, i.e the number of items at risk at immediately before x
        r = total_events - d.cumsum() + d
        # Find the survival function values (R) using the deaths and risk set
        # The 'official' way to do it, which is equivalent to using KM,
        # is to do p = (nu + mu).sum(axis=0)/(nu + mu).sum()
        R = func(r, d)
        # Calculate the probability mass in each interval
        p = np.abs(np.diff(np.hstack([[1], R])))
        expected.clear()

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

    def _fit(self, x, c, n, t, turnbull_estimator):
        return turnbull(x, c, n, t, turnbull_estimator)


Turnbull = Turnbull_()
