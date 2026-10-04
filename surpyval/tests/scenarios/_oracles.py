"""Independent likelihoods and helpers for the scenario cards.

Each oracle is written from the model's definition with numpy and
scipy only, so that it shares no code with the fit it checks: the
maximum a card finds with it is the maximum the package must report.
"""

import numpy as np
from scipy.optimize import minimize


def weibull_sf(t, alpha, beta):
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return np.exp(-((t / alpha) ** beta))


def interval_counts_neg_ll(F, x, c, n):
    """The negative log-likelihood of grouped data under the CDF ``F``:
    rows of ``x`` are ``[lower, upper]``, ``c`` 2 for a failure in the
    interval and 1 for a survivor at ``lower``, ``n`` the counts."""
    x = np.asarray(x, dtype=float)
    lik = np.where(c == 2, F(x[:, 1]) - F(x[:, 0]), 1 - F(x[:, 0]))
    return -(np.asarray(n) * np.log(np.clip(lik, 1e-300, None))).sum()


def best_of(neg_ll, starts, **options):
    """The lowest minimum of ``neg_ll`` from each start (Nelder-Mead,
    tight tolerances), as a ``scipy.optimize.OptimizeResult``."""
    options = {"maxiter": 60_000, "xatol": 1e-9, "fatol": 1e-10, **options}
    results = [
        minimize(neg_ll, s, method="Nelder-Mead", options=options)
        for s in starts
    ]
    return min(results, key=lambda r: r.fun)


def contains(interval, value):
    lo, hi = np.min(interval), np.max(interval)
    return lo <= value <= hi
