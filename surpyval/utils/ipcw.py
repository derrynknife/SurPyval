"""
The inverse-probability-of-censoring-weighting (IPCW) toolkit: the
Kaplan-Meier estimate of the *censoring* distribution and the
right-continuous step lookup used to evaluate it.

Three modules -- Gray's test, Fine-Gray regression and the prediction
metrics -- each carried their own copy of both functions. The copies had
already started to drift: the metrics copy silently ignored count
weights, consistent with its callers but a trap for the next reuse. This
module is the single, weighted implementation; passing no ``n`` is the
unweighted case.

The bodies are transplants from the competing-risks copies, kept
bit-identical so consolidating changed no fitted numbers.
"""

import numpy as np
import numpy.typing as npt


def censoring_survival(
    x: npt.NDArray,
    censored: npt.NDArray,
    n: "npt.NDArray | None" = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Kaplan-Meier estimate of the censoring survival ``G(t) = P(C > t)``.

    The roles are reversed relative to an ordinary survival fit:
    right-censored rows (``censored`` true) are the "events" for the
    censoring distribution and observed events are treated as censored.
    Returns the sorted unique times and the right-continuous ``G``
    evaluated at each.

    Parameters
    ----------
    x : ndarray
        Observed times.
    censored : ndarray of bool
        True where the observation was right-censored.
    n : ndarray, optional
        Count weight per observation; default 1 each.
    """
    x = np.asarray(x, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if n is None:
        n = np.ones(x.size)
    times = np.unique(x)
    G = np.ones(times.size)
    surv = 1.0
    for i, t in enumerate(times):
        at_risk = n[x >= t].sum()
        cens_here = n[(x == t) & censored].sum()
        if at_risk > 0:
            surv *= 1.0 - cens_here / at_risk
        G[i] = surv
    return times, G


def step_at(
    times: npt.NDArray,
    values: npt.NDArray,
    query: npt.ArrayLike,
    before: float,
) -> npt.NDArray:
    """
    Right-continuous step function: the value carried by the largest
    ``times`` entry ``<= query``; ``before`` is returned where ``query``
    precedes the first time. For a survival curve ``before`` is 1, for a
    cumulative hazard it is 0.
    """
    idx = np.searchsorted(times, query, side="right") - 1
    return np.where(idx < 0, before, values[np.clip(idx, 0, values.size - 1)])
