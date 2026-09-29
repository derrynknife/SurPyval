"""
The inverse-probability-of-censoring-weighting (IPCW) toolkit: the
Kaplan-Meier estimate of the *censoring* distribution and the step
lookups used to evaluate it (right-continuous, ``step_at``, and its left
limit, ``step_left_limit``).

Three modules -- Gray's test, Fine-Gray regression and the prediction
metrics -- each carried their own copy of both functions. The copies had
already started to drift: the metrics copy silently ignored count
weights, consistent with its callers but a trap for the next reuse. This
module is the single, weighted implementation; passing no ``n`` is the
unweighted case.

The bodies are transplants from the competing-risks copies, kept
bit-identical so consolidating changed no fitted numbers. The callers do
differ in one convention, how an event and a censoring at the same time
are ordered, which :func:`censoring_survival` selects with ``ties``:
Fine-Gray regression uses the default (``cmprsk::crr``'s), the
prediction metrics use ``"events_first"`` (``prodlim``'s and
scikit-survival's reverse Kaplan-Meier).
"""

import numpy as np
import numpy.typing as npt


def censoring_survival(
    x: npt.NDArray,
    censored: npt.NDArray,
    n: "npt.NDArray | None" = None,
    ties: str = "censoring_first",
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
    ties : {"censoring_first", "events_first"}, optional
        Which risk set a censoring at ``t`` is measured against when an
        event also falls at ``t``.

        * ``"censoring_first"`` (default): everyone with ``x >= t``,
          including the events at ``t`` -- they count as still at risk of
          being censored at ``t``. This is the convention of
          ``cmprsk::crr``, which Fine-Gray regression reproduces.
        * ``"events_first"``: everyone with ``x > t`` plus those censored
          at ``t``. An event at ``t`` is taken to precede a censoring at
          ``t`` (the convention the data themselves follow: a tie is
          recorded as an event), so it is no longer at risk of censoring.
          This is the reverse Kaplan-Meier of ``prodlim`` (``reverse =
          TRUE``) and scikit-survival, used by the prediction metrics in
          :mod:`surpyval.metrics.validation`.

        The two agree unless an event and a censoring share a time.

    Examples
    --------
    An event and a censoring both at ``t = 2``:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 2.0, 3.0])
    >>> censored = np.array([False, False, True, False])
    >>> censoring_survival(x, censored)[1]  # 1 - 1/3 at t = 2
    array([1.        , 0.66666667, 0.66666667])
    >>> censoring_survival(x, censored, ties="events_first")[1]  # 1 - 1/2
    array([1. , 0.5, 0.5])
    """
    if ties not in ("censoring_first", "events_first"):
        raise ValueError(
            "ties must be 'censoring_first' or 'events_first', "
            "got {!r}".format(ties)
        )
    x = np.asarray(x, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if n is None:
        n = np.ones(x.size)
    times = np.unique(x)
    G = np.ones(times.size)
    surv = 1.0
    for i, t in enumerate(times):
        cens_here = n[(x == t) & censored].sum()
        if ties == "events_first":
            at_risk = n[x > t].sum() + cens_here
        else:
            at_risk = n[x >= t].sum()
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


def step_left_limit(
    times: npt.NDArray,
    values: npt.NDArray,
    query: npt.ArrayLike,
    before: float,
) -> npt.NDArray:
    """
    The left limit ``f(query-)`` of the step function :func:`step_at`
    evaluates: the value carried by the largest ``times`` entry strictly
    ``< query``; ``before`` is returned where no time precedes ``query``.

    It differs from :func:`step_at` only where ``query`` is one of
    ``times``: there it is the value before that step rather than after
    it. Fine-Gray regression evaluates the censoring survival this way,
    ``G(t-)``, so that censorings at a time do not yet count against an
    event at the same time (see
    :mod:`surpyval.univariate.competing_risks.regression.fine_gray`).

    Examples
    --------
    >>> import numpy as np
    >>> times, values = np.array([1.0, 2.0]), np.array([0.8, 0.5])
    >>> step_at(times, values, [1.0, 1.5, 2.0], before=1.0)
    array([0.8, 0.8, 0.5])
    >>> step_left_limit(times, values, [1.0, 1.5, 2.0], before=1.0)
    array([1. , 0.8, 0.8])
    """
    idx = np.searchsorted(times, query, side="left") - 1
    return np.where(idx < 0, before, values[np.clip(idx, 0, values.size - 1)])
