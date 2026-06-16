"""
Trend / goodness-of-fit tests for recurrent-event (repairable-system) data.

These are standalone hypothesis tests on the raw event times of one or more
repairable systems; they do **not** require a fitted model. Both test the null
hypothesis that the events follow a homogeneous Poisson process (HPP) -- i.e.
the system shows *no trend* in its rate of occurrence of failures (ROCOF) --
against the alternative that the intensity is monotonically changing:

- an *increasing* intensity means failures arrive ever more frequently
  (wear-out / deterioration -- a "sad" system);
- a *decreasing* intensity means failures arrive ever less frequently
  (reliability growth -- a "happy" system).

Two classical statistics are provided:

``laplace``
    The Laplace (centroid) test. Compares the mean event time with the centre
    of the observation window; events bunched late are evidence of an
    increasing intensity. Asymptotically standard-normal under the HPP null.

``mil_hdbk_189c``
    The Military Handbook (MIL-HDBK-189C) test, equivalent to the total-time-
    on-test statistic. Derived from the power-law (Crow-AMSAA) NHPP, so it is
    most powerful against a power-law alternative. Chi-squared under the null.

Both functions take the same ``(x, i, T)`` data description and an
``alternative`` direction, and return a :class:`TrendTestResult`.

Systems are assumed to be observed from time ``0``. When the observation time
``T`` is not supplied each system is treated as *failure-truncated* (observed
up to its last event), and that last event -- which is the truncation point,
not a random event -- is dropped from the statistic. When ``T`` is supplied the
data are *time-truncated* (observed up to a fixed ``T``) and every event is
used.

References
----------
Ascher, H. and Feingold, H. (1984), "Repairable Systems Reliability".
Modarres, M., Kaminskiy, M. and Krivtsov, V. (2017), "Reliability Engineering
and Risk Analysis", 3rd ed., Chapter 10.
MIL-HDBK-189C (2011), "Reliability Growth Management".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

_ALTERNATIVES = ("two-sided", "increasing", "decreasing")


class TrendTestResult:
    """
    Result of a recurrent-event trend test (:func:`laplace` or
    :func:`mil_hdbk_189c`).

    Attributes
    ----------
    statistic : float
        The test statistic (a z-score for the Laplace test, a chi-squared
        value for the MIL-HDBK-189C test).
    p_value : float
        The p-value for the requested ``alternative``.
    alternative : str
        The alternative hypothesis tested: ``"two-sided"``, ``"increasing"``
        or ``"decreasing"``.
    test : str
        The name of the test.
    trend : str
        The direction of trend suggested by the statistic, independent of the
        ``alternative`` chosen: one of ``"increasing"``, ``"decreasing"`` or
        ``"none"``.
    dof : int or None
        Degrees of freedom (MIL-HDBK-189C only; ``None`` for Laplace).
    n_events : int
        The number of events contributing to the statistic.
    n_systems : int
        The number of systems (items) in the data.
    """

    def __init__(
        self,
        statistic: float,
        p_value: float,
        alternative: str,
        test: str,
        trend: str,
        n_events: int,
        n_systems: int,
        dof: int | None = None,
    ) -> None:
        self.statistic = statistic
        self.p_value = p_value
        self.alternative = alternative
        self.test = test
        self.trend = trend
        self.n_events = n_events
        self.n_systems = n_systems
        self.dof = dof

    def __repr__(self) -> str:
        lines = [
            self.test,
            "=" * len(self.test),
            "Null hypothesis  : no trend (homogeneous Poisson process)",
            "Alternative      : {a}".format(a=self.alternative),
            "Statistic        : {s:.6g}".format(s=self.statistic),
        ]
        if self.dof is not None:
            lines.append("DoF              : {d}".format(d=self.dof))
        lines.append("p-value          : {p:.6g}".format(p=self.p_value))
        lines.append("Suggested trend  : {t}".format(t=self.trend))
        return "\n".join(lines)


def _validate_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            "`alternative` must be one of {}; got {!r}".format(
                list(_ALTERNATIVES), alternative
            )
        )


def _resolve_truncation(
    T: npt.ArrayLike | dict | None, unique_i: npt.NDArray
) -> dict | None:
    """
    Normalise the observation-time argument into a ``{system_id: T}`` mapping,
    or ``None`` to signal failure-truncated data (each system observed up to
    its own last event).

    ``T`` may be a scalar (the same window for every system), a dict keyed by
    system id, or an array with one entry per (sorted-unique) system.
    """
    if T is None:
        return None
    if isinstance(T, dict):
        missing = [q for q in unique_i if q not in T]
        if missing:
            raise ValueError(
                "`T` is missing an observation time for system(s) "
                "{}".format(missing)
            )
        return {q: float(T[q]) for q in unique_i}
    if np.ndim(T) == 0:
        return {q: float(T) for q in unique_i}  # type: ignore[arg-type]
    T_arr = np.asarray(T, dtype=float)
    if T_arr.shape[0] != unique_i.shape[0]:
        raise ValueError(
            "array `T` must have one entry per system ({} systems, {} "
            "entries)".format(unique_i.shape[0], T_arr.shape[0])
        )
    return {q: float(T_arr[k]) for k, q in enumerate(unique_i)}


def _prepare(
    x: npt.ArrayLike, i: npt.ArrayLike | None, T: npt.ArrayLike | dict | None
) -> tuple[list[tuple[npt.NDArray, float]], int, int]:
    """
    Group the event times by system and resolve each system's observation
    window. Returns a list of ``(events_used, T_q)`` per system (with the
    truncating final event dropped for failure-truncated data), the total
    number of events used, and the number of systems.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1D array of event times")
    if x.size == 0:
        raise ValueError("`x` is empty; no event times to test")
    if not np.all(np.isfinite(x)):
        raise ValueError("event times must be finite")
    if np.any(x <= 0):
        raise ValueError(
            "event times must be strictly positive; systems are assumed to be "
            "observed from time 0"
        )

    if i is None:
        i = np.ones(x.shape[0])
    else:
        i = np.asarray(i)
        if i.shape[0] != x.shape[0]:
            raise ValueError("`x` and `i` must have the same length")

    unique_i = np.unique(i)
    windows = _resolve_truncation(T, unique_i)

    systems: list[tuple[npt.NDArray, float]] = []
    n_used = 0
    for q in unique_i:
        xq = np.sort(x[i == q])
        if windows is None:
            # Failure-truncated: the last event is the truncation point.
            Tq = float(xq[-1])
            used = xq[:-1]
        else:
            Tq = windows[q]
            if np.any(xq > Tq):
                raise ValueError(
                    "system {!r} has event time(s) after its observation "
                    "time T={}".format(q, Tq)
                )
            used = xq[xq <= Tq]
        if Tq <= 0:
            raise ValueError(
                "observation time for system {!r} must be positive".format(q)
            )
        systems.append((used, Tq))
        n_used += used.size

    if n_used < 2:
        raise ValueError(
            "at least two events are required to test for a trend"
        )

    return systems, n_used, int(unique_i.shape[0])


def laplace(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    T: npt.ArrayLike | dict | None = None,
    alternative: str = "two-sided",
) -> TrendTestResult:
    r"""
    The Laplace (centroid) trend test for recurrent-event data.

    Under the null hypothesis that the events of each system follow a
    homogeneous Poisson process, the event times are uniformly distributed on
    the observation window, so their mean sits at the centre. The standardised
    departure of the observed event-time total from its null expectation,

    .. math::
        U = \frac{\sum_q \sum_j t_{qj} - \sum_q n_q T_q / 2}
                 {\sqrt{\sum_q n_q T_q^2 / 12}},

    is asymptotically standard normal. ``U > 0`` (events bunched late) is
    evidence of an *increasing* intensity (deterioration); ``U < 0`` of a
    *decreasing* intensity (reliability growth).

    Parameters
    ----------
    x : array_like
        Event (failure) times, all strictly positive (systems are observed
        from time 0). For multiple systems, the times of every system are
        concatenated and identified by ``i``.
    i : array_like, optional
        System / item id for each event in ``x``. Defaults to a single system.
    T : scalar, array_like or dict, optional
        Observation (truncation) time. A scalar applies the same window to
        every system; an array gives one window per sorted-unique system; a
        dict is keyed by system id. If omitted, each system is treated as
        failure-truncated -- observed up to its last event, which is then
        excluded from the statistic.
    alternative : str, optional
        Direction of the alternative hypothesis: ``"two-sided"`` (default),
        ``"increasing"`` (upper tail; deterioration) or ``"decreasing"``
        (lower tail; reliability growth).

    Returns
    -------
    TrendTestResult
        Object carrying the ``statistic`` (the z-score ``U``), the
        ``p_value`` and the suggested ``trend``.

    Examples
    --------
    >>> from surpyval.recurrent.tests import laplace
    >>> # Inter-arrival times shrinking -> failures speeding up.
    >>> x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    >>> res = laplace(x, T=60)
    >>> bool(res.statistic > 0)
    True
    >>> res.trend
    'increasing'
    """
    _validate_alternative(alternative)
    systems, n_used, n_systems = _prepare(x, i, T)

    total = 0.0
    expected = 0.0
    variance = 0.0
    for used, Tq in systems:
        nq = used.size
        total += float(used.sum())
        expected += nq * Tq / 2.0
        variance += nq * Tq**2 / 12.0

    if variance <= 0:
        raise ValueError(
            "the null variance is zero; not enough events to compute the "
            "Laplace statistic"
        )

    u = (total - expected) / np.sqrt(variance)

    if alternative == "increasing":
        p_value = float(norm.sf(u))
    elif alternative == "decreasing":
        p_value = float(norm.cdf(u))
    else:
        p_value = float(2.0 * norm.sf(abs(u)))

    return TrendTestResult(
        statistic=float(u),
        p_value=p_value,
        alternative=alternative,
        test="Laplace Trend Test",
        trend=_trend_from_sign(u),
        n_events=n_used,
        n_systems=n_systems,
    )


def mil_hdbk_189c(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    T: npt.ArrayLike | dict | None = None,
    alternative: str = "two-sided",
) -> TrendTestResult:
    r"""
    The Military Handbook (MIL-HDBK-189C) trend test for recurrent-event data.

    Derived from the power-law (Crow-AMSAA) NHPP, the statistic

    .. math::
        \chi^2 = 2 \sum_q \sum_j \ln\!\left(\frac{T_q}{t_{qj}}\right)

    is chi-squared distributed with :math:`2N` degrees of freedom under the HPP
    null (where :math:`N` is the number of events used). It equals
    :math:`2N / \hat\beta` for the power-law shape estimate :math:`\hat\beta`,
    so a *small* statistic (large :math:`\hat\beta > 1`) indicates an
    *increasing* intensity (deterioration) and a *large* statistic (small
    :math:`\hat\beta < 1`) a *decreasing* intensity (reliability growth). It is
    the most powerful test against a power-law alternative.

    Parameters
    ----------
    x : array_like
        Event (failure) times, all strictly positive (systems are observed
        from time 0). For multiple systems, the times of every system are
        concatenated and identified by ``i``.
    i : array_like, optional
        System / item id for each event in ``x``. Defaults to a single system.
    T : scalar, array_like or dict, optional
        Observation (truncation) time. A scalar applies the same window to
        every system; an array gives one window per sorted-unique system; a
        dict is keyed by system id. If omitted, each system is treated as
        failure-truncated -- observed up to its last event, which is then
        excluded (giving :math:`2(n-1)` degrees of freedom per system).
    alternative : str, optional
        Direction of the alternative hypothesis: ``"two-sided"`` (default),
        ``"increasing"`` (deterioration; lower tail of the chi-squared) or
        ``"decreasing"`` (reliability growth; upper tail).

    Returns
    -------
    TrendTestResult
        Object carrying the chi-squared ``statistic``, its ``dof``, the
        ``p_value`` and the suggested ``trend``.

    Examples
    --------
    >>> from surpyval.recurrent.tests import mil_hdbk_189c
    >>> x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    >>> res = mil_hdbk_189c(x, T=60)
    >>> res.dof
    18
    >>> res.trend
    'increasing'
    """
    _validate_alternative(alternative)
    systems, n_used, n_systems = _prepare(x, i, T)

    statistic = 0.0
    dof = 0
    for used, Tq in systems:
        statistic += 2.0 * float(np.sum(np.log(Tq / used)))
        dof += 2 * used.size

    # Mean of a chi-squared is its dof; departures below indicate an
    # increasing intensity, above a decreasing one.
    if statistic < dof:
        trend = "increasing"
    elif statistic > dof:
        trend = "decreasing"
    else:
        trend = "none"

    lower = float(chi2.cdf(statistic, dof))
    upper = float(chi2.sf(statistic, dof))
    if alternative == "increasing":
        p_value = lower
    elif alternative == "decreasing":
        p_value = upper
    else:
        p_value = min(1.0, 2.0 * min(lower, upper))

    return TrendTestResult(
        statistic=float(statistic),
        p_value=p_value,
        alternative=alternative,
        test="MIL-HDBK-189C Trend Test",
        trend=trend,
        n_events=n_used,
        n_systems=n_systems,
        dof=dof,
    )


def _trend_from_sign(u: float) -> str:
    if u > 0:
        return "increasing"
    if u < 0:
        return "decreasing"
    return "none"
