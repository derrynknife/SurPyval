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
        The conclusion at the ``alpha_ci`` level: ``"increasing"`` or
        ``"decreasing"`` when the test rejects the no-trend null
        (``p_value < alpha_ci``), and ``"none"`` otherwise.
    direction : str
        The direction the statistic points to, whether or not it is
        significant: ``"increasing"``, ``"decreasing"`` or ``"none"`` (a
        statistic exactly at its null centre). It is not evidence of a
        trend on its own; ``trend`` is.
    alpha_ci : float
        The significance level ``trend`` is judged at.
    dof : int or None
        Degrees of freedom (MIL-HDBK-189C only; ``None`` for Laplace).
    n_events : int
        The number of events contributing to the statistic.
    n_systems : int
        The number of systems (items) in the data.

    Examples
    --------
    :func:`laplace` returns one. The gaps between these failures shrink,
    so the statistic points to an increasing rate, but with nine events
    it is not significant, so no trend is reported:

    >>> from surpyval.recurrent.tests import laplace
    >>> x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    >>> result = laplace(x, T=60)
    >>> result.direction, result.trend
    ('increasing', 'none')
    >>> round(result.statistic, 4), round(result.p_value, 4)
    (1.1547, 0.2482)
    """

    def __init__(
        self,
        statistic: float,
        p_value: float,
        alternative: str,
        test: str,
        direction: str,
        n_events: int,
        n_systems: int,
        dof: int | None = None,
        alpha_ci: float = 0.05,
    ) -> None:
        self.statistic = statistic
        self.p_value = p_value
        self.alternative = alternative
        self.test = test
        self.direction = direction
        self.alpha_ci = alpha_ci
        # A one-sided test that rejects always rejects towards its own
        # alternative, so the direction is the conclusion when significant.
        self.trend = direction if p_value < alpha_ci else "none"
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
        lines.append("Direction        : {d}".format(d=self.direction))
        if self.trend == "none":
            conclusion = "no trend detected (p >= {a:g})".format(
                a=self.alpha_ci
            )
        else:
            conclusion = "{t} (p < {a:g})".format(
                t=self.trend, a=self.alpha_ci
            )
        lines.append("Trend            : {c}".format(c=conclusion))
        return "\n".join(lines)


def _validate_alternative(alternative: str, alpha_ci: float) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            "`alternative` must be one of {}; got {!r}".format(
                list(_ALTERNATIVES), alternative
            )
        )
    if not (
        isinstance(alpha_ci, (int, float, np.integer, np.floating))
        and not isinstance(alpha_ci, bool)
        and 0 < alpha_ci < 1
    ):
        raise ValueError(
            "`alpha_ci`, the significance level, must be a number strictly "
            "between 0 and 1; got {!r}".format(alpha_ci)
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


def _events_and_windows(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None,
    T: npt.ArrayLike | dict | None,
    c: npt.ArrayLike | None,
) -> tuple:
    """
    Resolve the fitters' ``c`` into events and windows, and catch a ``c``
    passed as ``T`` (#485): ``laplace(x, i, c)``, copied from
    ``CrowAMSAA.fit(x, i, c)``, failed with "array `T` must have one entry
    per system".
    """
    if c is None:
        if T is not None and not isinstance(T, dict) and np.ndim(T) == 1:
            T_arr = np.asarray(T, dtype=float)
            n_rows = np.size(x)
            n_systems = 1 if i is None else np.unique(np.asarray(i)).size
            if (
                T_arr.size == n_rows != n_systems
                and np.isin(T_arr, [0, 1]).all()
            ):
                raise ValueError(
                    "`T` has one entry per row of `x` ({}), all 0 or 1: it "
                    "looks like the censoring flags `c` the fitters take "
                    "third. `T` is each system's observation end ({} "
                    "systems); pass the flags by keyword instead, "
                    "c=...".format(n_rows, n_systems)
                )
        return x, i, T
    if T is not None:
        raise ValueError("Give either `T` or `c`, not both")
    x_arr = np.asarray(x, dtype=float)
    c_arr = np.asarray(c)
    i_arr = np.ones(x_arr.shape[0]) if i is None else np.asarray(i)
    if not (c_arr.shape == x_arr.shape == i_arr.shape):
        raise ValueError("`x`, `i` and `c` must have the same length")
    if not np.isin(c_arr, [0, 1]).all():
        raise ValueError(
            "`c` must be 0 (an event) or 1 (the end of a system's "
            "observation)"
        )
    xs, items, windows = [], [], {}
    for q in np.unique(i_arr):
        mask = i_arr == q
        events = np.sort(x_arr[mask][c_arr[mask] == 0])
        ends = x_arr[mask][c_arr[mask] == 1]
        if ends.size:
            windows[q] = float(ends.max())
        elif events.size:
            # Failure-truncated: the last event closes the window, and is
            # not itself counted (as for T=None).
            windows[q] = float(events[-1])
            events = events[:-1]
        else:
            continue
        xs.extend(events)
        items.extend([q] * events.size)
    return np.asarray(xs), np.asarray(items), windows


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
    *,
    c: npt.ArrayLike | None = None,
    alpha_ci: float = 0.05,
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
    c : array_like, optional
        Keyword only: censoring flags in the fitters' form (0 an event, 1
        the end of a system's observation), in place of ``T``, as
        ``CrowAMSAA.fit(x, i, c)`` takes them. A system with a ``c = 1``
        row is observed to that time; one without is failure-truncated.
    alpha_ci : float, optional
        The significance level at which ``trend`` is judged (default
        0.05, keyword only): the result names a trend only when
        ``p_value < alpha_ci``.

    Returns
    -------
    TrendTestResult
        Object carrying the ``statistic`` (the z-score ``U``), the
        ``p_value``, the ``direction`` of the statistic and the ``trend``
        concluded at ``alpha_ci``.

    Examples
    --------
    >>> from surpyval.recurrent.tests import laplace
    >>> # Inter-arrival times shrinking -> failures speeding up.
    >>> x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    >>> res = laplace(x, T=60)
    >>> bool(res.statistic > 0)
    True
    >>> res.direction, round(res.p_value, 3)
    ('increasing', 0.248)
    >>> res.trend  # not significant at the default alpha_ci = 0.05
    'none'
    >>> laplace(x, T=60, alternative="increasing", alpha_ci=0.2).trend
    'increasing'
    """
    _validate_alternative(alternative, alpha_ci)
    x, i, T = _events_and_windows(x, i, T, c)
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
        direction=_trend_from_sign(u),
        n_events=n_used,
        n_systems=n_systems,
        alpha_ci=alpha_ci,
    )


def mil_hdbk_189c(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    T: npt.ArrayLike | dict | None = None,
    alternative: str = "two-sided",
    *,
    c: npt.ArrayLike | None = None,
    alpha_ci: float = 0.05,
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
    c : array_like, optional
        Keyword only: censoring flags in the fitters' form (0 an event, 1
        the end of a system's observation), in place of ``T``, as
        ``CrowAMSAA.fit(x, i, c)`` takes them. A system with a ``c = 1``
        row is observed to that time; one without is failure-truncated.
    alpha_ci : float, optional
        The significance level at which ``trend`` is judged (default
        0.05, keyword only): the result names a trend only when
        ``p_value < alpha_ci``.

    Returns
    -------
    TrendTestResult
        Object carrying the chi-squared ``statistic``, its ``dof``, the
        ``p_value``, the ``direction`` of the statistic and the ``trend``
        concluded at ``alpha_ci``.

    Examples
    --------
    >>> from surpyval.recurrent.tests import mil_hdbk_189c
    >>> x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    >>> res = mil_hdbk_189c(x, T=60)
    >>> res.dof
    18
    >>> res.direction, res.trend, round(res.p_value, 3)
    ('increasing', 'none', 0.203)
    """
    _validate_alternative(alternative, alpha_ci)
    x, i, T = _events_and_windows(x, i, T, c)
    systems, n_used, n_systems = _prepare(x, i, T)

    statistic = 0.0
    dof = 0
    for used, Tq in systems:
        statistic += 2.0 * float(np.sum(np.log(Tq / used)))
        dof += 2 * used.size

    # Mean of a chi-squared is its dof; departures below indicate an
    # increasing intensity, above a decreasing one.
    if statistic < dof:
        direction = "increasing"
    elif statistic > dof:
        direction = "decreasing"
    else:
        direction = "none"

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
        direction=direction,
        n_events=n_used,
        n_systems=n_systems,
        dof=dof,
        alpha_ci=alpha_ci,
    )


def _trend_from_sign(u: float) -> str:
    if u > 0:
        return "increasing"
    if u < 0:
        return "decreasing"
    return "none"
