"""The actuarial life table of right censored, possibly late-entry data:
:func:`life_table`, the counts in each time interval (entered, at risk,
censored, events) with the actuarial (Cutler-Ederer) estimates of the
survival, density and hazard and their standard errors.

It imports no plotting library.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.surpyval_data import SurpyvalData

from .nonparametric import NonParametric

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["life_table"]

#: The columns of the table, in order.
COLUMNS = (
    "start",
    "end",
    "entered",
    "at_risk_start",
    "censored",
    "events",
    "at_risk",
    "q",
    "p",
    "surv",
    "se_surv",
    "pdf",
    "se_pdf",
    "hazard",
    "se_hazard",
)


def life_table(
    x: Any,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
    intervals: int | npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""
    The actuarial life table (the Cutler-Ederer estimate) of right
    censored data, with or without late entry: the number entering, at
    risk, censored and having the event in each time interval, and the
    estimated survival, density and hazard there with their standard
    errors, as Klein and Moeschberger (2003, section 5.4) and R's
    ``KMsurv::lifetab`` give them.

    The intervals are ``[start, end)``: an event or a censoring at a
    boundary counts in the interval that starts there. Data already
    grouped into intervals (as ``lifetab`` takes them) is given as one
    row per interval and outcome, at the interval's start, with the count
    in ``n``: ``x=[0, 0, 2, 2]``, ``c=[0, 1, 0, 1]``, ``n=[77, 2, 71, 3]``
    with ``intervals=[0, 2, 3]``.

    Parameters
    ----------
    x : array like, SurpyvalData or NonParametric
        The times of the events or censorings; or a
        :class:`~surpyval.SurpyvalData` holding the data, or a fitted
        non-parametric model (``KaplanMeier.fit(...)``), whose data is
        used. ``c``, ``n`` and ``tl`` are then left out. Data or a model
        with weights ``w`` is refused (a weighted life table is not built
        yet; ``KaplanMeier.fit`` takes the weights).
    c : array like, optional
        The censoring flags: 0 an event, 1 right censored. Left (-1) and
        interval (2) censored data have no life table of counts; estimate
        their survival with :class:`~surpyval.Turnbull` instead.
    n : array like, optional
        The count of each row. Defaults to 1.
    tl : array like or float, optional
        Each row's entry time (left truncation): it is at risk on
        ``(tl, x]``. Defaults to entry at the start of the table.
    intervals : int or array like, optional
        The interval boundaries, increasing; or the number of intervals,
        equal in width from the start of the table, the last of which is
        open (``[start, inf)``). Rows from the last boundary on are put in
        an open last interval, and a last boundary of ``inf`` makes it
        open. Without it the width is the Freedman-Diaconis rule's on the
        times, from the start of the table (0, or the smallest time if it
        is negative) past the largest time.

    Returns
    -------
    pandas.DataFrame
        One row per interval, with the columns:

        - ``start``, ``end``: the interval ``[start, end)``.
        - ``entered``: the number entering during the interval (``start <
          tl < end``); those with ``tl <= start`` are at risk from its
          start.
        - ``at_risk_start``: the number at risk at its start (``lifetab``'s
          ``nsubs``).
        - ``censored``, ``events``: the numbers censored (withdrawn,
          ``nlost``) and having the event (``nevent``) in it.
        - ``at_risk``: the effective number at risk (``nrisk``),
          ``at_risk_start - censored / 2 + entered / 2``: a unit censored
          or entering during the interval is taken to be at risk for half
          of it.
        - ``q``, ``p``: the conditional probabilities of the event in the
          interval, ``events / at_risk``, and of surviving it, ``1 - q``.
        - ``surv``: the survival at the interval's start, the product of
          the earlier intervals' ``p``, with its Greenwood standard error
          ``se_surv``.
        - ``pdf``, ``hazard``: the density and the hazard rate at the
          interval's midpoint, per unit of time, ``surv * q / width`` and
          ``events / (width * (at_risk - events / 2))``, with their
          standard errors ``se_pdf`` and ``se_hazard``. They are ``nan``
          for an open last interval, which has no width.

        Where nobody is at risk, ``q``, ``p``, ``pdf`` and ``hazard`` and
        their standard errors are ``nan``.

    Notes
    -----
    With ``Y_j`` the effective number at risk in interval ``j``,
    ``d_j`` the events, ``b_j`` its width, ``q_j = d_j / Y_j`` and ``p_j =
    1 - q_j``:

    .. math::

        S_j &= \prod_{i < j} p_i, \qquad
        \widehat{\mathrm{Var}}(S_j) = S_j^2 \sum_{i < j}
        \frac{q_i}{Y_i p_i}, \\
        f_j &= \frac{S_j q_j}{b_j}, \qquad
        \widehat{\mathrm{Var}}(f_j) = \left(\frac{S_j}{b_j}\right)^2
        \left(q_j^2 \sum_{i < j} \frac{q_i}{Y_i p_i}
        + \frac{q_j p_j}{Y_j}\right), \\
        h_j &= \frac{2 q_j}{b_j (1 + p_j)}, \qquad
        \widehat{\mathrm{Var}}(h_j) = \frac{h_j^2}{Y_j q_j}
        \left(1 - \left(\frac{h_j b_j}{2}\right)^2\right)

    (Klein and Moeschberger 2003, equations 5.4.1 to 5.4.6; the
    variances are written so that an interval with no events has a
    standard error of 0). Counting a late entrant as half at risk is the
    actuarial assumption for withdrawals applied to entries; with no
    late entry the table is the classical one.

    References
    ----------
    Klein, J. P. and Moeschberger, M. L. (2003), *Survival Analysis:
    Techniques for Censored and Truncated Data*, 2nd edition, Springer,
    section 5.4.

    Examples
    --------
    >>> import surpyval
    >>> x = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    >>> table = surpyval.life_table(x, c, intervals=[0, 3, 6, 9])
    >>> table[["start", "end", "at_risk_start", "censored", "events"]]
       start  end  at_risk_start  censored  events
    0    0.0  3.0             10         1       2
    1    3.0  6.0              7         1       2
    2    6.0  9.0              4         1       2
    3    9.0  inf              1         0       1
    >>> table[["at_risk", "q", "surv", "hazard"]].round(4)
       at_risk       q    surv  hazard
    0      9.5  0.2105  1.0000  0.0784
    1      6.5  0.3077  0.7895  0.1212
    2      3.5  0.5714  0.5466  0.2667
    3      1.0  1.0000  0.2343     NaN
    """
    import pandas as pd

    x_, c_, n_, tl_ = _data(x, c, n, tl)
    origin = _origin(x_, tl_)
    edges = _edges(intervals, x_, n_, origin)
    before = x_ < edges[0]
    if before.any():
        warnings.warn(
            "{:g} of the {:g} units left (by an event or censoring) before "
            "the first interval starts at {:g}, and are not in the table; "
            "start the intervals at {:g} or earlier to count them.".format(
                n_[before].sum(), n_.sum(), edges[0], x_.min()
            ),
            stacklevel=2,
        )
        x_, c_, n_, tl_ = x_[~before], c_[~before], n_[~before], tl_[~before]
    counts = _counts(edges, x_, c_, n_, tl_)
    out = _actuarial(edges, *counts)
    table = pd.DataFrame({k: out[k] for k in COLUMNS})
    for name in ("entered", "at_risk_start", "censored", "events"):
        col = table[name]
        if np.all(col == np.round(col)):
            table[name] = col.astype(int)
    return table


def _data(
    x: Any, c: Any, n: Any, tl: Any
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """The times, flags, counts and entry times, as float arrays, from
    arrays, a SurpyvalData or a fitted non-parametric model; refusing
    what a life table cannot count."""
    held = isinstance(x, (SurpyvalData, NonParametric))
    if held and not (c is None and n is None and tl is None):
        raise TypeError(
            "x holds the data ({}); leave out 'c', 'n' and 'tl', or pass "
            "arrays.".format(type(x).__name__)
        )
    if held:
        from surpyval.utils.weights import refuse_weighted_data

        refuse_weighted_data(
            x,
            "life_table",
            "it counts the units entering, failing and withdrawn in each "
            "interval, and its standard errors treat those counts as units "
            "seen; a weighted life table is not built yet",
            "Estimate the survival with KaplanMeier.fit(data), which takes "
            "the weights and gives robust intervals, or pass the arrays "
            "without the weights (life_table(x, c, n, tl)) for the table "
            "of the units themselves.",
        )
    if isinstance(x, NonParametric):
        d = getattr(x, "data", None)
        if not isinstance(d, dict) or "x" not in d:
            raise ValueError(
                "This non-parametric model does not hold its data (it was "
                "made from a survival curve); pass the times and flags "
                "instead."
            )
        data = SurpyvalData(d["x"], d["c"], d.get("n"), t=d.get("t"))
    elif isinstance(x, SurpyvalData):
        data = x
    else:
        data = SurpyvalData(x, c, n, tl=tl)
    flags = np.asarray(data.c).astype(int)
    if np.any((flags == -1) | (flags == 2)):
        raise ValueError(
            "life_table counts events and right censorings in each "
            "interval; the data has left (c=-1) or interval (c=2) censored "
            "rows, whose event times are not known to fall in one "
            "interval. Estimate their survival with surpyval.Turnbull.fit "
            "instead."
        )
    t = np.asarray(data.t, dtype=float)
    if np.any(np.isfinite(t[:, 1])):
        raise ValueError(
            "life_table takes left truncation (late entry, 'tl') only; "
            "the data has right truncation ('tr'). Estimate its survival "
            "with surpyval.Turnbull.fit instead."
        )
    x_arr = np.asarray(data.x, dtype=float)
    if x_arr.ndim != 1:
        x_arr = x_arr[:, 0]
    return x_arr, flags, np.asarray(data.n, dtype=float), t[:, 0]


def _origin(x: npt.NDArray, tl: npt.NDArray) -> float:
    """The start of the table: 0, or the earliest time or entry if it is
    negative."""
    finite = tl[np.isfinite(tl)]
    low = min(x.min(), finite.min()) if finite.size else x.min()
    return float(min(0.0, low))


def _edges(
    intervals: Any, x: npt.NDArray, n: npt.NDArray, origin: float
) -> npt.NDArray:
    """The interval boundaries, the last ``inf`` where the last interval
    is open."""
    top = float(x.max())
    if intervals is None:
        times = np.repeat(x, n.astype(int)) if np.all(n == n.round()) else x
        q75, q25 = np.percentile(times, [75, 25])
        width = 2 * (q75 - q25) * times.size ** (-1 / 3)
        if not width > 0:
            width = (top - origin) / 10 if top > origin else 1.0
        k = int(np.floor((top - origin) / width)) + 1
        edges = origin + width * np.arange(k + 1)
        if top >= edges[-1]:
            # rounding left the largest time on the last boundary
            edges = np.append(edges, np.inf)
        return edges
    if np.ndim(intervals) == 0:
        k = intervals
        if isinstance(k, bool) or not float(k).is_integer() or int(k) < 1:
            raise ValueError(
                "'intervals' is a number of intervals or their boundaries; "
                "a number must be a whole number of at least 1, got "
                "{!r}.".format(k)
            )
        k = int(k)
        if not top > origin:
            return np.array([origin, np.inf])
        return np.append(np.linspace(origin, top, k + 1)[:-1], np.inf)
    edges = np.asarray(intervals, dtype=float).ravel()
    if (
        edges.size < 2
        or np.isnan(edges).any()
        or not np.all(np.diff(edges) > 0)
        or not np.isfinite(edges[:-1]).all()
    ):
        raise ValueError(
            "'intervals' must be at least two increasing, finite boundaries "
            "(the last may be inf), e.g. [0, 5, 10, 20]; got {}.".format(
                list(edges)
            )
        )
    if np.isfinite(edges[-1]) and top >= edges[-1]:
        edges = np.append(edges, np.inf)
    return edges


def _counts(
    edges: npt.NDArray,
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
) -> tuple[npt.NDArray, ...]:
    """Per interval: the number at risk at its start, entering, censored
    and having the event."""
    k = edges.size - 1
    exit_in = np.searchsorted(edges, x, side="right") - 1
    events = np.bincount(exit_in, weights=n * (c == 0), minlength=k)[:k]
    censored = np.bincount(exit_in, weights=n * (c == 1), minlength=k)[:k]
    # Entry strictly inside an interval is a late entrant there; at or
    # before its start, the unit is at risk from the start.
    enter_in = np.searchsorted(edges, tl, side="right") - 1
    late = (enter_in >= 0) & (tl > edges[np.clip(enter_in, 0, k)])
    entered = np.bincount(enter_in[late], weights=n[late], minlength=k)[:k]
    on_time = ~late
    starting = np.bincount(
        np.maximum(enter_in[on_time], 0), weights=n[on_time], minlength=k
    )[:k]
    # The number at risk at each start: those who start at or before it
    # and have not yet left.
    flow = (
        starting + np.concatenate([[0.0], (entered - events - censored)])[:k]
    )
    at_start = np.cumsum(flow)
    return at_start, entered, censored, events


def _actuarial(
    edges: npt.NDArray,
    at_start: npt.NDArray,
    entered: npt.NDArray,
    censored: npt.NDArray,
    events: npt.NDArray,
) -> dict[str, npt.NDArray]:
    """The actuarial estimates of Klein and Moeschberger's section 5.4."""
    start, end = edges[:-1], edges[1:]
    width = end - start
    Y = at_start - censored / 2 + entered / 2
    some = Y > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(some, events / np.where(some, Y, 1.0), np.nan)
        p = 1 - q
        S = np.concatenate([[1.0], np.cumprod(np.where(some, p, 1.0))])[:-1]
        term = np.where(some & (p > 0), q / (Y * p), 0.0)
        greenwood = np.concatenate([[0.0], np.cumsum(term)])[:-1]
        se_S = np.where(S > 0, S * np.sqrt(greenwood), 0.0)
        finite = np.isfinite(width)
        pdf = np.where(finite, S * q / width, np.nan)
        se_pdf = np.where(
            finite,
            S / width * np.sqrt(q**2 * greenwood + q * p / Y),
            np.nan,
        )
        hazard = np.where(finite, 2 * q / (width * (1 + p)), np.nan)
        var_h = (
            4
            * q
            / (width**2 * (1 + p) ** 2 * Y)
            * (1 - (hazard * width / 2) ** 2)
        )
        se_hazard = np.where(finite, np.sqrt(np.maximum(var_h, 0.0)), np.nan)
    return {
        "start": start,
        "end": end,
        "entered": entered,
        "at_risk_start": at_start,
        "censored": censored,
        "events": events,
        "at_risk": Y,
        "q": q,
        "p": p,
        "surv": S,
        "se_surv": se_S,
        "pdf": pdf,
        "se_pdf": se_pdf,
        "hazard": hazard,
        "se_hazard": se_hazard,
    }
