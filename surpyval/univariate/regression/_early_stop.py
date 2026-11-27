"""Where an additive-hazards estimate stops short of the data.

Aalen's additive increments are a least-squares fit over the units at
risk at each event time, ``(X'YX)^{-1} X' dN``. Once too few units are at
risk to separate the additive covariates (for example, every unit with a
binary covariate at 1 has left), ``X'YX`` is singular and the increments
cannot be estimated, there or at any later event time. The estimate stops
at the last event time it could estimate. R's ``timereg`` then holds the
cumulative coefficients at their last values to the end of the data,
silently, which understates the risk wherever events go on after the
stop; SurPyval instead says where it stopped and predicts ``nan`` after
it.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import caller_stacklevel


def warn_if_stopped(
    who: str,
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    times: npt.NDArray,
    advice: str,
) -> float:
    """Warn when the estimate stops before the last event time, naming
    the last time estimated, the next event time and the units at risk
    there, and the events left out; return the last time estimated (0
    when none was), ``nan`` when the estimate reaches the last event.

    ``times`` are the event times estimated, in order; ``tl`` the entry
    times (0 for none); ``advice`` ends the message with what to do.
    """
    events = (c == 0) & (n > 0)
    last = float(x[events].max())
    stop = float(times[-1]) if times.size else 0.0
    if times.size and stop >= last:
        return float("nan")
    later = events & (x > stop)
    first_bad = float(x[later].min())
    at_risk = float(n[(tl < first_bad) & (x >= first_bad)].sum())
    left = float(n[later].sum())
    warnings.warn(
        f"{who} stopped at t = {stop:.6g}, before the last event time "
        f"({last:.6g}): at the next event time ({first_bad:.6g}) the "
        f"{at_risk:g} units at risk cannot separate the additive "
        "covariates (X'YX is singular: too few units, or all alike in a "
        f"covariate), so {left:g} later events are left out and every "
        f"prediction after t = {stop:.6g} is nan (see stopped_at). "
        f"{advice}",
        stacklevel=caller_stacklevel(),
    )
    return stop


def beyond(t: npt.ArrayLike, stopped_at: float) -> npt.NDArray:
    """Whether each time is after the estimate stopped short of the data
    (always false when ``stopped_at`` is ``nan``)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(t, dtype=float) > stopped_at
