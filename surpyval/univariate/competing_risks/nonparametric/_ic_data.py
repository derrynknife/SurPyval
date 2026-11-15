"""Reading interval-censored competing-risks data: the rows' windows, their
causes and counts, shared by ``CompetingRisks.fit`` and ``gray_test`` when
the data have left- or interval-censored rows (or masked causes)."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from surpyval.univariate.competing_risks.labels import ordered_labels
from surpyval.utils import xcnt_handler
from surpyval.utils.data_formats import missing_events

#: The ``masked`` values: what to do with a failure whose cause was not
#: recorded.
MASKED = ("refuse", "include")


class Rows(NamedTuple):
    """Each row's window ``(xl, xr]`` (``xl == xr`` exact, ``xr = inf``
    right censored), its cause index (-1: none recorded), count and the
    causes in order."""

    xl: npt.NDArray
    xr: npt.NDArray
    cause: npt.NDArray
    n: npt.NDArray
    causes: list


def windowed_input(x: Any, c: Any, xl: Any, xr: Any, masked: str) -> bool:
    """Whether the data may hold a left- or interval-censored row (or a
    masked cause to include): then the rows are read as windows by
    :func:`read_rows`. One column of times with flags 0 and 1 only (and
    ``masked="refuse"``) is read as it always was."""
    if xl is not None or xr is not None or masked == "include":
        return True
    if x is not None and not np.isscalar(x) and np.ndim(x) == 2:
        return True
    if c is None:
        return False
    return bool(np.isin(np.asarray(c, dtype=object), (-1, 2)).any())


def _as_objects(e: Any) -> npt.NDArray:
    if isinstance(e, (list, tuple)):
        out = np.empty(len(e), dtype=object)
        for i, v in enumerate(e):
            out[i] = v
        return out
    return np.asarray(e, dtype=object)


def read_rows(
    x: Any,
    e: Any,
    c: Any,
    n: Any,
    xl: Any,
    xr: Any,
    masked: str,
    what: str,
) -> Rows:
    """Check and read the data of ``what``; every refusal says what to
    do."""
    if masked not in MASKED:
        raise ValueError(
            f"masked must be one of {list(MASKED)}; got {masked!r}. "
            "'refuse' (the default) refuses a failure with no cause; "
            "'include' counts it as a failure from some cause in its window."
        )
    if c is not None:
        from surpyval.utils.data_formats import refuse_boolean_censoring

        refuse_boolean_censoring(c)
    if c is not None and x is None and xl is not None and xr is not None:
        x = np.column_stack(
            [np.asarray(xl, dtype=float), np.asarray(xr, dtype=float)]
        )
        xl = xr = None
    if c is not None and x is not None and np.ndim(x) == 2:
        # With flags given, a right-censored row may be written [xl, inf]
        # and a left-censored one [-inf, xr]: read as the time alone, as
        # the handler wants it.
        x = np.array(x, dtype=float)
        flags = np.asarray(c)
        if flags.shape == (x.shape[0],):
            x[flags == 1, 1] = x[flags == 1, 0]
            x[flags == -1, 0] = x[flags == -1, 1]
    xd, cd, nd, _ = xcnt_handler(
        x=x, c=c, n=n, xl=xl, xr=xr, group_and_sort=False
    )
    cd = np.asarray(cd).astype(int)
    nd = np.asarray(nd, dtype=float)
    N = cd.size
    e = _as_objects(e)
    if e.shape != (N,):
        raise ValueError(
            f"e must have one cause per row: got {e.size} for {N} rows."
        )
    missing = missing_events(e)
    e = e.copy()
    e[missing] = None
    if c is None:
        # Without flags, a row with no cause and one time is right
        # censored there; a window's flag comes from its times.
        cd = np.where(missing & (cd == 0), 1, cd)

    xd = np.asarray(xd, dtype=float)
    if xd.ndim == 1:
        lower, upper = xd.copy(), xd.copy()
    else:
        lower, upper = xd[:, 0].copy(), xd[:, 1].copy()
    lower[cd == -1] = -np.inf
    upper[cd == 1] = np.inf
    if np.any(np.isnan(lower) | np.isnan(upper)):
        raise ValueError(f"{what} needs a time on every row (no NaN).")

    suspended = (cd == 1) & ~missing
    if suspended.any():
        from surpyval.utils.warnings import named, warn_suspension_types

        e[suspended] = None
        missing = missing | suspended
        warn_suspension_types(
            int(suspended.sum()),
            "rows " + named(np.flatnonzero(suspended).tolist()),
        )
    unknown_failure = (cd != 1) & missing
    if unknown_failure.any() and masked == "refuse":
        k = int(unknown_failure.sum())
        raise ValueError(
            f"{k} failure{'' if k == 1 else 's'} (c = 0, -1 or 2) with no "
            "cause in e (masked causes). Give each failure its cause, or "
            "pass masked='include' to count such a row as a failure from "
            "some cause in its window (this assumes whether the cause was "
            "recorded does not depend on which cause it was). A row that "
            "had not failed is right censored: c = 1."
        )
    causes = ordered_labels(e[~missing])
    if not causes:
        raise ValueError(
            f"{what} needs at least one failure with its cause in e."
        )
    index = {k: i for i, k in enumerate(causes)}
    cause = np.full(N, -1, dtype=np.int64)
    for i in np.flatnonzero(~missing):
        cause[i] = index[e[i]]
    if not (np.all(np.isfinite(nd)) and np.all(nd > 0)):
        raise ValueError("n must be finite and positive.")
    return Rows(lower, upper, cause, nd, causes)
