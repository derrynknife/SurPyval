"""Reshaping event history data on its arrays, where that gives what the
record route (:mod:`surpyval.utils.data_reshape`: each item's
:class:`~surpyval.utils.item_history.ItemHistory` reshaped, then
:meth:`EventHistoryData.from_items`) gives, without making a record per
item. Where the two could differ -- the record route refuses data that
the arrays allow, or the data holds what the records carry item by item
-- these return ``None`` and the caller takes the record route."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def without_covariates(data: Any) -> Any:
    """``data`` without its covariates, as
    ``EventHistoryData.from_items([r.without covariates for r in
    data.to_items()])`` builds it: the same rows, in the same order, with
    the same dtypes, through the same ``handle_xicn``. ``None`` where the
    records could refuse the data or change it in ways the arrays do not
    follow (see :func:`_records_follow_arrays`), or where the arrays are
    refused: the record route then runs, and refuses with its own message
    or gives its result."""
    from surpyval.utils.recurrent_utils import handle_xicn

    rows = _records_follow_arrays(data)
    if rows is None:
        return None
    x2, c, n, tl_rows, tr_rows = rows
    # As ItemHistory._rows writes each row: an interval as [lower, upper],
    # every other row at its lower time, counted once unless a count
    # before a first look; 1-D where there are neither intervals nor
    # counts before a first look (from_items)
    x = x2 if np.isin(c, (-1, 2)).any() else x2[:, 0]
    truncated = np.isfinite(tl_rows).any() or np.isfinite(tr_rows).any()
    try:
        return handle_xicn(
            x,
            np.array(np.asarray(data.i).tolist()),
            c,
            n,
            tl=tl_rows if truncated else None,
            tr=tr_rows if truncated else None,
            unit=data.unit,
        )
    except ValueError:
        return None


def _records_follow_arrays(data: Any) -> tuple | None:
    """The rows the records of ``data`` would give, as arrays ``(x, c,
    n, tl, tr)`` (``x`` 2-D), where every item's ItemHistory would accept
    them and give back the same rows; else ``None``.

    The records take more than the arrays in a few places, and the
    checks here follow them: what each item's record holds besides its
    events (event types, time-varying covariates, states, attributes)
    and observation windows, which a record has no place for; ids a
    record refuses (blank text) or could change (a numpy kind other than
    numbers or text, a negative zero); counts that are not whole; an
    interval that is not ``lower < upper``; an end of observation at or
    before entry; a failure at the end of an inspection window that
    counts the failures (conservatively: any item with both); covariates
    a record refuses; and the right-truncation check of the records."""
    if (
        getattr(data, "window_map", None) is not None
        or getattr(data, "e", None) is not None
        or getattr(data, "time_varying_covariates", None) is not None
        or getattr(data, "states", None) is not None
        or getattr(data, "attributes", None) is not None
    ):
        return None
    i = np.asarray(data.i)
    if i.ndim != 1 or i.dtype.kind not in "biufU":
        return None
    try:
        items, first, inverse = np.unique(
            i, return_index=True, return_inverse=True
        )
    except TypeError:
        return None
    inverse = inverse.reshape(-1)
    if i.dtype.kind == "U" and (np.char.strip(items) == "").any():
        return None
    if i.dtype.kind == "f" and np.signbit(i[i == 0]).any():
        return None
    if not _covariates_accepted(data, first):
        return None

    c = np.asarray(data.c)
    if not np.isin(c, (-1, 0, 1, 2)).all():
        return None
    c = c.astype(np.int64)
    n = np.asarray(data.n, dtype=float)
    if not np.isfinite(n).all() or (n % 1).any() or (n < 1).any():
        return None
    n = np.where((c == 0) | (c == 1), 1, n).astype(np.int64)

    x = np.asarray(data.x, dtype=float)
    x2 = np.column_stack([x, x]) if x.ndim == 1 else x.copy()
    point = c != 2
    x2[point, 1] = x2[point, 0]
    if not (x2[~point, 0] < x2[~point, 1]).all():
        return None

    # An item's bounds are its first row's (ItemHistory keeps one each)
    tl_item = np.asarray(data.tl, dtype=float)[first]
    tr_item = np.asarray(data.tr, dtype=float)[first]
    tl_item = np.where(np.isfinite(tl_item), tl_item, -np.inf)
    tr_item = np.where(np.isfinite(tr_item), tr_item, np.inf)
    if not (tl_item < tr_item).all():
        return None
    tl_rows, tr_rows = tl_item[inverse], tr_item[inverse]

    # Nothing at or before entry (an interval may start there)
    early = np.where(c == 2, x2[:, 0] < tl_rows, x2[:, 0] <= tl_rows)
    if early.any():
        return None
    # A failure at the top of an inspection window, which the record
    # counts twice and refuses (inside it, handle_xicn refuses too)
    if np.intersect1d(inverse[c == 0], inverse[c == 2]).size:
        return None
    # Watched to its right truncation age or beyond, an item has a
    # failure by then
    n_items = len(items)
    seen_to = np.full(n_items, -np.inf)
    np.maximum.at(seen_to, inverse, x2[:, 1])
    known = np.where(c == 1, np.inf, np.where(c == 2, x2[:, 1], x2[:, 0]))
    first_known = np.full(n_items, np.inf)
    np.minimum.at(first_known, inverse, known)
    if ((seen_to >= tr_item) & ~(first_known <= tr_item)).any():
        return None
    return x2, c, n, tl_rows, tr_rows


def _covariates_accepted(data: Any, first: np.ndarray) -> bool:
    """Whether each item's covariates (its first row's) are what a record
    takes: numbers, finite or NaN (not recorded), with names, where
    named, that are text, distinct, one per column."""
    Z = getattr(data, "Z", None)
    if Z is None:
        return True
    try:
        Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        return False
    if Z.ndim not in (1, 2) or Z.size == 0 or len(Z) != len(data.x):
        return False
    if np.isinf(Z[first]).any():
        return False
    names = getattr(data, "covariate_names", None)
    if names is None:
        return True
    names = list(names)
    return (
        Z.ndim == 2
        and len(names) == Z.shape[1]
        and len(set(names)) == len(names)
        and all(isinstance(name, str) and name for name in names)
    )


def covariates_dropped(data: Any) -> Any:
    """``data`` with its covariates (``Z`` and their names) set aside and
    everything else as it is: what a fit that ignores the covariates
    reads, for data the records refuse but the recurrent fits take (a
    blank id, an end of observation at entry, a failure at the top of an
    inspection window, observation windows)."""
    out = copy.copy(data)
    out.Z = None
    out.covariate_names = None
    return out
