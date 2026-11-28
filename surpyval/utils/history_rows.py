"""Histories as rows: a :class:`~surpyval.utils.surpyval_data.SurpyvalData`
with units ``i`` built from items' histories, and read back as histories.

``SurpyvalData`` with ``i`` is the form for several rows per unit, the one
every recurrent fit takes. This module is what lets it be built from, and
reshaped as, the histories of repeated events:

- :func:`rows_from_histories`, the rows of
  :class:`~surpyval.utils.item_history.ItemHistory` records
  (``SurpyvalData.from_items``, ``SurpyvalData.from_histories``);
- :func:`rows_from_event_history`, the rows of an
  :class:`~surpyval.utils.event_history_data.EventHistoryData`
  (``EventHistoryData.to_surpyval_data``), checked to read back as the
  same data;
- :func:`event_history_from_rows`, the other way, by the reading of
  :mod:`surpyval.utils.recurrent_data` (the one a recurrent ``fit`` makes
  of the rows), for ``SurpyvalData.to_first_events``, ``to_gaps`` and
  ``to_histories``.

How a history is written as rows: one row per event in time order (a
failure at its age, ``c = 0``; failures counted between two inspections,
``c = 2``, or before the first look, ``c = -1``, on one row with their
count ``n``), then the end of its observation (``c = 1``) where it has
one; the item's late entry ``tl`` on its first row, its right truncation
``tr`` on every row, and its covariates, attributes and id on each of its
rows. Watched in separate stretches (observation windows), each stretch
starts with its ``tl`` and ends with a ``c = 1`` row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.utils.warnings import (
    MaskedCauseWarning,
    ignoring_warnings,
    named,
)

# The arguments of ``handle_xicn`` the rows are read into.
_ARRAYS = {"x", "i", "c", "n", "tl", "tr", "windows", "Z", "e"}


def _row_key(row: tuple) -> tuple:
    """Time order within a history: by the row's upper end, the end of
    observation after any event at its age (the order of the notation's
    ``Item``)."""
    lower, upper, c = row[0], row[1], row[2]
    return (upper, c == 1, lower)


def _ordered_rows(history: Any) -> list:
    """``(lower, upper, c, n, event type)`` per row of an ItemHistory, in
    time order."""
    xs, cs, ns, es = history._rows()
    rows = [
        (float(x[0]), float(x[1]), int(c), n, e)
        for x, c, n, e in zip(xs, cs, ns, es)
    ]
    return sorted(rows, key=_row_key)


def _refuse_states(histories: list, what: str) -> None:
    held = [h.id for h in histories if h.has_states]
    if held:
        raise ValueError(
            f"{what}: changes of state (items {named(held)}), which a "
            "SurpyvalData holds as rows, one per stay in a state "
            "(SurpyvalData(x, s=state, e=state_entered, i=i)), not as a "
            "history of repeated events, have no place in these rows: "
            'build those rows, or drop them first (data.without("states"), '
            "or the records without their states)."
        )


def rows_from_histories(
    histories: list,
    unit: Any = None,
    what: str = "SurpyvalData.from_items",
    warn: bool = True,
) -> Any:
    """The :class:`~surpyval.utils.surpyval_data.SurpyvalData` of
    ``histories``, a list of
    :class:`~surpyval.utils.item_history.ItemHistory` records: their rows
    as the module describes, in the order of the records. Covariates and
    time-varying covariates, given for one item, are needed for every
    item; changes of state are refused (they are rows of another kind).
    The terminal flag ``a`` is 0 on every row (a history's end is the end
    of its observation); where no item has more than one row the data has
    no ``a``, as lifetime data has none."""
    from surpyval.utils.event_history_data import _all_or_none, _masked_items
    from surpyval.utils.item_history import (
        check_item_ids,
        covariate_names_of,
        covariate_row_in,
        units_of,
    )
    from surpyval.utils.surpyval_data import SurpyvalData
    from surpyval.utils.warnings import warn_masked

    check_item_ids([h.id for h in histories], "ItemHistory")
    unit = units_of(histories, unit, "items' ages")
    _refuse_states(histories, what)
    static = _all_or_none(histories, "covariates", lambda h: h.covariates)
    names = covariate_names_of(histories)
    varying = _all_or_none(
        histories,
        "time_varying_covariates",
        lambda h: (
            None
            if h.time_varying_covariates is None
            else [0.0] * h.time_varying_covariates.p
        ),
    )
    typed = any(h.has_event_types for h in histories)
    if typed and warn:
        masked = _masked_items(histories)
        if masked:
            warn_masked(len(masked), "items " + named(masked))
    lower: list = []
    upper: list = []
    c: list = []
    n: list = []
    i: list = []
    e: list = []
    tl: list = []
    tr: list = []
    Z: list = []
    attributes: list = []
    several = False
    for history in histories:
        rows = _ordered_rows(history)
        several = several or len(rows) > 1
        for k, (lo, hi, ck, nk, ek) in enumerate(rows):
            lower.append(lo)
            upper.append(hi)
            c.append(ck)
            n.append(nk)
            e.append(ek)
            i.append(history.id)
            entry = history.left_truncation
            tl.append(-np.inf if k or entry is None else entry)
            tr.append(
                np.inf
                if history.right_truncation is None
                else history.right_truncation
            )
        if static:
            Z += [covariate_row_in(history, names)] * len(rows)
        attributes += [history.attributes] * len(rows)
    extra: dict[str, Any] = {}
    if np.isfinite(tl).any() or np.isfinite(tr).any():
        extra.update(tl=tl, tr=tr)
    if static:
        extra.update(Z=np.array(Z, dtype=float), covariate_names=names)
    if typed:
        types = np.empty(len(e), dtype=object)
        types[:] = e
        extra["e"] = types
    if several:
        extra["a"] = [0] * len(c)
    x: Any = lower
    if 2 in c or -1 in c:
        # Counts at inspections: [lower, upper] rows, as handle_xicn has
        x = np.column_stack([lower, upper])
    with ignoring_warnings(MaskedCauseWarning):
        # Warned of above, by item
        data = SurpyvalData(x, c, n, i=i, unit=unit, **extra)
    if any(a is not None for a in attributes):
        data.attributes = attributes
    if varying:
        data.time_varying_covariates = {
            h.id: h.time_varying_covariates for h in histories
        }
    return data


def _window_rows(data: Any) -> Any:
    """The rows of an ``EventHistoryData`` with observation windows: each
    window of an item a stretch of its rows, from its start (``tl`` on the
    stretch's first row) to its end (a ``c = 1`` row). (Windows come with
    exact failure ages only: no covariates, event types, counts at
    inspections or truncation.)"""
    from surpyval.utils.surpyval_data import SurpyvalData, _plain

    held = [
        name
        for name, value in (
            ("covariates", data.Z),
            ("event types", data.e),
            ("time-varying covariates", data.time_varying_covariates),
            ("states", data.states),
        )
        if value is not None
    ]
    if held or np.ndim(data.x) != 1:
        raise ValueError(
            "to_surpyval_data: this data has observation windows and "
            + (" and ".join(held) or "counts at inspections")
            + ", which rows with gaps have no place for together: use its "
            "arrays (x, i, c, n, ...)."
        )
    x = np.asarray(data.x, dtype=float)
    # Each item's windows as the data was given them (an item watched
    # with no failure has windows and no event row), each with the rows
    # of its synthetic single-window item.
    parts = {
        (_plain(item), float(start), float(end)): part
        for part, (item, (start, end)) in data.window_map.items()
    }
    rows_of: dict = {}
    for r, part in enumerate(np.asarray(data.i).tolist()):
        if data.c[r] != 1:
            rows_of.setdefault(part, []).append(r)
    out: dict = {k: [] for k in ("x", "c", "n", "i", "tl", "at")}
    attributes = None
    if data.attributes is not None:
        attributes = dict(
            zip(
                dict.fromkeys(
                    _plain(item) for item, _ in data.window_map.values()
                ),
                data.attributes,
            )
        )
    for item, windows in data.observation_windows.items():
        key = _plain(item)
        for start, end in sorted((float(a), float(b)) for a, b in windows):
            rows = rows_of.get(parts.get((key, start, end)), [])
            ages = sorted(float(x[r]) for r in rows)
            count = len(ages) + 1
            out["x"] += [*ages, end]
            out["c"] += [0] * len(ages) + [1]
            out["n"] += [1] * count
            out["tl"] += [start] + [-np.inf] * len(ages)
            out["i"] += [key] * count
            if attributes is not None:
                out["at"] += [attributes.get(key)] * count
    rows_data = SurpyvalData(
        out["x"],
        out["c"],
        out["n"],
        i=out["i"],
        tl=out["tl"],
        a=[0] * len(out["c"]),
        unit=data.unit,
    )
    if attributes is not None:
        rows_data.attributes = out["at"]
    return rows_data


def rows_from_event_history(data: Any) -> Any:
    """The :class:`~surpyval.utils.surpyval_data.SurpyvalData` of an
    ``EventHistoryData``: the same histories as rows with units ``i``.
    The rows are read back (as a recurrent ``fit`` reads them) and
    compared with ``data`` before they are returned; data whose rows would
    not read back as it is refused, naming what to use."""
    what = "EventHistoryData.to_surpyval_data"
    with ignoring_warnings(MaskedCauseWarning):
        # The data warned of its masked causes when it was built
        if data.window_map is not None:
            out = _window_rows(data)
        elif np.any(np.asarray(data.n) == 0):
            # Inspections that found nothing: an item's record has no place
            # for them, so the rows are written from the arrays
            out = _rows_with_zero_counts(data, what)
        else:
            out = rows_from_histories(
                data.to_items(), unit=data.unit, what=what, warn=False
            )
        try:
            back = event_history_from_rows(out, what)
        except (TypeError, ValueError) as error:
            raise ValueError(_NOT_THE_SAME + f" ({error})") from error
        if np.ndim(data.x) == 1 and np.ndim(back.x) == 2:
            # Counts before a first look given as one age each: the rows
            # read them as [age, age], the same ages
            if (back.x[:, 0] == back.x[:, 1]).all():
                back.x = back.x[:, 1]
    if not data.equals(back, rtol=0):
        raise ValueError(_NOT_THE_SAME)
    return out


def _rows_with_zero_counts(data: Any, what: str) -> Any:
    """The rows of an ``EventHistoryData`` that holds inspections finding
    no events (``n = 0``), written from its arrays in the order
    :func:`rows_from_histories` uses (an item's rows in time order, its
    late entry ``tl`` on the first). A count of 0 is a real finding, "looked
    and saw nothing", and stays a row with ``n = 0`` (a ``SurpyvalData``
    with units holds it on a counted row)."""
    from surpyval.utils.surpyval_data import SurpyvalData

    if data.states is not None:
        raise ValueError(
            f"{what}: changes of state, which a SurpyvalData holds as rows "
            "of another kind, have no place in these rows: build those "
            'rows, or drop them first (data.without("states")).'
        )
    x = np.asarray(data.x, dtype=float)
    lower = x[:, 0] if x.ndim == 2 else x
    upper = x[:, 1] if x.ndim == 2 else x
    c = np.asarray(data.c)
    ids = np.asarray(data.i)
    # Items in the order of ``ids``, each item's rows in time order
    order: list = []
    for item in data.ids:
        rows = np.flatnonzero(ids == item)
        keys = [_row_key((lower[r], upper[r], c[r])) for r in rows]
        order += [
            rows[k] for k in sorted(range(len(rows)), key=keys.__getitem__)
        ]
    order_a = np.asarray(order)
    ordered_ids = ids[order_a]
    first = np.r_[True, ordered_ids[1:] != ordered_ids[:-1]]
    tl = np.where(first, np.asarray(data.tl)[order_a], -np.inf)
    tr = np.asarray(data.tr)[order_a]
    extra: dict[str, Any] = {}
    if np.isfinite(tl).any() or np.isfinite(tr).any():
        extra.update(tl=tl, tr=tr)
    if data.Z is not None:
        extra.update(
            Z=np.asarray(data.Z, dtype=float)[order_a],
            covariate_names=data.covariate_names,
        )
    if data.e is not None:
        extra["e"] = np.asarray(data.e, dtype=object)[order_a]
    if len(order_a) > len(data.ids):
        extra["a"] = [0] * len(order_a)
    out = SurpyvalData(
        x[order_a],
        c[order_a],
        np.asarray(data.n)[order_a],
        i=ordered_ids.tolist(),
        unit=data.unit,
        **extra,
    )
    if data.attributes is not None:
        by_item = dict(zip(data.ids, data.attributes))
        out.attributes = [by_item[item] for item in ordered_ids.tolist()]
    if data.time_varying_covariates is not None:
        out.time_varying_covariates = dict(data.time_varying_covariates)
    return out


_NOT_THE_SAME = (
    "to_surpyval_data: these histories, written as rows with units i, "
    "would not read back as the same data, so no rows are given: use "
    "data.to_items() for its records, or its arrays (x, i, c, n, ...)."
)


def _first_per_unit(ids: list, values: list, order: list) -> list:
    """Each unit's first value of ``values`` (one per row), in ``order``."""
    first: dict = {}
    for unit, value in zip(ids, values):
        first.setdefault(unit, value)
    return [first[unit] for unit in order]


def event_history_from_rows(data: Any, what: str) -> Any:
    """The ``EventHistoryData`` of a ``SurpyvalData`` with units ``i``:
    each unit's rows read as its history, as a recurrent ``fit`` reads
    them (:mod:`surpyval.utils.recurrent_data`). ``what`` names the caller
    in a refusal. Refused: data without ``i``, in more than one state,
    with weights or measurements, or whose covariates change between a
    unit's rows (none of which a history of repeated events holds)."""
    from surpyval.utils.recurrent_data import _Layout
    from surpyval.utils.recurrent_utils import handle_xicn
    from surpyval.utils.surpyval_data import _distinct_states, _plain

    if data.i is None:
        raise ValueError(
            f"{what} reads each unit's rows as its history, and this data "
            "has no units: every row is a unit of its own (a lifetime). "
            "Give each row's unit, SurpyvalData(x, c, ..., i=i)."
        )
    held = [
        f"{name} ({how})"
        for name, how, there in (
            (
                "states",
                'data[data.s == state] keeps one, data.without("states") '
                "drops them",
                len(_distinct_states(data)) > 1,
            ),
            ("weights", "build the data without w", data.w is not None),
            (
                "measurements",
                'data.without("measurements") drops them',
                data.m is not None,
            ),
        )
        if there
    ]
    if held:
        raise ValueError(
            f"{what} reads each unit's rows as its history of repeated "
            "events, which has no place for the data's "
            + "; ".join(held)
            + "."
        )
    layout = _Layout(data)
    if layout.varying:
        raise ValueError(
            f"{what}: the covariates of units {named(layout.varying)} "
            "change between their rows, and a history's covariates are "
            'fixed: drop them first, data.without("covariates"), or keep '
            "the rows as they are (the proportional-intensity models fit "
            "them by fit_tvc)."
        )
    arrays = layout.arrays(data, _ARRAYS)
    x = np.asarray(arrays["x"], dtype=float)
    if x.ndim == 1 and (np.asarray(arrays["c"]) == -1).any():
        # Counts before a first look: [age, age] rows, the form the
        # histories' own data has them in
        arrays["x"] = np.column_stack([x, x])
    with ignoring_warnings(MaskedCauseWarning):
        # The data warned of its masked causes when it was built
        history = handle_xicn(**arrays, unit=data.unit)
    if data.Z is not None:
        history.covariate_names = data.covariate_names
    ids = list(data.ids)
    order = [_plain(k) for k in history.ids]
    if history.window_map is not None:
        order = list(
            dict.fromkeys(
                _plain(item) for item, _ in history.window_map.values()
            )
        )
    if data.attributes is not None:
        history.attributes = _first_per_unit(ids, data.attributes, order)
    if data.time_varying_covariates is not None:
        history.time_varying_covariates = dict(data.time_varying_covariates)
    return history
