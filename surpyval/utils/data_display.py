"""How the datasets show themselves (#934): the ``repr`` of a
:class:`~surpyval.utils.surpyval_data.SurpyvalData` or an
:class:`~surpyval.utils.event_history_data.EventHistoryData` is its size
and a table of its first rows, with its ids, truncation, event types and
covariates where it has them; ``summary()`` describes all of it, with
counts of items (a row counted ``n`` is ``n`` items); ``_repr_html_``
is the same table for a notebook.
"""

from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any

import numpy as np

# The rows a repr shows before "... k more rows".
REPR_ROWS = 10


def plain(value: Any) -> Any:
    """A numpy scalar as the plain Python value it holds."""
    return value.item() if isinstance(value, np.generic) else value


def time_text(value: Any) -> str:
    """A time, count or ``[start, end]`` pair as the repr shows it."""
    if np.ndim(value):
        return "[" + ", ".join(time_text(v) for v in value) + "]"
    value = plain(value)
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return str(value)


def plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def columns_of(data: Any, shown: int, recurrent: bool) -> dict:
    """The columns of a dataset's table, each a list of ``shown`` texts:
    the item id (where there are ids), the time, the censoring flag, the
    count, the truncation bounds where any is finite, the event type and
    the covariates."""
    x = np.asarray(data.x)
    if x.ndim == 2 and not (x[:, 0] != x[:, 1]).any():
        x = x[:, 0]
    columns: dict = {}
    if recurrent:
        columns["i"] = [repr(plain(v)) for v in data.i[:shown]]
    elif getattr(data, "ids", None) is not None:
        columns["id"] = [repr(plain(v)) for v in data.ids[:shown]]
    columns["x"] = [time_text(v) for v in x[:shown]]
    columns["c"] = [str(int(v)) for v in data.c[:shown]]
    columns["n"] = [time_text(v) for v in data.n[:shown]]
    if getattr(data, "w", None) is not None:
        columns["w"] = [time_text(v) for v in data.w[:shown]]
    tl, tr = np.asarray(data.tl, dtype=float), np.asarray(data.tr, dtype=float)
    if np.isfinite(tl).any():
        columns["tl"] = [time_text(v) for v in tl[:shown]]
    if np.isfinite(tr).any():
        columns["tr"] = [time_text(v) for v in tr[:shown]]
    e = getattr(data, "e", None)
    if e is not None:
        columns["e"] = [repr(plain(v)) for v in e[:shown]]
    Z = getattr(data, "Z", None)
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        names = getattr(data, "covariate_names", None) or [
            f"Z{j}" for j in range(Z.shape[1])
        ]
        for j, name in enumerate(names):
            columns[name] = [time_text(v) for v in Z[:shown, j]]
    markers = getattr(data, "markers", None)
    if markers is not None and not recurrent:
        keys = data.ids if data.ids is not None else range(len(data.x))
        columns["markers"] = [
            plural(len(markers.get(key, {})), "reading")
            for key in list(keys)[:shown]
        ]
    attributes = getattr(data, "attributes", None)
    if attributes:
        if recurrent:
            by_item = dict(zip(data.ids, attributes))
            rows = [by_item.get(plain(v)) for v in data.i[:shown]]
        else:
            rows = attributes[:shown]
        names = list(dict.fromkeys(k for a in attributes if a for k in a))
        for name in names:
            columns[name] = [
                repr(None if a is None else a.get(name)) for a in rows
            ]
    return columns


def heading(data: Any, recurrent: bool) -> str:
    """``"SurpyvalData: 4 items, 3 rows"``: the number of items (the rows
    weighted by their counts, for lifetimes; the distinct ids, for
    histories) and of rows."""
    name = type(data).__name__
    if recurrent:
        items = len(data.ids)
    else:
        items = int(np.asarray(data.n).sum())
    head = f"{name}: {plural(items, 'item')}, {plural(len(data.x), 'row')}"
    unit = getattr(data, "unit", None)
    return head if unit is None else f"{head}, ages in {unit}"


def table_repr(data: Any, recurrent: bool) -> str:
    """The repr: the heading and the first rows as a table; a dataset of
    more rows says how many more."""
    shown = min(len(data.x), REPR_ROWS)
    columns = columns_of(data, shown, recurrent)
    widths = {
        k: max(len(k), *(len(v) for v in vals)) for k, vals in columns.items()
    }
    lines = [
        heading(data, recurrent),
        "  " + "  ".join(k.rjust(widths[k]) for k in columns),
    ]
    for r in range(shown):
        lines.append(
            "  " + "  ".join(columns[k][r].rjust(widths[k]) for k in columns)
        )
    if len(data.x) > shown:
        lines.append(
            f"  ... {plural(len(data.x) - shown, 'more row')}; "
            "summary() describes it all"
        )
    return "\n".join(lines)


def html_repr(data: Any, recurrent: bool) -> str:
    """The same table as HTML, for a notebook."""
    shown = min(len(data.x), REPR_ROWS)
    columns = columns_of(data, shown, recurrent)
    parts = [
        "<div><p><b>",
        escape(heading(data, recurrent)),
        "</b></p><table><thead><tr>",
        "".join(f"<th>{escape(k)}</th>" for k in columns),
        "</tr></thead><tbody>",
    ]
    for r in range(shown):
        parts.append(
            "<tr>"
            + "".join(f"<td>{escape(columns[k][r])}</td>" for k in columns)
            + "</tr>"
        )
    parts.append("</tbody></table>")
    if len(data.x) > shown:
        parts.append(
            f"<p>... {plural(len(data.x) - shown, 'more row')}; "
            "<code>summary()</code> describes it all</p>"
        )
    parts.append("</div>")
    return "".join(parts)


def held_lines(data: Any, recurrent: bool) -> list:
    """The summary lines for what a dataset holds beyond its times:
    truncation, event types, covariates, time-varying covariates and
    states, counting items (a lifetime row counted ``n`` is ``n`` items;
    a history's truncation is the item's) and, for histories, failures
    (a count of ``n`` in an interval is ``n`` failures)."""
    lines = []
    c, n = np.asarray(data.c), np.asarray(data.n)
    tl, tr = np.asarray(data.tl, dtype=float), np.asarray(data.tr, dtype=float)
    if recurrent:
        first, _ = data.item_rows()
        weight = np.zeros(len(c), dtype=int)
        weight[first] = 1
    else:
        weight = n
    if np.isfinite(tl).any() or np.isfinite(tr).any():
        lines.append(
            "  truncation: "
            + ", ".join(
                f"{int(weight[np.isfinite(b)].sum())} {side}"
                for side, b in (("left", tl), ("right", tr))
                if np.isfinite(b).any()
            )
        )
    e = getattr(data, "e", None)
    if e is not None:
        counts: Counter = Counter()
        masked = 0
        for v, flag, k in zip(e, c, n):
            if flag == 1:
                continue
            if v is None:
                masked += int(k)
            else:
                counts[v] += int(k)
        lines.append(
            "  event types: "
            + ", ".join(
                [f"{k} ({v})" for k, v in sorted(counts.items(), key=str)]
                + ([f"not recorded ({masked})"] if masked else [])
            )
        )
    Z = getattr(data, "Z", None)
    if Z is not None:
        Z = np.asarray(Z)
        width = 1 if Z.ndim == 1 else Z.shape[1]
        line = f"  covariates: {width} column{'' if width == 1 else 's'}"
        if getattr(data, "covariate_names", None) is not None:
            line += " (" + ", ".join(data.covariate_names) + ")"
        lines.append(line)
    schedules = getattr(data, "time_varying_covariates", None)
    if schedules is not None:
        kinds = Counter(type(v).__name__ for v in schedules.values())
        lines.append(
            "  time-varying covariates: "
            + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
        )
    markers = getattr(data, "markers", None)
    if markers is not None:
        readings = sum(len(m) for m in markers.values())
        read = sum(1 for m in markers.values() if m)
        lines.append(
            f"  markers: {plural(readings, 'reading')} on "
            f"{plural(read, 'item')}"
        )
    states = getattr(data, "states", None)
    if states is not None:
        found = sorted(
            {st for _, changes in states.values() for *_, st in changes}
            | {initial for initial, _ in states.values()},
            key=str,
        )
        lines.append("  states: " + ", ".join(map(str, found)))
    attributes = getattr(data, "attributes", None)
    if attributes:
        names = list(dict.fromkeys(k for a in attributes if a for k in a))
        lines.append("  attributes: " + ", ".join(names))
    return lines


# The attributes a dataset's ``dir()`` leaves out: the arrays the
# likelihoods read, split by kind of observation, and the pieces of the
# recurrent likelihoods, which are there but are not the API a user
# reaches for (#934); and the deprecated names.
LIFETIME_INTERNALS = frozenset(
    {
        "mask_o",
        "mask_r",
        "mask_l",
        "mask_i",
        "truncated_mask",
        "x_o",
        "n_o",
        "x_r",
        "n_r",
        "x_l",
        "n_l",
        "x_il",
        "x_ir",
        "n_i",
        "x_tl",
        "x_tr",
        "n_t",
        "tl_unique",
        "tr_unique",
        "n_t_unique",
        "Z_o",
        "Z_r",
        "Z_l",
        "Z_i",
        "Z_t",
        "x_min",
        "x_max",
        "add_covariates",
    }
)
HISTORY_INTERNALS = frozenset(
    {
        "get_previous_x",
        "split_for_nhpp_likelihood",
        "selection_terms",
        "item_rows",
        "item_observation_windows",
        "get_interarrival_times",
        "get_events_for_item",
        "to_cause_specific_xrd",
        "interarrival_times",
        "midpoints",
        "xrd",
        "window_map",
        "observation_windows",
        "items",
        "get_times_to_first_events",
    }
)


def public_dir(data: Any, internals: frozenset) -> list:
    """``dir(data)`` without the ``internals`` and the private names."""
    return sorted(
        name
        for name in object.__dir__(data)
        if not name.startswith("_") and name not in internals
    )
