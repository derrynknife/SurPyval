"""DataFrames in and out of the data classes (#936): a lifetime dataset
from a table with one row per item (:func:`lifetime_from_frame`), an
event history dataset from a log with one row per event
(:func:`history_from_frame`), and each dataset back as a table
(:func:`lifetime_to_frame`, :func:`history_to_frame`).

The column arguments are named after the ``fit`` argument each fills,
with a ``_col`` suffix, as every DataFrame entry point names them
(principle 21): ``x_col``, ``c_col``, ``n_col``, ``xl_col``, ``xr_col``,
``tl_col``, ``tr_col``, ``i_col``, ``e_col`` and ``Z_cols``. A
spreadsheet's status column (1 or True where the item failed) is
``event_col``, the opposite coding of ``c``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.utils import is_missing_event
from surpyval.utils.dataframe import (
    check_columns,
    frame_column,
    require_frame,
)


def _flags(df: Any, c_col: Any, event_col: Any) -> "np.ndarray | None":
    """The censoring flags: ``c_col`` as they are, or ``event_col`` (1 or
    True where the item failed, 0 or False where it was still running)
    turned round."""
    if c_col is not None and event_col is not None:
        raise ValueError(
            "Give c_col (the censoring flags: 0 observed, 1 right "
            "censored, -1 left censored, 2 interval censored) or event_col "
            "(1 or True where the item failed), not both."
        )
    if c_col is not None:
        return frame_column(df, c_col, "c_col")
    if event_col is None:
        return None
    column = frame_column(df, event_col, "event_col")
    values = np.array([None if is_missing_event(v) else v for v in column])
    if any(v is None for v in values):
        raise ValueError(
            f"event_col={event_col!r} has a blank cell: every row says "
            "whether its item failed (1 or True) or was still running (0 "
            "or False)."
        )
    allowed = {True, False, 0, 1, 0.0, 1.0}
    bad = sorted({repr(v) for v in values if v not in allowed})
    if bad:
        raise ValueError(
            f"event_col={event_col!r} must hold 1 or True where the item "
            "failed and 0 or False where it was still running; got "
            + ", ".join(bad[:5])
            + ". For a column of flags in the xcnt coding (0 observed, 1 "
            "right censored, -1, 2), give c_col."
        )
    return 1 - np.array(values, dtype=float).astype(int)


def _times(
    df: Any, x_col: Any, xl_col: Any, xr_col: Any
) -> tuple[np.ndarray, bool]:
    """The times: one column, or ``[lower, upper]`` from two, where a
    blank upper bound means the item had not failed when last seen
    (right censored at the lower bound) and a blank lower bound that it
    had already failed at the first look (left censored at the upper
    bound)."""
    if x_col is not None:
        if xl_col is not None or xr_col is not None:
            raise ValueError("Give x_col, or xl_col and xr_col, not both.")
        x = frame_column(df, x_col, "x_col", time=True).astype(float)
        if np.isnan(x).any():
            rows = np.flatnonzero(np.isnan(x)).tolist()
            raise ValueError(
                f"x_col={x_col!r} is blank on row"
                f"{'s' if len(rows) > 1 else ''} "
                + ", ".join(str(r) for r in rows[:5])
                + (" ..." if len(rows) > 5 else "")
                + ": every item needs a time. For an item known only to "
                "have failed by an age, or to have been working at one, "
                "give xl_col and xr_col with the unknown end blank."
            )
        return x, False
    if xl_col is None or xr_col is None:
        raise ValueError(
            "Give the times: x_col (one time per row), or xl_col and xr_col "
            "(the age last seen working and the age first seen failed; "
            "either may be blank)."
        )
    lower = frame_column(df, xl_col, "xl_col", time=True).astype(float)
    upper = frame_column(df, xr_col, "xr_col", time=True).astype(float)
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    both = ~np.isfinite(lower) & ~np.isfinite(upper)
    if both.any():
        rows = np.flatnonzero(both).tolist()
        raise ValueError(
            f"Row{'s' if len(rows) > 1 else ''} "
            + ", ".join(str(r) for r in rows[:5])
            + f" have both {xl_col!r} and {xr_col!r} blank: nothing is "
            "known of the item."
        )
    return np.column_stack([lower, upper]), True


def _bounds_by_flag(
    x: np.ndarray, c: Any, xl_col: Any, xr_col: Any, flag_col: Any
) -> np.ndarray:
    """``[lower, upper]`` rows given with flags: a blank bound on the
    censored side of a right (1) or left (-1) censored row is that row's
    known bound (the item last seen working at ``xl_col``, or found
    failed at ``xr_col``); a blank bound on an observed (0) or interval
    (2) row is refused naming the row."""
    c = np.asarray(c, dtype=float)
    lower, upper = x[:, 0].copy(), x[:, 1].copy()
    for row, (lo, hi, flag) in enumerate(zip(lower, upper, c)):
        blank_lo, blank_hi = not np.isfinite(lo), not np.isfinite(hi)
        if flag == 1 and blank_hi and not blank_lo:
            upper[row] = lo
        elif flag == -1 and blank_lo and not blank_hi:
            lower[row] = hi
        elif blank_lo or blank_hi:
            raise ValueError(
                f"Row {row}: {flag_col!r} says {int(flag)} but "
                f"{xl_col if blank_lo else xr_col!r} is blank; a blank "
                "bound is the unknown end of a right censored row "
                f"({xr_col!r} blank, flag 1) or a left censored one "
                f"({xl_col!r} blank, flag -1)."
            )
    return np.column_stack([lower, upper])


def _bound(df: Any, value: Any, arg: str, blanks: bool = False) -> Any:
    """A truncation column (every cell filled, unless ``blanks`` may be
    left out), or a number for every row, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        column = frame_column(df, value, arg, time=True).astype(float)
        if np.isnan(column).any() and not blanks:
            rows = np.flatnonzero(np.isnan(column)).tolist()
            raise ValueError(
                f"{arg}={value!r} is blank on row"
                f"{'s' if len(rows) > 1 else ''} "
                + ", ".join(str(r) for r in rows[:5])
                + (" ..." if len(rows) > 5 else "")
                + ": give every row its bound, -inf or inf for none."
            )
        return column
    if np.ndim(value) == 0:
        return float(value)
    raise ValueError(f"{arg} must be a column name or one number.")


def _counts(df: Any, n_col: Any) -> Any:
    """The counts column, every cell filled."""
    if n_col is None:
        return None
    column = frame_column(df, n_col, "n_col")
    values = np.asarray(column, dtype=float)
    if np.isnan(values).any():
        rows = np.flatnonzero(np.isnan(values)).tolist()
        raise ValueError(
            f"n_col={n_col!r} is blank on row"
            f"{'s' if len(rows) > 1 else ''} "
            + ", ".join(str(r) for r in rows[:5])
            + ": give every row its count (1 for one item)."
        )
    return column


def _covariates(
    df: Any, Z_cols: Any, purpose: str
) -> tuple["np.ndarray | None", "list | None"]:
    """The covariate columns as a float matrix and their names."""
    if Z_cols is None:
        return None, None
    names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
    if not names:
        raise ValueError("Z_cols names no column: give at least one.")
    check_columns(df, Z_cols=names)
    columns = []
    for name in names:
        column = df[name]
        try:
            columns.append(np.asarray(column, dtype=float))
        except (TypeError, ValueError):
            raise ValueError(
                f"Covariate column {name!r} is not numeric. A regression "
                "takes numbers: encode it (pd.get_dummies(df[[col]], "
                "drop_first=True)) and name the new columns in Z_cols, or "
                f"fit {purpose} with formula=, which codes a categorical "
                "column; a label that is not fitted (a site, a batch) is "
                "an attribute: attribute_cols=."
            ) from None
    Z = np.column_stack(columns)
    if np.isinf(Z).any():
        raise ValueError(
            "Covariates must be finite numbers (a blank cell is one not "
            "recorded)."
        )
    return Z, [str(n) for n in names]


def _attributes(df: Any, attribute_cols: Any) -> "list | None":
    """Each row's attributes, ``{name: value}``, from the columns."""
    if attribute_cols is None:
        return None
    names = (
        [attribute_cols]
        if isinstance(attribute_cols, str)
        else list(attribute_cols)
    )
    check_columns(df, attribute_cols=names)
    rows = [
        {
            str(name): (None if is_missing_event(v) else _plain(v))
            for name, v in zip(names, values)
        }
        for values in zip(*(df[name].tolist() for name in names))
    ]
    return rows


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _event_types(df: Any, e_col: Any) -> "np.ndarray | None":
    """The event type of each row, a blank cell None."""
    if e_col is None:
        return None
    column = frame_column(df, e_col, "e_col")
    out = np.empty(len(column), dtype=object)
    out[:] = [None if is_missing_event(v) else _plain(v) for v in column]
    return out


def lifetime_from_frame(
    cls: Any,
    df: Any,
    *,
    x_col: Any = None,
    xl_col: Any = None,
    xr_col: Any = None,
    c_col: Any = None,
    event_col: Any = None,
    n_col: Any = None,
    tl_col: Any = None,
    tr_col: Any = None,
    i_col: Any = None,
    e_col: Any = None,
    Z_cols: Any = None,
    attribute_cols: Any = None,
    unit: Any = None,
) -> Any:
    """A ``SurpyvalData`` from a DataFrame with one row per item (or per
    group of identical items, counted ``n_col``); see
    ``SurpyvalData.from_dataframe``."""
    from surpyval.utils.units import normalise_unit

    df = require_frame(df)
    if len(df) == 0:
        raise ValueError("The DataFrame has no rows.")
    x, intervals = _times(df, x_col, xl_col, xr_col)
    c = _flags(df, c_col, event_col)
    if intervals and c is not None:
        x = _bounds_by_flag(x, c, xl_col, xr_col, c_col or event_col)
    n = _counts(df, n_col)
    ids = None
    if i_col is not None:
        ids = [_plain(v) for v in frame_column(df, i_col, "i_col")]
    Z, names = _covariates(df, Z_cols, "a regression's fit_from_df(df, ...)")
    data = cls(
        x,
        c,
        n,
        tl=_bound(df, tl_col, "tl_col"),
        tr=_bound(df, tr_col, "tr_col"),
        Z=Z,
        e=_event_types(df, e_col),
        covariate_names=names,
        ids=ids,
    )
    attributes = _attributes(df, attribute_cols)
    if attributes is not None:
        data.attributes = attributes
    canonical = normalise_unit(unit)
    if canonical is not None:
        data.unit = canonical
    return data


def lifetime_to_frame(data: Any) -> Any:
    """The rows of a ``SurpyvalData`` as a DataFrame: ``id`` (where it has
    ids), ``x`` (or ``xl`` and ``xr`` where any row is an interval),
    ``c``, ``n``, ``tl`` and ``tr`` (where any bound is finite), ``e``
    (where it has event types), the covariates by name (``Z0``, ``Z1``,
    ... unnamed) and the attributes."""
    import pandas as pd

    if data.time_varying_covariates is not None:
        raise ValueError(
            "This data has time-varying covariates, which a table with one "
            "row per item cannot hold: use data.without("
            "'time_varying_covariates'), or keep the records "
            "(data.to_items()) or the JSON (data.to_json())."
        )
    columns: dict = {}
    if data.ids is not None:
        columns["id"] = list(data.ids)
    x = np.asarray(data.x, dtype=float)
    if x.ndim == 2:
        columns["xl"] = x[:, 0]
        columns["xr"] = x[:, 1]
    else:
        columns["x"] = x
    columns["c"] = np.asarray(data.c)
    columns["n"] = np.asarray(data.n)
    tl, tr = np.asarray(data.tl, dtype=float), np.asarray(data.tr, dtype=float)
    if np.isfinite(tl).any():
        columns["tl"] = tl
    if np.isfinite(tr).any():
        columns["tr"] = tr
    if data.e is not None:
        columns["e"] = list(data.e)
    _covariate_columns(columns, data)
    _attribute_columns(columns, getattr(data, "attributes", None))
    return pd.DataFrame(columns)


def _covariate_columns(columns: dict, data: Any) -> None:
    Z = getattr(data, "Z", None)
    if Z is None:
        return
    Z = np.asarray(Z, dtype=float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    names = getattr(data, "covariate_names", None) or [
        f"Z{j}" for j in range(Z.shape[1])
    ]
    for j, name in enumerate(names):
        columns[name] = Z[:, j]


def _attribute_columns(columns: dict, attributes: Any) -> None:
    if not attributes:
        return
    names = list(dict.fromkeys(k for row in attributes if row for k in row))
    for name in names:
        columns[name] = [
            None if row is None else row.get(name) for row in attributes
        ]


def history_from_frame(
    cls: Any,
    df: Any,
    *,
    i_col: Any,
    x_col: Any = None,
    xl_col: Any = None,
    xr_col: Any = None,
    c_col: Any = None,
    n_col: Any = None,
    e_col: Any = None,
    tl_col: Any = None,
    tr_col: Any = None,
    end_col: Any = None,
    ends: Any = None,
    Z_cols: Any = None,
    attribute_cols: Any = None,
    unit: Any = None,
) -> Any:
    """An ``EventHistoryData`` from a log with one row per event; see
    ``EventHistoryData.from_dataframe``."""
    from surpyval.utils.item_history import ItemHistory, item_id
    from surpyval.utils.units import normalise_unit

    df = require_frame(df)
    if len(df) == 0:
        raise ValueError("The DataFrame has no rows.")
    check_columns(df, i_col=i_col)
    ids = [item_id(v) for v in df[i_col].tolist()]
    if x_col is None and (xl_col is None or xr_col is None):
        raise ValueError(
            "Give the ages: x_col (each event's age), or xl_col and xr_col "
            "(the inspections between which failures were counted; equal "
            "for an exact event)."
        )
    if x_col is not None:
        if xl_col is not None or xr_col is not None:
            raise ValueError("Give x_col, or xl_col and xr_col, not both.")
        ages = frame_column(df, x_col, "x_col", time=True).astype(float)
        lower = upper = ages
    else:
        lower = frame_column(df, xl_col, "xl_col", time=True).astype(float)
        upper = frame_column(df, xr_col, "xr_col", time=True).astype(float)
    for name, column in (("x_col", lower), ("xr_col", upper)):
        if np.isnan(column).any():
            rows = np.flatnonzero(np.isnan(column)).tolist()
            raise ValueError(
                f"The age column is blank on row"
                f"{'s' if len(rows) > 1 else ''} "
                + ", ".join(str(r) for r in rows[:5])
                + (" ..." if len(rows) > 5 else "")
                + ": every event needs its age."
            )
    if c_col is None:
        c = np.where(lower == upper, 0, 2)
    else:
        c = np.asarray(frame_column(df, c_col, "c_col"), dtype=float)
        if np.isnan(c).any() or not np.isin(c, (-1, 0, 1, 2)).all():
            raise ValueError(
                f"c_col={c_col!r} must hold 0 (an event), 1 (the end of "
                "observation), -1 (failures before a first look) or 2 "
                "(failures counted between inspections) on every row."
            )
        c = c.astype(int)
    n = np.ones(len(df), dtype=int) if n_col is None else _counts(df, n_col)
    if n_col is not None:
        n = np.asarray(n, dtype=float)
        if (n < 1).any() or (n % 1).any():
            raise ValueError(
                f"n_col={n_col!r} must hold whole numbers, 1 or more."
            )
        n = n.astype(int)
    e = _event_types(df, e_col)
    Z, names = _covariates(df, Z_cols, "a proportional intensity model")
    attributes = _attributes(df, attribute_cols)
    tl = _bound(df, tl_col, "tl_col")
    tr = _bound(df, tr_col, "tr_col")
    end = _bound(df, end_col, "end_col", blanks=True)
    given_ends = _ends(ends, ids)
    order = list(dict.fromkeys(ids))
    rows_of: dict = {item: [] for item in order}
    for k, item in enumerate(ids):
        rows_of[item].append(k)
    histories = []
    for item in order + [i for i in given_ends if i not in rows_of]:
        rows = rows_of.get(item, [])
        histories.append(
            _history(
                ItemHistory,
                item,
                rows,
                lower,
                upper,
                c,
                n,
                e,
                Z,
                names,
                attributes,
                tl,
                tr,
                end,
                given_ends.get(item),
            )
        )
    data = cls.from_items(histories)
    canonical = normalise_unit(unit)
    if canonical is not None:
        data.unit = canonical
    return data


def _ends(ends: Any, ids: list) -> dict:
    """``ends`` as ``{item: age}``: a dict, a Series, or None."""
    from surpyval.utils.item_history import item_id

    if ends is None:
        return {}
    if type(ends).__name__ == "Series":
        ends = dict(ends.items())
    if not isinstance(ends, dict):
        raise ValueError(
            "ends must be {item: age}, each item's age at the end of its "
            "observation (a dict or a pandas Series indexed by item)."
        )
    return {item_id(k): v for k, v in ends.items()}


def _one_per_item(item: Any, name: str, values: Any, rows: list) -> Any:
    """The value of a per-item column on ``rows``: the same on every row
    of the item (blank cells left out), else refused."""
    if values is None or np.ndim(values) == 0:
        return values
    held = [values[r] for r in rows]
    if isinstance(values, np.ndarray) and values.ndim == 2:
        # A row of covariates: each column the same on every row of the
        # item, its blank cells left out
        block = np.asarray(held, dtype=float)
        out = np.full(block.shape[1], np.nan)
        for j in range(block.shape[1]):
            column = block[:, j]
            present = np.unique(column[~np.isnan(column)])
            if len(present) > 1:
                raise ValueError(
                    f"Item {item!r}: {name} differ between its rows; they "
                    "describe the item, so they are the same on every row "
                    "of the item (a blank cell is left out)."
                )
            if len(present):
                out[j] = present[0]
        return out
    filled = [v for v in held if not (isinstance(v, float) and np.isnan(v))]
    if not filled:
        return None
    first = filled[0]
    if any(not _same(v, first) for v in filled):
        raise ValueError(
            f"Item {item!r}: {name} differs between its rows; it describes "
            "the item, so it is the same on every row of the item (a blank "
            "cell is left out)."
        )
    return first


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a == b
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except Exception:
        try:
            return bool(np.all(a == b))
        except Exception:
            return a == b


def _history(
    ItemHistory: Any,
    item: Any,
    rows: list,
    lower: np.ndarray,
    upper: np.ndarray,
    c: np.ndarray,
    n: np.ndarray,
    e: Any,
    Z: Any,
    names: Any,
    attributes: Any,
    tl: Any,
    tr: Any,
    end: Any,
    given_end: Any,
) -> Any:
    """One item's ItemHistory from its rows of the log."""
    fields: dict = {}
    exact = [r for r in rows if c[r] == 0]
    for r in exact:
        if lower[r] != upper[r]:
            raise ValueError(
                f"Item {item!r}: row {r} is an event (c = 0) but its bounds "
                f"differ ({lower[r]:g}, {upper[r]:g}); failures counted "
                "between two inspections are c = 2."
            )
    if exact:
        # A count on an exact row: that many failures at the age
        fields["observed"] = [
            float(lower[r]) for r in exact for _ in range(int(n[r]))
        ]
    counted = [r for r in rows if c[r] == 2]
    if counted:
        fields["interval_censored"] = [
            [float(lower[r]), float(upper[r])] for r in counted
        ]
        fields["interval_censored_counts"] = [int(n[r]) for r in counted]
    before = [r for r in rows if c[r] == -1]
    if len(before) > 1:
        raise ValueError(
            f"Item {item!r} has {len(before)} rows of failures before a "
            "first look (c = -1); an item has one first look: give its "
            "count in n_col."
        )
    if before:
        fields["left_censored"] = float(upper[before[0]])
        fields["left_censored_counts"] = int(n[before[0]])
    watched = [r for r in rows if c[r] == 1]
    if len(watched) > 1:
        raise ValueError(
            f"Item {item!r} has {len(watched)} end-of-observation rows "
            "(c = 1); an item's observation ends once."
        )
    end_of = _one_per_item(item, "end_col", end, rows)
    if watched:
        fields["right_censored"] = float(upper[watched[0]])
    if end_of is not None:
        if watched and float(upper[watched[0]]) != float(end_of):
            raise ValueError(
                f"Item {item!r}: its end-of-observation row (c = 1) is at "
                f"{float(upper[watched[0]]):g} and end_col says "
                f"{float(end_of):g}; give one of them."
            )
        fields["right_censored"] = float(end_of)
    if given_end is not None:
        if "right_censored" in fields and fields["right_censored"] != float(
            given_end
        ):
            raise ValueError(
                f"Item {item!r}: the log ends its observation at "
                f"{fields['right_censored']:g} and ends= says "
                f"{float(given_end):g}; give one of them."
            )
        fields["right_censored"] = float(given_end)
    if e is not None:
        if exact:
            fields["observed_event_types"] = [e[r] for r in exact]
        if counted:
            fields["interval_censored_event_types"] = [e[r] for r in counted]
        if before:
            fields["left_censored_event_types"] = e[before[0]]
    if (Z is not None or attributes is not None) and not rows:
        raise ValueError(
            f"Item {item!r} is in ends= but has no row in the log, so its "
            + ("covariates" if Z is not None else "attributes")
            + " are not known: give it an end-of-observation row (c = 1) "
            "with them, or leave out "
            + ("Z_cols." if Z is not None else "attribute_cols.")
        )
    if Z is not None:
        row = _one_per_item(item, "the covariates", Z, rows)
        if row is not None:
            fields["covariates"] = dict(zip(names, row.tolist()))
    if attributes is not None:
        fields["attributes"] = _one_per_item(
            item, "the attributes", attributes, rows
        )
    for name, bound in (("left_truncation", tl), ("right_truncation", tr)):
        value = _one_per_item(item, f"{name}'s column", bound, rows)
        if value is not None and np.isfinite(value):
            fields[name] = float(value)
    return ItemHistory(item, **fields)


def history_to_frame(data: Any) -> Any:
    """The rows of an ``EventHistoryData`` as a DataFrame: ``i``, ``x``
    (or ``xl`` and ``xr`` where any row is a count between inspections),
    ``c``, ``n``, ``tl`` and ``tr`` (where any bound is finite), ``e``
    (where it has event types), the covariates by name and the
    attributes."""
    import pandas as pd

    problems = []
    if data.window_map is not None:
        problems.append("observation windows")
    if data.time_varying_covariates is not None:
        problems.append("time-varying covariates")
    if data.states is not None:
        problems.append("states")
    if problems:
        raise ValueError(
            "This data has "
            + " and ".join(problems)
            + ", which a table with one row per event cannot hold: keep "
            "the records (data.to_items()) or the JSON (data.to_json())"
            + (
                ", or use data.without(...) for the parts a table can hold."
                if data.window_map is None
                else "."
            )
        )
    columns: dict = {"i": [_plain(v) for v in data.i]}
    x = np.asarray(data.x, dtype=float)
    if x.ndim == 2:
        columns["xl"] = x[:, 0]
        columns["xr"] = x[:, 1]
    else:
        columns["x"] = x
    columns["c"] = np.asarray(data.c)
    columns["n"] = np.asarray(data.n)
    tl, tr = np.asarray(data.tl, dtype=float), np.asarray(data.tr, dtype=float)
    if np.isfinite(tl).any():
        columns["tl"] = tl
    if np.isfinite(tr).any():
        columns["tr"] = tr
    if data.e is not None:
        columns["e"] = list(data.e)
    _covariate_columns(columns, data)
    attributes = getattr(data, "attributes", None)
    if attributes:
        by_item = dict(zip(data.ids, attributes))
        _attribute_columns(columns, [by_item.get(_plain(v)) for v in data.i])
    return pd.DataFrame(columns)
