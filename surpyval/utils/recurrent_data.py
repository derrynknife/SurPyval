"""Recurrent fits from a :class:`~surpyval.utils.surpyval_data.SurpyvalData`.

A recurrent (repairable-system) model's ``fit`` takes the arrays of
:func:`~surpyval.utils.recurrent_utils.handle_xicn` (``x``, ``i``, ``c``,
``n``, the truncation ``tl`` / ``tr`` or the observation ``windows``,
``Z`` and ``e``), or an
:class:`~surpyval.utils.event_history_data.EventHistoryData`. This module
lets it take a ``SurpyvalData`` with units ``i`` as well: each unit's rows
are read as its history on its own clock, and turned into exactly the
arrays a user would have passed, so the fit is the fit of those arrays.

How a unit's rows are read:

- ``x`` is the unit's age at each row's end: an event (``c = 0``, or a
  count, ``c = -1`` or ``2``), or the end of the record (``c = 1``).
- A row starts at its ``tl`` where that is finite, and otherwise where the
  previous row of the unit ended (the first row at the unit's entry). A
  row whose ``tl`` is after the previous row's ``x`` starts a new
  observation window: the unit was not watched in between (a gap). Gapped
  data is given to the fit as ``windows``, as the arrays would be.
- A ``c = 1`` row before the unit's last row ends a stretch: where the
  next row starts at its ``x`` it is dropped (the unit was watched
  throughout), unless the covariates change there, in which case the fit
  is made by the model's ``fit_tvc`` from the rows as start-stop rows.
- ``a`` (the terminal flag) does not change the arrays: none of the
  recurrent models models a terminal event, and a unit's record without a
  ``c = 1`` row ends at its last event, which is where a terminal event
  ends it. What the fit made of ``a`` is said in the model's ``notes``.
"""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from surpyval.utils.surpyval_data import (
    _NO_MARKERS,
    SurpyvalData,
    _distinct_states,
    lost_message,
    named_fit,
    terminal_flags,
    with_data_unit,
)
from surpyval.utils.warnings import named

F = TypeVar("F", bound=Callable[..., Any])

# The arguments a recurrent fit takes its data in, besides ``x``: given
# with a SurpyvalData they are refused (the data holds them).
_BESIDE = ("i", "c", "n", "t", "tl", "tr", "windows", "xl", "xr", "w")

# The models that take what a fit without the argument cannot.
_TAKE_COVARIATES = (
    "the proportional-intensity models (ProportionalIntensityHPP, "
    "ProportionalIntensityNHPP, CauseSpecificProportionalIntensity) and "
    "the recurrent Cox models (AndersenGill, PrenticeWilliamsPeterson, "
    "WeiLinWeissfeld) take covariates"
)
_TAKE_EVENT_TYPES = (
    "the cause-specific models (CauseSpecificMCF, CauseSpecificNHPP, "
    "CauseSpecificProportionalIntensity) take the event types"
)
_TAKE_GAPS = (
    "HPP, CrowAMSAA, Duane, CoxLewis and NonParametricCounting take gaps "
    "without covariates, and the proportional-intensity models' fit_tvc "
    "with them"
)


def notes_text(model: Any) -> str:
    """The lines a model's ``repr`` ends with for its ``notes`` (what the
    fit made of the data it was given), ``""`` where it has none."""
    notes = getattr(model, "notes", ())
    if not notes:
        return ""
    return "\nNotes               :\n" + "\n".join(f"  - {n}" for n in notes)


def accepts_recurrent_surpyval_data(method: F) -> F:
    """Let a recurrent fitter's ``fit`` take a
    :class:`~surpyval.utils.surpyval_data.SurpyvalData` with units ``i``
    as its ``x``: ``CrowAMSAA.fit(data)``. The data is the whole of it, so
    the other data arguments (``i``, ``c``, ``n``, ``t``, ``tl``, ``tr``,
    ``windows``, and ``Z`` or ``e`` where the data has them) must be left
    out. The data's rows are turned into the arrays ``fit`` takes (see the
    module's description), so the fit is that of the arrays; what the fit
    made of the terminal flags ``a`` is in the model's ``notes``. Data the
    fit cannot take (no units, several states, weights, measurements,
    covariates or event types it has no argument for, gaps where it takes
    no windows) is refused, saying what to use."""
    signature = inspect.signature(method)
    params = list(signature.parameters)
    names = set(params)
    if "x" not in names:
        return method

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        data = bound.arguments.get("x")
        if not isinstance(data, SurpyvalData) or data.ids is None:
            # Without units: the fit refuses it as it did before, saying
            # to give each row's unit.
            return method(*args, **kwargs)
        given = [
            name
            for name in _BESIDE + ("Z", "e")
            if bound.arguments.get(name) is not None
            and not (name == "Z" and data.Z is None)
            and not (name == "e" and data.e is None)
        ]
        if given:
            raise TypeError(
                "x is a SurpyvalData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than a SurpyvalData)."
            )
        _refuse_what_no_recurrent_fit_takes(data)
        owner = bound.arguments.get(params[0]) if params else None
        fit_tvc = (
            getattr(owner, "fit_tvc", None)
            if params[0] in ("self", "cls")
            else None
        )
        layout = _Layout(data)
        lost = _lost(data, names, layout, callable(fit_tvc))
        if lost:
            raise ValueError(_refusal(lost))
        if data.Z is None and "Z" in names:
            parameter = signature.parameters["Z"]
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    "This fit needs covariates, Z, and the SurpyvalData "
                    "holds none: build it with SurpyvalData(x, c, ..., "
                    "i=i, Z=Z)."
                )
        if layout.needs_tvc(data):
            assert fit_tvc is not None  # _lost refused it otherwise
            options = _tvc_options(bound, signature, fit_tvc, params[0])
            rows = layout.tvc_rows(data)

            def run() -> Any:
                return fit_tvc(**rows, **options)

        else:
            bound.arguments.update(layout.arrays(data, names))

            def run() -> Any:
                return method(*bound.args, **bound.kwargs)

        model = with_data_unit(
            named_fit(
                run, data.covariate_names if data.Z is not None else None
            ),
            data,
        )
        try:
            model.notes = tuple(
                terminal_notes(data.c, data.a, data.a_inferred, data.ids)
            )
        except AttributeError:  # pragma: no cover - every model takes it
            pass
        return model

    setattr(wrapper, "_accepts_recurrent_surpyval_data", True)
    return wrapper  # type: ignore[return-value]


def _refuse_what_no_recurrent_fit_takes(data: SurpyvalData) -> None:
    """Refuse data that no recurrent fit takes: in more than one state,
    with measurements, weights or step time-varying covariates from
    records."""
    if len(_distinct_states(data)) > 1:
        raise ValueError(lost_message(["states"], False))
    if data.m is not None:
        raise ValueError(_NO_MARKERS)
    if data.w is not None:
        raise ValueError(
            "This fit takes no weights, which the SurpyvalData holds: no "
            "recurrent model takes weights w. To fit without them, use "
            'data.without("weights").'
        )
    if data.time_varying_covariates is not None:
        raise ValueError(
            "This fit takes no time-varying covariates from records, which "
            "the SurpyvalData holds: give a unit's covariates as rows "
            "instead, one per stretch with the covariates fixed in it "
            "(SurpyvalData(x, c, ..., i=i, tl=tl, Z=Z)), which the "
            "proportional-intensity models fit by their fit_tvc."
        )


def _lost(
    data: SurpyvalData, names: set, layout: _Layout, has_tvc: bool
) -> list:
    """What ``data`` holds that this fit (taking the arguments ``names``,
    with a ``fit_tvc`` where ``has_tvc``) cannot take, as ``(what, how)``
    pairs: ``how`` says what to use."""
    lost = []
    if data.Z is not None and "Z" not in names:
        lost.append(
            (
                "covariates",
                "data.without('covariates') to fit without them; "
                + _TAKE_COVARIATES,
            )
        )
    if data.e is not None and "e" not in names:
        lost.append(
            (
                "event types",
                "data.without('event_types') to fit without them, or "
                "data.for_event_type(t) for the events of one type; "
                + _TAKE_EVENT_TYPES,
            )
        )
    if lost:
        return lost
    tvc = layout.needs_tvc(data)
    if tvc and not has_tvc:
        units = layout.varying
        lost.append(
            (
                "covariates that change within a unit (unit"
                f"{'s' if len(units) > 1 else ''} {named(units)})",
                "give each unit one row of covariates, or use a model whose "
                "fit_tvc takes them (ProportionalIntensityHPP, "
                "ProportionalIntensityNHPP, "
                "CauseSpecificProportionalIntensity, AndersenGill, "
                "PrenticeWilliamsPeterson)",
            )
        )
    elif layout.gapped and not tvc and "windows" not in names:
        units = layout.gapped
        lost.append(
            (
                "gaps in observation (unit"
                f"{'s' if len(units) > 1 else ''} {named(units)}: a row "
                "starting, tl, after the previous row of its unit ended)",
                _TAKE_GAPS,
            )
        )
    if tvc:
        # fit_tvc reads a row as the stretch from where its unit's previous
        # row ended, and a range (xl, xr) starts after that.
        ranged = layout.lo < layout.hi
        if ranged.any():
            ids = list(data.ids)  # type: ignore[arg-type]
            units = list(dict.fromkeys(ids[r] for r in np.flatnonzero(ranged)))
            lost.append(
                (
                    "ranges (xl, xr) with covariates that change within a "
                    f"unit (unit{'s' if len(units) > 1 else ''} "
                    f"{named(units)})",
                    "data with each event at its time, x, or with one row "
                    "of covariates per unit",
                )
            )
        if np.isfinite(data.tr).any():
            lost.append(
                (
                    "right truncation (with time-varying covariates)",
                    "data.without('truncation') to fit without it",
                )
            )
    else:
        left = bool(np.isfinite(layout.entry).any())
        right = bool(np.isfinite(data.tr).any())
        if left and not ({"tl", "t"} & names) and not layout.gapped:
            lost.append(
                (
                    "left truncation",
                    "data.without('truncation') to fit without it",
                )
            )
        if right and not ({"tr", "t"} & names):
            lost.append(
                (
                    "right truncation",
                    "data.without('truncation') to fit without it",
                )
            )
    if "n" not in names and np.any(np.asarray(data.n) != 1):
        lost.append(("counts", "data with n = 1 on every row"))
    return lost


def _refusal(lost: list) -> str:
    """The refusal of a recurrent ``fit(data)`` for what it cannot take."""
    what = " or ".join(w for w, _ in lost)
    hows = list(dict.fromkeys(h for _, h in lost))
    return (
        f"This fit takes no {what}, which the SurpyvalData holds. Use "
        + "; or ".join(hows)
        + "."
    )


def _tvc_options(
    bound: inspect.BoundArguments,
    signature: inspect.Signature,
    fit_tvc: Callable,
    first: str,
) -> dict:
    """The options of ``fit`` given (or defaulted) in ``bound`` that its
    ``fit_tvc`` takes; one it does not take, given with a value other
    than its default, is refused."""
    takes = set(inspect.signature(fit_tvc).parameters)
    out = {}
    for name, value in bound.arguments.items():
        if name in (first, "x", "Z", "e") + _BESIDE:
            continue
        if name in takes:
            out[name] = value
        elif value is not signature.parameters[name].default:
            raise TypeError(
                f"The SurpyvalData's covariates change within a unit, so it "
                f"is fitted by fit_tvc, which takes no {name!r}: leave it "
                "out."
            )
    return out


class _Layout:
    """The rows of each unit read as its history: where each row starts,
    which rows are dropped, the unit's entry, its gaps and windows, and
    whether its covariates change."""

    def __init__(self, data: SurpyvalData) -> None:
        ids = list(data.ids)  # type: ignore[arg-type]
        x = np.asarray(data.x, dtype=float)
        self.lo, self.hi = (x[:, 0], x[:, 1]) if x.ndim == 2 else (x, x)
        c = np.asarray(data.c)
        tl = np.asarray(data.tl, dtype=float)
        k = len(c)
        self.rows_of: dict = {}
        for r, unit in enumerate(ids):
            self.rows_of.setdefault(unit, []).append(r)
        self.start = tl.copy()
        self.entry = np.full(k, -np.inf)
        # A c = 1 row before its unit's last row: a stretch boundary (the
        # next row starts at its x) or the close of a window (a gap).
        self.boundary = np.zeros(k, dtype=bool)
        self.gapped: list = []
        self.varying: list = []
        self.segments: dict = {}
        Z = data.Z
        for unit, rows in self.rows_of.items():
            first = rows[0]
            self.entry[rows] = tl[first]
            segments = [[first]]
            for p, q in zip(rows, rows[1:]):
                if np.isfinite(tl[q]) and tl[q] > self.hi[p]:
                    segments.append([q])
                else:
                    self.start[q] = self.hi[p]
                    segments[-1].append(q)
                if c[p] == 1:
                    self.boundary[p] = True
            if len(segments) > 1:
                self.gapped.append(unit)
            self.segments[unit] = segments
            if Z is not None and not np.all(Z[rows] == Z[first]):
                self.varying.append(unit)

    def needs_tvc(self, data: SurpyvalData) -> bool:
        """Whether the fit is made by ``fit_tvc``: covariates that change
        within a unit, or covariates with gaps (which only ``fit_tvc``
        takes)."""
        return bool(self.varying) or (bool(self.gapped) and data.Z is not None)

    def arrays(self, data: SurpyvalData, names: set) -> dict:
        """The arguments of ``fit`` (of those in ``names``): the arrays a
        user would have passed for the same histories."""
        c = np.asarray(data.c)
        if self.gapped:
            keep = c != 1
            windows = {}
            for unit, segments in self.segments.items():
                windows[unit] = [
                    (
                        (
                            float(self.start[seg[0]])
                            if np.isfinite(self.start[seg[0]])
                            else 0.0
                        ),
                        float(self.hi[seg[-1]]),
                    )
                    for seg in segments
                ]
        else:
            keep = ~self.boundary
            windows = None
        out: dict = {
            "x": np.asarray(data.x)[keep],
            "i": np.asarray(_id_array(data.ids))[keep],
            "c": c[keep],
            "n": np.asarray(data.n)[keep],
        }
        if windows is not None:
            out["windows"] = windows
        else:
            tl = self.entry[keep]
            tr = np.asarray(data.tr, dtype=float)[keep]
            left, right = bool(np.isfinite(tl).any()), bool(
                np.isfinite(tr).any()
            )
            if (left and "tl" not in names) or (right and "tr" not in names):
                out["t"] = np.column_stack([tl, tr])
            else:
                if left:
                    out["tl"] = tl
                if right:
                    out["tr"] = tr
        if data.Z is not None:
            out["Z"] = np.asarray(data.Z)[keep]
        if data.e is not None:
            out["e"] = np.asarray(data.e, dtype=object)[keep]
        return {k: v for k, v in out.items() if k in names}

    def tvc_rows(self, data: SurpyvalData) -> dict:
        """The rows as the start-stop rows of ``fit_tvc``: each from its
        start (the unit's entry, or 0 where it has none, for its first
        row) to its ``x``."""
        xl = np.where(np.isfinite(self.start), self.start, 0.0)
        out = {
            "i": np.asarray(_id_array(data.ids)),
            "xl": xl,
            "xr": self.hi.copy(),
            "c": np.asarray(data.c),
            "Z": np.asarray(data.Z),
        }
        if np.any(np.asarray(data.n) != 1):
            out["n"] = np.asarray(data.n)
        if data.e is not None:
            out["e"] = np.asarray(data.e, dtype=object)
        return out


def _id_array(ids: Any) -> np.ndarray:
    """The unit ids as an array of one kind (numbers or text)."""
    listed = list(ids)
    if all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        for v in listed
    ):
        return np.asarray(listed)
    return np.asarray(listed, dtype=object)


def terminal_notes(
    c: Any, a: Any, a_inferred: Any, ids: Any, model: str = "This model"
) -> list:
    """What a recurrent fit made of the terminal flags ``a``, as notes:
    the units whose last row is an event with ``a = 1`` (fitted as an
    event followed by the end of the unit's observation, since no
    recurrent model models terminal events), how many of those were
    inferred (no ``a`` given, or blank), and the units whose last row is
    an event with ``a = 0`` given and no end-of-record row (their
    observation ends at that event, as for the arrays). Rows are each
    unit's in time order; without ``a`` (``None``) an event on a unit's
    last row is read as terminal, as ``SurpyvalData`` reads it.

    Examples
    --------
    >>> from surpyval.utils.recurrent_data import terminal_notes
    >>> for note in terminal_notes(
    ...     c=[0, 0, 0, 1], a=[0, 1, 0, 0], a_inferred=[0, 1, 0, 0],
    ...     ids=["A", "A", "B", "B"],
    ... ):
    ...     print(note)  # doctest: +NORMALIZE_WHITESPACE
    1 unit ended in a terminal event (a = 1: 'A'). This model does not
    model terminal events, so it is fitted as an event followed by the end
    of that unit's observation.
    a was not given for 1 of them ('A'): an event on a unit's last row with
    no a is read as terminal (a = 1).
    """
    c = np.asarray(c)
    k = len(c)
    units = list(range(k)) if ids is None else list(ids)
    last: dict = {}
    for r, unit in enumerate(units):
        last[unit] = r
    if a is None:
        a = np.where(c != 1, 1, 0)
        a_inferred = c != 1
    a = np.asarray(a)
    a_inferred = (
        np.zeros(k, dtype=bool)
        if a_inferred is None
        else np.asarray(a_inferred, dtype=bool)
    )
    ended = [u for u, r in last.items() if c[r] != 1 and a[r] == 1]
    guessed = [u for u in ended if a_inferred[last[u]]]
    carried = [
        u
        for u, r in last.items()
        if c[r] != 1 and a[r] == 0 and not a_inferred[r]
    ]
    notes = []
    if ended:
        one = len(ended) == 1
        notes.append(
            f"{len(ended)} unit{'' if one else 's'} ended in a terminal "
            f"event (a = 1: {named(ended)}). {model} does not model "
            "terminal events, so "
            + ("it is" if one else "each is")
            + " fitted as an event followed by the end of that unit's "
            "observation."
        )
    if guessed:
        notes.append(
            f"a was not given for {len(guessed)} of them ({named(guessed)}):"
            " an event on a unit's last row with no a is read as terminal "
            "(a = 1)."
        )
    if carried:
        one = len(carried) == 1
        notes.append(
            f"{len(carried)} unit{'' if one else 's'} "
            + ("ends" if one else "end")
            + f" at an event with a = 0 and no end-of-record (c = 1) row "
            f"({named(carried)}): "
            + ("its" if one else "their")
            + " observation is taken to end at that event, as for the "
            "arrays."
        )
    return notes


def takes_terminal_column(method: F) -> F:
    """Give a recurrent ``fit_from_df`` an optional keyword ``a_col``,
    the column of terminal flags (1 where the row's event ended the
    unit's history for good, 0 or blank elsewhere), read as
    ``SurpyvalData.from_dataframe(a_col=...)`` reads it: a blank on an
    event is 1 on its unit's last row and 0 before, ``a = 1`` on a
    ``c = 1`` row or before a unit's last row is refused. It does not
    change the fit (no recurrent model models terminal events); the
    model's ``notes`` say what was made of it."""
    signature = inspect.signature(method)
    if "a_col" in signature.parameters:
        return method

    @functools.wraps(method)
    def wrapper(*args: Any, a_col: Any = None, **kwargs: Any) -> Any:
        if a_col is None:
            return method(*args, **kwargs)
        from surpyval.utils.dataframe import frame_column, require_frame

        bound = signature.bind_partial(*args, **kwargs)
        df = require_frame(bound.arguments.get("df"))
        flags = list(frame_column(df, a_col, "a_col"))
        model = method(*args, **kwargs)
        rows = len(df)

        def column(name: str, default: Any) -> np.ndarray:
            label = bound.arguments.get(name)
            if label is None:
                return np.full(rows, default)
            return np.asarray(df[label])

        x = column("x_col", 0.0).astype(float)
        i = column("i_col", 1)
        c = column("c_col", 0)
        notes = _terminal_column_notes(x, i, c, flags)
        try:
            model.notes = tuple(notes)
        except AttributeError:  # pragma: no cover - every model takes it
            pass
        return model

    params = list(signature.parameters.values())
    keyword = inspect.Parameter(
        "a_col", inspect.Parameter.KEYWORD_ONLY, default=None
    )
    at = len(params)
    if params and params[-1].kind is inspect.Parameter.VAR_KEYWORD:
        at -= 1
    params.insert(at, keyword)
    setattr(wrapper, "__signature__", signature.replace(parameters=params))
    return wrapper  # type: ignore[return-value]


def _terminal_column_notes(
    x: np.ndarray, i: np.ndarray, c: np.ndarray, flags: list
) -> list:
    """The notes of a ``fit_from_df`` given ``a_col``: the rows are put in
    each unit's time order as ``handle_xicn`` puts them (an end of record
    after an event at the same time), the flags read and checked as
    ``SurpyvalData`` reads them, naming the frame's rows."""
    try:
        order = np.lexsort((np.where(c == 1, 3, c), x, i))
    except TypeError:
        order = np.lexsort(
            (np.where(c == 1, 3, c), x, np.asarray([str(v) for v in i]))
        )
    ids = [
        i[r].item() if isinstance(i[r], np.generic) else i[r] for r in order
    ]
    try:
        a, inferred = terminal_flags(
            [flags[r] for r in order],
            c[order],
            x[order],
            np.full(len(order), -np.inf),
            ids,
            False,
        )
    except ValueError as error:
        # Name the frame's rows, not the sorted positions.
        text = re.sub(
            r"\brow (\d+)",
            lambda m: f"row {int(order[int(m.group(1))])}",
            str(error),
        )
        raise ValueError(text.replace("a = 1", "a_col = 1", 1)) from None
    return terminal_notes(c[order], a, inferred, ids)
