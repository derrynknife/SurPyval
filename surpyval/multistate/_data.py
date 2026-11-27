"""Multi-state data: checking it, and the long and stacked forms.

The models take the *long* form, one row per spell a unit spends in one
state: the state it was in (``state``), when the spell began (``tl``, its
entry) and ended (``x``), and the state it moved into at ``x`` (``e``;
missing where the spell ended by censoring). The *stacked* form (mstate's
``msprep`` output) has one row per transition a spell could have ended
in; :func:`to_stacked` and :func:`from_stacked` convert between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate.transitions import TransitionMatrix
from surpyval.univariate.competing_risks.labels import ordered_labels
from surpyval.utils.data_formats import missing_events


@dataclass
class SpellData:
    """Checked long-form data, with states as positions in ``tm``."""

    x: npt.NDArray  # exit times
    tl: npt.NDArray  # entry times (-inf: from the start)
    g: npt.NDArray  # state during the spell (index)
    h: npt.NDArray  # state entered at x (index; -1 censored)
    n: npt.NDArray  # counts
    cluster: npt.NDArray  # the unit each row belongs to (codes)
    n_clusters: int
    tm: TransitionMatrix
    ids_given: bool
    #: The earliest time each spell can have ended: ``x`` where its end
    #: is known, the left end of the interval where the move out of the
    #: state is known only to lie in ``(xl, x]``. ``None``: every end is
    #: known.
    xl: npt.NDArray | None = None
    #: On the reset clock, the rows that carry on the stay of their
    #: unit's row before (a stay split into rows): their time in the
    #: state runs on from that row's. ``None``: every row is its own stay.
    carry: npt.NDArray | None = None

    @property
    def interval(self) -> npt.NDArray:
        """Whether each spell's end is interval censored."""
        if self.xl is None:
            return np.zeros(self.x.size, dtype=bool)
        return self.xl < self.x

    @property
    def has_intervals(self) -> bool:
        return bool(np.any(self.interval))


def _object_array(values: Any, name: str, size: int | None) -> npt.NDArray:
    if values is None:
        raise ValueError(f"'{name}' is needed.")
    arr = np.empty(len(values), dtype=object)
    arr[:] = list(values)
    if size is not None and arr.size != size:
        raise ValueError(
            f"'{name}' has {arr.size} values but 'x' has {size}: give one "
            "per row."
        )
    return arr


def _times(values: Any, name: str, size: int) -> npt.NDArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(size, float(arr))
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(
            f"'{name}' must have one value per row ({size}); got shape "
            f"{np.shape(values)}."
        )
    return arr


def _exit_times(
    x: Any, intervals: bool
) -> tuple[npt.NDArray | None, npt.NDArray]:
    """``(xl, x)``: the exit times, and the left ends of interval-censored
    moves (``None`` where ``x`` is one column)."""
    x = np.asarray(x, dtype=float)
    xl = None
    if x.ndim == 2 and x.shape[1] == 2 and intervals:
        xl, x = x[:, 0].copy(), x[:, 1].copy()
    elif x.ndim != 1:
        raise ValueError(
            "'x' must be one exit time per row (1-D)"
            + (
                " or two columns [xl, xr] per row, a move known only to "
                "lie between them"
                if intervals
                else "; a move known only to lie between two times "
                "(interval censored) is taken by ParametricMultiState and "
                "ParametricMultiStateRegression, with 'x' as two columns "
                "[xl, xr]"
            )
            + f"; got shape {x.shape}."
        )
    return xl, x


def _check_intervals(
    xl: npt.NDArray, x: npt.NDArray, censored: npt.NDArray
) -> None:
    """Refuse an interval-censored move's ends that cannot be."""
    if np.any(~np.isfinite(xl)):
        raise ValueError(
            "'x' has a missing or infinite left end 'xl': give the "
            "last time each unit was seen in its state (xl = xr where "
            "the move's time is known)."
        )
    if np.any(xl > x):
        k = int(np.flatnonzero(xl > x)[0])
        raise ValueError(
            f"Row {k} has xl = {xl[k]:g} above xr = {x[k]:g}: an "
            "interval-censored move lies in (xl, xr], so xl <= xr."
        )
    wide = (xl < x) & censored
    if np.any(wide):
        k = int(np.flatnonzero(wide)[0])
        raise ValueError(
            f"Row {k} is censored (a missing 'e') with an interval "
            f"({xl[k]:g}, {x[k]:g}]: a censored spell ends at the last "
            "time the unit was seen in its state, so give xl = xr "
            "there."
        )


def check_spells(
    x: npt.ArrayLike,
    e: npt.ArrayLike,
    state: npt.ArrayLike,
    tl: npt.ArrayLike | float | None = None,
    i: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    transitions: Any = None,
    intervals: bool = False,
) -> SpellData:
    """Check long-form data and code its states (see the module
    docstring and ``MultiState.fit``). With ``intervals``, ``x`` may also
    be two columns, ``[xl, xr]`` per row: a move out of the state at an
    unknown time in ``(xl, xr]`` (``xl == xr`` where it is known)."""
    xl, x = _exit_times(x, intervals)
    size = x.size
    if size == 0:
        raise ValueError("No data: 'x' is empty.")
    if np.any(np.isnan(x)):
        raise ValueError(
            "'x' has missing (NaN) times: every spell needs its end time."
        )
    if np.any(~np.isfinite(x)):
        raise ValueError(
            "'x' has an infinite time: give the time the spell was last "
            "seen (censored) or ended."
        )
    e = _object_array(e, "e", size)
    state = _object_array(state, "state", size)
    censored = missing_events(e)
    if xl is not None:
        _check_intervals(xl, x, censored)
        if not np.any(xl < x):
            xl = None
    if np.any(missing_events(state)):
        raise ValueError(
            "'state' has missing values: every spell needs the state the "
            "unit was in."
        )
    tl_arr = np.full(size, -np.inf) if tl is None else _times(tl, "tl", size)
    if np.any(np.isnan(tl_arr)):
        raise ValueError(
            "'tl' has missing (NaN) entry times; use -inf (or leave 'tl' "
            "out) for a spell observed from the start."
        )
    if xl is not None:
        late = (tl_arr > xl) & (xl < x)
        if np.any(late):
            k = int(np.flatnonzero(late)[0])
            raise ValueError(
                f"Row {k} enters its state at tl = {tl_arr[k]:g}, after "
                f"the start of its interval ({xl[k]:g}, {x[k]:g}]: the "
                "unit was seen in its state at xl, so tl <= xl."
            )
    bad = tl_arr >= x
    if np.any(bad):
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{int(bad.sum())} rows have an entry time 'tl' at or after "
            f"their exit time 'x' (the first is row {k}: tl={tl_arr[k]:g}, "
            f"x={x[k]:g}); a spell is the time (tl, x], so tl must be "
            "below x."
        )
    if n is None:
        n_arr = np.ones(size)
    else:
        n_arr = _times(n, "n", size)
        if np.any(~np.isfinite(n_arr)) or np.any(n_arr <= 0):
            raise ValueError(
                "'n' (the counts) must be positive and finite numbers."
            )

    same = np.array(
        [not c and bool(a == b) for a, b, c in zip(state, e, censored)],
        dtype=bool,
    )
    if np.any(same):
        k = int(np.flatnonzero(same)[0])
        raise ValueError(
            f"Row {k} ends in a move from {state[k]!r} into {e[k]!r}, the "
            "same state; a spell ends in a move into another state or in "
            "censoring (a missing 'e')."
        )

    if transitions is None:
        labels = ordered_labels(list(state) + list(e[~censored]))
        pairs = sorted(
            {(a, b) for a, b, c in zip(state, e, censored) if not c},
            key=lambda p: (labels.index(p[0]), labels.index(p[1])),
        )
        if not pairs:
            raise ValueError(
                "No transitions were observed (every 'e' is missing) and "
                "none were given: pass 'transitions' to fit a model with "
                "no events."
            )
        tm = TransitionMatrix(pairs, states=labels)
    elif isinstance(transitions, TransitionMatrix):
        tm = transitions
    else:
        tm = TransitionMatrix(transitions)

    known = set(tm.states)
    for name, arr, mask in (
        ("state", state, np.ones(size, bool)),
        ("e", e, ~censored),
    ):
        unknown = [v for v in ordered_labels(arr[mask]) if v not in known]
        if unknown:
            raise ValueError(
                f"'{name}' holds states {unknown!r} that the transitions "
                f"do not have (the states are {list(tm.states)!r})."
            )
    allowed = set(tm.transitions)
    seen_pairs = {(a, b) for a, b, c in zip(state, e, censored) if not c}
    not_allowed = sorted(seen_pairs - allowed, key=str)
    if not_allowed:
        raise ValueError(
            f"The data have transitions {not_allowed!r} that 'transitions' "
            "does not allow: add them, or correct the data."
        )
    g = np.array([tm.index(s) for s in state], dtype=np.intp)
    h = np.array(
        [-1 if c else tm.index(s) for s, c in zip(e, censored)],
        dtype=np.intp,
    )

    if i is None:
        cluster = np.arange(size)
        n_clusters = size
        ids_given = False
    else:
        import pandas as pd

        ids = _object_array(i, "i", size)
        if np.any(missing_events(ids)):
            raise ValueError(
                "'i' (the unit of each row) has missing values: label "
                "every row."
            )
        codes, uniques = pd.factorize(pd.Series(ids, dtype=object))
        cluster = np.asarray(codes, dtype=np.intp)
        n_clusters = len(uniques)
        ids_given = True
        order = np.lexsort((tl_arr, cluster))
        # (the spell entered by an interval-censored move may start
        # anywhere in the move's interval)
        ends = x if xl is None else xl
        cs, ts, xs = cluster[order], tl_arr[order], ends[order]
        same_unit = cs[1:] == cs[:-1]
        overlap = same_unit & (ts[1:] < xs[:-1])
        if np.any(overlap):
            k = order[1:][overlap][0]
            raise ValueError(
                f"Unit {ids[k]!r} has spells that overlap in time (one "
                f"starts at tl={tl_arr[k]:g} before the one before it "
                "ends): each unit is in one state at a time. Give each "
                "spell its entry time 'tl'."
            )
        ns = n_arr[order]
        if np.any(same_unit & (ns[1:] != ns[:-1])):
            raise ValueError(
                "'n' differs between the rows of one unit: a count stands "
                "for identical units, so it is the same on each of their "
                "rows."
            )
    return SpellData(
        x=x,
        tl=tl_arr,
        g=g,
        h=h,
        n=n_arr,
        cluster=cluster,
        n_clusters=n_clusters,
        tm=tm,
        ids_given=ids_given,
        xl=xl,
    )


def to_stacked(
    x: npt.ArrayLike,
    e: npt.ArrayLike,
    state: npt.ArrayLike,
    transitions: Any,
    tl: npt.ArrayLike | float | None = None,
    i: npt.ArrayLike | None = None,
) -> Any:
    """
    Long-form multi-state data in the stacked form: one row for each
    transition each spell could have ended in.

    A spell in state ``a`` from ``tl`` to ``x`` gets one row for every
    transition out of ``a``, with ``status`` 1 on the transition it ended
    in (none if it was censored) and 0 on the others. This is the layout
    mstate's ``msprep`` gives, and the one a Cox model per transition is
    fitted to (one stratum per transition).

    Parameters
    ----------
    x, e, state, tl, i
        The long form, as for :meth:`MultiState.fit`.
    transitions : TransitionMatrix or iterable of (from, to)
        The allowed transitions.

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (the unit, ``i``, or the row of the spell when ``i``
        is not given),
        ``from``, ``to``, ``trans`` (the transition's number), ``tl``,
        ``x`` and ``status``.

    Examples
    --------
    >>> from surpyval.multistate import TransitionMatrix, to_stacked
    >>> tm = TransitionMatrix.illness_death()
    >>> to_stacked(
    ...     x=[2, 5], e=["ill", None], state=["healthy", "ill"],
    ...     tl=[0, 2], i=[1, 1], transitions=tm,
    ... )
       id     from   to  trans   tl    x  status
    0   1  healthy  ill      0  0.0  2.0       1
    1   1  healthy dead      1  0.0  2.0       0
    2   1      ill dead      2  2.0  5.0       0
    """
    import pandas as pd

    data = check_spells(x, e, state, tl, i, None, transitions)
    tm = data.tm
    ids = (
        np.arange(data.x.size)
        if i is None
        else _object_array(i, "i", data.x.size)
    )
    frm = tm.from_index()
    rows = []
    for k in range(data.x.size):
        for t in np.flatnonzero(frm == data.g[k]):
            a, b = tm.transitions[t]
            rows.append(
                (
                    ids[k],
                    a,
                    b,
                    int(t),
                    data.tl[k],
                    data.x[k],
                    int(data.h[k] == tm.index(b)),
                )
            )
    return pd.DataFrame(
        rows, columns=["id", "from", "to", "trans", "tl", "x", "status"]
    )


def from_stacked(
    i: npt.ArrayLike,
    from_state: npt.ArrayLike,
    to_state: npt.ArrayLike,
    tl: npt.ArrayLike,
    x: npt.ArrayLike,
    status: npt.ArrayLike,
) -> dict:
    """
    Stacked multi-state data (mstate's ``msprep`` layout) in the long
    form the models take.

    The rows of one spell (the same unit ``i``, ``from_state``, ``tl`` and
    ``x``) become one row, whose ``e`` is the ``to_state`` of the row
    with ``status`` 1, or missing if none has it.

    Returns
    -------
    dict
        ``{"x", "e", "state", "tl", "i"}``, to pass to
        ``MultiState.fit(**...)``.

    Examples
    --------
    >>> from surpyval.multistate import from_stacked
    >>> long = from_stacked(
    ...     i=[1, 1, 1], from_state=["healthy", "healthy", "ill"],
    ...     to_state=["ill", "dead", "dead"], tl=[0, 0, 2], x=[2, 2, 5],
    ...     status=[1, 0, 0],
    ... )
    >>> long["e"], long["state"]
    (['ill', None], ['healthy', 'ill'])
    """
    i = _object_array(i, "i", None)
    size = i.size
    frm = _object_array(from_state, "from_state", size)
    to = _object_array(to_state, "to_state", size)
    tl = _times(tl, "tl", size)
    x = _times(x, "x", size)
    status = _times(status, "status", size)
    if np.any(~np.isin(status, (0, 1))):
        raise ValueError("'status' must be 0 or 1 on every row.")
    spells: dict = {}
    for k in range(size):
        key = (i[k], frm[k], tl[k], x[k])
        if key not in spells:
            spells[key] = None
        if status[k] == 1:
            if spells[key] is not None:
                raise ValueError(
                    f"Unit {i[k]!r}'s spell in {frm[k]!r} ending at "
                    f"{x[k]:g} has status 1 on two transitions: a spell "
                    "ends in one transition at most."
                )
            spells[key] = to[k]
    keys = list(spells)
    return {
        "x": [k[3] for k in keys],
        "e": [spells[k] for k in keys],
        "state": [k[1] for k in keys],
        "tl": [k[2] for k in keys],
        "i": [k[0] for k in keys],
    }
