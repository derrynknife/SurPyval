"""Event history data: each item's history over time, the data of the
recurrent event models (and, as they arrive, of the multi-state and joint
models). Formerly ``RecurrentEventData`` in
``surpyval.utils.recurrent_event_data`` (renamed in v0.25)."""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

from surpyval.utils.data_reshape import HISTORY_PARTS, ReshapeMixin
from surpyval.utils.deprecation import REMOVED_IN_NEXT, RenamedAttribute
from surpyval.utils.surpyval_data import (
    SurpyvalData,
    _named_row,
    _plain,
    _plural,
    named_fit,
)
from surpyval.utils.validation import unknown_cause_error
from surpyval.utils.warnings import (
    MaskedCauseWarning,
    caller_stacklevel,
    named,
    warn_masked,
)


class EventHistoryData(ReshapeMixin):
    """
    Event history data: the history of each of several items over time --
    events that can happen more than once (recurrent failures and
    repairs), their types, each item's observation window and covariates,
    and its changes of state. The recurrent event models fit it; it was
    called ``RecurrentEventData`` before v0.25, a name still accepted, with
    a DeprecationWarning, until v0.26.

    Build one from each item's record with :meth:`from_items`
    (:class:`~surpyval.utils.item_history.ItemHistory`), from each item's
    failure ages with :meth:`from_histories`, or from ``x``, ``i``, ``c``,
    ``n`` arrays with ``surpyval.handle_xicn``, which also takes counts in
    intervals, truncation, windows and covariates; every recurrent ``fit``
    takes the result as ``x`` (``CrowAMSAA.fit(data)``). The constructor,
    ``EventHistoryData(x, i, c, n)``, takes arrays as ``handle_xicn``
    returns them, and checks them as it does: rows of one length, censoring
    codes -1, 0, 1 or 2, positive counts and finite times.

    Examples
    --------

    >>> from surpyval import EventHistoryData
    >>> data = EventHistoryData.from_histories(
    ...     observed={1: [1, 2], 2: [1, 2, 3, 4]},
    ...     right_censored={1: 5, 2: 5},
    ... )
    >>> x, r, d = data.to_xrd()
    >>> x, d
    (array([1., 2., 3., 4., 5.]), array([2, 2, 1, 1, 0]))
    >>> data[0:2]
    EventHistoryData: 1 item, 2 rows
      i    x  c  n
      1  1.0  0  1
      1  2.0  0  1

    As lifetime data: the time to each item's first failure, and the
    times between failures (each repair as good as new), the last gap of
    each item right censored:

    >>> data.to_first_events().x
    array([1., 1.])
    >>> gaps = data.to_gaps()
    >>> gaps.x, gaps.c
    (array([1., 1., 3., 1., 1., 1., 1., 1.]), array([0, 0, 1, 0, 0, 0, 0, 1]))
    """

    # Optional covariate matrix, attached by ``handle_xicn`` for regression.
    Z: npt.NDArray | None = None
    # Gapped (multi-window) observation metadata, attached by ``handle_xicn``
    # when ``windows`` is supplied: ``window_map`` maps each synthetic
    # single-window sub-item id to its ``(real_item, (start, end))`` and
    # ``observation_windows`` keeps the user's original per-item windows. Both
    # stay ``None`` for ordinary single-window data.
    window_map: dict | None = None
    observation_windows: dict | None = None
    # Time-varying covariates, ``{item: StepSchedule or CovariatePath}``,
    # attached by ``from_items`` when an ItemHistory has them; no model uses
    # them yet (#905), so every fit refuses data that has them.
    time_varying_covariates: dict | None = None
    # The name of each column of Z, from the items' covariates={name:
    # value}; they name a fitted model's coefficients.
    covariate_names: list | None = None
    # The item ids were ``items`` before v0.25 (``from_items`` and
    # ``to_items`` are about item records): accepted until v0.26.
    items = RenamedAttribute("ids", removed_in=REMOVED_IN_NEXT)
    # Each item's path through its states, ``{item: (initial_state,
    # changes)}``, attached by ``from_items``; no model uses them yet
    # (#887), so every fit refuses data that has them.
    states: dict | None = None

    def __init__(
        self,
        x: npt.ArrayLike,
        i: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike,
        e: "npt.ArrayLike | None" = None,
        tl: "npt.ArrayLike | None" = None,
        tr: "npt.ArrayLike | None" = None,
    ) -> None:
        self.x = np.atleast_1d(x).copy()  # the caller's stays theirs
        # One i, c or n is every row's, as for handle_xicn
        self.i, self.c, self.n = (
            (
                np.full(self.x.shape[0], value)
                if np.ndim(value) == 0
                else np.atleast_1d(value).copy()
            )
            for value in (i, c, n)
        )
        # Per-row truncation bounds defining each item's observation window.
        # The default window is the whole real line so that no non-negativity
        # assumption is imposed on ``x`` (e.g. a log-intensity variable may be
        # negative); untruncated NHPP fits use 0 as the integration origin.
        n_rows = self.x.shape[0]
        self.tl = (
            np.full(n_rows, -np.inf)
            if tl is None
            else (
                np.broadcast_to(
                    np.atleast_1d(tl).astype(float), (n_rows,)
                ).copy()
                if np.size(tl) == 1
                else np.atleast_1d(tl).astype(float)
            )
        )
        self.tr = (
            np.full(n_rows, np.inf)
            if tr is None
            else (
                np.broadcast_to(
                    np.atleast_1d(tr).astype(float), (n_rows,)
                ).copy()
                if np.size(tr) == 1
                else np.atleast_1d(tr).astype(float)
            )
        )
        # Optional event-type (mark) per observation. When provided this
        # turns the data into a competing-risks recurrent process: each
        # event belongs to one of several mutually-exclusive types. ``None``
        # marks (e.g. the end-of-observation censoring row) are permitted.
        self.e = np.atleast_1d(e).copy() if e is not None else None
        self._check_rows()
        self.ids = np.unique(self.i).tolist()

        if self.x.ndim == 1:
            self.interarrival_times = self.get_interarrival_times()
        else:
            x_midpoints = self.x.copy()
            x_midpoints[self.c == -1, 0] = x_midpoints.min()
            self.midpoints = x_midpoints.mean(axis=1)

    def _check_rows(self) -> None:
        """The arrays checked as ``handle_xicn`` checks them (#925): the
        constructor used to accept rows of different lengths, censoring
        codes such as 7 and negative counts, which went on to give
        negative event counts."""
        from surpyval.utils.data_formats import refuse_boolean_censoring
        from surpyval.utils.recurrent_utils import (
            _check_xicn_lengths,
            _check_xicn_values,
        )

        if self.x.shape[0] == 0:
            raise ValueError("An EventHistoryData needs at least one row.")
        if not np.issubdtype(self.x.dtype, np.number) or not (
            self.x.ndim == 1 or (self.x.ndim == 2 and self.x.shape[1] == 2)
        ):
            raise ValueError(
                "'x' must be numbers, one time per row or [start, end] "
                "pairs; build the data from records with "
                "EventHistoryData.from_items, or from arrays with "
                "surpyval.handle_xicn."
            )
        refuse_boolean_censoring(self.c)
        _check_xicn_lengths(self.x, self.i, self.c, self.n, None)
        for name, values in (("e", self.e), ("tl", self.tl), ("tr", self.tr)):
            if values is not None and values.shape[0] != self.x.shape[0]:
                raise ValueError(f"x and {name} must have the same length")
        _check_xicn_values(
            self.x, self.i, self.c, self.n, self.tl, self.tr, None
        )

    def to_xrd(
        self, estimator: str = "Nelson-Aalen"
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Convert the recurrent event data to xrd format.

        Parameters
        ----------
        estimator : str, optional
            The estimator to use, defaults to "Nelson-Aalen".

        Returns
        -------
        tuple
            A tuple containing unique event times, risk set sizes, and the
            event counts.
        """
        if not hasattr(self, "xrd"):
            # find the total number of times an event occurs at each x
            if self.x.ndim == 2:
                x_out = self.midpoints
            else:
                x_out = self.x

            entry, exit_ = self.item_observation_windows()

            # A finite right-truncation time closes the item's window just
            # as an end-of-observation (c=1) row there would, so it joins
            # the time grid (with no events) exactly as that row would.
            item_tr = np.asarray(self.tr)[self.item_rows()[0]]
            truncated_exit = exit_[np.isfinite(item_tr)]
            x_unique = np.unique(
                np.concatenate([x_out, truncated_exit])
                if truncated_exit.size
                else x_out
            )

            # TODO: consider having the presence of left-censored
            # data use the midpoints instead of the end value of the left
            # censored interval.

            # Counted and risk-set sizes by sorted lookup rather than one
            # pass over the data per time point: that was quadratic, and a
            # simulated MCF has tens of thousands of distinct times.
            is_event = (self.c == 0) | (self.c == 2) | (self.c == -1)
            d = np.bincount(
                np.searchsorted(x_unique, x_out[is_event]),
                weights=self.n[is_event],
                minlength=len(x_unique),
            ).astype(self.n.dtype)
            # Each item is at risk over its observation window (entry,
            # exit]: after its entry, up to and including its exit (see
            # ``item_observation_windows``). An item with a delayed entry
            # only joins the risk set once ``x`` passes its ``tl``, so event
            # times up to that entry see a correspondingly smaller risk set
            # (ignoring ``tl`` here would inflate the MCF); an item entering
            # at an event's time is not at risk for it, as in the NHPP
            # likelihood, which integrates from ``tl``, and the univariate
            # estimators (#658). The count is those entered before ``xi``
            # less those that left before it.
            r = np.searchsorted(
                np.sort(entry), x_unique, side="left"
            ) - np.searchsorted(np.sort(exit_), x_unique, side="left")

            self.xrd = x_unique, r, d
        return self.xrd

    def item_observation_windows(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        The ``(entry, exit)`` observation window of each item, aligned with
        :attr:`items`.

        The entry is the item's left-truncation bound ``tl`` (delayed
        entry); with no truncation it is -inf, so the item is at risk from
        the start. The window is open at the entry: the item is at risk
        only after it, so an event at ``tl`` is outside it (and refused by
        ``handle_xicn``). The exit is the item's last recorded time (its
        last event or its end-of-observation ``c=1`` row) or, when the item
        carries a
        finite right-truncation time ``tr``, that ``tr``: observation of the
        item ends there, exactly as if it had an end-of-observation row at
        ``tr``. This is the same window-close the NHPP likelihoods integrate
        to (see :meth:`get_right_truncation_close`).

        Returns
        -------
        tuple of numpy.ndarray
            ``(entry, exit)``, one value per item.
        """
        x_upper = self.x if self.x.ndim == 1 else self.x[:, 1]
        # From each item's first row and its rows' largest time, without a
        # pass over the rows per item, which was O(items x rows) (#521).
        first, inverse = self.item_rows()
        entry = np.asarray(self.tl, dtype=float)[first]
        last = np.full(len(first), -np.inf)
        np.maximum.at(last, inverse, np.asarray(x_upper, dtype=float))
        tr_item = np.asarray(self.tr, dtype=float)[first]
        exit_ = np.where(np.isfinite(tr_item), np.maximum(last, tr_item), last)
        return entry, exit_

    def item_rows(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        The first row of each item and the item of each row, as positions
        in :attr:`items` (which is sorted).

        Returns
        -------
        tuple of numpy.ndarray
            ``(first, inverse)``: ``first[k]`` is the index of item
            ``items[k]``'s first row, and ``inverse[j]`` the position in
            ``items`` of row ``j``'s item.

        Examples
        --------
        >>> from surpyval import handle_xicn
        >>> data = handle_xicn([1, 2, 3, 4], i=[2, 1, 2, 1])
        >>> data.ids, data.i
        ([1, 2], array([1, 1, 2, 2]))
        >>> data.item_rows()
        (array([0, 2]), array([0, 0, 1, 1]))
        """
        _, first, inverse = np.unique(
            self.i, return_index=True, return_inverse=True
        )
        return first, inverse.reshape(-1)

    @property
    def event_types(self) -> list:
        """
        The distinct event types (marks) present in the data, excluding the
        ``None`` mark used for censored / end-of-observation rows, in the
        order of ``ordered_labels`` (mixed ``str`` and ``int`` marks are
        ordered too). Returns an empty list when the data carries no marks.
        """
        from surpyval.univariate.competing_risks.labels import ordered_labels

        if self.e is None:
            return []
        return ordered_labels(self.e)

    def to_cause_specific_xrd(
        self, cause: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Convert the recurrent event data to xrd format for a single event
        type (cause). The at-risk set ``r`` is shared across all causes (an
        item remains at risk for every cause until it leaves observation);
        only the event count ``d`` is restricted to the requested cause.

        Parameters
        ----------
        cause : object
            The event type to compute the cause-specific counts for. Must be
            one of ``self.event_types``.

        Returns
        -------
        tuple
            A tuple ``(x_unique, r, d_cause)`` where ``d_cause`` counts only
            events of the requested cause and ``r`` is the shared at-risk set.
        """
        from surpyval.univariate.competing_risks.labels import label_mask

        if self.e is None:
            raise ValueError(
                "Data has no event-type marks; pass `e` to compute "
                "cause-specific curves."
            )
        if cause not in self.event_types:
            raise unknown_cause_error(cause, self.event_types)

        # Reuse the shared at-risk set and event-time grid from to_xrd.
        x_unique, r, _ = self.to_xrd()

        if self.x.ndim == 2:
            x_out = self.midpoints
        else:
            x_out = self.x

        observed = (self.c == 0) | (self.c == 2) | (self.c == -1)
        is_cause = label_mask(self.e, cause)
        d_cause = np.array(
            [
                self.n[(x_out == xi) & observed & is_cause].sum()
                for xi in x_unique
            ]
        )
        return x_unique, r, d_cause

    def get_interarrival_times(self) -> npt.NDArray:
        """
        Finds the interarrival times between events for each item. The class
        assumes that the time of the event is cumulative, sometimes it is
        necessary to know the interarrival times of the events. This method
        returns the interarrival times for each item. It is aligned with the
        items attribute.

        Returns
        -------
        numpy.ndarray
            An array of interarrival times.
        """
        _, idx = np.unique(self.i, return_index=True)
        arrival_times = np.split(self.x, idx)[1:]
        interarrival_times = [np.diff(arr, prepend=0) for arr in arrival_times]
        return np.concatenate(interarrival_times)

    def get_previous_x(self, min_x: float = 0) -> npt.NDArray:
        """
        Finds the previous event time for each event. This is useful for
        calculating the time since the last event. This method returns the
        previous event time for each event. It is aligned with the items
        attribute.

        Parameters
        ----------

        min_x : float, optional
            Fallback minimum for the first event of each item. The item's left
            truncation bound is used instead when it is greater, so a
            delayed-entry item's first interval begins at its entry time.

        Returns
        -------

        numpy.ndarray
            An array of previous event times.
        """
        unique_items = np.unique(self.i)
        x_previous = []
        for item in unique_items:
            mask_item = self.i == item
            x_item = self.x[mask_item]
            x_prev_item = np.roll(x_item, shift=1, axis=0)
            # The first interval of an item starts at its observation entry:
            # the left truncation bound when one is given (which may be
            # negative), otherwise the ``min_x`` origin for untruncated data.
            tl_item = float(self.tl[mask_item][0])
            entry = tl_item if np.isfinite(tl_item) else min_x
            x_prev_item[0] = entry
            x_previous.append(x_prev_item)

        return np.concatenate(x_previous)

    def _refuse_crossed_inspections(self) -> None:
        """Refuse an all-failure intensity fit of data whose inspection
        windows of different event types overlap or coincide (#928): the
        counts of each type are of its own inspections, so the total in a
        window is not known (overlapping) or would be read as two counts
        of one window (coinciding)."""
        if self.e is None or self.x.ndim != 2:
            return
        windows = (self.c != 1) & (self.x[:, 0] < self.x[:, 1])
        for item in np.unique(self.i[windows]):
            rows = np.flatnonzero(windows & (self.i == item))
            spans = sorted((tuple(self.x[r]), self.e[r]) for r in rows)
            for (a, kind_a), (b, kind_b) in zip(spans, spans[1:]):
                if b[0] < a[1] and kind_a != kind_b:
                    raise ValueError(
                        f"Item {_plain(item)!r} has inspection windows of "
                        "different event types that overlap or coincide "
                        f"({kind_a!r} "
                        f"in [{a[0]:g}, {a[1]:g}], {kind_b!r} in "
                        f"[{b[0]:g}, {b[1]:g}]), so a fit of every failure "
                        "cannot read its counts: fit "
                        "data.without('event_types') (which combines the "
                        "failures found at one inspection) or each type "
                        "with data.for_event_type(t)."
                    )

    def split_for_nhpp_likelihood(self) -> dict:
        """Split the data into the pieces every NHPP-style likelihood
        needs: per-censoring-type event and previous-event times, the
        right-truncation window close, and the row masks (so regression
        variants can gather covariates the same way). This preamble used
        to be copy-pasted between the plain and proportional-intensity
        NHPP fitters and drifted (#288); it now lives here once (#296).
        """
        self._refuse_crossed_inspections()
        x, c, n = self.x, self.c, self.n
        x_prev = self.get_previous_x()

        has_interval = 2 in c
        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None

        x_prev_l = (
            x_prev if x_prev.ndim == 1 else x_prev[:, 0]
        )  # not x[:, 0] (#288)
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else None
        # Interval counts need 2-D data, for the right columns. This was
        # an assert, which gave a bare AssertionError (and vanishes under
        # ``python -O``, #658).
        if has_interval:
            if x_r is None or x_prev_r is None:
                raise ValueError(
                    "Interval-counted rows (c=2) need x as [start, end] "
                    "pairs (a 2-D x, one row per count); x here is a single "
                    "time per row. Give each row as [start, end], with "
                    "[t, t] for an exact event (c=0)."
                )
            prev = x_prev_r
        else:
            prev = x_prev_l

        mask_o = c == 0
        mask_right = c == 1
        mask_left = c == -1
        mask_i = c == 2

        empty = np.array([])
        out = {
            "x_o": x_l[mask_o] if mask_o.any() else empty,
            "x_o_prev": prev[mask_o] if mask_o.any() else empty,
            "x_right": x_l[mask_right] if mask_right.any() else empty,
            "x_right_prev": prev[mask_right] if mask_right.any() else empty,
            "x_left": x_l[mask_left] if mask_left.any() else empty,
            # A left-censored count is the item's first row, so its previous
            # time is the item's entry (``tl``, or the origin 0): the count
            # covers (entry, x], not (0, x].
            "x_left_prev": x_prev_l[mask_left] if mask_left.any() else empty,
            "n_left": n[mask_left] if mask_left.any() else empty,
            "x_i_l": x_l[mask_i] if mask_i.any() else empty,
            "x_i_r": (
                x_r[mask_i] if (mask_i.any() and x_r is not None) else empty
            ),
            "n_i": n[mask_i] if mask_i.any() else empty,
            "mask_o": mask_o,
            "mask_right": mask_right,
            "mask_left": mask_left,
            "mask_i": mask_i,
        }
        # Right window-close: for items with a finite right-truncation
        # time ``tr`` the intensity is integrated out to ``tr`` rather
        # than merely to the last observed event / censoring row. Empty
        # for untruncated data.
        (
            out["x_close_last"],
            out["x_close_tr"],
            out["close_idx"],
        ) = self.get_right_truncation_close()
        return out

    def get_right_truncation_close(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Per-item integration bounds for the NHPP likelihood's right
        window-close.

        The NHPP integral runs from each item's entry time (its left
        truncation ``tl``, handled by :meth:`get_previous_x`) to the time its
        observation window closes. Historically that close was only known from
        an explicit right-censoring (``c=1``) row, so the integral stopped at
        the item's last recorded time. When an item instead carries a finite
        right-truncation time ``tr`` the window closes there, and the integral
        must be extended from the last in-window time out to ``tr``.

        Returns three aligned arrays ``(x_last, x_close, rep_idx)`` with one
        entry per item whose ``tr`` is finite: ``x_last`` is the item's last
        in-window time (its last event or right-censoring row), ``x_close`` is
        its ``tr``, and ``rep_idx`` is a representative row index for the item
        (so per-item covariates ``Z`` can be gathered). Items with the default
        ``tr = inf`` are omitted, so untruncated data yields empty arrays and
        contributes nothing to the integral. Adding
        ``cif(x_last) - cif(x_close)`` to the log-likelihood therefore extends
        the telescoped integral to ``tr`` (and is exactly zero when a ``c=1``
        row already sits at ``tr``, so the two ways of closing the window never
        double-count).

        Returns
        -------
        tuple of numpy.ndarray
            ``(x_last, x_close, rep_idx)`` as described above.
        """
        x_upper = self.x if self.x.ndim == 1 else self.x[:, 1]
        x_last, x_close, rep_idx = [], [], []
        for item in self.ids:
            mask = self.i == item
            tr_item = float(self.tr[mask][0])
            if np.isfinite(tr_item):
                x_last.append(float(x_upper[mask].max()))
                x_close.append(tr_item)
                rep_idx.append(int(np.flatnonzero(mask)[0]))
        return (
            np.array(x_last, dtype=float),
            np.array(x_close, dtype=float),
            np.array(rep_idx, dtype=int),
        )

    def get_events_for_item(
        self, item: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Get all events for a specific item or subject.

        Parameters
        ----------
        item : int or str
            The id of the item or subject.

        Returns
        -------
        tuple
            A tuple containing event times, censoring information and
            frequencies for the specified item.
        """
        mask = self.i == item
        return self.x[mask], self.c[mask], self.n[mask]

    # Until v0.25 the times to the first events came back with neither
    # the items' delayed entry, ids nor covariates (#926).
    get_times_to_first_events = RenamedAttribute(
        "to_first_events", removed_in=REMOVED_IN_NEXT
    )

    def to_first_events(self) -> "SurpyvalData":
        """
        Each item's time to its first failure, as lifetime data, one
        :class:`~surpyval.utils.item_lifetime.ItemLifetime` per item: its
        first failure (observed; between two inspections, interval
        censored; before its first look, left censored), or right censored
        at the end of its observation where it has none, with its delayed
        entry (left truncation), id, covariates, time-varying covariates
        and the first failure's event type. Fit it with any lifetime
        model: ``Weibull.fit(data.to_first_events())``.

        Examples
        --------
        >>> from surpyval import EventHistoryData, ItemHistory
        >>> data = EventHistoryData.from_items([
        ...     ItemHistory("A", [5, 9], right_censored=12, left_truncation=2),
        ...     ItemHistory("B", [4], right_censored=10),
        ...     ItemHistory("C", right_censored=8),
        ... ])
        >>> for item in data.to_first_events().to_items():
        ...     print(item)
        ItemLifetime('A', observed=5.0, left_truncation=2.0)
        ItemLifetime('B', observed=4.0)
        ItemLifetime('C', right_censored=8.0)
        """
        firsts = [_first_event(history) for history in self.to_items()]
        if self.e is None:
            return SurpyvalData.from_items(firsts)
        masked = [
            first.id
            for first in firsts
            if first.outcome != "right_censored" and first.event_type is None
        ]
        if masked:
            warnings.warn(
                f"{len(masked)} first failure{'' if len(masked) == 1 else 's'}"
                " of no known event type (masked causes): items "
                f"{named(masked)}, their type not recorded or the first "
                "failure found with failures of other types at one "
                "inspection. A fit of one type of first event is "
                "data.for_event_type(t).to_first_events().",
                MaskedCauseWarning,
                stacklevel=caller_stacklevel(),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaskedCauseWarning)
            return SurpyvalData.from_items(firsts)

    def to_gaps(self) -> "SurpyvalData":
        """
        The times between each item's failures, as lifetime data, each
        repair taken to make the item as good as new (a renewal process):
        one :class:`~surpyval.utils.item_lifetime.ItemLifetime` per gap,
        with id ``(item, k)`` for the item's ``k``-th gap, its covariates
        and the event type of the failure that ended it. The first gap runs
        from new to the first failure, left truncated at the item's entry
        where observation began late (taking it to have had no failure
        before then); each item's last gap, from its last failure to the
        end of its observation, is right censored. Fit it with any lifetime
        model: ``Weibull.fit(data.to_gaps())``.

        It needs each failure's age, so failures counted between
        inspections or before a first look are refused (their gaps are not
        known; :meth:`to_first_events` takes them), as are time-varying
        covariates, which follow an item's age rather than the time since
        its last repair, and two failures at one age.

        Examples
        --------
        >>> from surpyval import EventHistoryData, ItemHistory
        >>> data = EventHistoryData.from_items([
        ...     ItemHistory("A", [5, 9], right_censored=12),
        ...     ItemHistory("B", [4], right_censored=10),
        ... ])
        >>> for gap in data.to_gaps().to_items():
        ...     print(gap)
        ItemLifetime(('A', 1), observed=5.0)
        ItemLifetime(('A', 2), observed=4.0)
        ItemLifetime(('A', 3), right_censored=3.0)
        ItemLifetime(('B', 1), observed=4.0)
        ItemLifetime(('B', 2), right_censored=6.0)
        """
        gaps: list = []
        for history in self.to_items():
            gaps += _gaps(history)
        return SurpyvalData.from_items(gaps)

    def _first_event_times(self) -> "SurpyvalData":
        """Each item's first row, grouped (the renewal fits' start)."""
        x_ttff = np.array([self.x[self.i == item][0] for item in self.ids])
        c_ttff = np.array([self.c[self.i == item][0] for item in self.ids])
        n_ttff = np.array([self.n[self.i == item][0] for item in self.ids])
        return SurpyvalData(x_ttff, c_ttff, n_ttff)

    @classmethod
    def from_histories(
        cls,
        observed: "dict | None" = None,
        right_censored: "dict | None" = None,
        *,
        left_truncation: Any = None,
        right_truncation: Any = None,
        covariates: "dict | None" = None,
        gaps: bool = False,
    ) -> "EventHistoryData":
        """
        Recurrent event data from each item's history, keyed by the item's
        id: the ages at which it failed, ``observed``, and the age at which
        its observation ended, ``right_censored``. An item with no end of
        observation is taken to have been observed until its last failure;
        an item that has not failed has an end of observation and no
        failures.

        Parameters
        ----------
        observed : dict, optional
            ``{item: [age, age, ...]}``, the item's age (time since new or
            first observed) at each failure; a single failure may be a
            number.
        right_censored : dict, optional
            ``{item: age}``, the item's age when its observation ended
            (when it was last checked), at or after its last failure.
        left_truncation : dict or scalar, optional
            ``{item: age}``, the age at which an item came under
            observation (its failures before then are unknown), or one
            number for every item.
        right_truncation : dict or scalar, optional
            ``{item: age}``, the age after which an item's failures would
            not have been seen, or one number for every item.
        covariates : dict, optional
            ``{item: row}``, each item's covariates (a number, a row of
            them, or ``{name: value}``, which names them), for every item;
            a proportional intensity ``fit`` takes them as ``Z``.
        gaps : bool, default False
            Read ``observed`` as the times between failures (the first
            from the start of observation: ``left_truncation``, else 0) and
            ``right_censored`` as the time from the last failure to the end
            of observation; each item's ages are their running totals.

        Returns
        -------
        EventHistoryData

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"M1": [420, 1310, 2050], "M2": [980, 2400]},
        ...     right_censored={"M1": 2600, "M2": 2900, "M3": 3100},
        ... )
        >>> data.ids
        ['M1', 'M2', 'M3']
        >>> data.x[data.i == "M1"], data.c[data.i == "M1"]
        (array([ 420., 1310., 2050., 2600.]), array([0, 0, 0, 1]))

        The same history, recorded as the time between failures:

        >>> again = EventHistoryData.from_histories(
        ...     observed={"M1": [420, 890, 740], "M2": [980, 1420]},
        ...     right_censored={"M1": 550, "M2": 500, "M3": 3100},
        ...     gaps=True,
        ... )
        >>> bool((again.x == data.x).all())
        True
        """
        from surpyval.utils.item_history import ItemHistory

        observed = _history_dict("observed", observed)
        right_censored = _history_dict("right_censored", right_censored)
        items = list(dict.fromkeys([*observed, *right_censored]))
        if not items:
            raise ValueError(
                "Give each item's failures (observed) or end of observation "
                "(right_censored), keyed by the item's id."
            )
        starts = _per_item("left_truncation", left_truncation, items, None)
        ends = _per_item("right_truncation", right_truncation, items, None)
        rows = None
        if covariates is not None:
            rows = _item_covariate_rows(
                _history_dict("covariates", covariates), items
            )
        histories = []
        for k, item in enumerate(items):
            ages = _failure_times(item, observed.get(item, []))
            end = right_censored.get(item)
            if end is not None:
                end = _single_age("right_censored", item, end)
            if gaps:
                if (ages < 0).any() or (end is not None and end < 0):
                    raise ValueError(
                        f"Item {item!r}: with gaps=True the times are times "
                        "between failures, which cannot be negative."
                    )
                origin = 0.0 if starts[k] is None else starts[k]
                ages = origin + np.cumsum(ages)
                if end is not None:
                    end = (ages[-1] if len(ages) else origin) + end
            histories.append(
                ItemHistory(
                    item,
                    observed=ages,
                    right_censored=end,
                    left_truncation=starts[k],
                    right_truncation=ends[k],
                    covariates=None if rows is None else rows[k],
                )
            )
        return cls.from_items(histories)

    @classmethod
    def from_items(cls, items: Any) -> "EventHistoryData":
        """
        The dataset of several items' histories, each an
        :class:`~surpyval.utils.item_history.ItemHistory`: their failures
        (exact, in intervals or before an age), ends of observation,
        truncation and covariates. Covariates, given for one item, are
        needed for every item, with the same number of values; so are
        time-varying covariates, which no model uses yet (#905).

        Examples
        --------
        >>> from surpyval import ItemHistory, EventHistoryData
        >>> data = EventHistoryData.from_items([
        ...     ItemHistory("A", observed=[5, 9], right_censored=12),
        ...     ItemHistory("B", interval_censored=[[0, 10]],
        ...                 interval_censored_counts=3, right_censored=10),
        ... ])
        >>> data.c, data.n
        (array([0, 0, 1, 2, 1]), array([1, 1, 1, 3, 1]))
        """
        from surpyval.utils.item_history import ItemHistory
        from surpyval.utils.recurrent_utils import handle_xicn

        items = list(items)
        if not items:
            raise ValueError("Give at least one ItemHistory.")
        for history in items:
            if not isinstance(history, ItemHistory):
                raise TypeError(
                    "from_items takes ItemHistory objects; got "
                    f"{type(history).__name__}."
                )
        from surpyval.utils.item_history import (
            check_item_ids,
            covariate_names_of,
            covariate_row_in,
        )

        ids = [history.id for history in items]
        check_item_ids(ids, "ItemHistory")
        static = _all_or_none(items, "covariates", lambda h: h.covariates)
        names = covariate_names_of(items)
        varying = _all_or_none(
            items,
            "time_varying_covariates",
            lambda h: (
                None
                if h.time_varying_covariates is None
                else [0.0] * h.time_varying_covariates.p
            ),
        )
        typed = [h.has_event_types for h in items]
        events = [
            bool(len(h.observed) or len(h.interval_censored))
            or h.left_censored is not None
            for h in items
        ]
        masked = []
        if any(typed):
            # Failures whose type was not recorded: masked causes
            masked = [
                h.id
                for h, t, ev in zip(items, typed, events)
                if ev
                and (
                    not t
                    or any(
                        k is None
                        for k in (
                            *h.observed_event_types,
                            *h.interval_censored_event_types,
                        )
                    )
                    or (
                        h.left_censored is not None
                        and h.left_censored_event_types is None
                    )
                )
            ]
            if masked:
                warn_masked(len(masked), "items " + named(masked))
        x: list = []
        i: list = []
        c: list = []
        n: list = []
        e: list = []
        tl: list = []
        tr: list = []
        Z: list = []
        for history in items:
            xs, cs, ns, es = history._rows()
            x += xs
            c += cs
            n += ns
            e += es
            i += [history.id] * len(xs)
            tl += [
                (
                    -np.inf
                    if history.left_truncation is None
                    else history.left_truncation
                )
            ] * len(xs)
            tr += [
                (
                    np.inf
                    if history.right_truncation is None
                    else history.right_truncation
                )
            ] * len(xs)
            if static:
                Z += [covariate_row_in(history, names)] * len(xs)
        x_arr = np.array(x, dtype=float)
        if not (np.isin(c, (-1, 2))).any():
            x_arr = x_arr[:, 0]
        truncated = np.isfinite(tl).any() or np.isfinite(tr).any()
        with warnings.catch_warnings():
            # Warned of above, by item
            warnings.simplefilter("ignore", MaskedCauseWarning)
            data = handle_xicn(
                x_arr,
                np.array(i),
                np.array(c),
                np.array(n),
                tl=np.array(tl) if truncated else None,
                tr=np.array(tr) if truncated else None,
                Z=np.array(Z) if static else None,
                e=_object_array(e) if any(typed) else None,
            )
        if static:
            data.covariate_names = names
        if varying:
            data.time_varying_covariates = {
                history.id: history.time_varying_covariates
                for history in items
            }
        with_states = [history.has_states for history in items]
        if any(with_states):
            missing = [
                h.id for h, given in zip(items, with_states) if not given
            ]
            if missing:
                raise ValueError(
                    "States were given for some items but not for "
                    f"{missing!r}: every item needs its 'initial_state'."
                )
            data.states = {
                history.id: (history.initial_state, history.state_changes)
                for history in items
            }
        return data

    def to_histories(self) -> dict:
        """
        The data as each item's history, the form :meth:`from_histories`
        reads: ``"observed"`` (``{item: ages at its failures}``) and
        ``"right_censored"`` (``{item: age at the end of observation}``,
        for the items with one), with ``"left_truncation"`` /
        ``"right_truncation"`` for the items that are truncated and
        ``"covariates"`` where the data has them. So
        ``EventHistoryData.from_histories(**data.to_histories())`` is the
        same data. Interval or left censored counts, repeated rows,
        event types and observation windows have no place in a history,
        and are refused.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12, "B": 4}
        ... )
        >>> data.to_histories()["right_censored"]
        {'A': 12.0, 'B': 4.0}
        """
        problems = []
        if np.ndim(self.x) != 1 or np.isin(self.c, (-1, 2)).any():
            problems.append("left or interval censored counts")
        if (np.asarray(self.n) != 1).any():
            problems.append("rows counted more than once")
        if self.e is not None:
            problems.append("event types")
        if self.window_map is not None:
            problems.append("observation windows")
        if self.time_varying_covariates is not None:
            problems.append("time-varying covariates")
        if self.states is not None:
            problems.append("states")
        if problems:
            raise ValueError(
                "This data has " + " and ".join(problems) + ", which a "
                "history has no place for: use its arrays (x, i, c, n, ...)."
            )
        observed: dict = {}
        right_censored: dict = {}
        left: dict = {}
        right: dict = {}
        covariates: dict = {}
        Z = None if self.Z is None else np.asarray(self.Z, dtype=float)
        for item in self.ids:
            rows = self.i == item
            key = item.item() if isinstance(item, np.generic) else item
            observed[key] = self.x[rows & (self.c == 0)].astype(float)
            ends = self.x[rows & (self.c == 1)]
            if len(ends):
                right_censored[key] = float(ends[0])
            left[key] = float(self.tl[rows][0])
            right[key] = float(self.tr[rows][0])
            if Z is not None:
                covariates[key] = _named_row(Z[rows], 0, self.covariate_names)
        out: dict = {"observed": observed, "right_censored": right_censored}
        # Only the items with a bound: the others have none (-inf / inf)
        left = {key: age for key, age in left.items() if np.isfinite(age)}
        right = {key: age for key, age in right.items() if np.isfinite(age)}
        if left:
            out["left_truncation"] = left
        if right:
            out["right_truncation"] = right
        if Z is not None:
            out["covariates"] = covariates
        return out

    def to_items(self) -> list:
        """
        The data as one :class:`~surpyval.utils.item_history.ItemHistory`
        per item, in the order of ``ids``:
        ``EventHistoryData.from_items(data.to_items())`` is the same data.
        Observation windows have no place in an item's history, and are
        refused.
        """
        self._no_windows()
        # The rows of each item, found once for all items by one sort; the
        # groups come in the order of np.unique, as self.ids does
        _, inverse = np.unique(self.i, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse))
        groups = np.split(order, bounds[:-1])
        prepared = self._prepared()
        return [
            self._history_at(key, rows, prepared)
            for key, rows in zip(self.ids, groups)
        ]

    _PARTS = HISTORY_PARTS

    def _records(self) -> tuple[list, list]:
        records = self.to_items()
        return records, [1] * len(records)

    def _from_records(self, records: list, counts: list) -> "EventHistoryData":
        return EventHistoryData.from_items(records)

    def item(self, id: Any) -> Any:
        """
        The history of the item ``id``, as an
        :class:`~surpyval.utils.item_history.ItemHistory`.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"M1": [420, 1310], "M2": [980]},
        ...     right_censored={"M1": 2600, "M2": 2900},
        ... )
        >>> data.item("M2")
        ItemHistory(
            'M2',
            observed=[980.0],
            right_censored=2900.0
        )
        """
        from surpyval.utils.item_history import item_id

        self._no_windows()
        key = item_id(id)
        if key not in self.ids:
            raise KeyError(f"No item {key!r} in the data.")
        return self._history_at(
            key, np.flatnonzero(self.i == key), self._prepared()
        )

    def _no_windows(self) -> None:
        if self.window_map is not None:
            raise ValueError(
                "This data has observation windows, which an ItemHistory has "
                "no place for: use its arrays."
            )

    def _prepared(self) -> tuple:
        """The arrays every item's record reads, made once for all items."""
        x = self.x if np.ndim(self.x) == 2 else np.column_stack([self.x] * 2)
        Z = None if self.Z is None else np.asarray(self.Z, dtype=float)
        return x, Z

    def _history_at(self, key: Any, rows: np.ndarray, prepared: tuple) -> Any:
        from surpyval.utils.item_history import ItemHistory

        x, Z = prepared
        c, n, xi = self.c[rows], self.n[rows], x[rows]
        e = None if self.e is None else self.e[rows]
        left = xi[c == -1]
        end = xi[c == 1]
        tl, tr = self.tl[rows][0], self.tr[rows][0]
        state_args: dict = {}
        if self.states is not None:
            initial, changes = self.states[key]
            state_args = {
                "initial_state": initial,
                **_state_fields(changes),
            }
        return ItemHistory(
            key,
            observed=xi[c == 0, 0],
            right_censored=float(end[0, 0]) if len(end) else None,
            interval_censored=xi[c == 2],
            interval_censored_counts=n[c == 2],
            left_censored=float(left[0, 0]) if len(left) else None,
            left_censored_counts=int(n[c == -1][0]) if len(left) else None,
            left_truncation=float(tl) if np.isfinite(tl) else None,
            right_truncation=float(tr) if np.isfinite(tr) else None,
            covariates=(
                None
                if Z is None
                else _named_row(Z[rows], 0, self.covariate_names)
            ),
            time_varying_covariates=(
                None
                if self.time_varying_covariates is None
                else self.time_varying_covariates[key]
            ),
            observed_event_types=None if e is None else list(e[c == 0]),
            interval_censored_event_types=(
                None if e is None or not (c == 2).any() else list(e[c == 2])
            ),
            left_censored_event_types=(
                None if e is None or not (c == -1).any() else e[c == -1][0]
            ),
            **state_args,
        )

    def __len__(self) -> int:
        """The number of rows (events and ends of observation), as
        ``SurpyvalData``'s length is its rows; ``len(data.ids)`` is the
        number of items."""
        return len(self.x)

    def summary(self) -> str:
        """
        A short description of the data: its items and events, and what
        else it holds (counts, truncation, event types, covariates,
        time-varying covariates, states).

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"M1": [420, 1310], "M2": [980]},
        ...     right_censored={"M1": 2600, "M2": 2900},
        ... )
        >>> print(data.summary())
        EventHistoryData: 2 items, 5 rows
          events: 3 observed
          ends of observation: 2
        """
        c, n = np.asarray(self.c), np.asarray(self.n)
        kinds = [
            (f"{int(n[c == 0].sum())} observed", (c == 0).any()),
            (f"{int(n[c == 2].sum())} in intervals", (c == 2).any()),
            (f"{int(n[c == -1].sum())} before a first look", (c == -1).any()),
        ]
        lines = [
            f"EventHistoryData: {_plural(len(self.ids), 'item')}, "
            f"{_plural(len(self.x), 'row')}",
            (
                "  events: " + ", ".join(k for k, given in kinds if given)
                if any(given for _, given in kinds)
                else "  events: none"
            ),
            f"  ends of observation: {int((c == 1).sum())}",
        ]
        lines += _held(self, rows=None)
        return "\n".join(lines)

    def __getitem__(self, index: Any) -> "EventHistoryData":
        # As row numbers, so one row stays a row: a scalar index used to
        # turn an interval row [l, r] into two times and drop covariate
        # columns (#925).
        rows = np.atleast_1d(np.arange(len(self.x))[index])
        if rows.size == 0:
            raise ValueError(
                "The index selects no rows; an EventHistoryData needs at "
                "least one."
            )
        index = rows
        out = EventHistoryData(
            self.x[index],
            self.i[index],
            self.c[index],
            self.n[index],
            None if self.e is None else self.e[index],
            tl=self.tl[index],
            tr=self.tr[index],
        )
        # Covariates, schedules and states go with the rows they belong
        # to: a fit of a slice must see what a fit of the whole would.
        if self.Z is not None:
            out.Z = np.asarray(self.Z)[index]
            out.covariate_names = self.covariate_names
        for name in ("time_varying_covariates", "states"):
            held = getattr(self, name)
            if held is not None:
                setattr(out, name, {i: held[i] for i in out.ids if i in held})
        return out

    def __iter__(self) -> Any:
        # A stateless iterator: the old hand-rolled _index version broke
        # nested iteration over the same object.
        return zip(self.x, self.i, self.c, self.n)

    def __repr__(self) -> str:
        """Its size and its first rows, as a table: ``i``, ``x`` (a time,
        or ``[start, end]``), ``c``, ``n``, and the truncation, event types
        and covariates where it has them; :meth:`summary` describes all of
        it."""
        shown = min(len(self.x), _REPR_ROWS)
        columns: dict = {
            "i": [repr(_plain(v)) for v in self.i[:shown]],
            "x": [_time_text(v) for v in self.x[:shown]],
            "c": [str(int(v)) for v in self.c[:shown]],
            "n": [_time_text(v) for v in self.n[:shown]],
        }
        if np.isfinite(self.tl).any():
            columns["tl"] = [_time_text(v) for v in self.tl[:shown]]
        if np.isfinite(self.tr).any():
            columns["tr"] = [_time_text(v) for v in self.tr[:shown]]
        if self.e is not None:
            columns["e"] = [repr(_plain(v)) for v in self.e[:shown]]
        if self.Z is not None:
            Z = np.asarray(self.Z, dtype=float)
            Z = Z[:, None] if Z.ndim == 1 else Z
            names = self.covariate_names or [
                f"Z{j}" for j in range(Z.shape[1])
            ]
            for j, name in enumerate(names):
                columns[name] = [_time_text(v) for v in Z[:shown, j]]
        widths = {
            k: max(len(k), *(len(v) for v in vals))
            for k, vals in columns.items()
        }
        lines = [
            f"EventHistoryData: {_plural(len(self.ids), 'item')}, "
            f"{_plural(len(self.x), 'row')}",
            "  " + "  ".join(k.rjust(widths[k]) for k in columns),
        ]
        for r in range(shown):
            lines.append(
                "  "
                + "  ".join(columns[k][r].rjust(widths[k]) for k in columns)
            )
        if len(self.x) > shown:
            lines.append(
                f"  ... {_plural(len(self.x) - shown, 'more row')}; "
                "summary() describes it all"
            )
        return "\n".join(lines)


_REPR_ROWS = 10


def _time_text(value: Any) -> str:
    """A time, count or ``[start, end]`` pair as the repr shows it."""
    if np.ndim(value):
        return "[" + ", ".join(_time_text(v) for v in value) + "]"
    value = _plain(value)
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return str(value)


def _history_end(history: Any) -> "float | None":
    """Where ``history``'s observation ends: its ``right_censored``, else
    its ``right_truncation``, else its last failure."""
    for end in (history.right_censored, history.right_truncation):
        if end is not None:
            return float(end)
    ages = [*history.observed, *history.interval_censored[:, 1]]
    if history.left_censored is not None:
        ages.append(history.left_censored)
    return float(max(ages)) if ages else None


def _lifetime_parts(history: Any) -> dict:
    """What a lifetime made from ``history`` keeps of it: its covariates
    (named, where they are)."""
    covariates = history._fields().get("covariates")
    return {} if covariates is None else {"covariates": covariates}


def _first_event(history: Any) -> Any:
    """``history``'s time to its first failure, as an ItemLifetime: the
    earliest of its failures (seen, between inspections or before a first
    look). Where failures that could each be the first are of different
    types (a seal and a bearing found at one inspection), it is known only
    to lie between the earliest of their lower bounds and the first age
    by which one was known, and its type is not known: a masked cause
    (#928)."""
    from surpyval.utils.item_lifetime import ItemLifetime

    parts = _lifetime_parts(history)
    parts["left_truncation"] = history.left_truncation
    parts["time_varying_covariates"] = history.time_varying_covariates
    # (lower, upper, type) of each failure that could be the first
    candidates: list = []
    if history.left_censored is not None:
        candidates.append(
            (-np.inf, history.left_censored, history.left_censored_event_types)
        )
    observed_types = history.observed_event_types
    for k, age in enumerate(history.observed):
        kind = None if observed_types is None else observed_types[k]
        candidates.append((float(age), float(age), kind))
    interval_types = history.interval_censored_event_types
    for k, (lower, upper) in enumerate(history.interval_censored):
        kind = None if interval_types is None else interval_types[k]
        candidates.append((float(lower), float(upper), kind))
    if not candidates:
        return ItemLifetime(
            history.id, right_censored=_history_end(history), **parts
        )
    first = min(upper for _, upper, _ in candidates)
    possible = [
        (lower, upper, kind)
        for lower, upper, kind in candidates
        if lower < first or upper == first
    ]
    kinds = {kind for _, _, kind in possible}
    event_type = next(iter(kinds)) if len(kinds) == 1 else None
    lower = min(low for low, _, _ in possible)
    if lower == -np.inf:
        when: dict = {"left_censored": first}
    elif lower == first:
        when = {"observed": first}
    else:
        when = {"interval_censored": [lower, first]}
    return ItemLifetime(history.id, event_type=event_type, **when, **parts)


def _gaps(history: Any) -> list:
    """``history``'s times between failures, as ItemLifetimes."""
    from surpyval.utils.item_lifetime import ItemLifetime

    if len(history.interval_censored) or history.left_censored is not None:
        raise ValueError(
            f"Item {history.id!r} has failures counted between inspections "
            "(or before its first look), so the times between its failures "
            "are not known: to_gaps needs each failure's age. "
            "to_first_events() takes such items."
        )
    if history.time_varying_covariates is not None:
        raise ValueError(
            f"Item {history.id!r} has time-varying covariates, which follow "
            "its age, not the time since its last repair: to_gaps has no "
            "place for them; use without('time_varying_covariates') first."
        )
    order = np.argsort(history.observed, kind="stable")
    ages = [float(history.observed[k]) for k in order]
    types = history.observed_event_types
    types = [None] * len(ages) if types is None else [types[k] for k in order]
    twice = sorted({a for a, b in zip(ages, ages[1:]) if a == b})
    if twice:
        raise ValueError(
            f"Item {history.id!r} has more than one failure at age "
            f"{twice[0]:g}: the time between them is 0, which a lifetime "
            "cannot be."
        )
    parts = _lifetime_parts(history)
    gaps = []
    previous = 0.0
    for k, (age, event_type) in enumerate(zip(ages, types)):
        entry = (
            {"left_truncation": history.left_truncation}
            if k == 0 and history.left_truncation is not None
            else {}
        )
        gaps.append(
            ItemLifetime(
                (history.id, k + 1),
                observed=age - previous,
                event_type=event_type,
                **entry,
                **parts,
            )
        )
        previous = age
    end = _history_end(history)
    if end is not None and (not ages or end > previous):
        entry = (
            {"left_truncation": history.left_truncation}
            if not ages and history.left_truncation is not None
            else {}
        )
        gaps.append(
            ItemLifetime(
                (history.id, len(ages) + 1),
                right_censored=end - previous,
                **entry,
                **parts,
            )
        )
    return gaps


def _history_dict(name: str, value: Any) -> dict:
    """``value``, a dict keyed by item id, or empty for None."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"'{name}' must be a dict keyed by each item's id, such as "
            "{'M1': ..., 'M2': ...}; for flat arrays use the fit's x, i, c "
            "or handle_xicn."
        )
    return value


def _failure_times(item: Any, ages: Any) -> np.ndarray:
    """An item's failure ages (or gaps) as a 1-D float array."""
    try:
        arr = np.atleast_1d(np.asarray(ages, dtype=float))
    except (TypeError, ValueError):
        raise ValueError(f"Item {item!r}: 'observed' must be numbers.")
    if arr.ndim != 1:
        raise ValueError(f"Item {item!r}: 'observed' must be a list of times.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Item {item!r}: 'observed' times must be finite.")
    return arr


def _single_age(name: str, item: Any, value: Any) -> float:
    """One finite age for ``item``."""
    try:
        age = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Item {item!r}: '{name}' must be one number, the age at the end "
            "of observation."
        )
    if not np.isfinite(age):
        raise ValueError(f"Item {item!r}: '{name}' must be finite.")
    return age


def _per_item(name: str, value: Any, items: list, none: Any) -> list:
    """A bound for each item, in the order of ``items``: ``none`` where
    not given; a dict keyed by item, or one number for every item."""
    if value is None:
        return [none] * len(items)
    if isinstance(value, dict):
        unknown = [item for item in value if item not in items]
        if unknown:
            raise ValueError(
                f"'{name}' names items with no failures and no end of "
                f"observation: {unknown!r}."
            )
        return [
            none if item not in value else _single_age(name, item, value[item])
            for item in items
        ]
    if np.ndim(value) != 0:
        raise ValueError(
            f"'{name}' must be a dict keyed by item, or one number for every "
            "item."
        )
    return [float(value)] * len(items)


def _object_array(values: list) -> np.ndarray:
    """``values`` as a 1-D object array (types may be text or numbers)."""
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _all_or_none(items: list, name: str, get: Callable) -> bool:
    """Whether the ItemHistory objects have ``name``: all of them, or none
    (refused otherwise), with values of one length."""
    values = [get(history) for history in items]
    given = [value is not None for value in values]
    if not any(given):
        return False
    if not all(given):
        missing = [h.id for h, g in zip(items, given) if not g]
        raise ValueError(
            f"'{name}' must be given for every item; missing {missing!r}."
        )
    if len({len(np.atleast_1d(value)) for value in values}) > 1:
        raise ValueError(
            f"Every item's '{name}' must have the same number of values."
        )
    return True


def _item_covariate_rows(covariates: dict, items: list) -> list:
    """Each item's covariates as a 1-D row, in the order of ``items``:
    every item needs them, with the same number of values."""
    unknown = [item for item in covariates if item not in items]
    if unknown:
        raise ValueError(
            "'covariates' names items with no failures and no end of "
            f"observation: {unknown!r}."
        )
    missing = [item for item in items if item not in covariates]
    if missing:
        raise ValueError(
            f"'covariates' must be given for every item; missing {missing!r}."
        )
    rows = []
    for item in items:
        value = covariates[item]
        if isinstance(value, dict) or type(value).__name__ == "Series":
            rows.append(value)  # named: the ItemHistory reads it
            continue
        try:
            row = np.atleast_1d(np.asarray(value, dtype=float))
        except (TypeError, ValueError):
            raise ValueError(f"Item {item!r}: 'covariates' must be numbers.")
        if row.ndim != 1 or not np.isfinite(row).all():
            raise ValueError(
                f"Item {item!r}: 'covariates' must be one finite number or "
                "a row of them (they are constant over the item's history)."
            )
        rows.append(row)
    if len({len(row) for row in rows}) > 1:
        raise ValueError(
            "Every item's 'covariates' must have the same length."
        )
    return rows


# The arguments a recurrent fit takes its data in, besides ``x``.
_RECURRENT_ARGUMENTS = ("i", "c", "n", "t", "tl", "tr", "windows")

F = TypeVar("F", bound=Callable[..., Any])


def accepts_event_history_data(method: F) -> F:
    """Let a recurrent fitter's ``fit`` take a :class:`EventHistoryData`
    as its ``x``: ``CrowAMSAA.fit(data)``. The data is the whole of it, so
    the other data arguments (``i``, ``c``, ``n``, ``t``, ``tl``, ``tr``,
    ``windows``, and ``Z`` or ``e`` where the data carries them) must be
    left out. The data's truncation, covariates or event types that the
    fit has no argument for are refused, not dropped."""
    signature = inspect.signature(method)
    names = set(signature.parameters)
    if "x" not in names:
        return method

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        data = bound.arguments.get("x")
        if not isinstance(data, EventHistoryData):
            return method(*args, **kwargs)
        given = [
            name
            for name in _RECURRENT_ARGUMENTS + ("Z", "e")
            if bound.arguments.get(name) is not None
            and not (name == "Z" and data.Z is None)
            and not (name == "e" and data.e is None)
        ]
        if given:
            raise TypeError(
                "x is a EventHistoryData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than a EventHistoryData)."
            )
        if data.window_map is not None:
            raise ValueError(
                "This EventHistoryData was built with observation windows: "
                "pass its arrays and the windows to fit instead."
            )
        if (
            data.Z is None
            and "Z" in names
            and bound.arguments.get("Z") is None
            and signature.parameters["Z"].default is inspect.Parameter.empty
        ):
            raise TypeError(
                "This fit needs covariates, Z, and the EventHistoryData "
                "holds none: build it with from_histories(..., "
                "covariates={item: row, ...})."
            )
        filled: dict = {"x": data.x, "i": data.i, "c": data.c, "n": data.n}
        lost = []
        left = bool(np.isfinite(data.tl).any())
        right = bool(np.isfinite(data.tr).any())
        fill_tl = left and "tl" in names
        fill_tr = right and "tr" in names
        if (left and not fill_tl) or (right and not fill_tr):
            if "t" in names:
                filled["t"] = np.column_stack([data.tl, data.tr])
            else:
                if left and not fill_tl:
                    lost.append("left truncation")
                if right and not fill_tr:
                    lost.append("right truncation")
        else:
            if fill_tl:
                filled["tl"] = data.tl
            if fill_tr:
                filled["tr"] = data.tr
        if data.Z is not None:
            if "Z" in names:
                filled["Z"] = data.Z
            else:
                lost.append("covariates")
        if data.e is not None:
            if "e" in names:
                filled["e"] = data.e
            else:
                lost.append("event types")
        lost += _unmodelled(data)
        if lost:
            raise ValueError(
                f"This fit takes no {' or '.join(lost)}, which the "
                "EventHistoryData holds."
            )
        bound.arguments.update(filled)
        return named_fit(
            lambda: method(*bound.args, **bound.kwargs),
            data.covariate_names if "Z" in filled else None,
        )

    setattr(wrapper, "_accepts_event_history_data", True)
    return wrapper  # type: ignore[return-value]


def _unmodelled(data: Any) -> list:
    """What ``data`` holds that no model uses yet."""
    out = []
    if getattr(data, "time_varying_covariates", None) is not None:
        out.append("time-varying covariates (no model uses them yet, #905)")
    if getattr(data, "states", None) is not None:
        out.append("states (no model uses them yet, #887)")
    return out


def refuses_unmodelled_data(method: F) -> F:
    """Refuse, in a ``fit_from_recurrent_data``, data that holds what no
    model uses yet (time-varying covariates, #905; states, #887): fitting
    it as if they were not there would be a different model from the one
    asked for."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data = kwargs.get("data", args[1] if len(args) > 1 else None)
        unmodelled = _unmodelled(data)
        if unmodelled:
            raise ValueError(
                "This fit takes no "
                + " or ".join(unmodelled)
                + ", which the EventHistoryData holds."
            )
        return method(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _state_fields(changes: list) -> dict:
    """An item's changes of state as the ItemHistory arguments that make
    them (None for a kind with none)."""
    return {
        "observed_states": {
            u: st for k, _, u, st in changes if k == "observed"
        }
        or None,
        "interval_censored_states": [
            [lo, u, st] for k, lo, u, st in changes if k == "interval_censored"
        ]
        or None,
        "left_censored_states": {
            u: st for k, _, u, st in changes if k == "left_censored"
        }
        or None,
    }


def _held(data: Any, rows: "int | None") -> list:
    """The summary lines for what a data object holds beyond its times."""
    from collections import Counter

    lines = []
    tl, tr = np.asarray(data.tl), np.asarray(data.tr)
    if np.isfinite(tl).any() or np.isfinite(tr).any():
        lines.append(
            "  truncation: "
            + ", ".join(
                f"{int(np.isfinite(b).sum())} {side}"
                for side, b in (("left", tl), ("right", tr))
                if np.isfinite(b).any()
            )
        )
    if getattr(data, "e", None) is not None:
        counts = Counter(v for v in data.e if v is not None)
        failed = np.asarray(data.c) != 1
        masked = int(sum(1 for v, f in zip(data.e, failed) if f and v is None))
        lines.append(
            "  event types: "
            + ", ".join(
                [f"{k} ({v})" for k, v in sorted(counts.items(), key=str)]
                + ([f"not recorded ({masked})"] if masked else [])
            )
        )
    if getattr(data, "Z", None) is not None:
        Z = np.asarray(data.Z)
        width = 1 if Z.ndim == 1 else Z.shape[1]
        line = f"  covariates: {width} column{'' if width == 1 else 's'}"
        if getattr(data, "covariate_names", None) is not None:
            line += " (" + ", ".join(data.covariate_names) + ")"
        lines.append(line)
    if getattr(data, "time_varying_covariates", None) is not None:
        kinds = Counter(
            type(v).__name__ for v in data.time_varying_covariates.values()
        )
        lines.append(
            "  time-varying covariates: "
            + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
        )
    if getattr(data, "states", None) is not None:
        found = sorted(
            {st for _, changes in data.states.values() for *_, st in changes}
            | {initial for initial, _ in data.states.values()},
            key=str,
        )
        lines.append("  states: " + ", ".join(map(str, found)))
    return lines
