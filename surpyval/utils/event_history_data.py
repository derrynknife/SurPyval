"""Event history data: each item's history over time, the data of the
recurrent event models (and, as they arrive, of the multi-state and joint
models). Formerly ``RecurrentEventData`` in
``surpyval.utils.recurrent_event_data`` (renamed in v0.25)."""

from __future__ import annotations

import functools
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

from surpyval.utils.data_reshape import HISTORY_PARTS, ReshapeMixin
from surpyval.utils.deprecation import REMOVED_IN_NEXT, RenamedAttribute
from surpyval.utils.surpyval_data import (
    SurpyvalData,
    _named_row,
    _plain,
    named_fit,
    with_data_unit,
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
    # Each item's attributes (``{name: value}``, what describes the item
    # without being fitted), in the order of ``ids``, from the records or
    # ``from_dataframe``; None where the data has none.
    attributes: "list | None" = None
    # The unit of the ages (``"hour"``), a label as a model's ``unit`` is;
    # None where unknown (#945).
    unit: "str | None" = None

    def __init__(
        self,
        x: npt.ArrayLike,
        i: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike,
        e: "npt.ArrayLike | None" = None,
        tl: "npt.ArrayLike | None" = None,
        tr: "npt.ArrayLike | None" = None,
        unit: Any = None,
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
        if unit is not None:
            from surpyval.utils.item_history import _unit

            self.unit = _unit("(data)", unit)

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
            x_unique = np.unique(x_out)

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
        :attr:`ids`.

        The entry is the item's left-truncation bound ``tl`` (delayed
        entry); with no truncation it is -inf, so the item is at risk from
        the start. The window is open at the entry: the item is at risk
        only after it, so an event at ``tl`` is outside it (and refused by
        ``handle_xicn``). The exit is the item's last recorded time: its
        end-of-observation ``c=1`` row, or its last event where it has none
        (watched until it last failed). An item's right truncation ``tr``
        is not where its observation ends: it is the age by which the item
        failed, which is why it is in the data (#924; see
        :meth:`selection_terms`).

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
        return entry, last

    def item_rows(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        The first row of each item and the item of each row, as positions
        in :attr:`ids` (which is sorted).

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
        # Right truncation (#924): each truncated item is in the data
        # because it failed by its tr, so its likelihood is conditioned on
        # that. Empty for untruncated data.
        out.update(self.selection_terms())
        return out

    def selection_terms(self) -> dict:
        """
        The pieces of the right-truncation term of an intensity likelihood
        (#924). An item right truncated at ``tr`` is in the data because it
        had a failure in its window up to ``tr``, ``(entry, tr]``, so its
        likelihood is its history's divided by the chance of that:
        ``P(N(entry, tr] >= 1) = 1 - exp(-(Lambda(tr) - Lambda(entry)))``,
        ``entry`` its left truncation (else 0). Where its observation ended
        at ``T`` before ``tr`` with no failure seen (data cut by
        ``censor_at``), the failure that put it in the data came after
        ``T``, which multiplies the history's chance by ``P(N(T, tr] >=
        1)``.

        Returns
        -------
        dict
            ``sel_from``, ``sel_to`` and ``sel_idx`` (each truncated item's
            entry, ``tr`` and a row of it, for its covariates), and
            ``gap_from``, ``gap_to`` and ``gap_idx`` (the end of
            observation, ``tr`` and a row, of each item seen to ``T < tr``
            with no failure). Empty arrays for untruncated data.
        """
        x_upper = self.x if self.x.ndim == 1 else self.x[:, 1]
        first, inverse = self.item_rows()
        tr = np.asarray(self.tr, dtype=float)[first]
        tl = np.asarray(self.tl, dtype=float)[first]
        last = np.full(len(first), -np.inf)
        np.maximum.at(last, inverse, np.asarray(x_upper, dtype=float))
        seen = np.zeros(len(first), dtype=bool)
        np.logical_or.at(seen, inverse, np.asarray(self.c) != 1)
        truncated = np.isfinite(tr)
        gap = truncated & ~seen & (last < tr)
        entry = np.where(np.isfinite(tl), tl, 0.0)
        return {
            "sel_from": entry[truncated],
            "sel_to": tr[truncated],
            "sel_idx": np.asarray(first)[truncated].astype(int),
            "gap_from": last[gap],
            "gap_to": tr[gap],
            "gap_idx": np.asarray(first)[gap].astype(int),
        }

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
        unit: Any = None,
        **unknown: Any,
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
            ``{item: age}``, the age by which an item failed, which is why
            it is in the data (right truncation, #924: not the end of its
            observation, which is ``right_censored``), or one number for
            every item.
        covariates : dict, optional
            ``{item: row}``, each item's covariates (a number, a row of
            them, or ``{name: value}``, which names them), for every item;
            a proportional intensity ``fit`` takes them as ``Z``.
        gaps : bool, default False
            Read ``observed`` as the times between failures (the first
            from the start of observation: ``left_truncation``, else 0) and
            ``right_censored`` as the time from the last failure to the end
            of observation; each item's ages are their running totals.
        unit : str, optional
            The unit of the ages, kept with the data.

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

        if unknown:
            from surpyval.utils.record_keywords import (
                FROM_HISTORIES_ALIASES,
                accepted_names,
                refuse_unknown_keywords,
            )

            refuse_unknown_keywords(
                "EventHistoryData.from_histories",
                unknown,
                accepted_names(EventHistoryData.from_histories),
                FROM_HISTORIES_ALIASES,
            )
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
        return cls.from_items(histories, unit=unit)

    @classmethod
    def from_dataframe(
        cls,
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
    ) -> "EventHistoryData":
        """
        The data from a log held in a :class:`pandas.DataFrame`, one row
        per event (a maintenance log: the item and its age at each
        failure), naming the columns as every DataFrame entry point does
        (``x_col``, ``i_col``, ..., the ``fit`` argument with ``_col``).
        Each item's history is built as an
        :class:`~surpyval.utils.item_history.ItemHistory`, so every check
        names the item.

        Parameters
        ----------
        df : pandas.DataFrame
            The log.
        i_col : str
            The column of item ids.
        x_col : str, optional
            Each event's age (the item's own age); or give ``xl_col`` and
            ``xr_col``.
        xl_col, xr_col : str, optional
            The inspections between which a row's failures were counted
            (equal for an exact event).
        c_col : str, optional
            What each row is: 0 an event (the default), 1 the end of the
            item's observation (one row per item at most), -1 failures
            before a first look, 2 failures counted between inspections.
            Left out, a row with equal (or one) ages is an event and one
            with different ages a count.
        n_col : str, optional
            How many failures a counted row holds (c = 2 or -1).
        e_col : str, optional
            The type of each failure (a failure mode or cause); a blank
            cell on an end-of-observation row is nothing, on a failure a
            type not recorded (a masked cause).
        tl_col, tr_col : str or number, optional
            Each item's left truncation (the age it came under
            observation) and right truncation (the age by which it
            failed, which is why it is in the data), the same on every
            row of the item (or one number for every item).
        end_col : str, optional
            Each item's age at the end of its observation, on its rows
            (the same on each, blank cells left out); an alternative to
            a ``c = 1`` row. Left out, observation ends at the last
            failure.
        ends : dict or pandas.Series, optional
            ``{item: age}``, the end of each item's observation, for the
            items it names: a way to give the items that never failed,
            which have no row in the log.
        Z_cols : list of str, optional
            The covariate columns (numbers; the same on every row of an
            item), which name the covariates.
        attribute_cols : list of str, optional
            Columns describing the items without being fitted (a site, a
            product line), the same on every row of an item, kept as
            each record's ``attributes``.
        unit : str, optional
            The unit of the ages (``"hours"``), kept with the data.

        Returns
        -------
        EventHistoryData

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import EventHistoryData
        >>> log = pd.DataFrame({
        ...     "machine": ["M1", "M1", "M2"],
        ...     "hours": [420, 1310, 980],
        ...     "part": ["seal", "bearing", "seal"],
        ... })
        >>> data = EventHistoryData.from_dataframe(
        ...     log, i_col="machine", x_col="hours", e_col="part",
        ...     ends={"M1": 2600, "M2": 2900, "M3": 3100},
        ... )
        >>> data.ids, data.c.tolist()
        (['M1', 'M2', 'M3'], [0, 0, 1, 0, 1, 1])
        """
        from surpyval.utils.data_frames import history_from_frame

        return history_from_frame(
            cls,
            df,
            i_col=i_col,
            x_col=x_col,
            xl_col=xl_col,
            xr_col=xr_col,
            c_col=c_col,
            n_col=n_col,
            e_col=e_col,
            tl_col=tl_col,
            tr_col=tr_col,
            end_col=end_col,
            ends=ends,
            Z_cols=Z_cols,
            attribute_cols=attribute_cols,
            unit=unit,
        )

    def to_dataframe(self) -> Any:
        """
        The rows as a :class:`pandas.DataFrame`: ``i``, ``x`` (or ``xl``
        and ``xr`` where any row counts failures between inspections),
        ``c``, ``n``, ``tl`` and ``tr`` (where any bound is finite),
        ``e`` (where it has event types), the covariates by name and the
        attributes. ``EventHistoryData.from_dataframe(data.to_dataframe(),
        i_col="i", x_col="x", c_col="c", ...)`` is the same data.
        Observation windows, time-varying covariates and states have no
        place in it, and are refused.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12}
        ... )
        >>> data.to_dataframe()
           i     x  c  n
        0  A   5.0  0  1
        1  A   9.0  0  1
        2  A  12.0  1  1
        """
        from surpyval.utils.data_frames import history_to_frame

        return history_to_frame(self)

    @classmethod
    def from_items(
        cls, items: Any, unit: Any = None, **unknown: Any
    ) -> "EventHistoryData":
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
        from surpyval.utils.record_keywords import (
            FROM_ITEMS_ALIASES,
            refuse_unknown_keywords,
        )
        from surpyval.utils.recurrent_utils import handle_xicn
        from surpyval.utils.surpyval_data import _record_list

        refuse_unknown_keywords(
            "EventHistoryData.from_items",
            unknown,
            ["items", "unit"],
            {
                k: v
                for k, v in FROM_ITEMS_ALIASES.items()
                if k not in ("n", "count", "weights")
            },
        )
        items = _record_list(items, ItemHistory, "EventHistoryData")
        from surpyval.utils.item_history import (
            check_item_ids,
            covariate_names_of,
            covariate_row_in,
        )

        ids = [history.id for history in items]
        check_item_ids(ids, "ItemHistory")
        from surpyval.utils.item_history import units_of

        unit = units_of(items, unit, "items' ages")
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
                # (a covariate not recorded, NaN, is kept with the data,
                # though the recurrent fits refuse it: handle_xicn checks
                # the arrays a fit is given)
                Z=np.nan_to_num(np.array(Z), nan=0.0) if static else None,
                e=_object_array(e) if any(typed) else None,
                unit=unit,
            )
        if static:
            rows = {
                history.id: covariate_row_in(history, names)
                for history in items
            }
            data.Z = np.array([rows[_plain(k)] for k in data.i], dtype=float)
        if static:
            data.covariate_names = names
        if any(h.attributes is not None for h in items):
            held = {h.id: h.attributes for h in items}
            data.attributes = [held[_plain(i)] for i in data.ids]
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
                "This data has "
                + " and ".join(problems)
                + ", which the {item: ages} dict form of from_histories "
                "cannot hold: use data.to_items() (one ItemHistory per "
                "item, which holds everything), or its arrays (x, i, c, n, "
                "...)."
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
            attributes=(
                None
                if self.attributes is None
                else self.attributes[self.ids.index(key)]
            ),
            unit=self.unit,
            **state_args,
        )

    def __len__(self) -> int:
        """The number of rows (events and ends of observation), as
        ``SurpyvalData``'s length is its rows; ``len(data.ids)`` is the
        number of items."""
        return len(self.x)

    def summary(self) -> str:
        """
        A short description of the data: its items and rows, its failures
        (a count of ``n`` in an interval is ``n`` failures), and what else
        it holds (truncation and covariates, counted in items; event
        types, counted in failures; time-varying covariates, states).

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
        from surpyval.utils.data_display import heading, held_lines

        c, n = np.asarray(self.c), np.asarray(self.n)
        kinds = [
            (f"{int(n[c == 0].sum())} observed", (c == 0).any()),
            (f"{int(n[c == 2].sum())} in intervals", (c == 2).any()),
            (f"{int(n[c == -1].sum())} before a first look", (c == -1).any()),
        ]
        lines = [
            heading(self, recurrent=True),
            (
                "  events: " + ", ".join(k for k, given in kinds if given)
                if any(given for _, given in kinds)
                else "  events: none"
            ),
            f"  ends of observation: {int((c == 1).sum())}",
        ]
        lines += held_lines(self, recurrent=True)
        return "\n".join(lines)

    @property
    def n_items(self) -> int:
        """The number of items (``len(data)`` is the number of rows)."""
        return len(self.ids)

    def records(self) -> Any:
        """
        The items as :class:`~surpyval.utils.item_history.ItemHistory`
        records, one at a time, in the order of ``ids`` (a generator:
        :meth:`to_items` builds the list).

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"M1": [420, 1310], "M2": [980]},
        ...     right_censored={"M1": 2600, "M2": 2900},
        ... )
        >>> [len(item.observed) for item in data.records()]
        [2, 1]
        """
        self._no_windows()
        _, inverse = np.unique(self.i, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse))
        prepared = self._prepared()
        for key, rows in zip(self.ids, np.split(order, bounds[:-1])):
            yield self._history_at(key, rows, prepared)

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
            unit=self.unit,
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
        if self.attributes is not None:
            by_item = dict(zip(self.ids, self.attributes))
            out.attributes = [by_item[i] for i in out.ids]
        return out

    def _covariate_levels(self, column: int) -> list:
        first, _ = self.item_rows()
        Z = np.asarray(self.Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        return Z[first, column].tolist()

    def _rows_of_items(self, keep: np.ndarray) -> "EventHistoryData":
        _, inverse = np.unique(self.i, return_inverse=True)
        return self[np.flatnonzero(np.asarray(keep)[inverse.reshape(-1)])]

    @property
    def attribute_names(self) -> list:
        """The names of the attributes any item has (an empty list where
        the data has none)."""
        from surpyval.utils.surpyval_data import _attribute_names

        return _attribute_names(self.attributes)

    def to_dict(self) -> dict:
        """
        The data as a dictionary of plain Python values, strict JSON when
        dumped: its ``"class"``, the ``x``, ``i``, ``c``, ``n``, ``tl``
        and ``tr`` arrays as lists, and the event types, covariates (with
        their names), time-varying covariates (step schedules; a
        ``CovariatePath`` is refused) and states where it has them. Data
        built with observation windows is refused: keep its arrays and
        windows, and build it again with ``handle_xicn``. Non-finite
        values are written as ``null`` and recorded under
        ``"non_finite"`` (see :doc:`/surpyval.serialisation`).
        ``surpyval.from_dict`` reads it too.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12}
        ... )
        >>> sorted(data.to_dict())
        ['c', 'class', 'i', 'n', 'non_finite', 'tl', 'tr', 'x']
        """
        from surpyval.serialisation import encode_non_finite
        from surpyval.utils.data_json import label, schedule_dict

        self._no_windows()
        data: dict = {
            "class": "EventHistoryData",
            "x": self.x.tolist(),
            "i": [label(v) for v in self.i],
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "tl": self.tl.tolist(),
            "tr": self.tr.tolist(),
        }
        if self.e is not None:
            data["e"] = [label(v) for v in self.e]
        if self.Z is not None:
            data["Z"] = np.asarray(self.Z).tolist()
        if self.covariate_names is not None:
            data["covariate_names"] = list(self.covariate_names)
        if self.attributes is not None:
            from surpyval.utils.surpyval_data import _attributes_json

            data["attributes"] = _attributes_json(self.attributes)
        if self.time_varying_covariates is not None:
            data["time_varying_covariates"] = [
                schedule_dict(self.time_varying_covariates[item], False)
                for item in self.ids
            ]
        if self.states is not None:
            data["states"] = [
                {
                    "initial_state": label(initial),
                    "changes": [
                        [kind, lo, hi, label(st)]
                        for kind, lo, hi, st in changes
                    ],
                }
                for initial, changes in (
                    self.states[item] for item in self.ids
                )
            ]
        unit = getattr(self, "unit", None)
        if unit is not None:
            data["unit"] = unit
        return encode_non_finite(data)

    @classmethod
    def from_dict(cls, data_dict: dict) -> "EventHistoryData":
        """
        The data back from its :meth:`to_dict` dictionary.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12}
        ... )
        >>> EventHistoryData.from_dict(data.to_dict()).x
        array([ 5.,  9., 12.])
        """
        from surpyval.serialisation import decode_non_finite
        from surpyval.utils.data_json import label_back, schedule_back

        if not isinstance(data_dict, dict):
            raise ValueError(
                "Expected a serialised EventHistoryData dict, got "
                f"{type(data_dict).__name__}."
            )
        tag = data_dict.get("class")
        if tag is not None and tag != "EventHistoryData":
            raise ValueError(
                f"This dictionary was written by {tag}, not "
                f"EventHistoryData: read it with {tag}.from_dict, or "
                "surpyval.from_dict."
            )
        for key in ("x", "i", "c", "n"):
            if key not in data_dict:
                raise ValueError(
                    f"A serialised EventHistoryData needs {key!r}; this "
                    "dictionary has no such entry."
                )
        data = decode_non_finite(dict(data_dict))
        i = np.empty(len(data["i"]), dtype=object)
        i[:] = [label_back(v) for v in data["i"]]
        if all(isinstance(v, (int, float)) for v in i):
            i = np.array(i.tolist())
        elif all(isinstance(v, str) for v in i):
            i = np.array(i.tolist(), dtype=str)
        e = None
        if "e" in data:
            e = np.empty(len(data["e"]), dtype=object)
            e[:] = [label_back(v) for v in data["e"]]
        out = cls(
            np.array(data["x"], dtype=float),
            i,
            np.array(data["c"]),
            np.array(data["n"]),
            e,
            tl=np.array(data["tl"], dtype=float) if "tl" in data else None,
            tr=np.array(data["tr"], dtype=float) if "tr" in data else None,
        )
        if "Z" in data:
            out.Z = np.array(data["Z"], dtype=float)
            out.covariate_names = data.get("covariate_names")
        if "time_varying_covariates" in data:
            from surpyval.utils.item_history import _time_varying

            out.time_varying_covariates = {
                item: _time_varying(item, schedule_back(held))[0]
                for item, held in zip(out.ids, data["time_varying_covariates"])
            }
        if "attributes" in data:
            from surpyval.utils.surpyval_data import _attributes_back

            out.attributes = _attributes_back(data["attributes"])
        if "states" in data:
            out.states = {
                item: (
                    label_back(entry["initial_state"]),
                    [
                        (kind, lo, hi, label_back(st))
                        for kind, lo, hi, st in entry["changes"]
                    ],
                )
                for item, entry in zip(out.ids, data["states"])
            }
        if data.get("unit") is not None:
            out.unit = data["unit"]
        return out

    def to_json(self, filepath: "str | Path | None" = None) -> "str | None":
        """
        The data as strict JSON text (:meth:`to_dict` dumped), or written
        to the file ``filepath``.

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12}
        ... )
        >>> data.to_json().startswith('{"class": "EventHistoryData"')
        True
        """
        from surpyval.utils.data_json import dump

        return dump(self.to_dict(), filepath)

    @classmethod
    def from_json(cls, source: "str | Path") -> "EventHistoryData":
        """
        The data back from JSON text (a string starting with ``{``), or
        from the file holding it (``surpyval.from_json`` reads it too).

        Examples
        --------
        >>> from surpyval import EventHistoryData
        >>> data = EventHistoryData.from_histories(
        ...     observed={"A": [5, 9]}, right_censored={"A": 12}
        ... )
        >>> EventHistoryData.from_json(data.to_json()).ids
        ['A']
        """
        from surpyval.utils.data_json import load

        return cls.from_dict(load(source))

    def __iter__(self) -> Any:
        # A stateless iterator: the old hand-rolled _index version broke
        # nested iteration over the same object.
        return zip(self.x, self.i, self.c, self.n)

    def __repr__(self) -> str:
        """Its size and its first rows, as a table: ``i``, ``x`` (a time,
        or ``[start, end]``), ``c``, ``n``, and the truncation, event types
        and covariates where it has them; :meth:`summary` describes all of
        it."""
        from surpyval.utils.data_display import table_repr

        return table_repr(self, recurrent=True)

    def _repr_html_(self) -> str:
        from surpyval.utils.data_display import html_repr

        return html_repr(self, recurrent=True)

    def __dir__(self) -> list:
        from surpyval.utils.data_display import HISTORY_INTERNALS, public_dir

        return public_dir(self, HISTORY_INTERNALS)


def _held(data: Any, rows: "int | None") -> list:
    """The summary lines for what a data object holds beyond its times
    (kept for callers of the old name; see ``data_display.held_lines``)."""
    from surpyval.utils.data_display import held_lines

    return held_lines(data, recurrent=rows is None)


def _history_end(history: Any) -> "float | None":
    """Where ``history``'s observation ends: its ``right_censored``, else
    its ``right_truncation``, else its last failure."""
    if history.right_censored is not None:
        return float(history.right_censored)
    ages = [*history.observed, *history.interval_censored[:, 1]]
    if history.left_censored is not None:
        ages.append(history.left_censored)
    return float(max(ages)) if ages else None


def _lifetime_parts(history: Any) -> dict:
    """What a lifetime made from ``history`` keeps of it: its covariates
    (named, where they are), attributes and unit."""
    fields = history._fields()
    return {
        name: fields[name]
        for name in ("covariates", "attributes", "unit")
        if name in fields
    }


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
    # In the data because it failed by tr: its first failure came by then
    parts["right_truncation"] = history.right_truncation
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


def _first_gap_bounds(history: Any) -> dict:
    """The truncation of an item's first gap: from its entry, and, where
    the item is in the data because it failed by ``tr``, by ``tr`` (the
    selection is of its first failure)."""
    bounds = {}
    if history.left_truncation is not None:
        bounds["left_truncation"] = history.left_truncation
    if history.right_truncation is not None:
        bounds["right_truncation"] = history.right_truncation
    return bounds


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
        entry = _first_gap_bounds(history) if k == 0 else {}
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
        entry = _first_gap_bounds(history) if not ages else {}
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
                "x is an EventHistoryData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than an EventHistoryData)."
            )
        if data.window_map is not None:
            raise ValueError(
                "This EventHistoryData was built with observation windows: "
                "pass its arrays and the windows to fit instead."
            )
        if bound.arguments.get("Z") is None:
            _check_needs_covariates(data, signature)
        filled: dict = {"x": data.x, "i": data.i, "c": data.c, "n": data.n}
        lost: list = []
        left = bool(np.isfinite(data.tl).any())
        right = bool(np.isfinite(data.tr).any())
        fill_tl = left and "tl" in names
        fill_tr = right and "tr" in names
        if (left and not fill_tl) or (right and not fill_tr):
            if "t" in names:
                filled["t"] = np.column_stack([data.tl, data.tr])
            else:
                if left and not fill_tl:
                    lost.append(("left truncation", "truncation"))
                if right and not fill_tr:
                    lost.append(("right truncation", "truncation"))
        else:
            if fill_tl:
                filled["tl"] = data.tl
            if fill_tr:
                filled["tr"] = data.tr
        if data.Z is not None and "Z" in names:
            filled["Z"] = data.Z
        if data.e is not None and "e" in names:
            filled["e"] = data.e
        _refuse_what_fit_cannot_take(data, names, lost)
        bound.arguments.update(filled)
        return with_data_unit(
            named_fit(
                lambda: method(*bound.args, **bound.kwargs),
                data.covariate_names if "Z" in filled else None,
            ),
            data,
        )

    setattr(wrapper, "_accepts_event_history_data", True)
    return wrapper  # type: ignore[return-value]


def _unmodelled(data: Any) -> list:
    """What ``data`` holds that no model uses yet, as ``(what, part)``:
    the ``part`` of :meth:`~ReshapeMixin.without` that leaves it out."""
    out = []
    if getattr(data, "time_varying_covariates", None) is not None:
        out.append(
            (
                "time-varying covariates (no model uses them yet, #905)",
                "time_varying_covariates",
            )
        )
    if getattr(data, "states", None) is not None:
        out.append(("states (no model uses them yet, #887)", "states"))
    return out


def _check_needs_covariates(data: Any, signature: inspect.Signature) -> None:
    """Refuse data with no covariates for a fit that needs them (its
    ``Z`` has no default)."""
    parameter = signature.parameters.get("Z")
    if (
        getattr(data, "Z", None) is None
        and parameter is not None
        and parameter.default is inspect.Parameter.empty
    ):
        raise TypeError(
            "This fit needs covariates, Z, and the EventHistoryData "
            "holds none: build it with from_histories(..., "
            "covariates={item: row, ...})."
        )


def _refuse_what_fit_cannot_take(
    data: Any, names: "set[str]", lost: "list | None" = None
) -> None:
    """Refuse ``data`` that holds what a fit taking the arguments
    ``names`` cannot use -- covariates without a ``Z``, event types
    without an ``e``, what no model uses yet, and the ``lost`` already
    found -- naming the way to fit without them (#959). Fitting as if
    they were not there would be a different model from the one the data
    describes."""
    lost = list(lost or [])
    if getattr(data, "Z", None) is not None and "Z" not in names:
        lost.append(("covariates", "covariates"))
    if getattr(data, "e", None) is not None and "e" not in names:
        lost.append(("event types", "event_types"))
    lost += _unmodelled(data)
    if not lost:
        return
    parts = list(dict.fromkeys(part for _, part in lost))
    hint = "data.without(" + ", ".join(repr(p) for p in parts) + ")"
    if "event_types" in parts:
        hint += ", or data.for_event_type(t) for the failures of one type"
    raise ValueError(
        "This fit takes no "
        + " or ".join(what for what, _ in lost)
        + ", which the EventHistoryData holds. To fit without them, use "
        + hint
        + "."
    )


def refuses_unmodelled_data(method: F) -> F:
    """Refuse, in a ``fit_from_recurrent_data``, data that holds what the
    fitter's ``fit`` has no argument for -- covariates (``Z``), event
    types (``e``) -- or what no model uses yet (time-varying covariates,
    #905; states, #887), as ``fit(data)`` does (#959): fitting it as if
    they were not there would be a different model from the one the data
    describes. What the fitter takes is read from its ``fit``, so the two
    ways of fitting refuse the same data."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data = kwargs.get("data", args[1] if len(args) > 1 else None)
        fit = getattr(args[0], "fit", None) if args else None
        signature = inspect.signature(fit) if fit is not None else None
        names = set(signature.parameters) if signature is not None else set()
        if signature is not None:
            _check_needs_covariates(data, signature)
        _refuse_what_fit_cannot_take(data, names)
        return with_data_unit(method(*args, **kwargs), data)

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
