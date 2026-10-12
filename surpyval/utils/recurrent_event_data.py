from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import unknown_cause_error


class RecurrentEventData:
    """
    A class to handle and manipulate recurrent event data. Recurrent events are
    those that can occur more than once for each subject or item.

    Build one from each item's history with :meth:`from_histories`, or
    from ``x``, ``i``, ``c``, ``n`` arrays with ``surpyval.handle_xicn``;
    both validate the input, and every recurrent ``fit`` takes the result
    as ``x`` (``CrowAMSAA.fit(data)``). The constructor itself does no
    validation.

    Examples
    --------

    >>> import numpy as np
    >>> from surpyval import RecurrentEventData
    >>> x = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    >>> c = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])
    >>> n = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>> data = RecurrentEventData(x, i, c, n)
    >>> data.to_xrd()
    (array([1, 2, 3, 4, 5]), array([2, 2, 2, 2, 2]), array([2, 2, 1, 1, 0]))
    >>> data[0:2]
    RecurrentEventData(
        x=[1 2],
        i=[1 1],
        c=[0 0],
        n=[1 1]
    )
    >>> data.get_times_to_first_events()
    SurpyvalData(
        x=array([1.]),
        c=array([0]),
        n=array([2]),
        t=array([[-inf,  inf]])
    )
    >>> data.get_interarrival_times()
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
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
        self.x = np.atleast_1d(x)
        self.i = np.atleast_1d(i)
        self.c = np.atleast_1d(c)
        self.n = np.atleast_1d(n)
        # Per-row truncation bounds defining each item's observation window.
        # The default window is the whole real line so that no non-negativity
        # assumption is imposed on ``x`` (e.g. a log-intensity variable may be
        # negative); untruncated NHPP fits use 0 as the integration origin.
        n_rows = self.x.shape[0]
        self.tl = (
            np.full(n_rows, -np.inf)
            if tl is None
            else np.atleast_1d(tl).astype(float)
        )
        self.tr = (
            np.full(n_rows, np.inf)
            if tr is None
            else np.atleast_1d(tr).astype(float)
        )
        # Optional event-type (mark) per observation. When provided this
        # turns the data into a competing-risks recurrent process: each
        # event belongs to one of several mutually-exclusive types. ``None``
        # marks (e.g. the end-of-observation censoring row) are permitted.
        self.e = np.atleast_1d(e) if e is not None else None
        self.items = np.unique(self.i).tolist()

        if self.x.ndim == 1:
            self.interarrival_times = self.get_interarrival_times()
        else:
            x_midpoints = self.x.copy()
            x_midpoints[self.c == -1, 0] = x_midpoints.min()
            self.midpoints = x_midpoints.mean(axis=1)

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
        >>> data.items, data.i
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

    def split_for_nhpp_likelihood(self) -> dict:
        """Split the data into the pieces every NHPP-style likelihood
        needs: per-censoring-type event and previous-event times, the
        right-truncation window close, and the row masks (so regression
        variants can gather covariates the same way). This preamble used
        to be copy-pasted between the plain and proportional-intensity
        NHPP fitters and drifted (#288); it now lives here once (#296).
        """
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
        for item in self.items:
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

    def get_times_to_first_events(self) -> "SurpyvalData":
        """
        Get the times to the first events for each item or subject. In the
        estimation of recurrent or renewal events it can be helpful to know
        the distribution of the times to the first event per item. This method
        returns the times to the first events for each item. It is aligned with
        the items attribute.

        Returns
        -------

        SurpyvalData
            A transformed dataset containing times to the first events.
        """
        x_ttff = np.array([self.x[self.i == item][0] for item in self.items])
        c_ttff = np.array([self.c[self.i == item][0] for item in self.items])
        n_ttff = np.array([self.n[self.i == item][0] for item in self.items])
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
    ) -> "RecurrentEventData":
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
            ``{item: row}``, each item's covariates (a number, or a row of
            them), for every item; a proportional intensity ``fit`` takes
            them as ``Z``.
        gaps : bool, default False
            Read ``observed`` as the times between failures (the first
            from the start of observation: ``left_truncation``, else 0) and
            ``right_censored`` as the time from the last failure to the end
            of observation; each item's ages are their running totals.

        Returns
        -------
        RecurrentEventData

        Examples
        --------
        >>> from surpyval import RecurrentEventData
        >>> data = RecurrentEventData.from_histories(
        ...     observed={"M1": [420, 1310, 2050], "M2": [980, 2400]},
        ...     right_censored={"M1": 2600, "M2": 2900, "M3": 3100},
        ... )
        >>> data.items
        ['M1', 'M2', 'M3']
        >>> data.x[data.i == "M1"], data.c[data.i == "M1"]
        (array([ 420., 1310., 2050., 2600.]), array([0, 0, 0, 1]))

        The same history, recorded as the time between failures:

        >>> again = RecurrentEventData.from_histories(
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
    def from_items(cls, items: Any) -> "RecurrentEventData":
        """
        The dataset of several items' histories, each an
        :class:`~surpyval.utils.item_history.ItemHistory`: their failures
        (exact, in intervals or before an age), ends of observation,
        truncation and covariates. Covariates, given for one item, are
        needed for every item, with the same number of values; so are
        time-varying covariates, which no model uses yet (#905).

        Examples
        --------
        >>> from surpyval import ItemHistory, RecurrentEventData
        >>> data = RecurrentEventData.from_items([
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
        ids = [history.item for history in items]
        if len({type(item) for item in ids}) > 1:
            raise ValueError(
                "The item ids must all be of one type (all text or all "
                "numbers): mixed, '1' and 1 could not be told apart."
            )
        repeated = sorted({repr(i) for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(
                "Each item needs one ItemHistory; given twice: "
                + ", ".join(repeated)
                + "."
            )
        static = _all_or_none(items, "covariates", lambda h: h.covariates)
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
        if any(typed):
            untyped = [
                h.item
                for h, t, ev in zip(items, typed, events)
                if ev and not t
            ]
            if untyped:
                raise ValueError(
                    "Event types were given for some items' failures but not "
                    f"for {untyped!r}: every failure needs its type."
                )
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
            i += [history.item] * len(xs)
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
                Z += [history.covariates] * len(xs)
        x_arr = np.array(x, dtype=float)
        if not (np.isin(c, (-1, 2))).any():
            x_arr = x_arr[:, 0]
        truncated = np.isfinite(tl).any() or np.isfinite(tr).any()
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
        if varying:
            data.time_varying_covariates = {
                history.item: history.time_varying_covariates
                for history in items
            }
        with_states = [history.has_states for history in items]
        if any(with_states):
            missing = [
                h.item for h, given in zip(items, with_states) if not given
            ]
            if missing:
                raise ValueError(
                    "States were given for some items but not for "
                    f"{missing!r}: every item needs its 'initial_state'."
                )
            data.states = {
                history.item: (history.initial_state, history.state_changes)
                for history in items
            }
        return data

    def to_items(self) -> list:
        """
        The data as one :class:`~surpyval.utils.item_history.ItemHistory`
        per item, in the order of ``items``:
        ``RecurrentEventData.from_items(data.to_items())`` is the same
        data. Event types and observation windows have no place in an
        item's history, and are refused.
        """
        from surpyval.utils.item_history import ItemHistory

        if self.window_map is not None:
            raise ValueError(
                "This data has observation windows, which an ItemHistory has "
                "no place for: use its arrays."
            )
        x = self.x if np.ndim(self.x) == 2 else np.column_stack([self.x] * 2)
        Z = None if self.Z is None else np.asarray(self.Z, dtype=float)
        histories = []
        for item in self.items:
            rows = self.i == item
            key = item.item() if isinstance(item, np.generic) else item
            c, n, xi = self.c[rows], self.n[rows], x[rows]
            e = None if self.e is None else self.e[rows]
            left = xi[c == -1]
            end = xi[c == 1]
            tl, tr = self.tl[rows][0], self.tr[rows][0]
            varying = None
            if self.time_varying_covariates is not None:
                varying = self.time_varying_covariates[key]
            state_args: dict = {}
            if self.states is not None:
                initial, changes = self.states[key]
                state_args = {
                    "initial_state": initial,
                    "observed_states": {
                        u: st for k, _, u, st in changes if k == "observed"
                    }
                    or None,
                    "interval_censored_states": [
                        [lo, u, st]
                        for k, lo, u, st in changes
                        if k == "interval_censored"
                    ]
                    or None,
                    "left_censored_states": {
                        u: st
                        for k, _, u, st in changes
                        if k == "left_censored"
                    }
                    or None,
                }
            histories.append(
                ItemHistory(
                    key,
                    observed=xi[c == 0, 0],
                    right_censored=float(end[0, 0]) if len(end) else None,
                    interval_censored=xi[c == 2],
                    interval_censored_counts=n[c == 2],
                    left_censored=float(left[0, 0]) if len(left) else None,
                    left_censored_counts=(
                        int(n[c == -1][0]) if len(left) else None
                    ),
                    left_truncation=float(tl) if np.isfinite(tl) else None,
                    right_truncation=float(tr) if np.isfinite(tr) else None,
                    covariates=None if Z is None else Z[rows][0],
                    time_varying_covariates=varying,
                    observed_event_types=(
                        None if e is None else list(e[c == 0])
                    ),
                    interval_censored_event_types=(
                        None
                        if e is None or not (c == 2).any()
                        else list(e[c == 2])
                    ),
                    left_censored_event_types=(
                        None
                        if e is None or not (c == -1).any()
                        else e[c == -1][0]
                    ),
                    **state_args,
                )
            )
        return histories

    def to_histories(self) -> dict:
        """
        The data as each item's history, the form :meth:`from_histories`
        reads: ``"observed"`` (``{item: ages at its failures}``) and
        ``"right_censored"`` (``{item: age at the end of observation}``,
        for the items with one), with ``"left_truncation"`` /
        ``"right_truncation"`` for the items that are truncated and
        ``"covariates"`` where the data has them. So
        ``RecurrentEventData.from_histories(**data.to_histories())`` is the
        same data. Interval or left censored counts, repeated rows,
        event types and observation windows have no place in a history,
        and are refused.

        Examples
        --------
        >>> from surpyval import RecurrentEventData
        >>> data = RecurrentEventData.from_histories(
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
        for item in self.items:
            rows = self.i == item
            key = item.item() if isinstance(item, np.generic) else item
            observed[key] = self.x[rows & (self.c == 0)].astype(float)
            ends = self.x[rows & (self.c == 1)]
            if len(ends):
                right_censored[key] = float(ends[0])
            left[key] = float(self.tl[rows][0])
            right[key] = float(self.tr[rows][0])
            if Z is not None:
                covariates[key] = Z[rows][0]
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

    def __getitem__(self, index: Any) -> "RecurrentEventData":
        return RecurrentEventData(
            self.x[index],
            self.i[index],
            self.c[index],
            self.n[index],
            None if self.e is None else self.e[index],
            tl=self.tl[index],
            tr=self.tr[index],
        )

    def __iter__(self) -> Any:
        # A stateless iterator: the old hand-rolled _index version broke
        # nested iteration over the same object.
        return zip(self.x, self.i, self.c, self.n)

    def __repr__(self) -> str:
        return f"""
            RecurrentEventData(
    x={self.x},
    i={self.i},
    c={self.c},
    n={self.n}\n)
        """.strip()


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
        missing = [h.item for h, g in zip(items, given) if not g]
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
        try:
            row = np.atleast_1d(np.asarray(covariates[item], dtype=float))
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


def accepts_recurrent_data(method: F) -> F:
    """Let a recurrent fitter's ``fit`` take a :class:`RecurrentEventData`
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
        if not isinstance(data, RecurrentEventData):
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
                "x is a RecurrentEventData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than a RecurrentEventData)."
            )
        if data.window_map is not None:
            raise ValueError(
                "This RecurrentEventData was built with observation windows: "
                "pass its arrays and the windows to fit instead."
            )
        if (
            data.Z is None
            and "Z" in names
            and bound.arguments.get("Z") is None
            and signature.parameters["Z"].default is inspect.Parameter.empty
        ):
            raise TypeError(
                "This fit needs covariates, Z, and the RecurrentEventData "
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
                "RecurrentEventData holds."
            )
        bound.arguments.update(filled)
        return method(*bound.args, **bound.kwargs)

    setattr(wrapper, "_accepts_recurrent_data", True)
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
                + ", which the RecurrentEventData holds."
            )
        return method(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
