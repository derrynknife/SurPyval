import numpy as np
import numpy.typing as npt

from surpyval.utils.surpyval_data import SurpyvalData


class RecurrentEventData:
    # Optional covariate matrix, attached by ``handle_xicn`` for regression.
    Z: npt.NDArray | None = None

    """
    A class to handle and manipulate recurrent event data. Recurrent events are
    those that can occur more than once for each subject or item.

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
    x=[1.],
    c=[0],
    n=[2],
    t=[[-inf  inf]])
    >>> data.get_interarrival_times()
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """

    def __init__(self, x, i, c, n, e=None, tl=None, tr=None):
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
        self.items = list(set(self.i))

        if self.x.ndim == 1:
            self.interarrival_times = self.get_interarrival_times()
        else:
            x_midpoints = self.x.copy()
            x_midpoints[self.c == -1, 0] = x_midpoints.min()
            self.midpoints = x_midpoints.mean(axis=1)

        self._index = 0

    def to_xrd(self, estimator="Nelson-Aalen"):
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

            x_unique = np.unique(x_out)

            # TODO: consider having the presence of left-censored
            # data use the midpoints instead of the end value of the left
            # censored interval.

            d = np.array(
                [
                    self.n[
                        (x_out == xi)
                        & ((self.c == 0) | (self.c == 2) | (self.c == -1))
                    ].sum()
                    for xi in x_unique
                ]
            )
            # count the number of items at their maximum x for each x_unique
            # find the maximum x for each item
            max_x = np.array(
                [self.x[self.i == item].max() for item in self.items]
            )
            # sum the number of items at each x in x_unique
            # that are at their max
            r = np.array([(max_x == xi).sum() for xi in x_unique])

            r = len(self.items) * np.ones_like(x_unique) - r.cumsum() + r

            self.xrd = x_unique, r, d
        return self.xrd

    @property
    def event_types(self):
        """
        The distinct event types (marks) present in the data, excluding the
        ``None`` mark used for censored / end-of-observation rows. Returns an
        empty list when the data carries no marks.
        """
        if self.e is None:
            return []
        return sorted({e for e in self.e if e is not None})

    def to_cause_specific_xrd(self, cause):
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
        if self.e is None:
            raise ValueError(
                "Data has no event-type marks; pass `e` to compute "
                "cause-specific curves."
            )
        if cause not in self.event_types:
            raise ValueError(
                "Unrecognised cause {!r}; known causes are {}".format(
                    cause, self.event_types
                )
            )

        # Reuse the shared at-risk set and event-time grid from to_xrd.
        x_unique, r, _ = self.to_xrd()

        if self.x.ndim == 2:
            x_out = self.midpoints
        else:
            x_out = self.x

        observed = (self.c == 0) | (self.c == 2) | (self.c == -1)
        is_cause = np.array([ei == cause for ei in self.e])
        d_cause = np.array(
            [
                self.n[(x_out == xi) & observed & is_cause].sum()
                for xi in x_unique
            ]
        )
        return x_unique, r, d_cause

    def get_interarrival_times(self):
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

    def get_previous_x(self, min_x=0):
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

    def get_events_for_item(self, item):
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

    def get_times_to_first_events(self):
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

    def __getitem__(self, index):
        return RecurrentEventData(
            self.x[index],
            self.i[index],
            self.c[index],
            self.n[index],
            None if self.e is None else self.e[index],
            tl=self.tl[index],
            tr=self.tr[index],
        )

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.x):
            result = (
                self.x[self._index],
                self.i[self._index],
                self.c[self._index],
                self.n[self._index],
            )
            self._index += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return f"""
            RecurrentEventData(
    x={self.x},
    i={self.i},
    c={self.c},
    n={self.n}\n)
        """.strip()
