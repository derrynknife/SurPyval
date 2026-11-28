"""The warranty "Nevada" table: shipments by period and the returns of
each shipment in each later period.

A warranty or field-return database is often kept as a triangle. Each
row is a shipment (or sales, or build) period with the number of units
shipped in it; each column is a later period, holding how many of that
shipment came back in it. The newest shipment has had the least time in
service, so its row is the shortest, and the table looks like a
staircase (on its side, like the outline of the state of Nevada).

:class:`NevadaTable` reads that triangle into lifetime data -- each
return a failure at its age in periods, each shipment's survivors
right-censored at their age on the analysis date -- writes it back from
a list of returns, and forecasts the survivors' future returns with
:func:`surpyval.forecast`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["NevadaTable"]

_AGES = ("interval", "end", "middle")
_LAYOUTS = ("calendar", "age")
_SHIPPING = ("start", "spread")


def _counts(values: Any, name: str, allow_nan: bool = False) -> npt.NDArray:
    """``values`` as a float array of whole, non-negative counts (``nan``
    kept where ``allow_nan``, for a cell that was not observed)."""
    try:
        arr = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"'{name}' must hold numbers of units (whole numbers); a blank "
            "cell is None or nan."
        )
    seen = ~np.isnan(arr) if allow_nan else np.ones(arr.shape, bool)
    vals = arr[seen]
    if not np.isfinite(vals).all():
        raise ValueError(
            f"'{name}' must hold finite numbers of units"
            + ("" if allow_nan else " with no blanks")
            + "."
        )
    if (vals < 0).any() or (vals != np.floor(vals)).any():
        bad = vals[(vals < 0) | (vals != np.floor(vals))][0]
        raise ValueError(
            f"'{name}' must hold whole, non-negative numbers of units; it "
            f"has {float(bad)!r}."
        )
    return arr


class NevadaTable:
    r"""
    A warranty table of shipments by return period (a "Nevada chart").

    Each row is a shipment period with the number of units ``shipped`` in
    it; ``returns[i][j]`` is how many units of shipment ``i`` came back in
    return period ``j``. The table reads into lifetime data: a unit that
    came back in its :math:`k`-th period in service failed at an age of
    :math:`k` periods, and the units of a shipment that have not come back
    are still working at their age on the analysis date (right censored).
    Fit any model to that, then :meth:`forecast` gives the returns still
    to come from the units in service.

    Parameters
    ----------
    shipped : array_like
        The number of units shipped in each shipment period, oldest
        first: one per row of ``returns``.
    returns : array_like, 2-D
        The returns of each shipment (rows) in each return period
        (columns), in one of two layouts:

        - ``"calendar"`` (the default, how the chart is usually drawn):
          the columns are calendar periods, the first column being the
          first period in which the first shipment could come back, and
          each later shipment starting one column further right. Cells
          before a shipment's first period are blank (``None``, ``nan``)
          or 0. Every shipment is observed up to the last column, the
          analysis date.
        - ``"age"``: the columns are periods in service; column ``k - 1``
          holds the returns in each shipment's ``k``-th period, so every
          row starts in the first column. A row ends at its first blank
          cell (``nan`` or ``None``; a list of lists may also be ragged),
          or runs to the last column. ``observed`` gives the lengths
          instead.
    layout : {"calendar", "age"}, optional
        How ``returns`` is laid out (above). Keyword only.
    cumulative : bool, optional
        If True, each cell is the shipment's returns *to date* (by the end
        of that period), as some databases keep them; the table holds the
        differences. Default False. Keyword only.
    observed : array_like of int, optional
        For the ``"age"`` layout: the number of periods each shipment has
        been in service on the analysis date (the length of its row). By
        default each row's length. Keyword only.
    period : float, optional
        The length of one period in the time unit of the fit (30.44 for
        months in days); 1 by default, so ages are in periods. Keyword
        only.
    labels : sequence, optional
        A label for each shipment period (``"Jun 2010"``), shown by
        ``print``. Keyword only.
    shipping : {"start", "spread"}, optional
        When in its period a shipment went into service. ``"start"`` (the
        default, the warranty tools' reading): every unit at the start of
        its shipment period, so a return in its :math:`k`-th period in
        service is between :math:`k - 1` and :math:`k` periods old.
        ``"spread"``: the units went into service evenly through their
        shipment period, so on average half a period later; every age
        (returns, survivors and totals) is read half a period younger
        (see Notes). Keyword only.
    totals : array_like, optional
        For shipments whose returns are known only as a total to date, with
        no breakdown by period: the number returned by the analysis date,
        one value per shipment, ``None`` or ``nan`` for a shipment whose
        returns are in its row. A shipment with a total has no returns in
        its row (blank or 0 cells, as many as its periods in service). Each
        such return is a failure at some age up to the shipment's age on the
        analysis date (left censored). Keyword only.

    Attributes
    ----------
    shipped : numpy.ndarray
        Units shipped in each period.
    returns : numpy.ndarray
        ``returns[i, k - 1]``: returns of shipment ``i`` in its ``k``-th
        period in service (the ``"age"`` layout), ``nan`` past its last
        observed period.
    observed : numpy.ndarray
        The number of periods each shipment has been in service.
    totals : numpy.ndarray
        The total returns to date of each shipment given by ``totals``,
        ``nan`` for the others.
    shipping : str
        ``"start"`` or ``"spread"``.

    Raises
    ------
    ValueError
        If a count is negative, fractional or infinite; ``shipped`` does
        not have one value per row; a shipment has more returns than units
        shipped (the row is named); a calendar-layout cell before its
        shipment's first period holds returns; a cumulative row decreases;
        or ``layout`` or ``period`` is not one the table takes.

    Notes
    -----
    Ages are counted in whole periods from the shipment period, as the
    warranty tools do: a return in a shipment's :math:`k`-th period in
    service is :math:`k` periods old, and its survivors are :math:`m`
    periods old after :math:`m` periods. :meth:`to_data` treats a return
    as failed somewhere in its period, in
    :math:`((k - 1) \Delta, k \Delta]` (interval censored, the default,
    which assumes nothing about when in the period it failed), or at the
    end of it (``ages="end"``, Weibull++'s convention) or at its middle.
    By default every unit of a shipment is taken to start at the start of
    its shipment period; a unit shipped late in its period has been in
    service a little less than its shipment's age, so the ages run a
    little old. With ``shipping="spread"`` the units start evenly through
    the period: a unit shipped a fraction :math:`u` into it and returned in
    its shipment's :math:`k`-th period has an age between
    :math:`(k - 1 - u) \Delta` and :math:`(k - u) \Delta`. Taking each
    unit at the middle of its period (:math:`u = 1/2`) reads the return as
    failed in :math:`((k - 3/2) \Delta, (k - 1/2) \Delta]` (from 0 for
    :math:`k = 1`) and the survivors as :math:`(m - 1/2) \Delta` old. This
    is the half-period offset; it removes the reading's bias in the ages
    (on average half a period), though not the extra spread of each
    return's age, which the interval does not widen to carry.

    Examples
    --------
    Three monthly shipments and their returns in the months after (the
    Nevada chart in ReliaSoft's warranty documentation):

    >>> import surpyval as surv
    >>> table = surv.NevadaTable(
    ...     shipped=[100, 140, 150],
    ...     returns=[[3, 3, 5], [None, 2, 4], [None, None, 4]],
    ...     labels=["Jun", "Jul", "Aug"],
    ... )
    >>> print(table)
    Nevada table: 3 shipments, 390 units, 21 returns (calendar layout)
    shipment  shipped  period 1  period 2  period 3  in service  survivors
         Jun      100         3         3         5           3         89
         Jul      140                   2         4           2        134
         Aug      150                             4           1        146
    >>> table.survivors
    array([ 89, 134, 146])
    >>> model = surv.Weibull.fit(table.to_data(ages="end"))
    >>> model.params.round(4)  # published: eta 6.6951, beta 2.4928
    array([6.695 , 2.4928])
    >>> ahead = table.forecast(model, horizon=1)
    >>> ahead.unit_expected.ravel().round(2)  # published: 11.75, 11.04, 5.80
    array([11.76, 11.04,  5.79])
    >>> ahead.expected.round(2)
    array([28.59])
    """

    def __init__(
        self,
        shipped: npt.ArrayLike,
        returns: Any,
        *,
        layout: str = "calendar",
        cumulative: bool = False,
        observed: npt.ArrayLike | None = None,
        period: float = 1.0,
        labels: Any = None,
        shipping: str = "start",
        totals: npt.ArrayLike | None = None,
    ) -> None:
        if shipping not in _SHIPPING:
            raise ValueError(
                f"'shipping' must be one of {_SHIPPING}, not {shipping!r}: "
                "'start' when every unit of a shipment went into service at "
                "the start of its period, 'spread' when they went in evenly "
                "through it."
            )
        if layout not in _LAYOUTS:
            raise ValueError(
                f"'layout' must be one of {_LAYOUTS}, not {layout!r}: "
                "'calendar' when the columns are calendar periods (the "
                "staircase), 'age' when they are periods in service."
            )
        period = float(period)
        if not np.isfinite(period) or period <= 0:
            raise ValueError(
                "'period' must be a positive, finite length of one period "
                f"in the fit's time unit; got {period!r}."
            )
        shipped_arr = _counts(shipped, "shipped").ravel()
        rows = _ragged(returns)
        if len(shipped_arr) != rows.shape[0]:
            raise ValueError(
                f"'shipped' has {len(shipped_arr)} values and 'returns' "
                f"{rows.shape[0]} rows: give one shipment quantity per row."
            )
        rows = _counts(rows, "returns", allow_nan=True)
        if layout == "calendar":
            if observed is not None:
                raise ValueError(
                    "'observed' is for the 'age' layout; in the 'calendar' "
                    "layout every shipment is observed to the last column."
                )
            ages, m = _calendar_to_age(rows)
        else:
            ages, m = _age_rows(rows, observed)
        if cumulative:
            ages = _uncumulate(ages, m)
        total_arr = _totals(totals, len(shipped_arr), ages, m)
        returned = np.nansum(ages, axis=1) + np.nan_to_num(total_arr)
        over = np.flatnonzero(returned > shipped_arr)
        if over.size:
            i = int(over[0])
            raise ValueError(
                f"Shipment {i} (row {i} of 'returns') has {returned[i]:g} "
                f"returns but {shipped_arr[i]:g} units shipped; check the "
                "row (are its cells cumulative? pass cumulative=True)."
            )
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(shipped_arr):
                raise ValueError(
                    f"'labels' has {len(labels)} values; give one per "
                    f"shipment ({len(shipped_arr)})."
                )
        self.shipped = shipped_arr.astype(int)
        self.returns = ages
        self.observed = m.astype(int)
        self.period = period
        self.labels = labels
        self.layout = layout
        self.shipping = shipping
        self.totals = total_arr

    # ------------------------------------------------------------------
    # Reading the table

    @property
    def survivors(self) -> npt.NDArray:
        """The units of each shipment not returned by the analysis date."""
        returned = np.nansum(self.returns, axis=1) + np.nan_to_num(self.totals)
        return self.shipped - returned.astype(int)

    @property
    def age(self) -> npt.NDArray:
        """Each shipment's age on the analysis date, in the fit's time
        unit (``observed * period``, or half a period less with
        ``shipping="spread"``): the age at which its survivors are
        censored, and from which :meth:`forecast` looks ahead."""
        return (self.observed - self._offset) * self.period

    @property
    def _offset(self) -> float:
        """How much younger, in periods, the spread reading takes each
        age."""
        return 0.5 if self.shipping == "spread" else 0.0

    def to_xcnt(
        self, ages: str = "interval"
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        r"""
        The table as lifetime data, ``x``, ``c`` and ``n``, pooled over
        the shipments.

        Each return is a failure in its period of service, each
        shipment's survivors are right censored (``c = 1``) at its age, and
        the returns given only as a total (``totals``) are left censored
        (``c = -1``) at it. With ``shipping="spread"`` every age is half a
        period younger (see :class:`NevadaTable`).

        Parameters
        ----------
        ages : {"interval", "end", "middle"}, optional
            When in its period a return failed: anywhere in it, an
            interval-censored row :math:`((k - 1) \Delta, k \Delta]`
            (``c = 2``, the default); at its end, :math:`k \Delta`, an
            exact failure (Weibull++'s convention); or at its middle,
            :math:`(k - 1/2) \Delta`.

        Returns
        -------
        x : numpy.ndarray
            Shape ``(N,)`` for exact ages, ``(N, 2)`` (``[lower, upper]``,
            the right- and left-censored rows ``[age, age]``) for
            ``"interval"``.
        c, n : numpy.ndarray
            The censoring flags and counts.
        """
        if ages not in _AGES:
            raise ValueError(
                f"'ages' must be one of {_AGES}, not {ages!r}: 'interval' "
                "when a return is known only by its period, 'end' or "
                "'middle' to take it as failed at that point of it."
            )
        S, K = self.returns.shape
        width = self.period
        k = (np.arange(1, K + 1) - self._offset) * width
        fail = np.zeros(K)
        for row in self.returns:
            fail += np.nan_to_num(row)
        keep = fail > 0
        k, fail = k[keep], fail[keep]
        lower = np.maximum(k - width, 0.0)
        surv_age = self.age.astype(float)
        # Pool the survivors (and the totals) of shipments of the same age
        u_age, u_n = _pool(surv_age, self.survivors)
        t_age, t_n = _pool(surv_age, np.nan_to_num(self.totals))
        if ages == "interval":
            x = np.vstack(
                [
                    np.column_stack([lower, k]),
                    np.column_stack([u_age, u_age]),
                    np.column_stack([t_age, t_age]),
                ]
            )
            c = np.concatenate(
                [
                    np.full(len(k), 2),
                    np.ones(len(u_age)),
                    np.full(len(t_age), -1),
                ]
            )
        else:
            if ages == "end":
                at = k
            elif self._offset:
                at = (lower + k) / 2
            else:
                at = k - width / 2
            x = np.concatenate([at, u_age, t_age])
            c = np.concatenate(
                [
                    np.zeros(len(k)),
                    np.ones(len(u_age)),
                    np.full(len(t_age), -1),
                ]
            )
        n = np.concatenate([fail, u_n, t_n])
        return x, c.astype(int), n.astype(int)

    def to_data(self, ages: str = "interval") -> Any:
        """
        The table as a :class:`~surpyval.utils.surpyval_data.SurpyvalData`
        (see :meth:`to_xcnt` for ``ages``), which every ``fit`` takes:
        ``surv.Weibull.fit(table.to_data())``.
        """
        from surpyval.utils.surpyval_data import SurpyvalData

        x, c, n = self.to_xcnt(ages)
        return SurpyvalData(x, c, n)

    def to_array(self, layout: str = "calendar") -> npt.NDArray:
        """
        The triangle of returns as a float array, ``nan`` in the cells a
        shipment was not in service: the ``"calendar"`` staircase (the
        columns calendar periods) or the ``"age"`` layout (the columns
        periods in service). The table's own layout is :attr:`layout`.
        """
        if layout not in _LAYOUTS:
            raise ValueError(
                f"'layout' must be one of {_LAYOUTS}, not {layout!r}."
            )
        if layout == "age":
            return self.returns.copy()
        S = len(self.shipped)
        # The analysis date is shipment i's period i + observed[i] - 1
        P = int(np.max(np.arange(S) + self.observed)) if S else 0
        out = np.full((S, P), np.nan)
        for i in range(S):
            m = self.observed[i]
            out[i, i : i + m] = self.returns[i, :m]
        return out

    # ------------------------------------------------------------------
    # Writing the table

    @classmethod
    def from_returns(
        cls,
        shipped: npt.ArrayLike,
        shipment: npt.ArrayLike,
        age: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        *,
        observed: npt.ArrayLike | None = None,
        period: float = 1.0,
        labels: Any = None,
        shipping: str = "start",
    ) -> NevadaTable:
        """
        The table from a list of returns: each return's shipment period
        and the period of service it came back in.

        Parameters
        ----------
        shipped : array_like
            Units shipped in each shipment period, oldest first.
        shipment : array_like of int
            Each return's shipment period: its row, from 0 (the oldest).
        age : array_like of int
            The period of service it came back in, from 1: the return
            period minus the shipment period, plus 1 if a return can come
            in the shipment period itself (so that period is the first in
            service).
        n : array_like of int, optional
            The number of returns each row stands for (1 each by default).
        observed : array_like of int, optional
            The periods each shipment has been in service on the analysis
            date. By default the staircase: the oldest shipment has been
            in service for its latest return's period (or for as many
            periods as there are shipments, if longer), and each later one
            for one period less.
        period, labels, shipping
            As for :class:`NevadaTable`.

        Returns
        -------
        NevadaTable
            In the ``"calendar"`` layout when ``observed`` is the
            staircase, otherwise the ``"age"`` one.

        Examples
        --------
        >>> import surpyval as surv
        >>> table = surv.NevadaTable.from_returns(
        ...     shipped=[100, 140, 150],
        ...     shipment=[0, 0, 0, 1, 1, 2],
        ...     age=[1, 2, 3, 1, 2, 1],
        ...     n=[3, 3, 5, 2, 4, 4],
        ... )
        >>> table.to_array()
        array([[ 3.,  3.,  5.],
               [nan,  2.,  4.],
               [nan, nan,  4.]])
        """
        shipped_arr = _counts(shipped, "shipped").ravel()
        S = len(shipped_arr)
        row = _counts(shipment, "shipment").ravel().astype(int)
        k = _counts(age, "age").ravel().astype(int)
        if n is None:
            count = np.ones(len(row), int)
        else:
            count = _counts(n, "n").ravel().astype(int)
            if count.size == 1 and row.size != 1:
                count = np.full(len(row), int(count[0]))
        if not (len(row) == len(k) == len(count)):
            raise ValueError(
                "'shipment', 'age' and 'n' must have one value per return "
                f"(got {len(row)}, {len(k)} and {len(count)})."
            )
        if row.size and row.max() >= S:
            raise ValueError(
                f"'shipment' names row {int(row.max())}, but 'shipped' has "
                f"{S} shipments (rows 0 to {S - 1})."
            )
        if (k < 1).any():
            raise ValueError(
                "'age' must be at least 1, the first period in service; "
                "a return in the shipment period itself is age 1 when such "
                "returns are possible."
            )
        if observed is None:
            first = max(int(k[row == 0].max()) if (row == 0).any() else 0, S)
            m = first - np.arange(S)
            # A later shipment's returns may need it to be older
            for i in range(S):
                if (row == i).any():
                    m[i] = max(m[i], int(k[row == i].max()))
            staircase = bool(np.all(np.diff(m) == -1)) if S > 1 else True
        else:
            m = _counts(observed, "observed").ravel().astype(int)
            if len(m) != S:
                raise ValueError(
                    f"'observed' has {len(m)} values; give one per "
                    f"shipment ({S})."
                )
            staircase = bool(np.all(np.diff(m) == -1)) if S > 1 else True
            late = np.flatnonzero(k > m[row])
            if late.size:
                j = int(late[0])
                raise ValueError(
                    f"Return {j} is in period {int(k[j])} of service, but "
                    f"shipment {int(row[j])} has been in service for "
                    f"{int(m[row[j]])} periods ('observed')."
                )
        K = int(m.max()) if S else 0
        grid = np.full((S, K), np.nan)
        for i in range(S):
            grid[i, : m[i]] = 0.0
        np.add.at(grid, (row, k - 1), count)
        table = cls(
            shipped_arr,
            grid,
            layout="age",
            observed=m,
            period=period,
            labels=labels,
            shipping=shipping,
        )
        if staircase:
            table.layout = "calendar"
        return table

    def to_returns(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        The returns as a list, the inverse of :meth:`from_returns`: each
        non-empty cell's shipment period (row, from 0), period of service
        (from 1) and number of returns.
        """
        filled = np.nan_to_num(self.returns) > 0
        row, col = np.nonzero(filled)
        return row, col + 1, self.returns[row, col].astype(int)

    # ------------------------------------------------------------------
    # Forecasting

    def forecast(
        self,
        model: Any,
        horizon: npt.ArrayLike,
        limit: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        The returns still to come from each shipment's survivors:
        :func:`surpyval.forecast` with the shipments' ages
        (:attr:`age`) and survivors (:attr:`survivors`). The returns given
        only as totals have already happened, so they are not forecast.

        Parameters
        ----------
        model : fitted model
            A model fitted to the table (``Weibull.fit(table.to_data())``)
            or any other with ``sf``.
        horizon : scalar or array_like
            The time ahead, in the fit's time unit (periods when
            ``period`` is 1): one time, or the end of each period ahead.
        limit : scalar or array_like, optional
            The age at which the warranty ends: returns after it are not
            counted. One for every shipment or one each.
        alpha_ci : float, optional
            The significance level of the prediction intervals.

        Returns
        -------
        Forecast
            See :func:`surpyval.forecast`; its rows are the shipments.
        """
        from surpyval.forecasting import forecast

        return forecast(
            model,
            age=self.age.astype(float),
            n=self.survivors,
            horizon=horizon,
            limit=limit,
            alpha_ci=alpha_ci,
        )

    def __repr__(self) -> str:
        S = len(self.shipped)
        grid = self.to_array(self.layout)
        n_returns = int(np.nansum(self.returns) + np.nansum(self.totals))
        head = (
            f"Nevada table: {S} shipments, {int(self.shipped.sum())} units, "
            f"{n_returns} returns ({self.layout} layout"
            + (", shipments spread" if self.shipping == "spread" else "")
            + ")"
        )
        word = "period" if self.layout == "calendar" else "age"
        names = ["shipment", "shipped"]
        names += [f"{word} {j + 1}" for j in range(grid.shape[1])]
        names += ["in service", "survivors"]
        labels = (
            [str(v) for v in self.labels]
            if self.labels is not None
            else [str(i) for i in range(S)]
        )
        cells = []
        for i in range(S):
            vals = ["" if np.isnan(v) else f"{v:g}" for v in grid[i]]
            if not np.isnan(self.totals[i]):
                # A total with no breakdown: say so in its first cell
                first = i if self.layout == "calendar" else 0
                vals = [""] * len(vals)
                vals[first] = f"total {self.totals[i]:g}"
            cells.append(
                [labels[i], f"{self.shipped[i]:d}"]
                + vals
                + [f"{self.observed[i]:d}", f"{self.survivors[i]:d}"]
            )
        cols = list(zip(names, *cells))
        widths = [max(len(c) for c in col) for col in cols]
        lines = [head]
        for r in range(S + 1):
            lines.append(
                "  ".join(col[r].rjust(w) for col, w in zip(cols, widths))
            )
        return "\n".join(lines)


def _pool(ages: npt.NDArray, n: npt.NDArray) -> tuple[Any, Any]:
    """The distinct ages with a count, and the counts at each."""
    u_age, inv = np.unique(ages, return_inverse=True)
    u_n = np.bincount(inv, weights=n, minlength=len(u_age))
    has = u_n > 0
    return u_age[has], u_n[has]


def _totals(
    totals: Any, S: int, ages: npt.NDArray, m: npt.NDArray
) -> npt.NDArray:
    """``totals`` as a float array, one per shipment (``nan`` where the
    returns are in the row), checked against the rows."""
    if totals is None:
        return np.full(S, np.nan)
    try:
        arr = np.array(
            [np.nan if v is None else v for v in np.ravel(totals)],
            dtype=float,
        )
    except (TypeError, ValueError):
        raise ValueError(
            "'totals' must hold numbers of units, None or nan for a "
            "shipment whose returns are in its row."
        )
    if arr.size != S:
        raise ValueError(
            f"'totals' has {arr.size} values; give one per shipment ({S}), "
            "None for a shipment whose returns are in its row."
        )
    arr = _counts(arr, "totals", allow_nan=True)
    for i in np.flatnonzero(~np.isnan(arr)):
        if np.nansum(ages[i]) > 0:
            raise ValueError(
                f"Shipment {i} has a total in 'totals' and returns in its "
                "row: give its returns one way, by period in the row or as "
                "a total (its row blank or 0)."
            )
        if m[i] < 1:
            raise ValueError(
                f"Shipment {i} has a total in 'totals' but no periods in "
                "service: give its row as 0 cells for each period it has "
                "been in service, or pass 'observed'."
            )
    return arr


def _ragged(returns: Any) -> npt.NDArray:
    """A list of rows (perhaps ragged, perhaps with ``None``) as a 2-D
    float array padded with ``nan``."""
    if (
        isinstance(returns, np.ndarray)
        and returns.ndim == 2
        and returns.dtype != object
    ):
        return returns.astype(float)
    try:
        rows = [list(r) for r in returns]
    except TypeError:
        raise ValueError(
            "'returns' must be 2-D: one row of returns per shipment."
        )
    K = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), K), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if v is not None:
                try:
                    out[i, j] = float(v)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"'returns' row {i}, column {j} is {v!r}: give a "
                        "number of units, or None for a blank cell."
                    )
    return out


def _calendar_to_age(rows: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """The staircase as the age layout: shipment ``i``'s ``k``-th period
    in service is column ``i + k - 1``."""
    S, P = rows.shape
    if S > P:
        raise ValueError(
            f"'returns' has {S} shipments but {P} return periods: in the "
            "'calendar' layout the last shipment's first period in service "
            f"is column {S - 1}. Add the empty columns, or pass "
            "layout='age' if the columns are periods in service."
        )
    m = P - np.arange(S)
    ages = np.full((S, P), np.nan)
    for i in range(S):
        before = np.nan_to_num(rows[i, :i])
        if (before > 0).any():
            j = int(np.flatnonzero(before > 0)[0])
            raise ValueError(
                f"'returns' row {i}, column {j} holds {before[j]:g} returns "
                f"before shipment {i}'s first period in service (column "
                f"{i}). In the 'calendar' layout each shipment starts one "
                "column right of the one before; pass layout='age' if every "
                "row starts in the first column."
            )
        ages[i, : m[i]] = np.nan_to_num(rows[i, i:])
    return ages, m


def _age_rows(
    rows: npt.NDArray, observed: npt.ArrayLike | None
) -> tuple[npt.NDArray, npt.NDArray]:
    S, K = rows.shape
    if observed is None:
        m = np.empty(S, int)
        for i in range(S):
            blank = np.flatnonzero(np.isnan(rows[i]))
            m[i] = blank[0] if blank.size else K
            rest = rows[i, m[i] :]
            if (np.nan_to_num(rest) > 0).any():
                raise ValueError(
                    f"'returns' row {i} has returns after a blank cell "
                    f"(column {m[i]}): a row ends at its first blank. Fill "
                    "the gap with 0, or give 'observed'."
                )
    else:
        m = _counts(observed, "observed").ravel().astype(int)
        if len(m) != S:
            raise ValueError(
                f"'observed' has {len(m)} values; give one per shipment "
                f"({S})."
            )
        if (m > K).any():
            K = int(m.max())
            rows = np.hstack([rows, np.full((S, K - rows.shape[1]), np.nan)])
        for i in range(S):
            if (np.nan_to_num(rows[i, m[i] :]) > 0).any():
                raise ValueError(
                    f"'returns' row {i} has returns past its "
                    f"{int(m[i])} observed periods ('observed')."
                )
    ages = np.full((S, K), np.nan)
    for i in range(S):
        ages[i, : m[i]] = np.nan_to_num(rows[i, : m[i]])
    return ages, m


def _uncumulate(ages: npt.NDArray, m: npt.NDArray) -> npt.NDArray:
    out = ages.copy()
    for i in range(len(m)):
        row = ages[i, : m[i]]
        step = np.diff(row, prepend=0.0)
        if (step < 0).any():
            j = int(np.flatnonzero(step < 0)[0])
            raise ValueError(
                f"'returns' row {i} is cumulative (cumulative=True) but "
                f"falls at its period {j + 1} of service: cumulative "
                "returns cannot decrease. Are the cells per period? Pass "
                "cumulative=False."
            )
        out[i, : m[i]] = step
    return out
