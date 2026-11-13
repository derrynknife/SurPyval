from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from numbers import Number
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

import surpyval
from surpyval.serialisation import (
    check_schema,
    decode_non_finite,
    encode_non_finite,
)
from surpyval.utils import xcnt_handler
from surpyval.utils.data_reshape import LIFETIME_PARTS, ReshapeMixin
from surpyval.utils.item_history import missing_type
from surpyval.utils.measurements import MeasurementsMixin
from surpyval.utils.numeric import unique_pairs
from surpyval.utils.warnings import (
    MaskedCauseWarning,
    ignoring_warnings,
    named,
    warn_masked,
    warn_suspension_types,
)


class SurpyvalData(MeasurementsMixin, ReshapeMixin):
    """
    Lifetime data: one row per item (or per group of identical items,
    counted ``n``), each observed exactly, right censored, left censored
    or interval censored, within its truncation window, with the item's
    id, event type and covariates where they are given. The parametric,
    non-parametric and regression ``fit`` methods take one in place of the
    arrays: ``Weibull.fit(data)``, ``CoxPH.fit(data)``,
    ``CompetingRisks.fit(data)``.

    Build one from each item's record,
    :meth:`from_items` (:class:`~surpyval.utils.item_lifetime.ItemLifetime`),
    from the times of each kind, :meth:`from_observations`, from a
    DataFrame, :meth:`from_dataframe`, from the ``xrd`` layout,
    :meth:`from_xrd`, or from the ``xcnt`` arrays (:doc:`/Conventions`)
    with the constructor, which checks them as every fitter does. The
    data is read back as records (:meth:`item`, :meth:`to_items`,
    :meth:`records`), tables (:meth:`to_dataframe`,
    :meth:`to_observations`, :meth:`to_xcnt`) or JSON (:meth:`to_json`),
    described by :meth:`summary`, and reshaped for another analysis
    (:meth:`without`, :meth:`for_event_type`, :meth:`select`,
    :meth:`censor_at`, ...). ``len(data)`` is its rows; ``n_items`` its
    items.

    Its arrays are ``x`` (a ``[lower, upper]`` row where any observation
    is an interval), ``c``, ``n`` and ``t`` (``tl`` and ``tr``, its
    columns), ``Z`` (the covariates, named by ``covariate_names``), ``e``
    (the event types), ``ids`` (each row's unit, also called ``i``),
    for multi-state data ``s`` (the state each row's unit was in), and
    ``a`` (1 where a row's event ended its unit's history for good). For
    the likelihoods it also holds the
    rows split by kind of observation (``x_o``, ``n_o``; ``x_r``, ``n_r``;
    ``x_l``, ``n_l``; ``x_il``, ``x_ir``, ``n_i``; and ``Z_o``, ... with
    covariates): a censored row with a finite truncation bound on its
    censored side is stored as an interval ending at that bound, so the
    likelihood uses the probability of the part of the window it could
    have fallen in. These are left out of ``dir(data)``.

    Examples
    --------
    >>> from surpyval import ItemLifetime, SurpyvalData, Weibull
    >>> data = SurpyvalData.from_items([
    ...     ItemLifetime("B7", observed=1820),
    ...     ItemLifetime("B8", right_censored=2400),
    ...     ItemLifetime("B9", interval_censored=[1000, 1500]),
    ... ], counts=[1, 40, 2])
    >>> data
    SurpyvalData: 43 items, 3 rows
        id                 x  c   n
      'B7'  [1820.0, 1820.0]  0   1
      'B8'  [2400.0, 2400.0]  1  40
      'B9'  [1000.0, 1500.0]  2   2
    >>> print(data.summary())
    SurpyvalData: 43 items, 3 rows (with item ids)
      outcomes: 1 observed, 40 right censored, 2 interval censored
    >>> data.item("B8")
    ItemLifetime('B8', right_censored=2400.0)
    >>> Weibull.fit(data).parameter_names
    ['alpha', 'beta']

    From the arrays, with a right truncation bound on two rows:

    >>> inf = float("inf")
    >>> data = SurpyvalData(x=[1, 2, 3, 4], c=[0, 1, 0, 1], tr=[9, 3, 9, inf])
    >>> data.x_o, data.x_r
    (array([1., 3.]), array([4.]))
    >>> data.x_il, data.x_ir
    (array([2.]), array([3.]))

    With states: each row is a stay of a unit (``i``) in a state (``s``),
    entered at ``tl`` and left at ``x`` for the state ``e`` (blank where
    the unit was still in ``s`` when last seen, which makes ``c`` 1):

    >>> data = SurpyvalData(
    ...     x=[3, 5, 7, 4],
    ...     tl=[0, 3, 0, 0],
    ...     s=["well", "ill", "well", "well"],
    ...     e=["ill", "dead", None, "dead"],
    ...     i=["A", "A", "B", "C"],
    ... )
    >>> data.c
    array([0, 0, 1, 0])
    >>> data.n_items
    3

    Several rows per unit without states: repairs at 2 and 5, then
    scrapped at 9 (``a`` is read as 1 on a unit's last event, 0 before):

    >>> data = SurpyvalData(x=[2, 5, 9], i=["P1", "P1", "P1"])
    >>> data.a, data.n_items
    (array([0, 0, 1]), 1)
    """

    # Attached by ``from_items``: each row's item id, the event type of
    # each failure (None for a suspension), each item's time-varying
    # covariates (``{item: StepSchedule or CovariatePath}``) and each
    # item's marker readings (``{item: {age: value}}``, ``{}`` for an item
    # with none; #891).
    ids: list | None = None
    e: np.ndarray | None = None
    # Each row's state (the state the unit was in during the row), for
    # multi-state data; None where every row starts from the one initial
    # state (ordinary survival and competing-risks data).
    s: np.ndarray | None = None
    # Each row's terminal flag: 1 where the row's event ended the unit's
    # history for good (scrapped, written off, died), 0 elsewhere; None
    # where it was not given and no unit has more than one row. Where it
    # was filled by the rule (an event row with no ``a``: 1 on a unit's
    # last row, 0 before), ``a_inferred`` is True on that row.
    a: np.ndarray | None = None
    a_inferred: np.ndarray | None = None
    time_varying_covariates: dict | None = None
    # The markers are the measurements m (MeasurementsMixin), one store
    # The name of each column of Z, where they have names: given as
    # ``covariate_names``, the columns of a DataFrame ``Z``, or the items'
    # ``covariates={name: value}``. They name a fitted model's
    # coefficients.
    covariate_names: list | None = None
    # Each row's attributes (``{name: value}``, what describes the item
    # without being fitted), from the records or ``from_dataframe``; None
    # where the data has none, and None for a row without any.
    attributes: list | None = None
    # The unit of the times (``"hour"``), a label as a model's ``unit``
    # is; None where unknown (#945).
    unit: str | None = None
    # Each row's sampling, propensity or censoring weight (any positive
    # number), beside its count ``n``; None where the data is unweighted.
    w: np.ndarray | None = None

    def __init__(
        self,
        x: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        xl: ArrayLike | None = None,
        xr: ArrayLike | None = None,
        tl: ArrayLike | Number | None = None,
        tr: ArrayLike | Number | None = None,
        Z: ArrayLike | None = None,
        group_and_sort: bool = True,
        handle: bool = True,
        e: ArrayLike | None = None,
        covariate_names: list | None = None,
        ids: Any = None,
        unit: Any = None,
        w: ArrayLike | Number | None = None,
        s: Any = None,
        i: Any = None,
        a: Any = None,
        m: Any = None,
    ) -> None:
        """Initialize a SurpyvalData instance for survival analysis.

        Validates, sorts, and stores survival data in the xcnt format. Supports
        uncensored, right/left/interval-censored, and truncated observations.
        Can convert to xrd format and select subsets by censoring type.

        Parameters
        ----------
        x : array-like, optional
            The primary data array of failure/event times. When c is 2 the
            corresponding x entry is a 2-element array [left, right].
        c : array-like, optional
            Censoring flags for each value in x:
            * 0 = uncensored (the event, a failure or a death, was at x)
            * 1 = right censored
            * -1 = left censored
            * 2 = interval censored
        n : array-like, optional
            Number of occurrences for each value in x (positive whole
            numbers).
        t : array-like, optional
            2D array of truncation bounds [left, right] for each value in x.
        xl : array-like, optional
            Left interval bounds for interval censored data, paired with
            'xr'. Given with 'x', each row is read on its own: an exact
            time in 'x' (with 'xl' and 'xr' blank: None, NaN or pandas'
            NA), or a range in 'xl' and 'xr' (with 'x' blank), read by its
            bounds as a ``[xl, xr]`` row is (``[v, inf]`` right censored,
            ``[-inf, v]`` left censored, ``[v1, v2]`` interval censored,
            ``[v, v]`` exact). A row giving both is left out with a
            warning, and so are its entries in every other per-row
            argument; a row giving neither is refused. A ``c`` given must
            agree with each range (a blank ``c`` on a range row is read
            from the range).
        xr : array-like, optional
            Right interval bounds for interval censored data, paired with
            'xl'; with 'x', as for 'xl'.
        tl : array-like or scalar, optional
            Left truncation bounds. Cannot be used with 't'. May be given
            alone (the right bound is then infinite) or with 'tr'.
        tr : array-like or scalar, optional
            Right truncation bounds. Cannot be used with 't'. May be given
            alone (the left bound is then minus infinity) or with 'tl'.
        Z : array-like, optional
            Covariates, one row per value of x (see also
            :meth:`add_covariates`). Giving ``Z`` turns ``group_and_sort``
            off, so the rows stay aligned with x. A DataFrame's columns
            name the covariates.
        covariate_names : list of str, optional
            The name of each column of ``Z``; ``fit(data)`` names the
            model's coefficients by them, and ``with_covariates`` picks
            columns by them.
        e : array-like, optional
            The event type (failure mode or cause) of each row, for the
            competing-risks fitters, which take it from the data; a
            suspension (``c == 1``) has none (None). Giving ``e`` turns
            ``group_and_sort`` off, as ``Z`` does. With states ``s``,
            ``e`` is the state the unit moved into at ``x`` (a failure
            mode is a state entered from the one starting state), blank
            (None, NaN or "") where it was still in ``s``.
        ids : array-like, optional
            Each row's item (unit) id (text or numbers, each once), as
            ``from_items`` gives them: ``data.item(id)`` and
            ``select(ids=...)`` then find the rows. Giving ``ids`` turns
            ``group_and_sort`` off, as ``Z`` does. ``i`` is the same
            argument; give one or the other. With states ``s`` a unit
            has a row for each stay in a state, so an id may repeat.
        unit : str, optional
            The unit of the times (``"hours"``, ``"km"``), a label kept
            with the data as a model keeps its ``unit``; ``fit(data)``
            gives it to the model.
        w : array-like or scalar, optional
            Sampling, propensity or censoring weights: how many units of
            the population each unit of a row stands for (any positive
            finite number), beside its count ``n`` of identical units. A
            fit of weighted data gives the weighted estimate with a
            robust (sandwich) variance, and refuses the bounds weights
            make invalid (see :doc:`/Weighted Data`). Rows are grouped
            only where their weights are equal too. Without ``w`` the
            data is unweighted (``data.w`` is None).
        s : array-like, optional
            The state each row's unit was in during the row (from its
            entry, ``tl``, to ``x``), for multi-state data: one state per
            row, text or numbers, kept as given. Without ``s`` every row
            starts from the one initial state, as ordinary survival and
            competing-risks data do. With ``s``, ``c`` is read from ``e``
            where ``c`` is left out (blank ``e``: 1, still in ``s``;
            otherwise 0, moved into ``e`` at ``x``), and a ``c`` given
            must agree with ``e`` (a row with ``c == 1`` has a blank
            ``e``, any other row a state in ``e``). Giving ``s`` turns
            ``group_and_sort`` off, as ``Z`` does. Fits that are not
            multi-state refuse data with more than one state in ``s``.
        i : array-like, optional
            Each row's unit (item) id; the same argument as ``ids``
            (``data.i`` and ``data.ids`` are the same list). A unit may
            have several rows, in time order: stretches of its own clock,
            ``(tl, x]``, each with its covariates fixed (still running at
            ``x``, ``c == 1``, then the next row from ``x``), or repeated
            events (``a == 0``), ending with a terminal event (``a == 1``)
            or still running (``c == 1``). A row that starts before the
            previous row of its unit ends is refused.
        a : array-like, optional
            Each row's terminal flag: 1 where the row's event ended the
            unit's history for good (scrapped, written off, died: an
            absorbing state, as in multi-state data), 0 where the unit
            carried on (a repair, a recurrent event) or the row has no
            event (``c == 1``, where 1 is refused). A blank (None, NaN)
            on an event row is read as 1 on the unit's last row and 0 on
            an earlier one; ``data.a_inferred`` marks those rows. A row
            after one with ``a == 1`` of the same unit is refused.
            Without ``a``, ``data.a`` is None unless a unit has several
            rows (then it is inferred), and with states ``s`` it is
            kept and checked only (the transitions say which states are
            absorbing).
        m : Measurements or dict, optional
            Readings taken on the units over their lives, on their own
            timeline (:class:`~surpyval.utils.measurements.Measurements`,
            or a dict it is built from: ``{i: {x: value}}`` or ``{name:
            {i: {x: value}}}``), linked to the rows by ``i``, which it
            needs; ``data.m`` (``data.markers`` is the same store). A
            unit with readings but no rows is warned of and kept.
        group_and_sort : bool, default=True
            Whether to group and sort the data. Set False when using covariates
            to maintain data order.
        handle : bool, default=True
            Whether to validate and process the input data.
            Set False for pre-validated data: ``x``, ``c``, ``n`` and a
            two-column ``t`` exactly as
            :func:`~surpyval.utils.xcnt_handler` returns them.

        Examples
        --------
        Basic usage with uncensored data:
        >>> import numpy as np
        >>> from surpyval import SurpyvalData
        >>> x = np.array([1, 2, 3])
        >>> data = SurpyvalData(x)

        Right censored data:
        >>> x = np.array([1, 2, 3])
        >>> c = np.array([0, 1, 1])  # 2 and 3 are censored
        >>> data = SurpyvalData(x, c)

        Interval censored data:
        >>> xl = np.array([1, 2, 3])
        >>> xr = np.array([2, 3, 4])
        >>> data = SurpyvalData(xl=xl, xr=xr)

        Interval Censored with nested 2 arrays:
        >>> x = [1, 2, [2, 5], 3, 6]
        >>> c = [0, 1, 2, 0, 0]
        >>> data = SurpyvalData(x=x, c=c)

        With truncation:
        >>> x = np.array([1, 2, 3])
        >>> t = np.array([[0, 5], [0, 5], [0, 5]])
        >>> data = SurpyvalData(x, t=t)
        """

        if i is not None:
            if ids is not None:
                raise TypeError(
                    "Both i= and ids= were given: they are the same "
                    "argument (each row's unit id), so give one."
                )
            ids, id_name = i, "i"
        else:
            id_name = "ids"
        if (
            handle
            and (xl is not None or xr is not None)
            and (x is not None or (s is not None and c is None))
        ):
            from surpyval.utils.data_formats import drop_rows, exact_or_range

            # Read row by row: an exact time in x or a range in xl and xr.
            # Rows giving both are left out (with a warning) of every
            # per-row argument alike.
            x, c, keep = exact_or_range(x, xl, xr, c)
            xl = xr = None
            n, tl, tr, w, Z, e, ids, s, a = (
                drop_rows(v, keep) for v in (n, tl, tr, w, Z, e, ids, s, a)
            )
            t = drop_rows(t, keep, matrix=True)
        if s is not None and handle:
            if c is None:
                # A range row's flag from its bounds, an exact row's from e
                x, c = _state_flags(x, e)
            else:
                c = _flags_from_entered(c, e, x, xl)
        if any(v is not None for v in (Z, e, ids, s, a)):
            group_and_sort = False

        w_arr: np.ndarray | None = None
        if w is not None and handle:
            from surpyval.utils.data_formats import group_sort_weighted
            from surpyval.utils.weights import check_weights

            # Checked against the rows as given (before any grouping),
            # then grouped by weight as well.
            x, c, n, t = xcnt_handler(
                x, c, n, t, xl, xr, tl, tr, group_and_sort=False
            )
            w_arr = check_weights(w, len(c))
            if group_and_sort:
                x, c, n, t, w_arr = group_sort_weighted(x, c, n, t, w_arr)
        elif w is not None:
            w_arr = np.asarray(w, dtype=float)
        elif handle:
            x, c, n, t = xcnt_handler(
                x, c, n, t, xl, xr, tl, tr, group_and_sort=group_and_sort
            )
        self.w = w_arr

        x_arr: np.ndarray = np.asarray(x)
        c_arr: np.ndarray = np.asarray(c)
        n_arr: np.ndarray = np.asarray(n)
        t_arr: np.ndarray = np.asarray(t)

        self.x, self.c, self.n, self.t = x_arr, c_arr, n_arr, t_arr
        self.tl = self.t[:, 0]
        self.tr = self.t[:, 1]
        if x_arr.ndim == 2:
            self.x_min, self.x_max = np.min(x_arr[:, 0]), np.max(x_arr[:, 1])
        else:
            self.x_min, self.x_max = np.min(x_arr), np.max(x_arr)
        self._split_to_observation_types()
        if Z is not None:
            self.add_covariates(Z)
        else:
            self.Z: np.ndarray | None = None
        if covariate_names is not None:
            self.covariate_names = _checked_names(covariate_names, self.Z)
        if s is not None:
            self.s = _state_array(s, len(self.c))
            if e is not None:
                self.e = _entered_states(e, self.c)
        elif e is not None:
            self.e = _event_type_array(e, self.c)
        if ids is not None:
            listed = _kind_ids(id_name, ids, len(self.c))
            check_unit_ids(listed)
            self.ids = listed
        if handle:
            self.a, self.a_inferred = terminal_flags(
                a, self.c, self.x, self.tl, self.ids, s is not None
            )
        elif a is not None:
            self.a = np.asarray(a, dtype=int)
        if unit is not None:
            from surpyval.utils.item_history import _unit

            self.unit = _unit("(data)", unit)
        if m is not None:
            self.m = m

    def add_covariates(self, Z: ArrayLike) -> None:
        """
        Method to add covariates to the data. The covariates are stored in
        the Z attribute of the object. When doing regression survival analysis
        this method allows for the covariates to be added to the data in a
        consistent manner that also allows for the data to be converted to be
        passed to the fitters.

        Parameters
        ----------

        Z : numpy.ndarray
            The covariate array.
        """
        self.covariate_names = (
            _checked_names(
                [str(c) for c in getattr(Z, "columns")], np.asarray(Z)
            )
            if type(Z).__name__ == "DataFrame"
            else None
        )
        # A copy: the caller's array stays theirs (#927)
        Z_arr: np.ndarray = np.array(Z)
        if Z_arr.ndim == 0 or Z_arr.shape[0] != len(self.x):
            raise ValueError(
                f"Z has {Z_arr.shape[0] if Z_arr.ndim else 1} row"
                f"{'' if Z_arr.ndim and Z_arr.shape[0] == 1 else 's'} for "
                f"{len(self.x)} values of x: give one row of covariates "
                "per value."
            )
        self.Z = Z_arr
        # The same masks the observation split used, not fresh ``c``
        # comparisons: censored rows with a finite truncation bound are
        # handed to the likelihood as intervals (#310), and their
        # covariates have to move with them or every observation in the
        # regression likelihood is paired with the wrong covariate row.
        self.Z_o = Z_arr[self.mask_o]
        self.Z_r = Z_arr[self.mask_r]
        self.Z_l = Z_arr[self.mask_l]
        self.Z_i = Z_arr[self.mask_i]
        # Covariates of the truncated subset, aligned with x_tl/x_tr/n_t so
        # that regression likelihoods can apply the truncation correction
        # using each truncated observation's own covariates.
        self.Z_t = Z_arr[self.truncated_mask]

    def _split_to_observation_types(self) -> None:
        """Split the data into the buckets the likelihoods consume.

        A censored observation is only ever known to lie *inside its own
        truncation window* -- it could not have been observed otherwise.
        Its likelihood numerator must therefore be the probability of
        that intersection, not of the raw censoring event:

        =========================  =================  ===============
        row                        correct numerator  unconditional
        =========================  =================  ===============
        left censored at x, tl     F(x) - F(tl)       F(x)
        right censored at x, tr    F(tr) - F(x)       S(x)
        =========================  =================  ===============

        Using the unconditional form makes the contribution exceed one
        -- it counts territory the truncation has already ruled out --
        and the excess grows without bound as the fitted distribution's
        mass slides outside the window. Every likelihood in the package
        then has an unbounded direction and the optimiser runs off down
        it, reporting success at nonsense parameters (#310).

        Rather than teach each likelihood about truncation, such rows
        are handed over as *intervals*: a left-censored row truncated at
        ``tl`` becomes ``[tl, x]``, a right-censored row truncated at
        ``tr`` becomes ``[x, tr]``. The interval term is already a
        difference of CDFs, so it computes the correct numerator with no
        change to any likelihood -- which is why interval-censored data
        never had the bug.

        Only rows with a *finite* bound on the relevant side are
        recast. That is exactly where the defect lives, and it keeps
        right censoring on the exact ``log_sf`` path: expressing it as
        ``1 - F(x)`` loses all precision once ``F(x)`` rounds to one
        (``log_sf`` of -49 comes back as ``-inf``), and the optimiser
        evaluates the likelihood at parameters far enough from the data
        for that to bite. Untruncated fits are therefore bit-identical.
        """
        t = self.t
        has_tl = np.isfinite(t[:, 0])
        has_tr = np.isfinite(t[:, 1])
        recast_l = (self.c == -1) & has_tl
        recast_r = (self.c == 1) & has_tr

        # Kept as attributes because ``add_covariates`` must slice Z with
        # the *same* masks: if a row changes bucket here and its
        # covariates do not follow, the regression likelihood pairs every
        # observation with the wrong covariate row.
        self.mask_o = self.c == 0
        self.mask_r = (self.c == 1) & ~has_tr
        self.mask_l = (self.c == -1) & ~has_tl
        self.mask_i = (self.c == 2) | recast_l | recast_r

        self.x_o, self.n_o = self._split_by_mask(self.mask_o)
        self.x_r, self.n_r = self._split_by_mask(self.mask_r)
        self.x_l, self.n_l = self._split_by_mask(self.mask_l)

        if self.mask_i.any():
            lead = self.x if self.x.ndim == 1 else self.x[:, 0]
            trail = self.x if self.x.ndim == 1 else self.x[:, -1]
            # Interval rows keep their own bounds; recast rows take the
            # truncation bound on the censored side.
            lo = np.where(recast_l, t[:, 0], lead)
            hi = np.where(
                recast_r, t[:, 1], np.where(self.c == 2, trail, lead)
            )
            self.x_il = lo[self.mask_i]
            self.x_ir = hi[self.mask_i]
            self.n_i = self._likelihood_counts()[self.mask_i]
        else:
            self.x_il = np.array([], dtype=float)
            self.x_ir = np.array([], dtype=float)
            self.n_i = np.array([], dtype=int)

        self.truncated_mask = np.isfinite(self.t).any(axis=1)
        self.x_tl, self.x_tr, self.n_t = (
            self.t[self.truncated_mask, 0],
            self.t[self.truncated_mask, 1],
            self._likelihood_counts()[self.truncated_mask],
        )

        # The truncation correction depends only on the observation
        # *window*, not on where in it the observation fell, so it needs
        # evaluating once per distinct window rather than once per row.
        # Truncation is nearly always common to the whole sample -- a
        # single burn-in time, one study entry date -- so this typically
        # collapses hundreds of CDF evaluations per likelihood call into
        # one, and the likelihood is called hundreds of times per fit.
        if self.x_tl.size:
            self.tl_unique, self.tr_unique, inverse = unique_pairs(
                self.x_tl, self.x_tr
            )
            self.n_t_unique = np.bincount(inverse, weights=self.n_t)
        else:
            self.tl_unique = np.array([])
            self.tr_unique = np.array([])
            self.n_t_unique = np.array([])

    @classmethod
    def from_observations(
        cls,
        observed: ArrayLike | None = None,
        right_censored: ArrayLike | None = None,
        left_censored: ArrayLike | None = None,
        interval_censored: ArrayLike | None = None,
        *,
        left_truncation: Number | None = None,
        right_truncation: Number | None = None,
        observed_left_truncation: ArrayLike | None = None,
        observed_right_truncation: ArrayLike | None = None,
        right_censored_left_truncation: ArrayLike | None = None,
        right_censored_right_truncation: ArrayLike | None = None,
        left_censored_left_truncation: ArrayLike | None = None,
        left_censored_right_truncation: ArrayLike | None = None,
        interval_censored_left_truncation: ArrayLike | None = None,
        interval_censored_right_truncation: ArrayLike | None = None,
        observed_counts: ArrayLike | None = None,
        right_censored_counts: ArrayLike | None = None,
        left_censored_counts: ArrayLike | None = None,
        interval_censored_counts: ArrayLike | None = None,
        observed_covariates: ArrayLike | None = None,
        right_censored_covariates: ArrayLike | None = None,
        left_censored_covariates: ArrayLike | None = None,
        interval_censored_covariates: ArrayLike | None = None,
        observed_event_types: Any = None,
        left_censored_event_types: Any = None,
        interval_censored_event_types: Any = None,
        covariate_names: list | None = None,
        observed_ids: Any = None,
        right_censored_ids: Any = None,
        left_censored_ids: Any = None,
        interval_censored_ids: Any = None,
        unit: Any = None,
        **unknown: Any,
    ) -> SurpyvalData:
        """
        The data from a list of each kind of observation: the times at
        which a failure was observed, the times right censored at (the
        suspensions), the times left censored at, and the intervals a
        failure is known to lie in. Any of them may be left out, but not
        all; a time repeated is counted, and a single time may be given as
        a number.

        Each kind takes counts, ``<kind>_counts`` (how many items had
        each value: one number for every item of the kind or one per item;
        an item counted 0 is left out, as in a life table's empty
        interval), and covariates, ``<kind>_covariates`` (one row per
        item, the same columns for every kind; given for one kind, they
        are needed for every kind with items).

        Each kind has its own truncation bounds, ``<kind>_left_truncation``
        and ``<kind>_right_truncation``: one number for every item of that
        kind, or a list with one bound per item, in the same order
        (``-inf`` or ``inf`` for an item with no bound).
        ``left_truncation`` and ``right_truncation`` are one bound shared
        by every item of every kind; giving both it and a kind's own bound
        on the same side is refused, as one would silently replace the
        other.

        Parameters
        ----------
        observed : array like, optional
            The observed failure times.
        right_censored : array like, optional
            The right censored times (suspensions): the unit had not failed
            by then.
        left_censored : array like, optional
            The left censored times: the unit had failed by then.
        interval_censored : array like, optional
            ``[lower, upper]`` pairs, one per unit failed within the
            interval, with ``lower < upper``.
        left_truncation, right_truncation : scalar, optional
            One truncation bound shared by every observation: the window
            the units could be seen in.
        <kind>_left_truncation, <kind>_right_truncation : array like, optional
            The truncation bounds of one kind's items, where ``<kind>`` is
            ``observed``, ``right_censored``, ``left_censored`` or
            ``interval_censored``: one number for every item of the kind,
            or one per item, in the order of its list.
        <kind>_counts : array like, optional
            How many items had each value: one whole number (0 or more) for
            every item of the kind, or one per item.
        <kind>_event_types : optional
            What each failure was (a failure mode or cause, for the
            competing-risks fitters), for ``observed``, ``left_censored``
            and ``interval_censored``: one type for every item of the kind,
            or one per item; given for one kind, needed for every kind of
            failure with items. Suspensions have none.
        <kind>_covariates : array like, optional
            The covariates of each item, one row per item (a single column
            may be given as a list, one value per item); a regression
            ``fit`` takes them as ``Z``.
        covariate_names : list of str, optional
            The name of each covariate column, which names the fitted
            model's coefficients.
        <kind>_ids : array like, optional
            Each item's id, one per entry of the kind's list (given for
            every kind with entries, or for none), as ``from_items`` gives
            them: ``data.item(id)`` and ``select(ids=...)`` then find them.
        unit : str, optional
            The unit of the times, kept with the data.

        Returns
        -------
        SurpyvalData

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData.from_observations(
        ...     observed=[1, 3, 3, 7], right_censored=[5, 9]
        ... )
        >>> data.x
        array([1., 3., 5., 7., 9.])
        >>> data.c, data.n
        (array([0, 0, 1, 0, 1]), array([1, 2, 1, 1, 1]))

        A window shared by everything, and bounds of each kind's own:

        >>> data = SurpyvalData.from_observations(
        ...     observed=[2, 4],
        ...     interval_censored=[[1, 3]],
        ...     left_truncation=0.5,
        ... )
        >>> data.tl
        array([0.5, 0.5, 0.5])
        >>> data = SurpyvalData.from_observations(
        ...     observed=[600, 700],
        ...     observed_left_truncation=[500, 0],
        ...     right_censored=[1000],
        ...     right_censored_left_truncation=500,
        ... )
        >>> data.x, data.tl
        (array([ 600.,  700., 1000.]), array([500.,   0., 500.]))

        A life table, with counts:

        >>> data = SurpyvalData.from_observations(
        ...     interval_censored=[[0, 100], [100, 200], [200, 300]],
        ...     interval_censored_counts=[3, 0, 9],
        ...     right_censored=[300],
        ...     right_censored_counts=12,
        ... )
        >>> data.n
        array([ 3,  9, 12])
        """
        if unknown:
            from surpyval.utils.record_keywords import (
                FROM_OBSERVATIONS_ALIASES,
                accepted_names,
                refuse_unknown_keywords,
            )

            refuse_unknown_keywords(
                "SurpyvalData.from_observations",
                unknown,
                accepted_names(SurpyvalData.from_observations),
                FROM_OBSERVATIONS_ALIASES,
            )
        lists = {
            "observed": observed,
            "right_censored": right_censored,
            "left_censored": left_censored,
            "interval_censored": interval_censored,
        }
        own = {
            "observed_left_truncation": observed_left_truncation,
            "observed_right_truncation": observed_right_truncation,
            "right_censored_left_truncation": right_censored_left_truncation,
            "right_censored_right_truncation": (
                right_censored_right_truncation
            ),
            "left_censored_left_truncation": left_censored_left_truncation,
            "left_censored_right_truncation": left_censored_right_truncation,
            "interval_censored_left_truncation": (
                interval_censored_left_truncation
            ),
            "interval_censored_right_truncation": (
                interval_censored_right_truncation
            ),
        }
        counts = {
            "observed": observed_counts,
            "right_censored": right_censored_counts,
            "left_censored": left_censored_counts,
            "interval_censored": interval_censored_counts,
        }
        event_types = {
            "observed": observed_event_types,
            "left_censored": left_censored_event_types,
            "interval_censored": interval_censored_event_types,
        }
        covariates = {
            "observed": observed_covariates,
            "right_censored": right_censored_covariates,
            "left_censored": left_censored_covariates,
            "interval_censored": interval_censored_covariates,
        }
        ids = {
            "observed": observed_ids,
            "right_censored": right_censored_ids,
            "left_censored": left_censored_ids,
            "interval_censored": interval_censored_ids,
        }
        shared: dict[str, Any] = {
            "left": left_truncation,
            "right": right_truncation,
        }
        if all(times is None for times in lists.values()):
            raise ValueError(
                "Give at least one of observed, right_censored, "
                "left_censored or interval_censored."
            )
        for side, bound in shared.items():
            if bound is not None and np.ndim(bound) != 0:
                raise ValueError(
                    f"'{side}_truncation' must be one bound shared by every "
                    "observation. For a bound per observation, give it "
                    f"with each kind's list: 'observed_{side}_truncation', "
                    f"'right_censored_{side}_truncation' and so on, one "
                    "bound per item in the same order."
                )

        per_item = [
            (name.rsplit("_", 2)[0], name, v) for name, v in own.items()
        ]
        per_item += [(k, f"{k}_counts", v) for k, v in counts.items()]
        per_item += [(k, f"{k}_covariates", v) for k, v in covariates.items()]
        per_item += [
            (k, f"{k}_event_types", v) for k, v in event_types.items()
        ]
        per_item += [(k, f"{k}_ids", v) for k, v in ids.items()]
        for kind, name, values in per_item:
            if values is not None and lists[kind] is None:
                raise ValueError(
                    f"'{name}' was given without '{kind}': it is one value "
                    f"per item of the {kind.replace('_', ' ')} list."
                )
            _check_same_index(kind, lists[kind], name, values)

        xs, cs, ns, tls, trs = [], [], [], [], []
        Zs: dict[str, np.ndarray] = {}
        es: list = []
        id_parts: list = []
        typed: dict[str, bool] = {}
        for kind, flag in _OBSERVATION_KINDS:
            times = _observation_times(kind, lists[kind])
            bounds = {}
            for side, none in (("left", -np.inf), ("right", np.inf)):
                name = f"{kind}_{side}_truncation"
                bound = own[name]
                if bound is None:
                    bound = shared[side]
                elif shared[side] is not None:
                    raise ValueError(
                        f"Both '{side}_truncation' (shared by every "
                        f"observation) and '{name}' were given: give one."
                    )
                bounds[side] = _truncation_bounds(
                    name, bound, len(times), none
                )
            n = _item_counts(f"{kind}_counts", counts[kind], len(times))
            keep = n > 0
            if covariates[kind] is not None:
                Zs[kind] = _item_covariates(
                    f"{kind}_covariates", covariates[kind], len(times)
                )[keep]
            types = event_types.get(kind)
            typed[kind] = types is not None
            if types is not None:
                kind_types = _kind_event_types(
                    f"{kind}_event_types", types, len(times)
                )[keep]
            else:
                kind_types = np.full(len(times), None, dtype=object)[keep]
            es.append(kind_types)
            if ids[kind] is not None:
                kind_ids = _kind_ids(f"{kind}_ids", ids[kind], len(times))
                id_parts.append([kind_ids[k] for k in np.flatnonzero(keep)])
            elif len(times[keep]):
                id_parts.append(None)
            times = times[keep]
            xs.append(times if flag == 2 else np.column_stack([times, times]))
            cs.append(np.full(len(times), flag))
            ns.append(n[keep])
            tls.append(bounds["left"][keep])
            trs.append(bounds["right"][keep])
        x = np.concatenate(xs)
        if len(x) == 0:
            raise ValueError(
                "The lists hold no observations: give at least one time "
                "(with a count above 0)."
            )
        e = None
        if any(typed.values()):
            # A kind's failures given without types are masked causes,
            # warned of with the rest (#928)
            e = np.concatenate(es)
        item_ids = None
        if any(part is not None for part in id_parts):
            if any(part is None for part in id_parts):
                raise ValueError(
                    "Ids were given for some kinds of observation but not "
                    "for all: give '<kind>_ids' for every kind with "
                    "entries, or for none."
                )
            from surpyval.utils.item_history import check_item_ids, item_id

            item_ids = [item_id(i) for part in id_parts for i in part]
            check_item_ids(item_ids, "item (identical items are its counts)")
        Z = _stack_covariates(
            Zs,
            {kind: len(xk) for (kind, _), xk in zip(_OBSERVATION_KINDS, xs)},
        )
        data = cls(
            x,
            np.concatenate(cs),
            np.concatenate(ns),
            tl=np.concatenate(tls),
            tr=np.concatenate(trs),
            Z=Z,
            e=e,
            covariate_names=covariate_names,
            unit=unit,
        )
        if item_ids is not None:
            data.ids = item_ids
        return data

    @classmethod
    def from_dataframe(
        cls,
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
        w_col: Any = None,
        s_col: Any = None,
        a_col: Any = None,
        m: Any = None,
    ) -> SurpyvalData:
        """
        The data from a :class:`pandas.DataFrame` with one row per item
        (or per group of identical items, counted ``n_col``), naming the
        columns as every DataFrame entry point does (``x_col``, ``c_col``,
        ..., the ``fit`` argument with ``_col``). The rows stay in the
        frame's order where ids, event types, states or covariates are
        given.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per item.
        x_col : str, optional
            The column of times, one per row; or give ``xl_col`` and
            ``xr_col``.
        xl_col, xr_col : str, optional
            The age the item was last seen working and the age it was
            first seen failed (``[lower, upper]``): equal for an item
            observed to fail, different for one found failed between two
            looks (interval censored); a blank ``xr_col`` is an item that
            had not failed when last seen (right censored at ``xl_col``),
            a blank ``xl_col`` one already failed at the first look (left
            censored at ``xr_col``). The kind of each row is read from
            its bounds; a ``c_col`` given with them must agree (a row with
            different bounds is flagged 2). Given with ``x_col``, each row
            is read on its own: an exact time in ``x_col`` (the bounds
            blank) or bounds (``x_col`` blank); a row giving both is left
            out, with a warning, and a blank ``c_col`` cell on a row with
            bounds is read from them.
        c_col : str, optional
            The censoring flags, in the ``xcnt`` coding (0 observed, 1
            right censored, -1 left censored, 2 interval censored).
        event_col : str, optional
            A status column, 1 or True where the item failed and 0 or
            False where it was still running (the opposite coding of
            ``c``, as spreadsheets and R have it): ``c = 1 - event``.
        n_col : str, optional
            How many identical items each row stands for.
        tl_col, tr_col : str or number, optional
            The left and right truncation of each row (a column), or one
            bound for every row (a number).
        i_col : str, optional
            The item (unit) ids, ``data.i`` (each once; with ``s_col``, a
            unit's id on each of its rows): ``data.item(id)`` and
            ``select(ids=...)`` then find them.
        e_col : str, optional
            The event type of each failure (a failure mode or cause); a
            blank cell on a suspension is nothing, on a failure a type
            not recorded (a masked cause, warned of). With ``s_col``, the
            state the unit moved into at ``x`` (blank where it had not
            moved), from which ``c`` is read where no ``c_col`` or
            ``event_col`` is given.
        s_col : str, optional
            The state each row's unit was in during the row, for
            multi-state data (one row per stay in a state, entered at
            ``tl_col``): ``data.s``.
        a_col : str, optional
            The terminal flag of each row, ``data.a``: 1 where the row's
            event ended the unit's history for good, 0 where the unit
            carried on, blank where it is to be read from the rows (see
            the constructor's ``a``).
        Z_cols : list of str, optional
            The covariate columns, which name the covariates
            (``covariate_names``); numbers (a blank cell is a covariate
            not recorded). A text column is refused, naming what to do.
        attribute_cols : list of str, optional
            Columns describing the items without being fitted (a site, a
            batch, a product line: any values), kept as each record's
            ``attributes`` for ``groupby`` and ``select``.
        unit : str, optional
            The unit of the times (``"hours"``), kept with the data and
            taken by ``fit(data)``.
        w_col : str, optional
            The weight of each row, ``w``.
        m : Measurements or dict, optional
            The units' readings, ``data.m`` (a frame of readings is read
            with ``Measurements.from_dataframe``); needs ``i_col``.

        Returns
        -------
        SurpyvalData

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import SurpyvalData
        >>> df = pd.DataFrame({
        ...     "unit": ["B7", "B8", "B9"],
        ...     "hours": [1820, 2400, 1500],
        ...     "failed": [True, False, True],
        ...     "mode": ["spall", None, "crack"],
        ...     "load": [1.0, 0.5, 0.8],
        ... })
        >>> data = SurpyvalData.from_dataframe(
        ...     df, i_col="unit", x_col="hours", event_col="failed",
        ...     e_col="mode", Z_cols=["load"],
        ... )
        >>> data.c, data.ids
        (array([0, 1, 0]), ['B7', 'B8', 'B9'])
        >>> data.item("B9").event_type, data.covariate_names
        ('crack', ['load'])

        Last seen working and first seen failed, with blanks:

        >>> df = pd.DataFrame({
        ...     "last_ok": [1000, 2400, None],
        ...     "first_failed": [1500, None, 800],
        ... })
        >>> data = SurpyvalData.from_dataframe(
        ...     df, xl_col="last_ok", xr_col="first_failed"
        ... )
        >>> data.c
        array([-1,  2,  1])

        Stays in states (an illness-death study), the flags read from the
        state entered:

        >>> df = pd.DataFrame({
        ...     "patient": [1, 1, 2, 3],
        ...     "entered": [0, 3, 0, 0],
        ...     "left": [3, 5, 7, 4],
        ...     "state": ["well", "ill", "well", "well"],
        ...     "to": ["ill", "dead", None, "dead"],
        ... })
        >>> data = SurpyvalData.from_dataframe(
        ...     df, i_col="patient", tl_col="entered", x_col="left",
        ...     s_col="state", e_col="to",
        ... )
        >>> data.c, data.n_items
        (array([0, 0, 1, 0]), 3)
        """
        from surpyval.utils.data_frames import lifetime_from_frame

        data = lifetime_from_frame(
            cls,
            df,
            x_col=x_col,
            xl_col=xl_col,
            xr_col=xr_col,
            c_col=c_col,
            event_col=event_col,
            n_col=n_col,
            tl_col=tl_col,
            tr_col=tr_col,
            i_col=i_col,
            e_col=e_col,
            Z_cols=Z_cols,
            attribute_cols=attribute_cols,
            unit=unit,
            w_col=w_col,
            s_col=s_col,
            a_col=a_col,
        )
        if m is not None:
            data.m = m
        return data

    def to_dataframe(self) -> Any:
        """
        The rows as a :class:`pandas.DataFrame`: ``id`` (where the data
        has ids), ``x`` (or ``xl`` and ``xr`` where any row is an
        interval), ``c``, ``n``, ``tl`` and ``tr`` (where any bound is
        finite), ``s`` (where it has states), ``e`` (where it has event
        types), the covariates by name (``Z0``, ``Z1``, ... where
        unnamed) and the attributes.
        ``SurpyvalData.from_dataframe(data.to_dataframe(), x_col="x",
        c_col="c", n_col="n", i_col="id", s_col="s", ...)`` is the same
        data. Time-varying covariates have no place in it, and are
        refused.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1]).to_dataframe()
             x  c  n
        0  1.0  0  1
        1  2.0  0  2
        2  5.0  1  1
        """
        from surpyval.utils.data_frames import lifetime_to_frame

        return lifetime_to_frame(self)

    @classmethod
    def from_xrd(
        cls, x: ArrayLike, r: ArrayLike, d: ArrayLike
    ) -> SurpyvalData:
        """
        The data from the ``xrd`` format: the distinct times ``x``, the
        number at risk ``r`` and the number failing ``d`` at each. Each
        failure becomes an observed row, and the units that leave the risk
        set without failing between ``x[j]`` and ``x[j + 1]`` (``r[j] -
        d[j] - r[j + 1]``) right censored rows at ``x[j]``. The format
        cannot hold left truncation (the numbers at risk have absorbed
        it): give that data as ``SurpyvalData(x, c, n, tl=...)``.

        Parameters
        ----------
        x : array like
            The distinct times.
        r : array like
            The number at risk at each time (those failing there
            included).
        d : array like
            The number failing at each time.

        Returns
        -------
        SurpyvalData

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData.from_xrd([1, 2, 3], [5, 3, 1], [1, 1, 1])
        >>> data.x, data.c
        (array([1., 1., 2., 2., 3.]), array([0, 1, 0, 1, 0]))
        """
        from surpyval.utils.data_formats import xrd_to_xcnt

        x_, c_, n_, t_ = xrd_to_xcnt(x, r, d)
        return cls(x_, c_, n_, t_)

    @classmethod
    def from_items(
        cls, items: Any, counts: Any = None, unit: Any = None, **unknown: Any
    ) -> SurpyvalData:
        """
        The data of several items' lifetimes, each an
        :class:`~surpyval.utils.item_lifetime.ItemLifetime`, with ``counts``
        (one number for every item, one per item, or a dict ``{id:
        count}``, the items it leaves out counting 1) saying how many
        identical items each stands for; an item counted 0 is left out.
        Ids must differ and be of one type; covariates, given for one item,
        are needed for every item (the same number of values), and so are
        time-varying covariates; event types, given for one failure, are
        needed for every failure. The rows stay in the order of the items.
        The data's ``unit`` is the records' (which must agree), or
        ``unit=`` where they have none.
        Items' marker readings (``markers``, #891) are kept as
        ``data.markers``, ``{id: {age: value}}`` (``{}`` for an item with
        none), for the joint models.

        Examples
        --------
        >>> from surpyval import ItemLifetime, SurpyvalData
        >>> data = SurpyvalData.from_items(
        ...     [
        ...         ItemLifetime("A", observed=5),
        ...         ItemLifetime("B", right_censored=9),
        ...     ],
        ...     counts=[3, 10],
        ... )
        >>> data.x, data.c, data.n
        (array([5., 9.]), array([0, 1]), array([ 3, 10]))
        """
        from surpyval.utils.item_lifetime import ItemLifetime
        from surpyval.utils.record_keywords import (
            FROM_ITEMS_ALIASES,
            refuse_unknown_keywords,
        )

        refuse_unknown_keywords(
            "SurpyvalData.from_items",
            unknown,
            ["items", "counts", "unit"],
            FROM_ITEMS_ALIASES,
        )
        items = _record_list(items, ItemLifetime, "SurpyvalData")
        from surpyval.utils.item_history import (
            check_item_ids,
            covariate_names_of,
            covariate_row_in,
        )

        if isinstance(counts, dict):
            from surpyval.utils.item_history import item_id

            given = {item_id(k): v for k, v in counts.items()}
            not_given = sorted(
                (repr(k) for k in set(given) - {i.id for i in items}), key=str
            )
            if not_given:
                raise ValueError(
                    "'counts' names items not given: "
                    + ", ".join(not_given)
                    + "."
                )
            counts = [given.get(i.id, 1) for i in items]
        n = _item_counts("counts", counts, len(items))
        items = [item for item, k in zip(items, n) if k > 0]
        n = n[n > 0]
        if not items:
            raise ValueError("Every count is 0: there is no data.")
        ids = [item.id for item in items]
        check_item_ids(ids, "ItemLifetime (identical items are its counts)")
        from surpyval.utils.item_history import units_of

        unit = units_of(items, unit, "items' ages")
        static = _every_or_none(items, "covariates", lambda i: i.covariates)
        names = covariate_names_of(items)
        varying = _every_or_none(
            items,
            "time_varying_covariates",
            lambda i: (
                None
                if i.time_varying_covariates is None
                else [0.0] * i.time_varying_covariates.p
            ),
        )
        failures = [i for i in items if i.outcome != "right_censored"]
        typed = [i.event_type is not None for i in failures]
        masked = [i.id for i, t in zip(failures, typed) if not t]
        rows, flags = zip(*(item._row() for item in items))
        x = np.array(rows, dtype=float)
        if 2 not in flags:
            x = x[:, 0]
        tl = np.array(
            [
                -np.inf if i.left_truncation is None else i.left_truncation
                for i in items
            ]
        )
        tr = np.array(
            [
                np.inf if i.right_truncation is None else i.right_truncation
                for i in items
            ]
        )
        truncated = np.isfinite(tl).any() or np.isfinite(tr).any()
        e = None
        if any(typed):
            e = np.empty(len(items), dtype=object)
            e[:] = [i.event_type for i in items]
            if masked:
                warn_masked(len(masked), "items " + named(masked))
        with ignoring_warnings(MaskedCauseWarning):
            # Warned of above, by item
            data = cls(
                x,
                np.array(flags),
                n,
                tl=tl if truncated else None,
                tr=tr if truncated else None,
                Z=(
                    np.vstack([covariate_row_in(i, names) for i in items])
                    if static
                    else None
                ),
                group_and_sort=False,
                e=e,
                covariate_names=names,
                unit=unit,
            )
        data.ids = ids
        if any(i.attributes is not None for i in items):
            data.attributes = [i.attributes for i in items]
        if varying:
            data.time_varying_covariates = {
                i.id: i.time_varying_covariates for i in items
            }
        if any(i.markers is not None for i in items):
            data.markers = {i.id: i.markers or {} for i in items}
        return data

    def to_items(self) -> list:
        """
        One :class:`~surpyval.utils.item_lifetime.ItemLifetime` per row, in
        order (the item ids of :meth:`from_items`, else the row numbers);
        ``data.n`` gives how many identical items each stands for, so
        ``SurpyvalData.from_items(data.to_items(), counts=data.n)`` is the
        same data.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> SurpyvalData([3, 7], c=[0, 1]).to_items()
        [ItemLifetime(0, observed=3.0), ItemLifetime(1, right_censored=7.0)]
        """
        self._refuse_records("make records of")
        self._refuse_state_records("to_items()")
        self._refuse_measurement_records("to_items()")
        prepared = self._prepared()
        return [self._item_at(k, prepared) for k in range(len(self.x))]

    _PARTS = LIFETIME_PARTS

    def _records(self) -> tuple[list, list]:
        # The rows' records, for a reshape: with states, each record is
        # one row (one stay), its state kept beside it by the reshape
        # (``_from_kept``).
        self._refuse_records("reshape")
        prepared = self._prepared()
        records = [self._item_at(k, prepared) for k in range(len(self.x))]
        return records, list(np.asarray(self.n))

    def _refuse_state_records(self, what: str) -> None:
        """Refuse to give records of data with states, or with units of
        several rows or a terminal flag a record cannot say, which an
        ``ItemLifetime`` has no place for."""
        units = units_with_several_rows(self)
        if units and self.s is None:
            raise ValueError(
                f"{what} gives ItemLifetime records, one per unit, and "
                f"unit{'s' if len(units) > 1 else ''} {named(units)} "
                f"{'have' if len(units) > 1 else 'has'} several rows (one "
                "per stretch or event), which would be several records "
                "under one id. Use data.to_dataframe() for the rows, "
                "data.select(ids=...) for a unit's rows, or data[rows]."
            )
        if (
            self.a is not None
            and ((self.a == 1) != (np.asarray(self.c) != 1)).any()
        ):
            raise ValueError(
                f"{what} gives ItemLifetime records, which have no place "
                "for an event that did not end the unit's history (a = "
                "0). Use data.to_dataframe() for the rows, or data[rows]."
            )
        if self.s is not None:
            raise ValueError(
                f"{what} gives ItemLifetime records, which have no place "
                "for the state s each row's unit was in (and a unit with "
                "several stays would be several records under one id). "
                "Use data.to_dataframe() for the rows, data.select(ids=...) "
                "for a unit's rows, or data[rows]."
            )

    def _refuse_records(self, what: str) -> None:
        """Refuse to make records of weighted data, which have no place
        for a weight."""
        if self.w is not None:
            raise ValueError(
                f"Cannot {what} weighted data yet: its records "
                "(ItemLifetime) have no place for a weight w. Reshape the "
                "arrays (data.x, data.c, data.n, data.t, data.w) and build "
                "a new SurpyvalData(..., w=...), or index rows with "
                "data[rows], which keeps the weights."
            )

    def _from_records(self, records: list, counts: list) -> SurpyvalData:
        out = SurpyvalData.from_items(records, counts=counts)
        if self.ids is None:
            out.ids = None  # its items were row numbers, not ids
        return self._carry_measurements(out)

    def _from_kept(self, rows: list, records: list, counts: list) -> Any:
        """The data of the reshaped ``records`` of the rows ``rows``. With
        states, a record has no place for its row's state, and a unit's
        id repeats over its stays, so the records are numbered by row and
        the ids and states are put back by row."""
        if self.s is None and self.a is None:
            return self._from_records(records, counts)
        from surpyval.utils.item_lifetime import ItemLifetime

        numbered = [
            ItemLifetime(k, **record._fields())
            for k, record in enumerate(records)
        ]
        out = SurpyvalData.from_items(numbered, counts=counts)
        out.ids = None if self.ids is None else [self.ids[r] for r in rows]
        kept = np.asarray(rows, dtype=int)
        out.s = None if self.s is None else self.s[kept]
        if self.a is not None:
            # A row a reshape censored has no terminal event
            out.a = np.where(out.c == 1, 0, self.a[kept])
            if self.a_inferred is not None:
                out.a_inferred = self.a_inferred[kept] & (out.c != 1)
        return self._carry_measurements(out)

    def _covariate_levels(self, column: int) -> list:
        Z = np.asarray(self.Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        return Z[:, column].tolist()

    def _rows_of_items(self, keep: np.ndarray) -> SurpyvalData:
        from surpyval.utils.lifetime_reshape import rebuilt

        return rebuilt(self, keep)

    @property
    def attribute_names(self) -> list:
        """The names of the attributes any item has (an empty list where
        the data has none)."""
        return _attribute_names(self.attributes)

    def item(self, id: Any) -> Any:
        """
        The lifetime of the item ``id`` (its id from :meth:`from_items`,
        else its row number), as an
        :class:`~surpyval.utils.item_lifetime.ItemLifetime`; how many
        identical items it stands for is its count in ``n``.

        Examples
        --------
        >>> from surpyval import ItemLifetime, SurpyvalData
        >>> data = SurpyvalData.from_items(
        ...     [
        ...         ItemLifetime("B7", observed=1820),
        ...         ItemLifetime("B8", right_censored=2400),
        ...     ]
        ... )
        >>> data.item("B8")
        ItemLifetime('B8', right_censored=2400.0)
        """
        from surpyval.utils.item_history import item_id

        self._refuse_state_records(f"item({id!r})")
        self._refuse_measurement_records(f"item({id!r})")
        key = item_id(id)
        if self.ids is not None:
            matches = [k for k, i in enumerate(self.ids) if i == key]
            if not matches:
                raise KeyError(f"No item {key!r} in the data.")
            return self._item_at(matches[0], self._prepared())
        if isinstance(key, (int, np.integer)) and 0 <= key < len(self.x):
            return self._item_at(int(key), self._prepared())
        raise KeyError(
            f"No item {key!r}: this data has no item ids, so its items are "
            f"its row numbers, 0 to {len(self.x) - 1}."
        )

    def _prepared(self) -> tuple:
        """The arrays every row's record reads, made once for all rows."""
        self._refuse_records("make records of")
        x = self.x if self.x.ndim == 2 else np.column_stack([self.x] * 2)
        Z = None
        if self.Z is not None:
            Z = np.asarray(self.Z, dtype=float)
            Z = Z[:, None] if Z.ndim == 1 else Z
        return x, Z

    def _item_at(self, k: int, prepared: tuple) -> Any:
        from surpyval.utils.item_lifetime import ItemLifetime

        outcome = {
            0: "observed",
            1: "right_censored",
            -1: "left_censored",
            2: "interval_censored",
        }
        x, Z = prepared
        item = self.ids[k] if self.ids is not None else k
        flag = int(self.c[k])
        value = x[k].tolist() if flag == 2 else float(x[k, 0])
        tl, tr = self.t[k]
        return ItemLifetime(
            item,
            **{outcome[flag]: value},
            left_truncation=float(tl) if np.isfinite(tl) else None,
            right_truncation=float(tr) if np.isfinite(tr) else None,
            event_type=None if self.e is None else self.e[k],
            covariates=_named_row(Z, k, self.covariate_names),
            time_varying_covariates=(
                None
                if self.time_varying_covariates is None
                else self.time_varying_covariates[item]
            ),
            markers=_readings_of(self._plain_markers(), item),
            attributes=(
                None if self.attributes is None else self.attributes[k]
            ),
            unit=self.unit,
        )

    def _markers_by_row(self) -> list | None:
        """Each row's marker readings, ``{age: value}`` (``{}`` for an item
        with none), in the order of the rows; None where the data has
        none."""
        markers = self.markers
        if markers is None:
            return None
        keys = self.ids if self.ids is not None else range(len(self.x))
        return [dict(markers.get(key, {})) for key in keys]

    def summary(self) -> str:
        """
        A short description of the data: its items (a row counted ``n``
        is ``n`` items) and rows, their outcomes, and what else it holds
        (truncation, event types, covariates, time-varying covariates,
        markers, attributes), each counted in items.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> print(SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1]).summary())
        SurpyvalData: 4 items, 3 rows
          outcomes: 3 observed, 1 right censored
        """
        from surpyval.utils.data_display import heading, held_lines

        c, n = np.asarray(self.c), np.asarray(self.n)
        names = (
            (0, "observed"),
            (1, "right censored"),
            (-1, "left censored"),
            (2, "interval censored"),
        )
        head = heading(self, recurrent=False)
        if self.ids is not None:
            head += " (with item ids)"
        lines = [
            head,
            "  outcomes: "
            + ", ".join(
                f"{int(n[c == flag].sum())} {name}"
                for flag, name in names
                if (c == flag).any()
            ),
        ]
        if self.a is not None:
            lines.append("  " + self._units_line())
        lines += held_lines(self, recurrent=False)
        if self.w is not None:
            from surpyval.utils.weights import effective_sample_size

            lines.append(
                f"  weights: {self.w.min():.4g} to {self.w.max():.4g}, "
                "effective sample size "
                f"{effective_sample_size(self.n, self.w):.4g}"
            )
        return "\n".join(lines)

    def _units_line(self) -> str:
        """``"units: 12 (3 ended for good, 9 still running)"``: each
        unit's history ended for good where its last row has ``a == 1``
        (counted by that row's ``n``)."""
        n = np.asarray(self.n)
        last: dict = {}
        units = self.ids if self.ids is not None else range(len(n))
        for r, unit in enumerate(units):
            last[unit] = r
        total = self.n_items
        a = np.asarray(self.a)
        ended = int(sum(n[r] for r in last.values() if a[r] == 1))
        return (
            f"units: {total} ({ended} ended for good, "
            f"{total - ended} still running)"
        )

    @property
    def n_items(self) -> int:
        """The number of items: the rows weighted by their counts ``n``
        (``len(data)`` is the number of rows). With unit ids a unit may
        have several rows (stays in states, stretches, repeated events),
        so the items are the distinct units, each counted by the ``n`` of
        its first row."""
        n = np.asarray(self.n)
        if self.ids is not None and units_with_several_rows(self):
            first: dict = {}
            for k, unit in enumerate(self.ids):
                first.setdefault(unit, k)
            return int(n[list(first.values())].sum())
        return int(n.sum())

    @property
    def i(self) -> list | None:
        """Each row's unit (item) id, as ``ids``: the two names are one
        list, and setting either sets both."""
        return self.ids

    @i.setter
    def i(self, value: list | None) -> None:
        self.ids = value

    @property
    def event_types(self) -> list:
        """The distinct event types of the failures, in order (an empty
        list where the data has none); a failure whose type was not
        recorded (None, a masked cause) is not one."""
        from surpyval.univariate.competing_risks.labels import ordered_labels

        if self.e is None:
            return []
        return ordered_labels(self.e)

    def records(self) -> Any:
        """
        The rows as :class:`~surpyval.utils.item_lifetime.ItemLifetime`
        records, one at a time (a generator: :meth:`to_items` builds the
        list), with the counts ``n`` saying how many identical items each
        stands for.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData([3, 7], c=[0, 1])
        >>> [item.outcome for item in data.records()]
        ['observed', 'right_censored']
        """
        self._refuse_records("make records of")
        self._refuse_state_records("records()")
        self._refuse_measurement_records("records()")
        prepared = self._prepared()
        for k in range(len(self.x)):
            yield self._item_at(k, prepared)

    def to_xcnt(self) -> tuple:
        """
        The data in the ``xcnt`` format, ``(x, c, n, t)``: the times (a
        ``[lower, upper]`` row where any observation is an interval), the
        censoring flags (0 observed, 1 right, -1 left, 2 interval), the
        counts and the ``[left, right]`` truncation bounds, as copies.
        Covariates are not part of the format: they are ``data.Z``.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> x, c, n, t = SurpyvalData([3, 1, 3], [0, 1, 0]).to_xcnt()
        >>> x, c, n
        (array([1., 3.]), array([1, 0]), array([1, 2]))
        """
        return self.x.copy(), self.c.copy(), self.n.copy(), self.t.copy()

    def to_observations(self) -> dict:
        """
        The data in the form :meth:`from_observations` reads, so that
        ``SurpyvalData.from_observations(**data.to_observations())`` is the
        same data: the ``"observed"``, ``"right_censored"`` and
        ``"left_censored"`` times and the ``"interval_censored"``
        ``[lower, upper]`` pairs, each value once, with each kind's
        ``"<kind>_counts"`` where any count is not 1, its
        ``"<kind>_left_truncation"`` / ``"<kind>_right_truncation"`` for each
        side on which anything is truncated (``-inf`` or ``inf`` for an item
        with no bound), and its ``"<kind>_covariates"`` where the data has
        covariates (with ``"covariate_names"`` where they are named), its
        ``"<kind>_event_types"`` and ``"<kind>_ids"`` where the data has
        them. ``np.repeat(times, counts)`` gives one entry per item.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData([1, 3, 5], c=[0, 0, 1], n=[1, 2, 1])
        >>> lists = data.to_observations()
        >>> lists["observed"], lists["observed_counts"]
        (array([1., 3.]), array([1, 2]))
        >>> data = SurpyvalData(
        ...     [600, 700, 1000], c=[0, 0, 1], tl=[500, 0, 500]
        ... )
        >>> data.to_observations()["observed_left_truncation"]
        array([500.,   0.])
        """
        if self.s is not None:
            raise ValueError(
                "to_observations lists the times by kind of observation, "
                "and from_observations has no place for the state s of "
                "each row: use data.to_dataframe() or data.to_dict(), "
                "which keep the states."
            )
        counts = self.n.astype(int)
        x = self.x if self.x.ndim == 2 else np.column_stack([self.x] * 2)
        Z = None
        if self.Z is not None:
            Z = np.asarray(self.Z, dtype=float)
            Z = Z[:, None] if Z.ndim == 1 else Z
        truncated = [
            (side, column)
            for column, side in enumerate(("left", "right"))
            if np.isfinite(self.t[:, column]).any()
        ]
        lists: dict[str, np.ndarray] = {}
        extra: dict[str, Any] = {}
        for kind, flag in _OBSERVATION_KINDS:
            mask = self.c == flag
            lists[kind] = (x[mask] if flag == 2 else x[mask, 0]).astype(float)
            if (counts != 1).any():
                extra[f"{kind}_counts"] = counts[mask]
            for side, column in truncated:
                extra[f"{kind}_{side}_truncation"] = self.t[
                    mask, column
                ].astype(float)
            if Z is not None:
                extra[f"{kind}_covariates"] = Z[mask]
            if self.e is not None and kind != "right_censored":
                extra[f"{kind}_event_types"] = list(self.e[mask])
            if self.ids is not None:
                extra[f"{kind}_ids"] = [
                    self.ids[k] for k in np.flatnonzero(mask)
                ]
        if self.covariate_names is not None:
            extra["covariate_names"] = list(self.covariate_names)
        return {**lists, **extra}

    def to_xrd(self, estimator: str = "Nelson-Aalen") -> tuple:
        """
        Converts the data into the xrd format. Observed and right censored
        data without right truncation is converted exactly with
        :func:`~surpyval.utils.xcnt_to_xrd`. If the data has right truncated
        observations or left or interval censored observations, the
        Turnbull estimator is
        fitted and its (possibly fractional) numbers at risk and deaths are
        returned; the ``estimator`` parameter is used only in that case.

        Parameters
        ----------

            estimator : str, optional
                The method for estimation if data requires the use of the
                Turnbull estimator to convert to xrd, defaults to
                "Nelson-Aalen".

        Returns
        -------

            tuple
                The xrd data, ``(x, r, d)``.
        """
        refuse_multi_state(self, "to_xrd")
        # Cache per estimator: the first call used to be returned for
        # every later call regardless of a different ``estimator``
        # argument (#282).
        if not hasattr(self, "_xrd_cache"):
            self._xrd_cache: dict = {}
        if estimator not in self._xrd_cache and self.w is not None:
            self._xrd_cache[estimator] = self._weighted_xrd()
        if estimator not in self._xrd_cache:
            if (
                np.isfinite(self.t[:, 1]).any()
                or (2 in self.c)
                or (-1 in self.c)
            ):
                data = surpyval.univariate.nonparametric.turnbull(
                    self.x, self.c, self.n, self.t, estimator=estimator
                )
                xrd = data["x"], data["r"], data["d"]
            else:
                xrd = surpyval.utils.xcnt_to_xrd(
                    self.x, self.c, self.n, self.t
                )
            self._xrd_cache[estimator] = xrd
        # Copies, as to_xcnt gives: an edit to the result must not change
        # what the next call returns.
        return tuple(np.copy(part) for part in self._xrd_cache[estimator])

    def _weighted_xrd(self) -> tuple:
        """``to_xrd`` of weighted data: the weighted numbers at risk and of
        deaths at each distinct time."""
        if np.isfinite(self.t[:, 1]).any() or (2 in self.c) or (-1 in self.c):
            raise ValueError(
                "to_xrd of weighted data needs right censored, left "
                "truncated data: the weighted Turnbull estimate of left or "
                "interval censored or right truncated data is not built "
                "yet."
            )
        from surpyval.univariate.nonparametric._weighted import (
            WeightedLadder,
        )

        assert self.w is not None
        ladder = WeightedLadder(self.x, self.c, self.n, self.t, self.w)
        return ladder.x, ladder.r, ladder.d

    def _split_by_mask(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.x[mask] if self.x.ndim == 1 else self.x[mask, 0]
        return x, self._likelihood_counts()[mask]

    def _likelihood_counts(self) -> np.ndarray:
        """The multiplier of each row's term in a likelihood: its count
        ``n``, times its weight where the data is weighted (the weights
        scaled to sum to the units, :attr:`likelihood_weights`)."""
        w = self.likelihood_weights
        if w is None:
            return self.n
        return self.n * w

    @property
    def likelihood_weights(self) -> np.ndarray | None:
        """The weights ``w`` scaled to sum (over the units, ``n``) to the
        number of units, the weights a weighted likelihood uses: a fit
        then searches the same function whatever the scale of ``w``, so
        its answer does not depend on that scale even to the optimiser's
        tolerance. None where the data is unweighted."""
        w = getattr(self, "w", None)
        if w is None:
            return None
        n = np.asarray(self.n, dtype=float)
        return w * (n.sum() / np.sum(n * w))

    @property
    def weighted(self) -> bool:
        """Whether the data carries weights ``w``."""
        return getattr(self, "w", None) is not None

    def __getitem__(self, index: Any) -> SurpyvalData:
        if np.atleast_1d(np.arange(len(self.x))[index]).size == 0:
            # It used to fail on numpy's "zero-size array to reduction
            # operation minimum" (#925).
            raise ValueError(
                "The index selects no rows; a SurpyvalData needs at least "
                "one."
            )
        # A scalar index on interval-censored (2-D) data used to flatten
        # one [xl, xr] row into two 1-D observations and crash the type
        # split; keep the row structure instead (#282).
        if self.x.ndim == 2:
            x_idx = np.atleast_2d(self.x[index])
        else:
            x_idx = np.atleast_1d(self.x[index])
        out = SurpyvalData(
            x_idx,
            np.atleast_1d(self.c[index]),
            np.atleast_1d(self.n[index]),
            np.atleast_2d(self.t[index]),
            handle=False,
            w=None if self.w is None else np.atleast_1d(self.w[index]),
        )
        # Carry the covariates through (slicing used to silently return
        # Z=None, losing alignment, #282).
        if getattr(self, "Z", None) is not None:
            Z_arr = np.asarray(self.Z)
            Z_idx = Z_arr[index]
            if Z_arr.ndim == 2:
                Z_idx = np.atleast_2d(Z_idx)
            else:
                Z_idx = np.atleast_1d(Z_idx)
            out.add_covariates(Z_idx)
            out.covariate_names = self.covariate_names
        # By row number, so a tuple id or type stays one value (#928)
        rows = np.atleast_1d(np.arange(len(self.x))[index])
        if self.e is not None:
            out.e = self.e[rows]
        if self.s is not None:
            out.s = self.s[rows]
        if self.a is not None:
            out.a = self.a[rows]
            if self.a_inferred is not None:
                out.a_inferred = self.a_inferred[rows]
        if self.ids is not None:
            out.ids = [self.ids[r] for r in rows]
        if self.attributes is not None:
            out.attributes = [self.attributes[r] for r in rows]
        out.unit = self.unit
        if self.time_varying_covariates is not None:
            kept = out.ids if out.ids is not None else []
            out.time_varying_covariates = {
                i: self.time_varying_covariates[i] for i in kept
            }
        if self.m is not None and not self.m._plain:
            # Named measurements: the readings of the units kept
            out._m = self.m._for_units(out.ids)
            return out
        by_row = self._markers_by_row()
        if by_row is not None:
            # By id, else by the new row numbers
            keys = out.ids if out.ids is not None else range(len(rows))
            out.markers = {k: by_row[r] for k, r in zip(keys, rows)}
        return out

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self) -> Any:
        return zip(self.x, self.c, self.n, self.t)

    def __repr__(self) -> str:
        """Its size and its first rows, as a table: the item id where it
        has ids, ``x`` (a time, or ``[lower, upper]``), ``c``, ``n``, and
        the truncation, event types and covariates where it has them;
        :meth:`summary` describes all of it."""
        from surpyval.utils.data_display import table_repr

        return table_repr(self, recurrent=False)

    def _repr_html_(self) -> str:
        from surpyval.utils.data_display import html_repr

        return html_repr(self, recurrent=False)

    def __dir__(self) -> list:
        from surpyval.utils.data_display import LIFETIME_INTERNALS, public_dir

        return public_dir(self, LIFETIME_INTERNALS)

    def to_dict(self) -> dict:
        """
        The data as a dictionary of plain Python values, strict JSON when
        dumped: its ``"class"``, the ``x``, ``c``, ``n`` and ``t`` arrays
        as lists, and the covariates, their names, the event types, the
        ids and the time-varying covariates (step schedules; a
        ``CovariatePath`` is a function and is refused) where it has
        them. The infinite truncation bounds (and any other non-finite
        value) are written as ``null`` and recorded under
        ``"non_finite"``, which :meth:`from_dict` reads back (see
        :doc:`/surpyval.serialisation`). ``surpyval.from_dict`` reads it
        too. Data with weights ``w`` is stamped ``"schema": 3`` (a
        weighted model is schema 4), so that a reader without weights
        refuses it rather than drop them, and data with states ``s``
        ``"schema": 4`` for the same reason, as is data with terminal
        flags ``a``; other data has no ``"schema"``. The unit ids are
        under ``"ids"`` (``i``'s other name), the states under ``"s"``,
        the terminal flags under ``"a"`` (and where they were inferred
        under ``"a_inferred"``).

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> SurpyvalData([1, 2], c=[0, 1]).to_dict()["class"]
        'SurpyvalData'
        """
        from surpyval.utils.data_json import label, schedule_dict

        data: dict[str, Any] = {
            "class": "SurpyvalData",
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "t": self.t.tolist(),
        }
        if self.w is not None:
            data["w"] = self.w.tolist()
        if self.Z is not None:
            data["Z"] = self.Z.tolist()
        if self.covariate_names is not None:
            data["covariate_names"] = list(self.covariate_names)
        if self.e is not None:
            data["e"] = [label(v) for v in self.e]
        if self.s is not None:
            data["s"] = [label(v) for v in self.s]
        if self.a is not None:
            data["a"] = [int(v) for v in self.a]
            if self.a_inferred is not None:
                data["a_inferred"] = [bool(v) for v in self.a_inferred]
        if self.ids is not None:
            data["ids"] = [label(v) for v in self.ids]
        if self.attributes is not None:
            data["attributes"] = _attributes_json(self.attributes)
        if self.time_varying_covariates is not None:
            data["time_varying_covariates"] = [
                schedule_dict(schedule, False)
                for schedule in self.time_varying_covariates.values()
            ]
        unit = getattr(self, "unit", None)
        if unit is not None:
            data["unit"] = unit
        if self.m is not None and not self.m._plain:
            # Named measurements (or one with a unit): under "m"
            data["m"] = self.m.to_dict()
            by_row = None
        else:
            by_row = self._markers_by_row()
        if by_row is not None:
            # One list of [age, value] pairs per row (JSON keys are text)
            data["markers"] = [
                [[age, value] for age, value in readings.items()]
                for readings in by_row
            ]
        # Strict JSON, as for the models: the -inf/inf truncation bounds
        # (and any other non-finite value) are written as null with a
        # "non_finite" record of what each stood for (see
        # surpyval.serialisation.encode_non_finite), not as the
        # ``Infinity`` literals strict parsers reject.
        out = encode_non_finite(data)
        if (
            self.w is not None
            or self.s is not None
            or self.a is not None
            or "m" in data
        ):
            # Schema 3 (a weighted model's dict is 4): a reader that
            # checks the schema and has no weights refuses it rather than
            # drop them. Data with states is schema 4, which a reader
            # without them refuses rather than read the stays of every
            # state as one sample; so is data with terminal flags, whose
            # units' several rows such a reader would take as units of
            # their own. Other data keeps its unstamped layout.
            from surpyval.serialisation import required_schema

            out["schema"] = required_schema(out)
        return out

    @classmethod
    def from_dict(cls, data_dict: dict) -> SurpyvalData:
        """
        The data back from its :meth:`to_dict` dictionary. The data is
        not re-validated.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1], tl=0.5)
        >>> again = SurpyvalData.from_dict(data.to_dict())
        >>> again.x, again.n, again.tl
        (array([1., 2., 5.]), array([1, 2, 1]), array([0.5, 0.5, 0.5]))
        """
        from surpyval.utils.data_json import label_back, schedule_back

        if not isinstance(data_dict, dict):
            raise ValueError(
                "Expected a serialised SurpyvalData dict, got "
                f"{type(data_dict).__name__}."
            )
        tag = data_dict.get("class")
        if tag is not None and tag != "SurpyvalData":
            raise ValueError(
                f"This dictionary was written by {tag}, not SurpyvalData: "
                f"read it with {tag}.from_dict, or surpyval.from_dict."
            )
        for key in ("x", "c", "n", "t"):
            if key not in data_dict:
                raise ValueError(
                    f"A serialised SurpyvalData needs {key!r}; this "
                    "dictionary has no such entry."
                )
        # A newer layout is refused, asking for an upgrade. Weights are
        # read wherever they are, whatever the schema: a weighted dict
        # written before it was stamped (no "schema") keeps them.
        check_schema(data_dict)
        # Files written before the strict-JSON convention hold the
        # ``Infinity`` literals, which json.loads still reads.
        data = decode_non_finite(dict(data_dict))

        x = np.array(data["x"])
        c = np.array(data["c"])
        n = np.array(data["n"])
        t = np.array(data["t"])
        Z = np.array(data["Z"]) if "Z" in data else None

        out = cls(
            x,
            c,
            n,
            t,
            Z=Z,
            handle=False,
            covariate_names=data.get("covariate_names"),
            w=None if data.get("w") is None else np.array(data["w"]),
        )
        if "e" in data:
            out.e = _event_type_array(
                [label_back(v) for v in data["e"]], out.c, warn=False
            )
        if "s" in data:
            out.s = _object_array([label_back(v) for v in data["s"]])
        if "a" in data:
            out.a = np.asarray(data["a"], dtype=int)
            if "a_inferred" in data:
                out.a_inferred = np.asarray(data["a_inferred"], dtype=bool)
        if "ids" in data:
            out.ids = [label_back(v) for v in data["ids"]]
        if "attributes" in data:
            out.attributes = _attributes_back(data["attributes"])
        if "time_varying_covariates" in data:
            from surpyval.utils.item_history import _time_varying

            ids = out.ids if out.ids is not None else list(range(len(x)))
            out.time_varying_covariates = {
                i: _time_varying(i, schedule_back(v))[0]
                for i, v in zip(ids, data["time_varying_covariates"])
            }
        if data.get("unit") is not None:
            out.unit = data["unit"]
        if "markers" in data:
            keys = out.ids if out.ids is not None else range(len(out.x))
            out.markers = {
                key: {float(age): float(value) for age, value in pairs}
                for key, pairs in zip(keys, data["markers"])
            }
        if "m" in data:
            from surpyval.utils.measurements import Measurements

            out._m = Measurements.from_dict(data["m"])
        return out

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """
        The data as strict JSON text (:meth:`to_dict` dumped), or written
        to the file ``filepath``.

        Parameters
        ----------
        filepath : str | Path, optional
            If provided, saves JSON to this file path

        Returns
        -------
        str | None
            JSON string if no filepath provided, None if saved to file

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> text = SurpyvalData([1, 2], c=[0, 1]).to_json()
        >>> text.startswith('{"class": "SurpyvalData"')
        True
        """
        from surpyval.utils.data_json import dump

        return dump(self.to_dict(), filepath)

    @classmethod
    def from_json(cls, source: str | Path) -> SurpyvalData:
        """
        The data back from JSON text, or from the file holding it
        (``surpyval.from_json`` reads it too).

        Parameters
        ----------
        source : str | Path
            The JSON text ``to_json()`` returned (a string starting with
            ``{``), or the path of a file ``to_json(path)`` wrote. The
            data is not re-validated.

        Returns
        -------
        SurpyvalData

        Examples
        --------
        >>> import os, tempfile
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData(x=[1, 2, 2, 5], c=[0, 0, 0, 1])
        >>> restored = SurpyvalData.from_json(data.to_json())
        >>> restored.x, restored.n
        (array([1., 2., 5.]), array([1, 2, 1]))
        >>> path = os.path.join(tempfile.mkdtemp(), "data.json")
        >>> data.to_json(path)
        >>> SurpyvalData.from_json(path).n
        array([1, 2, 1])
        """
        from surpyval.utils.data_json import load

        return cls.from_dict(load(source))


# The kinds of observation from_observations reads, with their flags.
_OBSERVATION_KINDS = (
    ("observed", 0),
    ("right_censored", 1),
    ("left_censored", -1),
    ("interval_censored", 2),
)


def _observation_times(kind: str, times: Any) -> np.ndarray:
    """The ``kind`` list of :meth:`SurpyvalData.from_observations` as a
    float array: one-dimensional (a number is a list of one), or
    ``(k, 2)`` ``[lower, upper]`` pairs with ``lower < upper`` for the
    intervals; empty where it was not given."""
    from surpyval.utils.data_formats import validate_float_array

    interval = kind == "interval_censored"
    arr = validate_float_array(times, kind)
    if interval:
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "'interval_censored' must be [lower, upper] pairs, one per "
                "interval (shape (k, 2)); for a single interval, "
                "[[lower, upper]]."
            )
    else:
        arr = arr.reshape(1) if arr.ndim == 0 else arr
        if arr.ndim != 1:
            raise ValueError(f"'{kind}' must be a list of times.")
    if np.isnan(arr).any():
        raise ValueError(f"'{kind}' cannot contain NaN values.")
    if not interval and not np.isfinite(arr).all():
        raise ValueError(
            f"'{kind}' times must be finite: an infinite time carries no "
            "information. For an interval open at one end, give it in "
            "interval_censored as [lower, inf] or [-inf, upper]."
        )
    if interval and (arr[:, 0] >= arr[:, 1]).any():
        raise ValueError(
            "Each interval in 'interval_censored' needs lower < upper."
        )
    return arr


def _check_same_index(kind: str, times: Any, name: str, values: Any) -> None:
    """Refuse a list and its per-item ``values`` that are both pandas
    objects with different indexes: they are matched by position, and
    pandas users expect matching by index, so a reordered or filtered
    index would pair the wrong values silently."""
    index = getattr(times, "index", None)
    other = getattr(values, "index", None)
    # A list's .index is a method; only a pandas index has .equals
    if index is None or other is None:
        return
    if not (hasattr(index, "equals") and hasattr(other, "equals")):
        return
    if not index.equals(other):
        raise ValueError(
            f"'{kind}' and '{name}' are pandas objects with different "
            "indexes. They are matched by position, not by index: give them "
            "the same index (for example both from one DataFrame), or pass "
            "arrays."
        )


def _every_or_none(items: list, name: str, get: Callable) -> bool:
    """Whether the items have ``name``: every one, or none (refused
    otherwise), with values of one length."""
    values = [get(item) for item in items]
    given = [value is not None for value in values]
    if not any(given):
        return False
    if not all(given):
        missing = [i.id for i, g in zip(items, given) if not g]
        raise ValueError(
            f"'{name}' must be given for every item; missing {missing!r}."
        )
    if len({len(np.atleast_1d(value)) for value in values}) > 1:
        raise ValueError(
            f"Every item's '{name}' must have the same number of values."
        )
    return True


def _item_counts(name: str, counts: Any, k: int) -> np.ndarray:
    """A kind's counts, ``name``, one per item: 1 each where not given, a
    number for every item, or a list of ``k``; whole numbers, 0 or
    more."""
    from surpyval.utils.data_formats import validate_float_array

    if counts is None:
        return np.ones(k, dtype=int)
    arr = validate_float_array(counts, name)
    if arr.ndim == 0:
        arr = np.full(k, float(arr))
    elif arr.ndim != 1 or len(arr) != k:
        raise ValueError(
            f"'{name}' must be one count for every item or one per item "
            f"({k}, in the order of its list); got {np.shape(counts)}."
        )
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr % 1).any():
        raise ValueError(f"'{name}' must be whole numbers, 0 or more.")
    return arr.astype(int)


def _item_covariates(name: str, Z: Any, k: int) -> np.ndarray:
    """A kind's covariates, ``name``, as ``(k, p)``: one row per item (a
    list of ``k`` is one covariate; a number, one covariate the same for
    every item)."""
    from surpyval.utils.data_formats import validate_float_array

    arr = validate_float_array(Z, name)
    if arr.ndim == 0:
        arr = np.full((k, 1), float(arr))
    elif arr.ndim == 1 and len(arr) == k:
        arr = arr[:, None]
    if arr.ndim != 2 or len(arr) != k:
        raise ValueError(
            f"'{name}' must have one row per item ({k}, in the order of its "
            f"list); got {np.shape(Z)}. For one item with several "
            "covariates, give a list of one row: [[z1, z2, ...]]."
        )
    # A covariate not recorded is NaN, kept as the datasets keep it
    # (#929); an infinite one is refused
    if np.isinf(arr).any():
        raise ValueError(
            f"'{name}' must be finite numbers (NaN for one not recorded)."
        )
    return arr


def _stack_covariates(
    Zs: dict[str, np.ndarray], sizes: dict[str, int]
) -> np.ndarray | None:
    """The covariates of every kind, stacked in the order of the rows, or
    None where none were given. Given for one kind, they are needed for
    every kind with items, with the same columns."""
    if not Zs:
        return None
    missing = [kind for kind, k in sizes.items() if k and kind not in Zs]
    if missing:
        raise ValueError(
            "Covariates were given for "
            + ", ".join(sorted(Zs))
            + " but not for "
            + ", ".join(missing)
            + ": every item needs its covariates, so give "
            + ", ".join(f"'{kind}_covariates'" for kind in missing)
            + " too."
        )
    widths = {kind: Z.shape[1] for kind, Z in Zs.items() if sizes[kind]}
    if len(set(widths.values())) > 1:
        raise ValueError(
            "Each kind's covariates must have the same columns; got "
            + ", ".join(f"{w} for {kind}" for kind, w in widths.items())
            + "."
        )
    p = next(iter(widths.values()), 1)
    return np.vstack(
        [
            Zs[kind] if sizes[kind] else np.empty((0, p))
            for kind, _ in _OBSERVATION_KINDS
            if kind in Zs or not sizes[kind]
        ]
    )


def _truncation_bounds(
    name: str, bound: Any, k: int, none: float
) -> np.ndarray:
    """A kind's truncation bounds, ``name``, as one per item: ``none``
    (an infinity) for no bound, a number for every item, or a list of
    ``k``, one per item."""
    from surpyval.utils.data_formats import validate_float_array

    if bound is None:
        return np.full(k, none)
    arr = validate_float_array(bound, name)
    if arr.ndim == 0:
        arr = np.full(k, float(arr))
    elif arr.ndim != 1 or len(arr) != k:
        raise ValueError(
            f"'{name}' must be one bound for every item or one per item "
            f"({k}, in the order of its list); got {np.shape(bound)}."
        )
    if np.isnan(arr).any():
        raise ValueError(
            f"'{name}' cannot contain NaN: use {'-inf' if none < 0 else 'inf'}"
            " for an item with no bound."
        )
    return arr


# The arguments a fit takes its data in, besides ``x``.
_DATA_ARGUMENTS = ("c", "n", "t", "tl", "tr", "xl", "xr", "w")

F = TypeVar("F", bound=Callable[..., Any])


def refuses_surpyval_data_with(message: str) -> Callable[[F], F]:
    """A decorator for a ``fit`` whose arrays' rows mean something other
    than a :class:`SurpyvalData`'s rows (the panel models' visits): given
    a ``SurpyvalData`` as its ``x``, it raises a ``TypeError`` with
    ``message``, which says what to pass instead. The wrapper keeps the
    message as ``_refuses_surpyval_data``."""

    def decorate(method: F) -> F:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                return method(*args, **kwargs)
            if isinstance(bound.arguments.get("x"), SurpyvalData):
                raise TypeError(message)
            return method(*args, **kwargs)

        setattr(wrapper, "_refuses_surpyval_data", message)
        return wrapper  # type: ignore[return-value]

    return decorate


def accepts_surpyval_data(method: F) -> F:
    """Let a fitter's ``fit`` take a :class:`SurpyvalData` as its ``x``:
    ``Weibull.fit(data)``. The data is the whole of it, so its other data
    arguments (``c``, ``n``, ``t``, ``tl``, ``tr``, ``xl``, ``xr``, and
    ``Z`` or ``e`` where the data carries covariates or event types) must
    be left out; the fit then runs on the data's own arrays, as from the
    arrays themselves. Data with step time-varying covariates (from
    ``from_items``) is fitted by the model's ``fit_tvc``. A data's
    censoring, counts, truncation, covariates or event types that the fit
    has no argument for are refused, not dropped; so are its marker
    readings, which a fit takes only where it has a ``markers`` argument
    (one ``{age: value}`` per row, the joint models), and data whose rows
    are in more than one state (``s``), which only a fit with a ``state``
    (or ``s``) argument takes. Such a fit is given the data's states
    there, and its unit ids as ``i`` where it has that argument; its
    censoring is then carried by ``e``, so a fit with no ``c`` takes data
    whose flags are 0 and 1. Data without states whose units have
    several rows (stretches or repeated events) is refused, naming
    ``fit_tvc`` or the recurrent models, except by a fit with an ``a``
    argument, which is given the terminal flags ``data.a`` (and the unit
    ids as ``i``)."""
    signature = inspect.signature(method)
    params = list(signature.parameters)
    names = set(params)
    if "x" not in names:
        return method

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        data = bound.arguments.get("x")
        if not isinstance(data, SurpyvalData):
            return method(*args, **kwargs)
        given = _given_beside_data(bound.arguments, data)
        if given:
            raise TypeError(
                "x is a SurpyvalData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than a SurpyvalData)."
            )
        markers = _markers_taken(data, names)
        if data.time_varying_covariates is not None:
            if data.e is not None and "e" not in names:
                raise ValueError(
                    "This fit takes no event types, which the SurpyvalData "
                    "holds."
                )
            owner = bound.arguments.get(params[0]) if params else None
            fit_tvc = getattr(owner, "fit_tvc", None)
            if params[0] in ("self", "cls") and callable(fit_tvc):
                return with_data_unit(
                    named_fit(
                        lambda: _fit_tvc_from_data(
                            fit_tvc, data, _options(bound, signature)
                        ),
                        _tvc_names(data),
                    ),
                    data,
                )
        state = _state_argument(names)
        _check_needed(data, names, bound, signature, state)
        filled: dict[str, Any] = {"x": data.x}
        lost = []
        if "c" in names:
            filled["c"] = data.c
        elif np.any(data.c != 0) and not _flags_in_entered(data, names, state):
            lost.append("censoring")
        lost += _fill_counts(data, names, filled)
        left = bool(np.isfinite(data.tl).any())
        right = bool(np.isfinite(data.tr).any())
        if "t" in names:
            filled["t"] = data.t
        else:
            for side, flag, column in (
                ("left", left, data.tl),
                ("right", right, data.tr),
            ):
                arg = "tl" if side == "left" else "tr"
                if arg in names:
                    filled[arg] = column
                elif flag:
                    lost.append(f"{side} truncation")
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
        if data.time_varying_covariates is not None:
            lost.append("time-varying covariates")
        lost += _fill_histories(data, names, state, filled)
        filled.update(markers)
        if lost:
            raise ValueError(lost_message(lost, "e" in names))
        bound.arguments.update(filled)
        return with_data_unit(
            named_fit(
                lambda: method(*bound.args, **bound.kwargs),
                data.covariate_names if data.Z is not None else None,
            ),
            data,
        )

    setattr(wrapper, "_accepts_surpyval_data", True)
    return wrapper  # type: ignore[return-value]


def _fill_histories(
    data: SurpyvalData, names: set, state: str | None, filled: dict
) -> list:
    """Fill a fit's states (``state`` or ``s``), terminal flags ``a`` and
    units ``i`` from ``data``, where it takes them; what it cannot take
    (rows in several states, units of several rows) is returned."""
    lost = []
    if state is not None and data.s is not None:
        filled[state] = data.s
        if "i" in names and data.ids is not None:
            filled["i"] = data.ids
    elif len(_distinct_states(data)) > 1:
        lost.append("states")
    if "a" in names and data.a is not None:
        # A fit of units' histories: given the terminal flags and units
        filled["a"] = data.a
        if "i" in names and data.ids is not None:
            filled["i"] = data.ids
    elif state is None:
        several = several_rows_clause(data)
        if several is not None:
            lost.append(several)
    return lost


def _state_argument(names: set) -> str | None:
    """The argument a fit takes the states in, ``state`` or ``s``; None
    where it takes none."""
    return next((name for name in ("state", "s") if name in names), None)


def _flags_in_entered(data: SurpyvalData, names: set, state: Any) -> bool:
    """Whether a multi-state fit without ``c`` takes the data's censoring
    in ``e``: with states, a blank ``e`` is a row still in its state
    (``c == 1``), so flags of 0 and 1 are carried by ``e``."""
    return (
        state is not None
        and "e" in names
        and data.s is not None
        and data.e is not None
        and bool(np.isin(data.c, (0, 1)).all())
    )


def _check_needed(
    data: SurpyvalData,
    names: set,
    bound: inspect.BoundArguments,
    signature: inspect.Signature,
    state: str | None,
) -> None:
    """Refuse data that holds none of what the fit needs (an argument
    with no default): covariates, event types or states."""
    needs = [
        (
            "Z",
            data.Z,
            "covariates",
            "SurpyvalData(x, c, ..., Z=Z), or "
            "from_observations(..., observed_covariates=..., ...)",
        ),
        (
            "e",
            data.e,
            "event types",
            "SurpyvalData(x, c, ..., e=e), or "
            "from_observations(..., observed_event_types=..., ...)",
        ),
    ]
    if state is not None:
        needs.append(
            (
                state,
                data.s,
                "states",
                "SurpyvalData(x, ..., s=s, e=e, i=i), or "
                "SurpyvalData.from_dataframe(df, ..., s_col=..., e_col=...)",
            )
        )
    for name, held, what, how in needs:
        if (
            held is None
            and name in names
            and bound.arguments.get(name) is None
            and signature.parameters[name].default is inspect.Parameter.empty
        ):
            raise TypeError(
                f"This fit needs {what}, {name}, and the SurpyvalData "
                f"holds none: build it with {how}."
            )


def _fill_counts(data: SurpyvalData, names: set, filled: dict) -> list:
    """Fill a fit's ``n`` and ``w`` from ``data``; what it cannot take
    (counts other than 1, weights) is returned."""
    lost = []
    if "n" in names:
        filled["n"] = data.n
    elif np.any(data.n != 1):
        lost.append("counts")
    if data.w is not None:
        if "w" in names:
            filled["w"] = data.w
        else:
            lost.append("weights")
    return lost


# What a fit cannot take, and the part of ``without`` that leaves it out.
_WITHOUT_PARTS = {
    "covariates": "covariates",
    "event types": "event_types",
    "time-varying covariates": "time_varying_covariates",
    "left truncation": "truncation",
    "right truncation": "truncation",
}


# What a fit cannot take that is not a thing the data holds but a shape of
# it, each said in a clause of its own.
_OWN_CLAUSE = {
    "states": (
        "takes data in one state only (this data's rows are in more than "
        "one, s)"
    ),
}


def lost_message(lost: list, takes_event_types: bool) -> str:
    """The refusal of ``fit(data)`` for data holding what the fit has no
    argument for, ``lost``, naming the call that fits without it and the
    model that takes it (#933)."""
    own = [p for p in lost if p in _OWN_CLAUSE or p.startswith("takes ")]
    what = " or ".join(
        part.split(" (")[0] for part in lost if part not in own
    )  # ("... (fit them with fit_tvc for now)" is a note of its own)
    parts = list(
        dict.fromkeys(
            _WITHOUT_PARTS[part.split(" (")[0]]
            for part in lost
            if part.split(" (")[0] in _WITHOUT_PARTS
        )
    )
    ways = []
    if parts:
        ways.append("data.without(" + ", ".join(repr(p) for p in parts) + ")")
    if "event_types" in parts:
        ways.append("data.for_event_type(t) for the failures of one mode")
    takes = []
    if "covariates" in parts:
        takes.append(
            "a regression (WeibullPH, CoxPH, ...) takes the covariates"
            if not takes_event_types
            else "CompetingRisksProportionalHazards or FineGray takes the "
            "covariates with the event types"
        )
    if "event_types" in parts:
        takes.append(
            "CompetingRisks.fit(data) takes the event types"
            + (
                " (CompetingRisksProportionalHazards with covariates)"
                if "covariates" in parts
                else ""
            )
        )
    if "weights" in lost:
        takes.append(
            "weights w are taken by KaplanMeier, NelsonAalen, "
            "FlemingHarrington, the parametric distributions (MLE), the "
            "parametric regressions, CoxPH and CompetingRisks"
        )
    if "time_varying_covariates" in parts:
        takes.append(
            "fit_tvc takes the time-varying covariates (fit(data) calls it "
            "for a model that has one)"
        )
    if "states" in lost:
        ways.append("data[data.s == state] for the stays in one state")
        takes.append(
            "the multi-state models (surpyval.multistate.MultiState, "
            "MultiStateProportionalHazards, ParametricMultiState) take the "
            "states"
        )
    clauses = (
        [f"takes no {what}, which the SurpyvalData holds"] if what else []
    )
    clauses += [_OWN_CLAUSE.get(part, part) for part in own]
    message = "This fit " + ", and ".join(clauses) + "."
    if ways:
        message += " To fit without them, use " + ", or ".join(ways) + "."
    if takes:
        message += " " + "; ".join(takes) + "."
    message = message.replace(". a regression", ". A regression")
    return message.replace(". the multi-state", ". The multi-state")


def _given_beside_data(arguments: dict, data: SurpyvalData) -> list:
    """The data arguments given alongside a SurpyvalData ``x``, which holds
    them: ``Z``, ``e`` and ``markers`` count only where the data has them
    (``fit(data, Z=None)`` is fine)."""
    held = {
        "Z": data.Z,
        "e": data.e,
        "markers": data.m,
        "s": data.s,
        "state": data.s,
        "i": data.ids,
        "a": data.a,
    }
    return [
        name
        for name in _DATA_ARGUMENTS
        + ("Z", "e", "markers", "s", "state", "i", "a")
        if arguments.get(name) is not None
        and not (name in held and held[name] is None)
    ]


def _markers_taken(data: SurpyvalData, names: set) -> dict:
    """The data's marker readings as a fit's ``markers`` argument (one
    ``{age: value}`` per row), ``{}`` where it has none; refused for a fit
    that takes none."""
    if data.m is None:
        return {}
    if "markers" not in names:
        raise ValueError(_NO_MARKERS)
    if len(data.m.names) > 1:
        raise ValueError(
            "This fit takes one measurement as its markers, and the "
            f"SurpyvalData holds {len(data.m.names)} ("
            + ", ".join(map(str, data.m.names))
            + "): keep one with data.with_measurement(name), or leave "
            'them out with data.without("m").'
        )
    several = units_with_several_rows(data)
    if several:
        # A unit's readings are the unit's: handing them to each of its
        # rows would count them once per row
        clause = several_rows_clause(data)
        if clause is None and len(_distinct_states(data)) > 1:
            # Stays in several states: the multi-state models take them.
            clause = "states"
        raise ValueError(
            lost_message([clause], "e" in names)
            if clause is not None
            else "This fit takes one {age: value} of markers per unit on "
            f"one row, and unit{'s' if len(several) > 1 else ''} "
            f"{named(several)} {'have' if len(several) > 1 else 'has'} "
            "several rows (stays in states), which would hand a unit's "
            "readings to each of them."
        )
    return {"markers": data._markers_by_row()}


# The refusal of marker readings by a fit that does not take them (#891).
_NO_MARKERS = (
    "This fit takes no markers, which the SurpyvalData holds; use "
    'data.without("markers") (the joint models, such as WeibullJointPH, '
    "fit them)."
)


def _not_taken_yet(data: SurpyvalData) -> list:
    """What a SurpyvalData from ``from_items`` holds that ``fit(data)``
    does not take yet: fitting without it would be another model. (Its
    covariates are not refused here: the regression fitters fit through
    this route with the data's ``Z``.)"""
    out = []
    if data.e is not None:
        out.append("event types")
    if data.time_varying_covariates is not None:
        out.append("time-varying covariates (fit them with fit_tvc for now)")
    if len(_distinct_states(data)) > 1:
        out.append("states")
    several = several_rows_clause(data)
    if several is not None:
        out.append(several)
    return out


def refuses_what_fit_cannot_take(method: F) -> F:
    """Refuse, in a ``fit_from_surpyval_data``, data that holds event
    types or time-varying covariates, which it would otherwise ignore."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data = kwargs.get("surv_data", args[1] if len(args) > 1 else None)
        if isinstance(data, SurpyvalData):
            if data.m is not None:
                raise ValueError(_NO_MARKERS)
            missing = _not_taken_yet(data)
            if data.w is not None and not _takes_weights(args[0]):
                missing.append("weights")
            if missing:
                raise ValueError(lost_message(missing, False))
            return with_data_unit(method(*args, **kwargs), data)
        return method(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _takes_weights(fitter: Any) -> bool:
    """Whether ``fitter``'s ``fit`` takes weights ``w``."""
    fit = getattr(fitter, "fit", None)
    try:
        return "w" in inspect.signature(fit).parameters  # type: ignore
    except (TypeError, ValueError):
        return False


def with_data_unit(model: Any, data: Any) -> Any:
    """``model``, fitted to ``data``, with the data's unit where the
    data has one (#945): a unit the model already has (from ``unit=`` at
    the fit, applied first by ``takes_unit``) must agree."""
    unit = getattr(data, "unit", None)
    if unit is None or model is None:
        return model
    held = getattr(model, "unit", None)
    if held is not None and held != unit:
        raise ValueError(
            f"The data's ages are in {unit!r}, but unit={held!r} was given "
            "to the fit: a model is in the unit of its data (no conversion "
            "is made), so leave unit= out, or convert the data."
        )
    try:
        model.unit = unit
    except AttributeError:
        pass
    return model


def _event_type_array(e: Any, c: np.ndarray, warn: bool = True) -> np.ndarray:
    """``e`` as a 1-D object array, one event type per row: a missing one
    (None, NaN, pandas NA, "") is None. A suspension's type is dropped,
    with a warning (it has not failed); a failure without one is a masked
    cause, warned of where ``warn``."""
    if isinstance(e, (str, bytes)) or (
        not isinstance(e, (list, tuple)) and np.ndim(e) == 0
    ):
        raise ValueError(
            "'e' must hold one event type per row, not a single value."
        )
    out = np.empty(len(c), dtype=object)
    listed = list(e)
    if len(listed) != len(c):
        raise ValueError(
            f"'e' must hold one event type per row ({len(c)}); got "
            f"{len(listed)}."
        )
    out[:] = listed
    out[:] = [None if missing_type(v) else _plain(v) for v in out]
    c = np.asarray(c)
    typed = np.array([v is not None for v in out], dtype=bool)
    suspended = (c == 1) & typed
    if suspended.any():
        out[suspended] = None
        if warn:
            warn_suspension_types(
                int(suspended.sum()),
                "rows " + named(np.flatnonzero(suspended).tolist()),
            )
    masked = (c != 1) & ~typed
    if warn and masked.any():
        warn_masked(
            int(masked.sum()), "rows " + named(np.flatnonzero(masked).tolist())
        )
    return out


def _object_array(values: list) -> np.ndarray:
    """``values`` as a 1-D object array, one entry each (a tuple stays one
    value)."""
    out = np.empty(len(values), dtype=object)
    for k, value in enumerate(values):
        out[k] = value
    return out


def _per_row(name: str, what: str, values: Any, k: int) -> list:
    """``values`` as a list of ``k`` plain values, one per row; a single
    value is refused (as a single ``e`` is)."""
    if isinstance(values, (str, bytes)) or (
        not isinstance(values, (list, tuple)) and np.ndim(values) == 0
    ):
        raise ValueError(
            f"'{name}' must hold one {what} per row, not a single value: "
            f"give a list as long as x ([value] * {k} for the same one on "
            "every row)."
        )
    listed = list(values)
    if len(listed) != k:
        raise ValueError(
            f"'{name}' must hold one {what} per row ({k}); got "
            f"{len(listed)}."
        )
    return [_plain(v) for v in listed]


def _state_array(s: Any, k: int) -> np.ndarray:
    """The states ``s`` as a 1-D object array of plain values (text or
    numbers, as given): one per row, none missing."""
    listed = _per_row("s", "state", s, k)
    missing = [r for r, v in enumerate(listed) if missing_type(v)]
    if missing:
        raise ValueError(
            "'s' must give the state of every row (the state the unit was "
            f"in during the row); {_rows_have(missing).lower()} none (a "
            "blank, None or NaN). Give each row its state."
        )
    for value in listed:
        try:
            hash(value)
        except TypeError:
            raise ValueError(
                "A state in 's' must be text, a number or a tuple of them; "
                f"got {value!r}."
            )
    return _object_array(listed)


def _blank_entries(e: Any, rows: int) -> list:
    """Whether each of the ``rows`` entries of ``e`` is blank (no state
    entered)."""
    return [missing_type(v) for v in _per_row("e", "state", e, rows)]


def _flags_from_entered(c: Any, e: Any, x: Any, xl: Any) -> Any:
    """The censoring flags of data with states: ``c`` as given, or, where
    it is left out, read from ``e`` (a blank: 1, still in the state; a
    state: 0, moved into it at ``x``)."""
    if c is not None:
        return c
    if e is None:
        raise ValueError(
            "Data with states s needs e, the state each row's unit moved "
            "into at x (blank where it was still in its state s at x), or "
            "c (1 where it was still in s, 0 where it moved): give e, and "
            "c is read from it."
        )
    times = x if x is not None else xl
    rows = len(times) if times is not None and np.ndim(times) else 1
    return np.array(
        [1 if blank else 0 for blank in _blank_entries(e, rows)], dtype=int
    )


def _state_flags(x: Any, e: Any) -> tuple[Any, np.ndarray]:
    """The times and flags of data with states given no ``c``: an exact
    row's flag read from ``e`` (blank: 1, still in the state; a state: 0),
    a range row's from its bounds: ``(v, inf)`` still in the state at
    ``v`` (1), ``(-inf, v)`` moved by ``v`` (-1), ``(v1, v2)`` moved
    between them (2); the one-sided ranges are written as the time ``v``
    with their flag. ``e`` must then agree, as for a ``c`` given."""
    flags = _flags_from_entered(None, e, x, None)
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return x, flags
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) != len(flags):
        return x, flags
    lower, upper = arr[:, 0].copy(), arr[:, 1].copy()
    wide = lower != upper
    right = wide & np.isposinf(upper)
    left = wide & np.isneginf(lower) & ~right
    flags = np.where(
        right, 1, np.where(left, -1, np.where(wide, 2, flags))
    ).astype(int)
    upper[right] = lower[right]
    lower[left] = upper[left]
    return np.column_stack([lower, upper]), flags


def _entered_states(e: Any, c: np.ndarray) -> np.ndarray:
    """``e`` of data with states: the state each row moved into (None
    where blank), which must agree with ``c``: a row still in its state
    (``c == 1``) has none, every other row one."""
    listed = _per_row("e", "state", e, len(c))
    out = _object_array([None if missing_type(v) else v for v in listed])
    c = np.asarray(c)
    blank = np.array([v is None for v in out], dtype=bool)
    stayed = np.flatnonzero((c == 1) & ~blank).tolist()
    if stayed:
        raise ValueError(
            f"{_rows_have(stayed)} c = 1 (still in the state s at x) and "
            "a state entered in e. Blank e where the unit had not moved, "
            "or set c to 0 where it moved into e at x."
        )
    moved = np.flatnonzero((c != 1) & blank).tolist()
    if moved:
        raise ValueError(
            f"{_rows_have(moved)} no state entered in e, but c says the "
            "unit moved (0; -1 or 2 for a move seen only at a look). Give "
            "the state it moved into, or set c to 1 where it was still in "
            "its state s at x."
        )
    return out


def _rows_have(rows: list) -> str:
    """'Row 2 has' or 'Rows 2, 5 have'."""
    if len(rows) == 1:
        return f"Row {rows[0]} has"
    return f"Rows {named(rows)} have"


def _distinct_states(data: Any) -> list:
    """The distinct states of ``data``'s ``s``, in the order first seen
    (an empty list where it has none)."""
    s = getattr(data, "s", None)
    return [] if s is None else list(dict.fromkeys(s.tolist()))


def refuse_multi_state(data: Any, what: str) -> None:
    """Refuse, in ``what``, data whose units were in more than one state
    (``s``): taking all its rows as one sample would pool the stays in
    different states into one risk set, another model."""
    several = several_rows_clause(data)
    if several is not None:
        raise ValueError(
            f"{what} {several}: taking every row as a unit of its own "
            "would count a unit several times."
        )
    states = _distinct_states(data)
    if len(states) > 1:
        raise ValueError(
            f"{what} takes data with one starting state, and this data's "
            f"rows are in {len(states)} states (s: "
            f"{', '.join(repr(v) for v in states[:5])}"
            f"{', ...' if len(states) > 5 else ''}); pooling the stays in "
            "different states is another model. " + _STATES_INSTEAD
        )


# What to use for data in more than one state.
_STATES_INSTEAD = (
    "The multi-state models (surpyval.multistate.MultiState, "
    "MultiStateProportionalHazards, ParametricMultiState) take the "
    "states; data[data.s == state] keeps the stays in one state."
)


def check_unit_ids(ids: list) -> None:
    """The rule for the rows' unit ids, in one place: of one kind (all
    text or all numbers). A unit may have several rows: a stay in each
    state, stretches with different covariates, or repeated events."""
    from surpyval.utils.item_history import check_item_ids

    check_item_ids(
        list(dict.fromkeys(ids)), "row (identical items are its count, n)"
    )


def units_with_several_rows(data: Any) -> list:
    """The units (ids) that have more than one row, in the order first
    seen; an empty list where every unit has one (or there are no ids)."""
    ids = getattr(data, "ids", None)
    if ids is None:
        return []
    seen: dict = {}
    for unit in ids:
        seen[unit] = seen.get(unit, 0) + 1
    return [unit for unit, count in seen.items() if count > 1]


def several_rows_clause(data: Any) -> str | None:
    """What a fit of one row per unit says of data without states whose
    units have several rows (None where none has): stretches with
    covariates fixed in each, for the TVC route, or repeated events, for
    the recurrent models."""
    if getattr(data, "s", None) is not None:
        return None
    units = units_with_several_rows(data)
    if not units:
        return None
    a = getattr(data, "a", None)
    repeated = a is not None and bool(
        ((np.asarray(data.c) != 1) & (np.asarray(a) == 0)).any()
    )
    which = f"unit{'s' if len(units) > 1 else ''} {named(units)}"
    if repeated:
        return (
            f"takes one row per unit ({which} "
            f"{'have' if len(units) > 1 else 'has'} several: repeated "
            "events, a = 0 where the unit carried on; the recurrent models "
            "in surpyval.recurrent take them)"
        )
    return (
        f"takes one row per unit ({which} "
        f"{'have' if len(units) > 1 else 'has'} several: stretches with "
        "the covariates fixed in each; the model's fit_tvc takes them, "
        "fit_tvc(i=data.i, xl=data.tl, xr=data.x, c=data.c, Z=data.Z))"
    )


def _terminal_given(a: Any, k: int) -> list:
    """``a`` as one entry per row: 1, 0 or None (blank)."""
    if a is None:
        return [None] * k
    out: list = []
    for value in _per_row("a", "terminal flag", a, k):
        if missing_type(value):
            out.append(None)
        elif isinstance(value, (bool, np.bool_)) or value in (0, 1):
            out.append(int(value))
        else:
            raise ValueError(
                "'a' marks the row whose event ended the unit's history "
                "for good: 1 there, 0 (or blank) on every other row; got "
                f"{value!r}."
            )
    return out


def terminal_flags(
    a: Any, c: Any, x: Any, tl: Any, ids: Any, states: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Each row's terminal flag ``a`` and where it was inferred, checked
    against the rows of each unit (``ids``; without them each row is its
    own unit), which are stretches on the unit's own clock, ``(tl, x]``,
    in time order.

    An event row (``c`` other than 1) with no ``a`` is terminal (1) on
    its unit's last row and not (0) on an earlier one; a ``c == 1`` row
    has no terminal event (0; a 1 there is refused). A row after one with
    ``a == 1`` of the same unit, and a row that starts before the
    previous row of its unit ends (``tl`` before that ``x``, or ``x``
    not after it, which only an event may share: simultaneous events), are
    refused, naming the rows. Returns ``(None, None)`` where ``a`` is not
    given and no unit has more than one row, or where the data has states
    and no ``a`` (the transitions say which states are absorbing).
    """
    c = np.asarray(c)
    k = len(c)
    units = list(range(k)) if ids is None else list(ids)
    rows_of: dict = {}
    for r, unit in enumerate(units):
        rows_of.setdefault(unit, []).append(r)
    several = len(rows_of) < k
    if a is None and (states or not several):
        return None, None
    given = _terminal_given(a, k)
    running = [r for r in range(k) if c[r] == 1 and given[r] == 1]
    if running:
        raise ValueError(
            f"{_rows_have(running)} a = 1 (the unit's history ended for "
            "good at the row's event) and c = 1 (still running at x, so no "
            "event ended it). Set a to 0 or leave it blank on those rows, "
            "or give the event's flag in c where the unit did end there."
        )
    x = np.asarray(x, dtype=float)
    lo, hi = (x[:, 0], x[:, 1]) if x.ndim == 2 else (x, x)
    tl = np.asarray(tl, dtype=float)
    after, overlap = [], []
    for unit, rows in rows_of.items():
        for p, q in zip(rows, rows[1:]):
            if given[p] == 1:
                after.append((unit, p, q))
            elif (
                lo[q] < hi[p]
                or (lo[q] == hi[p] and c[p] == 1)
                or (np.isfinite(tl[q]) and tl[q] < hi[p])
            ):
                overlap.append((unit, p, q))
    if after:
        shown = "; ".join(
            f"row {q} after row {p} of unit {unit!r}"
            for unit, p, q in after[:5]
        )
        raise ValueError(
            f"{len(after)} row{'' if len(after) == 1 else 's'} "
            f"{'comes' if len(after) == 1 else 'come'} after a row with "
            f"a = 1, which ended its unit's history for good ({shown}"
            f"{'; ...' if len(after) > 5 else ''}). Leave the later rows "
            "out, or set a to 0 on the earlier row where the unit carried "
            "on."
        )
    if overlap:
        shown = "; ".join(
            f"row {q} (x = {hi[q]:g}"
            + (f", tl = {tl[q]:g}" if np.isfinite(tl[q]) else "")
            + f") after row {p} (x = {hi[p]:g}) of unit {unit!r}"
            for unit, p, q in overlap[:5]
        )
        raise ValueError(
            f"{len(overlap)} row{'' if len(overlap) == 1 else 's'} "
            f"{'overlaps' if len(overlap) == 1 else 'overlap'} the "
            f"previous row of {'its' if len(overlap) == 1 else 'their'} "
            f"unit or {'comes' if len(overlap) == 1 else 'come'} before it "
            f"({shown}{'; ...' if len(overlap) > 5 else ''}). "
            "Give a unit's rows in time order, each starting (tl) at or "
            "after the previous row's x and ending (x) after it; only "
            "events may share a time (simultaneous events)."
        )
    last = {rows[-1] for rows in rows_of.values()}
    out = np.zeros(k, dtype=int)
    inferred = np.zeros(k, dtype=bool)
    for r in range(k):
        if c[r] == 1:
            continue
        if given[r] is None:
            out[r] = 1 if r in last else 0
            inferred[r] = True
        else:
            out[r] = given[r]
    return out, inferred


def _kind_ids(name: str, value: Any, k: int) -> list:
    """A kind's item ids, ``name``: one per entry of its list."""
    from surpyval.utils.item_history import checked_id

    listed = [value] if isinstance(value, (str, bytes)) else list(value)
    if len(listed) != k:
        raise ValueError(
            f"'{name}' must hold one id per entry ({k}); got {len(listed)}."
        )
    return [checked_id(i) for i in listed]


def _record_list(items: Any, record: type, dataset: str) -> list:
    """``items`` as a list of ``record`` objects (an ``ItemLifetime`` or
    ``ItemHistory``) for ``dataset.from_items``: a single record, a
    dataset or anything else is refused, naming what to give (#933)."""
    name = record.__name__
    if isinstance(items, record):
        raise TypeError(
            f"{dataset}.from_items takes a list of {name} records; got one "
            f"{name}: give [item]."
        )
    if isinstance(items, (str, bytes)) or type(items).__name__ in (
        "SurpyvalData",
        "EventHistoryData",
        "DataFrame",
    ):
        raise TypeError(
            f"{dataset}.from_items takes a list of {name} records; got "
            f"{_a(type(items).__name__)}"
            + (
                ": its records are its to_items()."
                if type(items).__name__.endswith("Data")
                else (
                    f": build the records from its rows, or use "
                    f"{dataset}.from_dataframe(df, ...)."
                    if type(items).__name__ == "DataFrame"
                    else "."
                )
            )
        )
    try:
        listed = list(items)
    except TypeError:
        raise TypeError(
            f"{dataset}.from_items takes a list of {name} records; got "
            f"{_a(type(items).__name__)}."
        )
    if not listed:
        raise ValueError(f"Give at least one {name}.")
    other = next((i for i in listed if not isinstance(i, record)), None)
    if other is not None:
        kind = type(other).__name__
        hint = ""
        if kind in ("ItemLifetime", "ItemHistory"):
            built = (
                "EventHistoryData" if kind == "ItemHistory" else "SurpyvalData"
            )
            hint = f": a list of {kind} records builds {_a(built)}"
            if kind == "ItemHistory":
                hint += (
                    " (EventHistoryData.from_items(items).to_first_events() "
                    "for their first failures as lifetimes)"
                )
        raise TypeError(
            f"{dataset}.from_items takes {name} records; got "
            f"{_a(kind)}{hint}."
        )
    return listed


def _a(word: str) -> str:
    """'a list', 'an ItemHistory'."""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _kind_event_types(name: str, value: Any, k: int) -> np.ndarray:
    """A kind's event types, ``name``: one for every item or one per item,
    as an object array of ``k``; a missing one (None, NaN, "") is a masked
    cause, None (#928)."""
    out = np.empty(k, dtype=object)
    if isinstance(value, (str, bytes)) or np.ndim(value) == 0:
        out[:] = [value] * k
    else:
        listed = list(value)
        if len(listed) != k:
            raise ValueError(
                f"'{name}' must be one type for every item or one per item "
                f"({k}, in the order of its list); got {len(listed)}."
            )
        out[:] = listed
    out[:] = [None if missing_type(v) else _plain(v) for v in out]
    return out


# Arguments a fit takes its data in, or names the fitter itself.
_NOT_OPTIONS = frozenset(
    _DATA_ARGUMENTS + ("x", "Z", "e", "markers", "self", "cls")
)


def _options(
    bound: inspect.BoundArguments, signature: inspect.Signature
) -> dict:
    """The fit options given (``init``, ``fixed``, ``how``, ...), with any
    ``**kwargs`` spread out."""
    options = {}
    for name, value in bound.arguments.items():
        if name in _NOT_OPTIONS:
            continue
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            options.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            options[name] = value
    return options


#: A fit on continuous covariate paths starts with this many steps in each
#: stretch of a path, and halves them until no estimate moves by more than
#: ``_PATH_TOLERANCE`` of its standard error, for at most ``_PATH_ROUNDS``
#: halvings (8, 16, ..., 2048 steps a stretch) and ``_PATH_ROWS`` rows.
_PATH_STEPS = 8
_PATH_TOLERANCE = 0.01
_PATH_ROUNDS = 8
_PATH_ROWS = 2_000_000
#: The share of a failure's last step that holds the path's value at the
#: failure.
_PATH_END = 1e-6


def _fit_tvc_from_data(
    fit_tvc: Callable, data: SurpyvalData, options: dict
) -> Any:
    """Fit ``data`` with time-varying covariates by the model's
    ``fit_tvc``: each item's life, from the start of its observation
    (``left_truncation``, else 0) to its outcome, split where its schedule
    steps into ``(start, stop]`` rows, the covariates of each row being the
    item's static ones (if any) followed by its time-varying ones in force
    there.

    An item whose covariates are a continuous ``CovariatePath`` is fitted
    on the path cut into steps (``CovariatePath.to_steps``; the step a
    failure falls in ends with the path's value at the failure): the fit is
    repeated with the steps halved until no estimate moves by more than
    ``_PATH_TOLERANCE`` of its standard error, and the model says what it
    took in ``path_steps``."""
    from surpyval.univariate.regression.tvc_path import CovariatePath

    if np.isin(data.c, (-1, 2)).any():
        raise ValueError(
            "Time-varying covariates are fitted for observed or right "
            "censored lifetimes (fit_tvc); this data has left or interval "
            "censored items."
        )
    if np.isfinite(data.tr).any():
        raise ValueError(
            "Time-varying covariates are fitted without right truncation "
            "(fit_tvc); this data has right truncated items."
        )
    ids = data.ids if data.ids is not None else list(range(len(data.x)))
    accepted = inspect.signature(fit_tvc).parameters
    if not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()
    ):
        unknown = sorted(set(options) - set(accepted))
        if unknown:
            takes = [
                name
                for name in accepted
                if name not in ("i", "xl", "xr", "c", "Z", "n")
            ]
            raise TypeError(
                "Data with time-varying covariates is fitted by the model's "
                "fit_tvc, which takes no "
                + ", ".join(repr(u) for u in unknown)
                + "; it takes "
                + (", ".join(repr(t) for t in takes) or "no options")
                + "."
            )

    def fitted(steps: int) -> tuple[Any, int]:
        rows = _tvc_rows(fit_tvc, data, ids, steps)
        model = fit_tvc(
            np.array(rows["i"]),
            np.array(rows["xl"], dtype=float),
            np.array(rows["xr"], dtype=float),
            np.array(rows["c"]),
            np.array(rows["Z"], dtype=float),
            np.array(rows["n"]),
            **options,
        )
        return model, len(rows["i"])

    schedules = data.time_varying_covariates or {}
    continuous = any(
        isinstance(schedule, CovariatePath) for schedule in schedules.values()
    )
    model, count = fitted(_PATH_STEPS)
    if not continuous:
        return model
    return _refined_on_paths(fitted, model, count)


def _refined_on_paths(fitted: Callable, model: Any, count: int) -> Any:
    """The fit on continuous paths, refined: ``model`` is the fit with
    ``_PATH_STEPS`` steps a stretch (``count`` rows), and ``fitted(steps)``
    fits with another number. The steps are halved until the estimates
    move by no more than ``_PATH_TOLERANCE`` of their standard errors
    (where a standard error is missing, of ``1e-2 max(|estimate|, 1)``),
    and the model returned holds ``path_steps``: the steps a stretch and
    the rows of its fit, the largest move of an estimate at the last
    halving (in standard errors), the tolerance and whether it was met."""
    import warnings

    from surpyval.utils.warnings import caller_stacklevel

    steps, change = _PATH_STEPS, 0.0
    converged = False
    for _ in range(_PATH_ROUNDS):
        finer, finer_count = fitted(2 * steps)
        if finer_count == count:
            # every path is flat or made of steps: nothing was cut finer
            converged = True
            break
        change = _largest_move(model, finer)
        model, count, steps = finer, finer_count, 2 * steps
        if change <= _PATH_TOLERANCE:
            converged = True
            break
        if 2 * count > _PATH_ROWS:
            break
    report = {
        "steps": steps,
        "rows": count,
        "change": float(change),
        "tolerance": _PATH_TOLERANCE,
        "converged": converged,
    }
    try:
        model.path_steps = report
    except AttributeError:
        pass
    if not converged:
        warnings.warn(
            "The fit on continuous covariate paths still moved when its "
            f"steps were halved to {steps} a stretch ({count:,} rows): the "
            f"largest move of an estimate was {change:.2g} of its standard "
            f"error, above the tolerance of {_PATH_TOLERANCE:g}. The last "
            "fit is returned. List the path's kinks and jumps as "
            "breakpoints, or give the items' covariates as StepSchedules "
            "(CovariatePath.to_steps(t_max, steps)) of your own fineness.",
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )
    return model


def _largest_move(coarse: Any, fine: Any) -> float:
    """The largest change of an estimate between two fits, in the finer
    fit's standard errors."""
    old = np.atleast_1d(np.asarray(coarse.params, dtype=float)).ravel()
    new = np.atleast_1d(np.asarray(fine.params, dtype=float)).ravel()
    if old.shape != new.shape:
        return np.inf
    scale = 1e-2 * np.maximum(np.abs(new), 1.0)
    try:
        with ignoring_warnings():
            se = np.asarray(fine.standard_errors(), dtype=float).ravel()
        if se.shape == new.shape:
            scale = np.where(np.isfinite(se) & (se > 0), se, scale)
    except Exception:
        pass
    both = np.isfinite(old) & np.isfinite(new)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(new - old)[both] / scale[both]))


def _tvc_rows(
    fit_tvc: Callable, data: SurpyvalData, ids: Any, steps: int
) -> dict[str, list]:
    """The ``(start, stop]`` rows of every item for ``fit_tvc``, a
    continuous path cut into ``steps`` steps a stretch."""
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.utils.item_history import late_covariates

    static = None
    if data.Z is not None:
        static = np.asarray(data.Z, dtype=float)
        static = static[:, None] if static.ndim == 1 else static
    x = data.x if data.x.ndim == 1 else data.x[:, 0]
    fitter = getattr(fit_tvc, "__self__", None)
    support = getattr(fitter, "support", (0.0, np.inf))
    real_line = float(support[0]) < 0
    rows: dict[str, list] = {k: [] for k in ("i", "xl", "xr", "c", "Z", "n")}
    for k, item in enumerate(ids):
        schedule = data.time_varying_covariates[item]  # type: ignore[index]
        path = schedule if isinstance(schedule, CovariatePath) else None
        if np.isfinite(data.tl[k]):
            start = float(data.tl[k])
        elif real_line:
            # Not left truncated: followed since -inf, as the ordinary fit
            # reads it, on a baseline defined below 0 (#952).
            if path is not None:
                raise ValueError(
                    f"Item {item!r} has no left_truncation, so on this "
                    "model's baseline, defined below 0, it is followed "
                    "since -inf, and a continuous CovariatePath is cut "
                    "into steps over a finite stretch only. Give the age "
                    "it came under observation as left_truncation (0 if "
                    "at 0)."
                )
            if schedule.start > -np.inf:
                raise ValueError(
                    f"Item {item!r} has no left_truncation, so on this "
                    "model's baseline, defined below 0, it is followed "
                    "since -inf, but its time-varying covariates start at "
                    f"{schedule.start:g}. Give their value since -inf "
                    "(StepSchedule.from_changepoints([-np.inf, ...], ...)), "
                    "or left_truncation=0 if it came under observation at "
                    "0."
                )
            start = -np.inf
        else:
            start = 0.0
        stop = float(x[k])
        if schedule.start > start:
            # Nothing says what the covariates were before (#930).
            raise late_covariates(item, schedule.start, start)
        if path is not None:
            schedule = path.to_steps(
                stop if stop > start else start + 1.0, steps, start=start
            )
        starts, ends, values = schedule.segments(
            max(stop, schedule.start + 1e-12)
        )
        keep = (ends > start) & (starts < stop)
        starts = np.maximum(starts[keep], start)
        ends = np.minimum(ends[keep], stop)
        values = values[keep]
        if path is not None and int(data.c[k]) == 0 and len(starts):
            # A failure's hazard is that of the covariate at the failure,
            # not at the middle of the step it fell in (an error that
            # falls only with the step's width, not its square): the last
            # millionth of that step holds the path's value at its end.
            at_end = path(stop)[0]
            if not np.array_equal(values[-1], at_end):
                cut = ends[-1] - _PATH_END * (ends[-1] - starts[-1])
                if starts[-1] < cut < ends[-1]:
                    starts = np.append(starts, cut)
                    ends = np.append(ends[:-1], [cut, ends[-1]])
                    values = np.vstack([values, at_end])
        last = len(starts) - 1
        for j, (a, b, z) in enumerate(zip(starts, ends, values)):
            rows["i"].append(item)
            rows["xl"].append(a)
            rows["xr"].append(b)
            rows["c"].append(int(data.c[k]) if j == last else 1)
            rows["Z"].append(z if static is None else np.r_[static[k], z])
            rows["n"].append(data.n[k])
    return rows


def named_fit(fit: Callable, names: list | None) -> Any:
    """``fit()``, with ``names`` (where given) naming the columns of its
    covariates, as a DataFrame's columns do: the coefficients take them,
    and the model keeps them as its ``feature_names``."""
    if names is None:
        return fit()
    from surpyval.univariate.regression._aliasing import covariate_columns

    with covariate_columns(list(names)):
        model = fit()
    if getattr(model, "feature_names", None) is None:
        try:
            model.feature_names = list(names)
        except AttributeError:
            pass
    return model


def _tvc_names(data: SurpyvalData) -> list | None:
    """The names of the columns ``fit_tvc`` is given: the named fixed
    covariates, then ``time_varying_0``, ``time_varying_1``, ... for the
    time-varying ones."""
    schedules = data.time_varying_covariates
    if data.covariate_names is None or data.Z is None or not schedules:
        return None
    width = next(iter(schedules.values())).p
    return list(data.covariate_names) + [
        f"time_varying_{j}" for j in range(width)
    ]


def _checked_names(names: Any, Z: Any) -> list:
    """``names`` for the columns of ``Z``: text, one per column, with no
    name twice."""
    if Z is None:
        raise ValueError("covariate_names were given without covariates, Z.")
    names = [names] if isinstance(names, str) else list(names)
    width = 1 if np.ndim(Z) == 1 else np.shape(Z)[1]
    if len(names) != width:
        raise ValueError(
            f"{len(names)} covariate_names for {width} covariate "
            f"column(s)."
        )
    bad = [name for name in names if not isinstance(name, str) or not name]
    if bad:
        raise ValueError(f"covariate_names must be text; got {bad!r}.")
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"covariate_names has {twice!r} more than once.")
    return names


def _readings_of(markers: dict | None, item: Any) -> dict | None:
    """An item's marker readings in ``markers`` (None where it has none)."""
    return None if markers is None else markers.get(item)


def _named_row(Z: np.ndarray | None, k: int, names: list | None) -> Any:
    """Row ``k`` of ``Z`` as a record's covariates: ``{name: value}``
    where the columns are named."""
    if Z is None:
        return None
    return Z[k] if names is None else dict(zip(names, Z[k].tolist()))


def _plain(value: Any) -> Any:
    """A numpy scalar as the plain Python value JSON stores."""
    return value.item() if isinstance(value, np.generic) else value


def _attribute_names(attributes: list | None) -> list:
    """The names of the attributes any record of a dataset has."""
    if not attributes:
        return []
    return list(dict.fromkeys(k for a in attributes if a for k in a))


def _attributes_json(attributes: list) -> list:
    """A dataset's attributes as JSON keeps them (tuples as
    ``{"tuple": [...]}``)."""
    from surpyval.utils.data_json import label

    return [
        None if a is None else {k: label(v) for k, v in a.items()}
        for a in attributes
    ]


def _attributes_back(attributes: list) -> list:
    from surpyval.utils.data_json import label_back

    return [
        None if a is None else {k: label_back(v) for k, v in a.items()}
        for a in attributes
    ]
