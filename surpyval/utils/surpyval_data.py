from __future__ import annotations

import functools
import inspect
import json
import warnings
from numbers import Number
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike

import surpyval
from surpyval.serialisation import decode_non_finite, encode_non_finite
from surpyval.utils import xcnt_handler
from surpyval.utils.data_reshape import LIFETIME_PARTS, ReshapeMixin
from surpyval.utils.item_history import missing_type
from surpyval.utils.numeric import unique_pairs
from surpyval.utils.warnings import (
    MaskedCauseWarning,
    named,
    warn_masked,
    warn_suspension_types,
)


class SurpyvalData(ReshapeMixin):
    """
    Validated univariate survival data in the ``xcnt`` format, split by
    type of observation for the likelihoods.

    Every univariate fitter builds one from its arrays (with
    :func:`~surpyval.utils.xcnt_handler`); parametric fitters also accept
    one directly through ``fit_from_surpyval_data``. Besides ``x``, ``c``,
    ``n`` and ``t`` (and ``tl``/``tr``, the columns of ``t``), it holds
    the observed (``x_o``, ``n_o``), right censored (``x_r``, ``n_r``),
    left censored (``x_l``, ``n_l``) and interval (``x_il``, ``x_ir``,
    ``n_i``) rows separately. A censored row with a finite truncation
    bound on its censored side is stored as an interval ending at that
    bound, so the likelihood uses the probability of the part of the
    window it could have fallen in.

    Examples
    --------
    >>> from surpyval import SurpyvalData
    >>> inf = float("inf")
    >>> data = SurpyvalData(x=[1, 2, 3, 4], c=[0, 1, 0, 1], tr=[9, 3, 9, inf])
    >>> data.x_o, data.x_r
    (array([1., 3.]), array([4.]))
    >>> data.x_il, data.x_ir
    (array([2.]), array([3.]))
    """

    # Attached by ``from_items``: each row's item id, the event type of
    # each failure (None for a suspension), and each item's time-varying
    # covariates (``{item: StepSchedule or CovariatePath}``).
    ids: "list | None" = None
    e: "np.ndarray | None" = None
    time_varying_covariates: "dict | None" = None
    # The name of each column of Z, where they have names: given as
    # ``covariate_names``, the columns of a DataFrame ``Z``, or the items'
    # ``covariates={name: value}``. They name a fitted model's
    # coefficients.
    covariate_names: "list | None" = None

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
        covariate_names: "list | None" = None,
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
            * 0 = uncensored
            * 1 = right censored
            * -1 = left censored
            * 2 = interval censored
        n : array-like, optional
            Number of occurrences for each value in x (positive whole
            numbers).
        t : array-like, optional
            2D array of truncation bounds [left, right] for each value in x.
        xl : array-like, optional
            Left interval bounds for interval censored data.
            Cannot be used with 'x'. Must be paired with 'xr'.
        xr : array-like, optional
            Right interval bounds for interval censored data.
            Cannot be used with 'x'. Must be paired with 'xl'.
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
            ``group_and_sort`` off, as ``Z`` does.
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

        if Z is not None or e is not None:
            group_and_sort = False

        if handle:
            x, c, n, t = xcnt_handler(
                x, c, n, t, xl, xr, tl, tr, group_and_sort=group_and_sort
            )

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
        if e is not None:
            self.e = _event_type_array(e, self.c)

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
            _checked_names([str(c) for c in Z.columns], np.asarray(Z))
            if type(Z).__name__ == "DataFrame"
            else None
        )
        # A copy: the caller's array stays theirs (#927)
        Z_arr: np.ndarray = np.array(Z)
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
            self.n_i = self.n[self.mask_i]
        else:
            self.x_il = np.array([], dtype=float)
            self.x_ir = np.array([], dtype=float)
            self.n_i = np.array([], dtype=int)

        self.truncated_mask = np.isfinite(self.t).any(axis=1)
        self.x_tl, self.x_tr, self.n_t = (
            self.t[self.truncated_mask, 0],
            self.t[self.truncated_mask, 1],
            self.n[self.truncated_mask],
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
        covariate_names: "list | None" = None,
        observed_ids: Any = None,
        right_censored_ids: Any = None,
        left_censored_ids: Any = None,
        interval_censored_ids: Any = None,
    ) -> "SurpyvalData":
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
        shared = {"left": left_truncation, "right": right_truncation}
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
        )
        if item_ids is not None:
            data.ids = item_ids
        return data

    @classmethod
    def from_xrd(
        cls, x: ArrayLike, r: ArrayLike, d: ArrayLike
    ) -> "SurpyvalData":
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
    def from_items(cls, items: Any, counts: Any = None) -> "SurpyvalData":
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

        items = list(items)
        if not items:
            raise ValueError("Give at least one ItemLifetime.")
        for item in items:
            if not isinstance(item, ItemLifetime):
                raise TypeError(
                    "from_items takes ItemLifetime objects; got "
                    f"{type(item).__name__}."
                )
        from surpyval.utils.item_history import (
            check_item_ids,
            covariate_names_of,
            covariate_row_in,
        )

        if isinstance(counts, dict):
            from surpyval.utils.item_history import item_id

            given = {item_id(k): v for k, v in counts.items()}
            unknown = sorted(
                (repr(k) for k in set(given) - {i.id for i in items}), key=str
            )
            if unknown:
                raise ValueError(
                    "'counts' names items not given: "
                    + ", ".join(unknown)
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
        with warnings.catch_warnings():
            # Warned of above, by item
            warnings.simplefilter("ignore", MaskedCauseWarning)
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
            )
        data.ids = ids
        if varying:
            data.time_varying_covariates = {
                i.id: i.time_varying_covariates for i in items
            }
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
        prepared = self._prepared()
        return [self._item_at(k, prepared) for k in range(len(self.x))]

    _PARTS = LIFETIME_PARTS

    def _records(self) -> tuple[list, list]:
        return self.to_items(), list(np.asarray(self.n))

    def _from_records(self, records: list, counts: list) -> "SurpyvalData":
        out = SurpyvalData.from_items(records, counts=counts)
        if self.ids is None:
            out.ids = None  # its items were row numbers, not ids
        return out

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
        )

    def summary(self) -> str:
        """
        A short description of the data: its rows, items and outcomes, and
        what else it holds (truncation, event types, covariates,
        time-varying covariates).

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> print(SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1]).summary())
        SurpyvalData: 3 rows, 4 items
          outcomes: 3 observed, 1 right censored
        """
        from surpyval.utils.event_history_data import _held

        c, n = np.asarray(self.c), np.asarray(self.n)
        names = (
            (0, "observed"),
            (1, "right censored"),
            (-1, "left censored"),
            (2, "interval censored"),
        )
        head = (
            f"SurpyvalData: {_plural(len(self.x), 'row')}, "
            f"{_plural(int(n.sum()), 'item')}"
        )
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
        lines += _held(self, rows=len(self.x))
        return "\n".join(lines)

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
        extra: dict[str, np.ndarray] = {}
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
        # Cache per estimator: the first call used to be returned for
        # every later call regardless of a different ``estimator``
        # argument (#282).
        if not hasattr(self, "_xrd_cache"):
            self._xrd_cache: dict = {}
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

    def _split_by_mask(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.x[mask] if self.x.ndim == 1 else self.x[mask, 0]
        return x, self.n[mask]

    def __getitem__(self, index: Any) -> "SurpyvalData":
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
        if self.ids is not None:
            out.ids = [self.ids[r] for r in rows]
        if self.time_varying_covariates is not None:
            kept = out.ids if out.ids is not None else []
            out.time_varying_covariates = {
                i: self.time_varying_covariates[i] for i in kept
            }
        return out

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self) -> Any:
        return zip(self.x, self.c, self.n, self.t)

    def __repr__(self) -> str:
        return (
            f"SurpyvalData(\n"
            f"    x={self.x!r},\n"
            f"    c={self.c!r},\n"
            f"    n={self.n!r},\n"
            f"    t={self.t!r}\n"
            f")"
        )

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """
        Serialize SurpyvalData to JSON format.

        The JSON is strict: the infinite truncation bounds (and any other
        non-finite value) are written as ``null`` and recorded under
        ``"non_finite"``, which :meth:`from_json` reads back (see
        :doc:`/surpyval.serialisation`).

        Parameters
        ----------
        filepath : str | Path, optional
            If provided, saves JSON to this file path

        Returns
        -------
        str | None
            JSON string if no filepath provided, None if saved to file
        """
        if self.time_varying_covariates is not None:
            raise ValueError(
                "Data with time-varying covariates has no JSON form (a "
                "CovariatePath is a function): keep it with pickle, or keep "
                "its items."
            )
        data: dict[str, Any] = {
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "t": self.t.tolist(),
        }

        if self.Z is not None:
            data["Z"] = self.Z.tolist()
        if self.covariate_names is not None:
            data["covariate_names"] = list(self.covariate_names)
        if self.e is not None:
            data["e"] = [_json_label(v) for v in self.e]
        if self.ids is not None:
            data["ids"] = [_json_label(v) for v in self.ids]

        # Strict JSON, as for the models: the -inf/inf truncation bounds
        # (and any other non-finite value) are written as null with a
        # "non_finite" record of what each stood for (see
        # surpyval.serialisation.encode_non_finite), not as the
        # ``Infinity`` literals strict parsers reject.
        encode_non_finite(data)
        if filepath:
            Path(filepath).write_text(
                json.dumps(data, indent=2, allow_nan=False)
            )
            return None

        return json.dumps(data, allow_nan=False)

    @classmethod
    def from_json(cls, source: str | Path) -> "SurpyvalData":
        """
        Create SurpyvalData instance from JSON text or a file.

        Parameters
        ----------
        source : str | Path
            Pass a ``pathlib.Path`` to load from a file; a ``str`` is always
            parsed as JSON text, so ``from_json("data.json")`` fails even
            though ``to_json("data.json")`` writes that file. The data is
            not re-validated.

        Returns
        -------
        SurpyvalData
            New instance created from JSON data

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData(x=[1, 2, 2, 5], c=[0, 0, 0, 1])
        >>> restored = SurpyvalData.from_json(data.to_json())
        >>> restored.x, restored.n
        (array([1., 2., 5.]), array([1, 2, 1]))
        """
        if isinstance(source, Path):
            text = source.read_text()
        else:
            text = source

        # Files written before the strict-JSON convention hold the
        # ``Infinity`` literals, which json.loads still reads.
        data = decode_non_finite(json.loads(text))

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
        )
        if "e" in data:
            out.e = _event_type_array(
                [_label_from_json(v) for v in data["e"]], out.c, warn=False
            )
        if "ids" in data:
            out.ids = [_label_from_json(v) for v in data["ids"]]
        return out


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
_DATA_ARGUMENTS = ("c", "n", "t", "tl", "tr", "xl", "xr")

F = TypeVar("F", bound=Callable[..., Any])


def accepts_surpyval_data(method: F) -> F:
    """Let a fitter's ``fit`` take a :class:`SurpyvalData` as its ``x``:
    ``Weibull.fit(data)``. The data is the whole of it, so its other data
    arguments (``c``, ``n``, ``t``, ``tl``, ``tr``, ``xl``, ``xr``, and
    ``Z`` or ``e`` where the data carries covariates or event types) must
    be left out; the fit then runs on the data's own arrays, as from the
    arrays themselves. Data with step time-varying covariates (from
    ``from_items``) is fitted by the model's ``fit_tvc``. A data's
    censoring, counts, truncation, covariates or event types that the fit
    has no argument for are refused, not dropped."""
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
        given = [
            name
            for name in _DATA_ARGUMENTS + ("Z", "e")
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
        if data.time_varying_covariates is not None:
            if data.e is not None and "e" not in names:
                raise ValueError(
                    "This fit takes no event types, which the SurpyvalData "
                    "holds."
                )
            owner = bound.arguments.get(params[0]) if params else None
            fit_tvc = getattr(owner, "fit_tvc", None)
            if params[0] in ("self", "cls") and callable(fit_tvc):
                return named_fit(
                    lambda: _fit_tvc_from_data(
                        fit_tvc, data, _options(bound, signature)
                    ),
                    _tvc_names(data),
                )
        for name, held, how in (
            (
                "Z",
                data.Z,
                "SurpyvalData(x, c, ..., Z=Z), or "
                "from_observations(..., observed_covariates=..., ...)",
            ),
            (
                "e",
                data.e,
                "SurpyvalData(x, c, ..., e=e), or "
                "from_observations(..., observed_event_types=..., ...)",
            ),
        ):
            if (
                held is None
                and name in names
                and bound.arguments.get(name) is None
                and signature.parameters[name].default
                is inspect.Parameter.empty
            ):
                what = "covariates" if name == "Z" else "event types"
                raise TypeError(
                    f"This fit needs {what}, {name}, and the SurpyvalData "
                    f"holds none: build it with {how}."
                )
        filled = {"x": data.x}
        lost = []
        if "c" in names:
            filled["c"] = data.c
        elif np.any(data.c != 0):
            lost.append("censoring")
        if "n" in names:
            filled["n"] = data.n
        elif np.any(data.n != 1):
            lost.append("counts")
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
        if lost:
            raise ValueError(
                f"This fit takes no {' or '.join(lost)}, which the "
                "SurpyvalData holds."
            )
        bound.arguments.update(filled)
        return named_fit(
            lambda: method(*bound.args, **bound.kwargs),
            data.covariate_names if data.Z is not None else None,
        )

    setattr(wrapper, "_accepts_surpyval_data", True)
    return wrapper  # type: ignore[return-value]


def _not_taken_yet(data: SurpyvalData) -> list:
    """What a SurpyvalData from ``from_items`` holds that ``fit(data)``
    does not take yet: fitting without it would be another model."""
    out = []
    if data.e is not None:
        out.append("event types")
    if data.time_varying_covariates is not None:
        out.append("time-varying covariates (fit them with fit_tvc for now)")
    return out


def refuses_what_fit_cannot_take(method: F) -> F:
    """Refuse, in a ``fit_from_surpyval_data``, data that holds event
    types or time-varying covariates, which it would otherwise ignore."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data = kwargs.get("surv_data", args[1] if len(args) > 1 else None)
        if isinstance(data, SurpyvalData):
            missing = _not_taken_yet(data)
            if missing:
                raise ValueError(
                    f"This fit takes no {' or '.join(missing)}, which the "
                    "SurpyvalData holds."
                )
        return method(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


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
    out[:] = [None if missing_type(v) else v for v in out]
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


def _kind_ids(name: str, value: Any, k: int) -> list:
    """A kind's item ids, ``name``: one per entry of its list."""
    listed = [value] if isinstance(value, (str, bytes)) else list(value)
    if len(listed) != k:
        raise ValueError(
            f"'{name}' must hold one id per entry ({k}); got {len(listed)}."
        )
    return listed


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
    out[:] = [None if missing_type(v) else v for v in out]
    return out


# Arguments a fit takes its data in, or names the fitter itself.
_NOT_OPTIONS = frozenset(_DATA_ARGUMENTS + ("x", "Z", "e", "self", "cls"))


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


def _fit_tvc_from_data(
    fit_tvc: Callable, data: SurpyvalData, options: dict
) -> Any:
    """Fit ``data`` with step time-varying covariates by the model's
    ``fit_tvc``: each item's life, from the start of its observation
    (``left_truncation``, else 0) to its outcome, split where its schedule
    steps into ``(start, stop]`` rows, the covariates of each row being the
    item's static ones (if any) followed by its time-varying ones in force
    there."""
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.utils.item_history import late_covariates

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
    static = None
    if data.Z is not None:
        static = np.asarray(data.Z, dtype=float)
        static = static[:, None] if static.ndim == 1 else static
    x = data.x if data.x.ndim == 1 else data.x[:, 0]
    rows: dict[str, list] = {k: [] for k in ("i", "xl", "xr", "c", "Z", "n")}
    for k, item in enumerate(ids):
        schedule = data.time_varying_covariates[item]  # type: ignore[index]
        if isinstance(schedule, CovariatePath):
            raise ValueError(
                f"Item {item!r}'s time-varying covariates are a continuous "
                "CovariatePath; fitting on continuous paths is not provided "
                "yet (#908). Give the path as a StepSchedule to fit it."
            )
        start = data.tl[k] if np.isfinite(data.tl[k]) else 0.0
        stop = float(x[k])
        if schedule.start > start:
            # Nothing says what the covariates were before (#930).
            raise late_covariates(item, schedule.start, start)
        starts, ends, values = schedule.segments(
            max(stop, schedule.start + 1e-12)
        )
        keep = (ends > start) & (starts < stop)
        starts = np.maximum(starts[keep], start)
        ends = np.minimum(ends[keep], stop)
        values = values[keep]
        last = len(starts) - 1
        for j, (a, b, z) in enumerate(zip(starts, ends, values)):
            rows["i"].append(item)
            rows["xl"].append(a)
            rows["xr"].append(b)
            rows["c"].append(int(data.c[k]) if j == last else 1)
            rows["Z"].append(z if static is None else np.r_[static[k], z])
            rows["n"].append(data.n[k])
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
    return fit_tvc(
        np.array(rows["i"]),
        np.array(rows["xl"], dtype=float),
        np.array(rows["xr"], dtype=float),
        np.array(rows["c"]),
        np.array(rows["Z"], dtype=float),
        np.array(rows["n"]),
        **options,
    )


def named_fit(fit: Callable, names: "list | None") -> Any:
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


def _tvc_names(data: SurpyvalData) -> "list | None":
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


def _named_row(Z: "np.ndarray | None", k: int, names: "list | None") -> Any:
    """Row ``k`` of ``Z`` as a record's covariates: ``{name: value}``
    where the columns are named."""
    if Z is None:
        return None
    return Z[k] if names is None else dict(zip(names, Z[k].tolist()))


def _json_label(value: Any) -> Any:
    """An id or event type as JSON keeps it: a tuple (a gap's id,
    ``("A", 2)``) as ``{"tuple": [...]}``, which JSON's lists would
    otherwise turn into a list (#928)."""
    if isinstance(value, tuple):
        return {"tuple": [_json_label(v) for v in value]}
    return _plain(value)


def _label_from_json(value: Any) -> Any:
    """An id or event type read back from JSON (:func:`_json_label`)."""
    if isinstance(value, dict) and set(value) == {"tuple"}:
        return tuple(_label_from_json(v) for v in value["tuple"])
    return value


def _plain(value: Any) -> Any:
    """A numpy scalar as the plain Python value JSON stores."""
    return value.item() if isinstance(value, np.generic) else value


def _plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")
