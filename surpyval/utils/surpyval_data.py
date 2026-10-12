from __future__ import annotations

import functools
import inspect
import json
from numbers import Number
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike

import surpyval
from surpyval.serialisation import decode_non_finite, encode_non_finite
from surpyval.utils import xcnt_handler
from surpyval.utils.numeric import unique_pairs


class SurpyvalData:
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
            off, so the rows stay aligned with x.
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

        if Z is not None:
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
        Z_arr: np.ndarray = np.asarray(Z)
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
    ) -> "SurpyvalData":
        """
        The data from a list of each kind of observation: the times at
        which a failure was observed, the times right censored at (the
        suspensions), the times left censored at, and the intervals a
        failure is known to lie in. Any of them may be left out, but not
        all; a time repeated is counted, and a single time may be given as
        a number.

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

        xs, cs, tls, trs = [], [], [], []
        for kind, flag in _OBSERVATION_KINDS:
            times = _observation_times(kind, lists[kind])
            bounds = {}
            for side, none in (("left", -np.inf), ("right", np.inf)):
                name = f"{kind}_{side}_truncation"
                bound = own[name]
                if bound is None:
                    bound = shared[side]
                elif lists[kind] is None:
                    raise ValueError(
                        f"'{name}' was given without '{kind}': it bounds "
                        f"each of the {kind.replace('_', ' ')} times."
                    )
                elif shared[side] is not None:
                    raise ValueError(
                        f"Both '{side}_truncation' (shared by every "
                        f"observation) and '{name}' were given: give one."
                    )
                bounds[side] = _truncation_bounds(
                    name, bound, len(times), none
                )
            xs.append(times if flag == 2 else np.column_stack([times, times]))
            cs.append(np.full(len(times), flag))
            tls.append(bounds["left"])
            trs.append(bounds["right"])
        x = np.concatenate(xs)
        if len(x) == 0:
            raise ValueError(
                "The lists hold no observations: give at least one time."
            )
        return cls(
            x,
            np.concatenate(cs),
            tl=np.concatenate(tls),
            tr=np.concatenate(trs),
        )

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

    def to_xcnt(self) -> tuple:
        """
        The data in the ``xcnt`` format, ``(x, c, n, t)``: the times (a
        ``[lower, upper]`` row where any observation is an interval), the
        censoring flags (0 observed, 1 right, -1 left, 2 interval), the
        counts and the ``[left, right]`` truncation bounds, as copies.

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
        The data as a list of each kind of observation, the form
        :meth:`from_observations` reads: ``"observed"``,
        ``"right_censored"`` and ``"left_censored"`` times and
        ``"interval_censored"`` ``[lower, upper]`` pairs, each repeated by
        its count. Truncated data adds each kind's bounds, one per item
        (``"observed_left_truncation"`` and so on, ``-inf`` or ``inf``
        where an item has none), for each side on which anything is
        truncated; so ``from_observations(**data.to_observations())`` is
        the same data.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData([1, 3, 5], c=[0, 0, 1], n=[1, 2, 1])
        >>> data.to_observations()["observed"]
        array([1., 3., 3.])
        >>> data = SurpyvalData(
        ...     [600, 700, 1000], c=[0, 0, 1], tl=[500, 0, 500]
        ... )
        >>> data.to_observations()["observed_left_truncation"]
        array([500.,   0.])
        """
        counts = self.n.astype(int)
        x = self.x if self.x.ndim == 2 else np.column_stack([self.x] * 2)
        truncated = [
            (side, column)
            for column, side in enumerate(("left", "right"))
            if np.isfinite(self.t[:, column]).any()
        ]
        lists: dict[str, np.ndarray] = {}
        bounds: dict[str, np.ndarray] = {}
        for kind, flag in _OBSERVATION_KINDS:
            mask = self.c == flag
            values = x[mask] if flag == 2 else x[mask, 0]
            lists[kind] = np.repeat(values, counts[mask], axis=0).astype(float)
            for side, column in truncated:
                bounds[f"{kind}_{side}_truncation"] = np.repeat(
                    self.t[mask, column], counts[mask]
                ).astype(float)
        return {**lists, **bounds}

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
        return self._xrd_cache[estimator]

    def _split_by_mask(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.x[mask] if self.x.ndim == 1 else self.x[mask, 0]
        return x, self.n[mask]

    def __getitem__(self, index: Any) -> "SurpyvalData":
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
        data = {
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "t": self.t.tolist(),
        }

        if self.Z is not None:
            data["Z"] = self.Z.tolist()

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

        return cls(x, c, n, t, Z=Z, handle=False)


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
    if interval and (arr[:, 0] >= arr[:, 1]).any():
        raise ValueError(
            "Each interval in 'interval_censored' needs lower < upper."
        )
    return arr


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
    ``Z`` where the data carries covariates) must be left out; the fit
    then runs on the data's own arrays, as from the arrays themselves.
    A data's censoring, counts or truncation that the fit has no argument
    for is refused, not dropped."""
    signature = inspect.signature(method)
    names = set(signature.parameters)
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
            for name in _DATA_ARGUMENTS + ("Z",)
            if bound.arguments.get(name) is not None
            and not (name == "Z" and data.Z is None)
        ]
        if given:
            raise TypeError(
                "x is a SurpyvalData, which holds the data: leave out "
                + ", ".join(repr(g) for g in given)
                + " (or pass arrays rather than a SurpyvalData)."
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
        if "t" in names:
            filled["t"] = data.t
        elif "tl" in names and "tr" in names:
            filled["tl"], filled["tr"] = data.tl, data.tr
        elif np.isfinite(data.t).any():
            lost.append("truncation")
        if data.Z is not None:
            if "Z" in names:
                filled["Z"] = data.Z
            else:
                lost.append("covariates")
        if lost:
            raise ValueError(
                f"This fit takes no {' or '.join(lost)}, which the "
                "SurpyvalData holds."
            )
        bound.arguments.update(filled)
        return method(*bound.args, **bound.kwargs)

    setattr(wrapper, "_accepts_surpyval_data", True)
    return wrapper  # type: ignore[return-value]
