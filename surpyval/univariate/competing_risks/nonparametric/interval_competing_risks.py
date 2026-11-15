"""The non-parametric estimate of the cumulative incidence functions of
interval-censored competing-risks data (Hudgens, Satten and Longini 2001).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval import plotting
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import label_from_native
from surpyval.univariate.competing_risks.nonparametric import _ic_npmle
from surpyval.univariate.competing_risks.nonparametric._ic_data import (
    read_rows,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import axis_label, takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

_CB_ON = ("cif", "sf", "ff", "R", "F")


class IntervalCompetingRisks(SerialisableMixin):
    r"""
    Non-parametric competing-risks estimate for interval-censored data: the
    non-parametric maximum-likelihood estimate (NPMLE) of each cause's
    cumulative incidence function (CIF).

    When failures are found at inspections, each unit is known to have
    failed from its cause somewhere between its last good inspection and
    the one that found it failed. ``CompetingRisks`` (the Aalen-Johansen
    estimate) needs each failure's time; this estimate takes any mix of
    exact, right, left and interval-censored rows, and failures whose
    cause was not recorded (``masked="include"``). For exact and right
    censored data it is the Aalen-Johansen estimate; for one cause it is
    Turnbull's estimate.

    The cumulative incidence of cause :math:`k` is the probability of
    failing from that cause by time :math:`t` while the others still act.
    The estimate puts each cause's probability mass on a set of intervals
    (the innermost intervals of the rows that can hold that cause) and
    chooses the masses that make the data most likely:

    .. math::
        L = \prod_i \Big(\sum_{(k, j) \subseteq \text{row } i}
        p_{kj}\Big)^{n_i},

    a row that failed from cause :math:`k` in :math:`(x_l, x_r]` covering
    cause :math:`k`'s intervals inside it, a row with no recorded cause
    every cause's, and a right-censored row every cause's after
    :math:`x_l` plus the mass of the units still working at the end of
    follow-up (``beyond``). Where in its interval a cause's mass lies is
    not known: the CIF is held at its value at the interval's start and
    rises at its end, as ``Turnbull``'s curve does. See Notes.

    Call ``IntervalCompetingRisks.fit`` (or ``fit_from_df``); it returns a
    fitted instance.

    Notes
    -----
    The method is Hudgens, Satten and Longini's (2001) NPMLE of the
    sub-distribution functions, the competing-risks form of Turnbull's
    (1976) estimate; Groeneboom, Maathuis and Wellner (2008) study the
    same maximum likelihood estimate for current-status data. Cause
    :math:`k`'s mass lies on the innermost intervals (Turnbull's maximal
    intersections) of the windows of the rows that can hold it: those that
    failed from cause :math:`k`, those whose cause was not recorded, and the
    right-censored rows. An innermost interval that reaches to infinity is
    covered only by right-censored rows, so its mass, the units not failed
    by the end of follow-up, is not given to any cause (``beyond``); each
    CIF then stops below its limit, as the Aalen-Johansen CIFs stop at
    :math:`1 - S` at the last time.

    The log-likelihood is concave in the masses, so its maximum is unique,
    but the masses at it need not be: with causes not recorded, how the
    mass of an interval two causes share splits between them may not be
    determined by the data. The masses are found by the EM
    (self-consistency) algorithm, accelerated by SQUAREM (Varadhan and
    Roland 2008), and the search stops when the Karush-Kuhn-Tucker
    conditions hold to ``tol``: then the log-likelihood is within
    ``N log(1 + tol)`` of its maximum (``gap`` is the final value).

    The bounds (:meth:`cb`) are by the case bootstrap: the units are
    resampled and the estimate refitted. The NPMLE at a point converges
    more slowly than :math:`\sqrt{n}`, and for current-status data the
    plain bootstrap of it is known not to be consistent (Sen, Banerjee
    and Woodroofe 2010), so the bounds are approximate there.

    References
    ----------
    Hudgens, M. G., Satten, G. A. and Longini, I. M. (2001),
    "Nonparametric maximum likelihood estimation for competing risks
    survival data subject to interval censoring and truncation",
    Biometrics 57, 74-80.

    Groeneboom, P., Maathuis, M. H. and Wellner, J. A. (2008), "Current
    status data with competing risks: consistency and rates of
    convergence of the MLE", Annals of Statistics 36, 1031-1063.

    Turnbull, B. W. (1976), "The empirical distribution function with
    arbitrarily grouped, censored and truncated data", Journal of the
    Royal Statistical Society B 38, 290-295.

    Varadhan, R. and Roland, C. (2008), "Simple and globally convergent
    methods for accelerating the convergence of any EM algorithm",
    Scandinavian Journal of Statistics 35, 335-353.

    Sen, B., Banerjee, M. and Woodroofe, M. (2010), "Inconsistency of
    bootstrap: the Grenander estimator", Annals of Statistics 38,
    1953-1977.
    """

    #: The source DataFrame when fitted via ``fit_from_df``.
    source_df: Any
    event_idx_map: dict
    n_event_types: int
    #: Each atom's cause index, lower and upper time and mass: cause
    #: ``k``'s CIF rises by ``mass`` at ``upper``. The data say only that
    #: the mass is somewhere in ``(lower, upper]`` (at ``upper`` for an
    #: exact time, ``lower == upper``).
    atom_cause: np.ndarray
    atom_lower: np.ndarray
    atom_upper: np.ndarray
    atom_mass: np.ndarray
    #: The mass of the units not failed by the end of follow-up, not
    #: given to any cause: ``1 - sum_k CIF_k`` after the last interval.
    beyond: float
    #: The maximised log-likelihood, whether the search met ``tol``, the
    #: final Karush-Kuhn-Tucker gap and the iterations taken.
    log_likelihood: float
    converged: bool
    gap: float
    iterations: int
    #: The rows the estimate was fitted to (windows, cause indices with -1
    #: for none recorded, counts), kept for the bootstrap bounds.
    data_xl: np.ndarray
    data_xr: np.ndarray
    data_cause: np.ndarray
    data_n: np.ndarray
    tol: float = 1e-9
    max_iter: int = 10000
    _structure: Any = None

    _ARRAYS = (
        "atom_cause",
        "atom_lower",
        "atom_upper",
        "atom_mass",
        "data_xl",
        "data_xr",
        "data_cause",
        "data_n",
    )

    def __init__(self) -> None:
        self._boot: dict = {}

    # -- fitting -----------------------------------------------------------

    @classmethod
    @takes_unit
    @accepts_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike | None = None,
        e: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        masked: str = "refuse",
        tol: float = 1e-9,
        max_iter: int = 10000,
    ) -> IntervalCompetingRisks:
        """
        Fit the non-parametric estimate of the cumulative incidences.

        Parameters
        ----------
        x : array_like, optional
            The times: one column, with the flags ``c``, or two columns,
            each row the window ``[xl, xr]`` the failure lies in (as every
            SurPyval fit takes it). Leave it out to give ``xl`` and ``xr``.
        e : array_like
            The cause of each failure: any hashable labels, sorted to fix
            their order in ``event_idx_map``. A missing value (``None``,
            ``NaN``) marks a right-censored row (or, with
            ``masked="include"``, a failure whose cause was not recorded).
        c : array_like, optional
            Censoring flags: 0 a failure at ``x``, 1 right censored (still
            working at ``x``), -1 left censored (found failed at ``x``,
            some time before), 2 interval censored (failed in ``(xl,
            xr]``). If omitted, a row with no cause and one time is right
            censored and every other row's flag comes from its times.
        n : array_like, optional
            Counts. Defaults to 1.
        xl, xr : array_like, optional
            In place of ``x``: each row's lower and upper time (equal for
            an exact time, ``xr`` infinite for a right-censored row).
        masked : str, optional
            A failure (``c`` 0, -1 or 2) with no cause in ``e``:
            ``"refuse"`` (the default) refuses it, as every competing-risks
            fit does; ``"include"`` counts it as a failure from some cause
            in its window (its likelihood is the sum over the causes). This
            assumes that whether a cause was recorded does not depend on
            which cause it was.
        tol : float, optional
            The convergence tolerance: the largest relative gap in the
            Karush-Kuhn-Tucker conditions. Defaults to 1e-9.
        max_iter : int, optional
            The most accelerated EM iterations. Defaults to 10000.

        Returns
        -------
        IntervalCompetingRisks
            The fitted estimate.

        Warns
        -----
        UserWarning
            If the search did not meet ``tol`` in ``max_iter`` iterations.

        Examples
        --------
        Ten units inspected every 2 time units, each failure found at an
        inspection with its cause:

        >>> from surpyval import IntervalCompetingRisks
        >>> import numpy as np
        >>> x = [[0, 2], [0, 2], [2, 4], [2, 4], [4, 6],
        ...      [4, 6], [6, 8], [8, 10], [10, np.inf], [10, np.inf]]
        >>> e = ["a", "b", "a", "a", "b", "a", "b", "a", None, None]
        >>> model = IntervalCompetingRisks.fit(x, e)
        >>> model.cif([2, 4, 10], "a").round(4)
        array([0.1, 0.3, 0.5])
        >>> model.ff([10]).round(4)
        array([0.8])

        For exact and right-censored data it is the Aalen-Johansen
        estimate of ``CompetingRisks``:

        >>> from surpyval import CompetingRisks
        >>> t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> IntervalCompetingRisks.fit(t, e).cif([5, 10], 'a').round(4)
        array([0.3167, 0.6083])
        >>> CompetingRisks.fit(t, e).cif([5, 10], 'a').round(4)
        array([0.3167, 0.6083])
        """
        rows = read_rows(x, e, c, n, xl, xr, masked, "IntervalCompetingRisks")
        model = cls._from_rows(
            rows.xl,
            rows.xr,
            rows.cause,
            rows.n,
            rows.causes,
            tol,
            max_iter,
        )
        if not model.converged:
            warnings.warn(
                "IntervalCompetingRisks: the estimate did not converge to "
                f"tol={tol} in max_iter={max_iter} iterations (the "
                f"Karush-Kuhn-Tucker gap is {model.gap:.3g}); raise "
                "max_iter.",
                stacklevel=3,
            )
        return model

    @classmethod
    def _from_rows(
        cls,
        xl: npt.NDArray,
        xr: npt.NDArray,
        cause: npt.NDArray,
        n: npt.NDArray,
        causes: list,
        tol: float,
        max_iter: int,
    ) -> IntervalCompetingRisks:
        s = _ic_npmle.structure(xl, xr, cause, n, len(causes))
        res = _ic_npmle.fit_masses(s, tol=tol, max_iter=max_iter)
        model = cls()
        model.event_idx_map = {k: i for i, k in enumerate(causes)}
        model.n_event_types = len(causes)
        finite = s.cause >= 0
        order = np.lexsort((s.upper[finite], s.cause[finite]))
        model.atom_cause = s.cause[finite][order].astype(float)
        model.atom_lower = s.lower[finite][order]
        model.atom_upper = s.upper[finite][order]
        model.atom_mass = res.p[finite][order]
        model.beyond = float(res.p[~finite].sum())
        model.log_likelihood = res.loglik
        model.converged = bool(res.converged)
        model.gap = float(res.gap)
        model.iterations = int(res.iterations)
        model.data_xl = np.asarray(xl, dtype=float)
        model.data_xr = np.asarray(xr, dtype=float)
        model.data_cause = np.asarray(cause, dtype=float)
        model.data_n = np.asarray(n, dtype=float)
        model.tol = float(tol)
        model.max_iter = int(max_iter)
        model._structure = s
        return model

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str | None,
        e_col: str,
        c_col: str | None = None,
        n_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        masked: str = "refuse",
        tol: float = 1e-9,
        max_iter: int = 10000,
    ) -> IntervalCompetingRisks:
        """
        Fit from the columns of a :class:`pandas.DataFrame`; see
        :meth:`fit`.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str or None
            The column of times; ``None`` where they are given as windows
            by ``xl_col`` and ``xr_col``.
        e_col : str
            The column of causes (missing for a censored row).
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        xl_col, xr_col : str, optional
            In place of ``x_col``: each row's lower and upper time (a
            blank lower end is a row left censored at the upper one, a
            blank upper end one right censored at the lower one).
        masked, tol, max_iter : optional
            As for :meth:`fit`.

        Returns
        -------
        IntervalCompetingRisks
            The fitted estimate; the frame is kept as ``source_df``.
        """
        from surpyval.utils.dataframe import frame_column

        if xl_col is not None or xr_col is not None:
            if x_col is not None or xl_col is None or xr_col is None:
                raise ValueError(
                    "Give the times as x_col, or as xl_col and xr_col "
                    "(with x_col=None), not both."
                )
            lower = np.asarray(
                frame_column(df, xl_col, "xl_col", time=True), dtype=float
            )
            upper = np.asarray(
                frame_column(df, xr_col, "xr_col", time=True), dtype=float
            )
            x: Any = np.column_stack(
                [
                    np.where(np.isnan(lower), -np.inf, lower),
                    np.where(np.isnan(upper), np.inf, upper),
                ]
            )
        elif x_col is None:
            raise ValueError(
                "Give the times: x_col, or xl_col and xr_col for windows."
            )
        else:
            x = frame_column(df, x_col, "x_col", time=True)
        e = df[e_col].to_numpy(dtype=object)
        c = None if c_col is None else df[c_col].to_numpy()
        n = None if n_col is None else df[n_col].to_numpy()
        model = cls.fit(
            x, e, c=c, n=n, masked=masked, tol=tol, max_iter=max_iter
        )
        model.source_df = df
        return model

    # -- the estimate --------------------------------------------------------

    def _cause_index(self, event: Any) -> int:
        from surpyval.utils import validate_event

        validate_event(self.event_idx_map, event)
        if event is None:
            raise ValueError(
                "Name the cause: cif(x, event) with event one of "
                f"{list(self.event_idx_map)}."
            )
        return self.event_idx_map[event]

    def _cif_from(
        self, mass: npt.NDArray, k: int, x: npt.NDArray
    ) -> npt.NDArray:
        sel = self.atom_cause == k
        up, m = self.atom_upper[sel], mass[sel]
        order = np.argsort(up, kind="stable")
        up, cum = up[order], np.cumsum(m[order])
        idx = np.searchsorted(up, x, side="right")
        out = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
        return np.where(np.isnan(x), np.nan, out)

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """
        The cumulative incidence of cause ``event``: the probability of
        failing from it by ``x``. It rises by each of the cause's masses at
        the upper end of its interval (see ``atom_lower`` /
        ``atom_upper``).
        """
        k = self._cause_index(event)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        return self._cif_from(self.atom_mass, k, q)

    def _ff_from(self, mass: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
        return np.sum(
            [self._cif_from(mass, k, x) for k in range(self.n_event_types)],
            axis=0,
        )

    def _all_causes(self, event: Any) -> None:
        if event is not None:
            raise ValueError(
                "IntervalCompetingRisks has the all-cause sf, ff and Hf "
                "only (event=None): a cause's net survival needs the "
                "cause-specific hazard, which interval-censored data do "
                "not give without a model. Use cif(x, event) for a "
                "cause's incidence, or ParametricCompetingRisks."
            )

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The all-cause failure probability, the sum of the causes'
        cumulative incidences."""
        self._all_causes(event)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        return self._ff_from(self.atom_mass, q)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The all-cause survival, ``1 - ff``."""
        return 1.0 - self.ff(x, event)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The all-cause cumulative hazard, ``-log(sf)``."""
        with np.errstate(divide="ignore"):
            return 0.0 - np.log(self.sf(x, event))

    def innermost_intervals(self, event: Any) -> dict:
        """
        The intervals cause ``event``'s mass lies on, and their masses:
        ``{"lower": ..., "upper": ..., "mass": ...}``, in time order. The
        data say only that each mass is somewhere in ``(lower, upper]``
        (at the time, for an exact time); masses of 0 are intervals the
        estimate could have used but did not.
        """
        k = self._cause_index(event)
        sel = self.atom_cause == k
        return {
            "lower": self.atom_lower[sel].copy(),
            "upper": self.atom_upper[sel].copy(),
            "mass": self.atom_mass[sel].copy(),
        }

    # -- bounds --------------------------------------------------------------

    def _bootstrap(self, n_boot: int, random_state: Any) -> npt.NDArray:
        """``n_boot`` case-bootstrap refits' masses, one row each (cached
        for an integer seed)."""
        from surpyval.utils.rng import as_generator

        key = (
            (int(n_boot), int(random_state))
            if isinstance(random_state, (int, np.integer))
            and not isinstance(random_state, bool)
            else None
        )
        if key is not None and key in self._boot:
            return self._boot[key]
        s = self._structure
        if s is None:
            s = _ic_npmle.structure(
                self.data_xl,
                self.data_xr,
                self.data_cause.astype(np.int64),
                self.data_n,
                self.n_event_types,
            )
            self._structure = s
        rng = as_generator(random_state)
        N = float(s.n.sum())
        units = int(round(N))
        share = s.n / N
        finite = s.cause >= 0
        order = np.lexsort((s.upper[finite], s.cause[finite]))
        out = np.empty((n_boot, int(finite.sum())))
        for b in range(n_boot):
            counts = rng.multinomial(units, share).astype(float)
            res = _ic_npmle.fit_masses(
                s, n=counts, tol=self.tol, max_iter=self.max_iter
            )
            out[b] = res.p[finite][order]
        if key is not None:
            self._boot[key] = out
        return out

    def cb(
        self,
        x: npt.ArrayLike,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        n_boot: int = 500,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Pointwise bootstrap confidence bounds on the cumulative incidence
        of cause ``event`` (``on="cif"``, the default), or on the
        all-cause survival or failure probability (``on="sf"``/``"ff"``,
        ``event=None``).

        The units are resampled with replacement ``n_boot`` times, the
        estimate is refitted to each sample and the bounds are the
        percentiles of the refitted values (the case bootstrap, as
        ``IntervalCoxPH`` offers). The NPMLE at a point converges more
        slowly than root-n, and for current-status data the plain
        bootstrap of it is not consistent, so treat the bounds as
        approximate there (see the class Notes).

        Parameters
        ----------
        x : array_like
            The times.
        event : scalar, optional
            The cause, for ``on="cif"``; ``None`` for ``"sf"``/``"ff"``.
        on : str, optional
            ``"cif"`` (default), ``"sf"`` or ``"ff"`` (``"R"``/``"F"``
            alias them).
        alpha_ci : float, optional
            One minus the confidence level. Defaults to 0.05.
        bound : str, optional
            ``"two-sided"`` (default, ``(n, 2)``), ``"lower"`` or
            ``"upper"`` (one-sided at ``1 - alpha_ci``, ``(n,)``).
        n_boot : int, optional
            The bootstrap samples. Defaults to 500.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling. With an int the refits are kept,
            so later calls with the same ``n_boot`` and seed reuse them.

        Returns
        -------
        ndarray
            The bounds.

        Examples
        --------
        >>> from surpyval import IntervalCompetingRisks
        >>> import numpy as np
        >>> x = [[0, 2], [0, 2], [2, 4], [2, 4], [4, 6],
        ...      [4, 6], [6, 8], [8, 10], [10, np.inf], [10, np.inf]]
        >>> e = ["a", "b", "a", "a", "b", "a", "b", "a", None, None]
        >>> model = IntervalCompetingRisks.fit(x, e)
        >>> model.cb(4, "a", n_boot=200, random_state=1).round(2)
        array([0.1, 0.6])
        """
        check_option("on", on, _CB_ON)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        if (
            isinstance(n_boot, bool)
            or not float(n_boot).is_integer()
            or n_boot < 2
        ):
            raise ValueError(
                f"n_boot must be a whole number of at least 2; got {n_boot!r}"
            )
        on = {"R": "sf", "F": "ff"}.get(on, on)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        if on == "cif":
            k = self._cause_index(event)
        else:
            self._all_causes(event)
        masses = self._bootstrap(int(n_boot), random_state)
        if on == "cif":
            draws = np.array([self._cif_from(m, k, q) for m in masses])
        else:
            draws = np.array([self._ff_from(m, q) for m in masses])
        if on == "sf":
            draws = 1.0 - draws
        if bound == "two-sided":
            lo = np.quantile(draws, alpha_ci / 2, axis=0)
            hi = np.quantile(draws, 1 - alpha_ci / 2, axis=0)
            out = np.column_stack([lo, hi])
            return out.reshape(np.shape(x) + (2,))
        level = alpha_ci if bound == "lower" else 1 - alpha_ci
        out = np.quantile(draws, level, axis=0)
        return out.reshape(np.shape(x))

    # -- plotting ------------------------------------------------------------

    def plot(
        self,
        ax: Any = None,
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
        n_boot: int = 500,
        random_state: Any = 0,
    ) -> Any:
        """
        Plot each cause's cumulative incidence as a step curve, with its
        bootstrap bounds (:meth:`cb`) as a pale band of its colour unless
        ``plot_bounds=False``, as ``CompetingRisks.plot`` draws them.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to draw on; the current axes by default.
        plot_bounds : bool, optional
            Whether to draw the bounds. Defaults to True.
        alpha_ci, n_boot, random_state : optional
            Passed to :meth:`cb` (the seed defaults to 0, so the plot is
            reproducible).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from surpyval import IntervalCompetingRisks
        >>> model = IntervalCompetingRisks.fit(
        ...     [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        ...     ["a", "b", "a", "b", "a"],
        ... )
        >>> fig, ax = plt.subplots()
        >>> model.plot(ax=ax, n_boot=50).get_ylabel()
        'Cumulative incidence'
        >>> plt.close(fig)
        """
        check_alpha_ci(alpha_ci)
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        ends = self.atom_upper[np.isfinite(self.atom_upper)]
        starts = self.atom_lower[np.isfinite(self.atom_lower)]
        first = min(
            [0.0]
            + ([float(starts.min())] if starts.size else [])
            + ([float(ends.min())] if ends.size else [])
        )
        grid = np.unique(np.concatenate([[first], ends]))
        themed = plotting.theme_enabled()
        for cause in sorted(
            self.event_idx_map, key=lambda k: self.event_idx_map[k]
        ):
            (line,) = ax.step(
                grid,
                self.cif(grid, cause),
                where="post",
                label=str(cause),
                color=plotting.next_color(ax),
                linewidth=plotting.LINE_WIDTH if themed else None,
            )
            if plot_bounds:
                cb = self.cb(
                    grid,
                    cause,
                    alpha_ci=alpha_ci,
                    n_boot=n_boot,
                    random_state=random_state,
                )
                ax.fill_between(
                    grid,
                    cb[:, 0],
                    cb[:, 1],
                    step="post",
                    alpha=plotting.BAND_ALPHA if themed else 0.3,
                    color=line.get_color(),
                    linewidth=0,
                )
        ax.set_ylim(0, 1)
        if not ax.get_xlabel():
            ax.set_xlabel(axis_label(getattr(self, "unit", None)))
        ax.set_ylabel("Cumulative incidence")
        ax.set_title("Cumulative incidence by cause (interval-censored)")
        plotting.style_axes(ax)
        plotting.style_legend(ax, title="Cause")
        return ax

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted estimate to a plain, JSON-serialisable dict:
        the causes, the intervals and masses, the fit's summary and the
        rows (so the reloaded estimate has the bootstrap bounds too). The
        reloaded model reproduces every prediction exactly.
        """

        out: dict = {
            "model": "IntervalCompetingRisks",
            "event_idx_map": [
                [to_native(k), int(v)] for k, v in self.event_idx_map.items()
            ],
            "n_event_types": int(self.n_event_types),
            "beyond": float(self.beyond),
            "_neg_ll": -float(self.log_likelihood),
            "converged": bool(self.converged),
            "gap": float(self.gap),
            "iterations": int(self.iterations),
            "tol": float(self.tol),
            "max_iter": int(self.max_iter),
        }
        for name in self._ARRAYS:
            out[name] = np.asarray(getattr(self, name), dtype=float).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> IntervalCompetingRisks:
        """Rebuild the estimate from :meth:`to_dict`'s dict."""
        require_model_tag(
            model_dict,
            "IntervalCompetingRisks",
            "an interval-censored competing-risks estimate",
        )

        out = cls()
        out.event_idx_map = {
            label_from_native(k): int(v)
            for k, v in model_dict["event_idx_map"]
        }
        out.n_event_types = int(model_dict["n_event_types"])
        out.beyond = float(model_dict["beyond"])
        out.log_likelihood = -float(model_dict["_neg_ll"])
        out.converged = bool(model_dict["converged"])
        out.gap = float(model_dict["gap"])
        out.iterations = int(model_dict["iterations"])
        out.tol = float(model_dict["tol"])
        out.max_iter = int(model_dict["max_iter"])
        for name in cls._ARRAYS:
            setattr(out, name, np.array(model_dict[name], dtype=float))
        return out

    def __repr__(self) -> str:
        causes = list(self.event_idx_map)
        shares = {
            k: float(self.atom_mass[self.atom_cause == i].sum())
            for k, i in self.event_idx_map.items()
        }
        lines = [
            "Interval-censored competing-risks estimate (NPMLE)",
            f"Causes: {causes}",
            "Incidence by the end of follow-up: "
            + ", ".join(f"{k}: {v:.4g}" for k, v in shares.items()),
            f"Not failed by the end of follow-up: {self.beyond:.4g}",
            f"Log-likelihood: {self.log_likelihood:.6g}"
            + ("" if self.converged else " (not converged)"),
        ]
        return "\n".join(lines) + "\n"
