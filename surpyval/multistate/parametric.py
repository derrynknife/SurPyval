"""Parametric multi-state models: a parametric distribution (or a
parametric regression) for each transition, and the transition and state
occupation probabilities they imply, from the Kolmogorov forward
equations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate import _core, _forward, _semimarkov, _tvc
from surpyval.multistate._base import (
    check_bound_args,
    probability_bounds,
    time_fraction_bounds,
)
from surpyval.multistate._data import SpellData, check_spells
from surpyval.multistate._entry import multistate_data, other_name
from surpyval.multistate.regression import _covariates
from surpyval.multistate.transitions import TransitionMatrix
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

_CLOCKS = ("forward", "reset")


def _check_clock(clock: str) -> None:
    check_option("clock", clock, _CLOCKS)


def _check_reset(fitters: list, tm: TransitionMatrix, arg: str) -> None:
    """On the clock reset a hazard is a function of the time spent in a
    state, which starts at 0: refuse a distribution that does not live on
    (0, inf)."""
    for fitter, pair in zip(fitters, tm.transitions):
        support = getattr(fitter, "support", None)
        if support is None:
            continue
        lo, hi = (float(v) for v in np.ravel(support)[:2])
        if lo != 0.0 or hi != np.inf:
            raise ValueError(
                f"With clock='reset' each transition's hazard is a function "
                f"of the time spent in its 'from' state, which runs from 0 "
                f"without limit, but the '{arg}' of {pair!r} lives on "
                f"({lo:g}, {hi:g}). Use a distribution on (0, inf), such as "
                "Weibull, Gamma, LogNormal, LogLogistic or Exponential."
            )


def _previous_rows(data: SpellData) -> npt.NDArray:
    """Each row's unit's row before it (by entry time), ``-1`` for a
    unit's first row (every row, without ``i``)."""
    prev = np.full(data.x.size, -1, dtype=np.intp)
    if not data.ids_given:
        return prev
    order = np.lexsort((data.tl, data.cluster))
    same = data.cluster[order[1:]] == data.cluster[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def _with_carry_on(data: SpellData, carry_on: Any) -> SpellData:
    """``data`` with the rows marked in ``carry_on`` taken, on the reset
    clock, as carrying on the stay of their unit's row before; each
    marked row checked to be that: the same unit and state, starting
    where the row before was censored."""
    from dataclasses import replace

    if carry_on is None:
        return data
    carry = np.asarray(carry_on)
    if carry.shape != data.x.shape:
        raise ValueError(
            f"'carry_on' must be one True or False per row ({data.x.size}); "
            f"got shape {np.shape(carry_on)}."
        )
    if carry.dtype != bool:
        if not np.all(np.isin(carry, (0, 1))):
            raise ValueError(
                "'carry_on' must be True (the row carries on its unit's "
                "stay from the row before) or False for each row."
            )
        carry = carry.astype(bool)
    if not np.any(carry):
        return data
    if not data.ids_given:
        raise ValueError(
            "'carry_on' marks rows that carry on their unit's stay from "
            "the row before, so 'i' (the unit of each row) is needed."
        )
    prev = _previous_rows(data)
    marked = np.flatnonzero(carry)
    p = prev[marked]
    ok = p >= 0
    ok[ok] = (
        (data.g[p[ok]] == data.g[marked[ok]])
        & (data.h[p[ok]] < 0)
        & np.isfinite(data.tl[marked[ok]])
        & (data.tl[marked[ok]] == data.x[p[ok]])
    )
    if not np.all(ok):
        bad = marked[~ok]
        k = int(bad[0])
        raise ValueError(
            f"{bad.size} row(s) marked in 'carry_on' do not carry on the "
            f"row before them (the first is row {k}): a row that carries "
            "on a stay is the same unit's next row, in the same state, "
            "starting (tl) exactly where the row before ended censored "
            "(x, with no 'e'). Correct the rows' tl, state or e, or unmark "
            "the row if it starts a new stay."
        )
    return replace(data, carry=carry)


def _stays(data: SpellData) -> npt.NDArray:
    """On the reset clock, the time each row's stay in its state began:
    its ``tl`` (0 where it was observed from the start of the clock), or,
    for a row marked as carrying on its unit's stay from the row before
    (``data.carry``: a stay split into rows), that row's, so the time in
    the state runs on and does not restart. Rows are never taken to
    carry on unless marked: a unit censored in a state and renewed in it
    at the same time (a repair as good as new) starts a new stay."""
    stay = np.where(np.isfinite(data.tl), data.tl, 0.0)
    if data.carry is None:
        return stay
    prev = _previous_rows(data)
    order = np.lexsort((data.tl, data.cluster))
    for row in order.tolist():
        if data.carry[row]:
            stay[row] = stay[prev[row]]
    return stay


def _durations(data: SpellData) -> npt.NDArray:
    """The time each spell spent in its state by its end, ``x`` minus
    the start of its stay (:func:`_stays`: ``tl``, or ``0`` where the
    spell was observed from the start of the clock, or the start of the
    stay a row carries on)."""
    dur = data.x - _stays(data)
    if np.any(dur <= 0):
        bad = int(np.flatnonzero(dur <= 0)[0])
        raise ValueError(
            "With clock='reset' each spell is fitted on the time spent in "
            f"its state, x - tl, which must be positive; spell {bad} has "
            f"x = {data.x[bad]:g} and tl = {data.tl[bad]:g}. Remove "
            "spells of no length, or give each spell's start in 'tl'."
        )
    return dur


def _reset_entries(data: SpellData) -> npt.NDArray:
    """On the reset clock, the time already spent in the state when each
    row starts: 0 (no truncation, ``-inf``) for a row that starts its
    stay, the time since the stay began for a row that carries one on
    (a left truncation on the time in the state)."""
    entry = np.where(np.isfinite(data.tl), data.tl, 0.0) - _stays(data)
    return np.where(entry > 0, entry, -np.inf)


def _fitters(dist: Any, tm: TransitionMatrix, regression: bool) -> list:
    """The fitter of each transition, in the order of
    ``tm.transitions``: ``dist`` for every one, or its entry of a
    ``{(from, to): fitter}`` dict."""
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )
    from surpyval.univariate.regression.regression_data import (
        DataFrameRegressionMixin,
    )

    arg = "model" if regression else "dist"
    if isinstance(dist, dict):
        unknown = [k for k in dist if k not in tm.transitions]
        if unknown:
            raise ValueError(
                f"'{arg}' has entries for {unknown!r}, which are not "
                f"transitions (the transitions are {list(tm.transitions)!r})."
            )
        missing = [k for k in tm.transitions if k not in dist]
        if missing:
            raise ValueError(
                f"'{arg}' has no fitter for the transitions {missing!r}; "
                "give one for every transition, or one fitter for all."
            )
        out = [dist[k] for k in tm.transitions]
    else:
        out = [dist] * tm.n_transitions
    for fitter, pair in zip(out, tm.transitions):
        if regression:
            ok = isinstance(fitter, DataFrameRegressionMixin)
            what = (
                "a parametric regression fitter, such as WeibullPH, "
                "ExponentialPH, WeibullAFT or LogNormalAFT"
            )
        else:
            ok = isinstance(fitter, ParametricFitter)
            what = (
                "a parametric distribution, such as Weibull, Exponential, "
                "Gamma or LogNormal"
            )
        if not ok:
            raise ValueError(
                f"'{arg}' must be {what} (for every transition, or one per "
                f"transition in a dict); got {fitter!r} for {pair!r}."
            )
    return out


def _x_column(df: Any, x_col: Any) -> Any:
    """The exit times of a DataFrame: one column, or two (``[xl_col,
    xr_col]``) for moves known only to lie between two times."""
    from surpyval.utils.dataframe import frame_column

    if isinstance(x_col, (list, tuple)):
        if len(x_col) != 2:
            raise ValueError(
                "'x_col' is one column name, or two ([xl_col, xr_col]) "
                f"for interval-censored moves; got {x_col!r}."
            )
        return np.column_stack(
            [
                np.asarray(frame_column(df, c, "x_col"), dtype=float)
                for c in x_col
            ]
        )
    return frame_column(df, x_col, "x_col")


def _lows(fitters: list) -> list:
    """The start of each transition's support."""
    return [
        float(np.ravel(getattr(f, "support", [-np.inf]))[0]) for f in fitters
    ]


def _check_left_start(x: Any, tl: Any, fitters: list) -> None:
    """Refuse a left-censored move (``xl = -inf``) in a spell observed from
    the start of the clock (no ``tl``) where a transition's distribution
    is defined below 0: the move is read as lying between the spell's
    start and ``xr``, and there the clock has no start."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or not np.any(np.isneginf(arr[:, 0])):
        return
    if min(_lows(fitters)) >= 0:
        return
    left = np.isneginf(arr[:, 0])
    start = (
        np.full(arr.shape[0], -np.inf)
        if tl is None
        else np.broadcast_to(np.asarray(tl, dtype=float), arr.shape[:1])
    )
    bad = left & ~np.isfinite(start)
    if np.any(bad):
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{int(bad.sum())} left-censored move(s) (xl = -inf; the first "
            f"is row {k}) are in spells observed from the start of the "
            "clock (no tl), and a transition's distribution is defined "
            "below 0 (Normal, Gumbel, Logistic), so the spell has no "
            "start to read the move from. Give the spell's start in tl, "
            "or the last time the unit was seen in its state as xl."
        )


def _with_truncation(data: SpellData, tr: Any, reached: Any) -> SpellData:
    """``data`` with its right truncation (each row's cut-off ``tr``, the
    state ``reached`` that had to be entered by then), checked."""
    from dataclasses import replace

    if tr is None:
        if reached is not None:
            raise ValueError(
                "'reached' names the state a right-truncated unit had to "
                "reach by its cut-off; give the cut-offs in 'tr' too."
            )
        return data
    tau = np.asarray(tr, dtype=float)
    if tau.ndim == 0:
        tau = np.full(data.x.size, float(tau))
    if tau.shape != data.x.shape:
        raise ValueError(
            f"'tr' must be one cut-off per row ({data.x.size}); got shape "
            f"{np.shape(tr)}."
        )
    if np.any(np.isnan(tau)):
        raise ValueError(
            "'tr' has missing (NaN) cut-offs; use inf for a unit that was "
            "not right truncated."
        )
    if not np.any(np.isfinite(tau)):
        return data
    tm = data.tm
    if reached is None:
        absorbing = list(tm.absorbing)
        if len(absorbing) != 1:
            raise ValueError(
                "Right-truncated data need the state each unit had to "
                "reach by its cut-off to be seen: give it as reached= "
                f"(the transitions have {len(absorbing)} absorbing states, "
                "so none is taken by default)."
            )
        reached = absorbing[0]
    if reached not in tm.states:
        raise ValueError(
            f"reached={reached!r} is not a state (the states are "
            f"{list(tm.states)!r})."
        )
    if not data.ids_given:
        raise ValueError(
            "Right truncation is of units (seen only if they reached a "
            "state by a cut-off), so 'i' (the unit of each row) is "
            "needed."
        )
    return replace(data, tr=tau, reached=tm.index(reached))


class _ParametricMultiStateBase(SerialisableMixin):
    """What the parametric multi-state models share; the subclasses set
    how a transition is fitted and how its hazard is evaluated."""

    _regression = False

    #: The states and the allowed transitions.
    transitions: TransitionMatrix
    #: Each transition's fitted model, by ``(from, to)``.
    models: dict
    #: The time scale of the hazards: ``"forward"``, the time since the
    #: start (a Markov model).
    clock: str
    #: The default start ``s`` of a prediction: the earliest entry, or 0
    #: where a spell is observed from the start of the clock.
    start: float
    #: The last observed time.
    last: float
    _data: SpellData
    _Z: npt.NDArray
    _cache: dict
    #: The joint fit of every transition, where some moves are interval
    #: censored (the likelihood is then not a product of one factor per
    #: transition); ``None`` otherwise.
    _joint: Any = None

    # -- fitting ------------------------------------------------------------

    @classmethod
    def _fit(
        cls, data: SpellData, Z: npt.NDArray, fitters: list, clock: str
    ) -> Any:
        joint = None
        truncated = data.tr is not None
        if truncated:
            from surpyval.multistate import _truncation

            _truncation.check(data, np.asarray(Z, dtype=float), clock)
        if data.has_intervals or truncated:
            from surpyval.multistate import _interval, _joint

            if clock == "reset":
                _check_reset(
                    fitters, data.tm, "model" if cls._regression else "dist"
                )
                carried = np.isfinite(_reset_entries(data))
                if np.any(carried):
                    k = int(np.flatnonzero(carried)[0])
                    raise ValueError(
                        f"Row {k} is marked as carrying on its unit's stay "
                        "from the row before (carry_on), and the data have "
                        "interval-censored moves: with clock='reset' the "
                        "two are not fitted together. Give each stay as "
                        "one row."
                    )
            if truncated:
                _truncation.check_events(data)
            lik = _interval.IntervalLikelihood(
                data, clock, cls._row_hazard(Z), _lows(fitters)
            )
            start = _interval.midpoint_data(data, lik.succ)
            models = cls._fit_transitions(start, Z, fitters, clock)
            starts = [models[k] for k in data.tm.transitions]
            units = _joint.coefficient_units(starts, Z)
            what = f"The fit of {cls.__name__} to interval-censored moves"
            if truncated:
                what = f"The fit of {cls.__name__} to right-truncated units"
                lik = cls._truncated(lik, data, Z, fitters)
                # (the first-passage grids planned at the starting values,
                # and again at the estimate, the search going on from there
                # where they moved)
                nodes = lik.plan(starts)
                starts = _joint.fit(
                    starts,
                    lik.neg_ll,
                    what,
                    covariance=False,
                    central=True,
                    units=units,
                ).models
                again = lik.plan(starts)
                if len(again) != len(nodes) or any(
                    not np.array_equal(a, b) for a, b in zip(again, nodes)
                ):
                    starts = _joint.fit(
                        starts,
                        lik.neg_ll,
                        what,
                        covariance=False,
                        central=True,
                        units=units,
                    ).models
            joint = _joint.fit(
                starts,
                lik.neg_ll,
                what,
                polish=True,
                central=True,
                units=units,
            )
            if truncated:
                joint = _truncation.flag_boundary(
                    joint, data, cls._row_hazard(Z), _lows(fitters), lik.neg_ll
                )
            models = dict(zip(data.tm.transitions, joint.models))
        else:
            models = cls._fit_transitions(data, Z, fitters, clock)
        out = cls()
        out.transitions = data.tm
        out.models = models
        out.clock = clock
        out._data = data
        out._Z = Z
        out._joint = joint
        out._setup()
        return out

    @classmethod
    def _fit_transitions(
        cls, data: SpellData, Z: Any, fitters: list, clock: str
    ) -> dict:
        """Each transition's model fitted to its own factor of the
        likelihood (the whole fit, where every move's time is known)."""
        tm = data.tm
        frm, to = tm.from_index(), tm.to_index()
        counts = np.array(
            [
                data.n[(data.g == frm[h]) & (data.h == to[h])].sum()
                for h in range(tm.n_transitions)
            ]
        )
        if np.any(counts == 0):
            none = [tm.transitions[h] for h in np.flatnonzero(counts == 0)]
            raise ValueError(
                f"The transitions {none!r} have no events, so their "
                "distributions cannot be estimated: leave them out of "
                "'transitions'."
            )
        models = {}
        found: list = []
        if clock == "reset":
            _check_reset(fitters, tm, "model" if cls._regression else "dist")
            x_fit = _durations(data)
        else:
            x_fit = data.x
        for h, pair in enumerate(tm.transitions):
            rows = data.g == frm[h]
            c = np.where(data.h[rows] == to[h], 0, 1)
            if clock == "reset":
                # The clock starts at 0 on entry to the state: no
                # truncation, but for a row that carries on a stay (a
                # stay split into rows), which enters at the time
                # already spent in the state.
                tl = _reset_entries(data)[rows]
                extra: dict = {"tl": tl} if np.any(np.isfinite(tl)) else {}
            else:
                # An entry at or before the start of the distribution's
                # support truncates nothing: leave it out (the fit is then
                # the plain one, on the same likelihood).
                low = float(
                    np.ravel(getattr(fitters[h], "support", [-np.inf]))[0]
                )
                tl = np.where(data.tl[rows] <= low, -np.inf, data.tl[rows])
                extra = {"tl": tl} if np.any(np.isfinite(tl)) else {}
            models[pair] = cls._fit_one(
                fitters[h],
                x_fit[rows],
                Z[rows],
                c,
                data.n[rows],
                extra,
                found,
            )
        cls._warn_found(found)
        return models

    @classmethod
    def _row_hazard(cls, Z: Any) -> Any:
        """``hazard(model, t, rows, rate)``: a transition's cumulative
        hazard (hazard with ``rate``) at ``t`` for the spells ``rows``."""
        raise NotImplementedError

    @staticmethod
    def _fit_one(
        fitter: Any,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        extra: dict,
        found: list,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def _warn_found(found: list) -> None:
        return None

    @classmethod
    def _truncated(
        cls, base: Any, data: SpellData, Z: Any, fitters: list
    ) -> Any:
        """The likelihood of right-truncated units: ``base`` divided by
        each unit's chance of reaching the state by its cut-off."""
        from surpyval.multistate import _truncation

        entries = data.tl[np.isfinite(data.tl)]
        begin = min(float(entries.min()), 0.0) if entries.size else 0.0
        span = float(np.max(data.x)) - begin
        passage = _truncation.FirstPassage(
            data,
            cls._row_hazard(Z),
            _lows(fitters),
            span if span > 0 else 1.0,
            np.asarray(Z, dtype=float),
        )
        return _truncation.TruncatedLikelihood(base, passage)

    def _interval_likelihood(self) -> Any:
        """The likelihood a joint fit maximised over every transition's
        parameters at once (to interval-censored moves, or right-truncated
        units; on the latter's grids planned at the estimate), or
        ``None`` where every move's time is known, nothing is truncated
        and the fit is a transition at a time."""
        truncated = self._data.tr is not None
        if self._joint is None or not (self._data.has_intervals or truncated):
            return None
        from surpyval.multistate import _interval

        fitters = [m.dist for m in self._models()]
        lik = _interval.IntervalLikelihood(
            self._data,
            self.clock,
            self._row_hazard(self._Z),
            _lows(fitters),
        )
        if truncated:
            lik = self._truncated(lik, self._data, self._Z, fitters)
            lik.plan(self._models())
        return lik

    def _setup(self) -> None:
        d = self._data
        tl = d.tl
        finite = tl[np.isfinite(tl)]
        starts = list(finite)
        if np.any(~np.isfinite(tl)):
            starts.append(0.0)
        self.start = float(min(starts))
        self.last = float(np.max(d.x))
        self._t = np.unique(d.x[d.h >= 0])
        self._cache = {}
        if self.clock == "reset":
            dur = d.x - _stays(d)
            self._stay = float(np.max(dur)) if dur.size else 1.0

    def __getstate__(self) -> dict:
        # (the cache of predictions holds the hazards as closures, which do
        # not pickle; it is rebuilt on use)
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    # -- parameters ---------------------------------------------------------

    def _models(self) -> list:
        return [self.models[k] for k in self.transitions.transitions]

    @property
    def states(self) -> tuple:
        """The states, in order."""
        return self.transitions.states

    @property
    def params(self) -> npt.NDArray:
        """Every transition's parameters in one vector, transition by
        transition in the order of ``transitions.transitions``, named by
        :attr:`parameter_names`."""
        return np.concatenate(
            [
                np.ravel(np.asarray(m.params, dtype=float))
                for m in self._models()
            ]
        )

    @property
    def parameter_names(self) -> list[str]:
        """The names of :attr:`params`: ``"<from> -> <to>: <name>"``."""
        names = self.transitions.transition_names()
        return [
            f"{t}: {p}"
            for t, m in zip(names, self._models())
            for p in m.parameter_names
        ]

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows and columns of :meth:`covariance`:
        ``"<from> -> <to>: <name>"`` over each transition's own
        ``covariance_names``."""
        names = self.transitions.transition_names()
        return [
            f"{t}: {p}"
            for t, m in zip(names, self._models())
            for p in m.covariance_names
        ]

    def _estimates(self) -> npt.NDArray:
        return np.concatenate(
            [
                np.ravel(np.asarray(m._estimates(), float))
                for m in self._models()
            ]
        )

    def covariance(self) -> npt.NDArray:
        """
        The covariance of every transition's parameter estimates, in the
        order of :attr:`covariance_names`: each transition's own
        ``covariance()`` on the diagonal, zeros between transitions.

        The zeros are exact: the likelihood of a Markov multi-state model
        is a product of one factor per transition (its hazard where it
        happened, its survival over each spell at risk of it), so the
        transitions' parameters are estimated from their own factors.
        Where some moves are interval censored the likelihood does not
        factor, the transitions are fitted together, and this is the
        inverse of the whole likelihood's curvature, with terms between
        transitions.
        """
        if self._joint is not None:
            return self._joint.cov.copy()
        from surpyval.univariate.competing_risks._assembly import (
            block_diagonal,
        )

        return block_diagonal(
            [np.atleast_2d(m.covariance()) for m in self._models()]
        )

    def standard_errors(self) -> npt.NDArray:
        """The standard errors of the parameter estimates, the square
        roots of the diagonal of :meth:`covariance`."""
        from surpyval.utils.linalg import standard_errors_of

        return standard_errors_of(self.covariance())

    @property
    def maximum(self) -> str:
        """What the transitions' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of their own
        ``maximum``, since the likelihood is maximised a transition at a
        time."""
        if self._joint is not None:
            return self._joint.maximum
        from surpyval.utils.no_maximum import combined_maximum

        return combined_maximum(
            getattr(m, "maximum", "unknown") for m in self._models()
        )

    def neg_ll(self) -> float:
        """The negative log-likelihood: the sum of the transitions' (with
        interval-censored moves, that of the joint fit)."""
        if self._joint is not None:
            return float(self._joint.neg_ll)
        return float(sum(m.neg_ll() for m in self._models()))

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[int, float]:
        if self._joint is not None:
            d = self._data
            return self._joint.k, float(d.n[d.h >= 0].sum())
        from surpyval.univariate.competing_risks._assembly import (
            summed_ic_terms,
        )

        return summed_ic_terms(self._models())

    def aic(self) -> float:
        """Akaike's information criterion, ``2 K + 2 neg_ll``, ``K`` the
        parameters estimated over every transition."""
        k, _ = self._ic_terms()
        return float(2 * k + 2 * self.neg_ll())

    def bic(self) -> float:
        """The Bayesian information criterion, ``2 neg_ll + K ln(n)``,
        ``n`` the observed transitions (weighted by their counts), as the
        transitions' own fits count their events."""
        k, n = self._ic_terms()
        return float(2 * self.neg_ll() + k * np.log(n))

    def aic_c(self) -> float:
        """The small-sample corrected AIC, with the ``K`` and ``n`` of
        :meth:`bic`; ``nan`` where ``n <= K + 1``."""
        from surpyval.univariate.information_criteria import corrected_aic

        k, n = self._ic_terms()
        return corrected_aic(self.aic(), k, n)

    # -- the hazards and the forward equations ------------------------------

    @staticmethod
    def _Hf(
        model: Any, t: npt.NDArray, z: npt.NDArray, rate: bool = False
    ) -> npt.NDArray:
        raise NotImplementedError

    def _zkey(self, z: Any) -> Any:
        """The covariates ``z`` (a row, or a covariate path) as a key of
        the cache of predictions. A path is keyed by its identity, and
        held in the cache so that the identity is not reused."""
        if _tvc.is_path(z):
            self._cache.setdefault(("path", id(z)), z)
            return ("path", id(z))
        return tuple(z.tolist())

    @staticmethod
    def _path_origin(path: Any) -> float:
        """Where the cumulative hazards along a covariate path are taken
        from: its start, or 0 for a path with a value at every time."""
        return float(path.start) if np.isfinite(path.start) else 0.0

    def _cumhaz(self, models: list, z: npt.NDArray) -> Any:
        """The transitions' cumulative hazards at ``t`` (their hazards
        with ``rate``), as :func:`_forward.plan` takes them; ``z`` a row
        of covariates, or a covariate path."""
        if _tvc.is_path(z):
            return _tvc.path_cumhaz(models, z, self._path_origin(z))

        def H(t: npt.NDArray, rate: bool) -> npt.NDArray:
            out = np.empty((t.size, len(models)))
            with np.errstate(all="ignore"):
                for h, m in enumerate(models):
                    out[:, h] = np.ravel(self._Hf(m, t, z, rate))
            return out

        return H

    def _start(self, s: float | None) -> float:
        if s is None:
            return self.start
        s = float(s)
        if not np.isfinite(s):
            raise ValueError(
                f"'s', the start of the prediction, must be a finite time; "
                f"got {s}."
            )
        return s

    def _p0(self, s: float, from_state: Any) -> npt.NDArray:
        k = self.transitions.n_states
        if from_state is not None:
            p0 = np.zeros(k)
            p0[self.transitions.index(from_state)] = 1.0
            return p0
        if self.clock == "reset" and s != self.start:
            raise ValueError(
                "With clock='reset' a prediction from a start 's' other "
                f"than the earliest entry ({self.start:g}) needs "
                "'from_state': the units under observation at s have been "
                "in their states for different times, and a prediction "
                "takes each to have just entered its state at s. Give the "
                "state entered at s as from_state."
            )
        key = ("p0", s)
        if key not in self._cache:
            first = (
                float(self._t[0]) if s == self.start and self._t.size else None
            )
            self._cache[key] = _core.at_risk_distribution(
                self._data, s, first
            )[0]
        return self._cache[key]

    def _scale(self) -> float:
        """The time scale of the solver's grid and accuracy: the span of
        the data."""
        span = self.last - self.start
        return float(span) if span > 0 else 1.0

    def _grid(self, z: npt.NDArray, s: float, times: npt.NDArray) -> Any:
        """The plan of the product integral of :math:`P(s, t \\mid z)`
        at the unique ``times``."""
        key = ("grid", self._zkey(z), s, tuple(times.tolist()))
        if key not in self._cache:
            tm = self.transitions
            breaks = self._breaks()
            if _tvc.is_path(z) and times.size:
                # (grid nodes where the path jumps or bends)
                breaks += _tvc.path_breaks(z, s, float(np.max(times)))
            self._cache[key] = _forward.plan(
                self._cumhaz(self._models(), z),
                s,
                times,
                self._scale(),
                tm.from_index(),
                tm.to_index(),
                tm.n_states,
                breaks=breaks,
            )
        return self._cache[key]

    def _breaks(self) -> list:
        """The starts of the transitions' supports, where a hazard can
        jump from 0 (or be infinite)."""
        out = []
        for m in self._models():
            dist = getattr(m, "dist", None)
            support = getattr(dist, "support", None)
            if support is not None and np.isfinite(support[0]):
                out.append(float(support[0]))
        return out

    def _matrices(
        self,
        models: list | None,
        z: npt.NDArray,
        s: float,
        uniq: npt.NDArray,
        what: str,
    ) -> npt.NDArray:
        """``(len(uniq), K, K)`` at the unique times ``uniq`` (finite, at
        or after ``s``): the transition probabilities :math:`P(s, t)`
        (``what="P"``) or their integrals from ``s`` (``"los"``, the
        expected time spent in each state by ``t``), from each state at
        ``s``."""
        tm = self.transitions
        if self.clock == "forward":
            grid = self._grid(z, s, uniq)
            f = _forward.evaluate if what == "P" else _forward.integrate
            return f(
                self._cumhaz(self._models() if models is None else models, z),
                grid,
                tm.from_index(),
                tm.to_index(),
                tm.n_states,
            )
        k = tm.n_states
        h = self._reset_step(z)
        near = _semimarkov.MIN_CELLS * h
        out = np.empty((uniq.size, k, k))
        sol = None
        for n, t in enumerate(uniq):
            u = float(t - s)
            if 0 < u < near:
                # (its own grid, ending at u)
                one = self._reset_solution(
                    models, z, u, u / _semimarkov.MIN_CELLS
                )
                occ, los = one.at(u)
            else:
                if sol is None:
                    sol = self._reset_solution(
                        models, z, float(np.max(uniq) - s)
                    )
                occ, los = sol.at(u)
            out[n] = occ if what == "P" else los
        return out

    def _semi_model(self, models: list, z: npt.NDArray) -> Any:
        tm = self.transitions
        return _semimarkov.Model(
            self._cumhaz(models, z),
            tm.from_index(),
            tm.to_index(),
            tm.n_states,
        )

    def _reset_step(self, z: npt.NDArray) -> float:
        """The grid step of the semi-Markov solution for covariates ``z``,
        set by the fitted model (and kept for its bounds)."""
        key = ("step", self._zkey(z))
        if key not in self._cache:
            self._cache[key] = _semimarkov.step(
                self._semi_model(self._models(), z), self._stay
            )
        return self._cache[key]

    def _reset_solution(
        self,
        models: list | None,
        z: npt.NDArray,
        reach: float,
        step: float | None = None,
    ) -> Any:
        """The solved semi-Markov grid from each state, reaching past
        ``reach`` (a time spent since the start of the prediction); with
        ``step``, a grid of that step ending at ``reach``."""
        if step is None:
            h = self._reset_step(z)
            cells = _semimarkov.cells_for(h, reach, self._stay)
        else:
            h = step
            cells = _semimarkov.MIN_CELLS
        point = models is None
        key = ("solution", self._zkey(z), cells, h)
        if point and key in self._cache:
            return self._cache[key]
        model = self._semi_model(
            self._models() if models is None else models, z
        )
        weights = model.weights(h, cells)
        starts = np.eye(self.transitions.n_states)
        sol = _semimarkov.Solution(
            model, weights, starts, _semimarkov.solve(model, weights, starts)
        )
        if point:
            self._cache[key] = sol
        return sol

    def _point(
        self, z: npt.NDArray, s: float, times: npt.NDArray, what: str
    ) -> npt.NDArray:
        """:meth:`_matrices` of the fitted model at ``times``, cached."""
        k = self.transitions.n_states
        if times.size == 0:
            return np.zeros((0, k, k))
        uniq, inverse = np.unique(times, return_inverse=True)
        key = (what, self._zkey(z), s, tuple(uniq.tolist()))
        if key not in self._cache:
            models = None if self.clock == "reset" else self._models()
            self._cache[key] = self._matrices(models, z, s, uniq, what)
        return self._cache[key][np.ravel(inverse)]

    def _P_at(
        self, z: npt.NDArray, s: float, times: npt.NDArray
    ) -> npt.NDArray:
        """``(len(times), K, K)``: :math:`P(s, t)` at ``times`` (finite,
        at or after ``s``)."""
        return self._point(z, s, times, "P")

    def _occupancy_values(
        self,
        z: npt.NDArray,
        s: float,
        p0: npt.NDArray,
        b: int,
        times: npt.NDArray,
        what: str = "P",
    ) -> npt.NDArray:
        return (p0 @ self._point(z, s, times, what))[:, b]

    def _occupancy_se_values(
        self,
        z: npt.NDArray,
        s: float,
        p0: npt.NDArray,
        b: int,
        times: npt.NDArray,
        what: str = "P",
    ) -> npt.NDArray:
        """The delta-method standard errors of the occupation
        probabilities (``what="P"``) or the expected times in the state
        (``"los"``) at ``times``, over every transition's parameters, by
        central differences on the point estimate's grid."""
        from surpyval.univariate.competing_risks._assembly import (
            central_differences,
            checked_sd,
        )

        if times.size == 0:
            return np.zeros(0)
        uniq, inverse = np.unique(times, return_inverse=True)
        idx = np.ravel(inverse)
        models = self._models()
        sizes = [len(m.covariance_names) for m in models]
        est = self._estimates()
        cov = self.covariance()
        if self.clock == "forward":
            # (fix the grid at the point estimate's)
            self._grid(z, s, uniq)
        else:
            self._reset_step(z)

        def occ(values: npt.NDArray) -> npt.NDArray:
            rebuilt = []
            start = 0
            for m, size in zip(models, sizes):
                rebuilt.append(m.with_params(values[start : start + size]))
                start += size
            P = self._matrices(rebuilt, z, s, uniq, what)
            return (p0 @ P[idx])[:, b]

        from surpyval.utils.warnings import ignoring_warnings

        with ignoring_warnings():
            G = central_differences(occ, est, cov, times.size)
        var = np.einsum("ia,ij,ja->a", G, cov, G)
        on = "occupancy" if what == "P" else "length_of_stay"
        return checked_sd(var, times, on, np.ones(times.size, dtype=bool))

    # -- serialisation ------------------------------------------------------

    def _data_dict(self) -> dict:
        d = self._data
        tm = self.transitions
        x: list = (
            d.x.tolist()
            if d.xl is None
            else np.column_stack([d.xl, d.x]).tolist()
        )
        out = {
            "transitions": tm.to_dict(),
            "x": x,
            "tl": [None if not np.isfinite(v) else float(v) for v in d.tl],
            "state": [to_native(tm.states[g]) for g in d.g],
            "e": [None if h < 0 else to_native(tm.states[h]) for h in d.h],
            "n": d.n.tolist(),
            "i": d.cluster.tolist() if d.ids_given else None,
            "clock": self.clock,
            "models": [m.to_dict() for m in self._models()],
        }
        if self._joint is not None:
            out["joint"] = self._joint.to_dict()
        if d.tr is not None and d.reached is not None:
            out["tr"] = [
                None if not np.isfinite(v) else float(v) for v in d.tr
            ]
            out["reached"] = to_native(tm.states[int(d.reached)])
        if d.carry is not None:
            out["carry_on"] = [bool(v) for v in d.carry]
        return out

    @classmethod
    def _from_data_dict(cls, model_dict: dict, model_cls: Any) -> Any:
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        tl = [-np.inf if v is None else v for v in model_dict["tl"]]
        data = check_spells(
            model_dict["x"],
            [label_from_native(v) for v in model_dict["e"]],
            [label_from_native(v) for v in model_dict["state"]],
            tl,
            model_dict.get("i"),
            model_dict["n"],
            tm,
            intervals=True,
        )
        if model_dict.get("tr") is not None:
            data = _with_truncation(
                data,
                [np.inf if v is None else v for v in model_dict["tr"]],
                label_from_native(model_dict["reached"]),
            )
        if model_dict.get("carry_on") is not None:
            data = _with_carry_on(data, model_dict["carry_on"])
        out = cls()
        out.transitions = tm
        out.models = {
            pair: model_cls.from_dict(sub)
            for pair, sub in zip(tm.transitions, model_dict["models"])
        }
        out.clock = model_dict.get("clock", "forward")
        out._joint = None
        if "joint" in model_dict:
            from surpyval.multistate._joint import JointFit

            out._joint = JointFit.from_dict(model_dict["joint"], out._models())
        out._data = data
        # (no covariates; a regression's from_dict sets its own)
        out._Z = np.zeros((data.x.size, 0))
        out._setup()
        return out

    def _repr_lines(self, title: str) -> list[str]:
        tm = self.transitions
        lines = [
            title,
            f"States: {list(tm.states)!r}",
            f"Clock: {self.clock}",
            "Transitions:",
        ]
        for name, m in zip(tm.transition_names(), self._models()):
            params = ", ".join(
                f"{p}={v:.6g}"
                for p, v in zip(
                    m.parameter_names, np.ravel(np.asarray(m.params, float))
                )
            )
            lines.append(f"    {name}: {self._describe(m)} ({params})")
        return lines

    @staticmethod
    def _describe(model: Any) -> str:
        raise NotImplementedError


class ParametricMultiState(_ParametricMultiStateBase):
    r"""
    A parametric multi-state model: a parametric distribution for the
    hazard of each transition, and the transition and state occupation
    probabilities they imply.

    The hazard of the transition from state :math:`g` to state :math:`k`
    is that of a distribution (Weibull by default) on the time since the
    start (by default the clock goes forward: a Markov model). Each
    transition's distribution is fitted to the spells in its 'from'
    state, where the spells that ended in another transition, or were
    censored, are censored, and a spell that began at ``tl`` is left
    truncated there (delayed entry). That is the maximum-likelihood fit of
    the whole model, since its likelihood is a product of one factor per
    transition.

    With ``clock="reset"`` each hazard is instead a function of the time
    spent in the current state (a semi-Markov model): each transition is
    fitted on the durations ``x - tl``, and the probabilities come from
    the renewal equations, solved by convolution on a grid refined until
    they are accurate to about :math:`10^{-6}`; the forward equations
    below are those of the default ``clock="forward"``.

    The transition probabilities :math:`P(s, t)` solve the Kolmogorov
    forward equations :math:`\partial P(s, t) / \partial t = P(s, t)
    Q(t)`, :math:`Q(t)` the matrix of the transition hazards, which are
    solved by product integration on an adaptive grid to an accuracy of
    about :math:`10^{-8}` in each row (the sum of the absolute errors of
    its elements) over the span of the data, and in proportion beyond
    it. The bounds on the state occupation probabilities come
    from the delta method over every transition's parameters.

    A move found only at an inspection (``x`` as ``[xl, xr]``) is
    interval censored: the likelihood integrates over its unknown time
    (taking the move to be direct), which ties the transitions together,
    so they are then fitted at once (BFGS from their one-at-a-time fits)
    with a covariance from the whole likelihood's Hessian.

    Unlike the non-parametric :class:`~surpyval.multistate.MultiState`,
    its curves are smooth and can be extended past the data, as far as
    the distributions can be trusted there.

    Call the class method ``ParametricMultiState.fit`` (or
    ``fit_from_df``); it returns a fitted instance.

    References
    ----------
    Andersen, P. K. and Keiding, N. (2002), "Multi-state models for event
    history analysis", *Statistical Methods in Medical Research*, 11(2),
    91-115.

    Jackson, C. H. (2016), "flexsurv: a platform for parametric survival
    modeling in R", *Journal of Statistical Software*, 70(8), 1-33.
    """

    @classmethod
    @takes_unit
    @other_name("s", "state")
    @multistate_data(intervals=True)
    @accepts_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        i: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        tr: npt.ArrayLike | float | None = None,
        reached: Any = None,
        carry_on: npt.ArrayLike | None = None,
    ) -> ParametricMultiState:
        """
        Fit a parametric distribution to each transition of spells in
        states.

        Parameters
        ----------
        x, e, state, tl, i, n, transitions
            The spells, as for :meth:`MultiState.fit
            <surpyval.multistate.MultiState.fit>`: the time each ended
            (``x``), the state it ended in (``e``, missing where it was
            censored), the state it was in (``state``), when it began
            (``tl``), the unit (``i``), counts (``n``) and the allowed
            transitions. ``x`` may be a :class:`~surpyval.SurpyvalData`
            holding them all, and ``s`` is another name for ``state``.
            ``x`` may also be two columns, ``[xl, xr]`` per spell: a move
            out of the state known only to lie in ``(xl, xr]`` (the last
            time the unit was seen in the state and the first it was seen
            in the next; ``xl = xr`` where the time is known). The spell
            it began is the unit's next one, whose ``tl`` lies in the
            interval; the likelihood integrates over the move's time, and
            every transition is then fitted at once.
        dist : ParametricFitter or dict, optional
            The distribution of every transition (default ``Weibull``),
            or a ``{(from, to): distribution}`` dict with one for each.
        clock : ('forward', 'reset'), str, optional
            The time scale of the hazards. ``'forward'`` (the default):
            the time since the start (a Markov model), each spell left
            truncated at its ``tl``. ``'reset'``: the time spent in the
            current state (a semi-Markov model), each transition fitted
            on the durations ``x - tl`` (``tl`` the time the unit entered
            the state; ``x`` where it is missing), with a distribution on
            (0, inf).
        carry_on : array_like of bool, optional
            On the reset clock, the rows that carry on their unit's stay
            from its row before (a stay split into rows by hand): each
            such row is fitted on the time since the stay began, left
            truncated at the time already spent, so the time in the state
            does not restart. A marked row must be the same unit's next
            row, in the same state, starting where the row before was
            censored. By default (``None``) every row is a stay of its
            own, as for a unit renewed in its state (a repair as good as
            new) at the time its row before was censored. Ignored on the
            forward clock, where a row's ``tl`` already says it all.

        tr : array_like or float, optional
            Right truncation: each unit's cut-off (on each of its rows;
            ``inf`` for a unit not truncated). A truncated unit is in the
            data only because it reached the state ``reached`` by then
            (a register of deaths, of failures by a date), and its
            likelihood is divided by its chance of that, given its first
            spell. Needs ``i``; on the forward clock only.
        reached : state, optional
            The state a right-truncated unit had to reach by its cut-off
            (default the absorbing state, where the transitions have
            exactly one).

        A move at some unknown time before ``xr`` (a unit found to have
        moved, with no last time it was seen in its state: left censored)
        is ``x = [-inf, xr]``: the move lies between the start of its
        spell (``tl``, or 0) and ``xr``.

        Each kind of row's factor of the likelihood, for a spell in state
        ``g`` from ``a`` (its ``tl``, or 0), where ``S_g(a, t) = exp(-sum_k
        [H_gk(t) - H_gk(a)])`` over the transitions ``g -> k`` (on the
        forward clock the times since the start; on the reset clock the
        times since the stay began, ``a`` then 0 for a row that starts
        its stay):

        - a move into ``b`` at ``x``: ``S_g(a, x) h_gb(x)``;
        - censored at ``x``: ``S_g(a, x)``;
        - a move into ``b`` in ``(xl, xr]``: ``integral over u in (xl, xr]
          of S_g(a, u) h_gb(u) L_b(u) du``, ``L_b(u)`` the likelihood of
          the unit's next spell given it entered ``b`` at ``u`` (or, with
          no next spell, the chance of staying in ``b`` from ``u`` to
          ``xr``);
        - left censored, a move into ``b`` before ``xr`` (``xl = -inf``):
          the same with ``xl = a``; with two states, ``1 - S(a, xr)``, the
          univariate left-censored (and left-truncated) factor
          ``(F(xr) - F(a)) / S(a)``;
        - a right-truncated unit (cut-off ``tau``, state ``reached`` =
          ``r``): the product of its rows' factors divided by ``P(enter r
          by tau | in g0 at t0)``, ``g0`` and ``t0`` the state and entry
          of its first spell: element ``[g0, r]`` of the transition
          probabilities ``P(t0, tau)`` of the model with every transition
          out of ``r`` removed. With two states, ``1 - S(tau) / S(t0)``,
          so the unit's factor is the univariate right-truncated
          ``f(x) / (F(tau) - F(t0))``.

        Returns
        -------
        ParametricMultiState
            The fitted model.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import (
        ...     ParametricMultiState, TransitionMatrix,
        ... )
        >>> df = load_mgus2(spells=True)
        >>> tm = TransitionMatrix.illness_death(("mgus", "pcm", "death"))
        >>> model = ParametricMultiState.fit(
        ...     df["x"], df["e"], df["state"], tl=df["tl"], i=df["id"],
        ...     transitions=tm,
        ... )
        >>> model.parameter_names[:2]
        ['mgus -> pcm: alpha', 'mgus -> pcm: beta']
        >>> model.occupancy([120, 240], "death").round(3)
        array([0.581, 0.808])

        On the time spent in each state (a semi-Markov model):

        >>> reset = ParametricMultiState.fit(
        ...     df["x"], df["e"], df["state"], tl=df["tl"], i=df["id"],
        ...     transitions=tm, clock="reset",
        ... )
        >>> reset.occupancy([120, 240], "pcm").round(4)
        array([0.0145, 0.0091])
        """
        from surpyval import Weibull

        _check_clock(clock)
        data = check_spells(x, e, state, tl, i, n, transitions, intervals=True)
        data = _with_truncation(data, tr, reached)
        data = _with_carry_on(data, carry_on)
        fitters = _fitters(Weibull if dist is None else dist, data.tm, False)
        _check_left_start(x, tl, fitters)
        Z = np.zeros((data.x.size, 0))
        return cls._fit(data, Z, fitters, clock)

    @classmethod
    @takes_unit
    @other_name("s_col", "state_col")
    def fit_from_df(
        cls,
        df: Any,
        x_col: str | list,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        tr_col: str | None = None,
        reached: Any = None,
        carry_on_col: str | None = None,
    ) -> ParametricMultiState:
        """
        Fit the model to a DataFrame of spells, one row each; the columns
        are the arguments of :meth:`fit` (``s_col`` is another name for
        ``state_col``, ``tr_col`` holds the cut-offs ``tr`` and
        ``carry_on_col`` holds ``carry_on``).

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id", dist=Exponential,
        ... )
        >>> model.transitions.transition_names()
        ['mgus -> death', 'mgus -> pcm', 'pcm -> death']
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        from surpyval.utils.dataframe import frame_column

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        return cls.fit(
            _x_column(df, x_col),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            tl=col(tl_col, "tl_col"),
            i=col(i_col, "i_col"),
            n=col(n_col, "n_col"),
            transitions=transitions,
            dist=dist,
            clock=clock,
            tr=col(tr_col, "tr_col"),
            reached=reached,
            carry_on=col(carry_on_col, "carry_on_col"),
        )

    @classmethod
    @takes_unit
    @other_name("s_col", "state_col")
    def fit_bayes_from_df(
        cls,
        df: Any,
        x_col: str | list,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        *,
        priors: Any = None,
        method: str = "auto",
        points: int | None = None,
        importance: bool = False,
    ) -> Any:
        """
        :meth:`fit_bayes` on a DataFrame of spells, one row each, the
        columns named as :meth:`fit_from_df` names them (``s_col`` is
        another name for ``state_col``). ``priors``, ``method``,
        ``points`` and ``importance`` are :meth:`fit_bayes`'s, passed to
        it unchanged.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import Exponential
        >>> from surpyval.multistate import ParametricMultiState
        >>> spells = pd.DataFrame({
        ...     "t": [2.0, 5.0, 3.0, 7.0, 4.0, 6.0, 1.0, 8.0],
        ...     "to": ["ill", "dead", "dead", None, "ill", "dead", "ill",
        ...            None],
        ...     "from": ["well", "ill", "well", "well", "well", "ill",
        ...              "well", "ill"],
        ... })
        >>> model = ParametricMultiState.fit_bayes_from_df(
        ...     spells, "t", "to", "from", dist=Exponential)
        >>> sorted(model.models)
        [('ill', 'dead'), ('well', 'dead'), ('well', 'ill')]
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas
        from surpyval.utils.dataframe import frame_column

        df = _as_pandas(df)

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        return cls.fit_bayes(
            _x_column(df, x_col),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            tl=col(tl_col, "tl_col"),
            i=col(i_col, "i_col"),
            n=col(n_col, "n_col"),
            transitions=transitions,
            dist=dist,
            clock=clock,
            priors=priors,
            method=method,
            points=points,
            importance=importance,
        )

    @classmethod
    @takes_unit
    @other_name("s", "state")
    @multistate_data
    @accepts_surpyval_data
    def fit_bayes(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        i: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        *,
        priors: Any = None,
        method: str = "auto",
        points: int | None = None,
        importance: bool = False,
        carry_on: npt.ArrayLike | None = None,
    ) -> Any:
        """
        Fit each transition's distribution by Bayes' rule: priors on its
        parameters times the likelihood :meth:`fit` maximises.

        The likelihood is a product of one factor per transition, so
        with priors stated a transition at a time each transition has
        its own posterior, computed exactly as ``dist.fit_bayes``
        computes it on the transition's spells (censored where they
        ended otherwise, left truncated where they began; the grid for
        up to three free parameters: exact). The state occupation
        probabilities come from joint draws of the transitions'
        posteriors.

        Parameters
        ----------
        x, e, state, tl, i, n, transitions, dist, clock
            The spells and the model, as for :meth:`fit`.
        priors : optional
            One set of priors for every transition (as ``dist.fit_bayes``
            takes them), or a set a transition, ``{(from, to): {...}}``
            (a transition left out has none).
        method, points, importance
            As for ``dist.fit_bayes``.

        Returns
        -------
        surpyval.multistate.bayes.BayesianMultiState
            ``models`` (each transition's
            :class:`~surpyval.bayes.BayesianParametric`); ``occupancy``
            and ``length_of_stay`` for a new unit (posterior means) with
            ``occupancy_cri`` and ``length_of_stay_cri``, and
            ``transition_probabilities``.

        Examples
        --------
        >>> import surpyval as sp
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_bayes(
        ...     x=[2, 5, 3, 4, 6, 1, 7, 3, 8, 2], e=["ill", "dead", "dead",
        ...     None, "ill", "dead", None, "ill", None, "dead"],
        ...     state=["healthy", "ill", "healthy", "healthy", "healthy",
        ...     "ill", "ill", "healthy", "ill", "healthy"],
        ...     tl=[0, 2, 0, 0, 0, 0, 6, 0, 3, 0], dist=sp.Exponential,
        ... )
        >>> model.occupancy_cri([2.0, 6.0], "dead").shape
        (2, 2)
        """
        from surpyval.multistate import bayes

        return bayes.fit(
            cls,
            x,
            e,
            state,
            tl,
            i,
            n,
            transitions,
            dist,
            clock,
            priors=priors,
            method=method,
            points=points,
            importance=importance,
            carry_on=carry_on,
        )

    @staticmethod
    def _fit_one(
        fitter: Any,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        extra: dict,
        found: list,
    ) -> Any:
        return fitter.fit(x=x, c=c, n=n, **extra)

    @classmethod
    def _row_hazard(cls, Z: Any) -> Any:
        def hazard(
            model: Any, t: npt.NDArray, rows: npt.NDArray, rate: bool
        ) -> npt.NDArray:
            f = model.hf if rate else model.Hf
            return np.asarray(f(t), dtype=float)

        return hazard

    @staticmethod
    def _Hf(
        model: Any, t: npt.NDArray, z: npt.NDArray, rate: bool = False
    ) -> npt.NDArray:
        f = model.hf if rate else model.Hf
        return np.asarray(f(t), dtype=float)

    @staticmethod
    def _describe(model: Any) -> str:
        return model.dist.name

    # -- predictions --------------------------------------------------------

    def _times(self, x: Any, s: float) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        ok = np.isfinite(flat) & (flat >= s)
        return flat, ok

    def transition_probabilities(
        self, x: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """
        The transition probability matrices :math:`P(s, t)` at the times
        ``x``: element ``[..., a, b]`` is the probability of being in
        state ``b`` at ``x`` for a unit in state ``a`` at ``s`` (by
        default the earliest entry); with ``clock="reset"``, for a unit
        that entered ``a`` at ``s``. NaN before ``s``.

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit(
        ...     x=[2, 5, 3, 4, 6, 1, 7], e=["ill", "dead", "dead", None,
        ...     "ill", "dead", None], state=["healthy", "ill", "healthy",
        ...     "healthy", "healthy", "ill", "ill"], tl=[0, 2, 0, 0, 0, 0,
        ...     6], dist=Exponential,
        ... )
        >>> model.transition_probabilities(2.0).round(3)
        array([[1.   , 0.   , 0.   ],
               [0.182, 0.67 , 0.147],
               [0.551, 0.   , 0.449]])
        """
        s = self._start(s)
        k = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        flat, ok = self._times(xs, s)
        out = np.full((flat.size, k, k), np.nan)
        out[ok] = self._P_at(np.zeros(0), s, flat[ok])
        return out.reshape(xs.shape + (k, k))

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The state occupation probability: the probability of being in
        ``state`` at ``x``.

        Parameters
        ----------
        x : array_like
            The times.
        state : any
            The state.
        s : float, optional
            The time the prediction starts from. Defaults to the earliest
            entry (0 where spells are observed from the start). With
            ``clock="reset"`` a later start needs ``from_state``: each
            unit is taken to have just entered its state at ``s``.
        from_state : any, optional
            The unit's state at ``s``. By default the states of the units
            under observation at ``s``, in proportion (as for
            :meth:`MultiState.occupancy
            <surpyval.multistate.MultiState.occupancy>`).

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.occupancy([120, 240, 480], "pcm").round(4)
        array([0.0143, 0.0083, 0.0022])
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        flat, ok = self._times(x, s)
        out = np.full(flat.size, np.nan)
        out[ok] = self._occupancy_values(np.zeros(0), s, p0, b, flat[ok])
        return out

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The standard error of :meth:`occupancy`, by the delta method over
        every transition's parameters (the starting distribution taken
        as known).

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.occupancy_se(240, "death").round(4)
        np.float64(0.0114)
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        flat, ok = self._times(x, s)
        out = np.full(flat.size, np.nan)
        out[ok] = self._occupancy_se_values(np.zeros(0), s, p0, b, flat[ok])
        return out

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "log-log",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on the state occupation probability
        :meth:`occupancy`, from the delta-method standard error of
        :meth:`occupancy_se`.

        Parameters
        ----------
        x, state, s, from_state
            As for :meth:`occupancy`.
        alpha_ci : float, optional
            One minus the confidence level (default 0.05).
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Which bounds.
        bound_type : ('log-log', 'plain'), str, optional
            ``'log-log'`` (the default) works on the log(-log) scale and
            keeps the bounds in (0, 1); ``'plain'`` is the estimate plus
            and minus ``z`` standard errors. (``'exp'`` and ``'normal'``
            until v0.30, still accepted with a warning.)

        Notes
        -----
        In simulation the default bounds hold their level for common
        and rarely occupied states alike once each transition has a few
        dozen observed moves. With fewer (a rare state in about 100
        units, some 10 to 35 moves out of it) they are two to four
        points short of 90% and 95%, the band more often below the
        truth than above it; ``'normal'`` is further short there. The
        study is ``scripts/studies/multistate_occupancy`` in the
        repository.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.cb(240, "death").round(3)
        array([0.785, 0.829])
        """
        bound_type = check_bound_args(alpha_ci, bound, bound_type, self)
        p = self.occupancy(x, state, s, from_state)
        se = self.occupancy_se(x, state, s, from_state)
        return probability_bounds(
            np.ravel(p), np.ravel(se), alpha_ci, bound, bound_type
        )

    @keeps_query_shape
    def length_of_stay(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        r"""
        The expected length of stay: the expected time spent in ``state``
        between ``s`` and ``x``, the area under :meth:`occupancy` over
        that time,

        .. math::
            L_b(s, x) = \int_s^x P(\text{in } b \text{ at } u)\, du.

        Over every state it adds up to ``x - s``. ``s`` and
        ``from_state`` are as for :meth:`occupancy`; NaN before ``s``.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.length_of_stay(240, "mgus").round(1)
        np.float64(110.4)
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        flat, ok = self._times(x, s)
        out = np.full(flat.size, np.nan)
        out[ok] = self._occupancy_values(
            np.zeros(0), s, p0, b, flat[ok], "los"
        )
        return out

    @keeps_query_shape
    def length_of_stay_se(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The standard error of :meth:`length_of_stay`, by the delta method
        over every transition's parameters (the starting distribution
        taken as known).
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        flat, ok = self._times(x, s)
        out = np.full(flat.size, np.nan)
        out[ok] = self._occupancy_se_values(
            np.zeros(0), s, p0, b, flat[ok], "los"
        )
        return out

    @keeps_query_shape
    def length_of_stay_cb(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on :meth:`length_of_stay`: the Wald
        bound on the logit of the fraction of the time ``x - s`` spent in
        the state, from :meth:`length_of_stay_se`, so the bounds stay
        between 0 and ``x - s`` (as the restricted mean survival time's
        bounds do).

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.length_of_stay_cb(240, "mgus").round(1)
        array([105.7, 115. ])
        """
        check_bound_args(alpha_ci, bound, "log-log")
        x_arr = np.ravel(np.asarray(x, dtype=float))
        start = self._start(s)
        value = np.ravel(self.length_of_stay(x_arr, state, s, from_state))
        se = np.ravel(self.length_of_stay_se(x_arr, state, s, from_state))
        return time_fraction_bounds(value, se, x_arr - start, alpha_ci, bound)

    @keeps_query_shape
    def cumulative_hazard(
        self, x: npt.ArrayLike, from_state: Any, to_state: Any
    ) -> npt.NDArray:
        """
        The cumulative hazard of the transition from ``from_state`` to
        ``to_state`` at ``x`` (with ``clock="reset"``, ``x`` is the time
        spent in ``from_state``).

        Examples
        --------
        >>> from surpyval import Exponential
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id", dist=Exponential,
        ... )
        >>> model.cumulative_hazard(120, "mgus", "pcm").round(4)
        np.float64(0.1066)
        """
        self.transitions.transition_index(from_state, to_state)
        model = self.models[(self._label(from_state), self._label(to_state))]
        return np.asarray(model.Hf(np.asarray(x, dtype=float)), dtype=float)

    @keeps_query_shape
    def hazard(
        self, x: npt.ArrayLike, from_state: Any, to_state: Any
    ) -> npt.NDArray:
        """The hazard of the transition from ``from_state`` to
        ``to_state`` at ``x``."""
        self.transitions.transition_index(from_state, to_state)
        model = self.models[(self._label(from_state), self._label(to_state))]
        return np.asarray(model.hf(np.asarray(x, dtype=float)), dtype=float)

    def _label(self, state: Any) -> Any:
        return self.transitions.states[self.transitions.index(state)]

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        spells (for the starting distributions of predictions) and each
        transition's fitted distribution (its own ``to_dict``, with its
        covariance). The reloaded model predicts, and bounds, exactly as
        the original.
        """
        return stamp_schema(
            {"model": "ParametricMultiState", **self._data_dict()}
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> ParametricMultiState:
        """Rebuild the model from :meth:`to_dict`."""
        from surpyval.univariate.parametric.parametric import Parametric

        require_model_tag(
            model_dict,
            "ParametricMultiState",
            "a parametric multi-state model",
        )
        return cls._from_data_dict(model_dict, Parametric)

    def __repr__(self) -> str:
        return "\n".join(
            self._repr_lines(
                "Parametric multi-state model (a distribution per transition)"
            )
        )


class ParametricMultiStateRegression(_ParametricMultiStateBase):
    r"""
    A parametric multi-state regression: a parametric regression (a
    proportional-hazards, accelerated-failure-time or other parametric
    regression model) for each transition, and the transition and state
    occupation probabilities they predict for given covariates.

    Each transition's regression is fitted to the spells in its 'from'
    state with their covariates, the other spells censored and each spell
    left truncated at its start, as :class:`ParametricMultiState` fits
    its distributions (or, with ``clock="reset"``, on the time spent in
    the state); every transition has its own coefficients. The
    probabilities for a covariate row :math:`z` solve the Kolmogorov
    forward equations with the hazards :math:`h_{gk}(t \mid z)`, and
    their bounds come from the delta method over every transition's
    parameters.

    Call the class method ``ParametricMultiStateRegression.fit`` (or
    ``fit_from_df``); it returns a fitted instance.
    """

    _regression = True

    #: The covariates' names (``z0``, ``z1``, ... unless named).
    covariate_names: list

    @classmethod
    @takes_unit
    @other_name("s", "state")
    @multistate_data(intervals=True)
    @accepts_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        i: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        time_varying_covariates: dict | None = None,
        tr: npt.ArrayLike | float | None = None,
        reached: Any = None,
        carry_on: npt.ArrayLike | None = None,
    ) -> ParametricMultiStateRegression:
        """
        Fit a parametric regression to each transition of spells in
        states.

        Parameters
        ----------
        x, e, state, tl, i, n, transitions, clock, tr, reached, carry_on
            As for :meth:`ParametricMultiState.fit` (a right-truncated
            unit's chance of reaching the state taken with its
            covariates, which are then the same on each of its rows):
            ``x`` may be a
            :class:`~surpyval.SurpyvalData` holding the spells and their
            covariates ``Z`` (named by its ``covariate_names``), and
            ``s`` is another name for ``state``.
        Z : array_like
            The covariates, one row per spell (a unit's covariates on each
            of its spells). Missing values are refused: dropping a spell
            would cut a unit's path through its states. ``None`` where
            every covariate is in ``time_varying_covariates``.
        model : regression fitter or dict, optional
            The parametric regression of every transition (default
            ``WeibullPH``), such as ``ExponentialPH``, ``WeibullAFT`` or
            ``LogNormalAFT``, or a ``{(from, to): fitter}`` dict with one
            for each.
        time_varying_covariates : dict, optional
            Covariates that change during a stay: ``{unit: path}``, one
            :class:`~surpyval.StepSchedule` (or
            :class:`~surpyval.CovariatePath`) for each unit in ``i``, on
            the clock's time since the start. Each spell is split at its
            unit's changes into rows with the covariates fixed (on the
            reset clock the time in the state carries on across them),
            and they are the columns ``time_varying_0``, ... after those
            of ``Z``. Every transition's regression must have a hazard
            that depends only on the covariates at the time
            (proportional hazards, additive hazards or proportional
            odds); accelerated failure time is refused, and so are
            interval-censored moves. A ``CovariatePath`` is cut into
            steps, halved until no estimate moves by more than 1% of its
            standard error (``path_steps`` says what that took). The
            path is taken as external: known in advance, not driven by
            the unit's own moves.

        Returns
        -------
        ParametricMultiStateRegression
            The fitted model.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import (
        ...     ParametricMultiStateRegression, TransitionMatrix,
        ... )
        >>> df = load_mgus2(spells=True)
        >>> tm = TransitionMatrix.illness_death(("mgus", "pcm", "death"))
        >>> model = ParametricMultiStateRegression.fit(
        ...     df["x"], df[["male"]], df["e"], df["state"], tl=df["tl"],
        ...     i=df["id"], transitions=tm,
        ... )
        >>> model.parameter_names[:3]
        ['mgus -> pcm: alpha', 'mgus -> pcm: beta', 'mgus -> pcm: male']
        >>> model.occupancy([120, 120], [[1], [0]], "pcm").round(4)
        array([0.0128, 0.0161])
        """
        from surpyval import WeibullPH

        _check_clock(clock)
        data = check_spells(x, e, state, tl, i, n, transitions, intervals=True)
        data = _with_truncation(data, tr, reached)
        if time_varying_covariates is not None and data.tr is not None:
            raise ValueError(
                "Right truncation (tr) is not taken with time-varying "
                "covariates: the chance of reaching the state by the "
                "cut-off would need each unit's path. Give fixed "
                "covariates, or leave the truncation out."
            )
        data = _with_carry_on(data, carry_on)
        if Z is None and time_varying_covariates is not None:
            Z_arr, names = np.zeros((data.x.size, 0)), []
        else:
            Z_arr, names = _covariates(Z, data.x.size)
        fitters = _fitters(
            WeibullPH if model is None else model, data.tm, True
        )
        _check_left_start(x, tl, fitters)
        if time_varying_covariates is not None:
            return cls._fit_varying(
                data, Z_arr, names, fitters, clock, time_varying_covariates, i
            )
        out = cls._fit_named(data, Z_arr, names, fitters, clock)
        return out

    @classmethod
    def _fit_varying(
        cls,
        data: SpellData,
        Z: npt.NDArray,
        names: list,
        fitters: list,
        clock: str,
        paths: dict,
        i: Any,
    ) -> ParametricMultiStateRegression:
        """The fit with each unit's covariate path: every spell split at
        the changes (see :mod:`surpyval.multistate._tvc`)."""
        from surpyval.univariate.regression.tvc_path import CovariatePath
        from surpyval.utils.surpyval_data import (
            _PATH_STEPS,
            _refined_on_paths,
        )

        what = "time_varying_covariates"
        _tvc.check_families(fitters, list(data.tm.transitions), what)
        if data.has_intervals:
            raise ValueError(
                "time_varying_covariates cannot be taken with "
                "interval-censored moves (x as [xl, xr]): the move's "
                "unknown time would fall among the changes. Give each "
                "move at a time, or leave the time-varying covariates out."
            )
        units = _tvc.unit_paths(paths, i, data)
        width = units[0][1].p if units else 0
        all_names = list(names) + [f"time_varying_{j}" for j in range(width)]
        low = min(_lows(fitters)) if clock == "forward" else 0.0
        low = low if np.isfinite(low) else -np.inf

        def fitted(steps: int) -> tuple:
            split, Z_split = _tvc.split(data, Z, units, low, steps)
            model = cls._fit_named(split, Z_split, all_names, fitters, clock)
            return model, int(split.x.size)

        if any(isinstance(p, CovariatePath) for _, p in units):
            model, count = fitted(_PATH_STEPS)
            return _refined_on_paths(fitted, model, count)
        return fitted(_PATH_STEPS)[0]

    @classmethod
    @takes_unit
    @other_name("s", "state")
    def fit_bayes(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        i: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        *,
        priors: Any = None,
        method: str = "auto",
        importance: bool = False,
        carry_on: npt.ArrayLike | None = None,
    ) -> Any:
        """
        Fit each transition's regression by Bayes' rule: priors on its
        coefficients and baseline parameters times the likelihood
        :meth:`fit` maximises.

        The likelihood is a product of one factor per transition, so
        with priors stated a transition at a time each transition has
        its own posterior, computed exactly as its fitter's
        ``fit_bayes`` computes it on the transition's spells and their
        covariates (censored where they ended otherwise, left truncated
        where they began; the Laplace engine). The predictions are for
        one covariate row per call, from joint draws of the
        transitions' posteriors.

        Parameters
        ----------
        x, Z, e, state, tl, i, n, transitions, model, clock
            The spells, their covariates and the model, as for
            :meth:`fit`; exact moves only (a move known only to a window,
            ``x`` of two columns, is refused).
        priors : optional
            One set of priors for every transition (as the fitter's
            ``fit_bayes`` takes them), or a set a transition,
            ``{(from, to): {...}}`` (a transition left out has the
            fitter's defaults).
        method, importance
            As for the fitter's ``fit_bayes``.

        Returns
        -------
        surpyval.multistate.bayes.BayesianMultiStateRegression
            ``models`` (each transition's
            :class:`~surpyval.bayes.BayesianRegression`);
            ``occupancy(x, Z, state)`` and ``length_of_stay`` for a new
            unit with the covariate row ``Z`` (posterior means), with
            ``occupancy_cri`` and ``length_of_stay_cri``, and
            ``transition_probabilities(x, Z)``.

        Examples
        --------
        >>> from surpyval import ExponentialPH
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> df = load_mgus2(spells=True).iloc[::4]
        >>> model = ParametricMultiStateRegression.fit_bayes(
        ...     df["x"], df[["male"]], df["e"], df["state"], tl=df["tl"],
        ...     i=df["id"], model=ExponentialPH)
        >>> model.occupancy_cri([60.0, 120.0], [1], "death").shape
        (2, 2)
        """
        from surpyval.multistate import bayes

        return bayes.fit_regression(
            cls,
            x,
            Z,
            e,
            state,
            tl,
            i,
            n,
            transitions,
            model,
            clock,
            priors=priors,
            method=method,
            importance=importance,
            carry_on=carry_on,
        )

    @classmethod
    def _fit_named(
        cls,
        data: SpellData,
        Z: npt.NDArray,
        names: list,
        fitters: list,
        clock: str,
    ) -> ParametricMultiStateRegression:
        import pandas as pd

        frame = pd.DataFrame(Z, columns=names)
        out = cls._fit(data, frame, fitters, clock)
        out.covariate_names = list(names)
        out._Z = Z
        return out

    @classmethod
    @takes_unit
    @other_name("s_col", "state_col")
    def fit_from_df(
        cls,
        df: Any,
        x_col: str | list,
        Z_cols: Any,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        time_varying_covariates: dict | None = None,
        tr_col: str | None = None,
        reached: Any = None,
        carry_on_col: str | None = None,
    ) -> ParametricMultiStateRegression:
        """
        Fit the model to a DataFrame of spells, one row each; ``Z_cols``
        names the covariate columns (``None`` where every covariate is in
        ``time_varying_covariates``, keyed by the values of ``i_col``),
        and the other columns are the arguments of :meth:`fit` (``s_col``
        is another name for ``state_col``).

        Examples
        --------
        >>> from surpyval import ExponentialPH
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id", model=ExponentialPH,
        ... )
        >>> model.covariate_names
        ['male']
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        from surpyval.utils.dataframe import frame_column

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        cols = (
            []
            if Z_cols is None
            else [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        )
        _check_clock(clock)
        if time_varying_covariates is not None:
            return cls.fit(
                _x_column(df, x_col),
                (
                    None
                    if not cols
                    else df[cols]
                    .astype(float)
                    .set_axis([str(c) for c in cols], axis=1)
                ),
                col(e_col, "e_col"),
                col(state_col, "state_col"),
                tl=col(tl_col, "tl_col"),
                i=col(i_col, "i_col"),
                n=col(n_col, "n_col"),
                transitions=transitions,
                model=model,
                clock=clock,
                time_varying_covariates=time_varying_covariates,
                tr=col(tr_col, "tr_col"),
                reached=reached,
                carry_on=col(carry_on_col, "carry_on_col"),
            )
        Z = np.column_stack([frame_column(df, c, "Z_cols") for c in cols])
        data = check_spells(
            _x_column(df, x_col),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            col(tl_col, "tl_col"),
            col(i_col, "i_col"),
            col(n_col, "n_col"),
            transitions,
            intervals=True,
        )
        data = _with_carry_on(data, col(carry_on_col, "carry_on_col"))
        Z_arr, _ = _covariates(Z, data.x.size)
        from surpyval import WeibullPH

        fitters = _fitters(
            WeibullPH if model is None else model, data.tm, True
        )
        _check_left_start(_x_column(df, x_col), col(tl_col, "tl_col"), fitters)
        return cls._fit_named(
            data, Z_arr, [str(c) for c in cols], fitters, clock
        )

    @classmethod
    @takes_unit
    @other_name("s_col", "state_col")
    def fit_bayes_from_df(
        cls,
        df: Any,
        x_col: str | list,
        Z_cols: Any,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        *,
        priors: Any = None,
        method: str = "auto",
        importance: bool = False,
    ) -> Any:
        """
        :meth:`fit_bayes` on a DataFrame of spells, one row each, the
        columns named as :meth:`fit_from_df` names them (``Z_cols`` the
        covariate columns; ``s_col`` is another name for
        ``state_col``). ``priors``, ``method`` and ``importance`` are
        :meth:`fit_bayes`'s, passed to it unchanged; a coefficient is
        named in ``priors`` by its column (``{"age": ...}``), as
        :meth:`fit_from_df` names the covariates.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import ExponentialPH
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> spells = pd.DataFrame({
        ...     "t": [2.0, 5.0, 3.0, 7.0, 4.0, 6.0, 1.0, 8.0, 3.5, 2.5],
        ...     "to": ["ill", "dead", "ill", None, "ill", "dead", "ill",
        ...            None, "ill", "dead"],
        ...     "from": ["well", "ill", "well", "well", "well", "ill",
        ...              "well", "ill", "well", "ill"],
        ...     "age": [50, 62, 55, 48, 70, 66, 59, 61, 53, 68],
        ... })
        >>> model = ParametricMultiStateRegression.fit_bayes_from_df(
        ...     spells, "t", ["age"], "to", "from", model=ExponentialPH)
        >>> sorted(model.models)
        [('ill', 'dead'), ('well', 'ill')]
        >>> model.models[("well", "ill")].parameter_names
        ['failure_rate', 'age']
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas
        from surpyval.utils.dataframe import frame_column

        df = _as_pandas(df)

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        import pandas as pd

        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        # (a frame with the columns' names: each transition's coefficients
        # are named by them, as fit_from_df names the covariates)
        Z = pd.DataFrame(
            {
                str(c): np.asarray(frame_column(df, c, "Z_cols"), float)
                for c in cols
            }
        )
        return cls.fit_bayes(
            _x_column(df, x_col),
            Z,
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            tl=col(tl_col, "tl_col"),
            i=col(i_col, "i_col"),
            n=col(n_col, "n_col"),
            transitions=transitions,
            model=model,
            clock=clock,
            priors=priors,
            method=method,
            importance=importance,
        )

    @staticmethod
    def _fit_one(
        fitter: Any,
        x: npt.NDArray,
        Z: Any,
        c: npt.NDArray,
        n: npt.NDArray,
        extra: dict,
        found: list,
    ) -> Any:
        from surpyval.univariate.regression._aliasing import collect_aliased

        with collect_aliased() as aliased:
            out = fitter.fit(x, Z.reset_index(drop=True), c=c, n=n, **extra)
        found += aliased
        return out

    @staticmethod
    def _warn_found(found: list) -> None:
        from surpyval.univariate.regression._aliasing import warn_collected

        warn_collected(found, "in the fit of each transition")

    @classmethod
    def _row_hazard(cls, Z: Any) -> Any:
        Z_arr = np.asarray(Z, dtype=float)

        def hazard(
            model: Any, t: npt.NDArray, rows: npt.NDArray, rate: bool
        ) -> npt.NDArray:
            f = model.hf if rate else model.Hf
            return np.asarray(f(t, Z_arr[rows]), dtype=float)

        return hazard

    @staticmethod
    def _Hf(
        model: Any, t: npt.NDArray, z: npt.NDArray, rate: bool = False
    ) -> npt.NDArray:
        Z = np.repeat(z[None, :], t.size, axis=0)
        f = model.hf if rate else model.Hf
        return np.asarray(f(t, Z), dtype=float)

    @staticmethod
    def _describe(model: Any) -> str:
        return f"{model.distribution.name} ({model.kind})"

    @property
    def betas(self) -> npt.NDArray:
        """``(n_transitions, p)``: each transition's covariate
        coefficients, one per column of ``Z`` (NaN for an aliased one),
        in the order of ``transitions.transitions``."""
        p = len(self.covariate_names)
        return np.vstack(
            [
                np.asarray(m.params, dtype=float).ravel()[len(m.params) - p :]
                for m in self._models()
            ]
        ).reshape(self.transitions.n_transitions, p)

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficient the data cannot
        determine for some transition."""
        cols: set = set()
        for m in self._models():
            cols |= {int(j) for j in np.ravel(getattr(m, "aliased", []))}
        return np.array(sorted(cols), dtype=int)

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        The parameter table of every transition, stacked: each
        transition's own regression ``summary`` under its name
        (``"<from> -> <to>"``) as the first level of the index.
        """
        import pandas as pd

        from surpyval.utils.validation import check_alpha_ci

        check_alpha_ci(alpha_ci)
        names = self.transitions.transition_names()
        table = pd.concat(
            {t: m.summary(alpha_ci) for t, m in zip(names, self._models())},
            names=["transition"],
        )
        if self._joint is not None:
            table = self._joint_summary(table, alpha_ci)
        return table

    def _joint_summary(self, table: Any, alpha_ci: float) -> Any:
        """The table of :meth:`summary` with the standard errors, bounds
        and tests of the joint fit (where moves are interval censored):
        a positive parameter's bounds on the log scale, a coefficient's
        on its own."""
        from scipy.stats import norm

        from surpyval.multistate._joint import Packer

        pack = Packer.of(self._models())
        se = self.standard_errors()
        z = norm.ppf(1 - alpha_ci / 2)
        coef = table["coef"].to_numpy(dtype=float)
        if coef.size != se.size:
            return table
        cols = list(table.columns)
        lo_col = [c for c in cols if c.startswith("coef lower")][0]
        hi_col = [c for c in cols if c.startswith("coef upper")][0]
        elo = [c for c in cols if c.startswith("exp(coef) lower")][0]
        ehi = [c for c in cols if c.startswith("exp(coef) upper")][0]
        is_coef = table.index.get_level_values("part") == "coefficients"
        out = table.copy()
        positive = (pack.lower == 0) & ~np.isfinite(pack.upper)
        with np.errstate(all="ignore"):
            w = z * se / np.abs(coef)
            lower = np.where(
                positive & ~is_coef, coef * np.exp(-w), coef - z * se
            )
            upper = np.where(
                positive & ~is_coef, coef * np.exp(w), coef + z * se
            )
            stat = coef / se
        out["se(coef)"] = se
        out[lo_col] = lower
        out[hi_col] = upper
        out[elo] = np.where(is_coef, np.exp(lower), np.nan)
        out[ehi] = np.where(is_coef, np.exp(upper), np.nan)
        out["z"] = np.where(is_coef, stat, np.nan)
        out["p"] = np.where(is_coef, 2 * norm.sf(np.abs(stat)), np.nan)
        return out

    # -- predictions --------------------------------------------------------

    def _z(self, Z: Any) -> Any:
        if _tvc.is_path(Z):
            return self._checked_path(Z)
        z = np.asarray(Z, dtype=float).ravel()
        if z.size != len(self.covariate_names):
            raise ValueError(
                f"Z must be one row of {len(self.covariate_names)} "
                "covariate(s)."
            )
        return z

    def _checked_path(self, path: Any, s: float | None = None) -> Any:
        """A covariate path for a prediction, or a refusal saying why
        not."""
        p = len(self.covariate_names)
        if path.p != p:
            raise ValueError(
                f"The covariate path gives {path.p} covariate(s); the "
                f"model has {p} ({', '.join(self.covariate_names)}): give "
                "every covariate in the path (a constant column for one "
                "that does not change)."
            )
        if self.clock == "reset":
            raise ValueError(
                "With clock='reset' a prediction takes each state's "
                "hazards as functions of the time in it alone, but along a "
                "covariate path they depend on when the state was entered "
                "too, which the semi-Markov solution cannot take. Predict "
                "for fixed covariates (one row, or one row per time), or "
                "simulate the paths."
            )
        for pair, m in zip(self.transitions.transitions, self._models()):
            if getattr(m, "kind", None) not in _tvc.ADDITIVE_KINDS:
                raise ValueError(
                    "A prediction along a covariate path takes each "
                    "transition's hazard at a time with the covariates "
                    f"then, but {pair!r} is a {m.kind} regression, whose "
                    "hazard depends on the path before it. Predict for "
                    "fixed covariates, or fit that transition with a "
                    "proportional hazards, additive hazards or "
                    "proportional odds regression."
                )
        if s is not None and s < self._path_origin(path):
            raise ValueError(
                f"The covariate path starts at {path.start:g}, after the "
                f"start of the prediction, s = {s:g}: give the path from "
                "s on, or predict from a later s."
            )
        return path

    def _rows(self, Z: Any, size: int) -> npt.NDArray:
        p = len(self.covariate_names)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim <= 1:
            Z = Z.reshape(1, -1)
        if Z.shape[1] != p:
            raise ValueError(
                f"Z must have {p} covariate column(s); got {Z.shape[1]}."
            )
        if Z.shape[0] == 1:
            return np.repeat(Z, size, axis=0)
        if Z.shape[0] != size:
            raise ValueError(
                "Z must be one covariate row for every time, or one row "
                f"per time ({size}); got {Z.shape[0]} rows."
            )
        return Z

    def _by_row(self, x: Any, Z: Any, s: float, value: Any) -> npt.NDArray:
        """``value(z, times)`` for each distinct covariate row, at its
        query times; NaN for a row with a missing covariate or a time
        before ``s``."""
        x = np.asarray(x, dtype=float).ravel()
        if _tvc.is_path(Z):
            path = self._checked_path(Z, s if np.isfinite(s) else None)
            out = np.full(x.size, np.nan)
            ok = np.isfinite(x) & (x >= max(s, self._path_origin(path)))
            if np.any(ok):
                out[ok] = value(path, x[ok])
            return out
        rows = self._rows(Z, x.size)
        out = np.full(x.size, np.nan)
        bad = ~np.isfinite(x) | (x < s) | np.any(np.isnan(rows), axis=1)
        uniq, inverse = np.unique(
            np.where(np.isnan(rows), 0.0, rows), axis=0, return_inverse=True
        )
        inverse = np.asarray(inverse).ravel()
        for u in range(uniq.shape[0]):
            sel = (inverse == u) & ~bad
            if np.any(sel):
                out[sel] = value(uniq[u], x[sel])
        return out

    def transition_probabilities(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """
        The predicted transition probability matrices :math:`P(s, t \\mid
        z)` at the times ``x`` for one covariate row ``Z``: element
        ``[..., a, b]`` is the probability of being in state ``b`` at
        ``x`` for a unit in state ``a`` at ``s``.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.transition_probabilities(120, [1]).round(3)
        array([[1.   , 0.   , 0.   ],
               [0.615, 0.372, 0.013],
               [0.985, 0.   , 0.015]])
        """
        s = self._start(s)
        z = self._z(Z)
        if _tvc.is_path(z):
            self._checked_path(z, s)
        k = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        flat = xs.ravel()
        ok = np.isfinite(flat) & (flat >= s)
        out = np.full((flat.size, k, k), np.nan)
        out[ok] = self._P_at(z, s, flat[ok])
        return out.reshape(xs.shape + (k, k))

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The predicted state occupation probability: the probability of
        being in ``state`` at ``x`` for a unit with covariates ``Z`` (one
        row for every time, or one row per time). ``s`` and
        ``from_state`` are as for :meth:`ParametricMultiState.occupancy`.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.occupancy([120, 120], [[1], [0]], "death").round(3)
        array([0.615, 0.542])
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
            return self._occupancy_values(z, s, p0, b, times)

        return self._by_row(x, Z, s, value)

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The standard error of :meth:`occupancy`, by the delta method over
        every transition's parameters (the starting distribution taken as
        known).

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.occupancy_se(240, [1], "death").round(4)
        np.float64(0.0137)
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
            return self._occupancy_se_values(z, s, p0, b, times)

        return self._by_row(x, Z, s, value)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "log-log",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on the predicted state occupation
        probability :meth:`occupancy`, from the delta-method standard
        error of :meth:`occupancy_se`; ``alpha_ci``, ``bound`` and
        ``bound_type`` as for :meth:`ParametricMultiState.cb`.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.cb(240, [1], "death").round(3)
        array([0.807, 0.86 ])
        """
        bound_type = check_bound_args(alpha_ci, bound, bound_type, self)
        p = self.occupancy(x, Z, state, s, from_state)
        se = self.occupancy_se(x, Z, state, s, from_state)
        return probability_bounds(
            np.ravel(p), np.ravel(se), alpha_ci, bound, bound_type
        )

    @keeps_query_shape
    def length_of_stay(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The predicted expected length of stay: the expected time a unit
        with covariates ``Z`` spends in ``state`` between ``s`` and ``x``
        (:meth:`ParametricMultiState.length_of_stay`); ``Z``, ``s`` and
        ``from_state`` as for :meth:`occupancy`.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.length_of_stay([240, 240], [[1], [0]], "mgus").round(1)
        array([103.9, 117.9])
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
            return self._occupancy_values(z, s, p0, b, times, "los")

        return self._by_row(x, Z, s, value)

    @keeps_query_shape
    def length_of_stay_se(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """The standard error of :meth:`length_of_stay`, by the delta
        method over every transition's parameters (the starting
        distribution taken as known)."""
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
            return self._occupancy_se_values(z, s, p0, b, times, "los")

        return self._by_row(x, Z, s, value)

    @keeps_query_shape
    def length_of_stay_cb(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """Pointwise confidence bounds on :meth:`length_of_stay`, as
        :meth:`ParametricMultiState.length_of_stay_cb` (the Wald bound on
        the logit of the fraction of ``x - s`` spent in the state)."""
        check_bound_args(alpha_ci, bound, "log-log")
        x_arr = np.ravel(np.asarray(x, dtype=float))
        start = self._start(s)
        value = np.ravel(self.length_of_stay(x_arr, Z, state, s, from_state))
        se = np.ravel(self.length_of_stay_se(x_arr, Z, state, s, from_state))
        return time_fraction_bounds(value, se, x_arr - start, alpha_ci, bound)

    @keeps_query_shape
    def cumulative_hazard(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        from_state: Any,
        to_state: Any,
    ) -> npt.NDArray:
        """
        The predicted cumulative hazard of the transition from
        ``from_state`` to ``to_state`` at ``x`` for covariates ``Z``.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiStateRegression
        >>> model = ParametricMultiStateRegression.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", i_col="id",
        ... )
        >>> model.cumulative_hazard(120, [1], "mgus", "pcm").round(4)
        np.float64(0.1008)
        """
        h = self.transitions.transition_index(from_state, to_state)
        m = self._models()[h]

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
            if _tvc.is_path(z):
                H = _tvc.path_cumhaz([m], z, self._path_origin(z))
                return np.ravel(H(times, False))
            return np.ravel(self._Hf(m, times, z))

        return self._by_row(x, Z, -np.inf, value)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        spells and their covariates (for the starting distributions of
        predictions) and each transition's fitted regression (its own
        ``to_dict``, with its covariance). The reloaded model predicts,
        and bounds, exactly as the original.
        """
        out = {
            "model": "ParametricMultiStateRegression",
            **self._data_dict(),
            "Z": self._Z.tolist(),
            "covariate_names": list(self.covariate_names),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> ParametricMultiStateRegression:
        """Rebuild the model from :meth:`to_dict`."""
        from surpyval.univariate.regression.parametric_regression_model import (  # noqa: E501
            ParametricRegressionModel,
        )

        require_model_tag(
            model_dict,
            "ParametricMultiStateRegression",
            "a parametric multi-state regression",
        )
        out = cls._from_data_dict(model_dict, ParametricRegressionModel)
        out.covariate_names = list(model_dict["covariate_names"])
        out._Z = np.array(model_dict["Z"], dtype=float).reshape(
            len(model_dict["x"]), -1
        )
        return out

    def __repr__(self) -> str:
        lines = self._repr_lines(
            "Parametric multi-state regression (a parametric regression "
            "per transition)"
        )
        lines.insert(3, f"Covariates: {', '.join(self.covariate_names)}")
        return "\n".join(lines)
