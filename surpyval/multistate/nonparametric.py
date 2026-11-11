"""The non-parametric multi-state model: Aalen-Johansen transition
probabilities and state occupation probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate import _core
from surpyval.multistate._base import (
    check_bound_args,
    probability_bounds,
    step_index,
)
from surpyval.multistate._data import SpellData, check_spells
from surpyval.multistate.transitions import TransitionMatrix
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

#: The variance estimators of the non-parametric model.
_VARIANCES = ("greenwood", "aalen", "robust")


class MultiState(SerialisableMixin):
    r"""
    Non-parametric multi-state model: the Aalen-Johansen estimate of the
    probabilities of moving between states.

    Each unit moves through a set of states (see
    :class:`~surpyval.multistate.TransitionMatrix`), and the data are its
    *spells*: the time it spent in one state, from its entry ``tl`` to its
    exit ``x``, and the state it moved into at ``x`` (or none, if it was
    last seen then). From the numbers at risk in each state and the
    transitions seen at each time, the Nelson-Aalen increments
    :math:`dA_{gh}(t_j) = d_{gh}(t_j) / r_g(t_j)` give the transition
    probability matrix by the product integral

    .. math::
        P(s, t) = \prod_{s < t_j \le t} \left(I + dA(t_j)\right),

    whose element :math:`(a, b)` is the probability of being in state
    :math:`b` at :math:`t` for a unit in state :math:`a` at :math:`s`
    (Aalen and Johansen, 1978). It assumes the process is Markov: where a
    unit goes next depends on its state and the time, not on how it got
    there. The state occupation probabilities ``occupancy`` are a row of
    it, or the rows averaged over the states the units were in at
    :math:`s`; they still estimate the population's occupation
    probabilities when the process is not Markov (Datta and Satten,
    2001).

    With one transient state and absorbing states for the causes, this is
    the competing-risks cumulative incidence of
    :class:`~surpyval.univariate.competing_risks.CompetingRisks`; with one
    transition, one minus the Kaplan-Meier estimate.

    A spell is at risk in its state at :math:`t` when :math:`tl < t \le
    x`, so a unit observed from part-way through its history (delayed
    entry, left truncation) counts only from its entry.

    Call the class method ``MultiState.fit`` (or ``fit_from_df``); it
    returns a fitted instance.

    References
    ----------
    Aalen, O. O. and Johansen, S. (1978), "An empirical transition matrix
    for non-homogeneous Markov chains based on censored observations",
    *Scandinavian Journal of Statistics*, 5(3), 141-150.

    Datta, S. and Satten, G. A. (2001), "Validity of the Aalen-Johansen
    estimators of stage occupation probabilities and Nelson-Aalen
    estimators of integrated transition hazards for non-Markov models",
    *Statistics & Probability Letters*, 55(4), 403-411.

    de Wreede, L. C., Fiocco, M. and Putter, H. (2010), "The mstate
    package for estimation and prediction in non- and semi-parametric
    multi-state and competing risks models", *Computer Methods and
    Programs in Biomedicine*, 99(3), 261-274.
    """

    #: The states and the allowed transitions.
    transitions: TransitionMatrix
    #: The distinct event times.
    t: npt.NDArray
    #: ``(K, m)``: the number at risk in each state at each event time.
    r: npt.NDArray
    #: ``(K, K, m)``: the transitions from each state to each other one
    #: at each event time.
    d: npt.NDArray
    #: ``(m, K, K)``: the increments of the cumulative transition hazards
    #: (minus their row sums on the diagonal).
    dA: npt.NDArray
    #: The earliest entry: the default start ``s`` of a prediction.
    start: float
    #: The last observed time (an event or a censoring).
    last: float
    #: Whether units were labelled (``i``): the robust variance then sums
    #: each unit's spells.
    ids_given: bool = False
    _data: SpellData
    _table: _core.EventTable
    _cache: dict

    # -- fitting ------------------------------------------------------------

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        i: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
    ) -> MultiState:
        """
        Fit the Aalen-Johansen estimate to spells in states.

        Parameters
        ----------
        x : array_like
            The end of each spell: the time the unit left the state, or
            was last seen in it.
        e : array_like
            The state the unit moved into at ``x``; a missing value
            (``None``, ``NaN``) where the spell ended by censoring.
        state : array_like
            The state the unit was in during the spell.
        tl : array_like or scalar, optional
            The start of each spell, its entry: the unit is at risk in the
            state in ``(tl, x]``. A unit with several spells needs it (the
            second starts where the first ended). It is also delayed entry
            (left truncation) for a unit first seen part-way through.
            Defaults to observation from the start.
        i : array_like, optional
            The unit each spell belongs to. Used to check that a unit's
            spells do not overlap, and by the robust variance, which sums
            a unit's spells. Without it every row is a unit of its own.
        n : array_like, optional
            Counts: each row stands for ``n`` identical units (the same
            on each of a unit's rows). Defaults to 1.
        transitions : TransitionMatrix or iterable of (from, to), optional
            The states and the allowed transitions. Defaults to the
            transitions seen in the data, the states in sorted order.
            Transitions in the data that it does not allow are refused.

        Returns
        -------
        MultiState
            The fitted model.

        Examples
        --------
        Five units in an illness-death model: two fall ill and then die,
        one dies healthy and two are censored.

        >>> from surpyval.multistate import MultiState, TransitionMatrix
        >>> tm = TransitionMatrix.illness_death()
        >>> model = MultiState.fit(
        ...     x=[2, 6, 3, 8, 4, 7, 9],
        ...     e=["ill", "dead", "ill", None, "dead", None, None],
        ...     state=["healthy", "ill", "healthy", "ill", "healthy",
        ...            "healthy", "healthy"],
        ...     tl=[0, 2, 0, 3, 0, 0, 0],
        ...     i=[1, 1, 2, 2, 3, 4, 5],
        ...     transitions=tm,
        ... )
        >>> model.occupancy([1, 5, 9], "ill").round(4)
        array([0. , 0.4, 0.2])
        """
        data = check_spells(x, e, state, tl, i, n, transitions)
        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: SpellData) -> MultiState:
        table = _core.event_table(data)
        out = cls()
        out._data = data
        out.transitions = data.tm
        out.t = table.t
        out.r = table.r
        out.d = table.d
        out.dA = _core.increments(table)
        out._table = table
        out.start = float(np.min(data.tl))
        out.last = float(np.max(data.x))
        out.ids_given = data.ids_given
        out._cache = {}
        return out

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
    ) -> MultiState:
        """
        Fit the model to a DataFrame of spells, one row each: the columns
        are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.multistate import MultiState
        >>> df = pd.DataFrame({
        ...     "stop": [2, 6, 3, 4], "to": ["ill", "dead", "ill", "dead"],
        ...     "state": ["healthy", "ill", "healthy", "healthy"],
        ...     "start": [0, 2, 0, 0], "unit": [1, 1, 2, 3],
        ... })
        >>> model = MultiState.fit_from_df(
        ...     df, "stop", "to", "state", tl_col="start", i_col="unit"
        ... )
        >>> model.transitions.transitions
        (('healthy', 'dead'), ('healthy', 'ill'), ('ill', 'dead'))
        """
        from surpyval.utils.dataframe import frame_column

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        return cls.fit(
            col(x_col, "x_col"),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            tl=col(tl_col, "tl_col"),
            i=col(i_col, "i_col"),
            n=col(n_col, "n_col"),
            transitions=transitions,
        )

    # -- the estimate -------------------------------------------------------

    @property
    def states(self) -> tuple:
        """The states, in order."""
        return self.transitions.states

    def _start(self, s: float | None) -> float:
        if s is None:
            return self.start
        s = float(s)
        if np.isnan(s):
            raise ValueError("'s', the start of the prediction, is NaN.")
        return s

    def _P(self, s: float) -> npt.NDArray:
        key = ("P", s)
        if key not in self._cache:
            j0 = _core.start_index(self.t, s)
            self._cache[key] = _core.product_integral(self.dA, j0)
        return self._cache[key]

    def _first_event(self, s: float) -> float | None:
        """At the default start (the earliest entry), the first event
        time: the starting states are those of the units at risk then."""
        if s != self.start or self.t.size == 0:
            return None
        return float(self.t[0])

    def _p0(self, s: float, from_state: Any) -> npt.NDArray:
        k = self.transitions.n_states
        if from_state is not None:
            p0 = np.zeros(k)
            p0[self.transitions.index(from_state)] = 1.0
            return p0
        key = ("p0", s)
        if key not in self._cache:
            self._cache[key] = _core.at_risk_distribution(
                self._data, s, self._first_event(s)
            )
        return self._cache[key][0]

    def _at(
        self, values: npt.NDArray, x: npt.NDArray, s: float, initial: Any
    ) -> npt.NDArray:
        """``values`` (one per event time, on the first axis) at the
        query times: ``initial`` up to the first event after ``s``, NaN
        before ``s``."""
        idx = step_index(self.t, x)
        j0 = _core.start_index(self.t, s)
        take = values[np.clip(idx, 0, max(self.t.size - 1, 0))]
        before = (idx < j0)[(...,) + (None,) * (values.ndim - 1)]
        out = np.where(before, initial, take)
        out = np.where(
            (np.isnan(x) | (x < s))[(...,) + (None,) * (values.ndim - 1)],
            np.nan,
            out,
        )
        return out

    def transition_probabilities(
        self, x: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """
        The transition probability matrices :math:`P(s, t)` at the times
        ``x``: element ``[..., a, b]`` is the probability of being in
        state ``b`` at ``x`` for a unit in state ``a`` at ``s``, the
        states in the order of ``states``.

        Parameters
        ----------
        x : array_like
            The times.
        s : float, optional
            The time the units are in their starting state. Defaults to
            the earliest entry. A time ``x`` before ``s`` gives NaN.

        Returns
        -------
        numpy array
            The shape of ``x`` followed by ``(K, K)``.

        Examples
        --------
        >>> from surpyval.multistate import MultiState
        >>> model = MultiState.fit(
        ...     x=[1, 2, 5, 3, 6, 4], e=["b", "a", None, "b", None, None],
        ...     state=["a", "b", "a", "a", "b", "a"], tl=[0, 1, 2, 0, 3, 0],
        ...     i=[1, 1, 1, 2, 2, 3],
        ... )
        >>> model.states
        ('a', 'b')
        >>> model.transition_probabilities(3.5)
        array([[0.66666667, 0.33333333],
               [0.66666667, 0.33333333]])
        """
        s = self._start(s)
        xs = np.asarray(x, dtype=float)
        k = self.transitions.n_states
        out = self._at(self._P(s), xs.ravel(), s, np.eye(k))
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
            entry.
        from_state : any, optional
            The state the unit is in at ``s``: the probability is then the
            row of :meth:`transition_probabilities` for it. By default the
            units start in the states the units under observation were
            in, in proportion: at the first transition for the default
            ``s`` (those with ``tl < t_1 <= x``, as R's ``survfit``, since
            with delayed entry few may have entered at the earliest
            entry), else at ``s`` (``tl <= s < x``).

        Returns
        -------
        numpy array
            One probability per time; NaN before ``s``.

        Examples
        --------
        >>> from surpyval.multistate import MultiState
        >>> model = MultiState.fit(
        ...     x=[1, 2, 5, 3, 6, 4], e=["b", "a", None, "b", None, None],
        ...     state=["a", "b", "a", "a", "b", "a"], tl=[0, 1, 2, 0, 3, 0],
        ...     i=[1, 1, 1, 2, 2, 3],
        ... )

        Unit 1 moves to ``b`` at 1 and back at 2, unit 2 moves to ``b``
        at 3, so a third of the units are in ``b`` after 1, none after 2
        and a third again after 3:

        >>> model.occupancy([0.5, 1.5, 2.5, 3.5], "b").round(4)
        array([0.    , 0.3333, 0.    , 0.3333])
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        P = self._P(s)
        values = P[:, :, b].T  # (K, m): from each state
        p = p0 @ values
        return self._at(p, np.asarray(x, float), s, p0[b])

    @keeps_query_shape
    def cumulative_hazard(
        self, x: npt.ArrayLike, from_state: Any, to_state: Any
    ) -> npt.NDArray:
        """
        The Nelson-Aalen cumulative hazard of the transition from
        ``from_state`` to ``to_state`` at ``x``.

        Examples
        --------
        >>> from surpyval.multistate import MultiState
        >>> model = MultiState.fit(
        ...     x=[1, 2, 5, 3, 6, 4], e=["b", "a", None, "b", None, None],
        ...     state=["a", "b", "a", "a", "b", "a"], tl=[0, 1, 2, 0, 3, 0],
        ...     i=[1, 1, 1, 2, 2, 3],
        ... )
        >>> model.cumulative_hazard([1, 3], "a", "b").round(4)
        array([0.3333, 0.6667])
        """
        self.transitions.transition_index(from_state, to_state)
        a = self.transitions.index(from_state)
        b = self.transitions.index(to_state)
        H = np.cumsum(self.dA[:, a, b])
        x = np.asarray(x, dtype=float)
        idx = step_index(self.t, x)
        out = np.where(idx < 0, 0.0, H[np.maximum(idx, 0)])
        return np.where(np.isnan(x), np.nan, out)

    # -- uncertainty --------------------------------------------------------

    def _variance_blocks(self, s: float, variance: str) -> npt.NDArray:
        """``(m, K, K, K)``: ``[j, b, a, a']``, the covariance of
        :math:`P_{ab}(s, t_j)` and :math:`P_{a'b}(s, t_j)`."""
        key = (variance, s)
        if key not in self._cache:
            j0 = _core.start_index(self.t, s)
            cov = (
                _core.greenwood_increment_cov(self._table)
                if variance == "greenwood"
                else _core.aalen_increment_cov(self._table)
            )
            V = _core.propagate_variance(self.dA, cov, j0, variance)
            k = self.transitions.n_states
            # vec index a + K b -> [b, a]
            V = V.reshape(-1, k, k, k, k)  # [j, b, a, b', a']
            idx = np.arange(k)
            self._cache[key] = V[:, idx, :, idx, :].transpose(1, 0, 2, 3)
        return self._cache[key]

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        variance: str = "greenwood",
    ) -> npt.NDArray:
        """
        The standard error of :meth:`occupancy`.

        Parameters
        ----------
        x, state, s, from_state
            As for :meth:`occupancy`.
        variance : ('greenwood', 'aalen', 'robust'), str, optional
            ``'greenwood'`` (the default) and ``'aalen'`` are the
            Greenwood-type and Aalen-type variances of the Aalen-Johansen
            estimate, as R's ``mstate`` computes them (``msfit(...,
            vartype=)`` with ``probtrans(..., method=)``, the two set
            alike). They take the starting distribution as known.
            ``'robust'`` is the infinitesimal jackknife, R ``survival``'s
            ``survfit`` ``std.err``: it sums each unit's influence over
            its spells (pass ``i``), and includes the uncertainty of the
            default starting distribution.

        Examples
        --------
        >>> from surpyval.multistate import MultiState
        >>> model = MultiState.fit(
        ...     x=[1, 2, 5, 3, 6, 4], e=["b", "a", None, "b", None, None],
        ...     state=["a", "b", "a", "a", "b", "a"], tl=[0, 1, 2, 0, 3, 0],
        ...     i=[1, 1, 1, 2, 2, 3],
        ... )
        >>> model.occupancy_se(3.5, "b").round(4)
        np.float64(0.2722)
        """
        check_option("variance", variance, _VARIANCES)
        s = self._start(s)
        b = self.transitions.index(state)
        x = np.asarray(x, dtype=float)
        if variance == "robust":
            var = self._robust(s, from_state)[:, b]
        else:
            p0 = self._p0(s, from_state)
            blocks = self._variance_blocks(s, variance)[:, b]
            var = np.einsum("a,jab,b->j", p0, blocks, p0)
        se = np.sqrt(np.maximum(var, 0.0))
        return self._at(se, x, s, 0.0)

    def _robust(self, s: float, from_state: Any) -> npt.NDArray:
        key = ("robust", s, from_state)
        if key not in self._cache:
            j0 = _core.start_index(self.t, s)
            if from_state is None:
                p0 = self._p0(s, None)
                infl = self._cache[("p0", s)][1]
            else:
                p0 = self._p0(s, from_state)
                infl = None
            var = _core.robust_variance(
                self._data, self._table, self.dA, j0, p0[None, :], infl
            )
            self._cache[key] = var[:, 0, :]
        return self._cache[key]

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "exp",
        variance: str = "greenwood",
    ) -> npt.NDArray:
        r"""
        Pointwise confidence bounds on the state occupation probability
        :meth:`occupancy`.

        Parameters
        ----------
        x, state, s, from_state
            As for :meth:`occupancy`.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95%
            interval.
        bound : ('two-sided', 'lower', 'upper'), str, optional
            The two-sided interval (the default), with one row per ``x``
            and the columns ``[lower, upper]``, or one side, which puts
            all of ``alpha_ci`` on that side.
        bound_type : ('exp', 'normal'), str, optional
            ``'exp'`` (the default) forms the interval on the log(-log)
            scale, :math:`\hat{p}^{\exp(\pm z\,\hat{\sigma} / (\hat{p}
            \log \hat{p}))}`, which keeps it within [0, 1]; ``'normal'``
            is :math:`\hat{p} \pm z\,\hat{\sigma}`, which can leave it.
        variance : ('greenwood', 'aalen', 'robust'), str, optional
            The variance; see :meth:`occupancy_se`.

        Returns
        -------
        numpy array
            ``[lower, upper]`` rows two-sided, one value per ``x`` for
            one side. Where the variance is 0 (before the first
            transition, or a probability of exactly 0 or 1) the bounds are
            the estimate. Past the last observed time they are NaN, with a
            warning.

        Examples
        --------
        >>> from surpyval.multistate import MultiState
        >>> model = MultiState.fit(
        ...     x=[1, 2, 5, 3, 6, 4], e=["b", "a", None, "b", None, None],
        ...     state=["a", "b", "a", "a", "b", "a"], tl=[0, 1, 2, 0, 3, 0],
        ...     i=[1, 1, 1, 2, 2, 3],
        ... )
        >>> model.cb(3.5, "b").round(4)
        array([0.009 , 0.7741])
        """
        check_bound_args(alpha_ci, bound, bound_type)
        check_option("variance", variance, _VARIANCES)
        x = np.asarray(x, dtype=float)
        p = self.occupancy(x, state, s, from_state)
        se = self.occupancy_se(x, state, s, from_state, variance)
        out = probability_bounds(p, se, alpha_ci, bound, bound_type)
        past = x > self.last
        if np.any(past):
            from surpyval.univariate.nonparametric.nonparametric import (
                warn_bounds_past_data,
            )

            warn_bounds_past_data(
                "cb", self.last, np.unique(x[past]), "occupancy"
            )
            out = np.where(
                past if out.ndim == 1 else past[:, None], np.nan, out
            )
        return out

    # -- plotting -----------------------------------------------------------

    def plot(
        self,
        s: float | None = None,
        from_state: Any = None,
        ax: Any = None,
    ) -> Any:
        """
        Plot the state occupation probabilities over time, stacked: at
        each time the bands' heights are the probabilities of being in
        each state, adding up to 1.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        s0 = self._start(s)
        if ax is None:
            ax = plt.gca()
        lo = s0 if np.isfinite(s0) else min(0.0, float(self.t[0]))
        times = np.r_[lo, self.t[self.t > lo], self.last]
        probs = np.vstack(
            [self.occupancy(times, st, s0, from_state) for st in self.states]
        )
        ax.stackplot(
            times, probs, labels=[str(st) for st in self.states], step="post"
        )
        ax.set_xlabel("Time" if self.unit is None else f"Time ({self.unit})")
        ax.set_ylabel("State occupation probability")
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        return ax

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict. The data
        (the spells) are stored, so the reloaded model gives every
        estimate and variance exactly as the original.
        """
        d = self._data
        tm = self.transitions
        out = {
            "model": "MultiState",
            "transitions": tm.to_dict(),
            "x": d.x.tolist(),
            "tl": d.tl.tolist(),
            "state": [to_native(tm.states[g]) for g in d.g],
            "e": [None if h < 0 else to_native(tm.states[h]) for h in d.h],
            "n": d.n.tolist(),
            "i": d.cluster.tolist() if d.ids_given else None,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> MultiState:
        """Rebuild a multi-state model from :meth:`to_dict`."""
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        require_model_tag(model_dict, "MultiState", "a multi-state model")
        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        tl = [-np.inf if v is None else v for v in model_dict["tl"]]
        data = check_spells(
            model_dict["x"],
            [label_from_native(v) for v in model_dict["e"]],
            [label_from_native(v) for v in model_dict["state"]],
            tl,
            model_dict.get("i", model_dict.get("id")),
            model_dict["n"],
            tm,
        )
        return cls._from_data(data)

    def __repr__(self) -> str:
        tm = self.transitions
        return (
            "Non-parametric multi-state model (Aalen-Johansen)\n"
            f"States: {list(tm.states)!r}\n"
            f"Transitions: {', '.join(tm.transition_names())}\n"
            f"Transitions observed: {int(self.d.sum())} at "
            f"{self.t.size} times"
        )
