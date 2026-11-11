"""Parametric multi-state models: a parametric distribution (or a
parametric regression) for each transition, and the transition and state
occupation probabilities they imply, from the Kolmogorov forward
equations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate import _core, _forward
from surpyval.multistate._base import check_bound_args, probability_bounds
from surpyval.multistate._data import SpellData, check_spells
from surpyval.multistate.regression import _covariates
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

_CLOCKS = ("forward", "reset")


def _check_clock(clock: str) -> None:
    check_option("clock", clock, _CLOCKS)
    if clock == "reset":
        raise ValueError(
            "clock='reset' (a semi-Markov model, each transition's hazard "
            "a function of the time spent in the current state) is not "
            "built: its transition probabilities are not the solution of "
            "the forward equations, and need either nested integrals over "
            "the times of every earlier move or simulation. Use "
            "clock='forward' (the hazards a function of the time since "
            "the start, a Markov model), or fit each transition to the "
            "durations x - tl with the univariate fitters."
        )


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

    # -- fitting ------------------------------------------------------------

    @classmethod
    def _fit(
        cls, data: SpellData, Z: npt.NDArray, fitters: list, clock: str
    ) -> Any:
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
        for h, pair in enumerate(tm.transitions):
            rows = data.g == frm[h]
            c = np.where(data.h[rows] == to[h], 0, 1)
            # An entry at or before the start of the distribution's support
            # truncates nothing: leave it out (the fit is then the plain
            # one, on the same likelihood).
            low = float(np.ravel(getattr(fitters[h], "support", [-np.inf]))[0])
            tl = np.where(data.tl[rows] <= low, -np.inf, data.tl[rows])
            extra = {"tl": tl} if np.any(np.isfinite(tl)) else {}
            models[pair] = cls._fit_one(
                fitters[h],
                data.x[rows],
                Z[rows],
                c,
                data.n[rows],
                extra,
                found,
            )
        cls._warn_found(found)
        out = cls()
        out.transitions = tm
        out.models = models
        out.clock = clock
        out._data = data
        out._Z = Z
        out._setup()
        return out

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
        """
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
        from surpyval.utils.no_maximum import combined_maximum

        return combined_maximum(
            getattr(m, "maximum", "unknown") for m in self._models()
        )

    def neg_ll(self) -> float:
        """The negative log-likelihood: the sum of the transitions'."""
        return float(sum(m.neg_ll() for m in self._models()))

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[int, float]:
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

    def _cumhaz(self, models: list, z: npt.NDArray) -> Any:
        """The transitions' cumulative hazards at ``t`` (their hazards
        with ``rate``), as :func:`_forward.plan` takes them."""

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
        key = ("grid", tuple(z.tolist()), s, tuple(times.tolist()))
        if key not in self._cache:
            tm = self.transitions
            self._cache[key] = _forward.plan(
                self._cumhaz(self._models(), z),
                s,
                times,
                self._scale(),
                tm.from_index(),
                tm.to_index(),
                tm.n_states,
                breaks=self._breaks(),
            )
        return self._cache[key]

    def _solve(self, models: list, z: npt.NDArray, grid: Any) -> npt.NDArray:
        tm = self.transitions
        return _forward.evaluate(
            self._cumhaz(models, z),
            grid,
            tm.from_index(),
            tm.to_index(),
            tm.n_states,
        )

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

    def _P_at(
        self, z: npt.NDArray, s: float, times: npt.NDArray
    ) -> npt.NDArray:
        """``(len(times), K, K)``: :math:`P(s, t)` at ``times`` (finite,
        at or after ``s``)."""
        k = self.transitions.n_states
        if times.size == 0:
            return np.zeros((0, k, k))
        uniq, inverse = np.unique(times, return_inverse=True)
        key = ("P", tuple(z.tolist()), s, tuple(uniq.tolist()))
        if key not in self._cache:
            grid = self._grid(z, s, uniq)
            self._cache[key] = self._solve(self._models(), z, grid)
        return self._cache[key][np.ravel(inverse)]

    def _occupancy_values(
        self,
        z: npt.NDArray,
        s: float,
        p0: npt.NDArray,
        b: int,
        times: npt.NDArray,
    ) -> npt.NDArray:
        return (p0 @ self._P_at(z, s, times))[:, b]

    def _occupancy_se_values(
        self,
        z: npt.NDArray,
        s: float,
        p0: npt.NDArray,
        b: int,
        times: npt.NDArray,
    ) -> npt.NDArray:
        """The delta-method standard errors of the occupation
        probabilities at ``times``, over every transition's parameters,
        by central differences on the point estimate's grid."""
        from surpyval.univariate.competing_risks._assembly import (
            central_differences,
            checked_sd,
        )

        if times.size == 0:
            return np.zeros(0)
        uniq, inverse = np.unique(times, return_inverse=True)
        idx = np.ravel(inverse)
        grid = self._grid(z, s, uniq)
        models = self._models()
        sizes = [len(m.covariance_names) for m in models]
        est = self._estimates()
        cov = self.covariance()

        def occ(values: npt.NDArray) -> npt.NDArray:
            rebuilt = []
            start = 0
            for m, size in zip(models, sizes):
                rebuilt.append(m.with_params(values[start : start + size]))
                start += size
            P = self._solve(rebuilt, z, grid)
            return (p0 @ P[idx])[:, b]

        from surpyval.utils.warnings import ignoring_warnings

        with ignoring_warnings():
            G = central_differences(occ, est, cov, times.size)
        var = np.einsum("ia,ij,ja->a", G, cov, G)
        return checked_sd(
            var, times, "occupancy", np.ones(times.size, dtype=bool)
        )

    # -- serialisation ------------------------------------------------------

    def _data_dict(self) -> dict:
        d = self._data
        tm = self.transitions
        return {
            "transitions": tm.to_dict(),
            "x": d.x.tolist(),
            "tl": [None if not np.isfinite(v) else float(v) for v in d.tl],
            "state": [to_native(tm.states[g]) for g in d.g],
            "e": [None if h < 0 else to_native(tm.states[h]) for h in d.h],
            "n": d.n.tolist(),
            "i": d.cluster.tolist() if d.ids_given else None,
            "clock": self.clock,
            "models": [m.to_dict() for m in self._models()],
        }

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
        )
        out = cls()
        out.transitions = tm
        out.models = {
            pair: model_cls.from_dict(sub)
            for pair, sub in zip(tm.transitions, model_dict["models"])
        }
        out.clock = model_dict.get("clock", "forward")
        out._data = data
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
    start (the clock goes forward: a Markov model). Each transition's
    distribution is fitted to the spells in its 'from' state, where the
    spells that ended in another transition, or were censored, are
    censored, and a spell that began at ``tl`` is left truncated there
    (delayed entry). That is the maximum-likelihood fit of the whole
    model, since its likelihood is a product of one factor per transition.

    The transition probabilities :math:`P(s, t)` solve the Kolmogorov
    forward equations :math:`\partial P(s, t) / \partial t = P(s, t)
    Q(t)`, :math:`Q(t)` the matrix of the transition hazards, which are
    solved by product integration on an adaptive grid to an accuracy of
    about :math:`10^{-8}` in each row (the sum of the absolute errors of
    its elements). The bounds on the state occupation probabilities come
    from the delta method over every transition's parameters.

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
            transitions.
        dist : ParametricFitter or dict, optional
            The distribution of every transition (default ``Weibull``),
            or a ``{(from, to): distribution}`` dict with one for each.
        clock : ('forward',), str, optional
            The time scale of the hazards: ``'forward'``, the time since
            the start (a Markov model). ``'reset'`` (a semi-Markov model,
            the time in the current state) is refused, with the reason.

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
        array([0.627, 0.896])
        """
        from surpyval import Weibull

        _check_clock(clock)
        data = check_spells(x, e, state, tl, i, n, transitions)
        fitters = _fitters(Weibull if dist is None else dist, data.tm, False)
        Z = np.zeros((data.x.size, 0))
        return cls._fit(data, Z, fitters, clock)

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
        dist: Any = None,
        clock: str = "forward",
    ) -> ParametricMultiState:
        """
        Fit the model to a DataFrame of spells, one row each; the columns
        are the arguments of :meth:`fit`.

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
            dist=dist,
            clock=clock,
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
        default the earliest entry). NaN before ``s``.

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
               [0.339, 0.488, 0.173],
               [0.528, 0.   , 0.472]])
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
            entry (0 where spells are observed from the start).
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
        array([0.0139, 0.0094, 0.0012])
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
        np.float64(0.0085)
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
        bound_type: str = "exp",
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
        bound_type : ('exp', 'normal'), str, optional
            ``'exp'`` (the default) works on the log(-log) scale and
            keeps the bounds in (0, 1); ``'normal'`` is the estimate plus
            and minus ``z`` standard errors.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import ParametricMultiState
        >>> model = ParametricMultiState.fit_from_df(
        ...     load_mgus2(spells=True), "x", "e", "state", tl_col="tl",
        ...     i_col="id",
        ... )
        >>> model.cb(240, "death").round(3)
        array([0.879, 0.911])
        """
        check_bound_args(alpha_ci, bound, bound_type)
        p = self.occupancy(x, state, s, from_state)
        se = self.occupancy_se(x, state, s, from_state)
        return probability_bounds(
            np.ravel(p), np.ravel(se), alpha_ci, bound, bound_type
        )

    @keeps_query_shape
    def cumulative_hazard(
        self, x: npt.ArrayLike, from_state: Any, to_state: Any
    ) -> npt.NDArray:
        """
        The cumulative hazard of the transition from ``from_state`` to
        ``to_state`` at ``x``.

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
        np.float64(0.1135)
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
    its distributions; every transition has its own coefficients. The
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
    ) -> ParametricMultiStateRegression:
        """
        Fit a parametric regression to each transition of spells in
        states.

        Parameters
        ----------
        x, e, state, tl, i, n, transitions, clock
            As for :meth:`ParametricMultiState.fit`.
        Z : array_like
            The covariates, one row per spell (a unit's covariates on each
            of its spells). Missing values are refused: dropping a spell
            would cut a unit's path through its states.
        model : regression fitter or dict, optional
            The parametric regression of every transition (default
            ``WeibullPH``), such as ``ExponentialPH``, ``WeibullAFT`` or
            ``LogNormalAFT``, or a ``{(from, to): fitter}`` dict with one
            for each.

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
        array([0.0126, 0.0153])
        """
        from surpyval import WeibullPH

        _check_clock(clock)
        data = check_spells(x, e, state, tl, i, n, transitions)
        Z_arr, names = _covariates(Z, data.x.size)
        fitters = _fitters(
            WeibullPH if model is None else model, data.tm, True
        )
        out = cls._fit_named(data, Z_arr, names, fitters, clock)
        return out

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
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: Any,
        e_col: str,
        state_col: str,
        tl_col: str | None = None,
        i_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
    ) -> ParametricMultiStateRegression:
        """
        Fit the model to a DataFrame of spells, one row each; ``Z_cols``
        names the covariate columns, and the other columns are the
        arguments of :meth:`fit`.

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
        from surpyval.utils.dataframe import frame_column

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        Z = np.column_stack([frame_column(df, c, "Z_cols") for c in cols])
        _check_clock(clock)
        data = check_spells(
            col(x_col, "x_col"),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            col(tl_col, "tl_col"),
            col(i_col, "i_col"),
            col(n_col, "n_col"),
            transitions,
        )
        Z_arr, _ = _covariates(Z, data.x.size)
        from surpyval import WeibullPH

        fitters = _fitters(
            WeibullPH if model is None else model, data.tm, True
        )
        return cls._fit_named(
            data, Z_arr, [str(c) for c in cols], fitters, clock
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
        return pd.concat(
            {t: m.summary(alpha_ci) for t, m in zip(names, self._models())},
            names=["transition"],
        )

    # -- predictions --------------------------------------------------------

    def _z(self, Z: Any) -> npt.NDArray:
        z = np.asarray(Z, dtype=float).ravel()
        if z.size != len(self.covariate_names):
            raise ValueError(
                f"Z must be one row of {len(self.covariate_names)} "
                "covariate(s)."
            )
        return z

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
               [0.62 , 0.368, 0.013],
               [0.982, 0.   , 0.018]])
        """
        s = self._start(s)
        z = self._z(Z)
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
        array([0.62 , 0.509])
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
        np.float64(0.0133)
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
        bound_type: str = "exp",
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
        array([0.85, 0.9 ])
        """
        check_bound_args(alpha_ci, bound, bound_type)
        p = self.occupancy(x, Z, state, s, from_state)
        se = self.occupancy_se(x, Z, state, s, from_state)
        return probability_bounds(
            np.ravel(p), np.ravel(se), alpha_ci, bound, bound_type
        )

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
        np.float64(0.0958)
        """
        h = self.transitions.transition_index(from_state, to_state)
        m = self._models()[h]

        def value(z: npt.NDArray, times: npt.NDArray) -> npt.NDArray:
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
