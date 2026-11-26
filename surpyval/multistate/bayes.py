"""
Bayesian parametric multi-state models, by composition.

A parametric multi-state model's likelihood is a product of one factor
per transition: the spells in a transition's 'from' state, ended by that
transition (an event) or by anything else (censored), and left truncated
where they began (on the clock going forward). So, with priors stated a
transition at a time, the transitions' posteriors are independent, and
each is the ordinary Bayesian fit of that transition's distribution
(``dist.fit_bayes``) on those spells. Nothing new is computed for the
posterior; this module combines the transitions' fits.

The state occupation and transition probabilities follow from the
transitions' hazards through the Kolmogorov forward equations (or, on
the clock reset, the renewal equations), which are not linear in the
parameters, so the predictions come from joint draws: one draw from
each transition's posterior is a joint draw. A probability for a new
unit is the posterior mean over the draws; a credible interval is their
``alpha_ci / 2`` and ``1 - alpha_ci / 2`` points. The draws have a fixed
seed, so the numbers repeat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import check_alpha_ci, check_option
from surpyval.utils.warnings import ignoring_warnings

#: The joint draws behind every prediction (``BayesianMultiState.draws``
#: by default).
DRAWS = 1000
#: Their seed: transition ``h`` (in the order of the transitions) draws
#: with ``SEED + h``.
SEED = 20261126


def _levels(alpha_ci: float, bound: str) -> list[float]:
    check_alpha_ci(alpha_ci)
    check_option("bound", bound, ("two-sided", "lower", "upper"))
    if bound == "two-sided":
        return [alpha_ci / 2, 1 - alpha_ci / 2]
    return [alpha_ci] if bound == "lower" else [1 - alpha_ci]


class BayesianMultiState(SerialisableMixin):
    """
    A Bayesian parametric multi-state model
    (``ParametricMultiState.fit_bayes``): a distribution for each
    transition's hazard, each with its own posterior.

    The likelihood is a product of one factor per transition, so with
    priors stated a transition at a time their posteriors are
    independent, and each is the
    :class:`~surpyval.bayes.BayesianParametric` of its transition's
    spells, in ``models``. The state occupation probabilities, the
    transition probabilities and the expected time in a state come from
    ``draws`` joint draws (one from each transition's posterior, fixed
    seed): a prediction for a new unit is their mean, and a credible
    interval their quantiles.

    Attributes
    ----------
    models : dict
        Each transition's Bayesian fit, by ``(from, to)``.
    transitions : TransitionMatrix
        The states and the allowed transitions.
    mle : ParametricMultiState
        The maximum-likelihood fit to the same spells, whose start, data
        and solver the predictions use.
    draws : int
        How many joint draws a prediction uses (1,000 by default). More
        is slower in proportion: each draw solves the model's equations.

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
    >>> sorted(model.models)
    [('healthy', 'dead'), ('healthy', 'ill'), ('ill', 'dead')]
    >>> lo, hi = model.occupancy_cri(4.0, "dead")
    >>> bool(lo < model.occupancy(4.0, "dead") < hi)
    True
    """

    model_tag = "BayesianMultiState"

    def __init__(self, models: dict, mle: Any) -> None:
        self.mle = mle
        self.transitions = mle.transitions
        self.models = {
            pair: models[pair] for pair in self.transitions.transitions
        }
        self.clock = mle.clock
        self.method = {k: m.method for k, m in self.models.items()}
        self.draws = DRAWS
        self._cache: dict = {}
        self._joint: tuple | None = None

    @property
    def states(self) -> tuple:
        """The states, in order."""
        return self.transitions.states

    # -- joint draws ----------------------------------------------------------

    def _draw_models(self) -> list:
        """``draws`` lists of the transitions' ordinary models, one list
        a joint draw."""
        if self._joint is None or self._joint[0] != self.draws:
            per = [
                fit.draw_models(self.draws, random_state=SEED + h)
                for h, fit in enumerate(self.models.values())
            ]
            self._joint = (self.draws, [list(r) for r in zip(*per)])
            self._cache = {}
        return self._joint[1]

    def _values(
        self,
        what: str,
        x: npt.NDArray,
        s: float | None,
        from_state: Any,
        z: npt.NDArray | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(draws, m, K)``: from the starting distribution at ``s``,
        the probability of each state (``what="P"``) or the expected
        time in it (``"los"``) at the times ``x``, under each joint draw
        (for a regression, at the covariate row ``z``); and the mask of
        the times at or after ``s``."""
        mle = self.mle
        z = np.zeros(0) if z is None else np.asarray(z, dtype=float)
        s = mle._start(s)
        p0 = mle._p0(s, from_state)
        flat = np.asarray(x, dtype=float).ravel()
        ok = np.isfinite(flat) & (flat >= s)
        K = self.transitions.n_states
        out = np.full((self.draws, flat.size, K), np.nan)
        if not ok.any():
            return out, ok
        uniq, inverse = np.unique(flat[ok], return_inverse=True)
        key = (
            what,
            s,
            tuple(p0.tolist()),
            tuple(uniq.tolist()),
            tuple(z.tolist()),
        )
        draws = self._draw_models()
        if key not in self._cache:
            if self.clock == "forward":
                # (the solver's grid set by the fitted model, as for its
                # standard errors)
                mle._grid(z, s, uniq)
            else:
                mle._reset_step(z)
            vals = np.empty((len(draws), uniq.size, K))
            with ignoring_warnings(), np.errstate(all="ignore"):
                for d, models in enumerate(draws):
                    P = mle._matrices(models, z, s, uniq, what)
                    vals[d] = np.einsum("a,tab->tb", p0, P)
            self._cache[key] = vals
        out[:, ok, :] = self._cache[key][:, np.ravel(inverse), :]
        return out, ok

    def _summarise(
        self,
        what: str,
        x: npt.ArrayLike,
        state: Any,
        s: float | None,
        from_state: Any,
        levels: list[float] | None,
        z: npt.NDArray | None = None,
    ) -> npt.NDArray:
        b = self.transitions.index(state)
        vals, ok = self._values(
            what, np.asarray(x, dtype=float), s, from_state, z
        )
        v = vals[:, :, b]
        if levels is None:
            out = np.mean(v, axis=0)
            out[~ok] = np.nan
            return out
        q = np.stack([np.quantile(v, lv, axis=0) for lv in levels], axis=-1)
        q[~ok] = np.nan
        return q if len(levels) == 2 else q[:, 0]

    # -- predictions --------------------------------------------------------

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The probability that a new unit is in ``state`` at ``x``: the
        posterior mean of the state occupation probability over the
        joint draws (the posterior predictive). ``s`` and ``from_state``
        are as for :meth:`ParametricMultiState.occupancy
        <surpyval.multistate.ParametricMultiState.occupancy>`.
        """
        return self._summarise("P", x, state, s, from_state, None)

    @keeps_query_shape
    def occupancy_cri(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval on the state occupation probability: the
        range in which the population's probability of being in
        ``state`` at ``x`` lies with posterior probability ``1 -
        alpha_ci`` (equal-tailed, over the joint draws). Two-sided adds a
        last ``[lower, upper]`` axis.
        """
        levels = _levels(alpha_ci, bound)
        return self._summarise("P", x, state, s, from_state, levels)

    @keeps_query_shape
    def length_of_stay(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """The expected time a new unit spends in ``state`` from ``s`` to
        ``x``: the posterior mean over the joint draws."""
        return self._summarise("los", x, state, s, from_state, None)

    @keeps_query_shape
    def length_of_stay_cri(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval on the population's expected time in
        ``state`` from ``s`` to ``x``."""
        levels = _levels(alpha_ci, bound)
        return self._summarise("los", x, state, s, from_state, levels)

    def transition_probabilities(
        self, x: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """The posterior means of the transition probability matrices
        :math:`P(s, t)` at the times ``x`` (element ``[..., a, b]``: in
        ``b`` at ``x`` for a unit in ``a`` at ``s``); NaN before ``s``."""
        K = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        rows = [
            self._values("P", xs.ravel(), s, self.states[a])[0].mean(axis=0)
            for a in range(K)
        ]
        out = np.stack(rows, axis=1)  # (m, K from, K to)
        return out.reshape(xs.shape + (K, K))

    def param_cri(
        self,
        transition: tuple,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval for a parameter of one transition's
        distribution, the transition as ``(from, to)``."""
        if transition not in self.models:
            raise ValueError(
                f"No transition {transition!r}; the transitions are "
                f"{list(self.models)!r}."
            )
        return np.asarray(
            self.models[transition].param_cri(name, alpha_ci, bound),
            dtype=float,
        )

    # -- words and saving -----------------------------------------------------

    def summary(self, alpha_ci: float = 0.05) -> str:
        """Each transition's fit in words, one after another."""
        names = self.transitions.transition_names()
        return "\n\n".join(
            f"Transition {name}\n" + fit.summary(alpha_ci)
            for name, fit in zip(names, self.models.values())
        )

    def __repr__(self) -> str:
        return self.summary()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_joint"] = None
        return state

    def to_dict(self) -> dict:
        """A strict-JSON dictionary: the maximum-likelihood fit (the
        spells and the transitions) and each transition's Bayesian fit,
        in the order of the transitions."""
        import surpyval

        return stamp_schema(
            {
                "model": self.model_tag,
                "mle": self.mle.to_dict(),
                "models": [m.to_dict() for m in self.models.values()],
                "draws": int(self.draws),
                "surpyval_version": surpyval.__version__,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianMultiState:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes import BayesianParametric
        from surpyval.multistate.parametric import ParametricMultiState

        require_model_tag(
            model_dict, cls.model_tag, "a Bayesian multi-state fit"
        )
        mle = ParametricMultiState.from_dict(model_dict["mle"])
        fits = [BayesianParametric.from_dict(d) for d in model_dict["models"]]
        out = cls(dict(zip(mle.transitions.transitions, fits)), mle)
        out.draws = int(model_dict.get("draws", DRAWS))
        return out


def fit(
    cls: Any,
    x: Any,
    e: Any,
    state: Any,
    tl: Any = None,
    i: Any = None,
    n: Any = None,
    transitions: Any = None,
    dist: Any = None,
    clock: str = "forward",
    *,
    priors: Any = None,
    method: str = "auto",
    points: int | None = None,
    importance: bool = False,
) -> BayesianMultiState:
    """The Bayesian fit of a parametric multi-state model; see
    ``ParametricMultiState.fit_bayes``."""
    from surpyval import Weibull
    from surpyval.multistate._data import check_spells
    from surpyval.multistate.parametric import (
        _check_clock,
        _check_reset,
        _durations,
        _fitters,
    )
    from surpyval.univariate.competing_risks.parametric.bayes import (
        labelled,
        split_priors,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    _check_clock(clock)
    data = check_spells(x, e, state, tl, i, n, transitions)
    tm = data.tm
    fitters = _fitters(Weibull if dist is None else dist, tm, False)
    with quiet_maximum_warnings(), ignoring_warnings():
        mle = cls._fit(data, np.zeros((data.x.size, 0)), fitters, clock)
    own = split_priors(priors, list(tm.transitions))
    frm, to = tm.from_index(), tm.to_index()
    if clock == "reset":
        _check_reset(fitters, tm, "dist")
        x_fit = _durations(data)
    else:
        x_fit = data.x
    models = {}
    for h, pair in enumerate(tm.transitions):
        rows = data.g == frm[h]
        c = np.where(data.h[rows] == to[h], 0, 1)
        extra: dict = {}
        if clock == "forward":
            # As the fit: an entry at or before the start of the
            # distribution's support truncates nothing.
            low = float(np.ravel(getattr(fitters[h], "support", [-np.inf]))[0])
            tl_h = np.where(data.tl[rows] <= low, -np.inf, data.tl[rows])
            if np.any(np.isfinite(tl_h)):
                extra = {"tl": tl_h}
        try:
            models[pair] = labelled(
                f"Transition {tm.transition_names()[h]}",
                lambda h=h, rows=rows, c=c, pair=pair, extra=extra: fitters[
                    h
                ].fit_bayes(
                    x_fit[rows],
                    c,
                    data.n[rows],
                    priors=own[pair],
                    method=method,
                    points=points,
                    importance=importance,
                    **extra,
                ),
            )
        except ValueError as err:
            name = tm.transition_names()[h]
            raise ValueError(f"Transition {name}: {err}") from None
    return BayesianMultiState(models, mle)


class BayesianMultiStateRegression(BayesianMultiState):
    """
    A Bayesian parametric multi-state regression
    (``ParametricMultiStateRegression.fit_bayes``): a regression for
    each transition's hazard, each with its own posterior.

    The likelihood is a product of one factor per transition, so with
    priors stated a transition at a time their posteriors are
    independent, and each is the :class:`~surpyval.bayes.BayesianRegression`
    of its transition's spells and covariates, in ``models``. The
    predictions are for one covariate row ``Z`` per call: the state
    occupation probabilities, the transition probabilities and the
    expected time in a state come from ``draws`` joint draws (one from
    each transition's posterior, fixed seed) solved at that row through
    the fitted model's own equations; a prediction for a new unit is
    their mean, a credible interval their quantiles.

    Examples
    --------
    >>> from surpyval import ExponentialPH
    >>> from surpyval.datasets import load_mgus2
    >>> from surpyval.multistate import ParametricMultiStateRegression
    >>> df = load_mgus2(spells=True).iloc[::4]
    >>> model = ParametricMultiStateRegression.fit_bayes(
    ...     df["x"], df[["male"]], df["e"], df["state"], tl=df["tl"],
    ...     i=df["id"], model=ExponentialPH)
    >>> lo, hi = model.occupancy_cri(120.0, [1], "death")
    >>> bool(lo < model.occupancy(120.0, [1], "death") < hi)
    True
    """

    model_tag = "BayesianMultiStateRegression"

    def _z(self, Z: Any) -> npt.NDArray:
        z: npt.NDArray = self.mle._z(Z)
        return z

    def _summarise(
        self,
        what: str,
        x: npt.ArrayLike,
        state: Any,
        s: float | None,
        from_state: Any,
        levels: list[float] | None,
        z: npt.NDArray | None = None,
    ) -> npt.NDArray:
        if z is not None and np.isnan(z).any():
            # a missing covariate: NaN, as for every regression
            self.transitions.index(state)
            tail = () if levels is None or len(levels) == 1 else (2,)
            shape = np.asarray(x, dtype=float).ravel().shape + tail
            return np.full(shape, np.nan)
        return super()._summarise(what, x, state, s, from_state, levels, z)

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """The probability that a new unit with the covariate row ``Z``
        is in ``state`` at ``x``: the posterior mean over the joint
        draws."""
        return self._summarise("P", x, state, s, from_state, None, self._z(Z))

    @keeps_query_shape
    def occupancy_cri(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval on the state occupation probability at
        the covariate row ``Z``."""
        levels = _levels(alpha_ci, bound)
        return self._summarise(
            "P", x, state, s, from_state, levels, self._z(Z)
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
        """The expected time a new unit with the row ``Z`` spends in
        ``state`` from ``s`` to ``x``."""
        return self._summarise(
            "los", x, state, s, from_state, None, self._z(Z)
        )

    @keeps_query_shape
    def length_of_stay_cri(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval on the expected time in ``state`` at the
        row ``Z``."""
        levels = _levels(alpha_ci, bound)
        return self._summarise(
            "los", x, state, s, from_state, levels, self._z(Z)
        )

    def transition_probabilities(  # type: ignore[override]
        self, x: npt.ArrayLike, Z: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """The posterior means of :math:`P(s, t \\mid z)` at the times
        ``x`` for the covariate row ``Z``."""
        z = self._z(Z)
        K = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        rows = [
            self._values("P", xs.ravel(), s, self.states[a], z)[0].mean(axis=0)
            for a in range(K)
        ]
        out = np.stack(rows, axis=1)
        return out.reshape(xs.shape + (K, K))

    def to_dict(self) -> dict:
        """A strict-JSON dictionary: the maximum-likelihood fit and each
        transition's Bayesian regression, in the order of the
        transitions."""
        import surpyval

        return stamp_schema(
            {
                "model": self.model_tag,
                "mle": self.mle.to_dict(),
                "models": [m.to_dict() for m in self.models.values()],
                "draws": int(self.draws),
                "surpyval_version": surpyval.__version__,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianMultiStateRegression:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes import BayesianRegression
        from surpyval.multistate.parametric import (
            ParametricMultiStateRegression,
        )

        require_model_tag(
            model_dict, cls.model_tag, "a Bayesian multi-state regression"
        )
        mle = ParametricMultiStateRegression.from_dict(model_dict["mle"])
        fits = [BayesianRegression.from_dict(d) for d in model_dict["models"]]
        out = cls(dict(zip(mle.transitions.transitions, fits)), mle)
        out.draws = int(model_dict.get("draws", DRAWS))
        return out


def fit_regression(
    cls: Any,
    x: Any,
    Z: Any,
    e: Any,
    state: Any,
    tl: Any = None,
    i: Any = None,
    n: Any = None,
    transitions: Any = None,
    model: Any = None,
    clock: str = "forward",
    *,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
) -> BayesianMultiStateRegression:
    """The Bayesian fit of a parametric multi-state regression; see
    ``ParametricMultiStateRegression.fit_bayes``."""
    from surpyval import WeibullPH
    from surpyval.multistate._data import check_spells
    from surpyval.multistate.parametric import (
        _check_clock,
        _check_reset,
        _durations,
        _fitters,
    )
    from surpyval.multistate.regression import _covariates
    from surpyval.univariate.competing_risks.parametric.bayes import (
        labelled,
        split_priors,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    _check_clock(clock)
    data = check_spells(x, e, state, tl, i, n, transitions)
    tm = data.tm
    Z_arr, names = _covariates(Z, data.x.size)
    fitters = _fitters(WeibullPH if model is None else model, tm, True)
    with quiet_maximum_warnings(), ignoring_warnings():
        mle = cls._fit_named(data, Z_arr, names, fitters, clock)
    import pandas as pd

    frame = pd.DataFrame(Z_arr, columns=names)
    own = split_priors(priors, list(tm.transitions))
    frm, to = tm.from_index(), tm.to_index()
    if clock == "reset":
        _check_reset(fitters, tm, "model")
        x_fit = _durations(data)
    else:
        x_fit = data.x
    models = {}
    for h, pair in enumerate(tm.transitions):
        rows = data.g == frm[h]
        c = np.where(data.h[rows] == to[h], 0, 1)
        extra: dict = {}
        if clock == "forward":
            # As the fit: an entry at or before the start of the
            # distribution's support truncates nothing.
            low = float(np.ravel(getattr(fitters[h], "support", [-np.inf]))[0])
            tl_h = np.where(data.tl[rows] <= low, -np.inf, data.tl[rows])
            if np.any(np.isfinite(tl_h)):
                extra = {"tl": tl_h}
        try:
            models[pair] = labelled(
                f"Transition {tm.transition_names()[h]}",
                lambda h=h, rows=rows, c=c, pair=pair, extra=extra: fitters[
                    h
                ].fit_bayes(
                    x_fit[rows],
                    frame[rows].reset_index(drop=True),
                    c=c,
                    n=data.n[rows],
                    priors=own[pair],
                    method=method,
                    importance=importance,
                    **extra,
                ),
            )
        except ValueError as err:
            name = tm.transition_names()[h]
            raise ValueError(f"Transition {name}: {err}") from None
    return BayesianMultiStateRegression(models, mle)


__all__ = [
    "BayesianMultiState",
    "BayesianMultiStateRegression",
    "DRAWS",
    "SEED",
    "fit",
    "fit_regression",
]
