"""A Cox model for each transition of a multi-state model, and the
transition and state occupation probabilities it predicts."""

from __future__ import annotations

import warnings
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

_VARIANCES = ("aalen", "greenwood")
_TIES = ("efron", "breslow")


class MultiStateProportionalHazards(SerialisableMixin):
    r"""
    Multi-state proportional-hazards regression: a Cox model for each
    transition, and the transition and state occupation probabilities
    predicted from them for a unit with given covariates.

    The hazard of the transition :math:`h` (from state :math:`g` to state
    :math:`k`) of a unit with covariates :math:`z` is

    .. math::
        \lambda_h(t \mid z) = \lambda_{h,0}(t) \exp(\beta_h^\top z),

    each transition with its own baseline hazard and, by default, its own
    coefficients :math:`\beta_h`. A covariate named in ``shared`` has one
    coefficient for every transition instead. The model is fitted as one
    Cox model to the stacked data (one row for each transition a spell
    could have ended in; see :func:`~surpyval.multistate.to_stacked`),
    stratified by transition, with each covariate expanded into one
    column per transition (Andersen and Keiding, 2002; Putter, Fiocco and
    Geskus, 2007), which is R ``mstate``'s approach. With every effect
    transition-specific the coefficients are those of a separate Cox
    model of each transition.

    The prediction for a covariate row :math:`z` is the Aalen-Johansen
    product integral of the predicted cumulative hazards,
    :math:`dA_h(t \mid z) = d\hat{\Lambda}_{h,0}(t)
    \exp(\hat{\beta}_h^\top z)`, and its standard errors are those of
    ``mstate``'s ``msfit`` and ``probtrans``: they include the
    uncertainty of the coefficients. Like the non-parametric
    :class:`~surpyval.multistate.MultiState` it assumes the process is
    Markov.

    Call the class method ``MultiStateProportionalHazards.fit`` (or
    ``fit_from_df``); it returns a fitted instance.

    References
    ----------
    Andersen, P. K. and Keiding, N. (2002), "Multi-state models for event
    history analysis", *Statistical Methods in Medical Research*, 11(2),
    91-115.

    Putter, H., Fiocco, M. and Geskus, R. B. (2007), "Tutorial in
    biostatistics: competing risks and multi-state models", *Statistics
    in Medicine*, 26(11), 2389-2430.

    de Wreede, L. C., Fiocco, M. and Putter, H. (2010), "The mstate
    package for estimation and prediction in non- and semi-parametric
    multi-state and competing risks models", *Computer Methods and
    Programs in Biomedicine*, 99(3), 261-274.
    """

    #: The states and the allowed transitions.
    transitions: TransitionMatrix
    #: The distinct event times (of any transition).
    t: npt.NDArray
    #: ``(n_transitions, p)``: each transition's coefficients, one per
    #: column of ``Z`` (a shared covariate's repeated on every row).
    betas: npt.NDArray
    #: The stratified Cox model fitted to the stacked data.
    cox: Any
    #: The covariates' names (``z0``, ``z1``, ... unless named).
    covariate_names: list
    #: The columns of ``Z`` whose effect is shared by every transition.
    shared: tuple
    #: ``"efron"`` or ``"breslow"``: the ties of the partial likelihood
    #: and of the baseline hazards.
    tie_method: str
    #: The earliest entry: the default start ``s`` of a prediction.
    start: float
    #: The last observed time.
    last: float
    _data: SpellData
    _Z: npt.NDArray
    _beta: npt.NDArray
    _stacked: dict
    _baselines: dict
    _cache: dict

    # -- fitting ------------------------------------------------------------

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        state: npt.ArrayLike,
        tl: npt.ArrayLike | float | None = None,
        id: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        transitions: Any = None,
        shared: Any = None,
        tie_method: str = "efron",
    ) -> MultiStateProportionalHazards:
        """
        Fit a Cox model to each transition of spells in states.

        Parameters
        ----------
        x, e, state, tl, id, n, transitions
            The spells, as for :meth:`MultiState.fit
            <surpyval.multistate.MultiState.fit>`.
        Z : array_like
            The covariates, one row per spell (a unit's covariates on each
            of its spells). Missing values are refused: dropping a spell
            would cut a unit's path through its states.
        shared : sequence, optional
            The covariates (columns of ``Z``, by position, or by name for
            ``fit_from_df``) whose effect is the same on every transition:
            one coefficient for all of them. The others have one per
            transition (the default for all).
        tie_method : ('efron', 'breslow'), str, optional
            The handling of tied event times, in the partial likelihood
            and the baseline hazards, as :meth:`CoxPH.fit
            <surpyval.CoxPH.fit>`. Defaults to ``'efron'``.

        Returns
        -------
        MultiStateProportionalHazards
            The fitted model.

        Examples
        --------
        The MGUS data (``load_mgus2``): does a patient's sex change the
        hazard of progressing to a plasma-cell malignancy, of dying
        without it, or of dying after it?

        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import (
        ...     MultiStateProportionalHazards, TransitionMatrix,
        ... )
        >>> df = load_mgus2(spells=True)
        >>> tm = TransitionMatrix.illness_death(("mgus", "pcm", "death"))
        >>> model = MultiStateProportionalHazards.fit(
        ...     df["x"], df[["male"]], df["e"], df["state"], tl=df["tl"],
        ...     id=df["id"], transitions=tm,
        ... )
        >>> model.parameter_names
        ['male: mgus -> pcm', 'male: mgus -> death', 'male: pcm -> death']
        >>> model.params.round(3)
        array([-0.059,  0.228,  0.028])

        The probability that a man diagnosed with MGUS has progressed and
        is alive ten years (120 months) on:

        >>> model.occupancy(120, [1], "pcm", from_state="mgus").round(4)
        np.float64(0.0105)
        """
        check_option("tie_method", tie_method, _TIES)
        data = check_spells(x, e, state, tl, id, n, transitions)
        Z_arr, names = _covariates(Z, data.x.size)
        shared_cols = _shared_columns(shared, names)
        return cls._fit(data, Z_arr, names, shared_cols, tie_method)

    @classmethod
    def _fit(
        cls,
        data: SpellData,
        Z: npt.NDArray,
        names: list,
        shared: tuple,
        tie_method: str,
    ) -> MultiStateProportionalHazards:
        from surpyval import CoxPH

        tm = data.tm
        n_trans = tm.n_transitions
        frm, to = tm.from_index(), tm.to_index()
        counts = np.zeros(n_trans)
        for h in range(n_trans):
            counts[h] = data.n[(data.g == frm[h]) & (data.h == to[h])].sum()
        if np.any(counts == 0):
            none = [tm.transitions[h] for h in np.flatnonzero(counts == 0)]
            raise ValueError(
                f"The transitions {none!r} have no events, so their "
                "hazards (and coefficients) cannot be estimated: leave "
                "them out of 'transitions'."
            )
        out = cls()
        out.transitions = tm
        out.covariate_names = list(names)
        out.shared = tuple(shared)
        out.tie_method = tie_method
        out._data = data
        out._Z = Z
        stacked = out._stack()
        tl = stacked["tl"]
        finite = np.isfinite(tl)
        if not finite.any():
            tl_arg = None
        else:
            low = min(tl[finite].min(), stacked["x"].min()) - 1.0
            tl_arg = np.where(finite, tl, low)
        from surpyval.univariate.regression._aliasing import (
            collect_aliased,
            covariate_columns,
            warn_aliased,
        )

        with covariate_columns(None), collect_aliased() as found:
            out.cox = CoxPH.fit(
                stacked["x"],
                stacked["U"],
                1 - stacked["status"],
                stacked["n"],
                tl=tl_arg,
                tie_method=tie_method,
                strata=stacked["trans"],
            )
        if found:
            # Said of the columns of Z, not of their expansion.
            named = names != [f"z{j}" for j in range(len(names))]
            with covariate_columns(names if named else None):
                warn_aliased(
                    out.aliased,
                    f"{found[0][1]}, in the Cox model of each transition",
                )
        beta = np.asarray(out.cox.beta, dtype=float)
        out._beta = np.where(np.isnan(beta), 0.0, beta)
        out.betas = (
            np.vstack(
                [
                    beta[out._design_rows(h)].reshape(len(names))
                    for h in range(n_trans)
                ]
            )
            if names
            else np.zeros((n_trans, 0))
        )
        table = _core.event_table(data)
        out.t = table.t
        out.start = float(np.min(data.tl))
        out.last = float(np.max(data.x))
        out._stacked = stacked
        out._baselines = out._baseline_parts()
        out._cache = {}
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
        id_col: str | None = None,
        n_col: str | None = None,
        transitions: Any = None,
        shared: Any = None,
        tie_method: str = "efron",
    ) -> MultiStateProportionalHazards:
        """
        Fit the model to a DataFrame of spells, one row each; ``Z_cols``
        names the covariate columns, and ``shared`` may name some of
        them. The other columns are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> df = pd.DataFrame({
        ...     "stop": [2, 6, 3, 9, 4, 7, 5, 8],
        ...     "to": ["ill", "dead", "ill", None, "dead", "ill", "dead",
        ...            "dead"],
        ...     "state": ["healthy", "ill", "healthy", "ill", "healthy",
        ...               "healthy", "healthy", "ill"],
        ...     "start": [0, 2, 0, 3, 0, 0, 0, 7],
        ...     "unit": [1, 1, 2, 2, 3, 4, 5, 4],
        ...     "age": [60, 60, 70, 70, 65, 50, 75, 50],
        ... })
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     df, "stop", ["age"], "to", "state", tl_col="start",
        ...     id_col="unit", shared=["age"],
        ... )
        >>> model.parameter_names
        ['age']
        """
        from surpyval.utils.dataframe import frame_column

        def col(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        Z = np.column_stack([frame_column(df, c, "Z_cols") for c in cols])
        check_option("tie_method", tie_method, _TIES)
        data = check_spells(
            col(x_col, "x_col"),
            col(e_col, "e_col"),
            col(state_col, "state_col"),
            col(tl_col, "tl_col"),
            col(id_col, "id_col"),
            col(n_col, "n_col"),
            transitions,
        )
        Z_arr, _ = _covariates(Z, data.x.size)
        names = [str(c) for c in cols]
        return cls._fit(
            data, Z_arr, names, _shared_columns(shared, names), tie_method
        )

    # -- the design ---------------------------------------------------------

    def _columns(self) -> list[tuple[int, int | None]]:
        """The expanded columns: ``(covariate, transition)``, the
        transition None for a shared covariate."""
        out: list[tuple[int, int | None]] = []
        for j in range(len(self.covariate_names)):
            if j in self.shared:
                out.append((j, None))
            else:
                out += [(j, h) for h in range(self.transitions.n_transitions)]
        return out

    def _design_rows(self, h: int) -> npt.NDArray:
        """The expanded column of each covariate for transition ``h``."""
        cols = self._columns()
        return np.array(
            [
                next(
                    i
                    for i, (jj, hh) in enumerate(cols)
                    if jj == j and hh in (None, h)
                )
                for j in range(len(self.covariate_names))
            ],
            dtype=np.intp,
        )

    def _expand(self, z: npt.NDArray, h: int) -> npt.NDArray:
        """Covariate rows ``z`` (``(N, p)``) as the expanded design of
        transition ``h``."""
        cols = self._columns()
        U = np.zeros((z.shape[0], len(cols)))
        for i, (j, hh) in enumerate(cols):
            if hh is None or hh == h:
                U[:, i] = z[:, j]
        return U

    def _stack(self) -> dict:
        d = self._data
        tm = self.transitions
        frm, to = tm.from_index(), tm.to_index()
        parts: dict = {
            k: [] for k in ("x", "tl", "status", "n", "trans", "U", "row")
        }
        for h in range(tm.n_transitions):
            rows = np.flatnonzero(d.g == frm[h])
            parts["x"].append(d.x[rows])
            parts["tl"].append(d.tl[rows])
            parts["status"].append((d.h[rows] == to[h]).astype(int))
            parts["n"].append(d.n[rows])
            parts["trans"].append(np.full(rows.size, h))
            parts["U"].append(self._expand(self._Z[rows], h))
            parts["row"].append(rows)
        return {k: np.concatenate(v) for k, v in parts.items()}

    @property
    def parameter_names(self) -> list[str]:
        """The coefficients' names: the covariate and, for a
        transition-specific one, the transition (``"age: healthy ->
        ill"``)."""
        names = self.transitions.transition_names()
        return [
            (
                self.covariate_names[j]
                if h is None
                else f"{self.covariate_names[j]}: {names[h]}"
            )
            for j, h in self._columns()
        ]

    @property
    def params(self) -> npt.NDArray:
        """The coefficients, in the order of ``parameter_names`` (NaN for
        an aliased one)."""
        return np.asarray(self.cox.beta, dtype=float)

    def covariance(self) -> npt.NDArray:
        """The coefficients' covariance (the inverse of the observed
        information of the partial likelihood)."""
        return np.asarray(self.cox.covariance(), dtype=float)

    def standard_errors(self) -> npt.NDArray:
        """The coefficients' standard errors."""
        return np.sqrt(np.diag(self.covariance()))

    @property
    def maximum(self) -> str:
        """What the fit of the Cox model reached (as ``CoxPH``'s)."""
        return self.cox.maximum

    def neg_ll(self) -> float:
        """The negative of the maximised partial log-likelihood: the sum
        of the transitions' (the stratified model's)."""
        return float(self.cox.neg_ll())

    @property
    def log_likelihood(self) -> float:
        """The maximised partial log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def aic(self) -> float:
        """Akaike's information criterion of the partial likelihood, as
        ``CoxPH``'s: every estimated coefficient counts."""
        return float(self.cox.aic())

    def aic_c(self) -> float:
        """The small-sample corrected AIC, as ``CoxPH``'s."""
        return float(self.cox.aic_c())

    def bic(self) -> float:
        """The Bayesian information criterion, as ``CoxPH``'s (its sample
        size the number of events)."""
        return float(self.cox.bic())

    @property
    def p_values(self) -> npt.NDArray:
        """The coefficients' Wald p-values."""
        return np.asarray(self.cox.p_values, dtype=float)

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """The coefficients' table (as :meth:`CoxPH`'s ``summary``), one
        row each, named by ``parameter_names``."""
        table = self.cox.summary(alpha_ci=alpha_ci)
        table.index = self.parameter_names
        table.index.name = "covariate"
        return table

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficient the data cannot
        determine (for some transition): NaN in ``params`` and
        ``betas``."""
        cols = self._columns()
        nan = np.isnan(np.asarray(self.cox.beta, dtype=float))
        return np.array(
            sorted({cols[i][0] for i in np.flatnonzero(nan)}), dtype=int
        )

    # -- the baselines ------------------------------------------------------

    def _baseline_parts(self) -> dict:
        """Per transition, at each event time: the baseline hazard
        increment, the variance term and :math:`\\bar{x}` times the
        increment (survival's ``agsurv``)."""
        st = self._stacked
        t = self.t
        m = t.size
        q = st["U"].shape[1]
        lp = st["U"] @ self._beta
        out: dict = {"haz": [], "varhaz": [], "xbar": []}
        for h in range(self.transitions.n_transitions):
            rows = st["trans"] == h
            x, tl = st["x"][rows], st["tl"][rows]
            n, U = st["n"][rows], st["U"][rows]
            dead = st["status"][rows] == 1
            w = n * np.exp(lp[rows])
            first = np.searchsorted(t, tl, side="right")
            last = np.searchsorted(t, x, side="right") - 1
            ok = first <= last
            S0 = np.zeros(m + 1)
            S1 = np.zeros((m + 1, q))
            np.add.at(S0, first[ok], w[ok])
            np.add.at(S0, last[ok] + 1, -w[ok])
            np.add.at(S1, first[ok], (w[:, None] * U)[ok])
            np.add.at(S1, last[ok] + 1, -(w[:, None] * U)[ok])
            S0, S1 = np.cumsum(S0)[:m], np.cumsum(S1, axis=0)[:m]
            nd = np.zeros(m)
            D0 = np.zeros(m)
            D1 = np.zeros((m, q))
            np.add.at(nd, last[dead], n[dead])
            np.add.at(D0, last[dead], w[dead])
            np.add.at(D1, last[dead], (w[:, None] * U)[dead])
            haz = np.zeros(m)
            varhaz = np.zeros(m)
            xbar = np.zeros((m, q))
            for j in np.flatnonzero(nd > 0):
                if self.tie_method == "breslow":
                    haz[j] = nd[j] / S0[j]
                    varhaz[j] = nd[j] / S0[j] ** 2
                    xbar[j] = S1[j] / S0[j] * haz[j]
                else:
                    k = int(round(nd[j]))
                    frac = np.arange(k) / k
                    den = S0[j] - frac * D0[j]
                    haz[j] = np.sum(1.0 / den) * nd[j] / k
                    varhaz[j] = np.sum(1.0 / den**2) * nd[j] / k
                    num = S1[j][None, :] - frac[:, None] * D1[j][None, :]
                    xbar[j] = np.sum(num / den[:, None] ** 2, axis=0) * (
                        nd[j] / k
                    )
            out["haz"].append(haz)
            out["varhaz"].append(varhaz)
            out["xbar"].append(xbar)
        return {k: np.array(v) for k, v in out.items()}

    def _profile(self, z: npt.NDArray) -> dict:
        """The predicted increments and their covariance for one
        covariate row ``z``."""
        key = ("profile", tuple(z.tolist()))
        if key in self._cache:
            return self._cache[key]
        tm = self.transitions
        n_trans = tm.n_transitions
        k = tm.n_states
        frm, to = tm.from_index(), tm.to_index()
        base = self._baselines
        m = self.t.size
        phi = np.empty(n_trans)
        U = np.empty((n_trans, base["xbar"].shape[2]))
        for h in range(n_trans):
            U[h] = self._expand(z[None, :], h)[0]
            phi[h] = np.exp(U[h] @ self._beta)
        dH = base["haz"] * phi[:, None]  # (n_trans, m)
        dA = np.zeros((m, k, k))
        dA[:, frm, to] = dH.T
        dA = _core.with_diagonal(dA)
        # Where the predicted hazards of leaving a state add up to more
        # than 1, I + dA has a negative diagonal (mstate warns so too).
        over = np.flatnonzero(
            np.any(np.diagonal(dA, axis1=1, axis2=2) < -1, 1)
        )
        first_over = float(self.t[over[0]]) if over.size else np.inf
        # Cumulative covariances of the cumulative hazards (msfit).
        varh = np.cumsum(base["varhaz"] * phi[:, None] ** 2, axis=1)
        qv = np.cumsum(
            (base["xbar"] - U[:, None, :] * base["haz"][:, :, None])
            * phi[:, None, None],
            axis=1,
        )  # (n_trans, m, q)
        V = self.covariance()
        V = np.where(np.isnan(V), 0.0, V)
        cumcov = np.einsum("hjq,qr,gjr->jhg", qv, V, qv)
        idx = np.arange(n_trans)
        cumcov[:, idx, idx] += varh.T
        inc = np.diff(cumcov, axis=0, prepend=0.0)
        cov = np.zeros((m, k, k, k, k))
        for h in range(n_trans):
            for g in range(n_trans):
                cov[:, frm[h], to[h], frm[g], to[g]] = inc[:, h, g]
        cov_vec = _core._fill_diagonals(cov)
        out = {"dA": dA, "cov": cov_vec, "over": first_over}
        self._cache[key] = out
        return out

    # -- predictions --------------------------------------------------------

    @property
    def states(self) -> tuple:
        """The states, in order."""
        return self.transitions.states

    def _rows(self, Z: Any, size: int) -> npt.NDArray:
        """The covariate rows of a query of ``size`` times: one row for
        every time, or one per time."""
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

    def _start(self, s: float | None) -> float:
        if s is None:
            return self.start
        s = float(s)
        if np.isnan(s):
            raise ValueError("'s', the start of the prediction, is NaN.")
        return s

    def _p0(self, s: float, from_state: Any) -> npt.NDArray:
        k = self.transitions.n_states
        if from_state is not None:
            p0 = np.zeros(k)
            p0[self.transitions.index(from_state)] = 1.0
            return p0
        key = ("p0", s)
        if key not in self._cache:
            self._cache[key] = _core.at_risk_distribution(self._data, s)[0]
        return self._cache[key]

    def _P(self, z: npt.NDArray, s: float) -> npt.NDArray:
        key = ("P", tuple(z.tolist()), s)
        if key not in self._cache:
            j0 = _core.start_index(self.t, s)
            dA = self._profile(z)["dA"]
            self._cache[key] = _core.product_integral(dA, j0)
        return self._cache[key]

    def _V(self, z: npt.NDArray, s: float, variance: str) -> npt.NDArray:
        key = ("V", tuple(z.tolist()), s, variance)
        if key not in self._cache:
            j0 = _core.start_index(self.t, s)
            prof = self._profile(z)
            V = _core.propagate_variance(prof["dA"], prof["cov"], j0, variance)
            k = self.transitions.n_states
            V = V.reshape(-1, k, k, k, k)
            idx = np.arange(k)
            self._cache[key] = V[:, idx, :, idx, :].transpose(1, 0, 2, 3)
        return self._cache[key]

    def _by_row(
        self,
        x: npt.NDArray,
        Z: Any,
        s: float,
        value: Any,
        check: bool = True,
    ) -> npt.NDArray:
        """``value(z, j0, idx)`` for each distinct covariate row, at the
        query times; NaN for a row with a missing covariate or a time
        before ``s``. With ``check``, one warning for the rows queried
        past a time where their predicted probabilities can leave [0, 1].
        """
        rows = self._rows(Z, x.size)
        out = np.full(x.size, np.nan)
        idx = step_index(self.t, x)
        bad = np.isnan(x) | (x < s) | np.any(np.isnan(rows), axis=1)
        uniq, inverse = np.unique(
            np.where(np.isnan(rows), 0.0, rows), axis=0, return_inverse=True
        )
        inverse = np.asarray(inverse).ravel()
        j0 = _core.start_index(self.t, s)
        flagged: list = []
        for u in range(uniq.shape[0]):
            sel = (inverse == u) & ~bad
            if np.any(sel):
                out[sel] = value(uniq[u], j0, idx[sel])
                first = self._profile(uniq[u])["over"]
                if check and np.any(x[sel] >= first):
                    flagged.append((uniq[u].tolist(), first))
        if flagged:
            _warn_over(flagged)
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
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", id_col="id",
        ... )
        >>> model.states
        ('death', 'mgus', 'pcm')
        >>> model.transition_probabilities(120, [1]).round(3)
        array([[1.   , 0.   , 0.   ],
               [0.62 , 0.37 , 0.01 ],
               [0.99 , 0.   , 0.01 ]])
        """
        s = self._start(s)
        z = np.asarray(Z, dtype=float).ravel()
        if z.size != len(self.covariate_names):
            raise ValueError(
                f"Z must be one row of {len(self.covariate_names)} "
                "covariate(s)."
            )
        xs = np.asarray(x, dtype=float)
        flat = xs.ravel()
        k = self.transitions.n_states
        P = self._P(z, s)
        idx = step_index(self.t, flat)
        j0 = _core.start_index(self.t, s)
        out = np.where(
            (idx < j0)[:, None, None],
            np.eye(k),
            P[np.clip(idx, 0, max(self.t.size - 1, 0))],
        )
        out = np.where(
            (np.isnan(flat) | (flat < s))[:, None, None], np.nan, out
        )
        first = self._profile(z)["over"]
        if np.any(flat >= first):
            _warn_over([(z.tolist(), first)])
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
        being in ``state`` at ``x`` for a unit with covariates ``Z``.

        Parameters
        ----------
        x : array_like
            The times.
        Z : array_like
            The covariates: one row for every time, or one row per time.
        state : any
            The state.
        s : float, optional
            The time the prediction starts from. Defaults to the earliest
            entry.
        from_state : any, optional
            The unit's state at ``s``. By default the states of the units
            under observation at ``s``, in proportion (as for
            :meth:`MultiState.occupancy
            <surpyval.multistate.MultiState.occupancy>`).

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", id_col="id",
        ... )

        For a man and a woman, ten years on:


        >>> model.occupancy([120, 120], [[1], [0]], "pcm").round(4)
        array([0.0105, 0.0131])
        """
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        x = np.asarray(x, dtype=float)

        def value(z: npt.NDArray, j0: int, idx: npt.NDArray) -> npt.NDArray:
            p = p0 @ self._P(z, s)[:, :, b].T
            return np.where(idx < j0, p0[b], p[np.maximum(idx, 0)])

        return self._by_row(x, Z, s, value)

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
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", id_col="id",
        ... )
        >>> model.cumulative_hazard(120, [1], "mgus", "pcm").round(4)
        np.float64(0.0971)
        """
        h = self.transitions.transition_index(from_state, to_state)
        x = np.asarray(x, dtype=float)

        def value(z: npt.NDArray, j0: int, idx: npt.NDArray) -> npt.NDArray:
            prof = self._profile(z)["dA"]
            a = self.transitions.index(from_state)
            b = self.transitions.index(to_state)
            H = np.cumsum(prof[:, a, b])
            return np.where(idx < 0, 0.0, H[np.maximum(idx, 0)])

        del h
        return self._by_row(x, Z, -np.inf, value, check=False)

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        variance: str = "aalen",
    ) -> npt.NDArray:
        """
        The standard error of :meth:`occupancy`, which includes the
        uncertainty of the coefficients.

        Parameters
        ----------
        x, Z, state, s, from_state
            As for :meth:`occupancy`.
        variance : ('aalen', 'greenwood'), str, optional
            R ``mstate``'s ``probtrans(..., method=)``: ``'aalen'`` (the
            default, as there) or ``'greenwood'``. Both take the
            covariance of the predicted cumulative hazards of ``msfit``
            and the starting distribution as known. There is no robust
            (infinitesimal-jackknife) variance for this model.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", id_col="id",
        ... )
        >>> model.occupancy_se(240, [1], "death").round(4)
        np.float64(0.016)
        """
        if variance == "robust":
            raise ValueError(
                "There is no robust (infinitesimal-jackknife) variance for "
                "MultiStateProportionalHazards: use variance='aalen' or "
                "'greenwood', which include the coefficients' "
                "uncertainty, or MultiState's robust variance without "
                "covariates."
            )
        check_option("variance", variance, _VARIANCES)
        s = self._start(s)
        b = self.transitions.index(state)
        p0 = self._p0(s, from_state)
        x = np.asarray(x, dtype=float)

        def value(z: npt.NDArray, j0: int, idx: npt.NDArray) -> npt.NDArray:
            blocks = self._V(z, s, variance)[:, b]
            var = np.einsum("a,jab,b->j", p0, blocks, p0)
            se = np.sqrt(np.maximum(var, 0.0))
            return np.where(idx < j0, 0.0, se[np.maximum(idx, 0)])

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
        variance: str = "aalen",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on the predicted state occupation
        probability :meth:`occupancy`.

        Parameters
        ----------
        x, Z, state, s, from_state
            As for :meth:`occupancy`.
        alpha_ci, bound, bound_type
            As for :meth:`MultiState.cb <surpyval.multistate.MultiState.cb>`.
        variance : ('aalen', 'greenwood'), str, optional
            The variance; see :meth:`occupancy_se`.

        Examples
        --------
        >>> from surpyval.datasets import load_mgus2
        >>> from surpyval.multistate import MultiStateProportionalHazards
        >>> model = MultiStateProportionalHazards.fit_from_df(
        ...     load_mgus2(spells=True), "x", ["male"], "e", "state",
        ...     tl_col="tl", id_col="id",
        ... )
        >>> model.cb(240, [1], "death").round(4)
        array([0.8079, 0.8707])
        """
        check_bound_args(alpha_ci, bound, bound_type)
        x = np.asarray(x, dtype=float)
        p = self.occupancy(x, Z, state, s, from_state)
        with warnings.catch_warnings():
            # (occupancy has said where the probabilities can leave [0, 1])
            warnings.simplefilter("ignore")
            se = self.occupancy_se(x, Z, state, s, from_state, variance)
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

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict. The data
        (the spells and their covariates) are stored and the model is
        refitted on loading, so it predicts exactly as the original.
        """
        d = self._data
        tm = self.transitions
        out = {
            "model": "MultiStateProportionalHazards",
            "transitions": tm.to_dict(),
            "x": d.x.tolist(),
            "tl": d.tl.tolist(),
            "state": [to_native(tm.states[g]) for g in d.g],
            "e": [None if h < 0 else to_native(tm.states[h]) for h in d.h],
            "n": d.n.tolist(),
            "id": d.cluster.tolist() if d.ids_given else None,
            "Z": self._Z.tolist(),
            "covariate_names": list(self.covariate_names),
            "shared": [int(j) for j in self.shared],
            "tie_method": self.tie_method,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> MultiStateProportionalHazards:
        """Rebuild the model from :meth:`to_dict`."""
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        require_model_tag(
            model_dict,
            "MultiStateProportionalHazards",
            "a multi-state proportional-hazards model",
        )
        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        tl = [-np.inf if v is None else v for v in model_dict["tl"]]
        data = check_spells(
            model_dict["x"],
            [label_from_native(v) for v in model_dict["e"]],
            [label_from_native(v) for v in model_dict["state"]],
            tl,
            model_dict["id"],
            model_dict["n"],
            tm,
        )
        Z = np.array(model_dict["Z"], dtype=float).reshape(
            len(model_dict["x"]), -1
        )
        return cls._fit(
            data,
            Z,
            list(model_dict["covariate_names"]),
            tuple(model_dict["shared"]),
            model_dict["tie_method"],
        )

    def __repr__(self) -> str:
        tm = self.transitions
        lines = [
            "Multi-state proportional-hazards model (a Cox model per "
            "transition)",
            f"States: {list(tm.states)!r}",
            f"Transitions: {', '.join(tm.transition_names())}",
            f"Ties: {self.tie_method}",
            "Coefficients:",
        ]
        for name, value in zip(self.parameter_names, self.params):
            lines.append(f"    {name}: {value:.6g}")
        return "\n".join(lines)


def _warn_over(flagged: list) -> None:
    """Say where predicted probabilities can leave [0, 1]."""
    from surpyval.utils.warnings import caller_stacklevel

    z, first = flagged[0]
    more = f" (and {len(flagged) - 1} other rows)" if len(flagged) > 1 else ""
    warnings.warn(
        f"For the covariates {z}{more}, the predicted hazards of leaving a "
        f"state add up to more than 1 at time {first:g} and later, so the "
        "predicted probabilities there can leave [0, 1]. This happens "
        "where few are at risk or the covariates are far from the data's; "
        "predict to an earlier time, or for covariates within the data.",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def _covariates(Z: Any, size: int) -> tuple[npt.NDArray, list]:
    """``Z`` as an ``(N, p)`` float array and its column names."""
    names = None
    if hasattr(Z, "columns"):
        names = [str(c) for c in Z.columns]
    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != size:
        raise ValueError(
            f"Z must have one row of covariates per spell ({size}); got "
            f"shape {np.shape(Z)}."
        )
    bad = ~np.isfinite(arr)
    if np.any(bad):
        rows = np.flatnonzero(bad.any(axis=1))
        raise ValueError(
            f"Z has missing or infinite covariates on {rows.size} rows "
            f"(the first is row {int(rows[0])}). A multi-state model "
            "cannot drop a spell without cutting a unit's path through "
            "its states: impute them, or leave out the units they belong "
            "to."
        )
    if names is None:
        names = [f"z{j}" for j in range(arr.shape[1])]
    return arr, names


def _shared_columns(shared: Any, names: list) -> tuple:
    if shared is None:
        return ()
    if isinstance(shared, (str, int, np.integer)):
        shared = [shared]
    out = []
    for item in shared:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            if not 0 <= int(item) < len(names):
                raise ValueError(
                    f"shared names column {item}, but Z has {len(names)} "
                    "column(s)."
                )
            out.append(int(item))
        elif item in names:
            out.append(names.index(item))
        else:
            raise ValueError(
                f"shared names {item!r}, which is not a covariate; the "
                f"covariates are {names!r}."
            )
    return tuple(sorted(set(out)))
