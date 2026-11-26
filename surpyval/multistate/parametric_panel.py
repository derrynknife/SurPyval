"""Parametric multi-state models fitted to states seen only at visits
(panel data): a parametric distribution for each transition's hazard,
on the time since the start (a non-homogeneous Markov model) or on the
time in the state (a semi-Markov model)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate._entry import refuses_surpyval_data
from surpyval.multistate.parametric import (
    ParametricMultiState,
    ParametricMultiStateRegression,
    _check_clock,
    _check_reset,
    _fitters,
    _lows,
)
from surpyval.serialisation import require_model_tag, stamp_schema, to_native
from surpyval.utils.units import takes_unit

#: The accuracy of each row of the transition probabilities on the
#: forward clock's grid (see ``surpyval.multistate._forward``).
PANEL_TOL = 1e-9


def _exponential_starts(pdata: Any, fitters: list) -> list:
    """Each transition's starting model: its distribution fitted to the
    quantiles of the exponential with the intensity of a time-homogeneous
    Markov model (``PanelMultiState``) fitted to the same visits. Starting
    values only: the search then maximises the model's own likelihood."""
    from surpyval.multistate.panel import PanelMultiState
    from surpyval.utils.warnings import ignoring_warnings

    with ignoring_warnings():
        markov = PanelMultiState._fit(pdata, [], None)
        q = np.ravel(markov.params)[: pdata.tm.n_transitions]
    span = float(pdata.x.max() - pdata.x.min()) or 1.0
    p = (np.arange(1, 41) - 0.5) / 40.0
    out = []
    for fitter, rate in zip(fitters, q):
        # (an intensity at the boundary: one move in a hundred spans)
        rate = float(rate) if np.isfinite(rate) and rate > 0 else 0.01 / span
        x = -np.log1p(-p) / rate
        with ignoring_warnings():
            out.append(fitter.fit(x=x))
    return out


def _regression_starts(pdata: Any, fitters: list, names: list) -> list:
    """Each transition's starting regression: fitted to the quantiles of
    the exponential with the intensity, for each of up to 20 of the
    gaps' covariate rows, of a time-homogeneous Markov model with
    covariates (``PanelMultiStateRegression``) fitted to the same visits.
    Starting values only: the search then maximises the model's own
    likelihood."""
    import pandas as pd

    from surpyval.multistate.panel import PanelMultiStateRegression
    from surpyval.univariate.regression._aliasing import (
        collect_aliased,
        warn_collected,
    )
    from surpyval.utils.warnings import ignoring_warnings

    H = pdata.tm.n_transitions
    with ignoring_warnings():
        markov = PanelMultiStateRegression._fit(pdata, list(names), None)
        q = np.ravel(markov.params)[:H]
        B = np.nan_to_num(np.asarray(markov.betas, dtype=float))
    rows = pdata.intervals() - 1
    U = np.unique(pdata.Z[rows], axis=0)
    if U.shape[0] > 20:
        U = U[np.linspace(0, U.shape[0] - 1, 20).round().astype(int)]
    span = float(pdata.x.max() - pdata.x.min()) or 1.0
    p = (np.arange(1, 21) - 0.5) / 20.0
    out: list = []
    found: list = []
    for h, fitter in enumerate(fitters):
        rate = float(q[h]) if np.isfinite(q[h]) and q[h] > 0 else 0.0
        rate = rate if rate > 0 else 0.01 / span
        rates = rate * np.exp(np.clip(U @ B[h], -30.0, 30.0))
        x = (-np.log1p(-p)[None, :] / rates[:, None]).ravel()
        Z = pd.DataFrame(np.repeat(U, p.size, axis=0), columns=list(names))
        # (a column the gaps' covariates cannot tell from the others, or
        # from a constant, is aliased in these rows too, and is reported)
        with collect_aliased() as aliased:
            with ignoring_warnings():
                out.append(fitter.fit(x=x, Z=Z))
        found += aliased
    warn_collected(found, "in the fit of each transition")
    return out


def _patterns(pdata: Any) -> tuple[npt.NDArray, npt.NDArray]:
    """``(pattern, rows)``: the covariate pattern of the gap that starts
    at each row (0 for a row that starts none) and the distinct rows of
    covariates, one per pattern."""
    pattern = np.zeros(pdata.x.size, dtype=np.intp)
    Z = pdata.Z
    if Z.shape[1] == 0:
        return pattern, np.zeros((1, 0))
    starts = pdata.intervals() - 1
    if starts.size == 0:
        return pattern, Z[:1]
    rows, inverse = np.unique(Z[starts], axis=0, return_inverse=True)
    pattern[starts] = np.ravel(inverse)
    return pattern, rows


class _PanelFit:
    """The fit to states seen at visits that the parametric panel models
    share (with or without covariates): the likelihood on either clock,
    the joint search and what is counted."""

    _panel: Any
    _regression: bool
    _joint: Any

    @classmethod
    def _likelihood(cls, pdata: Any, fitters: list, clock: str) -> Any:
        from surpyval.multistate import _panel_likelihood as pl

        pattern, rows = _patterns(pdata)
        Hf = cls._Hf  # type: ignore[attr-defined]

        def hazard(m: Any, t: npt.NDArray, rate: bool, p: int) -> Any:
            t = np.asarray(t, dtype=float)
            return np.asarray(Hf(m, t.ravel(), rows[p], rate)).reshape(t.shape)

        if clock == "reset":
            if pl.has_cycle(pdata.tm):
                raise ValueError(
                    "With clock='reset' the panel model is for progressive "
                    "models only (no state can be come back to): with a way "
                    "back, a unit could pass through any number of states "
                    "unseen between two visits, each entered at an unknown "
                    "time. Use clock='forward' (a Markov model, which takes "
                    "cycles), or PanelMultiState."
                )
            _check_reset(
                fitters, pdata.tm, "model" if cls._regression else "dist"
            )
            return pl.ResetPanelLikelihood(
                pdata, hazard, pattern, cls._batched(rows)
            )

        def hazards(models: list, p: int) -> Any:
            def H(t: npt.NDArray, rate: bool) -> npt.NDArray:
                out = np.empty((t.size, len(models)))
                with np.errstate(all="ignore"):
                    for h, m in enumerate(models):
                        out[:, h] = np.ravel(hazard(m, t, rate, p))
                return out

            return H

        one = cls._batched(rows)

        def batched(
            models: list, t: npt.NDArray, p: npt.NDArray, rate: bool
        ) -> npt.NDArray:
            out = np.empty((t.size, len(models)))
            with np.errstate(all="ignore"):
                for h, m in enumerate(models):
                    out[:, h] = np.ravel(one(m, t, p, rate))
            return out

        return pl.ForwardPanelLikelihood(
            pdata,
            hazards,
            _lows(fitters),
            PANEL_TOL,
            pattern,
            None if one is None else batched,
        )

    @classmethod
    def _batched(cls, rows: npt.NDArray) -> Any:
        """``hazard(model, t, p, rate)`` of a model with covariates: the
        transition's hazards at the times ``t`` for the covariate rows
        ``rows[p]`` in one evaluation (``None`` without covariates)."""
        if not cls._regression:
            return None

        def hazard(
            model: Any, t: npt.NDArray, p: npt.NDArray, rate: bool
        ) -> npt.NDArray:
            f = model.hf if rate else model.Hf
            return np.asarray(f(t, rows[p]), dtype=float)

        return hazard

    @classmethod
    def _search(
        cls, pdata: Any, fitters: list, clock: str, models: list
    ) -> Any:
        """The joint fit from the starting ``models``."""
        from surpyval.multistate import _joint

        lik = cls._likelihood(pdata, fitters, clock)
        what = f"The fit of {cls.__name__}"
        units = None
        if cls._regression:
            units = _joint.coefficient_units(
                models, pdata.Z[pdata.intervals() - 1]
            )
        if clock == "forward":
            lik.plan(models)
            nodes = lik.nodes
            joint = _joint.fit(
                models,
                lik.neg_ll,
                what,
                covariance=False,
                central=True,
                units=units,
            )
            models = joint.models
            # (the grid again at the estimate, and the search from there
            # where it moved)
            lik.plan(models)
            if not np.array_equal(nodes, lik.nodes):
                models = _joint.fit(
                    models,
                    lik.neg_ll,
                    what,
                    covariance=False,
                    central=True,
                    units=units,
                ).models
        return _joint.fit(
            models, lik.neg_ll, what, polish=True, central=True, units=units
        )

    def _panel_likelihood(self) -> Any:
        """The likelihood the model was fitted by (on the forward clock,
        its grid planned at the fitted models)."""
        models = self._models()  # type: ignore[attr-defined]
        lik = self._likelihood(
            self._panel,
            [m.dist for m in models],
            self.clock,  # type: ignore[attr-defined]
        )
        if self.clock == "forward":  # type: ignore[attr-defined]
            lik.plan(models)
        return lik

    @property
    def n_obs(self) -> int:
        """The number of gaps between visits (the ``n`` of :meth:`bic`)."""
        return int(self._panel.intervals().size)

    def _ic_terms(self) -> tuple[int, float]:
        return self._joint.k, float(self.n_obs)

    def bic(self) -> float:
        """The Bayesian information criterion, ``2 neg_ll + K ln(n)``,
        ``n`` the number of gaps between visits (:attr:`n_obs`), as
        ``PanelMultiState`` counts them."""
        return super().bic()  # type: ignore[misc]

    def _panel_dict(self) -> dict:
        """The visits, as :meth:`to_dict` saves them."""
        from surpyval.multistate.panel import OBSTYPES

        p = self._panel
        return {
            "x": p.x.tolist(),
            "s": [to_native(v) for v in p.labels],
            "i": [to_native(p.ids[u]) for u in p.unit],
            "obstype": [OBSTYPES[k] for k in p.kind],
            "censored": {
                str(to_native(k)): [to_native(v) for v in states]
                for k, states in p.censored.items()
            },
            "censored_codes": [to_native(k) for k in p.censored],
            "exact_entry": [to_native(v) for v in p.exact_entry],
        }

    @staticmethod
    def _panel_from_dict(model_dict: dict, tm: Any, Z: Any) -> Any:
        from surpyval.multistate.panel import _check_panel
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        codes = model_dict.get("censored_codes", [])
        cens = {
            label_from_native(code): [
                label_from_native(v) for v in model_dict["censored"][str(code)]
            ]
            for code in codes
        }
        # (each row's obstype was saved as fitted: an exact entry stays one)
        return _check_panel(
            model_dict["x"],
            [label_from_native(v) for v in model_dict["s"]],
            [label_from_native(v) for v in model_dict["i"]],
            Z,
            model_dict["obstype"],
            tm,
            cens or None,
            [label_from_native(v) for v in model_dict["exact_entry"]] or None,
            None,
        )


class ParametricPanelMultiState(_PanelFit, ParametricMultiState):
    r"""
    A parametric multi-state model fitted to states seen only at visits
    (panel data): a parametric distribution for each transition's
    hazard, so that the intensities can rise or fall with time (a
    Weibull per transition, say), where
    :class:`~surpyval.multistate.PanelMultiState` holds them constant.

    The visits are the data of ``PanelMultiState`` (the time of each
    visit, the state seen, the unit, how it was seen); the fitted model
    is a :class:`~surpyval.multistate.ParametricMultiState`, with all of
    its predictions and bounds.

    With ``clock="forward"`` (the default) each hazard is a function of
    the time since the start (a non-homogeneous Markov model). The
    likelihood of a unit is the product over the gaps between its visits
    of the chance of going from the state seen at one visit to the state
    seen at the next, :math:`P(t_{j-1}, t_j)`, from the Kolmogorov
    forward equations by product integration (on a grid fixed for the
    search: every visit time, and the points between them that the
    accuracy needs at the starting values, chosen again at the estimate
    and the search repeated). With exponential distributions it is
    ``PanelMultiState``'s fit.

    With ``clock="reset"`` each hazard is a function of the time spent in
    the current state (a semi-Markov model). The time a state was entered
    is then not seen either, so the likelihood integrates over it (and
    over the entry into every state the unit may have passed through
    unseen between two visits). This is offered for progressive models
    only (no state can be come back to), where those paths are finite;
    each unit is taken to have entered its first state at its first
    visit.

    Every transition is fitted at once (BFGS on the log of positive
    parameters, from each distribution fitted to the exponential of a
    time-homogeneous fit), with the covariance from the whole
    likelihood's Hessian.

    Call the class method ``ParametricPanelMultiState.fit`` (or
    ``fit_from_df``); it returns a fitted instance.

    References
    ----------
    Titman, A. C. (2011). "Flexible nonhomogeneous Markov models for
    panel observed data", *Biometrics*, 67(3), 780-787.

    Kalbfleisch, J. D. and Lawless, J. F. (1985). "The analysis of panel
    data under a Markov assumption", *Journal of the American
    Statistical Association*, 80(392), 863-871.
    """

    _panel: Any

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: npt.ArrayLike,
        s: npt.ArrayLike,
        i: npt.ArrayLike,
        obstype: npt.ArrayLike | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        censored: dict | None = None,
        exact_entry: Any = None,
    ) -> ParametricPanelMultiState:
        """
        Fit a parametric distribution to each transition of states seen
        at visits.

        Parameters
        ----------
        x, s, i, obstype, transitions, censored, exact_entry
            The visits, as for :meth:`PanelMultiState.fit
            <surpyval.multistate.PanelMultiState.fit>`: the time of each
            (``x``), the state seen (``s``, or a code from ``censored``),
            the unit (``i``), how it was seen (``obstype``: ``'panel'``,
            ``'exact'`` or ``'exact_entry'``) and the allowed transitions
            (needed).
        dist : ParametricFitter or dict, optional
            The distribution of every transition (default ``Weibull``),
            or a ``{(from, to): distribution}`` dict with one for each.
        clock : ('forward', 'reset'), str, optional
            The time scale of the hazards: the time since the start
            (``'forward'``, the default; a Markov model) or the time in
            the current state (``'reset'``; a semi-Markov model, for
            progressive models only).

        Returns
        -------
        ParametricPanelMultiState
            The fitted model.

        Examples
        --------
        >>> from surpyval import Exponential, Weibull
        >>> from surpyval.multistate import ParametricPanelMultiState
        >>> visits = dict(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2, 4],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn", "ok", "ok",
        ...        "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model = ParametricPanelMultiState.fit(**visits, dist=Exponential)
        >>> model.params.round(3)
        array([0.658, 0.315])
        >>> model.occupancy(2.0, "worn").round(3)
        np.float64(0.507)

        With exponential distributions this is ``PanelMultiState``'s fit;
        the default, a Weibull per transition, lets each rate change with
        the time since the start (or, with ``clock="reset"``, with the
        time in the state).
        """
        from surpyval import Weibull
        from surpyval.multistate.panel import _check_panel

        _check_clock(clock)
        pdata = _check_panel(
            x, s, i, None, obstype, transitions, censored, exact_entry, None
        )
        fitters = _fitters(Weibull if dist is None else dist, pdata.tm, False)
        return cls._fit_panel(pdata, fitters, clock)

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        s_col: str,
        i_col: str,
        obstype_col: str | None = None,
        transitions: Any = None,
        dist: Any = None,
        clock: str = "forward",
        censored: dict | None = None,
        exact_entry: Any = None,
    ) -> ParametricPanelMultiState:
        """
        Fit the model to a DataFrame of visits, one row each; the columns
        are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import Exponential
        >>> from surpyval.multistate import ParametricPanelMultiState
        >>> df = pd.DataFrame({
        ...     "t": [0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     "state": ["ok", "worn", "failed", "ok", "worn", "failed",
        ...               "ok", "ok", "worn"],
        ...     "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ... })
        >>> model = ParametricPanelMultiState.fit_from_df(
        ...     df, "t", "state", "unit",
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     dist=Exponential,
        ... )
        >>> model.transitions.transition_names()
        ['ok -> worn', 'worn -> failed']
        """
        from surpyval.utils.dataframe import frame_column

        return cls.fit(
            frame_column(df, x_col, "x_col"),
            frame_column(df, s_col, "s_col"),
            frame_column(df, i_col, "i_col"),
            obstype=(
                None
                if obstype_col is None
                else frame_column(df, obstype_col, "obstype_col")
            ),
            transitions=transitions,
            dist=dist,
            clock=clock,
            censored=censored,
            exact_entry=exact_entry,
        )

    # -- fitting ------------------------------------------------------------

    @classmethod
    def _fit_panel(
        cls, pdata: Any, fitters: list, clock: str
    ) -> ParametricPanelMultiState:
        from surpyval.multistate import _panel_likelihood as pl

        spells = pl.midpoint_spells(pdata)
        cls._likelihood(pdata, fitters, clock)  # (its refusals first)
        joint = cls._search(
            pdata, fitters, clock, _exponential_starts(pdata, fitters)
        )
        out = cls()
        out.transitions = pdata.tm
        out.models = dict(zip(pdata.tm.transitions, joint.models))
        out.clock = clock
        out._data = spells
        out._Z = np.zeros((spells.x.size, 0))
        out._joint = joint
        out._panel = pdata
        out._setup()
        return out

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        visits and each transition's fitted distribution, with the joint
        fit's covariance. The reloaded model predicts, and bounds,
        exactly as the original.
        """
        return stamp_schema(
            {
                "model": "ParametricPanelMultiState",
                "transitions": self.transitions.to_dict(),
                **self._panel_dict(),
                "clock": self.clock,
                "models": [m.to_dict() for m in self._models()],
                "joint": self._joint.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> ParametricPanelMultiState:
        """Rebuild the model from :meth:`to_dict`."""
        from surpyval.multistate import _panel_likelihood as pl
        from surpyval.multistate._joint import JointFit
        from surpyval.multistate.transitions import TransitionMatrix
        from surpyval.univariate.parametric.parametric import Parametric

        require_model_tag(
            model_dict,
            "ParametricPanelMultiState",
            "a parametric multi-state model of panel data",
        )
        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        pdata = cls._panel_from_dict(model_dict, tm, None)
        out = cls()
        out.transitions = tm
        out.models = {
            pair: Parametric.from_dict(sub)
            for pair, sub in zip(tm.transitions, model_dict["models"])
        }
        out.clock = model_dict.get("clock", "forward")
        out._joint = JointFit.from_dict(model_dict["joint"], out._models())
        out._panel = pdata
        out._data = pl.midpoint_spells(pdata)
        out._Z = np.zeros((out._data.x.size, 0))
        out._setup()
        return out

    def __repr__(self) -> str:
        return "\n".join(
            self._repr_lines(
                "Parametric multi-state model of states seen at visits "
                "(a distribution per transition)"
            )
        )


class ParametricPanelMultiStateRegression(
    _PanelFit, ParametricMultiStateRegression
):
    r"""
    A parametric multi-state regression fitted to states seen only at
    visits (panel data): a parametric regression for each transition's
    hazard (a Weibull proportional hazards model per transition, say),
    the panel twin of
    :class:`~surpyval.multistate.ParametricMultiStateRegression`.

    The visits are the data of
    :class:`~surpyval.multistate.PanelMultiStateRegression`, with one
    row of covariates per visit: those of a row hold over the gap that
    starts at it, until the unit's next visit (as in R ``msm``). The
    fitted model is a ``ParametricMultiStateRegression``, with all of its
    predictions and bounds for given covariates.

    On the forward clock (the default; a non-homogeneous Markov model)
    a gap's factor is an element of :math:`P(t_{j-1}, t_j \mid z)`, the
    transition probabilities for the gap's covariates :math:`z`, by
    product integration on a grid of their own for each distinct row of
    covariates (refined at the starting values, again at the estimate).
    On the reset clock (a semi-Markov model, progressive models only) the
    unit is carried over its unknown entry times as in
    :class:`ParametricPanelMultiState`, each gap's hazards those of its
    covariates. With ``ExponentialPH`` regressions both are
    ``PanelMultiStateRegression``'s fit.

    Every transition is fitted at once (BFGS on central differences, each
    coefficient searched in its covariate's units, and a Newton step at
    the end), from regressions fitted to the exponentials of a
    ``PanelMultiStateRegression`` fit; the covariance is the inverse of
    the whole likelihood's numerical Hessian.

    Call the class method ``ParametricPanelMultiStateRegression.fit`` (or
    ``fit_from_df``); it returns a fitted instance.

    References
    ----------
    Titman, A. C. (2011). "Flexible nonhomogeneous Markov models for
    panel observed data", *Biometrics*, 67(3), 780-787.

    Marshall, G. and Jones, R. H. (1995), "Multi-state models and
    diabetic retinopathy", *Statistics in Medicine*, 14(18), 1975-1983.
    """

    @classmethod
    @takes_unit
    @refuses_surpyval_data
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        s: npt.ArrayLike,
        i: npt.ArrayLike,
        obstype: npt.ArrayLike | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        censored: dict | None = None,
        exact_entry: Any = None,
    ) -> ParametricPanelMultiStateRegression:
        """
        Fit a parametric regression to each transition of states seen at
        visits.

        Parameters
        ----------
        x, s, i, obstype, transitions, censored, exact_entry
            The visits, as for :meth:`PanelMultiState.fit
            <surpyval.multistate.PanelMultiState.fit>`: the time of each
            (``x``), the state seen (``s``, or a code from ``censored``),
            the unit (``i``), how it was seen (``obstype``) and the
            allowed transitions (needed).
        Z : array_like
            The covariates, one row per visit; a row's values hold until
            the unit's next visit. A table's column names name them.
            Missing values are refused.
        model : regression fitter or dict, optional
            The parametric regression of every transition (default
            ``WeibullPH``), such as ``ExponentialPH`` or ``WeibullAFT``,
            or a ``{(from, to): fitter}`` dict with one for each.
        clock : ('forward', 'reset'), str, optional
            The time scale of the hazards: the time since the start
            (``'forward'``, the default) or the time in the current state
            (``'reset'``, for progressive models only).

        Returns
        -------
        ParametricPanelMultiStateRegression
            The fitted model.

        Examples
        --------
        >>> from surpyval import ExponentialPH
        >>> from surpyval.multistate import (
        ...     ParametricPanelMultiStateRegression,
        ... )
        >>> model = ParametricPanelMultiStateRegression.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 1, 2,
        ...        4, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3],
        ...     Z=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        ...        0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "failed",
        ...        "ok", "worn", "failed", "ok", "worn", "worn", "ok", "ok",
        ...        "worn", "ok", "ok", "ok", "worn", "ok", "worn", "failed",
        ...        "ok", "ok", "worn", "ok", "worn", "failed", "ok", "ok",
        ...        "ok"],
        ...     i=[1] * 4 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3 + [6] * 4
        ...     + [7] * 3 + [8] * 3 + [9] * 3 + [10] * 3,
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     model=ExponentialPH,
        ... )
        >>> model.betas.round(2)
        array([[ 0.91],
               [-0.27]])
        >>> model.occupancy(2.0, [1], "worn").round(3)
        np.float64(0.546)
        """
        from surpyval import WeibullPH
        from surpyval.multistate.panel import _check_panel
        from surpyval.multistate.regression import _covariates

        _check_clock(clock)
        if Z is None:
            raise ValueError(
                "Z is needed; without covariates use "
                "ParametricPanelMultiState."
            )
        pdata = _check_panel(
            x, s, i, Z, obstype, transitions, censored, exact_entry, None
        )
        _, names = _covariates(Z, pdata.x.size)
        fitters = _fitters(
            WeibullPH if model is None else model, pdata.tm, True
        )
        return cls._fit_panel(pdata, fitters, clock, names)

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: Any,
        s_col: str,
        i_col: str,
        obstype_col: str | None = None,
        transitions: Any = None,
        model: Any = None,
        clock: str = "forward",
        censored: dict | None = None,
        exact_entry: Any = None,
    ) -> ParametricPanelMultiStateRegression:
        """
        Fit the model to a DataFrame of visits, one row each; ``Z_cols``
        names the covariate columns (a name or a list of them), and the
        other columns are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import ExponentialPH
        >>> from surpyval.multistate import (
        ...     ParametricPanelMultiStateRegression,
        ... )
        >>> df = pd.DataFrame({
        ...     "t": [0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2],
        ...     "state": ["ok", "worn", "failed", "ok", "worn", "failed",
        ...               "ok", "ok", "worn", "ok", "ok"],
        ...     "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        ...     "heavy": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        ... })
        >>> model = ParametricPanelMultiStateRegression.fit_from_df(
        ...     df, "t", "heavy", "state", "unit",
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     model=ExponentialPH,
        ... )
        >>> model.covariate_names
        ['heavy']
        """
        import pandas as pd

        from surpyval.utils.dataframe import frame_column

        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        Z = pd.DataFrame({c: frame_column(df, c, "Z_cols") for c in cols})
        return cls.fit(
            frame_column(df, x_col, "x_col"),
            Z,
            frame_column(df, s_col, "s_col"),
            frame_column(df, i_col, "i_col"),
            obstype=(
                None
                if obstype_col is None
                else frame_column(df, obstype_col, "obstype_col")
            ),
            transitions=transitions,
            model=model,
            clock=clock,
            censored=censored,
            exact_entry=exact_entry,
        )

    # -- fitting ------------------------------------------------------------

    @classmethod
    def _fit_panel(
        cls, pdata: Any, fitters: list, clock: str, names: list
    ) -> ParametricPanelMultiStateRegression:
        from surpyval.multistate import _panel_likelihood as pl

        spells = pl.midpoint_spells(pdata)
        cls._likelihood(pdata, fitters, clock)  # (its refusals first)
        starts = _regression_starts(pdata, fitters, names)
        joint = cls._search(pdata, fitters, clock, starts)
        out = cls()
        out.transitions = pdata.tm
        out.models = dict(zip(pdata.tm.transitions, joint.models))
        out.clock = clock
        out.covariate_names = list(names)
        out._data = spells
        out._Z = np.zeros((spells.x.size, len(names)))
        out._joint = joint
        out._panel = pdata
        out._setup()
        return out

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        visits with their covariates and each transition's fitted
        regression, with the joint fit's covariance. The reloaded model
        predicts, and bounds, exactly as the original.
        """
        return stamp_schema(
            {
                "model": "ParametricPanelMultiStateRegression",
                "transitions": self.transitions.to_dict(),
                **self._panel_dict(),
                "Z": self._panel.Z.tolist(),
                "covariate_names": list(self.covariate_names),
                "clock": self.clock,
                "models": [m.to_dict() for m in self._models()],
                "joint": self._joint.to_dict(),
            }
        )

    @classmethod
    def from_dict(
        cls, model_dict: dict
    ) -> ParametricPanelMultiStateRegression:
        """Rebuild the model from :meth:`to_dict`."""
        from surpyval.multistate import _panel_likelihood as pl
        from surpyval.multistate._joint import JointFit
        from surpyval.multistate.transitions import TransitionMatrix
        from surpyval.univariate.regression.parametric_regression_model import (  # noqa: E501
            ParametricRegressionModel,
        )

        require_model_tag(
            model_dict,
            "ParametricPanelMultiStateRegression",
            "a parametric multi-state regression of panel data",
        )
        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        names = list(model_dict["covariate_names"])
        Z = np.array(model_dict["Z"], dtype=float).reshape(
            len(model_dict["x"]), len(names)
        )
        pdata = cls._panel_from_dict(model_dict, tm, Z)
        out = cls()
        out.transitions = tm
        out.models = {
            pair: ParametricRegressionModel.from_dict(sub)
            for pair, sub in zip(tm.transitions, model_dict["models"])
        }
        out.clock = model_dict.get("clock", "forward")
        out.covariate_names = names
        out._joint = JointFit.from_dict(model_dict["joint"], out._models())
        out._panel = pdata
        out._data = pl.midpoint_spells(pdata)
        out._Z = np.zeros((out._data.x.size, len(names)))
        out._setup()
        return out

    def __repr__(self) -> str:
        lines = self._repr_lines(
            "Parametric multi-state regression of states seen at visits "
            "(a parametric regression per transition)"
        )
        lines.insert(3, f"Covariates: {', '.join(self.covariate_names)}")
        return "\n".join(lines)
