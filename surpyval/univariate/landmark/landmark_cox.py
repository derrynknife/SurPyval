"""Landmark Cox models: a Cox model of the landmark data at one landmark,
or a supermodel over many, and the dynamic predictions they give."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.utils.validation import check_alpha_ci, check_option

from ._data import Lives, check_landmarks, stack

_BASELINES = ("stratified", "pooled")
_TIES = ("efron", "breslow")


def _poly_names(degree: int) -> list[str]:
    return ["u" if k == 1 else f"u^{k}" for k in range(1, degree + 1)]


def _survival_bounds(
    H: npt.NDArray, se: npt.NDArray, z: float, bound_type: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """Bounds ``(H_lower, H_upper)`` on the cumulative hazard ``H`` with
    standard error ``se``: normal on ``log H`` (``"log-log"``), on ``log
    S`` (``"log"``, the upper survival bound at most 1) or on ``S``
    (``"plain"``, inside [0, 1]), as the Cox model's ``cb``; both are
    ``H`` where ``se`` is 0."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if bound_type == "log":
            lo = np.maximum(H - z * se, 0.0)
            hi = H + z * se
        elif bound_type == "log-log":
            ratio = np.where(H > 0, se / H, 0.0)
            lo = H * np.exp(-z * ratio)
            hi = H * np.exp(z * ratio)
        else:
            S = np.exp(-H)
            lo = -np.log(np.minimum(S + z * S * se, 1.0))
            hi = -np.log(np.maximum(S - z * S * se, 0.0))
        flat = se == 0
        return np.where(flat, H, lo), np.where(flat, H, hi)


def _refuse_unknown(unknown: dict, qualname: str, weights: str = "w") -> None:
    """Refuse weights (with the package's refusal) and unknown keywords."""
    from surpyval.utils.weights import refuse_unsupported_weights

    if weights in unknown:
        raise refuse_unsupported_weights(LandmarkCox, qualname, weights)
    if unknown:
        raise TypeError(
            "{}() got unexpected keyword argument(s) {}.".format(
                qualname, sorted(unknown)
            )
        )


class LandmarkCox(SerialisableMixin):
    r"""
    Landmark Cox models and the dynamic predictions they give
    (van Houwelingen, 2007; van Houwelingen and Putter, 2011).

    A dynamic prediction is the chance that a unit still event-free at
    time ``s`` has its event within the next ``w``, given what is known
    about it at ``s``: :math:`P(T \le s + w \mid T > s, Z(s))`. Landmarking
    answers it directly. At a landmark time ``s``, take the units still at
    risk, their covariates as known at ``s``, and their follow-up cut at
    ``s + w`` (:func:`~surpyval.landmark`), and fit a Cox model to
    them. Because only units still at risk at ``s`` are used, and only
    what was known about them at ``s``, the prediction never uses the
    future, so it has no immortal-time bias.

    With one landmark the model is that Cox model. With several, the
    landmark data sets are stacked and fitted together as a *supermodel*:
    the effect of each covariate may change smoothly with the landmark,

    .. math::

        h(t \mid Z(s), s) = h_0(t \mid s)
        \exp\big(\beta(s)' Z(s)\big), \qquad s \le t \le s + w,
        \qquad \beta(s) = \beta_0 + \beta_1 u + \dots + \beta_d u^d,

    with :math:`u = (s - s_1) / (s_L - s_1)` (0 at the first landmark, 1
    at the last), or any basis of ``s`` given as ``basis``. The baseline
    hazard is either a separate one at each landmark
    (``baseline="stratified"``) or one shared baseline
    :math:`h_0(t) \exp(\theta(s))` with :math:`\theta(s) = \theta_1 u +
    \dots + \theta_d u^d` (``baseline="pooled"``), which predicts at any
    ``s`` between the first and last landmark. A unit appears once for
    each landmark it is at risk at, so its rows are correlated; the
    standard errors are therefore the robust (sandwich) ones, clustered on
    the unit (R's ``coxph(... + cluster(id))``), and so are the bounds on
    the predictions.

    Fit with :meth:`fit` or :meth:`fit_from_df`.

    Attributes
    ----------
    params : numpy.ndarray
        The coefficients, named by ``parameter_names``: each covariate's
        effect at the first landmark (``"z"``), its change along the
        basis (``"z*u"``, ``"z*u^2"``, ...) and, for a pooled baseline,
        the baseline's terms (``"u"``, ``"u^2"``, ...).
    landmarks : numpy.ndarray
        The landmark times fitted.
    window : float
        The prediction window ``w``.
    baseline : str
        ``"stratified"`` or ``"pooled"``.
    feature_names : list of str
        The covariates' names.
    cox : SemiParametricRegressionModel
        The Cox model of the stacked data.
    """

    def __init__(self) -> None:
        self.params: npt.NDArray = np.empty(0)
        self.parameter_names: list[str] = []
        self.landmarks: npt.NDArray = np.empty(0)
        self.window: float = np.nan
        self.baseline: str = "stratified"
        self.degree: int = 0
        self.varying: tuple[int, ...] = ()
        self.feature_names: list[str] = []
        self.tie_method: str = "efron"
        self.n_units: int = 0
        self.n_rows: int = 0
        self.cox: Any = None
        self._basis: Callable | None = None
        self._basis_names: list[str] = []
        self._kept: npt.NDArray = np.empty(0, dtype=int)
        self._lives: Any = None
        self._life: Any = None
        self._stack: dict = {}
        self._X: npt.NDArray = np.empty((0, 0))
        self._robust: npt.NDArray = np.empty((0, 0))

    # -- fitting --------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        life: Any,
        at: npt.ArrayLike | float,
        horizon: float,
        baseline: str = "stratified",
        degree: int | None = None,
        basis: Callable[[npt.NDArray], npt.ArrayLike] | None = None,
        varying: Any = None,
        tie_method: str = "efron",
        **unknown: Any,
    ) -> LandmarkCox:
        """
        Fit a landmark Cox model (one landmark) or supermodel (several) to
        the landmark data of ``life`` (see :func:`~surpyval.landmark`).

        Parameters
        ----------
        life : SurpyvalData
            One row per unit: time ``x``, ``c`` (0 the event, 1 still
            running), counts ``n``, entry ``tl``, unit ids, fixed
            covariates ``Z``, time-varying ones and measurements ``m``;
            the covariates at each landmark are those known then (see
            :func:`~surpyval.landmark`).
        at : float or array_like
            The landmark times ``L``.
        horizon : float
            The prediction window ``w``; follow-up is cut at ``L + w``.
        baseline : {'stratified', 'pooled'}, optional
            A baseline hazard of its own at each landmark (the default),
            or one shared baseline moved by the basis terms in ``L``.
        degree : int, optional
            The degree of the polynomial in ``u`` for the covariate
            effects (and a pooled baseline's terms). Default 2, or one
            less than the number of landmarks if that is smaller (0 for a
            single landmark: a Cox model at that landmark).
        basis : callable, optional
            Instead of the polynomial, a function of the landmark times
            (a 1-D array) returning one column per basis term (a spline
            basis, for example); the constant is always added. A model
            with a basis of its own cannot be saved by :meth:`to_dict`.
        varying : list, optional
            The covariates (names, or positions in the landmark data's
            ``Z``) whose effect changes with the landmark. Default: every
            covariate. The others have one effect at every landmark.
        tie_method : {'efron', 'breslow'}, optional
            The Cox model's handling of tied times.

        Returns
        -------
        LandmarkCox

        Examples
        --------
        A fixed covariate ``z`` whose effect fades: the chance of the
        event within 2 time units of each landmark:

        >>> import numpy as np
        >>> from surpyval import LandmarkCox, SurpyvalData
        >>> rng = np.random.default_rng(1)
        >>> z = rng.binomial(1, 0.5, 400).astype(float)
        >>> t = rng.weibull(1.5, 400) * 4 * np.exp(-0.5 * z)
        >>> life = SurpyvalData(x=np.minimum(t, 6), c=(t > 6).astype(int),
        ...                     Z=z[:, None], covariate_names=["z"])
        >>> model = LandmarkCox.fit(life, at=[0, 1, 2], horizon=2,
        ...                         degree=1)
        >>> model.predict([0, 2], [[0], [1]]).round(3)
        array([[0.274, 0.517],
               [0.492, 0.675]])
        """
        _refuse_unknown(unknown, "LandmarkCox.fit")
        lives = Lives(life)
        return cls._fit(
            lives,
            life,
            at,
            horizon,
            baseline,
            degree,
            basis,
            varying,
            tie_method,
        )

    @classmethod
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        at: npt.ArrayLike | float,
        horizon: float,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        i_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        baseline: str = "stratified",
        degree: int | None = None,
        basis: Callable[[npt.NDArray], npt.ArrayLike] | None = None,
        varying: Any = None,
        tie_method: str = "efron",
        **unknown: Any,
    ) -> LandmarkCox:
        """
        Fit from a DataFrame with one row per unit: the columns of its
        time (``x_col``), censoring flag, count, entry, id and fixed
        covariates, read by :meth:`SurpyvalData.from_dataframe
        <surpyval.SurpyvalData.from_dataframe>`; the rest is as
        :meth:`fit`. Time-varying covariates and measurements need a
        ``SurpyvalData`` (``from_items``) and :meth:`fit`.
        """
        from surpyval.utils.surpyval_data import SurpyvalData

        _refuse_unknown(unknown, "LandmarkCox.fit_from_df", "w_col")
        life = SurpyvalData.from_dataframe(
            df,
            x_col=x_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            i_col=i_col,
            Z_cols=Z_cols,
        )
        return cls.fit(
            life,
            at,
            horizon,
            baseline=baseline,
            degree=degree,
            basis=basis,
            varying=varying,
            tie_method=tie_method,
        )

    @classmethod
    def _fit(
        cls,
        lives: Lives,
        life: Any,
        at: Any,
        horizon: Any,
        baseline: str,
        degree: int | None,
        basis: Callable | None,
        varying: Any,
        tie_method: str,
    ) -> LandmarkCox:
        from surpyval import CoxPH

        check_option("baseline", baseline, _BASELINES)
        check_option("tie_method", tie_method, _TIES)
        s, w = check_landmarks(at, horizon)
        p = lives.p
        if p == 0:
            raise ValueError(
                "A landmark Cox model needs covariates, and the data has "
                "none: for the survival from a landmark without them, use "
                "landmark_kaplan_meier."
            )
        if varying is None:
            vary = tuple(range(p))
        else:
            given = [varying] if isinstance(varying, str) else list(varying)
            missing = [
                v for v in given if isinstance(v, str) and v not in lives.names
            ]
            if missing:
                raise ValueError(
                    "'varying' names {} not among the covariates {}.".format(
                        missing, lives.names
                    )
                )
            vary = tuple(
                sorted(
                    {
                        lives.names.index(v) if isinstance(v, str) else int(v)
                        for v in given
                    }
                )
            )
            if any(v < 0 or v >= p for v in vary):
                raise ValueError(
                    "'varying' must list column positions of Z (0 to "
                    "{}); got {}.".format(p - 1, list(vary))
                )
        model = cls()
        model.landmarks = s
        model.window = w
        model.baseline = baseline
        model.varying = vary
        model.feature_names = list(lives.names)
        model.tie_method = tie_method
        model._basis = basis
        if basis is None:
            if degree is None:
                degree = min(2, s.size - 1)
            if int(degree) != degree or degree < 0:
                raise ValueError(
                    "'degree' must be a whole number, 0 or more; got "
                    "{!r}.".format(degree)
                )
            degree = int(degree)
            if degree > s.size - 1:
                raise ValueError(
                    "With {} landmark(s) the polynomial in s can have "
                    "degree at most {}; got degree={}. Add landmarks or "
                    "lower the degree.".format(s.size, s.size - 1, degree)
                )
            model.degree = degree
            model._basis_names = _poly_names(degree)
        else:
            if degree is not None:
                raise ValueError(
                    "Give 'degree' (a polynomial in s) or 'basis' (your "
                    "own), not both."
                )
            G = np.asarray(basis(s), dtype=float)
            G = G.reshape(s.size, -1) if G.size else np.empty((s.size, 0))
            if not np.isfinite(G).all():
                raise ValueError(
                    "'basis' returned non-finite values at the landmarks."
                )
            model.degree = G.shape[1]
            model._basis_names = [f"g{k}" for k in range(1, G.shape[1] + 1)]
        st = stack(lives, s, w)
        empty = [
            float(v) for k, v in enumerate(s) if not (st["lm"] == k).any()
        ]
        if empty:
            raise ValueError(
                "No unit is still at risk at the landmark(s) {}: drop "
                "them from 'at'.".format(empty)
            )
        if not (st["c"] == 0).any():
            raise ValueError(
                "No event happens within the window of any landmark, so "
                "there is nothing to fit: lengthen 'horizon' or choose "
                "earlier landmarks."
            )
        X = model._design(st["Z"], st["L"])
        model.parameter_names = model._design_names()
        strata = st["lm"] if baseline == "stratified" else None
        fit_args = dict(
            x=st["x"],
            c=st["c"],
            n=st["n"],
            tl=st["L"],
            tie_method=tie_method,
            strata=strata,
        )
        with warnings.catch_warnings():
            # Aliased columns are said in the covariates' terms below
            warnings.filterwarnings("ignore", message=".*cannot be estimated")
            cox = CoxPH.fit(Z=X, **fit_args)
        kept = np.flatnonzero(~np.isnan(np.asarray(cox.params, float)))
        if kept.size < X.shape[1]:
            # The robust variances need the residuals of the fit without
            # the aliased columns.
            model._warn_aliased(np.setdiff1d(np.arange(X.shape[1]), kept))
            if kept.size == 0:
                raise ValueError(
                    "None of the coefficients can be estimated: the "
                    "covariates do not vary within the risk sets."
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cox = CoxPH.fit(Z=X[:, kept], **fit_args)
        model.cox = cox
        model._kept = kept
        params = np.full(X.shape[1], np.nan)
        params[kept] = np.asarray(cox.params, dtype=float)
        model.params = params
        model._lives = lives
        model._life = life
        model._stack = st
        model._X = X
        model.n_units = int(np.unique(st["unit"]).size)
        model.n_rows = int(st["unit"].size)
        model._robust = model._robust_covariance()
        return model

    def _warn_aliased(self, columns: npt.NDArray) -> None:
        """Say which coefficients the data cannot determine: whole
        covariates (columns of ``Z``) and single terms."""
        from surpyval.utils.warnings import caller_stacklevel

        p = len(self.feature_names)
        whole = [
            j
            for j in range(p)
            if np.isin([j] + self._columns_of(j), columns).all()
        ]
        covered = set(whole) | {c for j in whole for c in self._columns_of(j)}
        terms = [self.parameter_names[c] for c in columns if c not in covered]
        parts = []
        if whole:
            parts.append(
                "Covariate column(s) {} of Z cannot be estimated".format(
                    ", ".join(str(j) for j in whole)
                )
            )
        if terms:
            parts.append(
                "{} the term(s) {} cannot be estimated".format(
                    "and" if whole else "In the landmark model,", terms
                )
            )
        warnings.warn(
            " ".join(parts) + ": the partial likelihood of the landmark "
            "data does not depend on them (a covariate that repeats "
            "others, is constant, or does not vary at the landmarks). "
            "Their coefficients are aliased -- reported as nan -- and the "
            "model is fitted without them; predictions take them as 0. "
            "Remove the covariate, or lower 'degree' or set 'varying', to "
            "fit without this warning.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )

    # -- the design -----------------------------------------------------------

    def _terms(self, s: npt.NDArray) -> npt.NDArray:
        """The basis terms at landmark times ``s``, one column each."""
        s = np.asarray(s, dtype=float)
        if self._basis is not None:
            G = np.asarray(self._basis(s), dtype=float)
            return G.reshape(s.size, -1) if G.size else np.empty((s.size, 0))
        span = self.landmarks[-1] - self.landmarks[0]
        u = (s - self.landmarks[0]) / span if span > 0 else np.zeros(s.size)
        G = np.empty((s.size, self.degree))
        for k in range(self.degree):
            G[:, k] = u ** (k + 1)
        return G

    def _design(self, Z: npt.NDArray, s: npt.NDArray) -> npt.NDArray:
        """The Cox model's columns for covariates ``Z`` (one row each)
        known at landmarks ``s``."""
        G = self._terms(s)
        cols = [Z]
        for j in self.varying:
            cols.append(Z[:, [j]] * G)
        if self.baseline == "pooled":
            cols.append(G)
        return np.column_stack(cols).astype(float)

    def _design_names(self) -> list[str]:
        names = list(self.feature_names)
        for j in self.varying:
            names += [
                f"{self.feature_names[j]}*{g}" for g in self._basis_names
            ]
        if self.baseline == "pooled":
            names += list(self._basis_names)
        return names

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` none of whose coefficients the data can
        determine (``nan`` in ``params``)."""
        nan = np.isnan(self.params)
        p = len(self.feature_names)
        out = []
        for j in range(p):
            cols = [j] + self._columns_of(j)
            if nan[cols].all():
                out.append(j)
        return np.array(out, dtype=int)

    def _columns_of(self, j: int) -> list[int]:
        """The design columns of covariate ``j``'s change with ``s``."""
        if j not in self.varying:
            return []
        k = self.varying.index(j)
        start = len(self.feature_names) + k * self.degree
        return list(range(start, start + self.degree))

    # -- the coefficients -----------------------------------------------------

    def _influence_beta(self) -> npt.NDArray:
        """Each stacked row's dfbeta (its influence on the estimated
        coefficients), in the stacked order, over the estimated
        columns."""
        cache = self.__dict__.get("_dfbeta")
        if cache is None:
            cache = np.asarray(self.cox.compute_residuals("dfbeta"), float)
            cache = cache.reshape(self.n_rows, -1)
            self._dfbeta = cache
        return cache

    def _cluster(self, values: npt.NDArray) -> npt.NDArray:
        """Rows' influences summed by unit and scaled for the counts: the
        robust variance is ``sum over units of n_u (U_u / n_u)^2``, a unit
        of count ``n_u`` being that many independent units."""
        st = self._stack
        codes = st["unit"]
        size = len(self._lives.ids)
        out = np.zeros((size,) + values.shape[1:])
        np.add.at(out, codes, values)
        weight = self._lives.n
        return out / np.sqrt(weight).reshape((-1,) + (1,) * (out.ndim - 1))

    def _robust_covariance(self) -> npt.NDArray:
        D = self._cluster(self._influence_beta())
        k = self._kept
        p = self.params.size
        out = np.full((p, p), np.nan)
        out[np.ix_(k, k)] = D.T @ D
        return out

    def covariance(self, robust: bool = True) -> npt.NDArray:
        """
        The coefficients' covariance: the robust one, clustered on the
        unit (the default), or, with ``robust=False``, the model-based
        one, which treats a unit's rows at different landmarks as
        independent and is too small for a supermodel.
        """
        if robust:
            return self._robust.copy()
        p = self.params.size
        out = np.full((p, p), np.nan)
        k = self._kept
        out[np.ix_(k, k)] = np.asarray(self.cox.covariance(), dtype=float)
        return out

    def standard_errors(self, robust: bool = True) -> npt.NDArray:
        """The coefficients' standard errors (robust by default; see
        :meth:`covariance`)."""
        return np.sqrt(np.diag(self.covariance(robust)))

    def summary(self, alpha_ci: float = 0.05, robust: bool = True) -> Any:
        """
        The coefficients' table: estimate, standard error (robust,
        clustered on the unit, by default), z, p-value, the Wald bounds
        and the hazard ratio ``exp(coef)``.
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        b = self.params
        se = self.standard_errors(robust)
        z = norm.isf(alpha_ci / 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = b / se
        table = pd.DataFrame(
            {
                "coef": b,
                "exp(coef)": np.exp(b),
                "se": se,
                "z": stat,
                "p": 2 * norm.sf(np.abs(stat)),
                "lower": b - z * se,
                "upper": b + z * se,
            },
            index=pd.Index(self.parameter_names, name="term"),
        )
        return table

    def _check_effect_s(self, s: npt.NDArray) -> None:
        lo, hi = self.landmarks[0], self.landmarks[-1]
        bad = (s < lo) | (s > hi)
        if bad.any():
            raise ValueError(
                "The landmark effects are fitted from s = {:g} to {:g}; "
                "got s = {}. Refit with landmarks covering it.".format(
                    lo, hi, s[bad].tolist()
                )
            )

    def beta(self, s: npt.ArrayLike) -> npt.NDArray:
        """
        Each covariate's effect (log hazard ratio) at landmark(s) ``s``:
        ``beta(s)``, shape ``np.shape(s) + (p,)``.
        """
        s_arr = np.asarray(s, dtype=float)
        flat = np.atleast_1d(s_arr).ravel()
        self._check_effect_s(flat)
        L = self._effect_matrix(flat)
        b = np.where(np.isnan(self.params), 0.0, self.params)
        out = np.einsum("mpk,k->mp", L, b)
        out[:, self.aliased] = np.nan
        return out.reshape(s_arr.shape + (len(self.feature_names),))

    def beta_cb(
        self,
        s: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Robust Wald bounds on :meth:`beta` at ``s``: shape ``np.shape(s)
        + (p, 2)`` two-sided, ``np.shape(s) + (p,)`` one-sided
        (``bound='lower'`` or ``'upper'``).
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, ("two-sided", "lower", "upper"))
        s_arr = np.asarray(s, dtype=float)
        flat = np.atleast_1d(s_arr).ravel()
        b = self.beta(flat)
        L = self._effect_matrix(flat)
        V = np.where(np.isnan(self._robust), 0.0, self._robust)
        se = np.sqrt(np.einsum("mpk,kl,mpl->mp", L, V, L))
        se[:, self.aliased] = np.nan
        shape = s_arr.shape + (b.shape[1],)
        if bound == "two-sided":
            z = norm.isf(alpha_ci / 2)
            out = np.stack([b - z * se, b + z * se], axis=-1)
            return out.reshape(shape + (2,))
        z = norm.isf(alpha_ci)
        return (b - z * se if bound == "lower" else b + z * se).reshape(shape)

    def _effect_matrix(self, s: npt.NDArray) -> npt.NDArray:
        """``L[m, j, k]``: the weight of coefficient ``k`` in covariate
        ``j``'s effect at ``s[m]``."""
        p = len(self.feature_names)
        G = self._terms(s)
        L = np.zeros((s.size, p, self.params.size))
        for j in range(p):
            L[:, j, j] = 1.0
            cols = self._columns_of(j)
            if cols:
                L[:, j, cols] = G
        return L

    # -- predictions ----------------------------------------------------------

    def _check_s(self, s: Any) -> tuple[float, Any]:
        """The landmark ``s`` of a prediction and its stratum."""
        try:
            s = float(s)
        except (TypeError, ValueError):
            raise ValueError(
                "'s' (the landmark of the prediction) must be one time; "
                "got {!r}.".format(s)
            ) from None
        if self.baseline == "stratified":
            hit = np.flatnonzero(self.landmarks == s)
            if hit.size == 0:
                raise ValueError(
                    "A stratified supermodel has a baseline only at its "
                    "landmarks {}; got s = {:g}. Predict at one of them, "
                    "or fit with baseline='pooled', which predicts at any "
                    "s from the first landmark to the last.".format(
                        self.landmarks.tolist(), s
                    )
                )
            return s, int(hit[0])
        lo, hi = self.landmarks[0], self.landmarks[-1]
        if not lo <= s <= hi:
            raise ValueError(
                "The supermodel is fitted from s = {:g} to {:g}; got s = "
                "{:g}. Predict within that range, or refit with landmarks "
                "covering it.".format(lo, hi, s)
            )
        return s, None

    def _n_rows(self, Z: Any) -> int:
        """The number of covariate rows in a query's ``Z``."""
        import pandas as pd

        if isinstance(Z, pd.DataFrame):
            return len(Z)
        arr = np.asarray(Z, dtype=float)
        return 1 if arr.ndim <= 1 else arr.shape[0]

    def _rows(self, Z: Any, size: int) -> npt.NDArray:
        """The covariate rows of a query: one (for every time) or one per
        time."""
        import pandas as pd

        if isinstance(Z, pd.DataFrame):
            Z = Z[self.feature_names].to_numpy(dtype=float)
        Z = np.asarray(Z, dtype=float)
        p = len(self.feature_names)
        if Z.ndim <= 1:
            if Z.size != p:
                raise ValueError(
                    "Z must have {} covariate value(s) per row; got "
                    "{}.".format(p, Z.size)
                )
            return np.broadcast_to(Z.reshape(1, p), (size, p))
        if Z.shape[1] != p:
            raise ValueError(f"Z must have {p} column(s); got {Z.shape[1]}.")
        if Z.shape[0] == 1:
            return np.broadcast_to(Z, (size, p))
        if Z.shape[0] != size:
            raise ValueError(
                "Z has {} rows for {} times: give one row for every time, "
                "or one row per time.".format(Z.shape[0], size)
            )
        return Z

    def _Hf(
        self, x: Any, Z: Any, s: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, tuple, Any, float]:
        """The cumulative hazard from ``s`` to each ``x``, flattened, with
        the design rows, the clipped times, the query's shape, the
        stratum and ``s``."""
        s, stratum = self._check_s(s)
        x_arr = np.asarray(x, dtype=float)
        shape = x_arr.shape
        t = x_arr.ravel()
        if t.size == 0:
            return np.empty(0), np.empty((0, 1)), t, shape, stratum, s
        k = self._n_rows(Z)
        if t.size == 1 and k > 1:  # one time, several rows
            shape = (k,)
            t = np.repeat(t, k)
        Zr = self._rows(Z, t.size)
        X = self._design(Zr, np.full(Zr.shape[0], s))[:, self._kept]
        horizon = s + self.window
        past = t > horizon * (1 + 1e-12)
        if past.any():
            warnings.warn(
                "{} of the times asked for are past the horizon s + w = "
                "{:g} the model was fitted to; their predictions are "
                "nan. Fit with a longer 'window' to predict further "
                "ahead.".format(int(past.sum()), horizon),
                UserWarning,
                stacklevel=3,
            )
        tc = np.clip(np.where(np.isnan(t), s, t), s, horizon)
        if stratum is None:
            H = self.cox.Hf(tc, X) - self.cox.Hf(np.full(tc.size, s), X)
        else:
            H = self.cox.Hf(tc, X, stratum) - self.cox.Hf(
                np.full(tc.size, s), X, stratum
            )
        H = np.asarray(H, dtype=float).copy()
        H[past | np.isnan(t) | np.isnan(Zr).any(axis=1)] = np.nan
        return H, X, tc, shape, stratum, s

    def Hf(self, x: npt.ArrayLike, Z: Any, s: float) -> npt.NDArray:
        """
        The cumulative hazard from the landmark ``s`` to ``x``, for a unit
        event-free at ``s`` with covariates ``Z`` known at ``s``. 0 for
        ``x <= s``; ``nan`` (with a warning) past the horizon ``s + w``.
        ``Z`` is one row for every ``x`` or one row per ``x``.
        """
        H, _, _, shape, _, _ = self._Hf(x, Z, s)
        return H.reshape(shape)

    def sf(self, x: npt.ArrayLike, Z: Any, s: float) -> npt.NDArray:
        """
        The dynamic survival :math:`P(T > x \\mid T > s, Z(s))`, for ``x``
        from ``s`` to the horizon ``s + w`` (1 before ``s``; ``nan`` past
        the horizon, with a warning).
        """
        return np.exp(-self.Hf(x, Z, s))

    def ff(self, x: npt.ArrayLike, Z: Any, s: float) -> npt.NDArray:
        """
        The dynamic failure probability :math:`P(T \\le x \\mid T > s,
        Z(s))` (see :meth:`sf`).
        """
        return -np.expm1(-self.Hf(x, Z, s))

    def predict(
        self,
        s: npt.ArrayLike,
        Z: Any,
        window: float | None = None,
    ) -> npt.NDArray:
        """
        The dynamic prediction: the chance of the event within ``window``
        (default the fitted ``w``, at most it) of the landmark ``s``, for
        a unit event-free at ``s`` with covariates ``Z`` known at ``s``,
        :math:`P(T \\le s + w \\mid T > s, Z(s))`.

        ``s`` is one landmark or several; ``Z`` one covariate profile (a
        1-D row) or several (rows). The result has shape ``np.shape(s)``
        for one profile, ``np.shape(s) + (k,)`` for ``k`` profiles.
        """
        w = self._check_window(window)
        s_arr = np.asarray(s, dtype=float)
        rows, single = self._profiles(Z)
        out = np.empty((s_arr.size, rows.shape[0]))
        for m, sv in enumerate(s_arr.ravel()):
            out[m] = self.ff(np.full(rows.shape[0], sv + w), rows, sv)
        shape = s_arr.shape if single else s_arr.shape + (rows.shape[0],)
        return out.reshape(shape)

    def _check_window(self, window: Any) -> float:
        if window is None:
            return self.window
        w = float(window)
        if not (0 < w <= self.window):
            raise ValueError(
                "'window' must be positive and at most the fitted window "
                "{:g}; got {!r}. Refit with a longer window to predict "
                "further ahead.".format(self.window, window)
            )
        return w

    def _profiles(self, Z: Any) -> tuple[npt.NDArray, bool]:
        import pandas as pd

        if isinstance(Z, pd.DataFrame):
            return Z[self.feature_names].to_numpy(dtype=float), False
        Z = np.asarray(Z, dtype=float)
        p = len(self.feature_names)
        if Z.ndim <= 1:
            if Z.size != p:
                raise ValueError(
                    "Z must have {} covariate value(s); got {}.".format(
                        p, Z.size
                    )
                )
            return Z.reshape(1, p), True
        return self._rows(Z, Z.shape[0]), False

    # -- bounds ---------------------------------------------------------------

    def _risk_parts(self, stratum: Any) -> dict:
        """At each event time of ``stratum`` (all rows for a pooled
        baseline): the Breslow increments of the baseline (at the mean of
        the columns), the cumulative sums the influence functions need,
        and each row's risk."""
        cache = self.__dict__.setdefault("_parts", {})
        if stratum in cache:
            return cache[stratum]
        st = self._stack
        X = self._X[:, self._kept]
        beta = np.asarray(self.cox.params, dtype=float)
        n = st["n"].astype(float)
        mean = (n[:, None] * X).sum(axis=0) / n.sum()
        lp = (X - mean) @ beta
        shift = lp.max()
        risk = n * np.exp(lp - shift)
        take = (
            np.ones(n.size, bool) if stratum is None else st["lm"] == stratum
        )
        x, tl, c = st["x"], st["L"], st["c"]
        ev = take & (c == 0)
        times = np.unique(x[ev])
        d = np.zeros(times.size)
        np.add.at(d, np.searchsorted(times, x[ev]), n[ev])
        # Rows at risk at t: tl < t <= x, from sorted x and tl.
        xs_order = np.argsort(x[take])
        tl_order = np.argsort(tl[take])
        r_t = risk[take]
        Xc = (X - mean)[take]

        def at_risk_sum(v: npt.NDArray) -> npt.NDArray:
            vx = v[xs_order]
            vt = v[tl_order]
            cx = np.concatenate(
                [np.cumsum(vx[::-1], axis=0)[::-1], 0 * vx[:1]]
            )
            ct = np.concatenate(
                [np.cumsum(vt[::-1], axis=0)[::-1], 0 * vt[:1]]
            )
            ix = np.searchsorted(x[take][xs_order], times, side="left")
            it = np.searchsorted(tl[take][tl_order], times, side="left")
            return cx[ix] - ct[it]

        S0 = at_risk_sum(r_t)
        S1 = at_risk_sum(r_t[:, None] * Xc)
        dL = d / S0
        parts = {
            "times": times,
            "S0": S0,
            "L": np.cumsum(dL),
            "G": np.cumsum(dL / S0),
            "Q": np.cumsum(dL[:, None] * S1 / S0[:, None], axis=0),
            "take": take,
            "risk": risk,
            "mean": mean,
        }
        cache[stratum] = parts
        return parts

    def _log_H_se(
        self, X: npt.NDArray, tc: npt.NDArray, stratum: Any, s: float
    ) -> npt.NDArray:
        """The robust standard error of the log cumulative hazard from
        ``s`` to each ``tc`` for the design rows ``X``: the infinitesimal
        jackknife, the rows' influences summed over each unit."""
        P = self._risk_parts(stratum)
        st = self._stack
        times = P["times"]

        def cum(arr: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
            idx = np.searchsorted(times, t, side="right") - 1
            pad = np.concatenate([np.zeros((1,) + arr.shape[1:]), arr])
            return pad[idx + 1]

        A = cum(P["L"], tc) - cum(P["L"], np.array([s]))
        q = cum(P["Q"], tc) - cum(P["Q"], np.array([s]))
        take = P["take"]
        x, tl, c, n = st["x"], st["L"], st["c"], st["n"].astype(float)
        D = self._influence_beta()
        # Baseline part, per row and query time (rows outside the
        # stratum have none).
        rows = np.flatnonzero(take)
        xr, tlr = x[rows], tl[rows]
        ev = c[rows] == 0
        S0_at = np.ones(rows.size)
        j = np.searchsorted(times, xr)
        ok = ev & (j < times.size)
        S0_at[ok] = P["S0"][j[ok]]
        hi = np.minimum(xr[:, None], tc[None, :])
        lo = np.maximum(tlr, s)[:, None]
        G_hi = cum(P["G"], hi.ravel()).reshape(hi.shape)
        G_lo = cum(P["G"], np.broadcast_to(lo, hi.shape).ravel()).reshape(
            hi.shape
        )
        own = np.where(
            ev[:, None] & (xr[:, None] > s) & (xr[:, None] <= tc[None, :]),
            (n[rows] / S0_at)[:, None],
            0.0,
        ) - P["risk"][rows][:, None] * np.where(hi > lo, G_hi - G_lo, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_A = np.where(A > 0, 1.0 / A, 0.0)
            # (rows, queries): baseline part / A, then the coefficients'
            # part, (X - mean - q / A)' D_r.
            lead = (X - P["mean"]) - q * inv_A[:, None]
        infl = D @ lead.T
        infl[rows] += own * inv_A[None, :]
        U = self._cluster(infl)
        se = np.sqrt(np.sum(U**2, axis=0))
        return np.where(A > 0, se, 0.0)

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        s: float,
        on: str = "ff",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "log-log",
    ) -> npt.NDArray:
        r"""
        Robust pointwise confidence bounds on the dynamic prediction
        (:meth:`ff`, the default), :meth:`sf` or :meth:`Hf` from the
        landmark ``s`` to ``x``.

        The standard error of :math:`\log \hat H(x \mid s, Z)` is the
        infinitesimal jackknife: each stacked row's influence on the
        baseline hazard over :math:`(s, x]` and on the coefficients,
        summed over the rows of each unit (a unit appears at several
        landmarks), and squared and added over the units. It is the
        robust variance that the coefficients' standard errors use, carried
        to the prediction; at tied times it takes Breslow's form of the
        baseline. The bound is normal on the scale of ``bound_type``:
        ``"log-log"`` (the default, :math:`H e^{\pm z\,\mathrm{se}}`),
        ``"log"`` or ``"plain"``, as for :meth:`~surpyval.CoxPH`'s
        ``cb``. Only these Wald bounds are offered: the stacked rows have
        no likelihood of their own for a likelihood-ratio bound, and no
        bootstrap is built (resample the units, refit, and take
        quantiles, for one).

        Returns ``[lower, upper]`` on a last axis of 2 for two-sided
        bounds, one value per ``x`` otherwise.
        """
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        on = {"R": "sf", "F": "ff"}.get(on, on)
        check_option("bound", bound, ("two-sided", "lower", "upper"))
        check_option(
            "bound_type",
            bound_type,
            ("log", "log-log", "plain", "exp", "normal"),
        )
        bound_type = {"exp": "log-log", "normal": "plain"}.get(
            bound_type, bound_type
        )
        check_alpha_ci(alpha_ci)
        H, X, tc, shape, stratum, s = self._Hf(x, Z, s)
        if H.size == 0:
            return np.empty(shape + ((2,) if bound == "two-sided" else ()))
        se_log = self._log_H_se(X, tc, stratum, s)
        se = np.where(np.isnan(H), np.nan, se_log * np.nan_to_num(H))
        z = norm.isf(alpha_ci / 2 if bound == "two-sided" else alpha_ci)
        H_lo, H_hi = _survival_bounds(H, se, z, bound_type)
        if bound != "two-sided":
            upper_H = (bound == "upper") != (on == "sf")
            H_lo = H_hi = H_hi if upper_H else H_lo
        if on == "Hf":
            lo, hi = H_lo, H_hi
        elif on == "sf":
            lo, hi = np.exp(-H_hi), np.exp(-H_lo)
        else:
            lo, hi = -np.expm1(-H_lo), -np.expm1(-H_hi)
        if bound == "two-sided":
            return np.stack([lo, hi], axis=-1).reshape(shape + (2,))
        return lo.reshape(shape)

    def table(
        self,
        Z: Any,
        landmarks: npt.ArrayLike | None = None,
        window: float | None = None,
        alpha_ci: float = 0.05,
        bound_type: str = "log-log",
    ) -> Any:
        """
        The dynamic prediction table: for each landmark (default: those
        fitted) and covariate profile (the rows of ``Z``; a DataFrame's
        index labels the profiles), the chance of the event within
        ``window`` (default the fitted ``w``) and its robust bounds.

        Returns a DataFrame with the columns ``landmark``, ``profile``,
        ``probability``, ``lower`` and ``upper``.
        """
        import pandas as pd

        w = self._check_window(window)
        s_all = (
            self.landmarks
            if landmarks is None
            else np.atleast_1d(np.asarray(landmarks, dtype=float))
        )
        rows, _ = self._profiles(Z)
        labels = (
            list(Z.index)
            if isinstance(Z, pd.DataFrame)
            else list(range(rows.shape[0]))
        )
        records = []
        for sv in s_all:
            t = np.full(rows.shape[0], sv + w)
            p = self.ff(t, rows, sv)
            b = self.cb(t, rows, sv, alpha_ci=alpha_ci, bound_type=bound_type)
            for k, label in enumerate(labels):
                records.append(
                    {
                        "landmark": float(sv),
                        "profile": label,
                        "probability": float(p[k]),
                        "lower": float(b[k, 0]),
                        "upper": float(b[k, 1]),
                    }
                )
        return pd.DataFrame(records)

    # -- saving ---------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict. The data
        (``life``, by its own ``to_dict``) and the options are stored and
        the model is refitted on loading, so it predicts and bounds
        exactly as the original. A model with a basis of its own
        (``basis=``) cannot be saved.
        """
        if self._basis is not None:
            raise ValueError(
                "A landmark model with a basis of its own (basis=) cannot "
                "be saved, as a function cannot be written to JSON: refit "
                "it, or use the polynomial basis (degree=)."
            )
        return stamp_schema(
            {
                "model": "LandmarkCox",
                "life": self._life.to_dict(),
                "at": self.landmarks.tolist(),
                "horizon": self.window,
                "baseline": self.baseline,
                "degree": self.degree,
                "varying": [to_native(v) for v in self.varying],
                "tie_method": self.tie_method,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> LandmarkCox:
        """Rebuild the model from :meth:`to_dict` (by refitting)."""
        from surpyval.utils.surpyval_data import SurpyvalData

        require_model_tag(model_dict, "LandmarkCox", "a landmark Cox model")
        life = SurpyvalData.from_dict(model_dict["life"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls._fit(
                Lives(life),
                life,
                model_dict["at"],
                model_dict["horizon"],
                model_dict["baseline"],
                int(model_dict["degree"]),
                None,
                model_dict["varying"],
                model_dict["tie_method"],
            )

    def __repr__(self) -> str:
        lines = [
            "Landmark Cox model"
            + (" (supermodel)" if self.landmarks.size > 1 else ""),
            "Landmarks: {}".format(
                ", ".join(f"{v:g}" for v in self.landmarks)
            ),
            f"Window: {self.window:g}",
            f"Baseline: {self.baseline}",
            f"Units: {self.n_units}, stacked rows: {self.n_rows}",
            "Coefficients (robust standard errors, clustered on the unit):",
        ]
        se = self.standard_errors()
        for name, b, e in zip(self.parameter_names, self.params, se):
            lines.append(f"    {name}: {b:.6g} (se {e:.4g})")
        return "\n".join(lines)
