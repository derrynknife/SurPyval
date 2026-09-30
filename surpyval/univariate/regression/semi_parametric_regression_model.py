from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils import is_missing_event
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)

from ._summary import (
    coefficient_names,
    coefficient_repr,
    coefficient_table,
)
from .regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd


class SemiParametricRegressionModel(SerialisableMixin):
    """
    The fitted Cox proportional hazards model returned by ``CoxPH.fit``,
    ``fit_from_df``, ``fit_tvc`` and ``fit_tvc_timeline``.

    ``params`` (also ``beta``) are the coefficients, ``p_values`` their
    Wald p-values, and ``x``, ``h0``, ``H0`` the baseline hazard
    increments (Breslow's estimator, with Efron's tie correction after an
    Efron fit) and cumulative hazard at the distinct observed times (the
    increment is 0 at a censoring time). The baseline is that of a unit
    at ``center``: ``Z = 0`` by default (zeros), or the covariate means
    for a fit with ``center=True``; ``phi(Z)``, the hazard multiplier, is
    relative to it, :math:`e^{\\beta' (Z - \\text{center})}`. The
    survival functions take
    the covariates as a second argument, ``sf(x, Z)`` (and a ``stratum``
    for a stratified fit); ``sf_tvc`` / ``Hf_tvc`` follow a time-varying
    covariate path. The model also provides residuals, the
    proportional-hazards test (``check_ph``), cluster-robust standard
    errors and serialisation.

    Examples
    --------
    Fitted to the Rossi recidivism data, where ``arrest`` is 1 for an
    arrest (so ``c = 1 - arrest``); ``exp(params)`` are the hazard ratios:

    >>> import numpy as np
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> model = CoxPH.fit(x, Z, c=c)
    >>> np.exp(model.params).round(4)
    array([0.7068, 0.9351, 1.1017])

    The chance of no arrest in the first year, without and with
    financial aid, for a 25-year-old with three prior convictions:

    >>> model.sf([52], [0, 25, 3]).round(4)
    array([0.7246])
    >>> model.sf([52], [1, 25, 3]).round(4)
    array([0.7963])
    """

    # Covariate metadata populated when the model is fit from a pandas
    # DataFrame via ``CoxPH.fit_from_df``.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    #: True when fitted from time-varying-covariate (start-stop) data via
    #: ``CoxPH.fit_tvc``; enables :meth:`predict_tvc`.
    is_tvc: bool = False
    #: True when fitted with ``strata=...`` (a separate baseline hazard per
    #: stratum, shared coefficients). Prediction then requires a ``stratum``.
    is_stratified: bool = False
    #: For a stratified fit, the list of stratum labels.
    strata_labels: Any = None
    #: For a stratified fit, ``{label: {"x", "r", "d", "h0", "H0"}}``.
    strata_baselines: Any = None
    #: The covariate values the baseline is at: zeros (``Z = 0``) by
    #: default, the ``n``-weighted column means of the fitted rows for a
    #: fit with ``center=True`` (#459, #463). The fit centres on the means
    #: either way. ``None`` (a model built by hand) is read as zeros.
    center: "npt.NDArray | None" = None
    #: The covariate means the fit centred on, which the residuals and
    #: diagnostics centre on too; not saved (they need the fitted data).
    _fit_center: "npt.NDArray | None" = None

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: ``beta_0``,
        ``beta_1``, ... for the covariate coefficients, as in the
        parametric regression models."""
        return ["beta_{}".format(i) for i in range(len(self.params))]

    # Attributes populated by the fitter (``CoxPH.fit`` / ``fit_from_df``).
    params: npt.NDArray
    beta: npt.NDArray
    x: npt.NDArray
    r: npt.NDArray
    d: npt.NDArray
    tl: Any
    h0: npt.NDArray
    H0: npt.NDArray
    p_values: npt.NDArray
    #: The coefficients' standard errors, from the observed information
    #: (``None`` for a model saved before they were stored).
    se: "npt.NDArray | None" = None
    #: The fit's score/Hessian and negative-partial-log-likelihood
    #: closures (the scalar value is ``_neg_log_like``).
    jac: Callable
    hess: Callable
    res: Any
    neg_ll: Callable
    _neg_log_like: float
    tie_method: str
    baseline_method: str
    #: Per-observation training data (``x``/``c``/``n``/``Z``/``tl``) retained
    #: by ``CoxPH.fit`` for residuals and the proportional-hazards test.
    _fit_data: dict
    #: For a TVC (start-stop) fit: subject id per *internal* (sorted) row,
    #: and the permutation from the caller's row order to the internal order
    #: (#259 — used to align user-supplied cluster labels).
    tvc_subject_ids: "npt.NDArray | None" = None
    tvc_row_order: "npt.NDArray | None" = None

    def __init__(self, kind: str, parameterization: str) -> None:
        self.kind = kind
        self.parameterization = parameterization

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        Convert ``Z`` to a numeric design matrix, selecting the covariate
        columns recorded at fit time when a pandas DataFrame is passed.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    def _center(self) -> npt.NDArray:
        """The centre as an array, zeros for a model without one."""
        beta = np.asarray(self.beta, dtype=float)
        if self.center is None:
            return np.zeros(beta.shape[0])
        return np.asarray(self.center, dtype=float)

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (#476): a constant column, one constant within each
        stratum, or a linear combination of the others. Their ``beta`` is
        ``nan`` (R's ``NA``), and predictions take it as 0."""
        return np.flatnonzero(np.isnan(np.asarray(self.beta, dtype=float)))

    def _coef(self) -> npt.NDArray:
        """``beta`` with an aliased coefficient as 0, as the predictions
        use it (R's ``predict.coxph`` does the same)."""
        beta = np.asarray(self.beta, dtype=float)
        return np.where(np.isnan(beta), 0.0, beta)

    def _log_risk(self, Z: npt.NDArray) -> npt.NDArray:
        """``beta'(Z - center)`` for numeric covariate rows ``Z``, the log
        of the multiplier of the baseline."""
        return (Z - self._center()) @ self._coef()

    @staticmethod
    def _times_risk(base: npt.NDArray, log_risk: npt.NDArray) -> npt.NDArray:
        """``base * exp(log_risk)`` computed as ``exp(log(base) +
        log_risk)``: a baseline at ``Z = 0`` far from the data is tiny and
        ``exp(beta'Z)`` huge, and their product is formed without either
        overflowing (#463). A zero baseline stays 0, and nan stays nan."""
        with np.errstate(divide="ignore", over="ignore"):
            return np.exp(np.log(base) + log_risk)

    def phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        The hazard multiplier :math:`e^{\\beta' (Z - \\text{center})}` for
        covariates ``Z``: the hazard ratio of ``Z`` against a unit at
        ``center``, whose hazard the baseline ``h0`` is -- :math:`e^{\\beta'
        Z}`, against ``Z = 0``, by default, and against the covariate means
        for a fit with ``center=True`` (R's ``predict(fit, type =
        "risk")``). ``Z`` is a single row, one row per prediction, a
        DataFrame for a model fitted with ``fit_from_df``, or a scalar for a
        one-covariate model (as the parametric families accept; it used to
        fail in the matrix product). The ratio of two multipliers is the
        hazard ratio of the two rows. On covariates far from ``center`` the
        multiplier can overflow to ``inf`` (the predictions do not: they
        combine it with the baseline on the log scale).
        """
        with np.errstate(over="ignore"):
            return np.exp(self._log_phi(Z))

    def _log_phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """The log of :meth:`phi`."""
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        if Z_arr.ndim == 0:
            Z_arr = Z_arr.reshape(1)
        return self._log_risk(Z_arr)

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Hazards"
            + "\nKind                : {kind}"
            + "\nParameterization    : {parameterization}"
        ).format(kind=self.kind, parameterization=self.parameterization)
        if np.any(self._center()):
            out += (
                "\nBaseline at         : the covariate means, Z = {}".format(
                    np.array2string(self._center(), separator=", ")
                )
            )

        tie_method = getattr(self, "tie_method", None)
        if tie_method is not None:
            out += "\nTie method          : {}".format(tie_method)
        out += (
            "\nCoefficients        : exp(coef) is the hazard ratio; Wald "
            "95% intervals\n"
        )
        return out + coefficient_repr(self.summary()) + "\n"

    def summary(
        self,
        alpha_ci: float = 0.05,
        robust: bool = False,
        cluster: "npt.ArrayLike | None" = None,
    ) -> "pd.DataFrame":
        """
        The coefficient table, as R's ``summary(coxph)`` and lifelines'
        ``summary`` give it (#484): one row per covariate (named by
        ``feature_names`` for a model fitted with ``fit_from_df``), with the
        coefficient, the hazard ratio ``exp(coef)``, the standard error, a
        two-sided ``1 - alpha_ci`` Wald interval for both, the Wald
        statistic ``z`` and its two-sided p-value. An aliased coefficient
        (#476) is ``nan`` throughout.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.
        robust : bool, optional
            Use the cluster-robust (sandwich) standard errors of
            :meth:`robust_summary` instead of the model-based ones.
        cluster : array_like, optional
            With ``robust=True``, the cluster label of each row (as for
            :meth:`robust_covariance`).

        Returns
        -------
        pandas.DataFrame
            Columns ``coef``, ``exp(coef)``, ``se(coef)``, ``coef lower
            95%``, ``coef upper 95%``, ``exp(coef) lower 95%``, ``exp(coef)
            upper 95%``, ``z`` and ``p`` (the level follows ``alpha_ci``).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]  # arrest is 1 for an arrest
        >>> model = CoxPH.fit_from_df(
        ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
        ... )
        >>> model.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(4)
                     coef  exp(coef)  se(coef)       p
        covariate
        fin       -0.3279     0.7204    0.1899  0.0841
        age       -0.0715     0.9310    0.0209  0.0006
        """
        beta = np.asarray(self.beta, dtype=float)
        names = coefficient_names(self, beta.size)
        if robust:
            se = np.asarray(self.robust_summary(cluster)["se"], dtype=float)
            return coefficient_table(names, beta, se, alpha_ci)
        se = getattr(self, "se", None)
        p_values = getattr(self, "p_values", None)
        if se is None and p_values is None:
            se = np.full(beta.shape, np.nan)
        elif se is None:
            # A model saved before the standard errors were: they follow
            # from the Wald p-values, |beta| / z.
            with np.errstate(divide="ignore", invalid="ignore"):
                z = norm.isf(np.asarray(p_values, dtype=float) / 2)
                se = np.abs(beta) / z
        return coefficient_table(names, beta, se, alpha_ci, p=p_values)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Cox model to a plain, JSON-serialisable dict.

        The Cox model is a proportional-hazards fit with a *nonparametric*
        baseline, so what is stored is the covariate coefficients ``beta`` and
        the fitted baseline step arrays (event times ``x`` and the baseline
        hazard ``h0`` / cumulative hazard ``H0``); the hazard multiplier
        ``phi(Z) = exp(beta'(Z - center))`` is rebuilt from ``beta`` on load,
        with the covariate means ``center`` for a fit with ``center=True``
        (stored only then, which makes the dict schema 2). Everything
        needed for ``hf``/``Hf``/``sf``/``ff``/``df`` (and, for a
        time-varying-covariate fit, ``predict_tvc``) round-trips exactly. The
        optimiser objects (the ``neg_ll`` closure, ``jac``, ``hess``, ``res``)
        are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        if self.is_stratified:
            raise NotImplementedError(
                "Serialisation of stratified Cox models is not supported "
                "(each stratum carries its own baseline hazard)."
            )
        out: dict[str, Any] = {
            "model": "SemiParametricRegressionModel",
            "kind": self.kind,
            "parameterization": self.parameterization,
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "x": np.asarray(self.x, dtype=float).tolist(),
            "r": np.asarray(self.r, dtype=float).tolist(),
            "d": np.asarray(self.d, dtype=float).tolist(),
            "h0": np.asarray(self.h0, dtype=float).tolist(),
            "H0": np.asarray(self.H0, dtype=float).tolist(),
            "tie_method": self.tie_method,
            "baseline_method": self.baseline_method,
            "is_tvc": bool(self.is_tvc),
        }
        if np.any(self._center()):
            # A baseline at the covariate means (center=True, #459): stored,
            # which makes the dict schema 2, as a schema-1 reader would read
            # the baseline as at Z = 0.
            out["center"] = self._center().tolist()
        if getattr(self, "tl", None) is not None:
            tl = np.asarray(self.tl, dtype=float)
            # No delayed entry is stored as -inf. Omit the array when no
            # row has an entry time; otherwise stamp_schema writes each
            # -inf as null with a "non_finite" record, like every other
            # model's non-finite values.
            if np.isfinite(tl).any():
                out["tl"] = tl.tolist()
        if getattr(self, "p_values", None) is not None:
            out["p_values"] = np.asarray(self.p_values, dtype=float).tolist()
        if getattr(self, "se", None) is not None:
            out["se"] = np.asarray(self.se, dtype=float).tolist()
        if getattr(self, "_neg_log_like", None) is not None:
            out["_neg_log_like"] = float(self._neg_log_like)
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "SemiParametricRegressionModel":
        """
        Rebuild a Cox model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "SemiParametricRegressionModel", "a Cox model"
        )
        out = cls(model_dict["kind"], model_dict["parameterization"])
        beta = np.array(model_dict["beta"], dtype=float)
        out.beta = beta
        out.params = np.array(model_dict["params"], dtype=float)
        # A dict without a "center" (the default fit, or one written
        # before #459) has its baseline at Z = 0.
        out.center = np.array(
            model_dict.get("center", np.zeros(beta.shape[0])), dtype=float
        )
        out.x = np.array(model_dict["x"], dtype=float)
        out.r = np.array(model_dict["r"], dtype=float)
        out.d = np.array(model_dict["d"], dtype=float)
        out.h0 = np.array(model_dict["h0"], dtype=float)
        out.H0 = np.array(model_dict["H0"], dtype=float)
        # phi is fully determined by beta and center (the ``phi`` method).
        out.tie_method = model_dict["tie_method"]
        out.baseline_method = model_dict["baseline_method"]
        out.is_tvc = bool(model_dict.get("is_tvc", False))
        # A bare null (no "non_finite" record) is how schema-1 dicts wrote
        # a row without delayed entry; it still reads as -inf.
        out.tl = (
            np.array(
                [-np.inf if v is None else v for v in model_dict["tl"]],
                dtype=float,
            )
            if model_dict.get("tl") is not None
            else None
        )
        if "p_values" in model_dict:
            out.p_values = np.array(model_dict["p_values"], dtype=float)
        if "se" in model_dict:
            out.se = np.array(model_dict["se"], dtype=float)
        if "_neg_log_like" in model_dict:
            out._neg_log_like = float(model_dict["_neg_log_like"])
        restore_covariate_meta(out, model_dict)
        return out

    def _missing_stratum(self, stratum: Any) -> bool:
        """Whether ``stratum`` is a missing label (``NaN`` or pandas
        ``NA``) for a stratified fit. ``None`` is not: it is the default
        and means no stratum was given."""
        return (
            self.is_stratified
            and stratum is not None
            and is_missing_event(stratum)
        )

    def _baseline_arrays(
        self, stratum: Any
    ) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray]":
        """Baseline ``(x, h0, H0)`` arrays, selecting a stratum if needed."""
        if self.is_stratified:
            if self._missing_stratum(stratum):
                # A missing stratum label predicts nan, as a missing
                # covariate does; the first stratum's times only give the
                # output its shape.
                b = self.strata_baselines[self.strata_labels[0]]
                nan = np.full(np.shape(b["h0"]), np.nan)
                return b["x"], nan, nan
            if stratum is None:
                raise ValueError(
                    "this is a stratified Cox model; pass stratum=... to "
                    "select which stratum's baseline hazard to use "
                    "(one of {})".format(self.strata_labels)
                )
            if stratum not in self.strata_baselines:
                raise ValueError(
                    "unknown stratum {!r}; known strata are {}".format(
                        stratum, self.strata_labels
                    )
                )
            b = self.strata_baselines[stratum]
            return b["x"], b["h0"], b["H0"]
        if stratum is not None:
            raise ValueError(
                "'stratum' was given but this model is not stratified"
            )
        return self.x, self.h0, self.H0

    @staticmethod
    def _baseline_step(
        bx: npt.NDArray, values: npt.NDArray, x: npt.ArrayLike
    ) -> npt.NDArray:
        """
        The baseline step function ``values`` (jumping at the event times
        ``bx``) evaluated at ``x``, in the order ``x`` was given. Before the
        first event time nothing has happened yet, so the value is 0 (nan
        where ``values`` is all nan, for a missing stratum). A missing
        (``NaN``) time gives nan: ``searchsorted`` places it after every
        event time, which read it as the value at ``t = inf``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(bx, x, side="right") - 1
        before = 0.0 * values[0] if values.size else 0.0
        out = np.where(idx >= 0, values[np.maximum(idx, 0)], before)
        return np.where(np.isnan(x), np.nan, out)

    def _scaled_step(
        self,
        values: npt.NDArray,
        bx: npt.NDArray,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        grid: bool,
    ) -> npt.NDArray:
        """The baseline step function ``values`` at ``x`` times ``phi(Z)``:
        paired (row ``i`` of ``Z`` with ``x[i]``, or one of them single),
        or on the grid of every row by every time, ``(len(Z), len(x))``."""
        base = self._baseline_step(bx, values, x)
        if grid:
            rows = covariate_rows(
                self._prepare_Z(Z), np.asarray(self.beta).shape[0]
            )
            log_risk = self._log_risk(rows)
            return self._times_risk(base[None, :], log_risk[:, None])
        log_risk = self._log_phi(Z)
        check_paired_rows(base.size, np.size(log_risk))
        return self._times_risk(base, log_risk)

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Hazard at ``x`` for covariates ``Z``: the baseline hazard
        increment at the latest baseline time at or before ``x``, times
        ``phi(Z)``. It is a step size, not a smooth hazard rate; the
        baseline times ``self.x`` include the censoring times, where the
        increment is 0. ``Z`` is one row (used for every ``x``) or one row
        per ``x``, paired in the order given.
        With ``grid=True`` every time is evaluated for every row of ``Z``
        (a survival curve per subject, lifelines'
        ``predict_survival_function``): the result has shape ``(len(Z),) +
        x.shape``, row ``i`` for row ``i`` of ``Z``, as a survival
        forest's. Without, rows and times of different counts (neither
        one) are refused (#488).
        """
        bx, bh0, _ = self._baseline_arrays(stratum)
        return self._scaled_step(bh0, bx, x, Z, grid)

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Cumulative hazard at ``x`` for covariates ``Z``: the baseline
        ``H0(x)`` (0 before the first event time) times
        ``phi(Z)``. ``Z`` is one row (used for every ``x``) or one row per
        ``x``, paired in the order given.
        With ``grid=True`` every time is evaluated for every row of ``Z``
        (a survival curve per subject, lifelines'
        ``predict_survival_function``): the result has shape ``(len(Z),) +
        x.shape``, row ``i`` for row ``i`` of ``Z``, as a survival
        forest's. Without, rows and times of different counts (neither
        one) are refused (#488).
        """
        bx, _, bH0 = self._baseline_arrays(stratum)
        return self._scaled_step(bH0, bx, x, Z, grid)

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Survival :math:`e^{-H_0(x) e^{\\beta' Z}}` at ``x`` for covariates
        ``Z`` (one row, or one row per ``x``); ``stratum`` selects the
        baseline of a stratified fit. A missing (``NaN``) time, covariate
        or stratum label gives ``nan`` in its place.
        With ``grid=True`` every time is evaluated for every row of ``Z``
        (a survival curve per subject, lifelines'
        ``predict_survival_function``): the result has shape ``(len(Z),) +
        x.shape``, row ``i`` for row ``i`` of ``Z``, as a survival
        forest's. Without, rows and times of different counts (neither
        one) are refused (#488).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = CoxPH.fit(x, df[["fin", "age"]].values, c=c)
        >>> subjects = [[0, 20], [1, 20], [0, 40]]
        >>> model.sf([10, 30, 50], subjects, grid=True).round(3)
        array([[0.949, 0.8  , 0.641],
               [0.963, 0.851, 0.726],
               [0.987, 0.948, 0.899]])
        """
        return np.exp(-self.Hf(x, Z, stratum, grid=grid))

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Failure probability ``1 - sf`` at ``x`` for covariates ``Z``;
        arguments as for :meth:`sf`.
        """
        return -np.expm1(-self.Hf(x, Z, stratum, grid=grid))

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        stratum: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        ``hf * sf`` at ``x`` for covariates ``Z``: the probability mass at
        each baseline event time (the baseline is a step function);
        arguments as for :meth:`sf`.
        """
        return self.hf(x, Z, stratum, grid=grid) * self.sf(
            x, Z, stratum, grid=grid
        )

    def compute_residuals(self, kind: str = "martingale") -> npt.NDArray:
        """
        Residuals for a fitted Cox proportional-hazards model.

        ``kind`` is one of ``"schoenfeld"``, ``"scaled_schoenfeld"``,
        ``"martingale"``, ``"deviance"``, ``"score"`` or ``"dfbeta"``. See
        :func:`~surpyval.univariate.regression.proportional_hazards.
        diagnostics.compute_residuals` for the definitions and uses.
        """
        from .proportional_hazards.diagnostics import compute_residuals

        return compute_residuals(self, kind)

    def check_ph(self, transform: str = "km") -> dict:
        """
        Test the proportional-hazards assumption (Grambsch-Therneau).

        Returns a dict with a joint ``global`` test and a ``per_covariate``
        list; a small ``p_value`` is evidence against proportional hazards.
        See :func:`~surpyval.univariate.regression.proportional_hazards.
        diagnostics.check_ph`.
        """
        from .proportional_hazards.diagnostics import check_ph

        return check_ph(self, transform)

    def robust_covariance(
        self, cluster: "npt.ArrayLike | None" = None
    ) -> npt.NDArray:
        """
        Cluster-robust ("sandwich") covariance of the coefficients.

        Pass ``cluster`` (one label per observation) for correlated /
        clustered data; ``None`` gives the ordinary robust variance. See
        :func:`~surpyval.univariate.regression.proportional_hazards.
        diagnostics.robust_covariance`.
        """
        from .proportional_hazards.diagnostics import robust_covariance

        return robust_covariance(self, cluster)

    def robust_summary(self, cluster: "npt.ArrayLike | None" = None) -> dict:
        """
        Cluster-robust standard errors, z-scores and p-values for the
        coefficients. See :func:`~surpyval.univariate.regression.
        proportional_hazards.diagnostics.robust_summary`.
        """
        from .proportional_hazards.diagnostics import robust_summary

        return robust_summary(self, cluster)

    def predict_tvc(
        self,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        Z: npt.ArrayLike,
        times: "npt.ArrayLike | None" = None,
        stratum: Any = None,
    ) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray]":
        r"""
        Survival for a subject whose covariates vary over time.

        With a time-varying covariate the survival function depends on the
        whole covariate path, not a single vector:

        .. math::
            H(t \mid Z(\cdot)) = \int_0^t e^{\beta' Z(u)}\, dH_0(u)
            = \sum_{u_j \le t} h_0(u_j)\, e^{\beta' Z(u_j)},

        summing the fitted baseline-hazard jumps ``h0`` weighted by the hazard
        multiplier of the covariate value *active* at each jump time (the
        baseline is that of a unit at ``center``, so the multiplier is
        :meth:`phi`, :math:`e^{\beta' (Z(u_j) - \text{center})}`). With a
        single constant interval this reduces exactly to ``sf(t, Z)``.

        Parameters
        ----------
        xl, xr : array_like
            The subject's covariate-path intervals ``(xl, xr]``, one per row
            (as given to :meth:`~...CoxPH.fit_tvc`). Usually contiguous from
            ``0``.
        Z : array_like
            The covariate row active on each interval, one row per interval.
        times : array_like, optional
            Times at which to return survival. Defaults to the fitted baseline
            jump times that fall within the covariate path.
        stratum : optional
            For a stratified fit, the stratum whose baseline hazard to use
            (required there, as for :meth:`sf`).

        Returns
        -------
        times, sf, Hf : ndarray
            The evaluation times and the survival and cumulative-hazard values
            of the subject along its covariate path. Outside the supplied path
            the nearest interval's covariate is held constant.
        """
        xl_a = np.atleast_1d(np.asarray(xl, dtype=float))
        xr_a = np.atleast_1d(np.asarray(xr, dtype=float))
        Z_a = np.asarray(Z, dtype=float)
        if Z_a.ndim == 1:
            Z_a = Z_a.reshape(-1, 1)
        if not (xl_a.shape[0] == xr_a.shape[0] == Z_a.shape[0]):
            raise ValueError("xl, xr and Z must have the same number of rows")
        # The intervals and covariates describe one subject's history, so a
        # missing value in them is refused rather than predicted around: a
        # nan interval end was ignored and a nan covariate made every time
        # nan, including those before it applied.
        for name, arr in (("xl", xl_a), ("xr", xr_a), ("Z", Z_a)):
            if np.isnan(arr).any():
                raise ValueError(
                    "'{}' has a missing (NaN) value; the covariate path "
                    "of one subject must be complete.".format(name)
                )
        if np.any(xl_a >= xr_a):
            raise ValueError("every interval must have xl < xr")

        order = np.argsort(xl_a)
        xl_a, xr_a, Z_a = xl_a[order], xr_a[order], Z_a[order]

        # The active interval at a baseline jump time u follows the fitted
        # (xl, xr] convention: the interval with xl < u <= xr, i.e. the OLD
        # covariate is still at risk at exactly its stop time (#259 —
        # ``side="right"`` credited a jump at a change time to the NEW
        # covariate, contradicting the likelihood). Times outside the path
        # are clamped to the first/last interval (covariate held constant).
        # A stratified fit has one baseline per stratum; the first stratum's
        # used to be taken silently.
        base_t, base_h0, _ = self._baseline_arrays(stratum)
        if times is None:
            within = (base_t > xl_a[0]) & (base_t <= xr_a[-1])
            query = base_t[within]
        else:
            query = np.atleast_1d(np.asarray(times, dtype=float))

        Hf = self._tvc_cumhaz(query, xl_a, Z_a, base_t, base_h0)
        return query, np.exp(-Hf), Hf

    def _tvc_cumhaz(
        self,
        query: npt.NDArray,
        starts: npt.NDArray,
        Zseg: npt.NDArray,
        base_t: npt.NDArray,
        base_h0: npt.NDArray,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard of the fitted baseline at each ``query`` time for a
        covariate that takes value ``Zseg[i]`` on the segment starting at
        ``starts[i]``:

        .. math::
            H(t) = \sum_{u_j \le t} h_0(u_j)\, e^{\beta' (Z(u_j) - c)},

        summing the baseline-hazard jumps ``h0`` at the fitted event times
        weighted by the multiplier of the covariate *active* at each jump
        (``c`` the model's ``center``).
        ``base_t``/``base_h0`` are the baseline (of the stratum, if any).
        """
        # (xl, xr] convention, matching the fit: the old covariate is at
        # risk at exactly its stop time (#259).
        active = np.searchsorted(starts, base_t, side="left") - 1
        active = np.clip(active, 0, starts.shape[0] - 1)
        H_cum = np.cumsum(
            self._times_risk(base_h0, self._log_risk(Zseg[active]))
        )
        idx = np.searchsorted(base_t, query, side="right") - 1
        last = H_cum.shape[0] - 1
        out = np.where(idx >= 0, H_cum[np.clip(idx, 0, last)], 0.0)
        # A missing query time is nan, not the value after the last jump.
        return np.where(np.isnan(query), np.nan, out)

    @keeps_query_shape
    def Hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        stratum: Any = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard for a covariate following a path ``Z(t)``: a step
        schedule, or a continuously varying path.

        The Cox analogue of :meth:`predict_tvc` written to the shared
        time-varying-covariate convention used by the parametric families:
        ``Z`` is a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
        a :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`, or
        an array of per-segment covariate rows with ``xl`` giving the segment
        start times. The cumulative hazard sums the fitted baseline-hazard
        jumps weighted by the covariate active at each jump (see
        :meth:`_tvc_cumhaz`). The baseline is a step function, so along a
        continuously varying path this is still exact, with no quadrature:
        only the covariate just before each jump time counts (the value
        before a jump in the path, as for ``(start, stop]`` rows). The path
        is measured from time zero (a path starting after zero has its first
        value held back to zero; the part before zero is ignored), and any
        time is a valid query: ``H`` is ``0`` up to the first baseline jump.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the cumulative hazard.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path -- a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            a :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`,
            or per-segment covariate rows (with ``xl`` giving the segment start
            times).
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        stratum : optional
            For a stratified fit, the stratum whose baseline hazard to use
            (required there, as for :meth:`sf`).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import CoxPH, CovariatePath
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> Z = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        >>> model = CoxPH.fit(x, Z)
        >>> ramp = CovariatePath.from_points([0, 6], [0.0, 1.0])
        >>> H = model.Hf_tvc([2.5, 6.0], ramp)

        The same as the baseline jumps weighted by the ramp at each jump:

        >>> w = model.h0 * np.exp(model.beta[0] * model.x / 6)
        >>> bool(np.allclose(H, [w[model.x <= 2.5].sum(), w.sum()]))
        True
        """
        return self._hf_tvc(x, Z, xl, stratum)

    def _hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: Any,
        xl: "npt.ArrayLike | None",
        stratum: Any,
        given: "float | None" = None,
    ) -> npt.NDArray:
        """:meth:`Hf_tvc`, less its value at ``given`` for a
        ``CovariatePath`` given one (summed from ``given`` on)."""
        base_t, base_h0, _ = self._baseline_arrays(stratum)
        from .tvc_path import CovariatePath
        from .tvc_schedule import as_covariate_path, segments_from_origin

        schedule = as_covariate_path(Z, xl)
        n_cov = np.asarray(self.beta).shape[0]
        if schedule.p != n_cov:
            raise ValueError(
                "the {} has {} covariate(s) but the model was fit with "
                "{}".format(
                    (
                        "path"
                        if isinstance(schedule, CovariatePath)
                        else "schedule"
                    ),
                    schedule.p,
                    n_cov,
                )
            )
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        if np.isnan(xq).all():
            # Nothing to evaluate: a missing time is nan (the schedule
            # cannot be materialised to a nan horizon).
            return np.full(xq.shape, np.nan)
        if isinstance(schedule, CovariatePath):
            return self._tvc_cumhaz_path(xq, schedule, base_t, base_h0, given)
        # A horizon at or below 0 materialises the segment in force at 0
        # (H is 0 there, before the first baseline jump).
        t_max = float(np.nanmax(xq))
        starts, _, Zseg = segments_from_origin(schedule, t_max)
        return self._tvc_cumhaz(xq, starts, Zseg, base_t, base_h0)

    def _tvc_cumhaz_path(
        self,
        query: npt.NDArray,
        path: Any,
        base_t: npt.NDArray,
        base_h0: npt.NDArray,
        given: "float | None" = None,
    ) -> npt.NDArray:
        r"""
        :meth:`_tvc_cumhaz` along a continuously varying ``path`` (#172):

        .. math::
            H(t) = \sum_{u_j \le t} h_0(u_j)\, e^{\beta' (Z(u_j-) - c)},

        exact, with the path's value just before each jump time (its value
        at 0 for a jump at or before 0, where the path is held back to).
        With ``given`` the jumps in ``(given, t]`` are summed (negated for
        ``t < given``).
        """
        from .tvc_path import sum_between

        Z_at = np.asarray(path._values(np.maximum(base_t, 0.0)), dtype=float)
        at0 = base_t <= 0
        if at0.any():
            Z_at[at0] = path._values(np.zeros(1), left=False)[0]
        weight = self._times_risk(base_h0, self._log_risk(Z_at))
        # The jumps between baseline times: panel i is (t_{i-1}, t_i], so
        # a time's cumulative sum is that of the last baseline time at or
        # before it.
        edges = np.concatenate([[-np.inf], base_t])
        origin = -np.inf if given is None else float(given)

        def at_edge(t: npt.NDArray) -> npt.NDArray:
            # The last baseline time at or before t (-inf before the first).
            idx = np.searchsorted(base_t, t, side="right") - 1
            return np.where(idx >= 0, base_t[np.clip(idx, 0, None)], -np.inf)

        out = sum_between(
            edges,
            weight,
            float(at_edge(np.array([origin]))[0]),
            at_edge(np.where(np.isnan(query), -np.inf, query)),
        )
        # A missing query time is nan, not the value after the last jump.
        return np.where(np.isnan(query), np.nan, out)

    @keeps_query_shape
    def sf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
        stratum: Any = None,
    ) -> npt.NDArray:
        r"""
        Survival for a covariate following a path ``Z(t)``: a step
        (piecewise-constant) schedule, or a continuously varying path.

        The Cox counterpart of the parametric ``sf_tvc``: ``S(x) = exp(-H(x))``
        with ``H`` the baseline-jump sum of :meth:`Hf_tvc`, so every regression
        family -- parametric proportional/additive hazards and semi-parametric
        Cox -- shares one calling convention. ``predict_tvc`` remains for the
        interval-oriented ``(xl, xr, Z)`` form and returning the baseline jump
        times.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate survival.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path. A
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            (change-points, intervals, a cyclic pattern, or a step-valued
            expression), a
            :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
            (a covariate that changes continuously; exact for Cox, see
            :meth:`Hf_tvc`), or an array of per-segment covariate rows with
            ``xl`` giving the segment start times.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            If supplied, return the *conditional* survival given the item has
            survived to age ``given``:
            ``S(x | given) = exp(-(H(x) - H(given)))``. Along a
            ``CovariatePath`` the baseline jumps after ``given`` are summed,
            so nothing is subtracted.
        stratum : optional
            For a stratified fit, the stratum whose baseline hazard to use
            (required there, as for :meth:`sf`).

        Returns
        -------
        ndarray
            Survival at each ``x`` (conditional on ``given`` when supplied).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import CoxPH, CovariatePath
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> Z = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        >>> model = CoxPH.fit(x, Z)
        >>> ramp = CovariatePath.from_points([0, 6], [0.0, 1.0])
        >>> S = model.sf_tvc([2.5, 4.5], ramp)
        >>> S_given = model.sf_tvc([2.5, 4.5], ramp, given=2.5)
        >>> bool(np.allclose(S_given, S / S[0]))
        True
        """
        from .tvc_path import CovariatePath

        if (
            isinstance(Z, CovariatePath)
            and given is not None
            and not np.isnan(float(given))
        ):
            # Summed from given on.
            return np.exp(-self._hf_tvc(x, Z, xl, stratum, given=float(given)))
        H = self._hf_tvc(x, Z, xl, stratum)
        if given is not None:
            given = float(given)
            if np.isnan(given):
                # A missing conditioning age: nothing is known.
                H = np.full(np.shape(H), np.nan)
            else:
                H = H - self._hf_tvc(given, Z, xl, stratum)
        return np.exp(-H)
