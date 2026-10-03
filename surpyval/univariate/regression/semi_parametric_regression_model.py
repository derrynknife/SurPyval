from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import InformationCriteriaMixin
from surpyval.utils import is_missing_event
from surpyval.utils.data_summary import data_summary
from surpyval.utils.deprecation import REMOVED_IN_NEXT, RenamedToMethod
from surpyval.utils.linalg import standard_errors_of
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.validation import no_covariance_error

from ._concordance import ConcordanceMixin
from ._prediction import ConditionalSurvivalMixin
from ._summary import (
    coefficient_names,
    coefficient_repr,
    coefficient_table,
)
from .regression_data import (
    LinearPredictorMixin,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd


class SemiParametricRegressionModel(
    ConditionalSurvivalMixin,
    InformationCriteriaMixin,
    LinearPredictorMixin,
    ConcordanceMixin,
    SerialisableMixin,
):
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
    #: What the fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``): ``"verified"`` (a zero score and a
    #: positive-definite information), ``"unverified"`` or ``"no finite
    #: maximum"`` (a monotone partial likelihood), each as the fit's
    #: warnings say; ``"unknown"`` for a model restored from a dict saved
    #: without it.
    maximum: str = "unknown"

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: each covariate's
        column (a formula, ``fit_from_df`` or a DataFrame ``Z``), else
        ``coef_0``, ``coef_1``, ... (#614), as in the parametric
        regression models."""
        return coefficient_names(self, len(self.params))

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
    #: The coefficients' covariance, ``covariance()``: the inverse of the
    #: observed information (#613; ``None`` for a model saved before it
    #: was stored).
    _covariance: "npt.NDArray | None" = None
    #: The coefficients' standard errors, ``standard_errors()`` (``None``
    #: for a model saved before they were stored).
    _se: "npt.NDArray | None" = None
    #: ``standard_errors()``'s name before v0.23, for one release.
    se = RenamedToMethod("standard_errors", "_se")
    #: The fit's score/Hessian closure, ``jac(beta) -> (score,
    #: information)``, and the negative partial log-likelihood as a
    #: function of the coefficients, ``neg_ll_of(beta)`` (``None`` for a
    #: model restored from a dict); its value at the fit is ``neg_ll()``.
    jac: Callable
    hess: Callable
    res: Any
    neg_ll_of: "Callable | None" = None
    tie_method: str
    baseline_method: str
    #: Per-observation training data (``x``/``c``/``n``/``Z``/``tl``) retained
    #: by ``CoxPH.fit`` for residuals and the proportional-hazards test.
    _fit_data: dict
    #: The printout's "Data" line of a model restored by ``from_dict``
    #: (#508), which does not hold its data.
    _data_summary: "str | None" = None
    #: For a TVC (start-stop) fit: subject id per *internal* (sorted) row,
    #: and the permutation from the caller's row order to the internal order
    #: (#259 — used to align user-supplied cluster labels).
    tvc_subject_ids: "npt.NDArray | None" = None
    tvc_row_order: "npt.NDArray | None" = None

    #: The model (``"Cox"``) and ``"Semi-Parametric"``, which the
    #: printout shows and ``to_dict`` stores.
    kind: str
    parameterization: str

    _ALIASED_WHY = (
        "a constant column, one constant within each stratum, or a linear "
        "combination of the others"
    )

    def __init__(self, kind: str, parameterization: str) -> None:
        self.kind = kind
        self.parameterization = parameterization

    # -- model comparison (#604) -------------------------------------------

    def neg_ll(self, beta: Any = None) -> float:
        """The negative partial log-likelihood at the fitted coefficients:
        a number, as every model's ``neg_ll()`` is (#604), so that
        :meth:`aic`, :meth:`bic` and :meth:`aic_c`, and ``log_likelihood``
        (its negative, R's ``logLik(fit)``), compare Cox models with each
        other. The partial likelihood is not the likelihood of the data,
        so these do not compare a Cox model with a parametric one.

        As a function of the coefficients it is ``neg_ll_of(beta)``;
        ``neg_ll(beta)``, its old spelling, still gives it with a
        ``DeprecationWarning`` until v0.24.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = CoxPH.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> round(model.log_likelihood, 3), round(model.aic(), 3)
        (-660.857, 1327.714)
        """
        if beta is not None:
            warnings.warn(
                "SemiParametricRegressionModel.neg_ll(beta) is deprecated "
                "and will be removed in v{}: neg_ll() is now the fitted "
                "value; use neg_ll_of(beta) for the negative partial "
                "log-likelihood at beta.".format(REMOVED_IN_NEXT),
                DeprecationWarning,
                stacklevel=2,
            )
            if self.neg_ll_of is None:
                raise ValueError(
                    "The partial likelihood is not stored with a model "
                    "restored from a dict; refit it to evaluate it at "
                    "other coefficients."
                )
            return float(self.neg_ll_of(beta))
        if getattr(self, "_neg_ll", None) is None:
            raise ValueError("Must have been fit with data")
        return float(self._neg_ll)

    def _ic_k(self) -> int:
        # The estimated coefficients: an aliased one (nan) is not, as R's
        # logLik(coxph) counts sum(!is.na(coef)).
        return int(onp.isfinite(onp.asarray(self.params, float)).sum())

    def _ic_sample_size_from_data(self) -> float:
        # A model restored from a dict saved without "ic_n": the events,
        # which the baseline counts at each time (R's ``nevent``).
        return float(onp.sum(self.d))

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

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        if getattr(self, "is_stratified", False):
            raise ValueError(
                "concordance is not available for a stratified Cox model: "
                "its strata have separate baselines, so risk scores rank "
                "subjects only within a stratum. Score each stratum's rows "
                "with surpyval.metrics.concordance_index."
            )
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        return self._log_risk(Z_arr.reshape(x.size, -1))

    def _concordance_data(self) -> "tuple | None":
        data = getattr(self, "_fit_data", None)
        if data is None or self.is_tvc:
            return None
        return data["x"], data["c"], data["n"], data["Z"]

    def _data_repr(self) -> str:
        """The data the model was fitted to, in one line, for the printout
        (#508): units weighted by ``n``, by kind of censoring and left
        truncation. A restored model gives the line it was saved with."""
        data = getattr(self, "_fit_data", None)
        if not isinstance(data, dict) or "c" not in data:
            return getattr(self, "_data_summary", None) or ""
        ids = getattr(self, "tvc_subject_ids", None)
        if getattr(self, "is_tvc", False) and ids is not None:
            # Start-stop rows are intervals of a unit, not units.
            c = np.asarray(data["c"])
            n = np.asarray(data.get("n", np.ones(len(c))))
            k = int(np.sum(n[c == 0]))
            units = len(np.unique(np.asarray(ids)))
            return "{} unit{} in {} start-stop intervals: {} event{}".format(
                units,
                "" if units == 1 else "s",
                len(c),
                k,
                "" if k == 1 else "s",
            )
        tl = data.get("tl")
        x = data.get("x")
        if tl is None:
            return data_summary(data["c"], data.get("n"), x=x)
        # Times are non-negative, where an entry at 0 truncates nothing.
        return data_summary(data["c"], data.get("n"), tl=tl, lower=0.0, x=x)

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Hazards"
            + "\nKind                : {kind}"
            + "\nParameterization    : {parameterization}"
        ).format(kind=self.kind, parameterization=self.parameterization)
        data_line = self._data_repr()
        if data_line:
            out += "\nData                : " + data_line
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
        se = self._se
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
        optimiser objects (the ``neg_ll_of`` closure, ``jac``, ``hess``,
        ``res``) are not stored; the fitted ``neg_ll()`` and the sample
        size of ``bic()`` (``"ic_n"``) are.

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
        if self._se is not None:
            out["se"] = np.asarray(self._se, dtype=float).tolist()
        if self._covariance is not None:
            # The key every model's dict stores it under (#605).
            out["covariance"] = np.asarray(
                self._covariance, dtype=float
            ).tolist()
        if getattr(self, "_neg_ll", None) is not None:
            # The key every model's dict stores it under (#605).
            out["_neg_ll"] = float(self._neg_ll)
        ic_n = self._ic_sample_size_or_none()
        if ic_n is not None:
            out["ic_n"] = ic_n
        out.update(maximum_entry(self.maximum))
        # The printout's "Data" line (#508), so the restored model prints
        # the same; the data themselves are not stored.
        if self._data_repr():
            out["data_summary"] = self._data_repr()
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
        # Plain numpy: autograd's ``array`` inspects a list item by item
        # for its boxes, 90% of reading a Cox model of 1e5 rows.
        beta = onp.array(model_dict["beta"], dtype=float)
        out.beta = beta
        out.params = onp.array(model_dict["params"], dtype=float)
        # A dict without a "center" (the default fit, or one written
        # before #459) has its baseline at Z = 0.
        out.center = onp.array(
            model_dict.get("center", np.zeros(beta.shape[0])), dtype=float
        )
        out.x = onp.array(model_dict["x"], dtype=float)
        out.r = onp.array(model_dict["r"], dtype=float)
        out.d = onp.array(model_dict["d"], dtype=float)
        out.h0 = onp.array(model_dict["h0"], dtype=float)
        out.H0 = onp.array(model_dict["H0"], dtype=float)
        # phi is fully determined by beta and center (the ``phi`` method).
        out.tie_method = model_dict["tie_method"]
        out.baseline_method = model_dict["baseline_method"]
        out.is_tvc = bool(model_dict.get("is_tvc", False))
        out._data_summary = model_dict.get("data_summary")
        # A bare null (no "non_finite" record) is how schema-1 dicts wrote
        # a row without delayed entry; it still reads as -inf.
        out.tl = (
            onp.array(
                [-np.inf if v is None else v for v in model_dict["tl"]],
                dtype=float,
            )
            if model_dict.get("tl") is not None
            else None
        )
        if "p_values" in model_dict:
            out.p_values = onp.array(model_dict["p_values"], dtype=float)
        if "se" in model_dict:
            out._se = onp.array(model_dict["se"], dtype=float)
        # A dict written before v0.23 has the standard errors only.
        if "covariance" in model_dict:
            out._covariance = onp.array(model_dict["covariance"], dtype=float)
        # "_neg_log_like" is the key of a dict written before v0.23.
        for key in ("_neg_ll", "_neg_log_like"):
            if key in model_dict:
                out._neg_ll = float(model_dict[key])
                break
        out._ic_n = cls._restored_ic_n(model_dict)
        out.maximum = restored_maximum(model_dict)
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

    def check_ph(self, transform: str = "km") -> "pd.DataFrame":
        """
        Test the proportional-hazards assumption (Grambsch-Therneau).

        The table R's ``cox.zph`` prints (#514): one row per covariate
        (named as in :meth:`summary`), each a 1-d.f. test of whether its
        scaled Schoenfeld residuals trend with time, and a last row
        ``GLOBAL``, the joint test on all of them. A small ``p`` is
        evidence against proportional hazards.

        Parameters
        ----------
        transform : {"km", "rank", "identity", "log"}, optional
            The function of time to test against; ``"km"`` (the default,
            as in R) is the scale-free choice. See
            :func:`~surpyval.univariate.regression.proportional_hazards.
            diagnostics.check_ph`.

        Returns
        -------
        pandas.DataFrame
            Columns ``statistic`` (chi-squared), ``df`` and ``p``; the
            transform is in ``attrs["transform"]``. Before v0.22 this was
            a dict, which :func:`~surpyval.univariate.regression.
            proportional_hazards.diagnostics.check_ph` (the same test as a
            function of the model) still returns.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]  # arrest is 1 for an arrest
        >>> model = CoxPH.fit_from_df(
        ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
        ... )
        >>> model.check_ph().round(4)
                   statistic  df       p
        covariate
        fin           0.0000   1  0.9983
        age           5.8491   1  0.0156
        GLOBAL        5.8570   2  0.0535
        """
        import pandas as pd

        from .proportional_hazards.diagnostics import check_ph

        res = check_ph(self, transform)
        rows = res["per_covariate"] + [res["global"]]
        names = coefficient_names(self, len(res["per_covariate"]))
        table = pd.DataFrame(
            {
                "statistic": [r["statistic"] for r in rows],
                "df": [int(r["df"]) for r in rows],
                "p": [r["p_value"] for r in rows],
            },
            index=pd.Index(list(names) + ["GLOBAL"], name="covariate"),
        )
        table.attrs["transform"] = res["transform"]
        return table

    def covariance(self) -> npt.NDArray:
        """
        The coefficients' covariance: the inverse of the observed
        information of the partial likelihood at the fit, R's
        ``vcov(coxph)`` (#613). An aliased coefficient's row and column
        are ``nan``. :meth:`robust_covariance` gives the sandwich.

        Raises a ``ValueError`` for a model saved before the covariance
        was stored, whose :meth:`standard_errors` are still available.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = CoxPH.fit(x, df[["fin", "age"]].values, c=c)
        >>> model.covariance().round(6)
        array([[ 0.036044, -0.000149],
               [-0.000149,  0.000436]])
        """
        if self._covariance is None:
            raise no_covariance_error(
                "it was saved before the covariance was stored; "
                "standard_errors() still gives the standard errors"
            )
        return self._covariance

    def standard_errors(self) -> npt.NDArray:
        """
        The coefficients' standard errors, the square roots of the diagonal
        of :meth:`covariance` in the order of ``params`` (``nan`` for an
        aliased coefficient), R's ``se(coef)``. ``se``, the attribute
        before v0.23, still gives them, with a ``DeprecationWarning``,
        until v0.24.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = CoxPH.fit(x, df[["fin", "age"]].values, c=c)
        >>> model.standard_errors().round(4)
        array([0.1899, 0.0209])
        """
        if self._covariance is not None:
            return standard_errors_of(self._covariance)
        if self._se is not None:
            # A model saved before the covariance was stored
            return onp.asarray(self._se, dtype=float)
        return standard_errors_of(self.covariance())

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
            ``S(x | given) = exp(-(H(x) - H(given)))`` for ``x > given``,
            and 1 for ``x <= given`` (survival to those times is certain).
            Along a ``CovariatePath`` the baseline jumps after ``given``
            are summed, so nothing is subtracted. A ``nan`` ``given`` gives
            ``nan``.
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

        g = None if given is None else float(given)
        if isinstance(Z, CovariatePath) and g is not None and not np.isnan(g):
            # Summed from given on.
            H = self._hf_tvc(x, Z, xl, stratum, given=g)
        else:
            H = self._hf_tvc(x, Z, xl, stratum)
            if g is not None:
                if np.isnan(g):
                    # A missing conditioning age: nothing is known.
                    H = np.full(np.shape(H), np.nan)
                else:
                    H = H - self._hf_tvc(g, Z, xl, stratum)
        if g is not None and not np.isnan(g):
            # Given survival to g, survival to any x <= g is certain: the
            # difference H(x) - H(g) is not a cumulative hazard there, and
            # gave a "survival" above 1 (#523).
            xq = np.atleast_1d(np.asarray(x, dtype=float))
            H = np.where(xq <= g, 0.0, H)
        return np.exp(-H)
