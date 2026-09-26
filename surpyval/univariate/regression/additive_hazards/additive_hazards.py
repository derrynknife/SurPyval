"""
Lin & Ying (1994) semi-parametric additive hazards model.

The model puts the covariates on the *hazard* scale additively, rather than
multiplicatively as Cox's proportional hazards does:

.. math::
    \\lambda(t \\mid Z) = \\lambda_0(t) + \\beta' Z

so ``beta_j`` is the change in the absolute hazard (a risk *difference*) per
unit of covariate ``j``, constant over time. The baseline hazard
``lambda_0(t)`` is left completely unspecified (semi-parametric).

Unlike Cox PH, the coefficient estimator is **closed form** -- no iterative
optimisation and no convergence concerns. Writing ``Ybar(t)`` for the
covariate mean over the risk set at time ``t``, Lin & Ying's estimating
equation solves to

.. math::
    \\hat\\beta = A^{-1} b, \\quad
    A = \\sum_i \\int_0^\\tau Y_i(t)\\,\\{Z_i - \\bar Z(t)\\}^{\\otimes 2}
    \\,dt, \\quad
    b = \\sum_i \\int_0^\\tau \\{Z_i - \\bar Z(t)\\}\\,dN_i(t)

where ``Y_i`` is the at-risk indicator and ``N_i`` the event counting
process. Both ``A`` (a time integral over the risk sets) and ``b`` (a sum
over the event times) are accumulated in closed form, and the variance comes
from the Lin & Ying sandwich ``A^{-1} B A^{-1}`` with
``B = sum_i int {Z_i - Zbar}^{2} dN_i``.

A caveat inherent to the model (not this implementation): an additive hazard
can go negative when ``beta'Z`` is sufficiently negative, so the fitted
cumulative hazard need not be monotone and the implied survival can exceed 1.
This is a known property of additive-hazards models; the raw estimates are
returned without clamping.

Reference
---------
Lin, D. Y. and Ying, Z. (1994), "Semiparametric analysis of the additive
risk model", Biometrika 81, 61-71.
"""

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils import (
    check_covariate_rows,
    finite_covariate_mask,
    xcnt_handler,
)
from surpyval.utils.linalg import safe_inv

from ..regression_data import (
    design_matrix_from_df,
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    x_h, c_h, n_h, _ = xcnt_handler(x, c, n, group_and_sort=False)
    c_arr = np.asarray(c_h, dtype=float)
    if not np.all((c_arr == 0) | (c_arr == 1)):
        raise ValueError(
            "The additive hazards model supports only observed (c=0) and "
            "right-censored (c=1) data."
        )
    x_arr = np.asarray(x_h, dtype=float)
    if x_arr.ndim == 2:
        # Two columns with no interval row: xl == xr on every row.
        x_arr = x_arr[:, 0]
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    elif Z_arr.ndim != 2:
        raise ValueError("Covariate matrix must be two dimensional")
    check_covariate_rows(Z_arr, x_arr.shape[0])
    # Rows with a NaN / infinite covariate are dropped with a warning, as in
    # every regression fitter (this one used to drop NaN rows silently).
    mask = finite_covariate_mask(Z_arr)
    x_arr, c_arr, Z_arr = x_arr[mask], c_arr[mask], Z_arr[mask]
    n_arr = np.asarray(n_h, dtype=float)[mask]
    if np.any(x_arr < 0):
        # The estimating equations integrate over the risk sets from time
        # 0; a negative time fed a negative width into that integral.
        raise ValueError(
            "The additive hazards model integrates the hazard from time 0; "
            "all times must be non-negative."
        )
    if not np.any(c_arr == 0):
        raise ValueError(
            "The additive hazards model needs at least one event (c=0); "
            "with every observation censored the coefficients are not "
            "estimable."
        )
    return x_arr, c_arr, n_arr, Z_arr


def _check_estimable(A: npt.NDArray, scale: npt.NDArray) -> None:
    """Refuse a design whose coefficients the data cannot determine.

    ``A`` is the integrated risk-set covariate scatter and ``scale`` the
    integrated raw second moment of each covariate, the yardstick for "no
    spread" (a constant covariate leaves only rounding in ``A``). A
    constant covariate, a single observation or collinear covariates make
    ``A`` singular, and the pseudo-inverse then returned ``beta = 0``
    without a word.
    """
    d = np.diag(A)
    flat = d <= 1e-10 * np.maximum(scale, np.finfo(float).tiny)
    if np.any(flat):
        raise ValueError(
            "Covariate(s) {} do not vary within the risk sets (a constant "
            "covariate, or too few observations), so the additive hazards "
            "coefficients cannot be estimated.".format(
                np.flatnonzero(flat).tolist()
            )
        )
    corr = A / np.sqrt(np.outer(d, d))
    if np.linalg.matrix_rank(corr, tol=1e-10) < A.shape[0]:
        raise ValueError(
            "The covariates are collinear within the risk sets, so the "
            "additive hazards coefficients cannot be estimated; drop the "
            "redundant covariate(s)."
        )


class AdditiveHazardsModel(SerialisableMixin):
    """
    A fitted Lin & Ying additive hazards model, returned by
    :meth:`AdditiveHazards.fit`.

    The covariate effect is additive on the hazard, so the prediction
    methods use ``h(t | Z) = h0(t) + beta'Z`` and the cumulative
    ``H(t | Z) = H0(t) + t * beta'Z``.
    """

    # Populated by ``fit`` / ``fit_from_df``.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: object = None

    # Fitted quantities set by ``AdditiveHazards.fit``.
    beta: npt.NDArray
    params: npt.NDArray
    cov: npt.NDArray
    se: npt.NDArray
    p_values: npt.NDArray
    x: npt.NDArray
    h0: npt.NDArray
    H0: npt.NDArray
    #: ``beta'Zbar(t)`` on each interval ``(x[j-1], x[j]]`` of the grid (the
    #: last value is held beyond it). ``None`` on a model restored from a
    #: dict written before it was stored; ``Hf`` then reads ``H0`` as a step.
    drift: "npt.NDArray | None" = None
    _A: npt.NDArray
    _b: npt.NDArray

    def __init__(self) -> None:
        self.kind = "Additive Hazards"
        self.parameterization = "Semi-Parametric"

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        Z = prepare_Z(Z, self.feature_names, self._model_spec)
        return np.atleast_2d(Z)

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Additive Hazards"
            + "\nKind                : Lin-Ying"
            + "\nParameterization    : Semi-Parametric"
            + "\nParameters          :\n"
        )
        for i, p in enumerate(self.beta):
            out += "   beta_{i}  :  {p}\n".format(i=i, p=p)
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Lin-Ying additive-hazards model to a plain,
        JSON-serialisable dict.

        The baseline is nonparametric, so what is stored is the coefficients
        ``beta`` and the fitted baseline step arrays (event times ``x`` and the
        baseline hazard ``h0`` / cumulative hazard ``H0``), along with the
        parameter covariance so the restored model can still report standard
        errors. Everything needed for ``hf``/``Hf``/``sf``/``ff``/``df`` (and
        ``se``/``cov``) round-trips exactly. The internal estimating-equation
        matrices (``_A``, ``_b``) are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict[str, Any] = {
            "model": "AdditiveHazardsModel",
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "x": np.asarray(self.x, dtype=float).tolist(),
            "h0": np.asarray(self.h0, dtype=float).tolist(),
            "H0": np.asarray(self.H0, dtype=float).tolist(),
            "cov": np.asarray(self.cov, dtype=float).tolist(),
            "se": np.asarray(self.se, dtype=float).tolist(),
        }
        if getattr(self, "p_values", None) is not None:
            out["p_values"] = np.asarray(self.p_values, dtype=float).tolist()
        if self.drift is not None:
            out["drift"] = np.asarray(self.drift, dtype=float).tolist()
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "AdditiveHazardsModel":
        """
        Rebuild a Lin-Ying additive-hazards model from a :meth:`to_dict`
        dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "AdditiveHazardsModel", "an additive-hazards model"
        )
        out = cls()
        out.beta = np.array(model_dict["beta"], dtype=float)
        out.params = np.array(model_dict["params"], dtype=float)
        out.x = np.array(model_dict["x"], dtype=float)
        out.h0 = np.array(model_dict["h0"], dtype=float)
        out.H0 = np.array(model_dict["H0"], dtype=float)
        out.cov = np.array(model_dict["cov"], dtype=float)
        out.se = np.array(model_dict["se"], dtype=float)
        if "p_values" in model_dict:
            out.p_values = np.array(model_dict["p_values"], dtype=float)
        if "drift" in model_dict:
            out.drift = np.array(model_dict["drift"], dtype=float)
        restore_covariate_meta(out, model_dict)
        return out

    def _h0_at(self, x: npt.NDArray) -> npt.NDArray:
        # Right-continuous step lookup of the baseline (cumulative) hazard at
        # each time in ``x``; zero before the first event time.
        idx = np.searchsorted(self.x, x, side="right") - 1
        return idx

    def _h0_rate(
        self, x: npt.NDArray, bandwidth: "float | None" = None
    ) -> npt.NDArray:
        # Kernel-smoothed (Ramlau-Hansen) baseline hazard *rate* from the
        # increments of the corrected baseline cumulative hazard H0 (the
        # raw self.h0 = d/S0 jumps include the covariate-mean drift and
        # are dimensionless besides -- adding such a jump to the rate
        # beta'Z was dimensionally incoherent and asymptotically dropped
        # the baseline from hf/df entirely, #277).
        if bandwidth is None:
            spread = float(np.std(self.x)) if self.x.size > 1 else 0.0
            scale = max(abs(float(np.mean(self.x))), 1.0)
            if spread <= 1e-8 * scale:
                # All event times (nearly) coincident relative to the time
                # scale: the normal-reference rule collapses to the floor
                # and hf returns Dirac spikes; fall back to a bandwidth on
                # the scale of the times themselves (#289).
                spread = scale
            bandwidth = max(
                1.06 * spread * max(self.x.size, 2) ** (-1 / 5), 1e-12
            )
        dH0 = np.diff(np.concatenate([[0.0], self.H0]))
        u = (x[:, None] - self.x[None, :]) / bandwidth
        kern = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)
        return (kern * dH0[None, :]).sum(axis=1) / bandwidth

    def hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        bandwidth: "float | None" = None,
    ) -> npt.NDArray:
        """
        Hazard rate ``h0(t) + beta'Z`` with a kernel-smoothed baseline.

        The semiparametric baseline is a step cumulative hazard, so the
        rate requires smoothing (Epanechnikov kernel over the increments;
        ``bandwidth`` defaults to a normal-reference rule on the event
        times). Estimates near the boundaries of the observed time range
        are attenuated by kernel truncation.
        """
        Z = self._prepare_Z(Z)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._h0_rate(x, bandwidth) + (Z @ self.beta)

    def Hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        """
        Cumulative hazard ``H0(x) + x * beta'Z`` at ``x`` for covariates
        ``Z`` (one row, or one row per ``x``). The baseline ``H0`` jumps
        by ``d / S0`` at each event time and, between the grid times,
        falls continuously by the covariate-mean drift
        ``beta' Zbar(t)`` (so ``H0`` is not 0 before the first event
        unless the covariates are centred there). The prediction
        ``H(x | Z)`` is the same however the covariates are centred.
        """
        Z = self._prepare_Z(Z)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = self._h0_at(x)
        last = self.x.size - 1
        H0 = np.where(idx < 0, 0.0, self.H0[np.clip(idx, 0, last)])
        if self.drift is not None:
            # The drift accrued since the last grid time at or before x, at
            # the rate of the interval x lies in (the last one beyond the
            # grid).
            since = x - np.where(idx < 0, 0.0, self.x[np.clip(idx, 0, last)])
            H0 = H0 - since * self.drift[np.clip(idx + 1, 0, last)]
        # H(t | Z) = H0(t) + integral_0^t beta'Z ds = H0(t) + t * beta'Z.
        return H0 + x * (Z @ self.beta)

    def sf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        """Survival ``exp(-Hf(x, Z))``."""
        return np.exp(-self.Hf(x, Z))

    def ff(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        """Failure probability ``1 - sf(x, Z)``."""
        return -np.expm1(-self.Hf(x, Z))

    def df(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        """Density ``hf(x, Z) * sf(x, Z)``, with the smoothed hazard."""
        return self.hf(x, Z) * self.sf(x, Z)

    def standard_errors(self) -> npt.NDArray:
        """Standard errors of the coefficients (Lin-Ying sandwich)."""
        return self.se

    def covariance(self) -> npt.NDArray:
        """Covariance matrix of the coefficients (Lin-Ying sandwich)."""
        return self.cov


class AdditiveHazards_:
    """
    The Lin & Ying semi-parametric additive hazards model: the covariates
    *add* a constant risk difference to a baseline hazard that is left to
    the data,

    .. math::
        h(x \\mid Z) = h_0(x) + \\beta' Z.

    The coefficients have a closed-form estimate (no iteration) with a
    sandwich variance. ``AdditiveHazards`` is an instance of this class;
    its ``fit`` returns an
    :class:`~surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazardsModel`.
    For a parametric baseline see the ``AH`` family.
    """

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
    ) -> AdditiveHazardsModel:
        """
        Fit the Lin & Ying additive hazards model.

        Parameters
        ----------

        x : array-like
            The observed event/censoring times.
        Z : array-like
            The covariate matrix (one row per observation). Rows with a
            missing or infinite covariate are dropped, with a warning; a
            covariate that does not vary within the risk sets (constant, or
            a single observation), collinear covariates, data with no
            event and negative times raise a ``ValueError``.
        c : array-like, optional
            Censoring flags: 0 observed (event), 1 right-censored. Defaults
            to all observed.
        n : array-like, optional
            Multiplicity (counts) for each row. Defaults to 1 each.

        Returns
        -------

        AdditiveHazardsModel
            The fitted model, carrying ``beta``, standard errors, the
            coefficient covariance, p-values, and the baseline hazard.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import AdditiveHazards
        >>> rng = np.random.default_rng(3)
        >>> Z = rng.binomial(1, 0.5, (300, 1)).astype(float)
        >>> x = rng.exponential(1 / (0.1 + 0.05 * Z[:, 0]))
        >>> c = (x > 15).astype(int)  # follow-up ends at 15
        >>> x = np.minimum(x, 15)
        >>> model = AdditiveHazards.fit(x, Z, c=c)
        >>> model.beta.round(4), model.se.round(4)
        (array([0.0291]), array([0.0162]))
        >>> model.sf([5, 10], [[1]]).round(4)
        array([0.4465, 0.2387])
        """
        x, c, n, Z = _validate(x, Z, c, n)
        p = Z.shape[1]

        # Group observations by their exit time. The risk-set sums S0/S1/S2
        # at each unique time are reverse cumulative sums of the per-time
        # totals (all subjects with exit time >= u_j are at risk at u_j).
        unique_x = np.unique(x)
        m = unique_x.size
        bucket = np.searchsorted(unique_x, x)

        nZ = n[:, None] * Z
        nZZ = n[:, None, None] * (Z[:, :, None] * Z[:, None, :])

        S0_at = np.bincount(bucket, weights=n, minlength=m)
        S1_at = np.zeros((m, p))
        np.add.at(S1_at, bucket, nZ)
        S2_at = np.zeros((m, p, p))
        np.add.at(S2_at, bucket, nZZ)

        S0 = np.cumsum(S0_at[::-1])[::-1]
        S1 = np.cumsum(S1_at[::-1], axis=0)[::-1]
        S2 = np.cumsum(S2_at[::-1], axis=0)[::-1]
        Zbar = S1 / S0[:, None]

        # A = integral over t of V(t) dt, with V(t) the risk-set covariate
        # scatter about its mean. V and the risk set are constant on each
        # interval (u_{j-1}, u_j], so the integral is a width-weighted sum
        # (lower limit 0).
        widths = np.diff(np.concatenate([[0.0], unique_x]))
        V = S2 - (S1[:, :, None] * S1[:, None, :]) / S0[:, None, None]
        A = (V * widths[:, None, None]).sum(axis=0)
        _check_estimable(A, np.einsum("jii,j->i", S2, widths))

        # b = sum over events of (Z_event - Zbar(t_event)); events aggregated
        # per unique time so ties share one Zbar.
        is_event = c == 0
        w_event = n * is_event
        d_at = np.bincount(bucket, weights=w_event, minlength=m)
        E1_at = np.zeros((m, p))
        np.add.at(E1_at, bucket, w_event[:, None] * Z)
        b = (E1_at - d_at[:, None] * Zbar).sum(axis=0)

        A_inv = safe_inv(A)
        beta = A_inv @ b

        # Lin-Ying sandwich variance: B = sum over events of the centered
        # outer product {Z_i - Zbar(t_i)}^2.
        Z_centered = Z - Zbar[bucket]
        B = np.einsum("i,ij,ik->jk", w_event, Z_centered, Z_centered)
        cov = A_inv @ B @ A_inv
        var = np.diag(cov)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(var)
            z_score = beta / se
            p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))

        # Baseline cumulative hazard on the event-time grid: the Breslow-type
        # step sum minus the accumulated covariate-mean drift.
        with np.errstate(invalid="ignore", divide="ignore"):
            dLambda = np.where(S0 > 0, d_at / S0, 0.0)
        Lambda = np.cumsum(dLambda)
        G = np.cumsum(Zbar * widths[:, None], axis=0)
        H0 = Lambda - G @ beta
        # The covariate-mean drift beta'Zbar(t) is a rate, constant on each
        # interval (u_{j-1}, u_j] of the risk-set grid; Hf integrates it
        # continuously between grid times. Reading H0 as a step (the drift
        # accrued only at the grid times) made predictions between event
        # times depend on how the covariates were centred.
        drift = Zbar @ beta

        model = AdditiveHazardsModel()
        model.beta = copy(beta)
        model.params = copy(beta)
        model.cov = cov
        model.se = se
        model.p_values = p_values
        model.x = unique_x
        model.h0 = dLambda
        model.H0 = H0
        model.drift = drift
        model._A = A
        model._b = b
        return model

    def fit_from_df(
        self,
        df: "pd.DataFrame",
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
    ) -> AdditiveHazardsModel:
        """
        Fit the additive hazards model from a pandas DataFrame, retaining the
        covariate names for prediction (see :meth:`fit` for the model).
        """
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values

        model = self.fit(x, Z, c=c, n=n)
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model


AdditiveHazards = AdditiveHazards_()
