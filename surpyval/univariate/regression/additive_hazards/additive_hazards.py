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

import json
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError, inv, pinv
from scipy.stats import norm

from surpyval.utils import check_Z_and_x, wrangle_Z, xcnt_handler

from ..regression_data import design_matrix_from_df, prepare_Z
from surpyval.serialisation import stamp_schema

if TYPE_CHECKING:
    import pandas as pd


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    x_h, c_h, n_h, _ = xcnt_handler(x, c, n, group_and_sort=False)
    Z_arr, mask = wrangle_Z(np.asarray(Z, dtype=float))
    x_arr = np.asarray(x_h, dtype=float)[mask]
    c_arr = np.asarray(c_h, dtype=float)[mask]
    n_arr = np.asarray(n_h, dtype=float)[mask]
    check_Z_and_x(Z_arr, x_arr)
    if not np.all((c_arr == 0) | (c_arr == 1)):
        raise ValueError(
            "The additive hazards model supports only observed (c=0) and "
            "right-censored (c=1) data."
        )
    return x_arr, c_arr, n_arr, Z_arr


class AdditiveHazardsModel:
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
    _A: npt.NDArray
    _b: npt.NDArray

    def __init__(self):
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
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        if self.formula is not None:
            out["formula"] = self.formula
        return stamp_schema(out)

    def to_json(self, fp: "str | Path") -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "AdditiveHazardsModel":
        """
        Rebuild a Lin-Ying additive-hazards model from a :meth:`to_dict`
        dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "AdditiveHazardsModel":
            raise ValueError(
                "Must create an additive-hazards model from an "
                "AdditiveHazardsModel dict"
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
        out.feature_names = model_dict.get("feature_names")
        out.formula = model_dict.get("formula")
        return out

    @classmethod
    def from_json(cls, fp: "str | Path") -> "AdditiveHazardsModel":
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def _h0_at(self, x: npt.NDArray) -> npt.NDArray:
        # Right-continuous step lookup of the baseline (cumulative) hazard at
        # each time in ``x``; zero before the first event time.
        idx = np.searchsorted(self.x, x, side="right") - 1
        return idx

    def hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        Z = self._prepare_Z(Z)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = self._h0_at(x)
        h0 = np.where(idx < 0, 0.0, self.h0[np.clip(idx, 0, self.x.size - 1)])
        return h0 + (Z @ self.beta)

    def Hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        Z = self._prepare_Z(Z)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = self._h0_at(x)
        H0 = np.where(idx < 0, 0.0, self.H0[np.clip(idx, 0, self.x.size - 1)])
        # H(t | Z) = H0(t) + integral_0^t beta'Z ds = H0(t) + t * beta'Z.
        return H0 + x * (Z @ self.beta)

    def sf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        return np.exp(-self.Hf(x, Z))

    def ff(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        return -np.expm1(-self.Hf(x, Z))

    def df(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        return self.hf(x, Z) * self.sf(x, Z)

    def standard_errors(self) -> npt.NDArray:
        """Standard errors of the coefficients (Lin-Ying sandwich)."""
        return self.se

    def covariance(self) -> npt.NDArray:
        """Covariance matrix of the coefficients (Lin-Ying sandwich)."""
        return self.cov


class AdditiveHazards_:
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
            The covariate matrix (one row per observation).
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

        # b = sum over events of (Z_event - Zbar(t_event)); events aggregated
        # per unique time so ties share one Zbar.
        is_event = c == 0
        w_event = n * is_event
        d_at = np.bincount(bucket, weights=w_event, minlength=m)
        E1_at = np.zeros((m, p))
        np.add.at(E1_at, bucket, w_event[:, None] * Z)
        b = (E1_at - d_at[:, None] * Zbar).sum(axis=0)

        try:
            A_inv = inv(A)
        except LinAlgError:
            A_inv = pinv(A)
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

        model = AdditiveHazardsModel()
        model.beta = copy(beta)
        model.params = copy(beta)
        model.cov = cov
        model.se = se
        model.p_values = p_values
        model.x = unique_x
        model.h0 = dLambda
        model.H0 = H0
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
