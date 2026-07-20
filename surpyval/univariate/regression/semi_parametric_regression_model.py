import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt

from surpyval.utils import _get_idx

from .regression_data import prepare_Z
from surpyval.serialisation import stamp_schema

if TYPE_CHECKING:
    import pandas as pd


class SemiParametricRegressionModel:
    # Covariate metadata populated when the model is fit from a pandas
    # DataFrame via ``CoxPH.fit_from_df``.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    #: True when fitted from time-varying-covariate (start-stop) data via
    #: ``CoxPH.fit_tvc``; enables :meth:`predict_tvc`.
    is_tvc: bool = False

    # Attributes populated by the fitter (``CoxPH.fit`` / ``fit_from_df``).
    params: npt.NDArray
    beta: npt.NDArray
    x: npt.NDArray
    r: npt.NDArray
    d: npt.NDArray
    tl: Any
    h0: npt.NDArray
    H0: npt.NDArray
    phi: Callable[..., npt.NDArray]
    p_values: npt.NDArray
    jac: npt.NDArray
    hess: npt.NDArray
    res: Any
    neg_ll: float
    _neg_log_like: float
    tie_method: str
    baseline_method: str
    #: Per-observation training data (``x``/``c``/``n``/``Z``/``tl``) retained
    #: by ``CoxPH.fit`` for residuals and the proportional-hazards test.
    _fit_data: dict

    def __init__(self, kind: str, parameterization: str) -> None:
        self.kind = kind
        self.parameterization = parameterization

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        Convert ``Z`` to a numeric design matrix, selecting the covariate
        columns recorded at fit time when a pandas DataFrame is passed.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Hazards"
            + "\nKind                : {kind}"
            + "\nParameterization    : {parameterization}"
        ).format(kind=self.kind, parameterization=self.parameterization)

        out = out + "\nParameters          :\n"
        for i, p in enumerate(self.params):
            out += "   beta_{i}  :  {p}\n".format(i=i, p=p)
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Cox model to a plain, JSON-serialisable dict.

        The Cox model is a proportional-hazards fit with a *nonparametric*
        baseline, so what is stored is the covariate coefficients ``beta`` and
        the fitted baseline step arrays (event times ``x`` and the baseline
        hazard ``h0`` / cumulative hazard ``H0``); the hazard multiplier
        ``phi(Z) = exp(beta'Z)`` is rebuilt from ``beta`` on load. Everything
        needed for ``hf``/``Hf``/``sf``/``ff``/``df`` (and, for a
        time-varying-covariate fit, ``predict_tvc``) round-trips exactly. The
        optimiser objects (the ``neg_ll`` closure, ``jac``, ``hess``, ``res``)
        are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
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
        if getattr(self, "tl", None) is not None:
            out["tl"] = np.asarray(self.tl, dtype=float).tolist()
        if getattr(self, "p_values", None) is not None:
            out["p_values"] = np.asarray(self.p_values, dtype=float).tolist()
        if getattr(self, "_neg_log_like", None) is not None:
            out["_neg_log_like"] = float(self._neg_log_like)
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
    def from_dict(cls, model_dict: dict) -> "SemiParametricRegressionModel":
        """
        Rebuild a Cox model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "SemiParametricRegressionModel":
            raise ValueError(
                "Must create a Cox model from a "
                "SemiParametricRegressionModel dict"
            )
        out = cls(model_dict["kind"], model_dict["parameterization"])
        beta = np.array(model_dict["beta"], dtype=float)
        out.beta = beta
        out.params = np.array(model_dict["params"], dtype=float)
        out.x = np.array(model_dict["x"], dtype=float)
        out.r = np.array(model_dict["r"], dtype=float)
        out.d = np.array(model_dict["d"], dtype=float)
        out.h0 = np.array(model_dict["h0"], dtype=float)
        out.H0 = np.array(model_dict["H0"], dtype=float)
        # phi is fully determined by beta
        out.phi = lambda Z: np.exp(np.asarray(Z, dtype=float) @ out.beta)
        out.tie_method = model_dict["tie_method"]
        out.baseline_method = model_dict["baseline_method"]
        out.is_tvc = bool(model_dict.get("is_tvc", False))
        out.tl = (
            np.array(model_dict["tl"], dtype=float)
            if "tl" in model_dict
            else None
        )
        if "p_values" in model_dict:
            out.p_values = np.array(model_dict["p_values"], dtype=float)
        if "_neg_log_like" in model_dict:
            out._neg_log_like = float(model_dict["_neg_log_like"])
        out.feature_names = model_dict.get("feature_names")
        out.formula = model_dict.get("formula")
        return out

    @classmethod
    def from_json(cls, fp: "str | Path") -> "SemiParametricRegressionModel":
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        Z = self._prepare_Z(Z)
        idx, rev = _get_idx(self.x, x)
        return (self.h0[idx] * self.phi(Z))[rev]

    def Hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        Z = self._prepare_Z(Z)
        idx, rev = _get_idx(self.x, x)
        return (self.H0[idx] * self.phi(Z))[rev]

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

    def robust_covariance(self, cluster=None) -> npt.NDArray:
        """
        Cluster-robust ("sandwich") covariance of the coefficients.

        Pass ``cluster`` (one label per observation) for correlated /
        clustered data; ``None`` gives the ordinary robust variance. See
        :func:`~surpyval.univariate.regression.proportional_hazards.
        diagnostics.robust_covariance`.
        """
        from .proportional_hazards.diagnostics import robust_covariance

        return robust_covariance(self, cluster)

    def robust_summary(self, cluster=None) -> dict:
        """
        Cluster-robust standard errors, z-scores and p-values for the
        coefficients. See :func:`~surpyval.univariate.regression.
        proportional_hazards.diagnostics.robust_summary`.
        """
        from .proportional_hazards.diagnostics import robust_summary

        return robust_summary(self, cluster)

    def predict_tvc(
        self,
        start: npt.ArrayLike,
        stop: npt.ArrayLike,
        Z: npt.ArrayLike,
        times: "npt.ArrayLike | None" = None,
    ) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray]":
        r"""
        Survival for a subject whose covariates vary over time.

        With a time-varying covariate the survival function depends on the
        whole covariate path, not a single vector:

        .. math::
            H(t \mid Z(\cdot)) = \int_0^t e^{\beta' Z(u)}\, dH_0(u)
            = \sum_{u_j \le t} h_0(u_j)\, e^{\beta' Z(u_j)},

        summing the fitted baseline-hazard jumps ``h0`` weighted by the hazard
        multiplier of the covariate value *active* at each jump time. With a
        single constant interval this reduces exactly to ``sf(t, Z)``.

        Parameters
        ----------
        start, stop : array_like
            The subject's covariate-path intervals ``(start, stop]``, one per
            row (as given to :meth:`~...CoxPH.fit_tvc`). Usually contiguous
            from ``0``.
        Z : array_like
            The covariate row active on each interval, one row per interval.
        times : array_like, optional
            Times at which to return survival. Defaults to the fitted baseline
            jump times that fall within the covariate path.

        Returns
        -------
        times, sf, Hf : ndarray
            The evaluation times and the survival and cumulative-hazard values
            of the subject along its covariate path. Outside the supplied path
            the nearest interval's covariate is held constant.
        """
        start_a = np.atleast_1d(np.asarray(start, dtype=float))
        stop_a = np.atleast_1d(np.asarray(stop, dtype=float))
        Z_a = np.asarray(Z, dtype=float)
        if Z_a.ndim == 1:
            Z_a = Z_a.reshape(-1, 1)
        if not (start_a.shape[0] == stop_a.shape[0] == Z_a.shape[0]):
            raise ValueError(
                "start, stop and Z must have the same number of rows"
            )
        if np.any(start_a >= stop_a):
            raise ValueError("every interval must have start < stop")

        order = np.argsort(start_a)
        start_a, stop_a, Z_a = start_a[order], stop_a[order], Z_a[order]

        # The active interval at a baseline jump time u is the last interval
        # whose start is at or before u; times outside the path are clamped to
        # the first/last interval (covariate held constant).
        base_t = self.x
        active = np.searchsorted(start_a, base_t, side="right") - 1
        active = np.clip(active, 0, start_a.shape[0] - 1)
        phi = np.exp(Z_a[active] @ self.beta)
        H_cum = np.cumsum(self.h0 * phi)

        if times is None:
            within = (base_t > start_a[0]) & (base_t <= stop_a[-1])
            query = base_t[within]
        else:
            query = np.atleast_1d(np.asarray(times, dtype=float))

        idx = np.searchsorted(base_t, query, side="right") - 1
        last = H_cum.shape[0] - 1
        Hf = np.where(idx >= 0, H_cum[np.clip(idx, 0, last)], 0.0)
        return query, np.exp(-Hf), Hf
