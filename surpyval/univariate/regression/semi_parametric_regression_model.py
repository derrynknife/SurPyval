from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt

from surpyval.utils import _get_idx

from .regression_data import prepare_Z

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
