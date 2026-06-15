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
