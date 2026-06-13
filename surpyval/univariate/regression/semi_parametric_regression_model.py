import autograd.numpy as np

from surpyval.utils import _get_idx

from .regression_data import prepare_Z


class SemiParametricRegressionModel:
    # Covariate metadata populated when the model is fit from a pandas
    # DataFrame via ``CoxPH.fit_from_df``.
    feature_names = None
    formula = None
    _model_spec = None

    def __init__(self, kind, parameterization):
        self.kind = kind
        self.parameterization = parameterization

    def _prepare_Z(self, Z):
        """
        Convert ``Z`` to a numeric design matrix, selecting the covariate
        columns recorded at fit time when a pandas DataFrame is passed.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    def __repr__(self):
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

    def hf(self, x, Z):
        Z = self._prepare_Z(Z)
        idx, rev = _get_idx(self.x, x)
        return (self.h0[idx] * self.phi(Z))[rev]

    def Hf(self, x, Z):
        Z = self._prepare_Z(Z)
        idx, rev = _get_idx(self.x, x)
        return (self.H0[idx] * self.phi(Z))[rev]

    def sf(self, x, Z):
        return np.exp(-self.Hf(x, Z))

    def ff(self, x, Z):
        return -np.expm1(-self.Hf(x, Z))

    def df(self, x, Z):
        return self.hf(x, Z) * self.sf(x, Z)
