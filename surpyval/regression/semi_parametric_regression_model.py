import autograd.numpy as np

from ..utils import _get_idx


class SemiParametricRegressionModel:
    def __init__(self, kind, parameterization):
        self.kind = kind
        self.parameterization = parameterization

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
        idx, rev = _get_idx(self.x, x)
        return (self.h0[idx] * self.phi(Z))[rev]

    def Hf(self, x, Z):
        idx, rev = _get_idx(self.x, x)
        return (self.H0[idx] * self.phi(Z))[rev]

    def sf(self, x, Z):
        return np.exp(-self.Hf(x, Z))

    def ff(self, x, Z):
        return -np.expm1(-self.Hf(x, Z))

    def df(self, x, Z):
        return self.hf(x, Z) * self.sf(x, Z)
