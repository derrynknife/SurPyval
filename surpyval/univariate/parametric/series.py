from autograd import elementwise_grad

import surpyval as surv
from surpyval import np
from surpyval.univariate import parametric as p


class SeriesModel:
    def __or__(self, other):
        if type(other) == surv.Parametric:
            return SeriesModel([*self.models, other])
        else:
            return SeriesModel([*self.models, *other.models])

    def __and__(self, other):
        if type(other) == surv.Parametric:
            return p.ParallelModel([*self.models, other])
        else:
            return p.ParallelModel([*self.models, *other.models])

    def __init__(self, models):
        self.models = models
        self._df = elementwise_grad(self.ff)
        self._hf = elementwise_grad(self.Hf)

    def sf(self, x):
        x = np.atleast_1d(x)
        sf = np.vstack([np.log(D.sf(x)) for D in self.models])
        return np.exp(sf.sum(axis=0))

    def ff(self, x):
        x = np.atleast_1d(x)
        return 1 - self.sf(x)

    def df(self, x):
        x = np.atleast_1d(x)
        return self._df(x)

    def hf(self, x):
        x = np.atleast_1d(x)
        return self._hf(x)

    def Hf(self, x):
        x = np.atleast_1d(x)
        return -np.log(self.sf(x))
