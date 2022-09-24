from surpyval import np
from autograd import elementwise_grad
import surpyval as surv
from surpyval import parametric as p

class ParallelModel(object):
    def __or__(self, other):
        if type(other) == surv.Parametric:
            return p.SeriesModel([*self.models, other])
        elif type(other) == p.ParallelModel:
            return ParallelModel([*self.models, other])
        else:
            return p.SeriesModel([*self.models, *other.models])

    def __and__(self, other):
        if type(other) == surv.Parametric:
            return ParallelModel([*self.models, other])
        elif type(other) == p.SeriesModel:
            return ParallelModel([*self.models, other])
        else:
            return ParallelModel([*self.models, *other.models])

    def __init__(self, models):
        self.models = models
        self._df = elementwise_grad(self.ff)
        self._hf = elementwise_grad(self.Hf)

    def ff(self, x):
        x = np.atleast_1d(x)
        ff = np.vstack([np.log(D.ff(x)) for D in self.models])
        return np.exp(ff.sum(axis=0))

    def sf(self, x):
        x = np.atleast_1d(x)
        return 1 - self.ff(x)

    def df(self, x):
        x = np.atleast_1d(x)
        return self._df(x)

    def hf(self, x):
        x = np.atleast_1d(x)
        return self._hf(x)

    def Hf(self, x):
        x = np.atleast_1d(x)
        return -np.log(self.sf(x))