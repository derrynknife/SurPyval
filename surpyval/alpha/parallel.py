from autograd import elementwise_grad

from surpyval import Distribution, np
from surpyval.alpha.series import SeriesModel


class ParallelModel(Distribution):
    def __or__(self, other):
        if isinstance(other, ParallelModel):
            return ParallelModel([*self.models, other])
        elif isinstance(other, SeriesModel):
            return SeriesModel([*self.models, *other.models])
        # Leaf model (Parametric, NonParametric, MixtureModel, ...)
        return SeriesModel([*self.models, other])

    def __and__(self, other):
        if isinstance(other, ParallelModel):
            return ParallelModel([*self.models, *other.models])
        # SeriesModel and leaf models alike nest as a single component.
        return ParallelModel([*self.models, other])

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
