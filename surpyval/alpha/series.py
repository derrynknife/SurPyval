from autograd import elementwise_grad

from surpyval import Distribution, np


class SeriesModel(Distribution):
    def __or__(self, other):
        from surpyval.alpha.parallel import ParallelModel

        # Composite operands contribute their components; any other
        # Distribution (Parametric, NonParametric, MixtureModel, ...) is
        # a leaf and is appended whole.
        if isinstance(other, (SeriesModel, ParallelModel)):
            return SeriesModel([*self.models, *other.models])
        return SeriesModel([*self.models, other])

    def __and__(self, other):
        from surpyval.alpha.parallel import ParallelModel

        if isinstance(other, (SeriesModel, ParallelModel)):
            return ParallelModel([*self.models, *other.models])
        return ParallelModel([*self.models, other])

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
