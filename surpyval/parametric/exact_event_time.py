import surpyval
from surpyval import np

from .parametric import Parametric


class ExactEventTime_:
    def __init__(self, name):
        super().__init__(name)
        # Set 'k', the number of parameters
        self.k = 1
        self.bounds = ((-np.inf, np.inf),)
        self.support = (-np.inf, np.inf)
        self.param_names = ["T"]
        self.param_map = {
            "T": 0,
        }

    def sf(self, x, T):
        x = np.atleast_1d(x)
        return (x < T).astype(float)

    def ff(self, x, T):
        x = np.atleast_1d(x)
        return (x >= T).astype(float)

    def df(self, x, T):
        x = np.atleast_1d(x)
        df = np.zeros_like(x).astype(float)
        df[x == T] = np.inf
        return df

    def hf(self, x, T):
        x = np.atleast_1d(x)
        hf = np.zeros_like(x).astype(float)
        hf[x >= T] = np.inf
        return hf

    def Hf(self, x):
        return self.hf(x)

    def random(self, size, T):
        return np.ones(size) * T

    def fit(self, x, c=None, n=None):
        x, c, n = surpyval.xcn_handler(x=x, c=c, n=n)

        if 0 in c:
            raise ValueError(
                "Fully observed observations in the data (c == 0). If you \
                have this data you know the failure time. Use `from_params` \
                method instead"
            )

        if 2 in c:
            raise NotImplementedError(
                "Exact failure time estimation not implemented for interval \
                censored data"
            )

        max_r = np.max(x[c == 1])
        min_l = np.min(x[c == -1])

        T = (max_r + min_l) / 2.0

        model = Parametric(self, "MLE", {}, False, False, False)
        model.params = [T]
        return model

    def from_params(self, T):
        model = Parametric(self, "from_params", {}, False, False, False)
        model.params = [T]
        return model


ExactEventTime = ExactEventTime_("ExactEventTime")
