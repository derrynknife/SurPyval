import surpyval
from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter

from ..parametric import Parametric


class ExactEventTime_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=1,
            bounds=((None, None),),
            support=(-np.inf, np.inf),
            param_names=["T"],
            param_map={"T": 0},
            plot_x_scale="linear",
        )

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

    def Hf(self, x, T):
        return self.hf(x, T)

    def random(self, size, T):
        return np.ones(size) * T

    def fit(self, x, c=None, n=None, t=None):
        x, c, n, t = surpyval.xcnt_handler(x=x, c=c, n=n, t=t)

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

        model = Parametric(self, "MLE", None, False, False, False)
        model.params = np.array([T])
        return model

    def from_params(self, T):
        model = Parametric(self, "from_params", None, False, False, False)
        model.params = np.array([T])
        return model


ExactEventTime = ExactEventTime_("ExactEventTime")
