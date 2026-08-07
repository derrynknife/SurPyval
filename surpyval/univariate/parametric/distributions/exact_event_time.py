import numpy.typing as npt

import surpyval
from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    ParametricFitter,
    reject_structural_params,
)

from ..parametric import Parametric


class ExactEventTime_(ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=1,
            bounds=((None, None),),
            support=(-np.inf, np.inf),
            param_names=["T"],
            param_map={"T": 0},
            plot_x_scale="linear",
        )

    def sf(self, x: Numeric, T: Boxable) -> npt.NDArray:
        x_arr = np.atleast_1d(x)
        return (x_arr < T).astype(float)

    def ff(self, x: Numeric, T: Boxable) -> npt.NDArray:
        x_arr = np.atleast_1d(x)
        return (x_arr >= T).astype(float)

    def df(self, x: Numeric, T: Boxable) -> npt.NDArray:
        x_arr = np.atleast_1d(x)
        df = np.zeros_like(x_arr).astype(float)
        df[x_arr == T] = np.inf
        return df

    def hf(self, x: Numeric, T: Boxable) -> npt.NDArray:
        x_arr = np.atleast_1d(x)
        hf = np.zeros_like(x_arr).astype(float)
        hf[x_arr >= T] = np.inf
        return hf

    def Hf(self, x: Numeric, T: Boxable) -> npt.NDArray:
        return self.hf(x, T)

    def random(self, size: int | tuple[int, ...], T: Boxable) -> npt.NDArray:
        return np.ones(size) * T

    # Narrower than OptimisedFitMixin.fit by design, and no longer a
    # Liskov violation: ExactEventTime_ does not inherit that mixin,
    # so there is no wider fit above this one. The event time is
    # bracketed exactly by the censoring bounds, so there is nothing
    # for how, offset, zi or lfp to do.
    def fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
    ) -> Parametric:
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

        # The estimator needs both a right-censored bound (below T) and a
        # left-censored bound (above T); raise informatively rather than an
        # opaque zero-size numpy reduction (#257).
        if not (c == 1).any() or not (c == -1).any():
            raise ValueError(
                "ExactEventTime needs at least one right-censored (c=1) and "
                "one left-censored (c=-1) observation to bracket the event "
                "time."
            )
        max_r = np.max(x[c == 1])
        min_l = np.min(x[c == -1])

        T = (max_r + min_l) / 2.0

        model = Parametric(self, "MLE", None, False, False, False)
        model.params = np.array([T])
        return model

    def from_params(
        self,
        params: Boxable,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        """Create an ExactEventTime model from the known event time.

        ``params`` is the event time, previously named ``T``. ``gamma``,
        ``p`` and ``f0`` are accepted so the signature matches
        :meth:`ParametricFitter.from_params`, and rejected.
        """
        reject_structural_params(self.name, gamma, p, f0)
        model = Parametric(self, "from_params", None, False, False, False)
        model.params = np.array([params])
        return model


ExactEventTime: ExactEventTime_ = ExactEventTime_("ExactEventTime")
