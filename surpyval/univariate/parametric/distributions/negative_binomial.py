import numpy.typing as npt
from autograd.scipy.special import gammaln
from scipy.stats import nbinom

from surpyval import np
from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.autograd_gamma_compat import betainc, betaincln
from surpyval.utils.surpyval_data import SurpyvalData


class NegativeBinomial_(OptimisedFitMixin, DiscreteParametricFitter):
    r"""

    The Negative Binomial distribution as a discrete lifetime on the
    positive integers :math:`\{1, 2, 3, \dots\}`. With ``T = 1 + Y`` and
    ``Y`` the number of failures before the ``r``-th success (each trial
    succeeding with probability ``p``), it models the number of cycles
    until an item accumulates enough shocks/successes to fail.

    .. math::
        P(T = k) = \frac{\Gamma(k - 1 + r)}{\Gamma(r)\,\Gamma(k)}
                   \, p^{r}\, (1 - p)^{k - 1}

    with ``r > 0`` (a real-valued shape / dispersion) and ``0 < p < 1``.
    It generalises the Geometric (``r = 1``) and, being overdispersed
    relative to the Poisson, is the natural discrete model for
    shock-accumulation lifetimes and heterogeneous count data.

    .. code:: python

        from surpyval import NegativeBinomial
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, 1)),
            # See ``Geometric``: true support is {1, 2, 3, ...}; declared as
            # 0 so k = 1 passes the interior check and zero-inflation is
            # permitted (structural zeros sit at x = 0).
            support=(0.0, np.inf),
            param_names=["r", "p"],
            param_map={"r": 0, "p": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # Method-of-moments seed from the shifted counts Y = T - 1: for the
        # negative binomial mean_Y = r(1-p)/p and var_Y = mean_Y / p, so
        # p = mean_Y / var_Y and r = mean_Y p / (1 - p). Falls back to a
        # neutral guess when the data are not overdispersed.
        x = data.x
        finite = x[np.isfinite(x)]
        y = finite - 1.0 if finite.size else np.array([1.0])
        mean_y = max(y.mean(), 1e-3)
        var_y = y.var()
        if var_y > mean_y:
            p = mean_y / var_y
            r = mean_y * p / (1.0 - p)
        else:
            p, r = 0.5, max(mean_y, 1.0)
        return np.array([min(max(r, 1e-2), 1e3), min(max(p, 1e-3), 1 - 1e-3)])

    def sf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = I_{1-p}(k, r)`."""
        # R = 1 below the first mass point at k = 1. The incomplete beta's
        # first argument must be positive, so it returns NaN for k < 0
        # rather than the 1 it happens to give at k = 0.
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 1.0, betainc(safe_x, r, 1.0 - p))

    def ff(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = I_{p}(r, k)`."""
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 0.0, betainc(r, safe_x, p))

    def df(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k)`."""
        return np.exp(self.log_df(x, r, p))

    def hf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        return self.df(x, r, p) / self.sf(x - 1.0, r, p)

    def Hf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, r, p)

    def qf(self, u: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        return nbinom.ppf(u, r, p) + 1.0

    def mean(self, r: Boxable, p: Boxable) -> Boxable:
        return 1.0 + r * (1.0 - p) / p

    def moment(self, m: int, r: Boxable, p: Boxable) -> Boxable:
        upper = int(self.qf(1.0 - 1e-9, r, p))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, r, p))

    def random(
        self, size: int | tuple[int, ...], r: Boxable, p: Boxable
    ) -> npt.NDArray:
        return nbinom.rvs(r, p, size=size) + 1.0

    def log_df(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        safe_x = np.where(x < 1.0, 1.0, x)
        return np.where(
            x < 1.0,
            -np.inf,
            gammaln(safe_x - 1.0 + r)
            - gammaln(r)
            - gammaln(safe_x)
            + r * np.log(p)
            + (safe_x - 1.0) * np.log(1.0 - p),
        )

    def log_sf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 0.0, betaincln(safe_x, r, 1.0 - p))


NegativeBinomial = NegativeBinomial_("NegativeBinomial")
