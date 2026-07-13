from autograd.scipy.special import gammaln
from scipy.stats import nbinom

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils.autograd_gamma_compat import betainc, betaincln


class NegativeBinomial_(ParametricFitter):
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

    def __init__(self, name: str):
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
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # Method-of-moments seed from the shifted counts Y = T - 1: for the
        # negative binomial mean_Y = r(1-p)/p and var_Y = mean_Y / p, so
        # p = mean_Y / var_Y and r = mean_Y p / (1 - p). Falls back to a
        # neutral guess when the data are not overdispersed.
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

    def sf(self, x, r, p):
        r"""Survival function :math:`R(k) = I_{1-p}(k, r)`."""
        return betainc(x, r, 1.0 - p)

    def ff(self, x, r, p):
        r"""CDF :math:`F(k) = I_{p}(r, k)`."""
        return betainc(r, x, p)

    def df(self, x, r, p):
        r"""PMF :math:`P(T = k)`."""
        return np.exp(self.log_df(x, r, p))

    def hf(self, x, r, p):
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        return self.df(x, r, p) / self.sf(x - 1.0, r, p)

    def Hf(self, x, r, p):
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, r, p)

    def qf(self, u, r, p):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        return nbinom.ppf(u, r, p) + 1.0

    def mean(self, r, p):
        return 1.0 + r * (1.0 - p) / p

    def moment(self, m, r, p):
        upper = int(self.qf(1.0 - 1e-9, r, p))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, r, p))

    def random(self, size, r, p):
        return nbinom.rvs(r, p, size=size) + 1.0

    def log_df(self, x, r, p):
        return (
            gammaln(x - 1.0 + r)
            - gammaln(r)
            - gammaln(x)
            + r * np.log(p)
            + (x - 1.0) * np.log(1.0 - p)
        )

    def log_sf(self, x, r, p):
        return betaincln(x, r, 1.0 - p)


NegativeBinomial = NegativeBinomial_("NegativeBinomial")
