from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class DiscreteWeibull_(ParametricFitter):
    r"""

    The (Type I) discrete Weibull distribution of Nakagawa & Osaki (1975):
    the discrete analogue of the continuous Weibull, and the discrete
    lifetime model with a flexible (increasing, constant, or decreasing)
    hazard on a cycle count. The support is the positive integers
    :math:`\{1, 2, 3, \dots\}`.

    .. math::
        R(k) = q^{\,k^{\beta}}

    with :math:`0 < q < 1` and :math:`\beta > 0`. ``beta`` controls the
    discrete hazard shape -- ``beta < 1`` decreasing (infant mortality),
    ``beta = 1`` constant (it reduces to the Geometric with ``p = 1 - q``),
    ``beta > 1`` increasing (wear-out). ``q`` is the probability of
    surviving the first cycle, ``R(1) = q``.

    .. code:: python

        from surpyval import DiscreteWeibull

    Reference
    ---------
    Nakagawa, T. and Osaki, S. (1975), "The discrete Weibull distribution",
    IEEE Transactions on Reliability R-24, 300-301.
    """

    def __init__(self, name: str):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, 1), (0, None)),
            # See ``Geometric``: the true support is {1, 2, 3, ...}; the
            # bound is declared as 0 so k = 1 passes the interior check and
            # zero-inflation (structural zeros at x = 0) is permitted.
            support=(0.0, np.inf),
            param_names=["q", "beta"],
            param_map={"q": 0, "beta": 1},
            plot_x_scale="linear",
        )
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # q ~ P(survive the first cycle) from the empirical fraction above 1;
        # start beta at 1 (the geometric special case).
        finite = x[np.isfinite(x)]
        q = (finite > 1).mean() if finite.size else 0.5
        return np.array([min(max(q, 1e-3), 1 - 1e-3), 1.0])

    def sf(self, x, q, beta):
        r"""Survival function :math:`R(k) = q^{k^{\beta}}`."""
        return q ** (x**beta)

    def ff(self, x, q, beta):
        r"""CDF :math:`F(k) = 1 - q^{k^{\beta}}`."""
        return 1.0 - q ** (x**beta)

    def df(self, x, q, beta):
        r"""PMF :math:`P(T=k) = q^{(k-1)^{\beta}} - q^{k^{\beta}}`."""
        return q ** ((x - 1.0) ** beta) - q ** (x**beta)

    def hf(self, x, q, beta):
        r"""Discrete hazard, :math:`1 - q^{k^{\beta} - (k-1)^{\beta}}`."""
        return 1.0 - q ** (x**beta - (x - 1.0) ** beta)

    def Hf(self, x, q, beta):
        r"""Cumulative hazard :math:`H(k) = -k^{\beta}\ln q`."""
        return -(x**beta) * np.log(q)

    def qf(self, u, q, beta):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = (np.log1p(-u) / np.log(q)) ** (1.0 / beta)
        return np.maximum(np.ceil(k), 1.0)

    def mean(self, q, beta):
        return self.moment(1, q, beta)

    def moment(self, m, q, beta):
        upper = int(self.qf(1.0 - 1e-9, q, beta))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, q, beta))

    def random(self, size, q, beta):
        U = uniform.rvs(size=size)
        return self.qf(U, q, beta)

    def log_sf(self, x, q, beta):
        return (x**beta) * np.log(q)

    def log_df(self, x, q, beta):
        # PMF = q^{(k-1)^beta} - q^{k^beta}. At k = 1 the first term is
        # q^{0^beta} = 1 with no beta dependence, but 0**beta has a NaN
        # gradient w.r.t. beta under autograd, so guard the base: where
        # k = 1 the term is the constant 1.
        km1 = x - 1.0
        safe_km1 = np.where(km1 > 0, km1, 1.0)
        term_low = np.where(km1 > 0, q ** (safe_km1**beta), 1.0)
        term_high = q ** (x**beta)
        return np.log(term_low - term_high)


DiscreteWeibull = DiscreteWeibull_("DiscreteWeibull")
