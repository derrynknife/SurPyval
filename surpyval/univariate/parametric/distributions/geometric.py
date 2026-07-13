from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class Geometric_(ParametricFitter):
    r"""

    The Geometric distribution: the discrete analogue of the Exponential.
    It models the number of cycles (or trials, shocks, periods) until the
    first failure when each cycle fails independently with probability
    ``p``. The support is the positive integers :math:`\{1, 2, 3, \dots\}`.

    Its discrete hazard is constant at ``p`` (memoryless), the discrete
    counterpart of the Exponential's constant continuous hazard.

    .. code:: python

        from surpyval import Geometric
    """

    def __init__(self, name: str):
        super().__init__(
            name=name,
            k=1,
            bounds=((0, 1),),
            # The true support is the positive integers {1, 2, 3, ...}. The
            # support bound is declared as 0 (an exclusive lower bound below
            # the first mass point) so that observations at k = 1 pass the
            # ``x <= support[0]`` interior check, and so that zero-inflation
            # -- whose structural zeros sit at x = 0 -- is permitted (the
            # fitter only allows ``zi`` when ``support[0] == 0``).
            support=(0.0, np.inf),
            param_names=["p"],
            param_map={"p": 0},
            plot_x_scale="linear",
        )
        # Probability plotting assumes a continuous inverse CDF; a discrete
        # lifetime is fit by maximum likelihood.
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # Method-of-moments seed: the mean of a geometric on {1, 2, ...} is
        # 1 / p, so p ~ 1 / mean(x). Kept inside (0, 1).
        finite = x[np.isfinite(x)]
        mean = finite.mean() if finite.size else 2.0
        p = 1.0 / max(mean, 1.0 + 1e-8)
        return np.array([min(max(p, 1e-8), 1 - 1e-8)])

    def sf(self, x, p):
        r"""Survival function :math:`R(k) = (1 - p)^{k}`."""
        return (1.0 - p) ** x

    def ff(self, x, p):
        r"""CDF :math:`F(k) = 1 - (1 - p)^{k}`."""
        return 1.0 - (1.0 - p) ** x

    def df(self, x, p):
        r"""PMF :math:`P(T = k) = (1 - p)^{k - 1}\,p`."""
        return (1.0 - p) ** (x - 1.0) * p

    def hf(self, x, p):
        r"""Discrete hazard :math:`h(k) = p` (constant, memoryless)."""
        return np.ones_like(x, dtype=float) * p

    def Hf(self, x, p):
        r"""Cumulative hazard :math:`H(k) = -\ln R(k) = -k\ln(1 - p)`."""
        return -x * np.log(1.0 - p)

    def qf(self, u, p):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        q = np.ceil(np.log1p(-u) / np.log(1.0 - p))
        return np.maximum(q, 1.0)

    def mean(self, p):
        return 1.0 / p

    def moment(self, m, p):
        # Non-central moment E[T^m] by a truncated sum over the pmf out to a
        # far quantile (no simple closed form for general m).
        upper = int(self.qf(1.0 - 1e-9, p))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, p))

    def random(self, size, p):
        U = uniform.rvs(size=size)
        return self.qf(U, p)

    def log_df(self, x, p):
        return (x - 1.0) * np.log(1.0 - p) + np.log(p)

    def log_sf(self, x, p):
        return x * np.log(1.0 - p)


Geometric = Geometric_("Geometric")
