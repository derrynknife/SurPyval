from autograd.scipy.special import gammainc, gammaincc, gammaln
from scipy.stats import poisson

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class Poisson_(ParametricFitter):
    r"""

    The Poisson distribution as a discrete count model on the non-negative
    integers :math:`\{0, 1, 2, \dots\}`. It models the number of events in a
    fixed interval when events occur independently at a constant rate
    :math:`\mu`:

    .. math::
        P(T = k) = \frac{\mu^{k} e^{-\mu}}{k!}, \quad k = 0, 1, 2, \dots

    This is the count *distribution*, distinct from the Poisson *process*
    fitters in :mod:`surpyval.recurrent`. Its survival :math:`P(T > k)` and
    discrete hazard are defined so it plugs into the same fit/predict API as
    every other distribution.

    .. code:: python

        from surpyval import Poisson
    """

    def __init__(self, name: str):
        super().__init__(
            name=name,
            k=1,
            bounds=((0, None),),
            # The first mass point is at k = 0; the support lower bound is
            # declared just below it (-1) so observations at k = 0 pass the
            # ``x <= support[0]`` interior check (see ``Geometric`` for the
            # same convention, with its first mass at k = 1).
            support=(-1.0, np.inf),
            param_names=["mu"],
            param_map={"mu": 0},
            plot_x_scale="linear",
        )
        # A discrete lifetime is fit by maximum likelihood, not probability
        # plotting (which assumes a continuous inverse CDF).
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # The Poisson mean is mu, so the sample mean is the moment seed.
        finite = x[np.isfinite(x)]
        mu = finite.mean() if finite.size else 1.0
        return np.array([max(float(mu), 1e-3)])

    def sf(self, x, mu):
        r"""Survival function :math:`R(k) = P(T > k)`."""
        # P(X > k) = P(X >= k + 1) is the regularised lower incomplete gamma
        # ``gammainc(k + 1, mu)``.
        return gammainc(np.floor(x) + 1.0, mu)

    def ff(self, x, mu):
        r"""CDF :math:`F(k) = P(T \le k)`."""
        return gammaincc(np.floor(x) + 1.0, mu)

    def df(self, x, mu):
        r"""PMF :math:`P(T = k) = \mu^{k} e^{-\mu} / k!`."""
        return np.exp(self.log_df(x, mu))

    def hf(self, x, mu):
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        return self.df(x, mu) / self.sf(x - 1.0, mu)

    def Hf(self, x, mu):
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, mu)

    def qf(self, u, mu):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        return poisson.ppf(u, mu)

    def mean(self, mu):
        return mu

    def moment(self, m, mu):
        # Non-central moment E[T^m] by a truncated sum over the pmf out to a
        # far quantile (no simple closed form for general m).
        upper = int(poisson.ppf(1.0 - 1e-9, mu))
        k = np.arange(0, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, mu))

    def random(self, size, mu):
        return poisson.rvs(mu, size=size).astype(float)

    def log_df(self, x, mu):
        return x * np.log(mu) - mu - gammaln(x + 1.0)

    def log_sf(self, x, mu):
        return np.log(gammainc(np.floor(x) + 1.0, mu))


Poisson = Poisson_("Poisson")
