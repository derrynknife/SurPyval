from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class DiscretizedFitter(ParametricFitter):
    r"""

    A continuous lifetime distribution discretized to the positive integers
    :math:`\{1, 2, 3, \dots\}` by grouping into unit-width bins:
    :math:`K = \lceil T \rceil` for a continuous lifetime ``T``, so

    .. math::
        P(K = k) = F(k) - F(k - 1), \qquad R_K(k) = R(k),

    where ``F`` and ``R`` are the continuous CDF and survival. The discrete
    survival at an integer therefore equals the continuous survival, and the
    probability mass is the continuous probability of falling in the interval
    ``(k - 1, k]``. This is the cheap, general way to obtain a discrete
    Gamma, Log-Normal, Normal-truncated, etc. from any non-negative
    continuous SurPyval distribution, fit by maximum likelihood on the same
    parameters as the underlying distribution.

    Created with the :func:`Discretize` factory rather than directly.
    """

    def __init__(self, distribution):
        if distribution.support[0] < 0:
            raise ValueError(
                "Discretize is defined for distributions supported on the "
                "non-negative reals (support[0] >= 0); {} is supported on "
                "{}.".format(distribution.name, distribution.support)
            )
        self.dist = distribution
        super().__init__(
            # e.g. "Discretize(Weibull)"; distinct from the standalone
            # ``DiscreteWeibull`` (Nakagawa-Osaki) distribution.
            name="Discretize(" + distribution.name + ")",
            k=distribution.k,
            bounds=distribution.bounds,
            # Mass is grouped onto {1, 2, ...}; support lower bound declared
            # as 0 (below the first mass at k = 1) for the interior check.
            support=(0.0, np.inf),
            param_names=list(distribution.param_names),
            param_map=dict(distribution.param_map),
            plot_x_scale=distribution.plot_x_scale,
        )
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        return self.dist._parameter_initialiser(x, c=c, n=n, t=t)

    def sf(self, x, *params):
        r"""Survival :math:`R_K(k) = R(k)` (the continuous survival)."""
        return self.dist.sf(x, *params)

    def ff(self, x, *params):
        r"""CDF :math:`F_K(k) = F(k)`."""
        return self.dist.ff(x, *params)

    def df(self, x, *params):
        r"""PMF :math:`P(K = k) = R(k - 1) - R(k)`."""
        return self.dist.sf(x - 1.0, *params) - self.dist.sf(x, *params)

    def hf(self, x, *params):
        r"""Discrete hazard :math:`h(k) = P(K = k)/R(k - 1)`."""
        return self.df(x, *params) / self.dist.sf(x - 1.0, *params)

    def Hf(self, x, *params):
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return self.dist.Hf(x, *params)

    def qf(self, u, *params):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        return np.maximum(np.ceil(self.dist.qf(u, *params)), 1.0)

    def mean(self, *params):
        upper = int(np.ceil(self.dist.qf(1.0 - 1e-9, *params)))
        k = np.arange(1, upper + 1, dtype=float)
        return float(np.sum(k * self.df(k, *params)))

    def moment(self, m, *params):
        upper = int(np.ceil(self.dist.qf(1.0 - 1e-9, *params)))
        k = np.arange(1, upper + 1, dtype=float)
        return float(np.sum(k**m * self.df(k, *params)))

    def random(self, size, *params):
        return np.ceil(self.dist.random(size, *params))

    def log_df(self, x, *params):
        return np.log(self.df(x, *params))

    def log_sf(self, x, *params):
        return self.dist.log_sf(x, *params)


def Discretize(distribution):
    r"""
    Discretize a continuous SurPyval distribution onto the positive integers.

    Wraps any non-negative continuous distribution so that
    :math:`K = \lceil T \rceil`: the discrete survival equals the continuous
    survival at each integer and the mass is
    :math:`P(K = k) = F(k) - F(k - 1)`. The wrapped model is fit by maximum
    likelihood on the underlying distribution's parameters.

    Parameters
    ----------
    distribution : ParametricFitter
        A continuous SurPyval distribution supported on ``[0, inf)``
        (e.g. ``Weibull``, ``Gamma``, ``LogNormal``).

    Returns
    -------
    DiscretizedFitter
        A discrete fitter with the usual ``fit`` / ``sf`` / ``df`` / ... API.

    Examples
    --------
    >>> from surpyval import Weibull, Discretize
    >>> DiscreteWeibull = Discretize(Weibull)
    >>> model = DiscreteWeibull.fit([1, 2, 2, 3, 4, 5, 3, 2])  # doctest: +SKIP
    """
    return DiscretizedFitter(distribution)
