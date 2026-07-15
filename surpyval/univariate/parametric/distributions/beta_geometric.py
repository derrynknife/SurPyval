from autograd.scipy.special import gammaln
from scipy.stats import beta as beta_rv
from scipy.stats import geom

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class BetaGeometric_(ParametricFitter):
    r"""

    The (shifted) Beta-Geometric distribution: a discrete-time frailty model
    on the positive integers :math:`\{1, 2, 3, \dots\}`. Each unit fails in a
    given cycle with its own probability ``p``, but ``p`` varies across the
    population as :math:`p \sim \mathrm{Beta}(a, b)`. Integrating the
    Geometric over that mixing distribution gives

    .. math::
        R(k) = P(T > k) = \frac{B(a,\, b + k)}{B(a,\, b)}, \qquad
        P(T = k) = \frac{B(a + 1,\, b + k - 1)}{B(a,\, b)} .

    The population heterogeneity makes the *marginal* discrete hazard
    **decrease** with time (the frailest units fail first, leaving a more
    robust survivor pool) -- behaviour a single Geometric cannot produce. It
    is the discrete-time counterpart of a continuous frailty / mixture model
    and is widely used for customer-retention ("shifted Beta-Geometric")
    modelling.

    .. code:: python

        from surpyval import BetaGeometric
    """

    def __init__(self, name: str):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            # See ``Geometric``: true support is {1, 2, 3, ...}; declared as
            # 0 so k = 1 passes the interior check.
            support=(0.0, np.inf),
            param_names=["a", "b"],
            param_map={"a": 0, "b": 1},
            plot_x_scale="linear",
        )
        self.supports_mpp = False

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # A neutral, proper starting point; the Beta(1, 1) mixing is the
        # uniform prior over p, i.e. a diffuse heterogeneity.
        return np.array([1.0, 1.0])

    def _log_beta(self, a, b):
        return gammaln(a) + gammaln(b) - gammaln(a + b)

    def sf(self, x, a, b):
        r"""Survival function :math:`R(k) = B(a, b + k)/B(a, b)`."""
        return np.exp(self.log_sf(x, a, b))

    def ff(self, x, a, b):
        r"""CDF :math:`F(k) = 1 - R(k)`."""
        return 1.0 - self.sf(x, a, b)

    def df(self, x, a, b):
        r"""PMF :math:`P(T = k) = B(a + 1, b + k - 1)/B(a, b)`."""
        return np.exp(self.log_df(x, a, b))

    def hf(self, x, a, b):
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        return self.df(x, a, b) / self.sf(x - 1.0, a, b)

    def Hf(self, x, a, b):
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, a, b)

    def qf(self, u, a, b):
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        out = np.ones_like(u)
        # The survival is monotone decreasing in k; find the smallest integer
        # k with sf(k) <= 1 - u by geometric bracketing then bisection.
        for idx, ui in enumerate(u):
            if ui <= 0.0:
                out[idx] = 1.0
                continue
            target = 1.0 - ui
            hi = 1
            while self.sf(float(hi), a, b) > target and hi < 2**40:
                hi *= 2
            lo = hi // 2
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self.sf(float(mid), a, b) > target:
                    lo = mid
                else:
                    hi = mid
            out[idx] = float(max(hi, 1))
        return out if out.size > 1 else out[0]

    def mean(self, a, b):
        # E[T] = E[1/p] with p ~ Beta(a, b) is (a + b - 1)/(a - 1) for a > 1;
        # the mean diverges for a <= 1 (heavy right tail).
        if a <= 1.0:
            return np.inf
        return (a + b - 1.0) / (a - 1.0)

    def moment(self, m, a, b):
        # Truncated sum of the pmf; the tail can be heavy, so integrate out to
        # a far survival quantile.
        upper = int(self.qf(1.0 - 1e-6, a, b))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, a, b))

    def random(self, size, a, b):
        # Draw each unit's failure probability from the Beta mixing law, then
        # a Geometric cycle count with that probability.
        p = beta_rv.rvs(a, b, size=size)
        p = np.clip(p, 1e-12, 1.0)
        return geom.rvs(p).astype(float)

    def log_sf(self, x, a, b):
        return self._log_beta(a, b + x) - self._log_beta(a, b)

    def log_df(self, x, a, b):
        return self._log_beta(a + 1.0, b + x - 1.0) - self._log_beta(a, b)


BetaGeometric = BetaGeometric_("BetaGeometric")
