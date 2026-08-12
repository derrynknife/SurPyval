import numpy.typing as npt
from autograd.scipy.special import gammaln
from scipy.stats import beta as beta_rv
from scipy.stats import geom

from surpyval import np
from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.surpyval_data import SurpyvalData


class BetaGeometric_(OptimisedFitMixin, DiscreteParametricFitter):
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

    def __init__(self, name: str) -> None:
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

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # A neutral, proper starting point; the Beta(1, 1) mixing is the
        # uniform prior over p, i.e. a diffuse heterogeneity.
        return np.array([1.0, 1.0])

    def _log_beta(self, a: Boxable, b: Boxable) -> Boxable:
        return gammaln(a) + gammaln(b) - gammaln(a + b)

    def sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = B(a, b + k)/B(a, b)`."""
        return np.exp(self.log_sf(x, a, b))

    def ff(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - R(k)`."""
        return 1.0 - self.sf(x, a, b)

    def df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k) = B(a + 1, b + k - 1)/B(a, b)`."""
        return np.exp(self.log_df(x, a, b))

    def hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        return self.df(x, a, b) / self.sf(x - 1.0, a, b)

    def Hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, a, b)

    def qf(self, u: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        out = np.ones_like(u_arr)
        # The survival is monotone decreasing in k; find the smallest integer
        # k with sf(k) <= 1 - u by geometric bracketing then bisection.
        for idx, ui in enumerate(u_arr):
            if ui <= 0.0:
                out[idx] = 1.0
                continue
            # ``target`` is reached by cancellation -- the caller almost
            # always passes u = F(k) = 1 - R(k), and 1 - (1 - R(k)) lands
            # one ulp below R(k). A strict comparison then rejects the
            # exact answer and returns k + 1, so F and its quantile did
            # not invert each other. Compare with a relative slack.
            target = (1.0 - ui) * (1.0 + 1e-12)
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

    def mean(self, a: Boxable, b: Boxable) -> Boxable:
        # E[T] = E[1/p] with p ~ Beta(a, b) is (a + b - 1)/(a - 1) for a > 1;
        # the mean diverges for a <= 1 (heavy right tail).
        if a <= 1.0:
            return np.inf
        return (a + b - 1.0) / (a - 1.0)

    def moment(self, m: int, a: Boxable, b: Boxable) -> Boxable:
        # The survival decays as k^-a, so E[T^m] converges only for a > m --
        # the same condition ``mean`` applies at m = 1. Without the test a
        # truncated sum reports a finite value for a moment that does not
        # exist: at a = 2, b = 3 the second moment is infinite and the old
        # sum returned about 25.
        if a <= m:
            return np.inf
        if m == 1:
            # Exact, and the reason mean() and moment(1) now agree: the
            # truncated sum lost 0.17% of a heavy tail even at the 1 - 1e-6
            # quantile.
            return self.mean(a, b)
        # No closed form for general m; sum out to a far quantile.
        upper = int(self.qf(1.0 - 1e-6, a, b))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, a, b))

    def random(
        self, size: int | tuple[int, ...], a: Boxable, b: Boxable
    ) -> npt.NDArray:
        # Draw each unit's failure probability from the Beta mixing law, then
        # a Geometric cycle count with that probability.
        p = beta_rv.rvs(a, b, size=size)
        p = np.clip(p, 1e-12, 1.0)
        return geom.rvs(p).astype(float)

    def log_sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        # R(k) = 1 for every k below the first mass point. The Beta-ratio
        # form does not know that -- at k = -1 it returns 2.0, a survival
        # above one -- so clamp the argument at zero, where it is already 1.
        safe_x = np.where(x < 0.0, 0.0, x)
        return self._log_beta(a, b + safe_x) - self._log_beta(a, b)

    def log_df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        # Zero mass below k = 1. The Beta-ratio form returns 1.0 at k = 0,
        # and B(a + 1, b + k - 1) is undefined once b + k - 1 <= 0, so the
        # argument is clamped before the guard chooses the branch.
        safe_x = np.where(x < 1.0, 1.0, x)
        log_df = self._log_beta(a + 1.0, b + safe_x - 1.0) - self._log_beta(
            a, b
        )
        return np.where(x < 1.0, -np.inf, log_df)


BetaGeometric = BetaGeometric_("BetaGeometric")
