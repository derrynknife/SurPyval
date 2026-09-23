import numpy.typing as npt
from scipy.stats import uniform

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


class Geometric_(OptimisedFitMixin, DiscreteParametricFitter):
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

    def __init__(self, name: str) -> None:
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

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # Method-of-moments seed: the mean of a geometric on {1, 2, ...} is
        # 1 / p, so p ~ 1 / mean(x). Kept inside (0, 1).
        x = data.x
        finite = x[np.isfinite(x)]
        mean = finite.mean() if finite.size else 2.0
        p = 1.0 / max(mean, 1.0 + 1e-8)
        return np.array([min(max(p, 1e-8), 1 - 1e-8)])

    def sf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = (1 - p)^{k}`."""
        # Nothing can fail before the first trial, so R = 1 below zero.
        # The algebraic form returns 1/(1 - p) there -- a survival above
        # one, which ``hf`` used to divide by.
        return np.where(x < 0.0, 1.0, (1.0 - p) ** x)

    def ff(self, x: Numeric, p: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - (1 - p)^{k}`."""
        return 1.0 - self.sf(x, p)

    def df(self, x: Numeric, p: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k) = (1 - p)^{k - 1}\,p`, zero below ``k = 1``."""
        # The algebraic form does not know where the support starts: at
        # k = 0 it evaluates to p/(1 - p), a positive "probability" below
        # the first mass point (0.43 at p = 0.3), and it grows without
        # bound as k decreases. The fitter's interior check keeps such a
        # value out of a likelihood, but df is public and a caller
        # plotting a pmf from zero would get it.
        return np.where(x < 1.0, 0.0, (1.0 - p) ** (x - 1.0) * p)

    def hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = p` (constant, memoryless)."""
        # Constant on the support, but zero below it: h(k) = P(T = k)/R(k - 1)
        # and there is no mass to condition on before k = 1.
        return np.where(x < 1.0, 0.0, np.ones_like(x, dtype=float) * p)

    def Hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k) = -k\ln(1 - p)`."""
        return np.where(x < 0.0, 0.0, -x * np.log(1.0 - p))

    def qf(self, u: Numeric, p: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = np.log1p(-u) / np.log(1.0 - p)
        # A caller inverting the CDF passes u = F(k), which was formed as
        # 1 - (1 - p)^k. Recovering k from it lands a few ulp above the
        # integer, and a bare ceil() then answers k + 1 -- so F and its
        # quantile did not invert each other. Snap first.
        k = np.where(np.abs(k - np.round(k)) < 1e-9, np.round(k), k)
        return np.maximum(np.ceil(k), 1.0)

    def mean(self, p: Boxable) -> Boxable:
        return 1.0 / p

    def moment(self, m: int, p: Boxable) -> Boxable:
        # Non-central moment E[T^m] by a truncated sum over the pmf out to a
        # far quantile (no simple closed form for general m).
        upper = int(self.qf(1.0 - 1e-9, p))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, p))

    def random(self, size: int | tuple[int, ...], p: Boxable) -> npt.NDArray:
        U = uniform.rvs(size=size)
        # qf is declared Boxable because a fit differentiates it;
        # sampling never does, so this is always a real array.
        return np.asarray(self.qf(U, p))

    def log_df(self, x: Numeric, p: Boxable) -> Boxable:
        # -inf below the support, matching ``df``'s zero.
        return np.where(
            x < 1.0, -np.inf, (x - 1.0) * np.log(1.0 - p) + np.log(p)
        )

    def log_sf(self, x: Numeric, p: Boxable) -> Boxable:
        return np.where(x < 0.0, 0.0, x * np.log(1.0 - p))


Geometric = Geometric_("Geometric")
