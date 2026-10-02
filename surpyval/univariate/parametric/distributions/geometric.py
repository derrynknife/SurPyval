from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
    eulerian_numbers,
)
from surpyval.univariate.parametric.parametric import uniform_draws
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
            parameter_names=["p"],
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
        return np.exp(self.log_sf(x, p))

    def ff(self, x: Numeric, p: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - (1 - p)^{k}`."""
        # -expm1 keeps a small F exact: 1 - R lost the digits of F below
        # about 1e-8 (ff(1, 1e-9) was 9.9999997e-10, #446).
        return -np.expm1(self.log_sf(x, p))

    def df(self, x: Numeric, p: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k) = (1 - p)^{k - 1}\,p`, zero below ``k = 1``."""
        # The algebraic form does not know where the support starts: at
        # k = 0 it evaluates to p/(1 - p), a positive "probability" below
        # the first mass point (0.43 at p = 0.3), and it grows without
        # bound as k decreases. The fitter's interior check keeps such a
        # value out of a likelihood, but df is public and a caller
        # plotting a pmf from zero would get it.
        return np.exp(self.log_df(x, p))

    def hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = p` (constant, memoryless)."""
        # Constant on the support, but zero below it: h(k) = P(T = k)/R(k - 1)
        # and there is no mass to condition on before k = 1.
        return np.where(x < 1.0, 0.0, np.ones_like(x, dtype=float) * p)

    def Hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k) = -k\ln(1 - p)`."""
        return -self.log_sf(x, p)

    def qf(self, u: Numeric, p: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = np.log1p(-u) / np.log1p(-p)
        # A caller inverting the CDF passes u = F(k), which was formed as
        # 1 - (1 - p)^k. Recovering k from it lands a few ulp above the
        # integer, and a bare ceil() then answers k + 1 -- so F and its
        # quantile did not invert each other. Snap first.
        k = np.where(np.abs(k - np.round(k)) < 1e-9, np.round(k), k)
        return np.maximum(np.ceil(k), 1.0)

    def mean(self, p: Boxable) -> Boxable:
        r"""Mean number of cycles to failure, :math:`E[T] = 1/p`.

        Examples
        --------
        >>> from surpyval import Geometric
        >>> Geometric.mean(0.2)
        5.0
        """
        return 1.0 / p

    def moment(self, m: int, p: Boxable) -> Boxable:
        r"""The ``m``-th raw moment :math:`E[T^{m}]`, exactly:

        .. math::
            E[T^{m}] = p^{-m} \sum_{i=0}^{m-1} A(m, i)\,(1 - p)^{i}

        with :math:`A(m, i)` the Eulerian numbers, so ``moment(1)`` is
        ``mean()``.

        Examples
        --------
        >>> from surpyval import Geometric
        >>> Geometric.moment(2, 0.2)
        45.0
        """
        if m == 0:
            return 1.0
        if m == 1:
            return self.mean(p)
        # The old sum over the mass function stopped at the 1 - 1e-9
        # quantile, so moment(1) and mean() disagreed in the eighth digit.
        q = 1.0 - p
        return sum(a * q**i for i, a in enumerate(eulerian_numbers(m))) / p**m

    def random(  # type: ignore[override]
        self,
        size: int | tuple[int, ...],
        p: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        """Draw ``size`` cycle counts by inverting the CDF (see ``qf``);
        ``random_state`` is as for :meth:`ParametricFitter.random`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Geometric
        >>> np.random.seed(1)
        >>> Geometric.random(5, 0.3)
        array([2., 4., 1., 2., 1.])
        """
        U = uniform_draws(size, random_state)
        # qf is declared Boxable because a fit differentiates it;
        # sampling never does, so this is always a real array.
        return np.asarray(self.qf(U, p))

    # log1p(-p), not log(1 - p): 1 - p rounds away the digits of a small
    # p (8 of them at p = 1e-9), and every function here is built on it
    # (#446).

    def log_df(self, x: Numeric, p: Boxable) -> Boxable:
        # -inf below the support, matching ``df``'s zero.
        return np.where(x < 1.0, -np.inf, (x - 1.0) * np.log1p(-p) + np.log(p))

    def log_sf(self, x: Numeric, p: Boxable) -> Boxable:
        return np.where(x <= 0.0, 0.0, x * np.log1p(-p))

    def log_ff(self, x: Numeric, p: Boxable) -> Boxable:
        # log F from F where F is small, and log1p(-R) where R is: the
        # base's log(-expm1(-H)) is log(1 - R), which rounds to 0 once R
        # is below 1e-16 (log_ff was 0.0 where it is -1e-30).
        # The unused branch of each ``where`` gets a harmless 1 or 0, so
        # log(0) warns nowhere; F = 0 (below k = 1) is -inf.
        log_sf = self.log_sf(x, p)
        F = -np.expm1(log_sf)
        small = F < 0.5
        return np.where(
            F <= 0.0,
            -np.inf,
            np.where(
                small,
                np.log(np.where(small & (F > 0.0), F, 1.0)),
                np.log1p(-np.where(small, 0.0, np.exp(log_sf))),
            ),
        )


Geometric = Geometric_("Geometric")
