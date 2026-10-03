from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric import uniform_draws
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.surpyval_data import SurpyvalData


class DiscreteWeibull_(OptimisedFitMixin, DiscreteParametricFitter):
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

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, 1), (0, None)),
            # See ``Geometric``: the true support is {1, 2, 3, ...}; the
            # bound is declared as 0 so k = 1 passes the interior check and
            # zero-inflation (structural zeros at x = 0) is permitted.
            support=(0.0, np.inf),
            parameter_names=["q", "beta"],
            param_map={"q": 0, "beta": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # q ~ P(survive the first cycle) from the empirical fraction above 1;
        # start beta at 1 (the geometric special case).
        x = data.x
        finite = x[np.isfinite(x)]
        q = (finite > 1).mean() if finite.size else 0.5
        return np.array([min(max(q, 1e-3), 1 - 1e-3), 1.0])

    def sf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = q^{k^{\beta}}`."""
        return np.exp(self.log_sf(x, q, beta))

    def ff(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - q^{k^{\beta}}`."""
        # -expm1 keeps a small F, which 1 - R lost (#458).
        return -np.expm1(self.log_sf(x, q, beta))

    def df(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""PMF :math:`P(T=k) = q^{(k-1)^{\beta}} - q^{k^{\beta}}`."""
        # From the log mass: the difference of the two powers cancelled
        # (to 0 at k = 1e12 with shape 0.1, where the mass is 1.6e-18,
        # #458).
        return np.exp(self.log_df(x, q, beta))

    def _steps(
        self, x: Numeric, q: Boxable, beta: Boxable
    ) -> tuple[Boxable, Boxable, Boxable]:
        """For k >= 1: log R(k - 1) = (k - 1)^b log q, the hazard's
        exponent (k^b - (k - 1)^b) log q, and k clamped to 1 below.

        The difference of powers is formed as (k - 1)^b expm1(b
        log1p(1 / (k - 1))), which keeps its digits where k^b and (k -
        1)^b agree in most of theirs (k = 1e12, shape 0.1), and at k = 1
        it is 1. Below k = 1 the argument is clamped, so the discarded
        branch has neither a negative base nor 0**b (whose gradient in b
        is NaN)."""
        k = np.where(x < 1.0, 1.0, x)
        km1 = np.where(k > 1.0, k - 1.0, 1.0)
        before = np.where(k > 1.0, km1**beta, 0.0)
        ratio = np.where(k > 1.0, np.expm1(beta * np.log1p(1.0 / km1)), 1.0)
        diff = np.where(k > 1.0, before * ratio, 1.0)
        log_q = np.log(q)
        return before * log_q, diff * log_q, k

    def hf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Discrete hazard, :math:`1 - q^{k^{\beta} - (k-1)^{\beta}}`."""
        # No mass to condition on below k = 1. The exponent is a
        # difference of powers; formed as one (see ``_steps``) it no
        # longer reads inf - inf = NaN at k = 1e6 with shape 1000 (#458).
        # At k = inf the exponent is inf * 0; its limit there is that of
        # beta k^(beta - 1) log q: the hazard tends to 1, 1 - q or 0 as
        # beta is above, at or below 1 (#561).
        top = np.asarray(x) == np.inf
        if np.any(top):
            out = self.hf(np.where(top, 1.0, x), q, beta)
            limit = np.where(beta > 1, 1.0, np.where(beta == 1, 1.0 - q, 0.0))
            return np.where(top, limit + np.zeros_like(out), out)
        _, step, _ = self._steps(x, q, beta)
        return np.where(x < 1.0, 0.0, -np.expm1(step))

    def Hf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -k^{\beta}\ln q`."""
        return -self.log_sf(x, q, beta)

    def qf(self, u: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = (np.log1p(-u) / np.log(q)) ** (1.0 / beta)
        # See ``Geometric.qf``: inverting a CDF built by cancellation
        # lands a few ulp above the integer, and ceil() would answer
        # k + 1 for a u that came straight out of ``ff``.
        # (u = 1 is k = inf, where inf - inf is NaN and k stands)
        with np.errstate(invalid="ignore"):
            k = np.where(np.abs(k - np.round(k)) < 1e-9, np.round(k), k)
        return np.maximum(np.ceil(k), 1.0)

    def mean(self, q: Boxable, beta: Boxable) -> Boxable:
        r"""Mean number of cycles, :math:`E[T]` (the first moment, see
        :meth:`moment`).

        Examples
        --------
        >>> from surpyval import DiscreteWeibull
        >>> DiscreteWeibull.mean(0.9, 1.5)
        np.float64(4.549546554642062)
        """
        return self.moment(1, q, beta)

    def moment(self, m: int, q: Boxable, beta: Boxable) -> Boxable:
        r"""The ``m``-th raw moment :math:`E[T^{m}]`.

        Summed over the mass function out to the ``1 - 1e-9`` quantile,
        so it agrees with the exact value to about seven significant
        figures.

        Examples
        --------
        >>> from surpyval import DiscreteWeibull
        >>> DiscreteWeibull.moment(2, 0.9, 1.5)
        np.float64(28.30743136203336)
        """
        upper = int(self.qf(1.0 - 1e-9, q, beta))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, q, beta))

    def random(  # type: ignore[override]
        self,
        size: int | tuple[int, ...],
        q: Boxable,
        beta: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        """Draw ``size`` cycle counts by inverting the CDF (see ``qf``);
        ``random_state`` is as for :meth:`ParametricFitter.random`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import DiscreteWeibull
        >>> np.random.seed(1)
        >>> DiscreteWeibull.random(5, 0.9, 1.5)
        array([3., 6., 1., 3., 2.])
        """
        U = uniform_draws(size, random_state)
        # qf is declared Boxable because a fit differentiates it;
        # sampling never does, so this is always a real array.
        return np.asarray(self.qf(U, q, beta))

    def log_sf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        # R = 1 below the first trial; the base of ``x**beta`` is clamped
        # to 1 there (a negative base to a fractional power is complex).
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 0.0, (safe_x**beta) * np.log(q))

    def log_ff(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        # log F from F where F is small, and log1p(-R) where R is: the
        # base's log(1 - R) is 0 once R is below 1e-16 (#458). The unused
        # branch of each ``where`` gets a harmless value, so log(0) warns
        # nowhere; F = 0 (below k = 1) is -inf.
        log_sf = self.log_sf(x, q, beta)
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

    def log_df(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        # P(T = k) = R(k - 1) h(k): log R(k - 1) plus the log of the
        # hazard, both formed without a difference of powers (see
        # ``_steps``). The log of the difference of the two masses was
        # -inf where they underflowed (1e-400 at shape 1) and lost digits
        # where they agreed (#458). No mass below k = 1.
        log_before, step, _ = self._steps(x, q, beta)
        hazard = -np.expm1(step)
        return np.where(
            x < 1.0,
            -np.inf,
            log_before + np.log(np.where(x < 1.0, 1.0, hazard)),
        )


DiscreteWeibull = DiscreteWeibull_("DiscreteWeibull")
