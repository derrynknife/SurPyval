from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData


class DiscretizedFitter(OptimisedFitMixin, DiscreteParametricFitter):
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

    def __init__(self, distribution: ParametricFitter) -> None:
        if distribution.support[0] < 0:
            raise ValueError(
                "Discretize is defined for distributions supported on the "
                "non-negative reals (support[0] >= 0); {} is supported on "
                "{}.".format(distribution.name, distribution.support)
            )
        # Held as Any, not ParametricFitter. The base class declares
        # none of sf, ff, df, hf, Hf, qf or _parameter_initialiser --
        # its docstring states the contract but the class does not
        # express it -- so every delegation below would be an
        # attr-defined error against the declared type. The parameter
        # stays annotated, because that is the contract for callers.
        self.dist: Any = distribution
        super().__init__(
            # e.g. "Discretize(Weibull)"; distinct from the standalone
            # ``DiscreteWeibull`` (Nakagawa-Osaki) distribution.
            name="Discretize(" + distribution.name + ")",
            k=distribution.k,
            bounds=distribution.bounds,
            # Mass is grouped onto {1, 2, ...}; support lower bound declared
            # as 0 (below the first mass at k = 1) for the interior check.
            support=(0.0, np.inf),
            parameter_names=list(distribution.parameter_names),
            param_map=dict(distribution.param_map),
            plot_x_scale=distribution.plot_x_scale,
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        return np.asarray(
            self.dist._parameter_initialiser(data),
            dtype=float,
        )

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        r"""Survival :math:`R_K(k) = R(k)` (the continuous survival)."""
        return self.dist.sf(x, *params)

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        r"""CDF :math:`F_K(k) = F(k)`."""
        return self.dist.ff(x, *params)

    def _sf_before(self, x: Numeric, *params: Boxable) -> Boxable:
        r""":math:`R(k - 1)`: exactly 1 at and below the start of the
        continuous support (``k = 1``), where the continuous survival is
        1 but its derivatives in the parameters through its formula are
        not finite (a Weibull's ``(0 / alpha)**beta`` has a Hessian of
        ``0 * inf``), and the fit's Hessian came out NaN (#562)."""
        before = x - 1.0
        start = self.dist.support[0]
        at_start = before <= start
        if not np.any(at_start):
            return self.dist.sf(before, *params)
        inside = np.where(at_start, start + 1.0, before)
        return np.where(at_start, 1.0, self.dist.sf(inside, *params))

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        r"""PMF :math:`P(K = k) = R(k - 1) - R(k)`."""
        return self._sf_before(x, *params) - self.dist.sf(x, *params)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(K = k)/R(k - 1)`."""
        before = self._sf_before(x, *params)
        gone = before == 0
        if not np.any(gone):
            return self.df(x, *params) / before
        # Once R(k - 1) underflows the ratio is 0 / 0 (NaN from k = 1e6
        # for a Weibull(4.4, 1.6), #561). The hazard is then 1 - R(k) /
        # R(k - 1) from the log survival while k - 1 and k are told apart
        # in it, and past that (and at k = inf) the limit 1 - exp(-h) of
        # the continuous hazard h, which varies slowly there.
        k = np.asarray(x, dtype=float)
        with np.errstate(invalid="ignore"):
            ratio = self.df(x, *params) / before
            log_before = self.dist.log_sf(k - 1.0, *params)
            from_logs = -np.expm1(self.dist.log_sf(k, *params) - log_before)
            limit = -np.expm1(-self.dist.hf(k, *params))
        resolved = np.isfinite(log_before) & (k < 1e15)
        tail = np.where(resolved, from_logs, limit)
        out = np.where(gone, tail, ratio)
        return out[()] if out.ndim == 0 else out

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return self.dist.Hf(x, *params)

    def qf(self, u: Numeric, *params: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = np.maximum(np.ceil(self.dist.qf(u, *params)), 1.0)
        # The continuous quantile of u = F(k) lands on k only up to
        # round-off, and ceil() of k + 1e-15 is k + 1: qf(ff(6)) was 7
        # (#383). Step back to k - 1 where F(k - 1) already reaches u, up
        # to a relative 1e-12 of the smaller of u and 1 - u (compared on
        # that side, so a tail probability keeps its digits), and above
        # 1/2 also up to u's own rounding (half an ulp below 1).
        before = k - 1.0
        lower = u <= 0.5
        half_ulp = np.finfo(float).eps / 4.0
        reached = np.where(
            lower,
            self.dist.ff(before, *params) >= u * (1.0 - 1e-12),
            self.dist.sf(before, *params)
            <= (1.0 - u) * (1.0 + 1e-12) + half_ulp,
        )
        k = np.where((before >= 1.0) & reached, before, k)
        return k[()] if k.ndim == 0 else k

    def mean(self, *params: Boxable) -> Boxable:
        upper = int(np.ceil(self.dist.qf(1.0 - 1e-9, *params)))
        k = np.arange(1, upper + 1, dtype=float)
        return float(np.sum(k * self.df(k, *params)))

    def moment(self, m: int, *params: Boxable) -> Boxable:
        upper = int(np.ceil(self.dist.qf(1.0 - 1e-9, *params)))
        k = np.arange(1, upper + 1, dtype=float)
        return float(np.sum(k**m * self.df(k, *params)))

    def random(
        self,
        size: int | tuple[int, ...],
        *params: Boxable,
        random_state: Any = None,
    ) -> npt.NDArray:
        return np.ceil(
            self.dist.random(size, *params, random_state=random_state)
        )

    def log_df(self, x: Numeric, *params: Boxable) -> Boxable:
        return np.log(self.df(x, *params))

    def log_sf(self, x: Numeric, *params: Boxable) -> Boxable:
        return self.dist.log_sf(x, *params)


def Discretize(distribution: ParametricFitter) -> DiscretizedFitter:
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
