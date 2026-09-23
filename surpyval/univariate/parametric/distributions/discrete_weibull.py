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
            param_names=["q", "beta"],
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
        # Below zero the base of ``x**beta`` is negative and a fractional
        # power of it is complex -- sf(-1) came back as 1.035+0.547j.
        # Nothing can fail before the first trial, so R = 1 there. The
        # dead branch is evaluated at 1 rather than 0 because ``0**beta``
        # has a NaN gradient with respect to beta (see ``log_df``).
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 1.0, q ** (safe_x**beta))

    def ff(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - q^{k^{\beta}}`."""
        return 1.0 - self.sf(x, q, beta)

    def df(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""PMF :math:`P(T=k) = q^{(k-1)^{\beta}} - q^{k^{\beta}}`."""
        # Below k = 1 the exponent base (k - 1) is negative, and a negative
        # base to a fractional power is complex: df(0) came back as
        # 0.0355+0.5468j. Guard the base as ``log_df`` already does, then
        # zero the whole thing below the support.
        # ``x`` is clamped, not just the result, so the discarded branch of
        # the np.where never evaluates a negative base at all -- otherwise
        # it still computes the NaN and warns before throwing it away.
        safe_x = np.where(x < 1.0, 1.0, x)
        km1 = safe_x - 1.0
        safe_km1 = np.where(km1 > 0, km1, 1.0)
        term_low = np.where(km1 > 0, q ** (safe_km1**beta), 1.0)
        return np.where(x < 1.0, 0.0, term_low - q ** (safe_x**beta))

    def hf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Discrete hazard, :math:`1 - q^{k^{\beta} - (k-1)^{\beta}}`."""
        # Same negative-base problem as ``df``, and no mass to condition
        # on below k = 1.
        safe_x = np.where(x < 1.0, 1.0, x)
        km1 = safe_x - 1.0
        safe_km1 = np.where(km1 > 0, km1, 1.0)
        exponent = safe_x**beta - np.where(km1 > 0, safe_km1**beta, 0.0)
        return np.where(x < 1.0, 0.0, 1.0 - q**exponent)

    def Hf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -k^{\beta}\ln q`."""
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 0.0, -(safe_x**beta) * np.log(q))

    def qf(self, u: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u = np.asarray(u, dtype=float)
        k = (np.log1p(-u) / np.log(q)) ** (1.0 / beta)
        # See ``Geometric.qf``: inverting a CDF built by cancellation
        # lands a few ulp above the integer, and ceil() would answer
        # k + 1 for a u that came straight out of ``ff``.
        k = np.where(np.abs(k - np.round(k)) < 1e-9, np.round(k), k)
        return np.maximum(np.ceil(k), 1.0)

    def mean(self, q: Boxable, beta: Boxable) -> Boxable:
        return self.moment(1, q, beta)

    def moment(self, m: int, q: Boxable, beta: Boxable) -> Boxable:
        upper = int(self.qf(1.0 - 1e-9, q, beta))
        k = np.arange(1, upper + 1, dtype=float)
        return np.sum(k**m * self.df(k, q, beta))

    def random(
        self, size: int | tuple[int, ...], q: Boxable, beta: Boxable
    ) -> npt.NDArray:
        U = uniform.rvs(size=size)
        # qf is declared Boxable because a fit differentiates it;
        # sampling never does, so this is always a real array.
        return np.asarray(self.qf(U, q, beta))

    def log_sf(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        safe_x = np.where(x < 0.0, 1.0, x)
        return np.where(x < 0.0, 0.0, (safe_x**beta) * np.log(q))

    def log_df(self, x: Numeric, q: Boxable, beta: Boxable) -> Boxable:
        # PMF = q^{(k-1)^beta} - q^{k^beta}. At k = 1 the first term is
        # q^{0^beta} = 1 with no beta dependence, but 0**beta has a NaN
        # gradient w.r.t. beta under autograd, so guard the base: where
        # k = 1 the term is the constant 1.
        # Below k = 1 there is no mass, so this is -inf. Clamping ``x`` to 1
        # rather than leaving it means the discarded branch evaluates the
        # k = 1 mass (1 - q, safely positive) instead of a negative base
        # and a log of zero, both of which warn before being thrown away.
        safe_x = np.where(x < 1.0, 1.0, x)
        km1 = safe_x - 1.0
        safe_km1 = np.where(km1 > 0, km1, 1.0)
        term_low = np.where(km1 > 0, q ** (safe_km1**beta), 1.0)
        term_high = q ** (safe_x**beta)
        return np.where(x < 1.0, -np.inf, np.log(term_low - term_high))


DiscreteWeibull = DiscreteWeibull_("DiscreteWeibull")
