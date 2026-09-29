from math import comb
from typing import Any

import numpy.typing as npt
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.stats import nbinom

from surpyval import np
from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
    stirling2_numbers,
)
from surpyval.univariate.parametric.parametric import draw_state
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.autograd_gamma_compat import (
    beta_cf,
    betainc,
    betainccln,
    betaincln,
    betaln_accurate,
)
from surpyval.utils.surpyval_data import SurpyvalData

from ._discrete_tails import refine_quantile


class NegativeBinomial_(OptimisedFitMixin, DiscreteParametricFitter):
    r"""

    The Negative Binomial distribution as a discrete lifetime on the
    positive integers :math:`\{1, 2, 3, \dots\}`. With ``T = 1 + Y`` and
    ``Y`` the number of failures before the ``r``-th success (each trial
    succeeding with probability ``p``), it models the number of cycles
    until an item accumulates enough shocks/successes to fail.

    .. math::
        P(T = k) = \frac{\Gamma(k - 1 + r)}{\Gamma(r)\,\Gamma(k)}
                   \, p^{r}\, (1 - p)^{k - 1}

    with ``r > 0`` (a real-valued shape / dispersion) and ``0 < p < 1``.
    It generalises the Geometric (``r = 1``) and, being overdispersed
    relative to the Poisson, is the natural discrete model for
    shock-accumulation lifetimes and heterogeneous count data.

    .. code:: python

        from surpyval import NegativeBinomial
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, 1)),
            # See ``Geometric``: true support is {1, 2, 3, ...}; declared as
            # 0 so k = 1 passes the interior check and zero-inflation is
            # permitted (structural zeros sit at x = 0).
            support=(0.0, np.inf),
            param_names=["r", "p"],
            param_map={"r": 0, "p": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # Method-of-moments seed from the shifted counts Y = T - 1: for the
        # negative binomial mean_Y = r(1-p)/p and var_Y = mean_Y / p, so
        # p = mean_Y / var_Y and r = mean_Y p / (1 - p). Falls back to a
        # neutral guess when the data are not overdispersed.
        x = data.x
        finite = x[np.isfinite(x)]
        y = finite - 1.0 if finite.size else np.array([1.0])
        mean_y = max(y.mean(), 1e-3)
        var_y = y.var()
        if var_y > mean_y:
            p = mean_y / var_y
            r = mean_y * p / (1.0 - p)
        else:
            p, r = 0.5, max(mean_y, 1.0)
        return np.array([min(max(r, 1e-2), 1e3), min(max(p, 1e-3), 1 - 1e-3)])

    def sf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = I_{1-p}(k, r)`."""
        # R = 1 below the first mass point at k = 1. The incomplete beta's
        # first argument must be positive, so it returns NaN for k < 0
        # rather than the 1 it happens to give at k = 0.
        # 1 - F where F is below 1/2, and the upper tail itself (from its
        # log) where it is small. The incomplete beta at 1 - p, the form
        # this used, rounds 1 - p, which a small p cannot afford (#458).
        ff = self.ff(x, r, p)
        return np.where(ff < 0.5, 1.0 - ff, np.exp(self.log_sf(x, r, p)))

    def ff(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = I_{p}(r, k)`."""
        # Nothing fails before k = 1 (I_p(r, 0) is 1, not 0).
        safe_x = np.where(x <= 0.0, 1.0, x)
        return np.where(x <= 0.0, 0.0, betainc(r, safe_x, p))

    def df(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k)`."""
        return np.exp(self.log_df(x, r, p))

    def hf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        # On the log scale: df/sf was 0/0 = nan once both underflowed
        # (#458).
        log_hf = self.log_df(x, r, p) - self.log_sf(x - 1.0, r, p)
        if isinstance(r, ArrayBox) or isinstance(p, ArrayBox):
            return np.exp(log_hf)
        # Deep in the right tail both logs are of size k ln(1 - p), and
        # their difference lost 3e-5 of the hazard at k = 1e12. There
        # R(k - 1) = I_{1-p}(k - 1, r) is exactly P(T = k) times the
        # incomplete beta's continued fraction, so the hazard is its
        # reciprocal, with no cancellation. (Values only: a fit
        # differentiates the log form above.)
        x = np.asarray(x, dtype=float)
        a = x - 1.0
        tail = (a >= 1.0) & (1.0 - p < (a + 1.0) / (a + r + 2.0))
        if np.any(tail):
            a_t = np.where(tail, a, 1.0)
            cf = beta_cf(a_t, r, np.where(tail, 1.0 - p, 0.5))
            log_hf = np.where(tail, -np.log(np.abs(cf)), log_hf)
        return np.exp(log_hf)

    def Hf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, r, p)

    def qf(self, u: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`."""
        u_arr = np.asarray(u, dtype=float)
        k = refine_quantile(
            nbinom.ppf(u_arr, r, p) + 1.0,
            u_arr,
            lambda k: self.log_sf(k, r, p),
            lambda k: self.log_ff(k, r, p),
            first=1.0,
        ).reshape(u_arr.shape)
        return k[()] if k.ndim == 0 else k

    def mean(self, r: Boxable, p: Boxable) -> Boxable:
        r"""Mean number of cycles, :math:`E[T] = 1 + r(1 - p)/p`.

        Examples
        --------
        >>> from surpyval import NegativeBinomial
        >>> NegativeBinomial.mean(3.0, 0.4)
        5.499999999999999
        """
        return 1.0 + r * (1.0 - p) / p

    def moment(self, m: int, r: Boxable, p: Boxable) -> Boxable:
        r"""The ``m``-th raw moment :math:`E[T^{m}]`, exactly.

        With :math:`T = 1 + Y`, the factorial moments of :math:`Y` are
        :math:`E[(Y)_{j}] = r (r + 1) \cdots (r + j - 1)\,((1 - p)/p)^{j}`;
        the Stirling numbers of the second kind turn them into raw moments
        of :math:`Y`, and the binomial expansion of :math:`(1 + Y)^{m}`
        into those of :math:`T`. ``moment(1)`` is ``mean()``.

        Examples
        --------
        >>> from surpyval import NegativeBinomial
        >>> NegativeBinomial.moment(2, 3.0, 0.4)
        41.5
        """
        if m == 0:
            return 1.0
        if m == 1:
            return self.mean(r, p)
        # A sum over the mass function to the 1 - 1e-9 quantile used to
        # stand in for this, and lost the eighth digit.
        odds = (1.0 - p) / p
        factorial: list = [1.0]
        for j in range(1, m + 1):
            factorial.append(factorial[-1] * (r + j - 1.0) * odds)
        raw_y = [
            sum(s * factorial[i] for i, s in enumerate(stirling2_numbers(j)))
            for j in range(m + 1)
        ]
        return float(sum(comb(m, j) * raw_y[j] for j in range(m + 1)))

    def random(  # type: ignore[override]
        self,
        size: int | tuple[int, ...],
        r: Boxable,
        p: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        """Draw ``size`` cycle counts, ``1 +`` a negative binomial draw;
        ``random_state`` is as for :meth:`ParametricFitter.random`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import NegativeBinomial
        >>> np.random.seed(1)
        >>> NegativeBinomial.random(5, 3.0, 0.4)
        array([ 7.,  4.,  2., 20.,  3.])
        """
        state = draw_state(random_state)
        return nbinom.rvs(r, p, size=size, random_state=state) + 1.0

    def log_df(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        # The coefficient G(k - 1 + r) / (G(r) G(k)) is 1 / ((k - 1)
        # B(r, k - 1)), exactly 1 at k = 1, with ln B taken without the
        # cancellation of two gammaln of size k ln k (5e-6 of the mass at
        # k = 1e9); and log1p(-p), not log(1 - p), which loses a small p
        # (#458).
        safe_x = np.where(x < 2.0, 2.0, x)
        log_coef = np.where(
            x < 2.0,
            0.0,
            -np.log(safe_x - 1.0) - betaln_accurate(r, safe_x - 1.0),
        )
        k = np.where(x < 1.0, 1.0, x)
        return np.where(
            x < 1.0,
            -np.inf,
            log_coef + r * np.log(p) + (k - 1.0) * np.log1p(-p),
        )

    def log_sf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        # R(k) = 1 - I_p(r, k), the upper tail's own log at p itself (see
        # ``betainccln``): log(sf) was capped at -708 and lost R near 1
        # (#458). R = 1 up to k = 0.
        safe_x = np.where(x <= 0.0, 1.0, x)
        return np.where(x <= 0.0, 0.0, betainccln(r, safe_x, p))

    def log_ff(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        safe_x = np.where(x <= 0.0, 1.0, x)
        return np.where(x <= 0.0, -np.inf, betaincln(r, safe_x, p))


NegativeBinomial = NegativeBinomial_("NegativeBinomial")
