from __future__ import annotations

import warnings
from math import comb
from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from autograd.numpy.numpy_boxes import ArrayBox

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
from surpyval.utils.no_maximum import warn_no_maximum
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

    As ``r`` grows with the mean fixed it tends to a (shifted) Poisson,
    ``T = 1 + Y`` with ``Y`` Poisson. On data no more dispersed than a
    Poisson the maximum likelihood fit runs towards that limit, which it
    never reaches, and warns ("No finite maximum", suggesting ``Poisson``
    on ``x - 1``); its ``r`` and ``p`` and their bounds are then
    meaningless.

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
            parameter_names=["r", "p"],
            param_map={"r": 0, "p": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # Method-of-moments seed from the shifted counts Y = T - 1: for the
        # negative binomial mean_Y = r(1-p)/p and var_Y = mean_Y / p, so
        # p = mean_Y / var_Y and r = mean_Y p / (1 - p). Data that are not
        # overdispersed start out towards the Poisson limit, where their
        # likelihood rises (``_warn_if_at_limit``): from r = 4, p = 1/2 the
        # search took twice as long to get there (#665).
        x = data.x
        finite = x[np.isfinite(x)]
        y = finite - 1.0 if finite.size else np.array([1.0])
        mean_y = max(y.mean(), 1e-3)
        var_y = y.var()
        if var_y > mean_y:
            p = mean_y / var_y
            r = mean_y * p / (1.0 - p)
        else:
            r = 1e3
            p = r / (r + mean_y)
        return np.array([min(max(r, 1e-2), 1e3), min(max(p, 1e-3), 1 - 1e-3)])

    def _warn_if_at_limit(
        self,
        surv_data: SurpyvalData,
        results: dict,
        zi: bool,
        lfp: bool,
    ) -> bool:
        """Warn when the likelihood is highest in the Poisson limit
        (#665), as ``BetaGeometric`` does for its Geometric limit.

        As ``r`` grows with the mean ``r (1 - p) / p`` fixed, ``T - 1``
        tends to a Poisson. On data no more dispersed than that (seven
        counts from 4 to 6) the likelihood keeps rising towards the limit,
        which it never reaches: the fit ended at r = 299 "unverified",
        with the generic message, below the shifted Poisson's
        log-likelihood (-11.917 against -11.876).

        The criterion compares the fit with its limit, the Poisson fitted
        to ``x - 1`` (with the same limited failure population): it is at
        least as likely, allowing the margin within which the fitter
        treats two starts' answers as equal. An interior maximum is
        strictly more likely than the limit it contains, so an ordinary
        fit is never flagged. Zero inflation is left to the generic
        checks: the NegativeBinomial's structural zeros are at 0, the
        shifted Poisson's would be at 1.
        """
        from .poisson import Poisson

        neg_ll = results.get("_neg_ll")
        params = np.asarray(results.get("params", []), dtype=float)
        if (
            zi
            or neg_ll is None
            or results.get("gamma", 0)
            or not np.all(np.isfinite(params))
        ):
            return super()._warn_if_at_limit(surv_data, results, zi, lfp)
        try:
            shifted = SurpyvalData(
                surv_data.x - 1.0, surv_data.c, surv_data.n, surv_data.t - 1.0
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                limit = Poisson.fit_from_surpyval_data(shifted, lfp=lfp)
        except ValueError:
            return super()._warn_if_at_limit(surv_data, results, zi, lfp)
        margin = 1e-9 * max(1.0, abs(limit._neg_ll))
        if neg_ll < limit._neg_ll - margin:
            return super()._warn_if_at_limit(surv_data, results, zi, lfp)
        r, p = params
        warn_no_maximum(
            "the NegativeBinomial likelihood keeps increasing towards its "
            "Poisson limit (r growing without bound, the mean r (1 - p) / p "
            "fixed), which the Poisson fit to x - 1 reaches "
            f"(mu = {limit.params[0]:.4g}, log-likelihood "
            f"{-limit._neg_ll:.6g} against {-neg_ll:.6g}): the data are "
            "no more dispersed than a Poisson",
            f"The reported r = {r:.4g} and p = {p:.4g}, their standard "
            "errors and their bounds are meaningless",
            "use Poisson on x - 1",
        )
        return True

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
        out = np.where(x <= 0.0, 0.0, betainc(r, safe_x, p))
        huge = x > _HUGE_K
        if np.any(huge):
            out = np.where(huge, -np.expm1(self._log_sf_huge(x, r, p)), out)
        return out

    def df(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k)`."""
        return np.exp(self.log_df(x, r, p))

    def hf(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1)`."""
        # Its limit p at k = inf, where the logs below are -inf - -inf
        # (#561)
        top = np.asarray(x) == np.inf
        if np.any(top):
            out = self.hf(np.where(top, 1.0, x), r, p)
            return np.where(top, p + np.zeros_like(out), out)
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
        # Past ``_HUGE_K`` the fraction's terms overflow, and the hazard is
        # p to a relative O(r / k) (see ``_log_sf_huge``), which the
        # difference of two logs of size k ln(1 - p) cannot resolve.
        huge = a > _HUGE_K
        if np.any(huge):
            log_hf = np.where(huge, np.log(p) + np.zeros_like(log_hf), log_hf)
        tail = (
            (a >= 1.0) & (a <= _HUGE_K) & (1.0 - p < (a + 1.0) / (a + r + 2.0))
        )
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
        from scipy.stats import nbinom

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
        >>> round(NegativeBinomial.moment(2, 3.0, 0.4), 9)
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
        from scipy.stats import nbinom

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
        out = np.where(
            x <= 0.0,
            0.0,
            betainccln(r, np.where(safe_x > _HUGE_K, 1.0, safe_x), p),
        )
        huge = x > _HUGE_K
        if np.any(huge):
            out = np.where(huge, self._log_sf_huge(x, r, p), out)
        return out

    def log_ff(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        safe_x = np.where(x <= 0.0, 1.0, x)
        out = np.where(
            x <= 0.0,
            -np.inf,
            betaincln(r, np.where(safe_x > _HUGE_K, 1.0, safe_x), p),
        )
        huge = x > _HUGE_K
        if np.any(huge):
            out = np.where(
                huge, np.log1p(-np.exp(self._log_sf_huge(x, r, p))), out
            )
        return out

    def _log_sf_huge(self, x: Numeric, r: Boxable, p: Boxable) -> Boxable:
        """log R(k) past ``_HUGE_K``, where the incomplete beta's terms
        overflow (it gave NaN from k = 1.3e154, #561): R(k) = P(T = k + 1)
        / p to a relative O(r / k), as successive masses there shrink by
        the factor 1 - p."""
        k = np.where(x > _HUGE_K, x, _HUGE_K)
        return self.log_df(k + 1.0, r, p) - np.log(p)


#: Past this many trials the tails are taken from the mass
#: (``NegativeBinomial_._log_sf_huge``), to a relative 1e-150.
_HUGE_K = 1e150

NegativeBinomial = NegativeBinomial_("NegativeBinomial")
