from __future__ import annotations

import warnings
from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from autograd.scipy.special import gammaln
from scipy.stats import beta as beta_rv
from scipy.stats import geom

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
    eulerian_numbers,
)
from surpyval.univariate.parametric.parametric import draw_state
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.surpyval_data import SurpyvalData

from ._discrete_tails import log_gamma_ratio


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

    As ``a`` and ``b`` grow with ``a / (a + b)`` fixed it tends to a
    Geometric. On data no more dispersed than a Geometric the maximum
    likelihood fit runs towards that limit, which it never reaches, and
    warns ("No finite maximum", suggesting ``Geometric``); its ``a`` and
    ``b`` and their bounds are then meaningless.

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
            parameter_names=["a", "b"],
            param_map={"a": 0, "b": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        # A neutral, proper starting point; the Beta(1, 1) mixing is the
        # uniform prior over p, i.e. a diffuse heterogeneity.
        return np.array([1.0, 1.0])

    def _warn_if_at_limit(
        self,
        surv_data: SurpyvalData,
        results: dict,
        zi: bool,
        lfp: bool,
    ) -> bool:
        """Warn when the likelihood is highest in the Geometric limit.

        As ``a`` and ``b`` grow with ``a / (a + b)`` fixed the Beta mixing
        law concentrates on one ``p`` and the BetaGeometric becomes a
        Geometric. On data that show no more heterogeneity than a single
        Geometric, the likelihood keeps rising towards that limit, which
        it never reaches: the fit to 12 rows from the registry ran to
        ``a, b = 9.8e4, 3.5e5``, in silence, and every bound was then nan.

        The criterion compares the fit with its limit: the Geometric fit
        to the same data (with the same zero inflation or limited failure
        population) is at least as likely, allowing the margin within
        which the fitter itself treats two starts' answers as equal. An
        interior maximum is strictly more likely than the limit it
        contains, so an ordinary fit is never flagged: on data drawn from
        a Geometric one sample in five was more dispersed by chance, and
        its fit (a, b = 49, 113) stands.
        """
        from .geometric import Geometric

        neg_ll = results.get("_neg_ll")
        params = np.asarray(results.get("params", []), dtype=float)
        if neg_ll is None or not np.all(np.isfinite(params)):
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                limit = Geometric.fit_from_surpyval_data(
                    surv_data, zi=zi, lfp=lfp
                )
        except ValueError:
            return False
        margin = 1e-9 * max(1.0, abs(limit._neg_ll))
        if neg_ll < limit._neg_ll - margin:
            return False
        a, b = params
        warn_no_maximum(
            "the BetaGeometric likelihood keeps increasing towards its "
            "Geometric limit (a and b growing without bound, a / (a + b) "
            "fixed), which the Geometric fit to the same data reaches "
            f"(p = {limit.params[0]:.4g}, log-likelihood "
            f"{-limit._neg_ll:.6g} against {-neg_ll:.6g}): the data show no "
            "variation in the per-cycle failure probability between units",
            f"The reported a = {a:.4g} and b = {b:.4g}, their standard "
            "errors and their bounds are meaningless",
            "use Geometric",
        )
        return True

    def _log_beta(self, a: Boxable, b: Boxable) -> Boxable:
        return gammaln(a) + gammaln(b) - gammaln(a + b)

    def sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Survival function :math:`R(k) = B(a, b + k)/B(a, b)`."""
        return np.exp(self.log_sf(x, a, b))

    def ff(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""CDF :math:`F(k) = 1 - R(k)`."""
        # -expm1 keeps a small F exact, where 1 - R lost it.
        return -np.expm1(self.log_sf(x, a, b))

    def df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""PMF :math:`P(T = k) = B(a + 1, b + k - 1)/B(a, b)`."""
        return np.exp(self.log_df(x, a, b))

    def hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Discrete hazard :math:`h(k) = P(T = k)/R(k - 1) =
        a / (a + b + k - 1)`, zero below ``k = 1``."""
        # The closed form. The ratio df/sf was 0/0 = nan once both
        # underflowed (at k = 1e6 with a = 1000, #449).
        safe_x = np.where(x < 1.0, 1.0, x)
        return np.where(x < 1.0, 0.0, a / (a + b + safe_x - 1.0))

    def Hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Cumulative hazard :math:`H(k) = -\ln R(k)`."""
        return -self.log_sf(x, a, b)

    def qf(self, u: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Quantile: the smallest integer ``k`` with :math:`F(k) \geq u`;
        infinite at ``u = 1`` (the support has no last point)."""
        u_in = np.asarray(u, dtype=float)
        u_arr, a_arr, b_arr = np.broadcast_arrays(
            u_in, np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        )
        # F(k) >= u is tested on the smaller side, on the log scale: log F
        # against log u below 1/2, log R against log(1 - u) above it, so
        # neither loses a small probability. The caller usually passes
        # u = F(k), and recovering the threshold from it can land an ulp
        # on the wrong side, so the test has a relative slack of 1e-12.
        lower = u_arr <= 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(
                lower, np.log(u_arr), np.log1p(-np.where(lower, 0.0, u_arr))
            )
        slack = 1e-12 * np.maximum(1.0, np.abs(target))
        slack = np.where(np.isfinite(slack), slack, 0.0)

        def reached(k: npt.NDArray) -> npt.NDArray:
            log_sf = self.log_sf(k, a_arr, b_arr)
            with np.errstate(divide="ignore"):
                log_ff = np.log(-np.expm1(log_sf))
            return np.where(
                lower, log_ff >= target - slack, log_sf <= target + slack
            )

        # Bracket by doubling, then bisect over the integers. The survival
        # decays like k^-a, so at a small ``a`` the answer can be beyond
        # the largest double: inf.
        todo = (u_arr > 0.0) & (u_arr < 1.0)
        hi = np.ones_like(u_arr)
        while True:
            grow = todo & np.isfinite(hi) & ~reached(hi)
            if not grow.any():
                break
            hi = np.where(grow, hi * 2.0, hi)
        lo = np.where(hi > 1.0, hi / 2.0, 1.0)
        search = todo & np.isfinite(hi) & (hi > 1.0)
        while True:
            mid = np.floor(0.5 * (lo + hi))
            step = search & (mid > lo) & (mid < hi)
            if not step.any():
                break
            ok = reached(mid)
            hi = np.where(step & ok, mid, hi)
            lo = np.where(step & ~ok, mid, lo)
        out = np.where(u_arr <= 0.0, 1.0, np.where(u_arr >= 1.0, np.inf, hi))
        # A missing or impossible probability has no quantile.
        out = np.where(np.isnan(u_arr) | (u_arr > 1.0), np.nan, out)
        # The shape of ``u``: a scalar for a scalar, empty for empty.
        return out[()] if out.ndim == 0 else out

    def mean(self, a: Boxable, b: Boxable) -> Boxable:
        r"""Mean number of cycles, :math:`E[T] = (a + b - 1)/(a - 1)`,
        infinite when :math:`a \leq 1`.

        Examples
        --------
        >>> from surpyval import BetaGeometric
        >>> BetaGeometric.mean(5.0, 3.0)
        1.75
        """
        # E[T] = E[1/p] with p ~ Beta(a, b) is (a + b - 1)/(a - 1) for a > 1;
        # the mean diverges for a <= 1 (heavy right tail).
        if a <= 1.0:
            return np.inf
        return (a + b - 1.0) / (a - 1.0)

    def moment(self, m: int, a: Boxable, b: Boxable) -> Boxable:
        r"""The ``m``-th raw moment :math:`E[T^{m}]`.

        Infinite unless :math:`a > m` (the survival decays like
        :math:`k^{-a}`), and otherwise exact: mixing the Geometric's
        :math:`E[T^{m} \mid p] = p^{-m} \sum_{i} A(m, i) (1 - p)^{i}`
        (:math:`A` the Eulerian numbers) over :math:`p \sim
        \mathrm{Beta}(a, b)` gives

        .. math::
            E[T^{m}] = \sum_{i=0}^{m-1} A(m, i)\,
            \frac{B(a - m,\, b + i)}{B(a, b)} .

        Examples
        --------
        >>> from surpyval import BetaGeometric
        >>> BetaGeometric.moment(2, 5.0, 3.0)
        5.25
        >>> BetaGeometric.moment(2, 2.0, 3.0)
        inf
        """
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
        if m == 2:
            # Also exact: E[T^2 | p] = 2/p^2 - 1/p for a Geometric, and
            # E[1/p^2] = (a + b - 1)(a + b - 2) / ((a - 1)(a - 2)) under the
            # Beta mixing. The truncated sum below lost the tail here too
            # (5.2413 against 5.25 at a = 5, b = 3), and the variance and
            # the method of moments both read this moment.
            c = a + b - 1.0
            return 2.0 * c * (c - 1.0) / ((a - 1.0) * (a - 2.0)) - c / (
                a - 1.0
            )
        # The general case of the two above. A sum over the mass function
        # to the 1 - 1e-6 quantile used to stand in for it, and lost the
        # heavy tail: 32.28 against 33.25 for m = 3 at a = 5, b = 3.
        log_norm = self._log_beta(a, b)
        return float(
            sum(
                A * np.exp(self._log_beta(a - m, b + i) - log_norm)
                for i, A in enumerate(eulerian_numbers(m))
            )
        )

    def _mom(self, x: npt.NDArray) -> tuple[float, float]:
        r"""Method-of-moments estimate, solved in closed form.

        With :math:`m_1` and :math:`m_2` the first two sample moments and
        :math:`s = (m_1 + m_2) / 2` the matching :math:`E[1/p^2]`, the
        moment equations :math:`m_1 = (a + b - 1)/(a - 1)` and
        :math:`s = m_1 (a + b - 2)/(a - 2)` give, with :math:`q = s / m_1`,

        .. math::
            a = \frac{2q - m_1 - 1}{q - m_1}, \qquad
            b = (m_1 - 1)(a - 1).

        The generic numerical route cannot do this: it starts at the
        ``_parameter_initialiser`` point ``a = b = 1``, where neither
        moment exists, so its objective was nan from the first step and it
        handed the start back as the fit.

        A solution exists only for a sample more dispersed than a
        Geometric with the same mean (variance above
        :math:`m_1 (m_1 - 1)`) -- the Beta mixing can only add
        dispersion -- and it always has :math:`a > 2`, where both moments
        are finite. Anything else is refused rather than answered.
        """
        m1 = float(np.mean(x))
        m2 = float(np.mean(np.asarray(x, dtype=float) ** 2))
        q = (m1 + m2) / (2.0 * m1) if m1 > 0 else np.nan
        if not (m1 > 1.0 and q > m1):
            raise ValueError(
                "Method of moments has no Beta-Geometric solution for this "
                "sample: it needs a mean above 1 and a variance above that "
                f"of a Geometric with the same mean (m1 (m1 - 1) = "
                f"{m1 * (m1 - 1.0):.4g}; the sample's is "
                f"{m2 - m1**2:.4g}). Use how='MLE', or a Geometric."
            )
        a = (2.0 * q - m1 - 1.0) / (q - m1)
        b = (m1 - 1.0) * (a - 1.0)
        return a, b

    def random(  # type: ignore[override]
        self,
        size: int | tuple[int, ...],
        a: Boxable,
        b: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        """Draw ``size`` cycle counts: a per-unit probability from the
        Beta(``a``, ``b``) mixing law, then a Geometric count with it;
        ``random_state`` is as for :meth:`ParametricFitter.random`."""
        # Draw each unit's failure probability from the Beta mixing law, then
        # a Geometric cycle count with that probability.
        state = draw_state(random_state)
        p = beta_rv.rvs(a, b, size=size, random_state=state)
        p = np.clip(p, 1e-12, 1.0)
        return geom.rvs(p, random_state=state).astype(float)

    def log_sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        # R(k) = 1 for every k below the first mass point. The Beta-ratio
        # form does not know that -- at k = -1 it returns 2.0, a survival
        # above one -- so clamp the argument at zero, where it is already 1.
        #
        # ln R(k) = ln B(a, b + k) - ln B(a, b)
        #         = [ln G(b + a) - ln G(b)] - [ln G(b + k + a) - ln G(b + k)],
        # each bracket taken by ``log_gamma_ratio``, which keeps its digits
        # at a large k where the four gammaln did not (#449).
        safe_x = np.where(x < 0.0, 0.0, x)
        return log_gamma_ratio(b, a) - log_gamma_ratio(b + safe_x, a)

    def log_ff(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        # log F from F where F is small, and log1p(-R) where R is: the
        # base's log(1 - R) is 0 once R is below 1e-16. The unused branch
        # of each ``where`` gets a harmless value, so log(0) warns nowhere;
        # F = 0 (below k = 1) is -inf.
        log_sf = self.log_sf(x, a, b)
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

    def log_df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        # Zero mass below k = 1; above it P(T = k) = R(k - 1) h(k), with
        # the closed-form hazard h(k) = a / (a + b + k - 1). The argument
        # is clamped before the guard chooses the branch.
        safe_x = np.where(x < 1.0, 1.0, x)
        log_df = (
            self.log_sf(safe_x - 1.0, a, b)
            + np.log(a)
            - np.log(a + b + safe_x - 1.0)
        )
        return np.where(x < 1.0, -np.inf, log_df)


BetaGeometric = BetaGeometric_("BetaGeometric")
