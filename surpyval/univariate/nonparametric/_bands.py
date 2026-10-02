"""The simultaneous confidence bands of the non-parametric estimates.

``BandsMixin``, which
:class:`~surpyval.univariate.nonparametric.nonparametric.NonParametric`
inherits, holds ``band`` (the Hall-Wellner and equal precision bands) and
its critical values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import norm

from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import alpha_ci_error, check_option

# The equal precision band's default range of a = N sigma^2 / (1 + N
# sigma^2) (#390): its standardized boundary is unbounded as a nears 0 or
# 1, and there the estimate rests on a few failures or a few at risk. Over
# the first to the last event it covered 0.87-0.93 for 0.95; over this
# range 0.94-0.96 (n = 40 to 400). Klein and Moeschberger tabulate it for
# a_L from 0.02 and a_U to 0.98.
_EP_RANGE = (0.1, 0.9)


class BandsMixin:
    """The confidence bands of a :class:`NonParametric` estimate, which
    inherits this mixin."""

    if TYPE_CHECKING:
        # Supplied by NonParametric, the one class that inherits this
        # mixin. Declared rather than defined so the methods below type
        # check without the mixin pretending to own them.
        x: npt.NDArray
        r: npt.NDArray
        R: npt.NDArray
        greenwood: npt.NDArray
        data: dict[str, Any]
        _band_n: "float | None"

    def _band_sample_size(self) -> float:
        """The sample size N of ``band``: the number of items fitted,
        from the data or, restored without them, as ``to_dict`` stored it.
        A model with neither (``from_xrd``, or a dictionary written before
        it was stored) takes the largest risk set, the same number unless
        the data were left truncated (#451)."""
        if getattr(self, "data", None) is not None and "n" in self.data:
            return float(self.data["n"].sum())
        if self._band_n is not None:
            return self._band_n
        return float(np.max(self.r))

    @staticmethod
    def _band_critical_value(
        a_l: float,
        a_u: float,
        alpha_ci: float,
        standardized: bool,
    ) -> float:
        r"""
        Critical value of the supremum of :math:`|B(a)|`, a Brownian bridge
        (the Hall-Wellner band), or of :math:`|B(a)|/\sqrt{a(1 - a)}` (the
        equal precision band), over :math:`[a_l, a_u]`: the ``c`` with
        :math:`P(\sup |\cdot| \le c) = 1 - \alpha`.

        It used to be simulated from bridge paths on a 1000-point grid,
        which misses the excursions between grid points and never looks
        below :math:`a = 0.001`: the value came out about 1.5% low (1.337
        against the Kolmogorov 1.358 over the whole range), for roughly
        94.4% coverage, and a valid range falling between grid points
        crashed. It is now computed numerically, and deterministically.

        With :math:`t = a/(1 - a)`, :math:`B(a) = W(t)/(1 + t)` for a
        Brownian motion :math:`W`, so the event is that :math:`W` stays
        inside :math:`\pm b(t)`, with :math:`b(t) = c(1 + t)` for
        Hall-Wellner and :math:`c\sqrt{t}` for the equal precision band.
        The density of :math:`W(t)/b(t)` on :math:`[-1, 1]` is propagated
        across a grid of times with the Gaussian transition kernel, each
        step weighted by the probability that the Brownian bridge between
        its two ends does not touch either boundary, which for a boundary
        linear over the step is :math:`1 - e^{-2(b - x)(b' - y)/\Delta t}`
        (exact for Hall-Wellner, whose boundary is linear in :math:`t`).
        The non-crossing probability is increasing in ``c``, which is found
        by root finding. It reproduces the Kolmogorov quantiles over the
        whole range to about 1e-8.
        """
        if not 0 < 1 - alpha_ci < 1:
            raise alpha_ci_error(alpha_ci)
        # t = a / (1 - a); a_u = 1 would put the end at infinity, which
        # the grid below cannot reach in finitely many steps.
        a_u = min(float(a_u), 1.0 - 1e-12)
        a_l = min(max(float(a_l), 0.0), a_u)
        t_l, t_u = a_l / (1 - a_l), a_u / (1 - a_u)
        if standardized and t_l <= 0:
            # The standardized bridge is unbounded near a = 0 (the law of
            # the iterated logarithm), so there is no finite value.
            raise ValueError(
                "The equal precision band needs a range [a_l, a_u] with "
                "a_l > 0"
            )

        u = np.linspace(-1.0, 1.0, 401)
        w = np.full(u.size, u[1] - u[0])
        w[[0, -1]] *= 0.5

        def kernel(m: float, s: float, A: float) -> npt.NDArray:
            # Row i: the (quadrature-weighted) density of reaching u[j]
            # from u[i] without touching either boundary. The two one-sided
            # survival factors multiply, which neglects touching both in
            # one step: with a step variance of at most a tenth of b^2
            # that is below e^-80.
            K = np.exp(-0.5 * ((u[None, :] - m * u[:, None]) / s) ** 2)
            K /= s * np.sqrt(2 * np.pi)
            K *= -np.expm1(-A * np.outer(1 - u, 1 - u))
            K *= -np.expm1(-A * np.outer(1 + u, 1 + u))
            return w[:, None] * K

        def inside(c: float) -> float:
            if standardized:
                # u = W(t) / (c sqrt(t)) starts as N(0, 1 / c^2). On a
                # geometric grid t_{k+1} = q t_k the step in these
                # coordinates is the same at every k, so one kernel serves
                # them all; with q <= 1.02 the chord replacing sqrt(t)
                # within a step is within 2e-5 of it, and q - 1 <= c^2 / 10
                # keeps the step variance, (q - 1) t, below b^2 / 10.
                g = norm.pdf(u, scale=1.0 / c)
                if t_u > t_l:
                    q_max = 1.0 + min(0.02, 0.1 * c**2)
                    n = int(np.ceil(np.log(t_u / t_l) / np.log(q_max)))
                    q = (t_u / t_l) ** (1.0 / n)
                    K = kernel(
                        1.0 / np.sqrt(q),
                        np.sqrt((q - 1.0) / q) / c,
                        2.0 * c**2 * np.sqrt(q) / (q - 1.0),
                    )
                    for _ in range(n):
                        g = g @ K
                return float(g @ w)
            # Hall-Wellner, u = W(t) / (c (1 + t)) = B(a) / c. Within
            # c^2 / 64 of either end of [0, 1] the bridge's standard
            # deviation is below c / 8, so the chance of it reaching c
            # there is below 4 * Phi(-8) ~ 3e-15. A range reaching into
            # those ends is cut back to them, where the density of u is
            # still wide enough for the grid (the bridge is pinned to 0 at
            # both ends, which no grid resolves); a start moved up to t_s
            # takes W(t_s) ~ N(0, t_s).
            edge = c**2 / 64.0
            t_s = max(t_l, min(edge, t_u))
            t_e = max(t_s, min(t_u, (1.0 - edge) / edge))
            b = c * (1.0 + t_s)
            g = norm.pdf(u * b, scale=np.sqrt(t_s)) * b
            # Steps of equal size in v = 1 / (1 + t) = 1 - a keep each
            # step's variance at about a tenth of b^2.
            v_s, v_u = 1.0 / (1.0 + t_s), 1.0 / (1.0 + t_e)
            n = int(np.ceil((v_s - v_u) / (0.1 * c**2)))
            ts = 1.0 / np.linspace(v_s, v_u, n + 1) - 1.0
            for t0, t1 in zip(ts[:-1], ts[1:]):
                b0, b1 = c * (1.0 + t0), c * (1.0 + t1)
                dt = t1 - t0
                g = g @ kernel(b0 / b1, np.sqrt(dt) / b1, 2 * b0 * b1 / dt)
            return float(g @ w)

        target = 1.0 - alpha_ci
        # The supremum is at least |B(a)| at any single a, so the two-sided
        # normal quantile there bounds c from below.
        z = norm.ppf(1.0 - alpha_ci / 2.0)
        if standardized:
            lo = z
        else:
            a_mid = min(max(0.5, a_l), a_u)
            lo = z * np.sqrt(a_mid * (1.0 - a_mid))
        if not t_u > t_l:
            # A single point (a_l == a_u): the bound is attained.
            return float(lo)
        # The number of grid steps grows as 1 / c^2, so ``inside`` is only
        # evaluated down to about the root. The search used to climb from
        # ``lo``, which is far below the root when alpha_ci is large (z is
        # 1.25e-6 at alpha_ci = 1 - 1e-6): alpha_ci = 0.9 took 13 s, and
        # 1 - 1e-6 asked for a 158 TiB grid (#420). At the root itself the
        # steps are bounded (about 8 log(1 / (1 - alpha_ci)), however narrow
        # the range: a path staying within +-c over a stretch of n steps has
        # a probability of about e^(-n / 8)). So start from an upper value
        # and come down: for Hall-Wellner the Kolmogorov tail bound,
        # P(sup |B| > c) <= 2 exp(-2 c^2) over the whole of [0, 1], and for
        # the equal precision band 1 (or ``lo``), climbing if it is short.
        if standardized:
            c = max(lo, 1.0)
        else:
            c = max(lo, np.sqrt(np.log(2.0 / alpha_ci) / 2.0))
        if inside(c) >= target:
            hi = c
            while True:
                c = max(hi / 1.5, lo)
                if c == lo:
                    # Within a factor 1.5 of the root, so the grid is too.
                    if inside(lo) >= target:
                        # A range too short for the grid to tell apart from
                        # a single point.
                        return float(lo)
                    break
                if inside(c) < target:
                    break
                hi = c
            lo = c
        else:
            lo, hi = c, 1.5 * c
            while inside(hi) < target:
                lo, hi = hi, 1.5 * hi
        return float(brentq(lambda c: inside(c) - target, lo, hi, xtol=1e-8))

    @keeps_query_shape
    def band(
        self,
        x: npt.ArrayLike | None = None,
        method: str = "hall-wellner",
        bound_type: str = "arcsine",
        alpha_ci: float = 0.05,
        x_range: "tuple[float, float] | None" = None,
    ) -> npt.NDArray:
        r"""
        Simultaneous confidence band of the survival function.

        The pointwise bounds from ``cb()`` cover the true value of the
        survival function at each single time with probability
        1 - alpha_ci, but the probability that the *whole* true curve
        lies between them is lower, since the curve has many
        opportunities to escape. A confidence band is widened so that,
        with probability 1 - alpha_ci, the entire survival function
        lies within the band over the observed range. Use the band, not
        the pointwise bounds, to assess whether a hypothesised curve
        (e.g. a fitted parametric distribution) is consistent with the
        data as a whole.

        Two classical bands are available:

        - "hall-wellner": width proportional to
          :math:`(1 + n\sigma^2(t))/\sqrt{n}`; tends to be relatively
          wider in the middle of the curve.
        - "nair" (equal precision): width proportional to the pointwise
          standard error, i.e. the band is the pointwise interval
          scaled by a larger critical value, so its width follows the
          pointwise bounds everywhere.

        Critical values are those of the limiting Brownian bridge
        process over the range the band covers, computed numerically
        rather than read from a table or simulated, so they are accurate
        for any range and results are reproducible.

        A band covers a range of times :math:`[t_L, t_U]` (``x_range``),
        and is NaN outside it. Its critical value depends on the range
        through :math:`a = N\sigma^2/(1 + N\sigma^2)` at its two ends,
        :math:`\sigma^2` the Greenwood sum, and the theory behind it holds
        for a range inside the data, :math:`0 < a_L < a_U < 1`. The
        Hall-Wellner band covers the first to the last observed event
        (where the variance estimate is positive and finite, and the
        estimate strictly between 0 and 1). The equal precision band's
        boundary grows without limit as :math:`a` approaches 0 or 1, and
        near either end of the data the estimate rests on a handful of
        failures, or of items at risk, whose error is far from normal; by
        default it covers the times with :math:`0.1 \le a \le 0.9`
        (inside the range of Klein and Moeschberger's tables, from 0.02
        to 0.98). Outside the band's range NaN is returned, whether or
        not the model has bounds (``set_support``). The asymptotic theory
        for these bands is for right censored data; for Turnbull models
        with interval censoring prefer ``bootstrap_cb()``.

        The band is applied on the arcsine-square-root scale by default
        (Klein and Moeschberger's transformed bands; Borgan and Liestøl,
        1990). In simulation (Weibull lifetimes, n = 40 to 400, 30%
        censored, the true curve checked over the band's whole range) the
        equal precision band over the first to the last event covered
        0.87 to 0.89 for a nominal 0.95 on the log(-log) scale, the
        default until v0.22, its misses mostly at the first events, 0.93
        on the arcsine scale, and over its default range 0.94 to 0.96 on
        the arcsine scale. The Hall-Wellner band covers about 0.95 over
        the whole range on either scale (#390).

        Parameters
        ----------

        x : array like or scalar, optional
            The values at which the band will be evaluated. Defaults to
            the observed values.
        method : ('hall-wellner', 'nair'), str, optional
            The type of band. Defaults to 'hall-wellner'.
        bound_type : ('arcsine', 'exp', 'normal'), str, optional
            The scale the band is applied on: 'arcsine' the
            arcsine-square-root of the survival function, 'exp' its
            log(-log), as the pointwise ``cb()`` default, and 'normal' the
            survival function itself. 'arcsine' and 'exp' keep the band
            within [0, 1]. Defaults to 'arcsine', the one that holds its
            level from the first event on (see above).
        alpha_ci : scalar, optional
            The level of significance of the band. Defaults to 0.05.
        x_range : (t_L, t_U), optional
            The times the band covers. Defaults to the first to the last
            event for the Hall-Wellner band, and to the times with
            :math:`0.1 \le a \le 0.9` (see above) for the equal precision
            band.

        Returns
        -------

        band : numpy array
            Array of shape (len(x), 2) with the ``[lower, upper]`` band
            values for the survival function at each x, NaN outside the
            band's range.

        Raises
        ------

        ValueError
            If no value in the band's range has a positive, finite
            variance with an estimate strictly between 0 and 1 (e.g. no
            failures), or the model has no variance estimate
            (``fit_from_ecdf``), or ``alpha_ci`` is not strictly between 0
            and 1, or ``x_range`` is not two increasing times.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> model = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8],
        ...                         c=[0, 1, 0, 0, 1, 0, 0, 1])
        >>> model.band([4, 6]).round(4)
        array([[0.1209, 0.9652],
               [0.0051, 0.9151]])
        >>> model.cb([4, 6]).round(4)
        array([[0.1802, 0.8441],
               [0.063 , 0.7242]])

        References
        ----------

        Borgan, Ø. and Liestøl, K. (1990), "A note on confidence intervals
        and bands for the survival function based on transformations",
        Scandinavian Journal of Statistics 17, 35-41.

        Hall, W. J. and Wellner, J. A. (1980), "Confidence bands for a
        survival curve from censored data", Biometrika 67, 133-143.

        Nair, V. N. (1984), "Confidence bands for survival functions
        with censored data: a comparative study", Technometrics 26,
        265-275.

        Klein, J. P. and Moeschberger, M. L. (2003), "Survival
        Analysis", 2nd ed., Section 4.4.
        """
        check_option("method", method, ("hall-wellner", "nair"))
        check_option("bound_type", bound_type, ("arcsine", "exp", "normal"))
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bands "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )

        N = self._band_sample_size()

        with np.errstate(all="ignore"):

            sigma2 = self.greenwood
            valid = (
                np.isfinite(sigma2)
                & (sigma2 > 0)
                & (self.R > 0)
                & (self.R < 1)
            )

            if not valid.any():
                raise ValueError(
                    "Band is undefined: no observations with a positive, "
                    + "finite variance estimate"
                )

            a = N * sigma2 / (1 + N * sigma2)
            ends = np.append(self.x[1:], np.inf)
            if x_range is not None:
                t_l, t_u = self._band_range(x_range)
                # The steps [x_j, x_j+1) that meet [t_L, t_U].
                valid &= (self.x <= t_u) & (ends > t_l)
                what = "in x_range"
            elif method == "nair":
                valid &= (a >= _EP_RANGE[0]) & (a <= _EP_RANGE[1])
                what = (
                    "with a = N sigma^2 / (1 + N sigma^2) between {} and {}, "
                    "the equal precision band's default range; give one "
                    "with x_range, or use the Hall-Wellner band".format(
                        *_EP_RANGE
                    )
                )
            if not valid.any():
                raise ValueError(
                    "Band is undefined: no observations {} with a "
                    "positive, finite variance estimate".format(what)
                )
            a_l = a[valid].min()
            a_u = a[valid].max()

            crit = self._band_critical_value(
                a_l, a_u, alpha_ci, standardized=(method == "nair")
            )

            if method == "nair":
                half_width = crit * np.sqrt(sigma2)
            else:
                half_width = crit * (1 + N * sigma2) / np.sqrt(N)

            if bound_type == "arcsine":
                # Klein and Moeschberger's arcsine-square-root bands
                # (Section 4.4): by the delta method arcsin(sqrt(S)) has
                # the standard error sigma sqrt(S / (1 - S)) / 2, sigma^2
                # the Greenwood sum.
                # Near S = 1 the upper end reaches 1, and the band does not
                # pull the cumulative hazard down by a factor e^-c at the
                # first event, as on the log(-log) scale, where a step of
                # a few failures is far from normal (#390).
                angle = np.arcsin(np.sqrt(self.R))
                se = 0.5 * half_width * np.sqrt(self.R / (1 - self.R))
                lower = np.sin(np.maximum(angle - se, 0.0)) ** 2
                upper = np.sin(np.minimum(angle + se, np.pi / 2)) ** 2
            elif bound_type == "exp":
                # Band applied on the log(-log) scale, mirroring the
                # pointwise exponential Greenwood bounds.
                theta = np.log(-np.log(self.R))
                se = half_width / np.abs(np.log(self.R))
                lower = np.exp(-np.exp(theta + se))
                upper = np.exp(-np.exp(theta - se))
            else:
                lower = self.R - half_width * self.R
                upper = self.R + half_width * self.R

            lower = np.where(valid, lower, np.nan)
            upper = np.where(valid, upper, np.nan)

            if x is None:
                x = self.x
            x = np.atleast_1d(x).astype(float)
            idx = np.searchsorted(self.x, x, side="right") - 1
            idx_c = np.clip(idx, 0, len(self.x) - 1)
            out = np.empty((x.size, 2))
            out[:, 0] = np.where(idx < 0, np.nan, lower[idx_c])
            out[:, 1] = np.where(idx < 0, np.nan, upper[idx_c])
            outside = (x < self.x.min()) | (x > self.x.max()) | np.isnan(x)
            if x_range is not None:
                outside |= (x < t_l) | (x > t_u)
            out[outside] = np.nan

        return out

    @staticmethod
    def _band_range(x_range: Any) -> tuple[float, float]:
        """``x_range`` as two increasing times ``(t_L, t_U)``."""
        try:
            t_l, t_u = (float(t) for t in x_range)
        except (TypeError, ValueError):
            raise ValueError(
                "'x_range' must be two times (t_L, t_U); got "
                "{!r}".format(x_range)
            ) from None
        if not t_l <= t_u:
            raise ValueError(
                "'x_range' must be two increasing times (t_L, t_U); got "
                "{!r}".format(x_range)
            )
        return t_l, t_u
