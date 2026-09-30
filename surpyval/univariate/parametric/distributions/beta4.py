from typing import Any

import numpy.typing as npt
from autograd.scipy.special import betaln as abetaln
from scipy.special import betaincinv, comb, digamma

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.autograd_gamma_compat import betainc as abetainc
from surpyval.utils.autograd_gamma_compat import betainccln as abetainccln
from surpyval.utils.autograd_gamma_compat import betaincln as abetaincln
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.surpyval_data import SurpyvalData


def _power_log(k: Boxable, z: Boxable) -> Boxable:
    r""":math:`k \ln z` for :math:`z \geq 0`, with its limit at
    :math:`z = 0`: :math:`-\infty`, 0 or :math:`\infty` as ``k`` is
    positive, 0 or negative. The log sees a positive argument only, so
    nothing warns and no nan enters a gradient."""
    positive = z > 0.0
    at_zero = np.where(k > 0.0, -np.inf, np.where(k < 0.0, np.inf, 0.0))
    return np.where(positive, k * np.log(np.where(positive, z, 1.0)), at_zero)


class Beta4_(OptimisedFitMixin, ParametricFitter):
    r"""
    The four-parameter (generalised) Beta distribution.

    The standard :class:`Beta` distribution is supported on ``[0, 1]``.
    The four-parameter Beta generalises it to an arbitrary finite
    interval ``[a, b]`` by introducing a location parameter ``a`` (the
    lower bound) and a scale that stretches the unit interval out to an
    upper bound ``b``. If ``Y`` is a standard Beta random variable then
    ``X = a + (b - a) Y`` is four-parameter Beta distributed.

    Because the support ``[a, b]`` is itself estimated, this is the
    distribution to reach for when data are bounded on *both* sides but
    neither bound is zero — the case where ``Beta(..., offset=True)``
    would (deliberately) refuse, since a one-sided offset cannot move the
    lower bound while keeping the upper bound pinned at 1.

    .. note::
       Fit it with ``how="MPS"`` (maximum product of spacings). Its
       likelihood is unbounded: with a shape below 1 the density is
       infinite at a support end, so a maximum-likelihood fit can run
       an end onto the smallest or largest observation, where there is no
       maximum, and its answer then depends on the data's units. Such a fit
       warns "No finite maximum". MPS scores an end gap of zero as minus
       infinity, so its estimates are finite, consistent (Cheng & Amin,
       1983) and the same in any units. MLE stays the default, as for
       every distribution, and is fine when the data put both shapes above
       1.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=4,
            bounds=((0, None), (0, None), (None, None), (None, None)),
            # The support [a, b] is data-dependent and resolved from the
            # fitted ``a`` (param 2) and ``b`` (param 3) parameters.
            support=(np.nan, np.nan),
            param_names=["alpha", "beta", "a", "b"],
            param_map={"alpha": 0, "beta": 1, "a": 2, "b": 3},
            plot_x_scale="linear",
        )
        # The four-parameter Beta has no linearising probability plot.
        self.supports_mpp = False
        # ``a`` and ``b`` supply the left and right support bounds.
        self.support_param_index = (2, 3)

    def _check_params(self, params: Any) -> None:
        # Each parameter is unbounded on its own, so from_params used to
        # accept a > b -- a model whose sf was 0 everywhere.
        if not params[2] < params[3]:
            raise ValueError(
                f"{self.name} needs a < b; got a = {params[2]}, "
                f"b = {params[3]}"
            )

    def _warn_if_at_limit(
        self,
        surv_data: SurpyvalData,
        results: dict,
        zi: bool,
        lfp: bool,
    ) -> bool:
        """Warn when the fit ran a support end onto an observation with its
        shape below 1, where the likelihood is unbounded (#385, #392).

        With ``alpha < 1`` the density grows without bound at ``a``, so
        moving ``a`` onto the smallest exactly observed value makes the
        likelihood infinite: the four-parameter Beta has no maximum
        likelihood estimate there (Smith, 1985; ``beta < 1`` at ``b``
        likewise). The fit then stops wherever its search gave up, and the
        answer depends on the data's units: shapes of 1.00 and 1.19 on
        the registry's fixture, 0.18 and 0.18 on the same data times 7.3.
        The criterion is that end resting on the extreme observation to
        half the digits of the support's width (``sqrt(eps)``), which a
        fit with an interior maximum -- the end strictly outside the data,
        where the density at the extreme is finite -- does not reach.
        Maximum product of spacings has no such limit: an end gap of zero
        scores minus infinity, and its fit is the same in any units.
        """
        params = np.asarray(results.get("params", []), dtype=float)
        if params.size != 4 or not np.all(np.isfinite(params)):
            return False
        alpha, beta, a, b = params
        x = np.asarray(surv_data.x, dtype=float)
        if x.ndim != 1:
            return False
        exact = x[np.asarray(surv_data.c) == 0]
        if exact.size == 0 or not b > a:
            return False
        close = np.sqrt(np.finfo(float).eps) * (b - a)
        ends = []
        if alpha < 1 and exact.min() - a <= close:
            ends.append(
                f"a = {a:.6g} on the smallest observation "
                f"{exact.min():.6g} with alpha = {alpha:.4g}"
            )
        if beta < 1 and b - exact.max() <= close:
            ends.append(
                f"b = {b:.6g} on the largest observation "
                f"{exact.max():.6g} with beta = {beta:.4g}"
            )
        if not ends:
            return False
        warn_no_maximum(
            "the Beta4 likelihood is unbounded: a shape below 1 makes the "
            "density infinite at a support end, and the fit ran "
            + " and ".join(ends),
            "The reported parameters are where the search stopped (they "
            "change with the data's units), and their standard errors and "
            "bounds are meaningless",
            "fit with how='MPS' (maximum product of spacings), which is "
            "finite here and the same in any units",
        )
        return True

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        x = np.asarray(data.x, dtype=float)
        if (data.c == 0).all():
            x = np.repeat(x, data.n)

        span = x.max() - x.min()
        if span <= 0:
            span = 1.0

        # Place the initial bounds just outside the observed range so that
        # every observed point sits strictly inside (a, b).
        a = x.min() - 0.05 * span
        b = x.max() + 0.05 * span

        u = (x - a) / (b - a)
        mean = u.mean()
        var = u.var()
        if var <= 0:
            var = 1e-3
        term1 = (mean * (1 - mean) / var) - 1
        alpha = max(term1 * mean, 0.5)
        beta = max(term1 * (1 - mean), 0.5)

        return np.array([alpha, beta, a, b], dtype=float)

    def _z(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        """Standardise ``x`` onto the unit interval."""
        return (x - a) / (b - a)

    def sf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Survival (or reliability) function for the four-parameter Beta
        distribution:

        .. math::
            R(x) = 1 - I_{z}\left(\alpha, \beta\right), \quad
            z = \frac{x - a}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta4
        >>> x = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
        >>> Beta4.sf(x, 3, 4, 2, 3)
        array([0.98415, 0.90112, 0.74431, 0.54432, 0.34375])
        """
        # 1 - F where F is below 1/2, and the upper tail itself (from its
        # log) where it is small: 1 - F was 0 where R is 1e-30 (#442).
        ff = self.ff(x, alpha, beta, a, b)
        return np.where(
            ff < 0.5, 1.0 - ff, np.exp(self.log_sf(x, alpha, beta, a, b))
        )

    def ff(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the four-parameter
        Beta distribution:

        .. math::
            F(x) = I_{z}\left(\alpha, \beta\right), \quad
            z = \frac{x - a}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta4
        >>> x = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
        >>> Beta4.ff(x, 3, 4, 2, 3)
        array([0.01585, 0.09888, 0.25569, 0.45568, 0.65625])
        """
        z = np.clip(self._z(x, a, b), 0.0, 1.0)
        return abetainc(alpha, beta, z)

    def df(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Density function for the four-parameter Beta distribution:

        .. math::
            f(x) = \frac{\left(x - a\right)^{\alpha - 1}
            \left(b - x\right)^{\beta - 1}}{B\left(\alpha, \beta\right)
            \left(b - a\right)^{\alpha + \beta - 1}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta4
        >>> x = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
        >>> Beta4.df(x, 3, 4, 2, 3)
        array([0.4374, 1.2288, 1.8522, 2.0736, 1.875 ])
        """
        # From the log density: the powers and B(alpha, beta) of the
        # algebraic form overflow separately at extreme shapes (at alpha =
        # 1000 on [-1e6, 1e6], (b - a)^999 raised OverflowError) although
        # the density is an ordinary number (#445).
        return np.exp(self.log_df(x, alpha, beta, a, b))

    def hf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Instantaneous hazard rate for the four-parameter Beta
        distribution.

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.
        """
        # df = 0 and sf = 0 above the support made hf return NaN (0/0)
        # for x > b; the hazard is 0 below the support (no mass yet) and
        # infinite at/above the upper bound (no survivors) (#289).
        # On the log scale: df/sf was inf (or 0/0) once sf underflowed
        # (#443).
        x_arr = np.asarray(x, dtype=float)
        log_hf = self.log_df(x_arr, alpha, beta, a, b) - self.log_sf(
            x_arr, alpha, beta, a, b
        )
        out = np.where(x_arr < a, 0.0, np.exp(log_hf))
        return np.where(x_arr >= b, np.inf, out)

    def Hf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Cumulative hazard rate for the four-parameter Beta distribution.

        .. math::
            H(x) = -\ln\left(R(x)\right)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.
        """
        return -self.log_sf(x, alpha, beta, a, b)

    def qf(
        self, u: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Quantile function for the four-parameter Beta distribution:

        .. math::
            q(u) = a + \left(b - a\right) I^{-1}_{u}\left(\alpha, \beta\right)

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Beta distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta4
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Beta4.qf(u, 3, 4, 2, 3)
        array([2.20090888, 2.26864915, 2.32332388, 2.37307973, 2.42140719])
        """
        return a + (b - a) * betaincinv(alpha, beta, u)

    def mean(
        self, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Mean of the four-parameter Beta distribution

        .. math::
            E = a + \left(b - a\right)\frac{\alpha}{\alpha + \beta}

        Parameters
        ----------

        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta4
        >>> Beta4.mean(3, 4, 2, 3)
        2.4285714285714284
        """
        return a + (b - a) * alpha / (alpha + beta)

    def moment(
        self, m: int, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        m-th (non central) moment of the four-parameter Beta distribution.

        Computed from the standard Beta moments via the binomial
        expansion of :math:`\left(a + (b - a) U\right)^m`.

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta4
        >>> Beta4.moment(1, 3, 4, 2, 3)
        np.float64(2.428571428571429)
        """
        scale = b - a
        total = 0.0
        for k in range(m + 1):
            # k-th raw moment of the standard Beta(alpha, beta)
            u_moment = np.exp(abetaln(k + alpha, beta) - abetaln(alpha, beta))
            total = total + comb(m, k) * a ** (m - k) * scale**k * u_moment
        return total

    def entropy(
        self, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        r"""

        Differential entropy of the four-parameter Beta distribution.

        Equal to the standard Beta entropy plus :math:`\ln(b - a)` for the
        change of scale.

        Parameters
        ----------

        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        a : numpy array or scalar
            The lower bound of the support
        b : numpy array or scalar
            The upper bound of the support

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Beta distribution
        """
        standard = (
            abetaln(alpha, beta)
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (alpha + beta - 2) * digamma(alpha + beta)
        )
        return standard + np.log(b - a)

    def log_df(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        # The density is zero outside [a, b]; evaluating the power terms
        # there returned arbitrary nonzero, negative, or NaN values
        # (fractional powers of negative bases), so hf inherited garbage
        # on any grid extending past the fitted support (#280). At an edge
        # the limit is taken: 0, the constant, or inf as the shape there
        # is above, at or below 1.
        x = np.asarray(x, dtype=float)
        inside = (x >= a) & (x <= b)
        xc = np.where(inside, x, 0.5 * (a + b))
        log_df = (
            _power_log(alpha - 1.0, (xc - a) / (b - a))
            + _power_log(beta - 1.0, (b - xc) / (b - a))
            - abetaln(alpha, beta)
            - np.log(b - a)
        )
        return np.where(inside, log_df, -np.inf)

    def log_ff(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        z = np.clip(self._z(x, a, b), 0.0, 1.0)
        return abetaincln(alpha, beta, z)

    def log_sf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, a: Boxable, b: Boxable
    ) -> Boxable:
        # The upper tail's own log (log(1 - F) lost R below 1e-16, #442,
        # and was -inf where R underflowed, #443), as the lower tail of the
        # mirrored Beta at (b - x) / (b - a) where that is small: taken from
        # x, not as 1 - z, it keeps its digits near b.
        z = np.clip(self._z(x, a, b), 0.0, 1.0)
        zc = np.clip((b - x) / (b - a), 0.0, 1.0)
        return np.where(
            zc < 0.5,
            abetaincln(beta, alpha, zc),
            abetainccln(alpha, beta, z),
        )

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return self.qf(y, *params)

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return self.ff(y, *params)

    def _plot_x_bounds(
        self, x: npt.NDArray, params: npt.NDArray
    ) -> tuple[float, float] | None:
        return float(params[2]), float(params[3])


Beta4: Beta4_ = Beta4_("Beta4")
