from autograd.scipy.special import beta as abeta
from autograd.scipy.special import betaln as abetaln
from scipy.special import betaincinv, comb, digamma

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils.autograd_gamma_compat import betainc as abetainc
from surpyval.utils.autograd_gamma_compat import betaincln as abetaincln


class Beta4_(ParametricFitter):
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
    """

    def __init__(self, name):
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

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        x = np.asarray(x, dtype=float)
        if (n is not None) and (c is not None) and (c == 0).all():
            x = np.repeat(x, n)

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

        return alpha, beta, a, b

    def _z(self, x, a, b):
        """Standardise ``x`` onto the unit interval."""
        return (x - a) / (b - a)

    def sf(self, x, alpha, beta, a, b):
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
        return 1 - self.ff(x, alpha, beta, a, b)

    def cs(self, x, X, alpha, beta, a, b):
        r"""

        Conditional survival (or reliability) function for the
        four-parameter Beta distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
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

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.
        """
        return self.sf(x + X, alpha, beta, a, b) / self.sf(
            X, alpha, beta, a, b
        )

    def ff(self, x, alpha, beta, a, b):
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

    def df(self, x, alpha, beta, a, b):
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
        num = (x - a) ** (alpha - 1) * (b - x) ** (beta - 1)
        den = abeta(alpha, beta) * (b - a) ** (alpha + beta - 1)
        return num / den

    def hf(self, x, alpha, beta, a, b):
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
        return self.df(x, alpha, beta, a, b) / self.sf(x, alpha, beta, a, b)

    def Hf(self, x, alpha, beta, a, b):
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
        return -np.log(self.sf(x, alpha, beta, a, b))

    def qf(self, p, alpha, beta, a, b):
        r"""

        Quantile function for the four-parameter Beta distribution:

        .. math::
            q(p) = a + \left(b - a\right) I^{-1}_{p}\left(\alpha, \beta\right)

        Parameters
        ----------

        p : numpy array or scalar
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
            The quantiles for the Beta distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta4
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Beta4.qf(p, 3, 4, 2, 3)
        array([2.20090888, 2.26864915, 2.32332388, 2.37307973, 2.42140719])
        """
        return a + (b - a) * betaincinv(alpha, beta, p)

    def mean(self, alpha, beta, a, b):
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

    def moment(self, m, alpha, beta, a, b):
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
        2.4285714285714284
        """
        scale = b - a
        total = 0.0
        for k in range(m + 1):
            # k-th raw moment of the standard Beta(alpha, beta)
            u_moment = np.exp(abetaln(k + alpha, beta) - abetaln(alpha, beta))
            total = total + comb(m, k) * a ** (m - k) * scale**k * u_moment
        return total

    def entropy(self, alpha, beta, a, b):
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

    def log_df(self, x, alpha, beta, a, b):
        return (
            (alpha - 1) * np.log(x - a)
            + (beta - 1) * np.log(b - x)
            - abetaln(alpha, beta)
            - (alpha + beta - 1) * np.log(b - a)
        )

    def log_ff(self, x, alpha, beta, a, b):
        z = np.clip(self._z(x, a, b), 0.0, 1.0)
        return abetaincln(alpha, beta, z)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        return self.qf(y, *params)

    def mpp_inv_y_transform(self, y, *params):
        return self.ff(y, *params)

    def mpp(self, *args, **kwargs):
        msg = "Probability Plotting Method for Beta4 distribution"
        raise NotImplementedError(msg)

    def _plot_x_bounds(self, x, params):
        return float(params[2]), float(params[3])


Beta4: ParametricFitter = Beta4_("Beta4")
