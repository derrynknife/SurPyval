from autograd.scipy.special import beta as abeta
from autograd.scipy.special import betaln as abetaln
from scipy.special import betaincinv
from scipy.special import gamma as gamma_func
from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils.autograd_gamma_compat import betainc as abetainc
from surpyval.utils.autograd_gamma_compat import betaincln as abetaincln


class Beta_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, 1),
            param_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        if (c is not None) & ((c == 0).all()):
            x = np.repeat(x, n)
            p = self._mom(x)
        else:
            p = 1.0, 1.0
        return p

    def sf(self, x, alpha, beta):
        r"""

        Survival (or reliability) function for the Beta Distribution:

        .. math::
            R(x) = 1 - \int_{0}^{x}t^{\alpha-1}\left (1 - t \right )^
            {\beta - 1}dt

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.sf(x, 3, 4)
        array([0.98415, 0.90112, 0.74431, 0.54432, 0.34375])
        """
        return 1 - self.ff(x, alpha, beta)

    def cs(self, x, X, alpha, beta):
        r"""

        Conditional survival (or reliability) function for the Beta
        Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.cs(x, 0.4, 3, 4)
        array([0.6315219 , 0.32921811, 0.12946429, 0.03115814, 0.00233319])
        """
        return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

    def ff(self, x, alpha, beta):
        r"""

        Failure (CDF or unreliability) function for the Beta Distribution:

        .. math::
            F(x) = \int_{0}^{x}t^{\alpha-1}\left (1 - t \right )^{\beta -1}dt

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.ff(x, 3, 4)
        array([0.01585, 0.09888, 0.25569, 0.45568, 0.65625])
        """
        return abetainc(alpha, beta, x)

    def df(self, x, alpha, beta):
        r"""

        Density function for the Beta Distribution:

        .. math::
            f(x) = \frac{x^{\alpha-1}\left(1 - x \right )^{\beta-1}}{B \left (
            \alpha , \beta \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.df(x, 3, 4)
        array([0.4374, 1.2288, 1.8522, 2.0736, 1.875 ])
        """
        return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / abeta(alpha, beta)

    def hf(self, x, alpha, beta):
        r"""

        Instantaneous hazard rate for the Beta distribution.

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.hf(x, 3, 4)
        array([0.44444444, 1.36363636, 2.48847926, 3.80952381, 5.45454545])
        """
        return self.df(x, alpha, beta) / self.sf(x, alpha, beta)

    def Hf(self, x, alpha, beta):
        r"""

        Cumulative hazard rate for the Beta distribution.

        .. math::
            H(x) = -\ln\left( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            The scale parameter for the Beta distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.hf(x, 3, 4)
        array([0.44444444, 1.36363636, 2.48847926, 3.80952381, 5.45454545])
        """
        return -np.log(self.sf(x, alpha, beta))

    def qf(self, p, alpha, beta):
        r"""

        Quantile function for the Beta Distribution:

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            Another scale parameter for the Beta distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Beta distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Beta.qf(p, 3, 4)
        array([0.20090888, 0.26864915, 0.32332388, 0.37307973, 0.42140719])
        """
        return betaincinv(alpha, beta, p)

    def mean(self, alpha, beta):
        r"""

        Mean of the Beta distribution

        .. math::
            E = \frac{\alpha}{\alpha + \beta}

        Parameters
        ----------

        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            Another scale parameter for the Beta distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.mean(3, 4)
        0.42857142857142855
        """
        return alpha / (alpha + beta)

    def moment(self, n, alpha, beta):
        r"""

        n-th (non central) moment of the Beta distribution

        .. math::
            E = \frac{\Gamma \left( n + \alpha\right )}{\beta^{n}\Gamma
            \left ( \alpha \right )}

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            Another scale parameter for the Beta distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.moment(2, 3, 4)
        0.75
        """
        return gamma_func(n + alpha) / (beta**n * gamma_func(alpha))

    def log_df(self, x, alpha, beta):
        return (
            (alpha - 1) * np.log(x)
            + (beta - 1) * np.log1p(-x)
            - abetaln(alpha, beta)
        )

    def log_ff(self, x, alpha, beta):
        return abetaincln(alpha, beta, x)

    def mpp_y_transform(self, y, *params):
        return self.qf(y, *params)

    def mpp_inv_y_transform(self, y, *params):
        return abetainc(y, *params)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp(
        self,
        x,
        c=None,
        n=None,
        heuristic="Nelson-Aalen",
        rr="y",
        on_d_is_0=False,
        offset=False,
    ):
        msg = "Probability Plotting Method for Beta distribution"
        raise NotImplementedError(msg)

    def _mom(self, x):
        """
        MOM: Method of Moments for the beta distribution has an analytic answer
        """
        mean = x.mean()
        var = x.var()
        term1 = (mean * (1 - mean) / var) - 1
        alpha = term1 * mean
        beta = term1 * (1 - mean)

        return alpha, beta


Beta: ParametricFitter = Beta_("Beta")
