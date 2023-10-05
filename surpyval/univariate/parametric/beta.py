from autograd.scipy.special import beta as abeta
from autograd.scipy.special import betaln as abetaln
from autograd_gamma import betainc as abetainc
from autograd_gamma import betaincln as abetaincln
from scipy.special import betaincinv
from scipy.special import gamma as gamma_func
from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


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
            y_ticks=[
                0.0001,
                0.0002,
                0.0003,
                0.001,
                0.002,
                0.003,
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
                0.99,
                0.999,
                0.9999,
            ],
        )

    def _parameter_initialiser(self, x, c=None, n=None):
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

        df : scalar or numpy array
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

        df : scalar or numpy array
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

        df : scalar or numpy array
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

        df : scalar or numpy array
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

        df : scalar or numpy array
            The value(s) of the cumulative hazard function at x.

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

    def random(self, size, alpha, beta):
        r"""
        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        alpha : numpy array or scalar
            One shape parameter for the Beta distribution
        beta : numpy array or scalar
            Another scale parameter for the Beta distribution

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> Beta.random(10, 3, 4)
        array([0.48969376, 0.50266801, 0.66913503, 0.3031257 , 0.60897905,
               0.34901845, 0.34196432, 0.37401123, 0.06191741, 0.17604693])
        >>> Beta.random((5, 5), 3, 4)
        array([[0.68575237, 0.39409872, 0.18546004, 0.51266116, 0.55043579],
               [0.62656723, 0.69497065, 0.68958488, 0.37330801, 0.57053267],
               [0.07330674, 0.68020665, 0.42907148, 0.51884251, 0.12159803],
               [0.3566228 , 0.55493446, 0.59288881, 0.43542773, 0.31740851],
               [0.82044756, 0.23062323, 0.35342936, 0.2902573 , 0.54522114]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, alpha, beta)

    def log_df(self, x, alpha, beta):
        return (
            (alpha - 1) * np.log(x)
            + (beta - 1) * np.log1p(-x)
            - abetaln(alpha, beta)
        )

    def log_ff(self, x, alpha, beta):
        return abetaincln(alpha, beta, x)

    def mpp_y_transform(self, y, alpha, beta):
        return self.qf(y, alpha, beta)

    def mpp_inv_y_transform(self, y, alpha, beta):
        return abetainc(y, alpha, beta)

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

    def var_R(self, dR, cv_matrix):
        dr_dalpha = dR[:, 0]
        dr_dbeta = dR[:, 1]
        var_r = (
            dr_dalpha**2 * cv_matrix[0, 0]
            + dr_dbeta**2 * cv_matrix[1, 1]
            + 2 * dr_dalpha * dr_dbeta * cv_matrix[0, 1]
        )
        return var_r


Beta: ParametricFitter = Beta_("Beta")
