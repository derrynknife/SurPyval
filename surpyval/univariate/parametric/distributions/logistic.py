from autograd import grad
from autograd.scipy.special import beta as abeta
from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils import xcnt_handler


class Logistic_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=(
                (None, None),
                (0, None),
            ),
            support=(-np.inf, np.inf),
            param_names=["mu", "sigma"],
            param_map={"mu": 0, "sigma": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        return self.fit(x, c, n, how="MPP").params
        x, c, n, _ = xcnt_handler(x, c, n, t)
        flag = (c == 0).astype(int)
        if offset:
            return x.sum() / (n * flag).sum(), 1.0, 1.0
        else:
            return x.sum() / (n * flag).sum(), 1.0

    def sf(self, x, mu, sigma):
        """

        Survival (or reliability) function for the Logistic Distribution:

        .. math::
            R(x) = 1 - \\frac{1}{1 + e^{- \\left (
                x - \\mu \\right )/ \\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.sf(x, 3, 4)
        array([0.62245933, 0.5621765 , 0.5       , 0.4378235 , 0.37754067])
        """
        exp_term = np.exp(-(x - mu) / sigma)
        return exp_term / (1 + exp_term)

    def cs(self, x, X, mu, sigma):
        return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

    def ff(self, x, mu, sigma):
        """

        Failure (CDF or unreliability) function for the Logistic Distribution:

        .. math::
            F(x) = \\frac{1}{1 + e^{- \\left ( x - \\mu \\right )/ \\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.ff(x, 3, 4)
        array([0.37754067, 0.4378235 , 0.5       , 0.5621765 , 0.62245933])
        """
        z = (x - mu) / sigma
        return 1.0 / (1 + np.exp(-z))

    def df(self, x, mu, sigma):
        """

        Failure (CDF or unreliability) function for the Logistic Distribution:

        .. math::
            f(x) = \\frac{e^{-\\left ( x - \\mu \\right ) / \\sigma}}{\\sigma
            \\left ( 1 + e^{-\\left ( x - \\mu \\right )/ \\sigma}\\right )^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.df(x, 3, 4)
        array([0.05875093, 0.06153352, 0.0625    , 0.06153352, 0.05875093])
        """
        z = (x - mu) / sigma
        return np.exp(-z) / (sigma * (1 + np.exp(-z)) ** 2)

    def hf(self, x, mu, sigma):
        """

        Instantaneous hazard rate for the Logistic Distribution:

        .. math::
            h(x) = \\frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.hf(x, 3, 4)
        array([0.09438517, 0.10945587, 0.125     , 0.14054413, 0.15561483])
        """
        return self.df(x, mu, sigma) / self.sf(x, mu, sigma)

    def Hf(self, x, mu, sigma):
        """

        Cumulative hazard rate for the Logistic distribution:

        .. math::
            H(x) = -\\ln \\left( R(x) \\right)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.Hf(x, 3, 4)
        array([0.47407698, 0.57593942, 0.69314718, 0.82593942, 0.97407698])
        """
        return -np.log(self.sf(x, mu, sigma))

    def qf(self, p, mu, sigma):
        """

        Quantile function for the Logistic distribution:

        .. math::
            q(p) = \\mu + \\sigma \\ln \\left ( \\frac{p}{1 - p} \\right)

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Logistic distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Logistic.qf(p, 3, 4)
        array([-5.78889831, -2.54517744, -0.38919144,  1.37813957,  3.       ])
        """
        return mu + sigma * (np.log(p) - np.log1p(-p))

    def mean(self, mu, sigma):
        r"""

        Mean of the Logistic distribution

        .. math::
            E = \mu

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        mu : scalar or numpy array
            The mean(s) of the Logistic distribution

        Examples
        --------
        >>> from surpyval import Logistic
        >>> Logistic.mean(3, 4)
        3
        """
        return mu

    def log_df(self, x, mu, sigma):
        z = (x - mu) / sigma
        return -(z + np.log(sigma) + 2 * np.log1p(np.exp(-z)))

    def log_sf(self, x, mu, sigma):
        z = (x - mu) / sigma
        return -(z + np.log1p(np.exp(-z)))

    def log_ff(self, x, mu, sigma):
        z = (x - mu) / sigma
        return -np.log1p(np.exp(-z))

    def mgf(self, t, mu, sigma):
        return np.exp(mu * t) * abeta(1 + sigma * t, 1 - sigma * t)

    def moment(self, n, mu, sigma):
        d = self.mgf
        for i in range(n):
            d = grad(d)
        return d(0.0, mu, sigma)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = -np.log(1.0 / y[~mask] - 1)
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1.0 / (np.exp(-y) + 1)

    def unpack_rr(self, params, rr):
        if rr == "y":
            sigma = 1.0 / params[0]
            mu = -sigma * params[1]
        elif rr == "x":
            sigma = params[0]
            mu = params[1]
        return mu, sigma


Logistic: ParametricFitter = Logistic_("Logistic")
